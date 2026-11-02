"""Dataset partitions, integrity checks and video segment export."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'data/clips_dataset'
LABELS = ('line', 'other', 'scrum')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_partition_files(output, rows, parts):
    """Keep optional provenance columns identical across every partition."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output/'manifest.csv', rows, fields)
    for name, values in parts.items():
        write_csv(output/f'{name}.csv', values, fields)


def local_path(value, root=ROOT):
    path = (root/Path(value)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Ruta fuera del repositorio: {value}')
    return path


def validate_partition_rows(parts, root=ROOT, verify_hashes=False):
    names, hashes = {}, {}
    checked = set()
    for name, rows in parts.items():
        for row in rows:
            if row['tag'] not in LABELS:
                raise ValueError('Clase desconocida')
            if 'path' in row:
                path = local_path(row['path'], root)
                if not path.is_file():
                    raise ValueError(f'Video no disponible: {path}')
                if verify_hashes and path not in checked:
                    if digest(path) != row['sha256']:
                        raise ValueError(f'Video modificado: {path}')
                    checked.add(path)
            for identity, seen in [(row['video_name'], names), (row['sha256'], hashes)]:
                if identity in seen and seen[identity] != name:
                    raise ValueError(f'Clip compartido entre {seen[identity]} y {name}: {identity}')
                seen[identity] = name


def balance(rows, target, seed):
    rng = random.Random(seed)
    result = []
    for tag in LABELS:
        group = sorted((r for r in rows if r['tag'] == tag), key=lambda r: r['video_name'])
        if not group:
            raise ValueError(f'La partición no contiene {tag}')
        result.extend(rng.sample(group, target) if len(group) >= target
                      else group + rng.choices(group, k=target-len(group)))
    rng.shuffle(result)
    return result


def build_partitions(rows, target=600, validation_fraction=.3, seed=42):
    from sklearn.model_selection import train_test_split
    if not 0 < validation_fraction < 1 or target < 2:
        raise ValueError('Fracción o cantidad inválida')
    if len({r['video_name'] for r in rows}) != len(rows) or len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Se requieren clips únicos por nombre y contenido antes del balanceo')
    if any(r['split'] not in ('train', 'test') for r in rows):
        raise ValueError('Partición inicial desconocida')
    development = sorted((r for r in rows if r['split'] == 'train'), key=lambda r: r['video_name'])
    test = sorted((r for r in rows if r['split'] == 'test'), key=lambda r: r['video_name'])
    train, validation = train_test_split(development, test_size=validation_fraction,
                                        random_state=seed, stratify=[r['tag'] for r in development])
    parts = dict(train=train, validation=validation, test=test)
    validate_partition_rows(parts)
    val_target = round(target*validation_fraction)
    if not 0 < val_target < target:
        raise ValueError('El balance debe conservar ejemplos en ambas particiones')
    parts['balanced_train'] = balance(train, target-val_target, seed)
    parts['balanced_validation'] = balance(validation, val_target, seed+1)
    validate_partition_rows(dict(train=parts['balanced_train'], validation=parts['balanced_validation'], test=test))
    return parts


def seven_second_interval(source_duration, old_start=None, old_end=None):
    values = [source_duration] + ([] if old_start is None else [old_start, old_end])
    if any(v is None or not math.isfinite(float(v)) for v in values) or source_duration < 7:
        raise ValueError('La fuente no contiene siete segundos reales')
    if old_start is None:
        center = source_duration/2
    else:
        if not 0 <= old_start < old_end <= source_duration or old_end-old_start > 7.000001:
            raise ValueError('Intervalo previo fuera de la fuente o mayor que siete segundos')
        center = (old_start+old_end)/2
    start = max(0., min(center-3.5, source_duration-7.))
    return start, start+7.


def probe_video(path, decode=False):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f'Video ilegible: {path}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(3)), int(cap.get(4))
        if fps <= 0 or frames <= 0:
            raise ValueError(f'Video sin duración: {path}')
        if decode:
            decoded = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError('Resolución variable')
                decoded += 1
            if decoded != frames:
                raise ValueError(f'Video truncado: {decoded}/{frames}: {path}')
        return dict(frames=frames, fps=fps, width=width, height=height, duration_s=frames/fps)
    finally:
        cap.release()


def extract_real_segment(source, output, start, end):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    info = probe_video(source)
    if abs(end-start-7) > 1e-6 or start < 0 or end > info['duration_s']+1e-6:
        raise ValueError('Se requieren siete segundos reales dentro de la fuente')
    # Apply the half-scale center crop used by the training videos.
    scale = max(.5, 224/min(info['width'], info['height']))
    width, height = round(info['width']*scale), round(info['height']*scale)
    spatial = (f'setpts=PTS-STARTPTS,scale={width}:{height}:flags=bilinear,'
               f'crop=224:224:{(width-224)//2}:{(height-224)//2},fps=25:start_time=0')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-threads', '1',
               '-ss', f'{start:.9f}', '-i', str(source), '-an', '-t', '7',
               '-filter_threads', '1', '-vf', spatial, '-c:v', 'libx264',
               '-threads', '1', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True, capture_output=True)
        actual = probe_video(output, decode=True)
        if (actual['frames'], actual['width'], actual['height']) != (175, 224, 224) or abs(actual['fps']-25) > 1e-6:
            raise ValueError(f'La fuente no produjo 175 cuadros reales a 25 FPS: {actual}')
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return actual


def verify(directory=DEFAULT, media=False):
    directory = local_path(directory)
    manifest = read_csv(directory/'manifest.csv')
    parts = {name: read_csv(directory/f'{name}.csv') for name in ['train', 'validation', 'test']}
    balanced = dict(train=read_csv(directory/'balanced_train.csv'),
                    validation=read_csv(directory/'balanced_validation.csv'), test=parts['test'])
    validate_partition_rows(parts, verify_hashes=True)
    validate_partition_rows(balanced)
    if len(manifest) != len({r['clip_id'] for r in manifest}):
        raise ValueError('Identidades duplicadas en manifiesto')
    by_name = {r['video_name']: r for r in manifest}
    if len(by_name) != len(manifest) or set(by_name) != {r['video_name'] for values in parts.values() for r in values}:
        raise ValueError('Las particiones no cubren el manifiesto')
    for name, values in parts.items():
        if len(values) != len({r['video_name'] for r in values}):
            raise ValueError(f'Clips repetidos dentro de {name}')
        for row in values:
            if row != by_name[row['video_name']]:
                raise ValueError('Partición no coincide con manifiesto')
        allowed = {r['video_name'] for r in values}
        if any(r != by_name[r['video_name']] or r['video_name'] not in allowed for r in balanced[name]):
            raise ValueError('El balanceo cambió la pertenencia de una partición')
        if name != 'test':
            counts = Counter(r['tag'] for r in balanced[name])
            if set(counts) != set(LABELS) or len(set(counts.values())) != 1:
                raise ValueError(f'Balanceo vacío o incorrecto en {name}')
    state = directory/'estado.json'
    if state.exists():
        settings = json.loads(state.read_text(encoding='utf-8'))
        if settings.get('partition_sha256'):
            for filename, expected_hash in settings['partition_sha256'].items():
                if digest(local_path((directory/filename).relative_to(ROOT))) != expected_hash:
                    raise ValueError(f'Partición modificada: {filename}')
        else:
            expected = build_partitions(manifest, target=settings['target_per_class'],
                                        validation_fraction=settings['validation_fraction'], seed=settings['seed'])
            for name, values in expected.items():
                if read_csv(directory/f'{name}.csv') != values:
                    raise ValueError(f'Partición modificada: {name}')
    if media:
        for row in manifest:
            info = probe_video(local_path(row['path']), decode=True)
            if (info['frames'], info['fps'], info['width'], info['height']) != (int(row['frames']), float(row['fps']), 224, 224):
                raise ValueError('Video no válido')
    return dict(status='ok', clips=len(manifest), shared_clips=0, shared_content_hashes=0,
                videos_decoded=media, original_repository_required=False)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    checking = commands.add_parser('verify')
    checking.add_argument('--output', type=Path, default=DEFAULT)
    checking.add_argument('--media', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.output, media=args.media)))

if __name__ == '__main__':
    main()
