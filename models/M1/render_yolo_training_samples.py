"""Renderiza muestras del train con cajas y esqueletos de YOLO11n-pose."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from models.M1.pose_cache import crop_training_roi, sample_indices
ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = CURRENT_DATASET / 'train.csv'
VIDEO_DIRS = (ROOT / 'data' / '7seg_unique_cropped', ROOT / 'data' / '7Seg', ROOT / 'data' / '7seg unique')
CACHE_DIR = experiment_directory(__file__) / 'pose_cache' / 'yolo11n-pose_pt'
DEFAULT_OUTPUT = experiment_directory(__file__) / 'visualizations' / 'yolo_training_samples'
SKELETON = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))

def locate_video(name: str) -> Path:
    for directory in VIDEO_DIRS:
        candidate = directory / name
        if _resolve_model_artifact(study_csv_path(candidate)).exists():
            return candidate
    raise FileNotFoundError(f'No se encontro el video de entrenamiento {name}')

def draw_frame(video_path: Path, cache_path: Path, cache_index: int, label: str, output_path: Path) -> dict:
    with _load_model_result_array(cache_path, allow_pickle=False) as data:
        boxes = np.asarray(data['boxes'], dtype=np.float32)
        person_conf = np.asarray(data['person_conf'], dtype=np.float32)
        keypoints = np.asarray(data['keypoints'], dtype=np.float32)
        keypoint_conf = np.asarray(data['keypoint_conf'], dtype=np.float32)
        valid = np.asarray(data['valid'], dtype=bool)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'No se pudo abrir {video_path}')
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_indices(total, len(boxes))
    source_index = int(indices[min(cache_index, len(indices) - 1)])
    capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f'No se pudo leer frame {source_index} de {video_path}')
    roi = crop_training_roi(frame, 640)
    if roi is None:
        raise RuntimeError(f'No se pudo reproducir el recorte de entrenamiento para {video_path}')
    canvas = roi.copy()
    people = 0
    for person_index in range(valid.shape[1]):
        if not valid[cache_index, person_index] or person_conf[cache_index, person_index] < 0.25:
            continue
        people += 1
        x1, y1, x2, y2 = boxes[cache_index, person_index] * 640.0
        x1, y1, x2, y2 = [int(round(value)) for value in (x1, y1, x2, y2)]
        confidence = float(person_conf[cache_index, person_index])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 220, 80), 2)
        cv2.putText(canvas, f'p{person_index + 1:02d} {confidence:.2f}', (max(2, x1), max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 220, 80), 1, cv2.LINE_AA)
        points = keypoints[cache_index, person_index] * 640.0
        point_conf = keypoint_conf[cache_index, person_index]
        for start, end in SKELETON:
            if point_conf[start] >= 0.2 and point_conf[end] >= 0.2:
                a = tuple((int(round(value)) for value in points[start]))
                b = tuple((int(round(value)) for value in points[end]))
                cv2.line(canvas, a, b, (255, 170, 40), 2, cv2.LINE_AA)
        for point, confidence_point in zip(points, point_conf):
            if confidence_point >= 0.2:
                center = tuple((int(round(value)) for value in point))
                cv2.circle(canvas, center, 3, (40, 120, 255), -1, cv2.LINE_AA)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (640, 53), (20, 20, 20), -1)
    canvas = cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0.0)
    cv2.putText(canvas, f'YOLO11n-pose | clase: {label} | frame cache: {cache_index + 1}/{len(boxes)}', (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, f'frame video: {source_index} | personas: {people} | verde=caja  azul=esqueleto  naranja=punto', (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return {'video': video_path.name, 'label': label, 'cache_frame': cache_index, 'video_frame': source_index, 'people': people, 'path': str(output_path)}

def make_contact_sheet(images: list[Path], output_path: Path) -> None:
    loaded = [cv2.imread(str(path)) for path in images]
    loaded = [image for image in loaded if image is not None]
    if not loaded:
        raise RuntimeError('No se pudieron cargar las muestras renderizadas')
    width = 640
    height = int(round(640 * loaded[0].shape[0] / loaded[0].shape[1]))
    resized = [cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) for image in loaded]
    while len(resized) < 6:
        resized.append(np.zeros_like(resized[0]))
    rows = [np.hstack(resized[index:index + 2]) for index in range(0, 6, 2)]
    sheet = np.vstack(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--per-class', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260811)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    train = pd.read_csv(_resolve_model_artifact(study_csv_path(TRAIN_CSV)))
    train['video_name'] = train['video_name'].astype(str).str.strip()
    train['tag'] = train['tag'].astype(str).str.strip().str.lower()
    selected: list[tuple[str, str]] = []
    for label in ('line', 'other', 'scrum'):
        candidates = train[train['tag'] == label].sample(frac=1.0, random_state=args.seed + len(label)).reset_index(drop=True)
        chosen = 0
        for name in candidates['video_name']:
            cache_path = CACHE_DIR / name.replace('.mp4', '.npz')
            if not _resolve_model_artifact(study_csv_path(cache_path)).exists():
                continue
            with _load_model_result_array(cache_path, allow_pickle=False) as data:
                if int(np.sum(np.any(data['valid'], axis=1))) == 0:
                    continue
            selected.append((name, label))
            chosen += 1
            if chosen >= args.per_class:
                break
    records: list[dict] = []
    images: list[Path] = []
    positions = (4, 16, 27)
    for clip_index, (name, label) in enumerate(selected):
        video_path = locate_video(name)
        cache_path = CACHE_DIR / name.replace('.mp4', '.npz')
        with _load_model_result_array(cache_path, allow_pickle=False) as data:
            frame_count = int(data['boxes'].shape[0])
        for position_index, cache_index in enumerate(positions[:1]):
            output_path = output_root / f'{clip_index + 1:02d}_{label}_{Path(name).stem}_f{cache_index + 1:02d}.png'
            records.append(draw_frame(video_path, cache_path, min(cache_index, frame_count - 1), label, output_path))
            images.append(output_path)
    make_contact_sheet(images, output_root / 'contact_sheet.png')
    (output_root / 'manifest.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Contact sheet: {output_root / 'contact_sheet.png'}")
if __name__ == '__main__':
    import json
    main()
