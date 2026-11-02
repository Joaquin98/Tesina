"""Frozen experiment helpers. View zero is ALWAYS the existing central crop."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import importlib.util
import json
import cv2
import numpy as np
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
CACHE = ROOT / 'data/multicrop_d2_stride25_vf_20261026'
REPO = ROOT / 'entregas/repositorio_tesina_20261022'
from temporal import STEPS, sample_count, sample_indices
PREVIOUS = ROOT / 'data/yolo_crop_classification_20261023'
spec = importlib.util.spec_from_file_location('legacy_protocol', _resolve_model_artifact(PREVIOUS / 'common.py'))
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
save, digest, score, audit_parts, paired_bootstrap = (legacy.save, legacy.digest, legacy.score, legacy.audit_parts, legacy.paired_bootstrap)
LABELS = ['line', 'other', 'scrum']
SEEDS = [17, 42, 2026]
MODES = ['fixed', 'multi']

def score(y, predicted):
    result = legacy.score(y, predicted)
    matrix = np.asarray(result['confusion_matrix'])
    den = matrix.sum(0)
    precision = np.divide(np.diag(matrix), den, out=np.zeros(3), where=den != 0)
    return {**result, **{f'{label}_precision': float(precision[i]) for i, label in enumerate(LABELS)}}
from runtime_io import save

def load(path):
    return json.loads(_resolve_model_artifact(Path(path)).read_text(encoding='utf-8'))

def context_geometry(width, height):
    scale = max(360 / height, 224 / width)
    nw, nh = (max(224, int(np.floor(width * scale + 0.5))), max(224, int(np.floor(height * scale + 0.5))))
    xs = np.rint(np.linspace(0, nw - 224, 4)).astype(int)
    ys = np.rint(np.linspace(0, nh - 224, 2)).astype(int)
    rects = [[int(x), int(y), 224, 224] for y in ys for x in xs]
    if len(set(map(tuple, rects))) != 8 or max(np.diff(xs)) > 224 or max(np.diff(ys)) > 224:
        raise ValueError('Unsupported aspect ratio for eight covering context crops')
    return (nw, nh, rects)

def views(frame):
    h, w = frame.shape[:2]
    hw, hh = (round(w * 0.5), round(h * 0.5))
    if min(hw, hh) < 224:
        raise ValueError('Original source too small for repository crop')
    half = cv2.resize(frame, (hw, hh), interpolation=cv2.INTER_LINEAR)
    x, y = ((hw - 224) // 2, (hh - 224) // 2)
    images = [half[y:y + 224, x:x + 224].copy()]
    meta = [{'view': 0, 'kind': 'repository_center', 'resize': [hw, hh], 'rect': [x, y, 224, 224]}]
    nw, nh, rects = context_geometry(w, h)
    normalized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    for i, (x, y, ww, hx) in enumerate(rects, 1):
        images.append(normalized[y:y + hx, x:x + ww].copy())
        meta.append({'view': i, 'kind': 'context', 'resize': [nw, nh], 'rect': [x, y, ww, hx]})
    return (np.stack(images), meta)

def read_samples(row):
    """Same source indices used to export the corrected 25 FPS clips."""
    cap = cv2.VideoCapture(row['source'])
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open ' + row['source'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        indices = sample_indices(row['frames'], fps, row['start_s'])
        if indices[-1] >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise RuntimeError('Window exceeds source')
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - indices[0]) > 0.1:
            raise RuntimeError('Seek failed')
        decoded = int(indices[0]) - 1
        for index in indices:
            while decoded < index:
                ok = cap.grab()
                if not ok:
                    raise RuntimeError('Incomplete source')
                decoded += 1
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError('Cannot retrieve sampled frame')
            yield (int(index), frame)
    finally:
        cap.release()

def check_frozen():
    frozen = HERE / 'implementation_hashes.json'
    if _resolve_model_artifact(frozen).exists():
        for name, expected in load(frozen).items():
            if digest(HERE / name) != expected:
                raise RuntimeError('Frozen implementation changed: ' + name)
    p = load(HERE / 'protocol.json')
    for path, expected in p['input_hashes'].items():
        if digest(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    for name, expected in p['partition_hashes'].items():
        if digest(HERE / 'partitions' / name) != expected:
            raise RuntimeError('Partition changed')
    if digest(HERE / 'manifest.json') != p['manifest_sha256']:
        raise RuntimeError('Manifest changed')
    return p

def writer_lock(name):
    import msvcrt
    lock = _resolve_model_artifact(HERE / name).open('a+b')
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError('Another writer owns ' + name)
    return lock
