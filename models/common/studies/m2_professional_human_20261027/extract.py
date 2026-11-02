"""Resumable official M2 global YOLO cadence, original single-frame detector."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import os, sys, json, time, ast, hashlib
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ['YOLO_CONFIG_DIR'] = str(HERE / 'runtime')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np, cv2, torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)
from ultralytics import YOLO
from models.common.spatial import normalize_yolo_source_roi
from models.M1.geometry_features import build_geometry_features
worker = int(sys.argv[1])
assert worker in (0, 1)
OUT = HERE / 'raw_pose'
OUT.mkdir(exist_ok=True)
start = time.perf_counter()

def save(p, v):
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(v, indent=2), encoding='utf-8')
    t.replace(p)

def progress(**v):
    save(HERE / f'extract_{worker}_progress.json', dict(worker=worker, elapsed_s=time.perf_counter() - start, updated=time.time(), **v))

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()
source = _resolve_model_artifact(ROOT / 'models/M1/pose_cache.py')
weights = ROOT / 'yolo11n-pose.pt'
hashes = {str(p): sha(p) for p in (source, weights, Path(__file__), _resolve_model_artifact(ROOT / 'models/common/spatial.py'))}
tree = ast.parse(_resolve_model_artifact(source).read_text(encoding='utf-8-sig'))
node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'detect_frame'))
ns = dict(np=np)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<original detect_frame>', 'exec'), ns)
video = ROOT / 'data/professional_rugby_m1_game1_20261024/video/jQqN_K0o5_I.mp4'
cap = cv2.VideoCapture(str(video))
assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 154161
assert abs(cap.get(cv2.CAP_PROP_FPS) - 25) < 1e-08
idx = np.unique(np.rint(np.arange(0.0, 154161, 1.75)).astype(np.int32).clip(0, 154160))
assert len(idx) == 88092
model = YOLO(_resolve_model_artifact(str(weights)))
model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=640, conf=0.25, device='cpu', verbose=False)
torch.set_num_threads(4)
assert torch.get_num_threads() == 4
done = 0
extracted = 0
resumed = 0
for chunk, a in enumerate(range(0, len(idx), 256)):
    if chunk % 2 != worker:
        continue
    z = min(a + 256, len(idx))
    path = OUT / f'{a:06d}_{z:06d}.npz'
    if _resolve_model_artifact(path).exists():
        with _load_model_result_array(path) as d:
            assert np.array_equal(d['indices'], idx[a:z]) and d['boxes'].shape == (z - a, 30, 4)
        resumed += z - a
        done += z - a
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx[a]))
    cursor = int(idx[a])
    raw = []
    tick = time.perf_counter()
    for source_index in idx[a:z]:
        while cursor < int(source_index):
            assert cap.grab()
            cursor += 1
        ok, f = cap.read()
        assert ok
        cursor += 1
        roi = normalize_yolo_source_roi(f, yolo_size=640)
        raw.append(ns['detect_frame'](model, roi, 640, 0.25, 30))
    boxes, conf, kp, kpc, valid = [np.asarray([r[i] for r in raw]) for i in range(5)]
    positions, poses, mask = build_geometry_features(boxes, conf, kp, kpc, valid, person_threshold=0.25, keypoint_threshold=0.2)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, indices=idx[a:z], boxes=boxes, person_conf=conf, keypoints=kp, keypoint_conf=kpc, valid=valid, position_features=positions, pose_features=poses, frame_mask=mask)
    temp.replace(path)
    done += z - a
    extracted += z - a
    progress(stage='extracting_yolo', observations_done=done, new_observations=extracted, resumed_observations=resumed, total_observations=88092, last_chunk=path.name, chunk_s=time.perf_counter() - tick)
cap.release()
assert hashes == {p: sha(Path(p)) for p in hashes}
progress(stage='complete', observations_done=done, new_observations=extracted, resumed_observations=resumed, total_observations=88092, source_hashes_unchanged=True, source_hashes=hashes)
