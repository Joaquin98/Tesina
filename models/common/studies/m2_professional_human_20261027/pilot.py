import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import os, sys, time, json, ast
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ['YOLO_CONFIG_DIR'] = str(HERE / 'runtime')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, cv2, torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)
from ultralytics import YOLO
from models.common.spatial import normalize_yolo_source_roi
tree = ast.parse(_resolve_model_artifact(_resolve_model_artifact(ROOT / 'models/M1/pose_cache.py')).read_text(encoding='utf-8-sig'))
node = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'detect_frame'))
ns = dict(np=np)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<original detector>', 'exec'), ns)
node.name = 'convert_result'
node.args = ast.parse('def f(result,frame,max_people): pass').body[0].args
node.body = node.body[2:]
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<same conversion>', 'exec'), ns)
cap = cv2.VideoCapture(str(ROOT / 'data/professional_rugby_m1_game1_20261024/video/jQqN_K0o5_I.mp4'))
frames = []
for i in [0, 14000, 30000, 50000, 75000, 100000, 125000, 150000]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, f = cap.read()
    assert ok
    frames.append(normalize_yolo_source_roi(f, yolo_size=640))
cap.release()
model = YOLO(_resolve_model_artifact(str(ROOT / 'yolo11n-pose.pt')))
ns['detect_frame'](model, frames[0], 640, 0.25, 30)
t = time.perf_counter()
single = [ns['detect_frame'](model, f, 640, 0.25, 30) for f in frames]
single_s = time.perf_counter() - t
t = time.perf_counter()
raw = model.predict(frames, imgsz=640, conf=0.25, device='cpu', verbose=False)
batched = [ns['convert_result'](r, f, 30) for r, f in zip(raw, frames)]
batch_s = time.perf_counter() - t
errors = [max((float(np.max(np.abs(a.astype(float) - b.astype(float)))) for a, b in zip(s, x))) for s, x in zip(single, batched)]
result = dict(single8_s=single_s, batch8_s=batch_s, errors=errors, valid_masks_equal=all((np.array_equal(s[-1], b[-1]) for s, b in zip(single, batched))), observations=int(len(np.unique(np.rint(np.arange(0.0, 154161, 1.75)).astype(np.int32).clip(0, 154160)))))
(HERE / 'pilot.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
