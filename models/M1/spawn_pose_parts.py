from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
py = sys.executable
script = str(Path(__file__).with_name('pose_cache_batch.py'))
out = root / 'data/player_geometry_fusion/pose_cache/frames_50_yolo11s_320'
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['PLAYER_GEOMETRY_LENGTH'] = '50'
env['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(out)
common = [script, '--model', 'yolo11s-pose.pt', '--frame-count', '50', '--image-size', '320', '--batch-size', '32', '--seed', '42']
for i, (offset, limit) in enumerate(((0, 2041), (2041, 2041)), 1):
    log = _resolve_model_artifact(out / f'part{i}.log').open('w', encoding='utf-8')
    err = _resolve_model_artifact(out / f'part{i}.err').open('w', encoding='utf-8')
    subprocess.Popen([py, *common, '--offset', str(offset), '--limit', str(limit)], cwd=root, env=env, stdout=log, stderr=err, creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0), close_fds=True)
    log.close()
    err.close()
print('spawned', flush=True)
