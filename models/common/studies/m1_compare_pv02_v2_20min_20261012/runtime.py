"""Isolated graph inference on the full final-test VF_S01 video, timed end to end.

The original window method is used unchanged. Only the dual model callable is
wrapped. No feature cache is used in end-to-end runs. Python write audit is a
second safeguard, not an OS-level sandbox for native libraries. The old saved
full-video result is a read-only prediction reference, with no timing baseline.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import types
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for key, value in {'CUDA_VISIBLE_DEVICES': '-1', 'TF_CPP_MIN_LOG_LEVEL': '2', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'NUMEXPR_NUM_THREADS': '8', 'PYTHONDONTWRITEBYTECODE': '1', 'YOLO_CONFIG_DIR': str(HERE / 'runtime/ultralytics'), 'KERAS_HOME': str(HERE / 'runtime/keras'), 'TORCH_HOME': str(HERE / 'runtime/torch'), 'MPLCONFIGDIR': str(HERE / 'runtime/matplotlib'), 'XDG_CACHE_HOME': str(HERE / 'runtime/cache'), 'PLAYER_GEOMETRY_POSE_CACHE_ROOT': str(HERE / 'runtime/pose_cache'), 'TEMP': str(HERE / 'runtime/tmp'), 'TMP': str(HERE / 'runtime/tmp')}.items():
    os.environ[key] = value
for name in ['runtime/tmp', 'runtime/keras', 'runtime/ultralytics', 'logs', 'results', 'inputs']:
    (HERE / name).mkdir(parents=True, exist_ok=True)
os.chdir(HERE)
WRITE_EVENTS = []

def audit(event, args):
    paths = []
    if event == 'open':
        path, mode, flags = args
        if isinstance(path, (str, bytes, os.PathLike)) and (mode and any((c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            paths = [path]
    elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'):
        paths = [args[0]]
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        paths = [args[0], args[1]]
    for path in paths:
        absolute = Path(os.fsdecode(path)).resolve()
        if not absolute.is_relative_to(HERE):
            raise PermissionError(f'WRITE BLOCKED by experiment audit: {event} {absolute}')
        WRITE_EVENTS.append({'event': event, 'path': str(absolute)})
sys.addaudithook(audit)

def save(path, value):
    with _resolve_model_artifact(HERE / path).open('x', encoding='utf-8') as file:
        json.dump(value, file, indent=2, ensure_ascii=False)

def snapshot():
    result = {}
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d != '.git' and Path(base, d).resolve() != HERE]
        for name in files:
            p = Path(base, name)
            stat = p.stat()
            result[str(p.relative_to(ROOT))] = [stat.st_size, stat.st_mtime_ns]
    return result

def sha(path):
    h = hashlib.sha256()
    with _resolve_model_artifact(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
