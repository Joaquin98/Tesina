import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, hashlib, os
import numpy as np
from runtime_io import save
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
PARENT = ROOT / 'data/multicrop_d2_stride25_vf_20261026'
LABELS = ['line', 'other', 'scrum']
SEEDS = [17, 42, 2026]
COSTS = [8, 16]

def load(p):
    return json.loads(_resolve_model_artifact(Path(p)).read_text(encoding='utf-8'))

def digest(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(p)).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def check():
    for name, h in load(HERE / 'implementation_hashes.json').items():
        if digest(HERE / name) != h:
            raise RuntimeError('Frozen code changed: ' + name)
    for name, h in load(HERE / 'protocol.json')['input_hashes'].items():
        if digest(name) != h:
            raise RuntimeError('Frozen input changed: ' + name)

def lock(name):
    import msvcrt
    f = _resolve_model_artifact(HERE / name).open('a+b')
    f.seek(0, os.SEEK_END)
    if f.tell() == 0:
        f.write(b'0')
        f.flush()
    f.seek(0)
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        raise RuntimeError('Another process owns ' + name)
    return f
