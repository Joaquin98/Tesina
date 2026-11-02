import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def fold_ids(y, groups):
    f = np.full(len(y), -1)
    for j, (tr, va) in enumerate(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(np.zeros(len(y)), y, groups)):
        assert not set(groups[tr]) & set(groups[va])
        assert (f[va] < 0).all()
        f[va] = j
    assert (f >= 0).all()
    return f

def weights(y):
    count = np.bincount(y, minlength=3)
    assert (count > 0).all()
    w = len(y) / (3 * count) * [4, 1, 4]
    return w / np.mean(w[y])

def combine(p, w, b):
    return np.log(np.maximum(p[:, 0, :], 1e-07)) + p.reshape(-1, 27) @ w + b
