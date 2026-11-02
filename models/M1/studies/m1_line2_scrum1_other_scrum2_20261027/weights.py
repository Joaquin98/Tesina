import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np

def effective_weights(labels, hard_indices):
    y = np.asarray(labels)
    idx = np.asarray(hard_indices)
    if y.ndim != 1 or len(y) == 0 or (not np.isin(y, [0, 1, 2]).all()) or (idx.ndim != 1) or (idx.dtype.kind not in 'iu') or (len(np.unique(idx)) != len(idx)) or (idx < 0).any() or (idx >= len(y)).any():
        raise ValueError('Invalid labels or negative indices')
    counts = np.bincount(y.astype(int), minlength=3)
    if (counts == 0).any() or not (y[idx] == 1).all():
        raise ValueError('All classes required; selected negatives must be other')
    base = len(y) / (3 * counts) * np.array([2.0, 1.0, 1.0])
    weights = base[y.astype(int)]
    weights[idx] *= 2.0
    return weights / weights.mean()
