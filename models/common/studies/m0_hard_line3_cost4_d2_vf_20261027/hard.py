import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np

def sample_weights(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    if y.ndim != 1 or pred.shape != y.shape or (not np.isin(y, [0, 1, 2]).all()) or (not np.isin(pred, [0, 1, 2]).all()):
        raise ValueError('invalid labels')
    count = np.bincount(y, minlength=3)
    if (count == 0).any():
        raise ValueError('missing class')
    base = len(y) / (3 * count) * np.array([4, 1, 4])
    hard = (y == 0) & (pred != 0)
    result = base[y] * np.where(hard, 3.0, 1.0)
    return (result / result.mean(), hard)
