import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np

def class_weights(labels, line_cost):
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or (not np.isin(y, [0, 1, 2]).all()) or (not np.isfinite(line_cost)) or (line_cost < 1):
        raise ValueError('Invalid labels or line cost')
    counts = np.bincount(y.astype(int), minlength=3)
    if (counts == 0).any():
        raise ValueError('All classes required')
    w = len(y) / (3 * counts) * np.array([line_cost, 1.0, 1.0])
    return w / np.average(w, weights=counts)
