"""Frozen, training-only hard-negative selection and effective sample weights."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np

def hard_negative_weights(labels, predictions, class_weights, factor):
    y = np.asarray(labels)
    p = np.asarray(predictions)
    c = np.asarray(class_weights, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0 or (not np.isin(y, [0, 1, 2]).all()) or (not np.isin(p, [0, 1, 2]).all()) or (c.shape != (3,)) or (not np.isfinite(c).all()) or (not (c > 0).all()) or (not np.isfinite(factor)) or (factor < 1):
        raise ValueError('Invalid labels, predictions, class weights or hard-negative factor')
    hard = (y == 1) & (p == 2)
    values = c[y.astype(int)] * np.where(hard, float(factor), 1.0)
    return (values / values.mean(), hard)
