"""Training costs only; inference always uses the untouched probability argmax."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np
LABELS = ['line', 'other', 'scrum']

def class_weights(y, cost):
    y = np.asarray(y)
    if not np.isfinite(cost) or cost < 1 or y.ndim != 1 or (not np.isin(y, [0, 1, 2]).all()):
        raise ValueError('Invalid labels/cost')
    counts = np.bincount(y.astype(int), minlength=3)
    if (counts == 0).any():
        raise ValueError('Every class must be present')
    w = len(y) / (3 * counts) * np.array([cost, 1, cost])
    return w / np.average(w, weights=counts)

def partitions(rows):
    train = []
    test = []
    for i, r in enumerate(rows):
        if r['domain'] == 'VF':
            if r['split'] != 'test':
                raise ValueError('VF leakage')
            test.append(i)
        else:
            if r['split'] != 'train':
                raise ValueError('Unexpected D2 split')
            train.append(i)
    return (np.array(train, dtype=int), np.array(test, dtype=int))

def decide(p):
    p = np.asarray(p)
    if p.ndim != 2 or p.shape[1] != 3 or (not np.isfinite(p).all()) or (p < 0).any():
        raise ValueError('Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-06)
    return p.argmax(1)

def classification_counts(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    return {k: dict(tp=int(((y == i) & (pred == i)).sum()), fn=int(((y == i) & (pred != i)).sum()), fp=int(((y != i) & (pred == i)).sum()), support=int((y == i).sum())) for i, k in enumerate(LABELS)}
