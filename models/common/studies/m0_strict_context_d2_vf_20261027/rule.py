import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np
GRID = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], dtype=float)
MINIMUM = 20
TARGET = 0.9

def scores(p):
    p = np.asarray(p)
    if p.ndim != 3 or p.shape[1:] != (9, 3) or (not np.isfinite(p).all()) or (p < 0).any():
        raise ValueError('expected probabilities [N,9,3]')
    np.testing.assert_allclose(p.sum(axis=2), 1, atol=1e-05)
    cls = p.argmax(2)
    central = cls[:, 0]
    result = []
    for event in [0, 2]:
        q = np.where(cls[:, 1:] == event, p[:, 1:, event], -1.0)
        result.append(np.sort(q, axis=1)[:, -2])
    return (central, np.stack(result, axis=1))

def select(p, y):
    central, s = scores(p)
    out = []
    audit = []
    for j, event in enumerate([0, 2]):
        tests = []
        picked = None
        for t in GRID:
            m = (central != event) & (s[:, j] >= t)
            n = int(m.sum())
            correct = int((y[m] == event).sum())
            precision = correct / n if n else None
            tests.append(dict(threshold=float(t), n=n, correct=correct, precision=precision))
            if picked is None and n >= MINIMUM and (precision >= TARGET):
                picked = float(t)
        out.append(picked)
        audit.append(dict(event=event, chosen=picked, candidates=tests))
    return (out, audit)

def decide(p, thresholds):
    central, s = scores(p)
    valid = np.zeros_like(s, dtype=bool)
    for j, t in enumerate(thresholds):
        if t is not None:
            valid[:, j] = (s[:, j] >= t) & (central != [0, 2][j])
    score = np.where(valid, s, -1.0)
    chosen = np.where(score[:, 1] >= score[:, 0], 2, 0)
    return np.where(valid.any(axis=1), chosen, central)
