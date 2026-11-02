import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np

def decide(p):
    p = np.asarray(p)
    if p.ndim != 3 or p.shape[1:] != (9, 3) or (not np.isfinite(p).all()) or (p < 0).any():
        raise ValueError('Expected probabilities [blocks,9,3]')
    np.testing.assert_allclose(p.sum(2), 1, atol=1e-05)
    votes = p.argmax(2)
    line = (votes == 0).sum(1)
    scrum = (votes == 2).sum(1)
    pred = np.where(line + scrum == 0, 1, np.where(scrum >= line, 2, 0))
    return (pred, np.full(len(p), -1, dtype=int))
