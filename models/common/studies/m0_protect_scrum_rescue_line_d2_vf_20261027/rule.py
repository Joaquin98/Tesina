import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np

def decide(p, threshold):
    p = np.asarray(p)
    if p.ndim != 3 or p.shape[1:] != (9, 3) or (not np.isfinite(p).all()) or (p < 0).any() or (not 0 <= threshold <= 1):
        raise ValueError('invalid')
    np.testing.assert_allclose(p.sum(2), 1, atol=1e-05)
    votes = p.argmax(2)
    central = votes[:, 0]
    line_votes = ((votes[:, 1:] == 0) & (p[:, 1:, 0] >= threshold)).sum(1)
    return np.where((central == 1) & (line_votes >= 2), 0, central)
