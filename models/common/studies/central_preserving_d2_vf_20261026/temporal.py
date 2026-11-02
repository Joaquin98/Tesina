"""Actual sparse observations, never temporal interpolation."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np
STEPS = 8
STRIDE = 25

def sample_count(frames):
    if frames <= 0:
        raise ValueError('Empty clip')
    return len(range(0, min(200, int(frames)), STRIDE))

def sample_indices(frames, fps, start_s):
    sample_count(frames)
    if not np.isfinite(fps) or fps <= 0 or start_s < 0:
        raise ValueError('Invalid source time')
    return np.rint(start_s * fps + np.arange(0, min(200, int(frames)), STRIDE) * fps / 25).astype(int)

def match_window_positions(indices, start):
    wanted = start + np.arange(0, 175, STRIDE)
    pos = np.searchsorted(indices, wanted)
    if (pos >= len(indices)).any() or not np.array_equal(indices[pos], wanted):
        raise ValueError('Missing exact sparse observations')
    return pos
