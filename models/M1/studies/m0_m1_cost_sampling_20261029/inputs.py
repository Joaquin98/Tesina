"""Small pure helpers: class costs and strictly selected RGB observations."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(source_directory(experiment_directory(__file__).parents[0] / 'm1_recall_cost_20261027')))
from core import SelectedRGB

def cost_weights(labels, line, scrum):
    y = np.asarray(labels)
    if y.ndim != 1 or not len(y) or (not np.isin(y, [0, 1, 2]).all()) or (not np.isfinite([line, scrum]).all()) or (min(line, scrum) < 1):
        raise ValueError('Invalid labels/costs')
    counts = np.bincount(y.astype(int), minlength=3)
    if (counts == 0).any():
        raise ValueError('All classes required')
    weights = len(y) / (3 * counts) * [line, 1, scrum]
    return weights / np.average(weights, weights=counts)

class RGBSampler:

    def __init__(self, values, stride):
        if stride < 1 or int(stride) != stride:
            raise ValueError('Invalid stride')
        self.n = len(values)
        self.indices = np.unique(np.r_[np.arange(0, self.n, stride), self.n - 1]).astype(np.int32)
        keep = np.zeros(self.n, bool)
        keep[self.indices] = True
        self.reader = SelectedRGB(values, keep)
        grid = np.arange(self.n)
        right = np.searchsorted(self.indices, grid).clip(0, len(self.indices) - 1)
        left = (right - 1).clip(0)
        self.li = self.indices[left]
        self.ri = self.indices[right]
        self.w = ((grid - self.li) / np.maximum(self.ri - self.li, 1)).astype(np.float32)

    def window(self, start, length=175):
        if start < 0 or length < 1 or length > 200 or (start + length > self.n):
            raise ValueError('Invalid window')
        x = np.zeros((200, 2048), np.float32)
        s = slice(start, start + length)
        w = self.w[s, None]
        x[:length] = self.reader[self.li[s]] * (1 - w) + self.reader[self.ri[s]] * w
        return (x, np.arange(200) < length)
