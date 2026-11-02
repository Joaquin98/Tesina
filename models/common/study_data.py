"""Current dataset inputs and shared numeric features for retained studies."""
from pathlib import Path

import numpy as np

from models.common.legacy_training import DATASET, ROOT, current_frames, cached_rgb

_CSV_NAMES = {
    'train.csv': 'train.csv', 'train_natural.csv': 'train.csv',
    'train_natural_fixed.csv': 'train.csv', 'train_unique.csv': 'train.csv',
    'balanced_train.csv': 'train.csv', 'train_balanced.csv': 'train.csv',
    'validation.csv': 'validation.csv', 'validation_fixed.csv': 'validation.csv',
    'validation_unique.csv': 'validation.csv',
    'test.csv': 'test.csv', 'test_natural.csv': 'test.csv',
    'test_natural_fixed.csv': 'test.csv', 'balanced_test.csv': 'test.csv',
    'test_historical_817.csv': 'test.csv', 'evaluation.csv': 'test.csv',
}


def study_csv_path(path):
    """Translate input partition names without redirecting experiment outputs."""
    if not isinstance(path, (str, Path)):
        return path
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else ROOT/candidate
    parents = {DATASET, ROOT/'data/clips_dataset', ROOT/'models/M0',
               ROOT/'data/class_imbalance_comparison/full/splits'}
    if absolute.parent in parents and absolute.name in _CSV_NAMES:
        return DATASET/_CSV_NAMES[absolute.name]
    return path


class _FeatureView:
    def __init__(self, names, component):
        self.names = names
        self.component = component
        self.shape = (len(names), 200, 2048) if component == 0 else (len(names), 200)
        self.dtype = np.dtype(np.float32 if component == 0 else bool)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, selection):
        tail = ()
        if isinstance(selection, tuple):
            selection, tail = selection[0], selection[1:]
        indexes = np.arange(len(self.names))[selection]
        if np.ndim(indexes) == 0:
            result = cached_rgb(self.names[int(indexes)])[self.component]
            return result[tail] if tail else result
        values = [cached_rgb(self.names[int(index)])[self.component] for index in indexes]
        result = np.asarray(values, dtype=self.dtype) if values else np.empty((0, *self.shape[1:]), dtype=self.dtype)
        return result[(slice(None), *tail)] if tail else result


class RGBFeatureStore:
    """Expose batch-indexable features without rebuilding an aggregate bank."""
    def __init__(self):
        self.names = [str(name) for frame in current_frames() for name in frame.video_name]
        if len(set(self.names)) != len(self.names):
            raise ValueError('Los clips deben pertenecer a particiones independientes')
        self.index = {name: i for i, name in enumerate(self.names)}
        self.features = _FeatureView(self.names, 0)
        self.masks = _FeatureView(self.names, 1)
