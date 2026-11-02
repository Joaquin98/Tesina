"""Shared training readers, feature transforms and prediction utilities.

ResNet-50 and YOLO are never imported or trained here.  Every trainable path
starts from their persisted cache files.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[1]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import copy
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_class_weight
from models.common.comparison.data import align_sequence, compact_features, compact_matrix, dynamic_matrix, fit_dynamic_pca, geometry_statistics, labels_for, load_m1_pose32, load_resnet, load_yolo
from models.common.comparison.metrics import normalize_probabilities
from models.common.comparison.protocol import LABELS
ROOT = Path(__file__).resolve().parents[1]
M0_PATH = ROOT / 'models/M0/checkpoints/best_model/best_model.keras'
M4_PATH = ROOT / 'data/historical_primera_grouped_20260818/outputs/historical_primera.keras'
M2_DEPLOYMENT = ROOT / 'data/player_geometry_highres_pose/outputs/deployment_quantile_100frames'
M2_PCA = ROOT / 'data/joint_multimodal_20260813/rgb_dynamics_seed42/dynamic_pca.joblib'
M3_TEMPORAL = ROOT / 'data/player_geometry_highres_pose/outputs/temporal_external_primera_20260815/temporal_models'
M3_VERIFIER = ROOT / 'data/line_cached_improvement_suite_20260818/outputs/stable_event_logreg_c0_1_pending_external_match.joblib'
M4_VERIFIER = ROOT / 'data/historical_primera_calibrated_20260818/outputs/line_verifier_historical_primera.joblib'
_TEMPORAL_ENSEMBLE_CACHE: dict[str, np.ndarray] = {}
_FROZEN_KERAS_MODELS: dict[str, object] = {}
_FROZEN_KERAS_PROBABILITY_CACHE: dict[tuple[str, str], np.ndarray] = {}
_STATIC_B_ARTIFACTS: tuple[object, object] | None = None
_STATIC_B_PROBABILITY_CACHE: dict[str, np.ndarray] = {}
M1_ROOT = ROOT / 'data/player_geometry_fusion/results'
M1_EXPORT = M1_ROOT / 'web_export'



@dataclass
class PredictionBundle:
    probabilities: np.ndarray
    development_probabilities: np.ndarray | None = None
    training_probabilities: np.ndarray | None = None
    branches: dict[str, np.ndarray] = field(default_factory=dict)
    development_branches: dict[str, np.ndarray] = field(default_factory=dict)
    training_branches: dict[str, np.ndarray] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _resolve_model_artifact(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

class ResNetCacheSequence(keras.utils.Sequence):
    """Keras-compatible lazy reader for cached ResNet sequences."""

    def __init__(self, frame: pd.DataFrame, *, batch_size: int, shuffle: bool, seed: int, dual: bool=False):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.frame))
        self.dual = bool(dual)
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_index):
        selected = self.indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        rgb, masks, pose, labels = ([], [], [], [])
        for index in selected:
            row = self.frame.iloc[int(index)]
            features, mask, _length = load_resnet(row)
            rgb.append(features)
            masks.append(mask)
            labels.append(int(labels_for(self.frame.iloc[[int(index)]])[0]))
            if self.dual:
                raw = load_yolo(row)
                from models.common.comparison.data import geometry_sequence
                pose.append(geometry_sequence(raw))
        rgb_values = np.asarray(rgb, dtype=np.float32)
        mask_values = np.asarray(masks, dtype=bool)
        if self.dual:
            inputs = {'rgb': rgb_values, 'pose': np.asarray(pose, dtype=np.float32), 'sequence_mask': mask_values}
        else:
            inputs = {'features': rgb_values, 'mask': mask_values}
        return (inputs, np.asarray(labels, dtype=np.int64))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

def _predict_keras(model, frame: pd.DataFrame, *, batch_size: int, dual: bool=False) -> np.ndarray:
    inference_batch_size = max(64, int(batch_size))
    sequence = ResNetCacheSequence(frame, batch_size=inference_batch_size, shuffle=False, seed=0, dual=dual)
    outputs = []
    for batch_index in range(len(sequence)):
        inputs, _labels = sequence[batch_index]
        outputs.append(np.asarray(model.predict_on_batch(inputs)))
    if not outputs:
        return np.empty((0, len(LABELS)), dtype=np.float64)
    return normalize_probabilities(np.concatenate(outputs, axis=0))

def _predict_frozen_keras(base_path: Path, frame: pd.DataFrame, *, batch_size: int) -> np.ndarray:
    """Predict an immutable published checkpoint once per unique block."""
    model_key = str(Path(base_path).resolve())
    keys = frame['block_id'].astype(str).tolist()
    if len(set(keys)) != len(keys):
        raise ValueError('Frozen Keras prediction frame contains duplicate block_id values')
    missing_positions = [index for index, key in enumerate(keys) if (model_key, key) not in _FROZEN_KERAS_PROBABILITY_CACHE]
    if missing_positions:
        model = _FROZEN_KERAS_MODELS.get(model_key)
        if model is None:
            model = keras.models.load_model(_resolve_model_artifact(base_path), compile=False)
            _FROZEN_KERAS_MODELS[model_key] = model
        probabilities = _predict_keras(model, frame.iloc[missing_positions], batch_size=batch_size)
        for position, probability in zip(missing_positions, probabilities, strict=True):
            _FROZEN_KERAS_PROBABILITY_CACHE[model_key, keys[position]] = probability.copy()
    return np.asarray([_FROZEN_KERAS_PROBABILITY_CACHE[model_key, key] for key in keys], dtype=np.float64)

def _class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.arange(3)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return {int(index): float(value) for index, value in zip(classes, weights)}

def _geometric_fuse(left, right, alpha: float) -> np.ndarray:
    values = np.exp((1.0 - alpha) * np.log(np.clip(left, 1e-07, 1.0)) + alpha * np.log(np.clip(right, 1e-07, 1.0)))
    return normalize_probabilities(values)

def _choose_alpha(labels, left, right, *, geometric: bool=False) -> float:
    best = (-1.0, 0.0)
    for alpha in np.arange(0.0, 1.001, 0.05):
        probability = _geometric_fuse(left, right, float(alpha)) if geometric else normalize_probabilities((1.0 - alpha) * left + alpha * right)
        score = f1_score(labels, probability.argmax(1), average='macro', zero_division=0)
        if score > best[0]:
            best = (float(score), float(alpha))
    return best[1]

__all__ = [name for name in globals() if not name.startswith("__")]
