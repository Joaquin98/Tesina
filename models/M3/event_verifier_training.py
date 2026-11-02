"""Fold-local training and application of the 208-feature line verifier."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from models.common.event_verifier import event_vector, window_meta_features
from models.common.comparison.metrics import temporal_metrics
from models.common.comparison.protocol import LABELS, LABEL_TO_ID

@dataclass(frozen=True)
class Candidate:
    positions: tuple[int, ...]
    vector: np.ndarray
    target: int

def _runs(mask: np.ndarray):
    start = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield (start, index)
            start = None
    if start is not None:
        yield (start, len(mask))

def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-09, None)
    return values / values.sum(axis=1, keepdims=True)

def candidates(frame: pd.DataFrame, branches: dict[str, np.ndarray], *, historical_weight: float) -> list[Candidate]:
    frame = frame.reset_index(drop=True)
    historical = _normalize(branches['historical'])
    static = _normalize(branches['static'])
    temporal = _normalize(branches['temporal'])
    official = _normalize(historical_weight * historical + (1.0 - historical_weight) * static)
    result = []
    for (_group, _section), rows in frame.groupby(['group_id', 'section'], sort=False):
        positions = rows.sort_values('block_index', kind='mergesort').index.to_numpy(np.int64)
        h, s, t, o = (historical[positions], static[positions], temporal[positions], official[positions])
        features = window_meta_features(h, s, t, historical_weight=historical_weight, static_weight=1.0 - historical_weight)
        for start, stop in _runs(o.argmax(axis=1) == LABEL_TO_ID['line']):
            selected = tuple((int(value) for value in positions[start:stop]))
            target = int((frame.loc[list(selected), 'label'] == 'line').any())
            result.append(Candidate(positions=selected, vector=event_vector(features, h, s, o, t, start, stop), target=target))
    return result

def fit_verifier(frame: pd.DataFrame, branches: dict[str, np.ndarray], *, historical_weight: float, seed: int, output: Path):
    values = candidates(frame, branches, historical_weight=historical_weight)
    labels = np.asarray([item.target for item in values], dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise RuntimeError('The training fold does not contain both positive and negative line candidates')
    model = LogisticRegression(C=0.1, class_weight='balanced', max_iter=5000, random_state=seed).fit(np.asarray([item.vector for item in values], dtype=np.float32), labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return model

def load_verifier(path: Path):
    return joblib.load(_resolve_model_artifact(path))

def apply_verifier(frame: pd.DataFrame, branches: dict[str, np.ndarray], *, model, threshold: float, historical_weight: float, base_prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = frame.reset_index(drop=True)
    result = np.asarray(base_prediction, dtype=np.int64).copy()
    scores = np.full(len(frame), np.nan, dtype=np.float64)
    event_ids = np.full(len(frame), '', dtype=object)
    values = candidates(frame, branches, historical_weight=historical_weight)
    if not values:
        return (result, scores, event_ids)
    matrix = np.asarray([item.vector for item in values], dtype=np.float32)
    classes = np.asarray(getattr(model, 'classes_', [0, 1]))
    positive = int(np.flatnonzero(classes == 1)[0])
    candidate_scores = np.asarray(model.predict_proba(matrix)[:, positive], dtype=float)
    for candidate_index, (candidate, score) in enumerate(zip(values, candidate_scores)):
        positions = np.asarray(candidate.positions, dtype=np.int64)
        scores[positions] = score
        non_scrum = result[positions] != LABEL_TO_ID['scrum']
        result[positions[non_scrum]] = LABEL_TO_ID['other']
        if score >= threshold:
            result[positions] = LABEL_TO_ID['line']
            event_ids[positions] = f'line_{candidate_index:05d}'
    return (result, scores, event_ids)

def select_verifier_threshold(frame: pd.DataFrame, branches: dict[str, np.ndarray], *, model, historical_weight: float, base_prediction: np.ndarray) -> float:
    values = candidates(frame, branches, historical_weight=historical_weight)
    if not values:
        return 1.0
    matrix = np.asarray([item.vector for item in values], dtype=np.float32)
    classes = np.asarray(getattr(model, 'classes_', [0, 1]))
    positive = int(np.flatnonzero(classes == 1)[0])
    scores = np.asarray(model.predict_proba(matrix)[:, positive], dtype=float)
    grid = np.unique(np.r_[0.0, 1.0, scores, scores + 1e-09, np.arange(0.05, 0.951, 0.025)])
    best = None
    for threshold in grid:
        prediction, _score, _events = apply_verifier(frame, branches, model=model, threshold=float(threshold), historical_weight=historical_weight, base_prediction=base_prediction)
        candidate_frame = frame.copy()
        candidate_frame['y_true'] = candidate_frame['label']
        candidate_frame['y_final'] = [LABELS[index] for index in prediction]
        metrics = temporal_metrics(candidate_frame)
        key = (metrics['line']['weighted_f1'], -metrics['line']['far_false_positives_per_hour'], -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]
