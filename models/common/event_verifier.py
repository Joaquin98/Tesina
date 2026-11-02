"""Event-level line-out verifier used by the high-resolution web candidate.

The binary head was fitted on complete line candidate runs built from
non-overlapping seven-second blocks.  The web detector produces a denser
one-second timeline, so this module first aligns it to the audited seven-second
grid and only then builds the exact 208-feature representation used at fit
time.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from typing import Iterable
import joblib
import numpy as np
LABELS = ('line', 'other', 'scrum')

def _normalize(values: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(values, dtype=np.float64), 1e-09, None)
    return result / np.maximum(result.sum(axis=-1, keepdims=True), 1e-12)

def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(probabilities, 1e-09, 1.0)
    return -np.sum(values * np.log(values), axis=1, keepdims=True)

def _margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    return (ordered[:, -1] - ordered[:, -2])[:, None]

def window_meta_features(historical: np.ndarray, static: np.ndarray, temporal: np.ndarray, *, historical_weight: float=0.4, static_weight: float=0.6) -> np.ndarray:
    """Reproduce the 40 window features used when fitting the verifier."""
    historical = _normalize(historical)
    static = _normalize(static)
    temporal = _normalize(temporal)
    weights = np.asarray([historical_weight, static_weight], dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError(f'Invalid historical/static verifier weights: {weights.tolist()}')
    weights /= weights.sum()
    official = _normalize(weights[0] * historical + weights[1] * static)
    branches = [historical, static, official, temporal]
    features = np.concatenate(branches + [np.log(np.clip(value, 1e-08, 1.0)) for value in branches] + [historical - static, official - temporal, *(_entropy(value) for value in branches), *(_margin(value) for value in branches), (historical.argmax(axis=1) != static.argmax(axis=1))[:, None].astype(float), (official.argmax(axis=1) != temporal.argmax(axis=1))[:, None].astype(float)], axis=1).astype(np.float32)
    if features.shape[1] != 40:
        raise ValueError(f'Unexpected verifier window feature count: {features.shape}')
    return features

def _runs(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    start = None
    for index, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield (start, index)
            start = None
    if start is not None:
        yield (start, len(mask))

def event_vector(window_features: np.ndarray, historical: np.ndarray, static: np.ndarray, official: np.ndarray, temporal: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Build the exact 208-feature vector expected by the persisted head."""
    values = window_features[start:stop]
    probabilities = np.concatenate([historical[start:stop], static[start:stop], official[start:stop], temporal[start:stop]], axis=1)
    before = official[max(0, start - 2):start]
    after = official[stop:min(len(official), stop + 2)]
    context = []
    for part in (before, after):
        context.extend((part.mean(axis=0) if len(part) else np.zeros(3)).tolist())
    line_curve = official[start:stop, 0]
    vector = np.concatenate([np.asarray([stop - start, (stop - start) * 7.0], dtype=float), values.mean(axis=0), values.std(axis=0), values.min(axis=0), values.max(axis=0), np.quantile(probabilities, 0.25, axis=0), np.quantile(probabilities, 0.5, axis=0), np.quantile(probabilities, 0.75, axis=0), np.asarray(context, dtype=float), np.asarray([float(np.mean(np.abs(np.diff(line_curve)))) if len(line_curve) > 1 else 0.0, float(np.max(line_curve)), float(np.min(line_curve)), float(np.mean(line_curve))])]).astype(np.float32)
    if vector.shape != (208,):
        raise ValueError(f'Unexpected line event feature shape: {vector.shape}')
    return vector

class LineEventVerifier:
    """Veto false line candidates after the base 40/60 model has classified them."""
    GRID_SECONDS = 7.0

    def __init__(self, artifact: str | Path | object, *, threshold: float=0.11, historical_weight: float=0.4, static_weight: float=0.6):
        self.artifact_path = Path(artifact).resolve() if isinstance(artifact, (str, Path)) else None
        self.model = joblib.load(_resolve_model_artifact(self.artifact_path)) if self.artifact_path is not None else artifact
        self.threshold = float(threshold)
        weights = np.asarray([historical_weight, static_weight], dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError(f'Invalid historical/static verifier weights: {weights.tolist()}')
        weights /= weights.sum()
        self.historical_weight = float(weights[0])
        self.static_weight = float(weights[1])
        expected = getattr(self.model, 'n_features_in_', None)
        if expected is not None and int(expected) != 208:
            raise ValueError(f'Line verifier expects {expected} features instead of 208')

    @staticmethod
    def _audited_grid(windows: list[dict]) -> list[dict]:
        ordered = sorted(windows, key=lambda item: float(item['start_time']))
        if not ordered:
            return []
        origin = float(ordered[0]['start_time'])
        selected = []
        for item in ordered:
            offset = (float(item['start_time']) - origin) / LineEventVerifier.GRID_SECONDS
            if abs(offset - round(offset)) <= 0.075:
                selected.append(item)
        return selected

    @staticmethod
    def _branch_matrix(windows: list[dict], name: str) -> np.ndarray:
        rows = []
        for item in windows:
            branches = item.get('branch_probabilities') or {}
            values = branches.get(name)
            if values is None:
                raise ValueError(f"Window is missing '{name}' branch probabilities")
            if isinstance(values, dict):
                rows.append([float(values.get(label, 0.0)) for label in LABELS])
            else:
                row = np.asarray(values, dtype=float).reshape(-1)
                if row.shape != (3,):
                    raise ValueError(f"Invalid '{name}' probabilities: {row.shape}")
                rows.append(row.tolist())
        return _normalize(np.asarray(rows, dtype=float))

    def verify(self, windows: list[dict], *, video_duration: float, source_fps: float) -> tuple[list[dict], list[dict]]:
        """Return accepted line events and their source audit-grid windows."""
        grid = self._audited_grid(windows)
        if not grid:
            return ([], [])
        historical = self._branch_matrix(grid, 'historical')
        static = self._branch_matrix(grid, 'static')
        temporal = self._branch_matrix(grid, 'temporal')
        official = _normalize(self.historical_weight * historical + self.static_weight * static)
        features = window_meta_features(historical, static, temporal, historical_weight=self.historical_weight, static_weight=self.static_weight)
        candidates = list(_runs(official.argmax(axis=1) == 0))
        if not candidates:
            return ([], [])
        vectors = np.asarray([event_vector(features, historical, static, official, temporal, start, stop) for start, stop in candidates], dtype=np.float32)
        classes = np.asarray(getattr(self.model, 'classes_', [0, 1]))
        positive = int(np.flatnonzero(classes == 1)[0])
        scores = np.asarray(self.model.predict_proba(vectors)[:, positive], dtype=float)
        accepted_events = []
        accepted_windows = []
        for (start, stop), score in zip(candidates, scores):
            if score < self.threshold:
                continue
            source = grid[start:stop]
            accepted_windows.extend(source)
            start_time = max(0.0, float(source[0]['start_time']))
            end_time = min(float(video_duration), float(source[-1]['end_time']))
            average = official[start:stop].mean(axis=0)
            peak_index = start + int(np.argmax(official[start:stop, 0]))
            peak = grid[peak_index]
            accepted_events.append({'play_type': 'line', 'start_time': start_time, 'end_time': end_time, 'duration': max(0.0, end_time - start_time), 'confidence': float(np.max(official[start:stop, 0])), 'mean_confidence': float(np.mean(official[start:stop, 0])), 'frame_start': int(round(start_time * source_fps)) if source_fps > 0 else int(source[0].get('frame_start', 0)), 'frame_end': int(round(end_time * source_fps)) if source_fps > 0 else int(source[-1].get('frame_end', 0)), 'probabilities': {label: float(value) for label, value in zip(LABELS, average)}, 'source_window_count': len(source), 'line_verifier_score': float(score), 'line_verifier_threshold': self.threshold, 'line_verifier_grid_seconds': self.GRID_SECONDS, 'line_historical_weight': self.historical_weight, 'line_static_weight': self.static_weight, 'peak_window': {'start_time': float(peak['start_time']), 'end_time': float(peak['end_time']), 'confidence': float(official[peak_index, 0]), 'original_confidence': float(official[peak_index, 0])}, 'contributing_windows': [{'start_time': float(item['start_time']), 'end_time': float(item['end_time']), 'confidence': float(official[index, 0]), 'original_confidence': float(official[index, 0])} for index, item in enumerate(grid[start:stop], start=start)]})
        return (accepted_events, accepted_windows)
