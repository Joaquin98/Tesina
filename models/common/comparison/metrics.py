"""Block-level and event-level metrics defined by the final protocol."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from .protocol import LABELS, LABEL_TO_ID

def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(probabilities, dtype=np.float64), nan=0.0)
    values = np.clip(values, 1e-08, None)
    return values / values.sum(axis=1, keepdims=True)

def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int=15) -> float:
    probabilities = normalize_probabilities(probabilities)
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = prediction == labels
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if selected.any():
            result += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(result)

def block_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = normalize_probabilities(probabilities)
    prediction = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=np.arange(3), zero_division=0)
    one_hot = np.eye(3, dtype=np.float64)[labels]
    return {'n': int(len(labels)), 'accuracy': float(accuracy_score(labels, prediction)), 'balanced_accuracy': float(balanced_accuracy_score(labels, prediction)), 'f1_macro': float(f1_score(labels, prediction, average='macro', zero_division=0)), 'log_loss': float(log_loss(labels, probabilities, labels=np.arange(3))), 'brier_multiclass': float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))), 'ece_15': expected_calibration_error(labels, probabilities), 'confusion_matrix': confusion_matrix(labels, prediction, labels=np.arange(3)).tolist(), 'per_class': {label: {'precision': float(precision[index]), 'recall': float(recall[index]), 'f1': float(f1[index]), 'support': int(support[index])} for index, label in enumerate(LABELS)}}

@dataclass(frozen=True)
class Event:
    group: str
    section: str
    label: str
    start: float
    end: float
    anchor: float
    blocks: tuple[int, ...]

def events_from_blocks(frame: pd.DataFrame, label: str, *, prediction_column: str, probability_column: str) -> list[Event]:
    events = []
    required = {'group_id', 'section', 'block_index', 'start_s', 'end_s', prediction_column, probability_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing event columns: {sorted(missing)}')
    for (group, section), rows in frame.groupby(['group_id', 'section'], sort=False):
        rows = rows.sort_values('block_index', kind='mergesort')
        active = rows[rows[prediction_column] == label]
        if active.empty:
            continue
        chunks = []
        current = []
        previous = None
        for index, row in active.iterrows():
            block = int(row['block_index'])
            if previous is None or block == previous + 1:
                current.append((index, row))
            else:
                chunks.append(current)
                current = [(index, row)]
            previous = block
        if current:
            chunks.append(current)
        for chunk in chunks:
            chunk_rows = pd.DataFrame([row for _index, row in chunk])
            event_start = float(chunk_rows['start_s'].min())
            event_end = float(chunk_rows['end_s'].max())
            if prediction_column == 'y_true':
                anchor = (event_start + event_end) / 2.0
            else:
                best_probability = chunk_rows[probability_column].astype(float).max()
                best = chunk_rows[chunk_rows[probability_column].astype(float) == best_probability].sort_values('start_s').iloc[0]
                anchor = (float(best['start_s']) + float(best['end_s'])) / 2.0
            events.append(Event(group=str(group), section=str(section), label=label, start=event_start, end=event_end, anchor=anchor, blocks=tuple((int(value) for value in chunk_rows['block_index']))))
    return events

def _iou(left: Event, right: Event) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union > 0.0 else 0.0

def match_events(truth: list[Event], prediction: list[Event], *, max_distance: float=35.0) -> list[tuple[int, int, float, float]]:
    """Globally maximize distance credit with one-to-one assignment."""
    if not truth or not prediction:
        return []
    credit = np.zeros((len(truth), len(prediction)), dtype=np.float64)
    distance = np.full_like(credit, np.inf)
    for i, true_event in enumerate(truth):
        for j, predicted_event in enumerate(prediction):
            if (true_event.group, true_event.section) != (predicted_event.group, predicted_event.section):
                continue
            value = abs(true_event.anchor - predicted_event.anchor)
            if value <= max_distance:
                distance[i, j] = value
                credit[i, j] = max(0.0, 1.0 - value / max_distance)
    rows, columns = linear_sum_assignment(-credit)
    return [(int(i), int(j), float(distance[i, j]), float(credit[i, j])) for i, j in zip(rows, columns) if np.isfinite(distance[i, j])]

def _f1(tp: float, predictions: int, truth: int) -> float:
    denominator = predictions + truth
    return float(2.0 * tp / denominator) if denominator else 1.0

def temporal_metrics_for_class(frame: pd.DataFrame, label: str) -> dict:
    true_events = events_from_blocks(frame, label, prediction_column='y_true', probability_column=f'p_{label}')
    predicted_events = events_from_blocks(frame, label, prediction_column='y_final', probability_column=f'p_{label}')
    matches = match_events(true_events, predicted_events)
    weighted_tp = sum((value[3] for value in matches))
    distances = np.asarray([value[2] for value in matches], dtype=np.float64)
    tolerance = {}
    for seconds in (0, 7, 14, 21, 35):
        tp = sum((distance <= seconds for _i, _j, distance, _credit in matches))
        tolerance[str(seconds)] = _f1(float(tp), len(predicted_events), len(true_events))
    iou_scores = {}
    for threshold in (0.1, 0.25, 0.5):
        tp = sum((_iou(true_events[i], predicted_events[j]) >= threshold for i, j, _distance, _credit in matches))
        iou_scores[f'{threshold:.2f}'] = _f1(float(tp), len(predicted_events), len(true_events))
    matched_prediction = {j for _i, j, _d, _w in matches}
    duplicates = 0
    far_false_positives = 0
    for index, predicted in enumerate(predicted_events):
        if index in matched_prediction:
            continue
        nearby = any(((truth.group, truth.section) == (predicted.group, predicted.section) and abs(truth.anchor - predicted.anchor) <= 35.0 for truth in true_events))
        if nearby:
            duplicates += 1
        else:
            far_false_positives += 1
    total_seconds = float(frame.groupby(['group_id', 'section'], sort=False)['end_s'].max().sum())
    fragments = 0
    fusions = 0
    for true_event in true_events:
        overlap_count = sum(((true_event.group, true_event.section) == (predicted.group, predicted.section) and _iou(true_event, predicted) > 0.0 for predicted in predicted_events))
        fragments += max(0, overlap_count - 1)
    for predicted_event in predicted_events:
        overlap_count = sum(((true_event.group, true_event.section) == (predicted_event.group, predicted_event.section) and _iou(true_event, predicted_event) > 0.0 for true_event in true_events))
        fusions += max(0, overlap_count - 1)
    predicted_seconds = sum((event.end - event.start for event in predicted_events))
    return {'true_events': len(true_events), 'predicted_events': len(predicted_events), 'matched_events': len(matches), 'weighted_precision': float(weighted_tp / len(predicted_events)) if predicted_events else 0.0, 'weighted_recall': float(weighted_tp / len(true_events)) if true_events else 0.0, 'weighted_f1': _f1(weighted_tp, len(predicted_events), len(true_events)), 'f1_at_seconds': tolerance, 'f1_at_iou': iou_scores, 'median_error_seconds': float(np.median(distances)) if len(distances) else None, 'p90_error_seconds': float(np.quantile(distances, 0.9)) if len(distances) else None, 'far_false_positives_per_hour': float(far_false_positives / (total_seconds / 3600.0)) if total_seconds else 0.0, 'duplicates': int(duplicates), 'fragmentations': int(fragments), 'fusions': int(fusions), 'predicted_minutes': float(predicted_seconds / 60.0), 'predicted_fraction': float(predicted_seconds / total_seconds) if total_seconds else 0.0}

def temporal_metrics(frame: pd.DataFrame) -> dict:
    result = {label: temporal_metrics_for_class(frame, label) for label in ('line', 'scrum')}
    result['weighted_f1_macro'] = float(np.mean([result['line']['weighted_f1'], result['scrum']['weighted_f1']]))
    return result

def apply_class_thresholds(probabilities: np.ndarray, *, line: float, scrum: float) -> np.ndarray:
    values = normalize_probabilities(probabilities)
    output = np.full(len(values), LABEL_TO_ID['other'], dtype=np.int64)
    line_score = values[:, LABEL_TO_ID['line']] / max(float(line), 1e-08)
    scrum_score = values[:, LABEL_TO_ID['scrum']] / max(float(scrum), 1e-08)
    line_wins = (values[:, 0] >= line) & (line_score >= scrum_score)
    scrum_wins = (values[:, 2] >= scrum) & (scrum_score > line_score)
    output[line_wins] = LABEL_TO_ID['line']
    output[scrum_wins] = LABEL_TO_ID['scrum']
    return output

def threshold_search(development: pd.DataFrame, probabilities: np.ndarray, *, values: np.ndarray | None=None) -> dict:
    values = np.asarray(values if values is not None else np.arange(0.1, 0.951, 0.05))
    best = None
    for line_threshold in values:
        for scrum_threshold in values:
            prediction = apply_class_thresholds(probabilities, line=float(line_threshold), scrum=float(scrum_threshold))
            candidate = development.copy()
            candidate['y_final'] = [LABELS[index] for index in prediction]
            metrics = temporal_metrics(candidate)
            truth = candidate['y_true'].map(LABEL_TO_ID).to_numpy()
            class_f1 = f1_score(truth, prediction, labels=np.arange(3), average=None, zero_division=0)
            key = (metrics['weighted_f1_macro'], min(float(class_f1[LABEL_TO_ID['line']]), float(class_f1[LABEL_TO_ID['scrum']])), -metrics['line']['far_false_positives_per_hour'] - metrics['scrum']['far_false_positives_per_hour'])
            if best is None or key > best['key']:
                best = {'key': key, 'line': float(line_threshold), 'scrum': float(scrum_threshold), 'temporal': metrics}
    return {key: value for key, value in best.items() if key != 'key'}
