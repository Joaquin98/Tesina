"""No-human post-processing and mining experiments for the rugby classifier.

Reads existing score/result files and writes analyses under
data/no_human_model_improvements/results/.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
REPO_ROOT = experiment_directory(__file__).parents[1]
BASE_DIR = experiment_directory(__file__)
RESULTS_DIR = BASE_DIR / 'results'
TRAIN_SCORES = REPO_ROOT / 'data' / 'cnn_rnn_detector_fusion' / 'results' / 'train_predictions_with_scores.csv'
TEST_SCORES = REPO_ROOT / 'data' / 'cnn_rnn_detector_fusion' / 'results' / 'test_predictions_with_scores.csv'
JSON_RESULTS_DIR = REPO_ROOT / 'data' / 'results'
LABELS = ['line', 'other', 'scrum']
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
PROB_COLS = ['cnn_p_line', 'cnn_p_other', 'cnn_p_scrum']

@dataclass(frozen=True)
class ThresholdPolicy:
    line_threshold: float
    scrum_threshold: float
    other_margin: float
    temperature: float = 1.0

def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve_model_artifact(path))
    df['true_label'] = df['true_label'].astype(str).str.lower().str.strip()
    return df[df['true_label'].isin(LABELS)].reset_index(drop=True)

def y_true(scores: pd.DataFrame) -> np.ndarray:
    return scores['true_label'].map(LABEL_TO_INDEX).to_numpy(dtype=np.int32)

def probs_from_scores(scores: pd.DataFrame, temperature: float=1.0) -> np.ndarray:
    probs = scores[PROB_COLS].to_numpy(dtype=np.float64)
    return apply_temperature_to_probs(probs, temperature)

def apply_temperature_to_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probs, 1e-08, 1.0)
    logits = np.log(clipped)
    scaled = logits / max(temperature, 1e-06)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)

def baseline_predict(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=1)

def threshold_predict(probs: np.ndarray, policy: ThresholdPolicy) -> np.ndarray:
    line_idx = LABEL_TO_INDEX['line']
    other_idx = LABEL_TO_INDEX['other']
    scrum_idx = LABEL_TO_INDEX['scrum']
    pred = np.argmax(probs, axis=1)
    out = pred.copy()
    for i, cls in enumerate(pred):
        if cls == line_idx:
            margin = probs[i, line_idx] - probs[i, other_idx]
            if probs[i, line_idx] < policy.line_threshold or margin < policy.other_margin:
                out[i] = other_idx
        elif cls == scrum_idx:
            margin = probs[i, scrum_idx] - probs[i, other_idx]
            if probs[i, scrum_idx] < policy.scrum_threshold or margin < policy.other_margin:
                out[i] = other_idx
    return out

def metric_row(name: str, y: np.ndarray, pred: np.ndarray, params: dict | None=None) -> dict:
    row = {'method': name, 'accuracy': round(float(accuracy_score(y, pred)), 6), 'f1_macro': round(float(f1_score(y, pred, average='macro', zero_division=0)), 6), 'f1_weighted': round(float(f1_score(y, pred, average='weighted', zero_division=0)), 6)}
    for label in LABELS:
        idx = LABEL_TO_INDEX[label]
        row[f'precision_{label}'] = round(float(precision_score(y, pred, labels=[idx], average='macro', zero_division=0)), 6)
        row[f'recall_{label}'] = round(float(recall_score(y, pred, labels=[idx], average='macro', zero_division=0)), 6)
    row['params'] = json.dumps(params or {}, sort_keys=True)
    return row

def tune_temperature(scores: pd.DataFrame) -> float:
    y = y_true(scores)
    raw_probs = scores[PROB_COLS].to_numpy(dtype=np.float64)
    best_temp = 1.0
    best_loss = math.inf
    for temp in np.linspace(0.5, 5.0, 91):
        calibrated = apply_temperature_to_probs(raw_probs, float(temp))
        loss = log_loss(y, calibrated, labels=[0, 1, 2])
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)
    return best_temp

def tune_thresholds(scores: pd.DataFrame, temperature: float=1.0) -> ThresholdPolicy:
    y = y_true(scores)
    probs = probs_from_scores(scores, temperature=temperature)
    best_policy = ThresholdPolicy(0.0, 0.0, -1.0, temperature)
    best_score = -1.0
    for line_threshold in np.linspace(0.45, 0.95, 26):
        for scrum_threshold in np.linspace(0.45, 0.95, 26):
            for other_margin in np.linspace(-0.05, 0.35, 17):
                policy = ThresholdPolicy(line_threshold=float(line_threshold), scrum_threshold=float(scrum_threshold), other_margin=float(other_margin), temperature=temperature)
                pred = threshold_predict(probs, policy)
                score = f1_score(y, pred, average='macro', zero_division=0)
                if score > best_score:
                    best_score = float(score)
                    best_policy = policy
    return best_policy

def tune_precision_guard_thresholds(scores: pd.DataFrame, temperature: float=1.0) -> ThresholdPolicy:
    """Find a stricter target policy while keeping macro-F1 close to baseline."""
    y = y_true(scores)
    probs = probs_from_scores(scores, temperature=temperature)
    baseline_f1 = f1_score(y, baseline_predict(probs), average='macro', zero_division=0)
    best_policy = ThresholdPolicy(0.8, 0.8, 0.1, temperature)
    best_score = -1.0
    for line_threshold in np.linspace(0.65, 0.98, 12):
        for scrum_threshold in np.linspace(0.65, 0.98, 12):
            for other_margin in np.linspace(0.0, 0.35, 8):
                policy = ThresholdPolicy(line_threshold=float(line_threshold), scrum_threshold=float(scrum_threshold), other_margin=float(other_margin), temperature=temperature)
                pred = threshold_predict(probs, policy)
                macro_f1 = f1_score(y, pred, average='macro', zero_division=0)
                if macro_f1 < baseline_f1 - 0.02:
                    continue
                score = precision_score(y, pred, labels=[LABEL_TO_INDEX['line']], average='macro', zero_division=0) + precision_score(y, pred, labels=[LABEL_TO_INDEX['scrum']], average='macro', zero_division=0) + recall_score(y, pred, labels=[LABEL_TO_INDEX['other']], average='macro', zero_division=0)
                if score > best_score:
                    best_score = float(score)
                    best_policy = policy
    return best_policy

def prediction_table(scores: pd.DataFrame, policy: ThresholdPolicy, name: str) -> pd.DataFrame:
    probs = probs_from_scores(scores, policy.temperature)
    pred = threshold_predict(probs, policy)
    out = scores[['video_name', 'true_label']].copy()
    out['baseline_pred'] = [LABELS[i] for i in baseline_predict(probs_from_scores(scores, 1.0))]
    out[f'{name}_pred'] = [LABELS[i] for i in pred]
    out[f'{name}_confidence'] = probs.max(axis=1)
    for i, label in enumerate(LABELS):
        out[f'{name}_p_{label}'] = probs[:, i]
    return out

def extract_windows_from_json(path: Path) -> list[dict]:
    try:
        data = json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8'))
    except Exception:
        return []
    windows = data.get('window_predictions')
    if isinstance(windows, list) and windows:
        return [dict(item, result_file=path.name, video_path=data.get('video_path', '')) for item in windows]
    detections = data.get('detections')
    if isinstance(detections, list) and detections:
        return [dict(item, result_file=path.name, video_path=data.get('video_path', '')) for item in detections]
    return []

def row_probs(row: dict) -> dict:
    probs = row.get('probabilities') or {}
    return {label: float(probs.get(label, 0.0) or 0.0) for label in LABELS}

def smooth_windows(windows: list[dict], radius: int=2, min_confidence: float=0.7, min_consecutive: int=2) -> list[dict]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda item: float(item.get('start_time', item.get('window_start', item.get('time', 0.0))) or 0.0))
    matrix = np.array([[row_probs(row)[label] for label in LABELS] for row in ordered], dtype=np.float64)
    smoothed = np.zeros_like(matrix)
    for i in range(len(matrix)):
        lo = max(0, i - radius)
        hi = min(len(matrix), i + radius + 1)
        smoothed[i] = matrix[lo:hi].mean(axis=0)
    candidates = []
    for i, row in enumerate(ordered):
        pred_idx = int(np.argmax(smoothed[i]))
        label = LABELS[pred_idx]
        conf = float(smoothed[i, pred_idx])
        if label == 'other' or conf < min_confidence:
            continue
        start = float(row.get('start_time', row.get('window_start', row.get('time', 0.0))) or 0.0)
        end = float(row.get('end_time', row.get('window_end', start)) or start)
        candidates.append({'play_type': label, 'start_time': start, 'end_time': end, 'confidence': conf})
    merged = []
    for cand in candidates:
        if not merged or cand['play_type'] != merged[-1]['play_type'] or cand['start_time'] > merged[-1]['end_time'] + 2.0:
            merged.append({**cand, 'source_window_count': 1, 'peak_confidence': cand['confidence']})
        else:
            merged[-1]['end_time'] = max(merged[-1]['end_time'], cand['end_time'])
            merged[-1]['source_window_count'] += 1
            merged[-1]['peak_confidence'] = max(merged[-1]['peak_confidence'], cand['confidence'])
    return [event for event in merged if event['source_window_count'] >= min_consecutive]

def analyze_temporal_jsons() -> pd.DataFrame:
    rows = []
    smoothed_payload = {}
    for path in sorted(_model_artifact_glob(JSON_RESULTS_DIR, '*.json', recursive=False)):
        windows = extract_windows_from_json(path)
        if not windows:
            continue
        original_target = [w for w in windows if str(w.get('play_type', '')).lower() in ('line', 'scrum')]
        smoothed = smooth_windows(windows)
        rows.append({'result_file': path.name, 'video_path': windows[0].get('video_path', ''), 'raw_target_windows_or_detections': len(original_target), 'smoothed_events': len(smoothed), 'smoothed_line_events': sum((1 for e in smoothed if e['play_type'] == 'line')), 'smoothed_scrum_events': sum((1 for e in smoothed if e['play_type'] == 'scrum'))})
        smoothed_payload[path.name] = smoothed
    (RESULTS_DIR / 'smoothed_events_by_result.json').write_text(json.dumps(smoothed_payload, indent=2), encoding='utf-8')
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / 'temporal_smoothing_summary.csv', index=False)
    return out

def mine_unlabeled_candidates() -> pd.DataFrame:
    rows = []
    for path in sorted(_model_artifact_glob(JSON_RESULTS_DIR, '*.json', recursive=False)):
        for row in extract_windows_from_json(path):
            probs = row_probs(row)
            sorted_probs = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            top_label, top_prob = sorted_probs[0]
            second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
            margin = top_prob - second_prob
            if top_label == 'other':
                continue
            start = float(row.get('start_time', row.get('window_start', row.get('time', 0.0))) or 0.0)
            end = float(row.get('end_time', row.get('window_end', start)) or start)
            reason = None
            if top_prob >= 0.9 and probs.get('other', 0.0) >= 0.05:
                reason = 'high_conf_target_with_other_mass'
            elif margin <= 0.15:
                reason = 'low_margin_ambiguous_target'
            elif top_label == 'line' and top_prob >= 0.9:
                reason = 'high_conf_line_candidate'
            if reason:
                rows.append({'result_file': path.name, 'video_path': row.get('video_path', ''), 'start_time': start, 'end_time': end, 'predicted_label': top_label, 'confidence': top_prob, 'margin': margin, 'p_line': probs['line'], 'p_other': probs['other'], 'p_scrum': probs['scrum'], 'reason': reason})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['reason', 'confidence'], ascending=[True, False]).head(500)
    out.to_csv(RESULTS_DIR / 'automatic_mining_candidates.csv', index=False)
    return out

def write_report(metrics: pd.DataFrame, smoothing: pd.DataFrame, mined: pd.DataFrame, temp: float, policy: ThresholdPolicy) -> None:
    metric_columns = list(metrics.columns)
    metric_table = ['| ' + ' | '.join(metric_columns) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(metric_columns) - 1)) + '|']
    for _, row in metrics.iterrows():
        metric_table.append('| ' + ' | '.join((str(row[col]) for col in metric_columns)) + ' |')
    lines = ['# No-human model improvement experiments', '', 'This report is additive: no current checkpoint, manifest, or app behavior was replaced.', '', '## Thresholds and calibration', '', f'Best temperature on the training score table: `{temp:.3f}`.', '', 'Best target-threshold policy tuned on training predictions:', '', f'- line threshold: `{policy.line_threshold:.3f}`', f'- scrum threshold: `{policy.scrum_threshold:.3f}`', f'- target-vs-other margin: `{policy.other_margin:.3f}`', '', 'Held-out test-set comparison:', '', '\n'.join(metric_table), '', '## Temporal smoothing', '', 'Saved sliding-window/live result JSON files were post-processed with a moving average over class probabilities and event merging.', '', f'- Result files analyzed: `{len(smoothing)}`', f"- Total raw target windows/detections: `{(int(smoothing['raw_target_windows_or_detections'].sum()) if not smoothing.empty else 0)}`", f"- Total smoothed target events: `{(int(smoothing['smoothed_events'].sum()) if not smoothing.empty else 0)}`", '', 'This is useful mainly for deployment behavior: it reduces repeated adjacent detections and one-window flicker without retraining.', '', '## Automatic mining', '', f'Candidate windows exported: `{len(mined)}`.', '', 'These are not new labels. They are a ranked queue of suspicious or useful windows that can be reviewed later if time allows.', '', '## Recommendation', '', 'Use thresholding/calibration and temporal smoothing as deployment/post-processing controls. Treat automatic mining as a no-human way to find where the current model is probably weak, and only label the top candidates if extra time appears.']
    (RESULTS_DIR / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def main() -> None:
    ensure_results_dir()
    train_scores = load_scores(TRAIN_SCORES)
    test_scores = load_scores(TEST_SCORES)
    train_temp = tune_temperature(train_scores)
    policy_raw = tune_thresholds(train_scores, temperature=1.0)
    policy_calibrated = tune_thresholds(train_scores, temperature=train_temp)
    precision_guard_policy = tune_precision_guard_thresholds(train_scores, temperature=train_temp)
    y_test = y_true(test_scores)
    raw_probs = probs_from_scores(test_scores, 1.0)
    calibrated_probs = probs_from_scores(test_scores, train_temp)
    rows = [metric_row('baseline_argmax', y_test, baseline_predict(raw_probs), {}), metric_row('temperature_only_argmax', y_test, baseline_predict(calibrated_probs), {'temperature': train_temp}), metric_row('threshold_policy', y_test, threshold_predict(raw_probs, policy_raw), policy_raw.__dict__), metric_row('temperature_plus_threshold_policy', y_test, threshold_predict(calibrated_probs, policy_calibrated), policy_calibrated.__dict__), metric_row('precision_guard_policy', y_test, threshold_predict(calibrated_probs, precision_guard_policy), precision_guard_policy.__dict__)]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / 'threshold_calibration_metrics.csv', index=False)
    prediction_table(test_scores, policy_raw, 'threshold_policy').to_csv(RESULTS_DIR / 'threshold_policy_test_predictions.csv', index=False)
    prediction_table(test_scores, policy_calibrated, 'temperature_threshold_policy').to_csv(RESULTS_DIR / 'temperature_threshold_policy_test_predictions.csv', index=False)
    smoothing = analyze_temporal_jsons()
    mined = mine_unlabeled_candidates()
    write_report(metrics, smoothing, mined, train_temp, policy_calibrated)
    print(metrics.to_string(index=False))
    print(f"Report: {RESULTS_DIR / 'REPORT.md'}")
if __name__ == '__main__':
    main()
