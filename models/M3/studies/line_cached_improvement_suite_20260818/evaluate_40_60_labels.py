"""Evaluate the saved 40â€“60 minute web result against manual 7-second labels."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import copy
import csv
import json
import math
import sys
from pathlib import Path
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
SESSION = ROOT / 'data' / 'active_learning_dataset' / 'sessions' / '20260818_121149_881514'
MANIFEST = SESSION / 'test_manifest.csv'
RESULT = ROOT / 'data' / 'results' / 'result_20260818_110229.json'
OUTPUT = ROOT / 'data' / 'line_cached_improvement_suite_20260818' / 'outputs' / 'LABELLED_40_60_METRICS.json'
sys.path.insert(0, str(source_directory(ROOT / 'src' / 'web')))
import app
from line_event_verifier import LineEventVerifier
LABELS = ('line', 'scrum')

def load_ground_truth() -> tuple[list[dict], float]:
    with _resolve_model_artifact(MANIFEST).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    events: list[dict] = []
    for row in rows:
        label = row['correct_label']
        if label not in LABELS:
            continue
        start = float(row['start_time'])
        end = float(row['end_time'])
        if events and events[-1]['play_type'] == label and math.isclose(events[-1]['end_time'], start):
            events[-1]['end_time'] = end
            events[-1]['duration'] = end - events[-1]['start_time']
            events[-1]['block_count'] += 1
        else:
            events.append({'play_type': label, 'start_time': start, 'end_time': end, 'duration': end - start, 'block_count': 1})
    duration = max((float(row['end_time']) for row in rows))
    return (events, duration)

def intersection(first: dict, second: dict) -> float:
    return max(0.0, min(float(first['end_time']), float(second['end_time'])) - max(float(first['start_time']), float(second['start_time'])))

def iou(first: dict, second: dict) -> float:
    overlap = intersection(first, second)
    union = max(float(first['end_time']), float(second['end_time'])) - min(float(first['start_time']), float(second['start_time']))
    return overlap / union if union > 0.0 else 0.0

def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

def event_metrics(ground_truth: list[dict], predictions: list[dict], *, minimum_iou: float=0.0) -> dict:
    result: dict = {}
    combined = {'tp': 0, 'fp': 0, 'fn': 0}
    for label in LABELS:
        truth = [item for item in ground_truth if item['play_type'] == label]
        predicted = [item for item in predictions if item['play_type'] == label]
        pairs = []
        for pred_index, pred in enumerate(predicted):
            for truth_index, target in enumerate(truth):
                value = iou(pred, target)
                if value > 0.0 and value + 1e-12 >= minimum_iou:
                    pairs.append((value, pred_index, truth_index))
        pairs.sort(reverse=True)
        used_pred: set[int] = set()
        used_truth: set[int] = set()
        matches = []
        for value, pred_index, truth_index in pairs:
            if pred_index in used_pred or truth_index in used_truth:
                continue
            used_pred.add(pred_index)
            used_truth.add(truth_index)
            matches.append({'prediction': [predicted[pred_index]['start_time'], predicted[pred_index]['end_time']], 'truth': [truth[truth_index]['start_time'], truth[truth_index]['end_time']], 'iou': value})
        tp = len(matches)
        fp = len(predicted) - tp
        fn = len(truth) - tp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        f2 = safe_div(5.0 * precision * recall, 4.0 * precision + recall)
        result[label] = {'ground_truth': len(truth), 'predictions': len(predicted), 'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1, 'f2': f2, 'matches': matches, 'false_positives': [[item['start_time'], item['end_time']] for index, item in enumerate(predicted) if index not in used_pred], 'false_negatives': [[item['start_time'], item['end_time']] for index, item in enumerate(truth) if index not in used_truth]}
        combined['tp'] += tp
        combined['fp'] += fp
        combined['fn'] += fn
    precision = safe_div(combined['tp'], combined['tp'] + combined['fp'])
    recall = safe_div(combined['tp'], combined['tp'] + combined['fn'])
    combined.update({'precision': precision, 'recall': recall, 'f1': safe_div(2.0 * precision * recall, precision + recall), 'f2': safe_div(5.0 * precision * recall, 4.0 * precision + recall)})
    result['combined'] = combined
    return result

def temporal_metrics(ground_truth: list[dict], predictions: list[dict], duration: float) -> dict:
    result = {}
    for label in LABELS:
        truth = [item for item in ground_truth if item['play_type'] == label]
        predicted = [item for item in predictions if item['play_type'] == label]
        step = 1.0
        samples = int(math.ceil(duration / step))
        tp = fp = fn = tn = 0
        for index in range(samples):
            point = min(duration - 1e-06, (index + 0.5) * step)
            actual = any((item['start_time'] <= point < item['end_time'] for item in truth))
            detected = any((item['start_time'] <= point < item['end_time'] for item in predicted))
            if actual and detected:
                tp += 1
            elif detected:
                fp += 1
            elif actual:
                fn += 1
            else:
                tn += 1
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        result[label] = {'tp_seconds': tp, 'fp_seconds': fp, 'fn_seconds': fn, 'tn_seconds': tn, 'precision': precision, 'recall': recall, 'f1': safe_div(2.0 * precision * recall, precision + recall)}
    return result

def merge_overlapping_label(predictions: list[dict], label: str) -> list[dict]:
    selected = sorted((copy.deepcopy(item) for item in predictions if item['play_type'] == label), key=lambda item: float(item['start_time']))
    merged: list[dict] = []
    for item in selected:
        if merged and float(item['start_time']) <= float(merged[-1]['end_time']):
            merged[-1]['end_time'] = max(float(merged[-1]['end_time']), float(item['end_time']))
            merged[-1]['start_time'] = min(float(merged[-1]['start_time']), float(item['start_time']))
            merged[-1]['duration'] = merged[-1]['end_time'] - merged[-1]['start_time']
            merged[-1]['confidence'] = max(float(merged[-1].get('confidence', 0.0)), float(item.get('confidence', 0.0)))
        else:
            merged.append(item)
    return [item for item in predictions if item['play_type'] != label] + merged
ground_truth, duration = load_ground_truth()
payload = json.loads(_resolve_model_artifact(RESULT).read_text(encoding='utf-8'))
current = list(payload['detections'])
scrums = [item for item in current if item['play_type'] == 'scrum']
baseline_payload = copy.deepcopy(payload)
baseline_payload.get('event_post_processing', {}).pop('line_mode', None)
baseline_payload.get('metadata', {}).get('event_post_processing', {}).pop('line_mode', None)
baseline = app._rethreshold_result_payload(baseline_payload, 0.7, 0.95)['detections']
artifact = ROOT / payload['event_post_processing']['line_verifier_artifact']
verifier = LineEventVerifier(artifact, threshold=0.0)
all_line_candidates, _ = verifier.verify(payload['window_predictions'], video_duration=duration, source_fps=float(payload.get('source_fps') or payload.get('metadata', {}).get('source_fps') or 25.0))
thresholds = sorted({0.0, 0.11, 1.0} | {round(float(item['line_verifier_score']), 12) for item in all_line_candidates} | {round(float(item['line_verifier_score']) + 1e-09, 12) for item in all_line_candidates})
sweep = []
for threshold in thresholds:
    lines = [item for item in all_line_candidates if float(item['line_verifier_score']) >= threshold]
    predictions = sorted(scrums + lines, key=lambda item: float(item['start_time']))
    metrics = event_metrics(ground_truth, predictions)
    sweep.append({'threshold': threshold, 'line_predictions': len(lines), 'line': {key: metrics['line'][key] for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'f2')}, 'combined': metrics['combined']})
best_line_f1 = max(sweep, key=lambda item: (item['line']['f1'], item['line']['recall'], -item['line']['fp'], item['threshold']))
full_recall = [item for item in sweep if math.isclose(item['line']['recall'], 1.0)]
best_full_recall = max(full_recall, key=lambda item: (item['line']['precision'], item['threshold'])) if full_recall else None
best_combined_f1 = max(sweep, key=lambda item: (item['combined']['f1'], item['combined']['recall'], -item['combined']['fp'], item['threshold']))
best_threshold = float(best_line_f1['threshold'])
best_threshold_lines = [item for item in all_line_candidates if float(item['line_verifier_score']) >= best_threshold]
merged_scrums = merge_overlapping_label(scrums, 'scrum')
merged_scrums = [item for item in merged_scrums if item['play_type'] == 'scrum']
post_hoc_variants = {}
for name, lines in {'threshold_0_11_plus_scrum_dedup': [item for item in current if item['play_type'] == 'line'], 'threshold_0_168_plus_scrum_dedup': best_threshold_lines, 'threshold_0_168_scrum_dedup_reject_line_over_28s': [item for item in best_threshold_lines if float(item['duration']) <= 28.0]}.items():
    predictions = sorted(lines + merged_scrums, key=lambda item: float(item['start_time']))
    post_hoc_variants[name] = {'event_any_overlap': event_metrics(ground_truth, predictions), 'temporal_1s': temporal_metrics(ground_truth, predictions, duration), 'prediction_count': len(predictions)}
output = {'source': {'session_id': SESSION.name, 'result_id': '20260818_110229', 'duration_seconds': duration, 'manual_block_seconds': 7.0, 'ground_truth_events': ground_truth}, 'current_threshold_0_11': {'event_any_overlap': event_metrics(ground_truth, current), 'event_iou_0_10': event_metrics(ground_truth, current, minimum_iou=0.1), 'event_iou_0_30': event_metrics(ground_truth, current, minimum_iou=0.3), 'event_iou_0_50': event_metrics(ground_truth, current, minimum_iou=0.5), 'temporal_1s': temporal_metrics(ground_truth, current, duration)}, 'previous_postprocess': {'event_any_overlap': event_metrics(ground_truth, baseline), 'temporal_1s': temporal_metrics(ground_truth, baseline, duration)}, 'verifier_candidates': all_line_candidates, 'post_hoc_threshold_search': {'best_line_f1': best_line_f1, 'best_line_precision_with_full_recall': best_full_recall, 'best_combined_f1': best_combined_f1, 'sweep': sweep, 'warning': 'Post-hoc diagnostic only: thresholds selected on this segment are not independently validated.'}, 'post_hoc_rule_diagnostics': {'variants': post_hoc_variants, 'warning': 'Exploratory rules measured on the same labelled segment; they require validation on a separate labelled match or holdout.'}}
OUTPUT.write_text(json.dumps(output, indent=2), encoding='utf-8')
print(json.dumps({'output': str(OUTPUT), 'current': output['current_threshold_0_11']['event_any_overlap'], 'previous': output['previous_postprocess']['event_any_overlap'], 'best_threshold': best_line_f1, 'rule_diagnostics': post_hoc_variants}, indent=2))
