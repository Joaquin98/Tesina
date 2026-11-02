"""Evaluate a post-hoc union of the best lineout and scrum specialists.

The lineout specialist is M1-H2-D2 and the scrum specialist is M3-H2-D2.
Each model keeps the threshold selected on the development match of its fold.
The test predictions are never used to fit or tune a parameter.
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
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from .metrics import temporal_metrics
HERE = experiment_directory(__file__)
RUNS = HERE / 'outputs' / 'runs'
OUTPUT = HERE / 'outputs' / 'specialist_union_m1h2_m3h2'
LABELS = ('line', 'other', 'scrum')
LINE_SPECIALIST = ('M1', 'H2', 'D2')
SCRUM_SPECIALIST = ('M3', 'H2', 'D2')

def completed_attempt(run_id: str) -> Path:
    attempts = sorted(_model_artifact_glob(RUNS / run_id, 'attempt_*', recursive=False), reverse=True)
    for attempt in attempts:
        if _resolve_model_artifact(attempt / 'predictions.csv').is_file() and _resolve_model_artifact(attempt / 'run.json').is_file():
            return attempt
    raise FileNotFoundError(f'No completed attempt for {run_id}')

def load_run(model: str, level: str, regime: str, fold_number: int) -> tuple[pd.DataFrame, dict]:
    fold = f'PV{fold_number:02d}'
    attempt = completed_attempt(f'PV-{model}-{level}-{regime}-{fold}-S42')
    predictions = pd.read_csv(_resolve_model_artifact(attempt / 'predictions.csv'))
    metadata = json.loads(_resolve_model_artifact(attempt / 'run.json').read_text(encoding='utf-8'))
    return (predictions, metadata)

def evaluate_fold(fold_number: int) -> tuple[dict, pd.DataFrame, dict]:
    line_model, line_metadata = load_run(*LINE_SPECIALIST, fold_number)
    scrum_model, scrum_metadata = load_run(*SCRUM_SPECIALIST, fold_number)
    for column in ('block_id', 'y_true', 'test_group', 'start_s', 'end_s'):
        if not line_model[column].equals(scrum_model[column]):
            raise ValueError(f'Unaligned {column} in fold {fold_number}')
    line_threshold = float(line_metadata['thresholds']['line'])
    scrum_threshold = float(scrum_metadata['thresholds']['scrum'])
    line_detected = line_model['y_final'].eq('line').to_numpy()
    scrum_detected = scrum_model['y_final'].eq('scrum').to_numpy()
    line_score = line_model['p_line'].to_numpy(float) / line_threshold
    scrum_score = scrum_model['p_scrum'].to_numpy(float) / scrum_threshold
    prediction = np.full(len(line_model), 'other', dtype=object)
    prediction[line_detected & ~scrum_detected] = 'line'
    prediction[scrum_detected & ~line_detected] = 'scrum'
    conflicts = line_detected & scrum_detected
    prediction[conflicts & (line_score >= scrum_score)] = 'line'
    prediction[conflicts & (scrum_score > line_score)] = 'scrum'
    frame = line_model[['block_id', 'test_group', 'section', 'block_index', 'start_s', 'end_s', 'duration', 'y_true']].copy()
    frame = frame.rename(columns={'test_group': 'group_id'})
    frame['y_final'] = prediction
    frame['p_line'] = line_model['p_line'].to_numpy(float)
    frame['p_other'] = np.where(prediction == 'other', 1.0, 0.0)
    frame['p_scrum'] = scrum_model['p_scrum'].to_numpy(float)
    temporal = temporal_metrics(frame)
    row = {'fold': f'PV{fold_number:02d}', 'test_group': str(frame['group_id'].iloc[0]), 'line_threshold': line_threshold, 'scrum_threshold': scrum_threshold, 'conflicting_segments': int(conflicts.sum()), 'f1w_macro': temporal['weighted_f1_macro'], 'f1w_line': temporal['line']['weighted_f1'], 'f1w_scrum': temporal['scrum']['weighted_f1'], 'fp_far_hour': temporal['line']['far_false_positives_per_hour'] + temporal['scrum']['far_false_positives_per_hour'], 'block_f1_macro_final': float(f1_score(frame['y_true'], frame['y_final'], labels=LABELS, average='macro', zero_division=0)), 'block_balanced_accuracy_final': float(balanced_accuracy_score(frame['y_true'], frame['y_final']))}
    return (row, frame, temporal)

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_dir = OUTPUT / 'predictions'
    prediction_dir.mkdir(exist_ok=True)
    rows = []
    details = []
    for fold_number in range(1, 7):
        row, frame, temporal = evaluate_fold(fold_number)
        rows.append(row)
        details.append({**row, 'temporal': temporal})
        frame.to_csv(prediction_dir / f'PV{fold_number:02d}_predictions.csv', index=False)
    folds = pd.DataFrame(rows)
    metrics = ('f1w_macro', 'f1w_line', 'f1w_scrum', 'fp_far_hour', 'block_f1_macro_final', 'block_balanced_accuracy_final')
    summary = {'configuration': 'M1-H2-D2_line__M3-H2-D2_scrum', 'rule': "Union of each specialist's final target-class decisions; simultaneous detections are resolved by probability divided by the fold-specific development threshold", 'folds': 6, 'conflicting_segments': int(folds['conflicting_segments'].sum())}
    for metric in metrics:
        summary[metric] = float(folds[metric].mean())
        summary[f'{metric}_std'] = float(folds[metric].std(ddof=1))
    folds.to_csv(OUTPUT / 'fold_results.csv', index=False)
    pd.DataFrame([summary]).to_csv(OUTPUT / 'summary.csv', index=False)
    (OUTPUT / 'details.json').write_text(json.dumps({'status': 'exploratory_post_hoc', 'selection_note': 'The two specialists were selected after reviewing test results. No threshold or tie-break parameter was fitted on a test match; the result therefore describes a post-hoc analysis and requires confirmation on new matches.', 'summary': summary, 'folds': details}, indent=2), encoding='utf-8')
    print(pd.DataFrame([summary]).to_string(index=False))
    print(folds.to_string(index=False))
if __name__ == '__main__':
    main()
