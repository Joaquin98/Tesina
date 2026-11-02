"""Evaluate M1-B for scrum plus M1-H1-D2 for line/other."""
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from .metrics import temporal_metrics
EXPERIMENT_DIR = experiment_directory(__file__)
OUTPUTS_DIR = EXPERIMENT_DIR / 'outputs'
RUNS_DIR = OUTPUTS_DIR / 'runs'
RESULTS_DIR = OUTPUTS_DIR / 'hybrid_m1b_m1h1d2'
LABELS = ('line', 'other', 'scrum')
RULE = 'M1-B alone decides scrum; otherwise M1-H1-D2 decides line vs other'

def _latest_completed_attempt(run_id: str) -> Path:
    attempts = sorted(_model_artifact_glob(RUNS_DIR / run_id, 'attempt_*', recursive=False), reverse=True)
    for attempt in attempts:
        if _resolve_model_artifact(attempt / 'predictions.csv').exists() and _resolve_model_artifact(attempt / 'run.json').exists():
            return attempt
    raise FileNotFoundError(f'No completed attempt found for {run_id}')

def _load_fold(fold_number: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fold = f'PV{fold_number:02d}'
    baseline_dir = _latest_completed_attempt(f'PV-M1-B-NA-{fold}-S0')
    line_dir = _latest_completed_attempt(f'PV-M1-H1-D2-{fold}-S42')
    baseline = pd.read_csv(_resolve_model_artifact(baseline_dir / 'predictions.csv'))
    line_model = pd.read_csv(_resolve_model_artifact(line_dir / 'predictions.csv'))
    metadata = json.loads(_resolve_model_artifact(line_dir / 'run.json').read_text(encoding='utf-8'))
    if not baseline['block_id'].equals(line_model['block_id']):
        raise ValueError(f'Prediction rows are not aligned for {fold}')
    if not baseline['y_true'].equals(line_model['y_true']):
        raise ValueError(f'Ground truth differs between paired runs for {fold}')
    return (baseline, line_model, metadata)

def _evaluate_fold(fold_number: int) -> tuple[dict, dict, pd.DataFrame]:
    fold = f'PV{fold_number:02d}'
    baseline, line_model, metadata = _load_fold(fold_number)
    line_threshold = float(metadata['thresholds']['line'])
    baseline_scrum = baseline['y_final'].eq('scrum').to_numpy()
    line_detected = line_model['p_line'].to_numpy(dtype=float) >= line_threshold
    prediction = np.where(baseline_scrum, 'scrum', np.where(line_detected, 'line', 'other'))
    frame = baseline[['block_id', 'test_group', 'section', 'block_index', 'start_s', 'end_s', 'duration', 'y_true']].copy()
    frame = frame.rename(columns={'test_group': 'group_id'})
    frame['y_final'] = prediction
    frame['p_line'] = line_model['p_line'].to_numpy(dtype=float)
    frame['p_other'] = line_model['p_other'].to_numpy(dtype=float)
    frame['p_scrum'] = baseline['p_scrum'].to_numpy(dtype=float)
    temporal = temporal_metrics(frame)
    block_f1 = f1_score(frame['y_true'], frame['y_final'], labels=list(LABELS), average='macro', zero_division=0)
    balanced_accuracy = balanced_accuracy_score(frame['y_true'], frame['y_final'])
    row = {'configuration': 'M1-B_scrum__M1-H1-D2_line_other', 'rule': RULE, 'fold': fold, 'test_group': str(frame['group_id'].iloc[0]), 'line_threshold_from_development': line_threshold, 'n_blocks': int(len(frame)), 'f1w_macro': temporal['weighted_f1_macro'], 'f1w_line': temporal['line']['weighted_f1'], 'f1w_scrum': temporal['scrum']['weighted_f1'], 'fp_far_hour': temporal['line']['far_false_positives_per_hour'] + temporal['scrum']['far_false_positives_per_hour'], 'fp_far_hour_line': temporal['line']['far_false_positives_per_hour'], 'fp_far_hour_scrum': temporal['scrum']['far_false_positives_per_hour'], 'block_f1_macro_final': float(block_f1), 'block_balanced_accuracy_final': float(balanced_accuracy)}
    details = {**row, 'temporal': temporal, 'block': {'confusion_matrix': confusion_matrix(frame['y_true'], frame['y_final'], labels=list(LABELS)).tolist(), 'label_order': list(LABELS)}}
    return (row, details, frame)

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_dir = RESULTS_DIR / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    for fold_number in range(1, 7):
        row, fold_details, frame = _evaluate_fold(fold_number)
        rows.append(row)
        details.append(fold_details)
        frame.to_csv(predictions_dir / f"{row['fold']}_predictions.csv", index=False)
    folds = pd.DataFrame(rows)
    metric_columns = ['f1w_macro', 'f1w_line', 'f1w_scrum', 'fp_far_hour', 'fp_far_hour_line', 'fp_far_hour_scrum', 'block_f1_macro_final', 'block_balanced_accuracy_final']
    summary = {'configuration': folds['configuration'].iloc[0], 'rule': RULE, 'folds': int(len(folds))}
    for column in metric_columns:
        summary[column] = float(folds[column].mean())
        summary[f'{column}_std'] = float(folds[column].std(ddof=1))
    folds.to_csv(RESULTS_DIR / 'fold_results.csv', index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / 'summary.csv', index=False)
    (RESULTS_DIR / 'details.json').write_text(json.dumps({'status': 'exploratory_post_hoc', 'selection_note': "The specialist combination was selected after inspecting test performance. Thresholds come only from each fold's development group and were not retuned on test.", 'summary': summary, 'folds': details}, indent=2), encoding='utf-8')
    print(pd.DataFrame([summary]).to_string(index=False))
if __name__ == '__main__':
    main()
