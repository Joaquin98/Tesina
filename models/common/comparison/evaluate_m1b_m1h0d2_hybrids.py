"""Evaluate post-hoc hybrids of M1-B and M1-H0-D2 from saved predictions."""
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
RESULTS_DIR = OUTPUTS_DIR / 'hybrid_m1b_m1h0d2'
LABELS = ('line', 'other', 'scrum')

def _run_dir(run_id: str) -> Path:
    attempts = sorted(_model_artifact_glob(RUNS_DIR / run_id, 'attempt_*', recursive=False))
    if not attempts:
        raise FileNotFoundError(f'No completed attempt found for {run_id}')
    return attempts[-1]

def _load_pair(fold_number: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    fold = f'PV{fold_number:02d}'
    baseline_dir = _run_dir(f'PV-M1-B-NA-{fold}-S0')
    historical_dir = _run_dir(f'PV-M1-H0-D2-{fold}-S42')
    baseline = pd.read_csv(_resolve_model_artifact(baseline_dir / 'predictions.csv'))
    historical = pd.read_csv(_resolve_model_artifact(historical_dir / 'predictions.csv'))
    metadata = json.loads(_resolve_model_artifact(historical_dir / 'run.json').read_text(encoding='utf-8'))
    if not baseline['block_id'].equals(historical['block_id']):
        raise ValueError(f'Prediction rows are not aligned for {fold}')
    if not baseline['y_true'].equals(historical['y_true']):
        raise ValueError(f'Ground truth differs between paired runs for {fold}')
    return (baseline, historical, metadata)

def _metric_frame(baseline: pd.DataFrame, historical: pd.DataFrame, prediction: np.ndarray, *, variant: str) -> pd.DataFrame:
    frame = baseline[['block_id', 'test_group', 'section', 'block_index', 'start_s', 'end_s', 'duration', 'y_true']].copy()
    frame = frame.rename(columns={'test_group': 'group_id'})
    frame['y_final'] = prediction
    frame['p_line'] = historical['p_line'].to_numpy(dtype=float)
    frame['p_other'] = historical['p_other'].to_numpy(dtype=float)
    if variant == 'union':
        frame['p_scrum'] = np.maximum(baseline['p_scrum'].to_numpy(dtype=float), historical['p_scrum'].to_numpy(dtype=float))
    else:
        frame['p_scrum'] = baseline['p_scrum'].to_numpy(dtype=float)
    return frame

def _evaluate_fold(fold_number: int, variant: str) -> tuple[dict, pd.DataFrame, dict]:
    fold = f'PV{fold_number:02d}'
    baseline, historical, metadata = _load_pair(fold_number)
    baseline_scrum = baseline['y_final'].eq('scrum').to_numpy()
    line_threshold = float(metadata['thresholds']['line'])
    if variant == 'union':
        prediction = np.where(baseline_scrum, 'scrum', historical['y_final'].to_numpy())
        rule = 'M1-B scrum override; otherwise M1-H0-D2 final prediction'
    elif variant == 'strict':
        historical_line = historical['p_line'].to_numpy(dtype=float) >= line_threshold
        prediction = np.where(baseline_scrum, 'scrum', np.where(historical_line, 'line', 'other'))
        rule = 'M1-B alone decides scrum; otherwise H0-D2 line threshold vs other'
    else:
        raise ValueError(f'Unknown variant: {variant}')
    frame = _metric_frame(baseline, historical, prediction, variant=variant)
    temporal = temporal_metrics(frame)
    block_f1 = f1_score(frame['y_true'], frame['y_final'], labels=list(LABELS), average='macro', zero_division=0)
    balanced_accuracy = balanced_accuracy_score(frame['y_true'], frame['y_final'])
    result = {'variant': variant, 'rule': rule, 'fold': fold, 'test_group': str(frame['group_id'].iloc[0]), 'line_threshold_from_development': line_threshold, 'n_blocks': int(len(frame)), 'f1w_macro': temporal['weighted_f1_macro'], 'f1w_line': temporal['line']['weighted_f1'], 'f1w_scrum': temporal['scrum']['weighted_f1'], 'fp_far_hour': temporal['line']['far_false_positives_per_hour'] + temporal['scrum']['far_false_positives_per_hour'], 'fp_far_hour_line': temporal['line']['far_false_positives_per_hour'], 'fp_far_hour_scrum': temporal['scrum']['far_false_positives_per_hour'], 'block_f1_macro_final': float(block_f1), 'block_balanced_accuracy_final': float(balanced_accuracy), 'line_true_events': temporal['line']['true_events'], 'line_predicted_events': temporal['line']['predicted_events'], 'line_matched_events': temporal['line']['matched_events'], 'scrum_true_events': temporal['scrum']['true_events'], 'scrum_predicted_events': temporal['scrum']['predicted_events'], 'scrum_matched_events': temporal['scrum']['matched_events']}
    details = {'variant': variant, 'fold': fold, 'rule': rule, 'line_threshold_from_development': line_threshold, 'temporal': temporal, 'block': {'f1_macro_final': float(block_f1), 'balanced_accuracy_final': float(balanced_accuracy), 'confusion_matrix': confusion_matrix(frame['y_true'], frame['y_final'], labels=list(LABELS)).tolist(), 'label_order': list(LABELS)}}
    return (result, frame, details)

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_results = []
    all_details = []
    for variant in ('union', 'strict'):
        variant_dir = RESULTS_DIR / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        for fold_number in range(1, 7):
            result, frame, details = _evaluate_fold(fold_number, variant)
            fold_results.append(result)
            all_details.append(details)
            frame.to_csv(variant_dir / f"{result['fold']}_predictions.csv", index=False)
    folds = pd.DataFrame(fold_results)
    numeric_summary = ['f1w_macro', 'f1w_line', 'f1w_scrum', 'fp_far_hour', 'fp_far_hour_line', 'fp_far_hour_scrum', 'block_f1_macro_final', 'block_balanced_accuracy_final']
    summaries = []
    for variant, rows in folds.groupby('variant', sort=False):
        summary = {'variant': variant, 'rule': rows['rule'].iloc[0], 'folds': int(len(rows))}
        for column in numeric_summary:
            summary[column] = float(rows[column].mean())
            summary[f'{column}_std'] = float(rows[column].std(ddof=1))
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    folds.to_csv(RESULTS_DIR / 'fold_results.csv', index=False)
    summary_frame.to_csv(RESULTS_DIR / 'summary.csv', index=False)
    (RESULTS_DIR / 'details.json').write_text(json.dumps({'status': 'exploratory_post_hoc', 'selection_note': 'The hybrid rules were proposed after inspecting test performance. No threshold was tuned on test, but results are exploratory until confirmed on unseen matches.', 'folds': all_details, 'summary': summaries}, indent=2), encoding='utf-8')
    print(summary_frame.to_string(index=False))
if __name__ == '__main__':
    main()
