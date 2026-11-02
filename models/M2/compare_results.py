"""Compare the high-resolution pose ensemble with the current best ensemble."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = experiment_directory(__file__)
DEFAULT_CANDIDATE = EXPERIMENT_DIR / 'outputs' / 'results' / 'ensemble_calibrated_fixed'
DEFAULT_BASELINE = REPO_ROOT / 'data' / 'player_geometry_fusion' / 'results' / 'ensemble_same_rnn_calibrated_fixed'
DEFAULT_OUTPUT = EXPERIMENT_DIR / 'outputs' / 'comparison'
LABELS = ('line', 'other', 'scrum')

def load_pair(candidate_root: Path, baseline_root: Path, seed: int) -> pd.DataFrame:
    candidate = pd.read_csv(_resolve_model_artifact(candidate_root / f'seed_{seed}' / 'predictions.csv'))
    baseline = pd.read_csv(_resolve_model_artifact(baseline_root / f'seed_{seed}' / 'predictions.csv'))
    required = {'video_name', 'true_label', 'predicted_label'}
    if not required.issubset(candidate.columns) or not required.issubset(baseline.columns):
        raise ValueError(f'Prediction files for seed {seed} do not contain the required columns')
    merged = candidate[list(required)].merge(baseline[list(required)], on='video_name', how='inner', suffixes=('_candidate', '_baseline'), validate='one_to_one')
    if len(merged) != len(candidate) or len(merged) != len(baseline):
        raise ValueError(f'Candidate and baseline clips are not identical for seed {seed}')
    if not np.array_equal(merged['true_label_candidate'], merged['true_label_baseline']):
        raise ValueError(f'True labels differ for seed {seed}')
    return merged.sort_values('video_name').reset_index(drop=True)

def global_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {'accuracy': float(accuracy_score(y_true, y_pred)), 'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))}

def class_metrics(y_true: pd.Series, y_pred: pd.Series) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS), zero_division=0)
    return [{'label': label, 'precision': float(precision[index]), 'recall': float(recall[index]), 'f1': float(f1[index]), 'support': int(support[index])} for index, label in enumerate(LABELS)]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument('--baseline-root', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(',') if value.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)
    global_rows: list[dict] = []
    per_class_rows: list[dict] = []
    pairs: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        merged = load_pair(args.candidate_root, args.baseline_root, seed)
        pairs[seed] = merged
        y_true = merged['true_label_candidate']
        for model in ('candidate', 'baseline'):
            prediction = merged[f'predicted_label_{model}']
            global_rows.append({'model': model, 'seed': seed, **global_metrics(y_true, prediction)})
            per_class_rows.extend(({'model': model, 'seed': seed, **row} for row in class_metrics(y_true, prediction)))
    global_frame = pd.DataFrame(global_rows)
    class_frame = pd.DataFrame(per_class_rows)
    global_frame.to_csv(args.output_root / 'metrics_per_seed.csv', index=False)
    class_frame.to_csv(args.output_root / 'class_metrics_per_seed.csv', index=False)
    aggregate_rows = []
    for model, group in global_frame.groupby('model', sort=True):
        row = {'model': model, 'runs': int(len(group))}
        for metric in ('accuracy', 'f1_macro', 'f1_weighted'):
            row[f'{metric}_mean'] = float(group[metric].mean())
            row[f'{metric}_std'] = float(group[metric].std(ddof=0))
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).to_csv(args.output_root / 'aggregate_metrics.csv', index=False)
    class_aggregate = []
    for (model, label), group in class_frame.groupby(['model', 'label'], sort=True):
        row = {'model': model, 'label': label, 'runs': int(len(group))}
        for metric in ('precision', 'recall', 'f1'):
            row[f'{metric}_mean'] = float(group[metric].mean())
            row[f'{metric}_std'] = float(group[metric].std(ddof=0))
        class_aggregate.append(row)
    pd.DataFrame(class_aggregate).to_csv(args.output_root / 'class_metrics_aggregate.csv', index=False)
    metric_names = ['accuracy', 'f1_macro', 'f1_weighted']
    metric_names.extend((f'{metric}_{label}' for label in LABELS for metric in ('precision', 'recall', 'f1')))
    deltas = {metric: [] for metric in metric_names}
    rng = np.random.default_rng(20260810)
    for seed in seeds:
        merged = pairs[seed]
        for _ in range(args.bootstrap_samples):
            indexes = rng.integers(0, len(merged), len(merged))
            sample = merged.iloc[indexes]
            y_true = sample['true_label_candidate']
            candidate_pred = sample['predicted_label_candidate']
            baseline_pred = sample['predicted_label_baseline']
            candidate_global = global_metrics(y_true, candidate_pred)
            baseline_global = global_metrics(y_true, baseline_pred)
            for metric in ('accuracy', 'f1_macro', 'f1_weighted'):
                deltas[metric].append(candidate_global[metric] - baseline_global[metric])
            candidate_class = {row['label']: row for row in class_metrics(y_true, candidate_pred)}
            baseline_class = {row['label']: row for row in class_metrics(y_true, baseline_pred)}
            for label in LABELS:
                for metric in ('precision', 'recall', 'f1'):
                    deltas[f'{metric}_{label}'].append(candidate_class[label][metric] - baseline_class[label][metric])
    bootstrap_row = {'candidate': str(args.candidate_root), 'baseline': str(args.baseline_root), 'seeds': ','.join((str(seed) for seed in seeds)), 'bootstrap_samples_per_seed': int(args.bootstrap_samples)}
    for metric, values in deltas.items():
        array = np.asarray(values, dtype=np.float64)
        bootstrap_row[f'{metric}_delta_mean'] = float(array.mean())
        bootstrap_row[f'{metric}_delta_low'] = float(np.quantile(array, 0.025))
        bootstrap_row[f'{metric}_delta_high'] = float(np.quantile(array, 0.975))
    pd.DataFrame([bootstrap_row]).to_csv(args.output_root / 'paired_bootstrap.csv', index=False)
    (args.output_root / 'comparison_metadata.json').write_text(json.dumps({'candidate': str(args.candidate_root.resolve()), 'baseline': str(args.baseline_root.resolve()), 'seeds': seeds, 'bootstrap_samples_per_seed': args.bootstrap_samples, 'selection_note': 'The high-resolution configuration was frozen before reading test results.'}, indent=2), encoding='utf-8')
    print(f'Comparison written to {args.output_root}')
if __name__ == '__main__':
    main()
