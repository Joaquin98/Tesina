"""Agregación y bootstrap pareado de los modelos M0--M5."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from models.M1.config import DEFAULT_MODELS, DEFAULT_SEEDS, LABELS, RESULTS_DIR
METRICS = ['test_accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum', 'f1_line', 'f1_other', 'f1_scrum']

def load_results(models: list[str], seeds: list[int]) -> tuple[pd.DataFrame, dict[tuple[str, int], pd.DataFrame]]:
    rows = []
    predictions = {}
    for model_name in models:
        for seed in seeds:
            run_dir = RESULTS_DIR / model_name / f'seed_{seed}'
            metrics_path = run_dir / 'metrics.json'
            predictions_path = run_dir / 'predictions.csv'
            if not _resolve_model_artifact(metrics_path).exists() or not _resolve_model_artifact(predictions_path).exists():
                continue
            rows.append(json.loads(_resolve_model_artifact(metrics_path).read_text(encoding='utf-8')))
            predictions[model_name, seed] = pd.read_csv(_resolve_model_artifact(predictions_path))
    if not rows:
        raise SystemExit('No hay métricas. Ejecutar primero train.py.')
    return (pd.DataFrame(rows), predictions)

def metric_from_predictions(frame: pd.DataFrame, metric: str) -> float:
    true = frame['true_label'].to_numpy()
    pred = frame['predicted_label'].to_numpy()
    if metric == 'test_accuracy':
        return float(np.mean(true == pred))
    if metric == 'f1_macro':
        return float(f1_score(true, pred, labels=list(LABELS), average='macro', zero_division=0))
    if metric == 'f1_weighted':
        return float(f1_score(true, pred, labels=list(LABELS), average='weighted', zero_division=0))
    if metric.startswith('recall_'):
        label = metric.removeprefix('recall_')
        return float(recall_score(true, pred, labels=[label], average='macro', zero_division=0))
    if metric.startswith('f1_'):
        label = metric.removeprefix('f1_')
        return float(f1_score(true, pred, labels=[label], average='macro', zero_division=0))
    raise ValueError(metric)

def bootstrap_against_baseline(predictions: dict[tuple[str, int], pd.DataFrame], models: list[str], seeds: list[int], samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model_name in models:
        if model_name == 'm0':
            continue
        common_seeds = [value for value in seeds if ('m0', value) in predictions and (model_name, value) in predictions]
        if not common_seeds:
            continue
        by_seed = {}
        for run_seed in common_seeds:
            baseline = predictions['m0', run_seed].set_index('video_name')
            candidate = predictions[model_name, run_seed].set_index('video_name')
            common = baseline.index.intersection(candidate.index)
            if not len(common):
                continue
            baseline = baseline.loc[common].reset_index()
            candidate = candidate.loc[common].reset_index()
            by_seed[run_seed] = (baseline, candidate)
        if not by_seed:
            continue
        draws = {metric: [] for metric in METRICS}
        available_seeds = list(by_seed)
        for _ in range(samples):
            selected_seed = int(rng.choice(available_seeds))
            baseline, candidate = by_seed[selected_seed]
            indexes = rng.integers(0, len(baseline), len(baseline))
            base_sample = baseline.iloc[indexes].reset_index(drop=True)
            candidate_sample = candidate.iloc[indexes].reset_index(drop=True)
            for metric in METRICS:
                draws[metric].append(metric_from_predictions(candidate_sample, metric) - metric_from_predictions(base_sample, metric))
        row = {'model': model_name, 'baseline': 'm0', 'seeds': ','.join((str(value) for value in available_seeds)), 'bootstrap_samples': samples}
        for metric, values in draws.items():
            values = np.asarray(values, dtype=np.float64)
            row[f'{metric}_delta_mean'] = float(values.mean())
            row[f'{metric}_delta_low'] = float(np.quantile(values, 0.025))
            row[f'{metric}_delta_high'] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(DEFAULT_MODELS))
    parser.add_argument('--seeds', default=','.join((str(seed) for seed in DEFAULT_SEEDS)))
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20260806)
    args = parser.parse_args()
    models = [item.strip().lower() for item in args.models.split(',') if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    summary, predictions = load_results(models, seeds)
    summary.to_csv(RESULTS_DIR / 'summary_reloaded.csv', index=False)
    aggregate_rows = []
    for model_name, group in summary.groupby('model'):
        row = {'model': model_name, 'runs': len(group)}
        for metric in METRICS:
            if metric in group:
                row[f'{metric}_mean'] = float(group[metric].mean())
                row[f'{metric}_std'] = float(group[metric].std(ddof=0))
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).to_csv(RESULTS_DIR / 'aggregate_metrics_reloaded.csv', index=False)
    bootstrap = bootstrap_against_baseline(predictions, models, seeds, args.bootstrap_samples, args.seed)
    bootstrap.to_csv(RESULTS_DIR / 'paired_bootstrap.csv', index=False)
    print(f"Resumen: {RESULTS_DIR / 'aggregate_metrics_reloaded.csv'}")
    print(f"Bootstrap: {RESULTS_DIR / 'paired_bootstrap.csv'}")
if __name__ == '__main__':
    main()
