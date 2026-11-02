"""Agrega y compara las variantes same-RNN contra M0."""
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
from sklearn.metrics import f1_score
from models.M1.config import RESULTS_DIR
VARIANTS = ('pose_concat', 'pose_projected', 'raw4_projected', 'raw8_projected', 'pose_residual')
METRICS = ('test_accuracy', 'f1_macro', 'f1_weighted')

def load_metrics(variants: list[str], seeds: list[int], results_root: Path) -> pd.DataFrame:
    rows = []
    for variant in variants:
        for seed in seeds:
            path = results_root / variant / f'seed_{seed}' / 'metrics.json'
            if _resolve_model_artifact(path).exists():
                rows.append(json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8')))
    if not rows:
        raise SystemExit('No hay mÃ©tricas same-RNN')
    return pd.DataFrame(rows)

def metric_from_predictions(frame: pd.DataFrame, metric: str) -> float:
    true = frame['true_label']
    pred = frame['predicted_label']
    if metric == 'test_accuracy':
        return float(np.mean(true == pred))
    average = 'macro' if metric == 'f1_macro' else 'weighted'
    return float(f1_score(true, pred, average=average, zero_division=0))

def bootstrap(variants: list[str], seeds: list[int], samples: int, results_root: Path, output: Path) -> None:
    rng = np.random.default_rng(20260808)
    rows = []
    for variant in variants:
        deltas = {metric: [] for metric in METRICS}
        available = []
        for seed in seeds:
            candidate_path = results_root / variant / f'seed_{seed}' / 'predictions.csv'
            baseline_path = RESULTS_DIR / 'm0' / f'seed_{seed}' / 'predictions.csv'
            if not _resolve_model_artifact(candidate_path).exists() or not _resolve_model_artifact(baseline_path).exists():
                continue
            candidate = pd.read_csv(_resolve_model_artifact(candidate_path))
            baseline = pd.read_csv(_resolve_model_artifact(baseline_path)).set_index('video_name')
            baseline = baseline.loc[candidate['video_name'].tolist()].reset_index()
            available.append(seed)
            for _ in range(samples):
                indices = rng.integers(0, len(candidate), len(candidate))
                c = candidate.iloc[indices].reset_index(drop=True)
                b = baseline.iloc[indices].reset_index(drop=True)
                for metric in METRICS:
                    deltas[metric].append(metric_from_predictions(c, metric) - metric_from_predictions(b, metric))
        row = {'model': variant, 'baseline': 'm0', 'seeds': ','.join((str(seed) for seed in available)), 'bootstrap_samples_per_seed': samples}
        for metric, values in deltas.items():
            values = np.asarray(values, dtype=np.float64)
            row[f'{metric}_delta_mean'] = float(values.mean())
            row[f'{metric}_delta_low'] = float(np.quantile(values, 0.025))
            row[f'{metric}_delta_high'] = float(np.quantile(values, 0.975))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(VARIANTS))
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--results-root', default=str(RESULTS_DIR / 'same_rnn'))
    args = parser.parse_args()
    variants = [item.strip() for item in args.models.split(',') if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    output_dir = Path(args.results_root)
    summary = load_metrics(variants, seeds, output_dir)
    rows = []
    for variant, group in summary.groupby('model'):
        rows.append({'model': variant, 'runs': len(group), 'test_accuracy_mean': group['test_accuracy'].mean(), 'test_accuracy_std': group['test_accuracy'].std(ddof=0), 'f1_macro_mean': group['f1_macro'].mean(), 'f1_macro_std': group['f1_macro'].std(ddof=0), 'f1_weighted_mean': group['f1_weighted'].mean(), 'f1_weighted_std': group['f1_weighted'].std(ddof=0)})
    pd.DataFrame(rows).sort_values('model').to_csv(output_dir / 'aggregate_metrics_reloaded.csv', index=False)
    bootstrap(variants, seeds, args.bootstrap_samples, output_dir, output_dir / 'paired_bootstrap.csv')
    print(f"Resumen: {output_dir / 'aggregate_metrics_reloaded.csv'}")
    print(f"Bootstrap: {output_dir / 'paired_bootstrap.csv'}")
if __name__ == '__main__':
    main()
