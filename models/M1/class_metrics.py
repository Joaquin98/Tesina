"""Compara mÃ©tricas por clase del candidato y M0."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
from models.M1.config import LABELS, RESULTS_DIR

def class_rows(frame: pd.DataFrame, model: str, seed: int) -> list[dict]:
    report = classification_report(frame['true_label'], frame['predicted_label'], labels=list(LABELS), target_names=list(LABELS), output_dict=True, zero_division=0)
    return [{'model': model, 'seed': seed, 'label': label, 'precision': report[label]['precision'], 'recall': report[label]['recall'], 'f1': report[label]['f1-score']} for label in LABELS]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='ensemble_same_rnn')
    parser.add_argument('--seeds', default='17,42,2026')
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    rows = []
    for seed in seeds:
        candidate = pd.read_csv(_resolve_model_artifact(RESULTS_DIR / args.candidate / f'seed_{seed}' / 'predictions.csv'))
        baseline = pd.read_csv(_resolve_model_artifact(RESULTS_DIR / 'm0' / f'seed_{seed}' / 'predictions.csv'))
        rows.extend(class_rows(baseline, 'm0', seed))
        rows.extend(class_rows(candidate, args.candidate, seed))
    output_dir = RESULTS_DIR / args.candidate
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / 'class_metrics_per_seed.csv', index=False)
    frame.groupby(['model', 'label'], as_index=False).agg(runs=('seed', 'count'), precision_mean=('precision', 'mean'), precision_std=('precision', 'std'), recall_mean=('recall', 'mean'), recall_std=('recall', 'std'), f1_mean=('f1', 'mean'), f1_std=('f1', 'std')).to_csv(output_dir / 'class_metrics_aggregate.csv', index=False)
    print(frame.groupby(['model', 'label'])[['precision', 'recall', 'f1']].mean().to_string())
if __name__ == '__main__':
    main()
