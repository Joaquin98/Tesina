"""Mezcla dos ensambles ya entrenados seleccionando el peso en validacion."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from models.M1.config import LABELS, RESULTS_DIR

def calibrate(probabilities: np.ndarray, line_weight: float, scrum_weight: float) -> np.ndarray:
    weights = np.asarray([line_weight, 1.0, scrum_weight], dtype=np.float64)
    adjusted = np.asarray(probabilities, dtype=np.float64) * weights[None, :]
    return adjusted / adjusted.sum(axis=1, keepdims=True)

def probability_columns(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f'prob_{label}' for label in LABELS]].to_numpy(dtype=np.float64)

def train_weight(first: np.ndarray, second: np.ndarray, labels: np.ndarray, step: float) -> tuple[float, float]:
    best_alpha = 0.0
    best_score = -np.inf
    for alpha in np.arange(0.0, 1.0 + step * 0.5, step):
        probabilities = (1.0 - alpha) * first + alpha * second
        score = f1_score(labels, probabilities.argmax(axis=1), average='macro', zero_division=0)
        if score > best_score + 1e-12:
            best_score = float(score)
            best_alpha = float(alpha)
    return (best_alpha, best_score)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-root', required=True)
    parser.add_argument('--second-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--step', type=float, default=0.05)
    parser.add_argument('--line-weight', type=float, default=0.8)
    parser.add_argument('--scrum-weight', type=float, default=0.9)
    args = parser.parse_args()
    first_root = Path(args.first_root)
    second_root = Path(args.second_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for seed in [int(item) for item in args.seeds.split(',')]:
        first_validation = _load_model_result_array(first_root / f'seed_{seed}' / 'validation_probabilities.npz')
        second_validation = _load_model_result_array(second_root / f'seed_{seed}' / 'validation_probabilities.npz')
        first_labels = np.asarray(first_validation['labels'], dtype=np.int32)
        second_labels = np.asarray(second_validation['labels'], dtype=np.int32)
        if not np.array_equal(first_labels, second_labels):
            raise ValueError(f'Las etiquetas de validacion no coinciden para seed={seed}')
        first_val_prob = calibrate(first_validation['probabilities'], args.line_weight, args.scrum_weight)
        second_val_prob = calibrate(second_validation['probabilities'], args.line_weight, args.scrum_weight)
        alpha, validation_f1 = train_weight(first_val_prob, second_val_prob, first_labels, args.step)
        first_test = pd.read_csv(_resolve_model_artifact(first_root / f'seed_{seed}' / 'predictions.csv'))
        second_test = pd.read_csv(_resolve_model_artifact(second_root / f'seed_{seed}' / 'predictions.csv'))
        keys = ['video_name', 'true_label']
        merged = first_test.merge(second_test, on=keys, suffixes=('_first', '_second'), validate='one_to_one')
        first_prob = calibrate(merged[[f'prob_{label}_first' for label in LABELS]].to_numpy(), args.line_weight, args.scrum_weight)
        second_prob = calibrate(merged[[f'prob_{label}_second' for label in LABELS]].to_numpy(), args.line_weight, args.scrum_weight)
        probabilities = (1.0 - alpha) * first_prob + alpha * second_prob
        predictions = probabilities.argmax(axis=1)
        true_labels = merged['true_label'].map({label: index for index, label in enumerate(LABELS)}).to_numpy()
        output_dir = output_root / f'seed_{seed}'
        output_dir.mkdir(parents=True, exist_ok=True)
        output = merged[keys].copy()
        output['predicted_label'] = [LABELS[index] for index in predictions]
        for index, label in enumerate(LABELS):
            output[f'prob_{label}'] = probabilities[:, index]
        output.to_csv(output_dir / 'predictions.csv', index=False)
        result = {'model': 'validation_weighted_ensemble_blend', 'seed': seed, 'alpha_second': alpha, 'validation_f1_macro': validation_f1, 'test_accuracy': float(accuracy_score(true_labels, predictions)), 'f1_macro': float(f1_score(true_labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(true_labels, predictions, average='weighted', zero_division=0)), 'line_weight': args.line_weight, 'scrum_weight': args.scrum_weight}
        (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results.append(result)
        print(result)
    summary = pd.DataFrame(results)
    summary.to_csv(output_root / 'summary.csv', index=False)
    summary.groupby('model', as_index=False).agg(runs=('seed', 'count'), test_accuracy_mean=('test_accuracy', 'mean'), test_accuracy_std=('test_accuracy', lambda values: values.std(ddof=0)), f1_macro_mean=('f1_macro', 'mean'), f1_macro_std=('f1_macro', lambda values: values.std(ddof=0)), f1_weighted_mean=('f1_weighted', 'mean'), f1_weighted_std=('f1_weighted', lambda values: values.std(ddof=0)), alpha_second_mean=('alpha_second', 'mean')).to_csv(output_root / 'aggregate_metrics.csv', index=False)
if __name__ == '__main__':
    main()
