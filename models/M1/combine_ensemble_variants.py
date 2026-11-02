"""Mezcla dos ensembles ya entrenados usando sólo sus probabilidades guardadas."""
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
from models.M1.calibrate_ensemble import apply_weights

def probabilities(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f'prob_{label}' for label in LABELS]].to_numpy(dtype=np.float64)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-root', default=str(RESULTS_DIR / 'ensemble_same_rnn'))
    parser.add_argument('--right-root', default=str(RESULTS_DIR / 'ensemble_same_rnn_residual'))
    parser.add_argument('--output-root', default=str(RESULTS_DIR / 'ensemble_variant_blend_fixed'))
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--alpha-step', type=float, default=0.01)
    parser.add_argument('--line-weight', type=float, default=0.8)
    parser.add_argument('--scrum-weight', type=float, default=0.9)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    left_root = Path(args.left_root)
    right_root = Path(args.right_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    results = []
    for seed in seeds:
        left_dir = left_root / f'seed_{seed}'
        right_dir = right_root / f'seed_{seed}'
        left_validation = _load_model_result_array(left_dir / 'validation_probabilities.npz')
        right_validation = _load_model_result_array(right_dir / 'validation_probabilities.npz')
        if not np.array_equal(left_validation['labels'], right_validation['labels']):
            raise AssertionError(f'Etiquetas de validación desalineadas para seed={seed}')
        validation_labels = left_validation['labels']
        best = (-np.inf, 0.0)
        for alpha in np.arange(0.0, 1.0 + args.alpha_step * 0.5, args.alpha_step):
            candidate = (1.0 - alpha) * left_validation['probabilities'] + alpha * right_validation['probabilities']
            candidate = apply_weights(candidate, args.line_weight, args.scrum_weight)
            score = f1_score(validation_labels, candidate.argmax(axis=1), average='macro', zero_division=0)
            if score > best[0]:
                best = (float(score), float(alpha))
        validation_mixed = (1.0 - best[1]) * left_validation['probabilities'] + best[1] * right_validation['probabilities']
        validation_mixed = apply_weights(validation_mixed, args.line_weight, args.scrum_weight)
        left_test = pd.read_csv(_resolve_model_artifact(left_dir / 'predictions.csv'))
        right_test = pd.read_csv(_resolve_model_artifact(right_dir / 'predictions.csv'))
        if left_test['video_name'].tolist() != right_test['video_name'].tolist():
            raise AssertionError(f'Predicciones de prueba desalineadas para seed={seed}')
        test_labels = np.asarray([LABELS.index(label) for label in left_test['true_label']], dtype=np.int32)
        mixed = (1.0 - best[1]) * probabilities(left_test) + best[1] * probabilities(right_test)
        mixed = apply_weights(mixed, args.line_weight, args.scrum_weight)
        predictions = mixed.argmax(axis=1)
        output_dir = output_root / f'seed_{seed}'
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({'video_name': left_test['video_name'], 'true_label': left_test['true_label'], 'predicted_label': [LABELS[index] for index in predictions]})
        for index, label in enumerate(LABELS):
            frame[f'prob_{label}'] = mixed[:, index]
        frame.to_csv(output_dir / 'predictions.csv', index=False)
        np.savez(output_dir / 'validation_probabilities.npz', probabilities=validation_mixed, labels=validation_labels)
        result = {'model': 'blend_pose_concat_residual_calibrated', 'seed': seed, 'alpha_residual': best[1], 'line_weight': args.line_weight, 'scrum_weight': args.scrum_weight, 'validation_f1_macro': best[0], 'test_accuracy': float(accuracy_score(test_labels, predictions)), 'f1_macro': float(f1_score(test_labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(test_labels, predictions, average='weighted', zero_division=0))}
        (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results.append(result)
        print(result)
    pd.DataFrame(results).to_csv(output_root / 'summary.csv', index=False)
    aggregate = {'model': results[0]['model'], 'runs': len(results)}
    for metric in ('test_accuracy', 'f1_macro', 'f1_weighted'):
        values = np.asarray([item[metric] for item in results], dtype=np.float64)
        aggregate[f'{metric}_mean'] = float(values.mean())
        aggregate[f'{metric}_std'] = float(values.std())
    pd.DataFrame([aggregate]).to_csv(output_root / 'aggregate_metrics.csv', index=False)
    rng = np.random.default_rng(20260808)
    deltas = {metric: [] for metric in ('test_accuracy', 'f1_macro', 'f1_weighted')}
    for seed in seeds:
        candidate = pd.read_csv(_resolve_model_artifact(output_root / f'seed_{seed}' / 'predictions.csv'))
        baseline = pd.read_csv(_resolve_model_artifact(RESULTS_DIR / 'm0' / f'seed_{seed}' / 'predictions.csv'))
        for _ in range(args.bootstrap_samples):
            indexes = rng.integers(0, len(candidate), len(candidate))
            c = candidate.iloc[indexes].reset_index(drop=True)
            b = baseline.iloc[indexes].reset_index(drop=True)
            deltas['test_accuracy'].append(float(np.mean(c['predicted_label'] == c['true_label']) - np.mean(b['predicted_label'] == b['true_label'])))
            deltas['f1_macro'].append(float(f1_score(c['true_label'], c['predicted_label'], average='macro', zero_division=0) - f1_score(b['true_label'], b['predicted_label'], average='macro', zero_division=0)))
            deltas['f1_weighted'].append(float(f1_score(c['true_label'], c['predicted_label'], average='weighted', zero_division=0) - f1_score(b['true_label'], b['predicted_label'], average='weighted', zero_division=0)))
    row = {'model': results[0]['model'], 'baseline': 'm0', 'seeds': ','.join((str(seed) for seed in seeds)), 'bootstrap_samples_per_seed': args.bootstrap_samples}
    for metric, values in deltas.items():
        values = np.asarray(values, dtype=np.float64)
        row[f'{metric}_delta_mean'] = float(values.mean())
        row[f'{metric}_delta_low'] = float(np.quantile(values, 0.025))
        row[f'{metric}_delta_high'] = float(np.quantile(values, 0.975))
    pd.DataFrame([row]).to_csv(output_root / 'paired_bootstrap.csv', index=False)
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
