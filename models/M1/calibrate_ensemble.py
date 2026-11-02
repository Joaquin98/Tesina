"""CalibraciÃ³n de pesos de clase del ensemble usando sÃ³lo validaciÃ³n."""
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
OUTPUT_DIR = RESULTS_DIR / 'ensemble_same_rnn_calibrated'

def apply_weights(probabilities: np.ndarray, line_weight: float, scrum_weight: float) -> np.ndarray:
    values = probabilities * np.asarray([line_weight, 1.0, scrum_weight], dtype=np.float64)[None, :]
    return values / values.sum(axis=1, keepdims=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--step', type=float, default=0.05)
    parser.add_argument('--fixed-line', type=float, default=None)
    parser.add_argument('--fixed-scrum', type=float, default=None)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--input-root', default=str(RESULTS_DIR / 'ensemble_same_rnn'))
    parser.add_argument('--output-root', default=None)
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    input_root = Path(args.input_root)
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = OUTPUT_DIR if args.fixed_line is None or args.fixed_scrum is None else RESULTS_DIR / 'ensemble_same_rnn_calibrated_fixed'
    model_name = 'same_rnn_plus_geometric_mean_calibrated' if input_root.name == 'ensemble_same_rnn' else f'{input_root.name}_calibrated'
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        source_dir = input_root / f'seed_{seed}'
        validation = _load_model_result_array(source_dir / 'validation_probabilities.npz')
        validation_probabilities = validation['probabilities']
        validation_labels = validation['labels']
        test = pd.read_csv(_resolve_model_artifact(source_dir / 'predictions.csv'))
        test_probabilities = test[[f'prob_{label}' for label in LABELS]].to_numpy(dtype=np.float64)
        test_labels = np.asarray([LABELS.index(label) for label in test['true_label']], dtype=np.int32)
        if args.fixed_line is not None and args.fixed_scrum is not None:
            best = (float(f1_score(validation_labels, apply_weights(validation_probabilities, args.fixed_line, args.fixed_scrum).argmax(axis=1), average='macro', zero_division=0)), float(args.fixed_line), float(args.fixed_scrum))
        else:
            best = (-np.inf, 1.0, 1.0)
            values = np.arange(0.8, 1.6 + args.step * 0.5, args.step)
            for line_weight in values:
                for scrum_weight in values:
                    candidate = apply_weights(validation_probabilities, line_weight, scrum_weight)
                    score = f1_score(validation_labels, candidate.argmax(axis=1), average='macro', zero_division=0)
                    if score > best[0]:
                        best = (float(score), float(line_weight), float(scrum_weight))
        calibrated = apply_weights(test_probabilities, best[1], best[2])
        predictions = calibrated.argmax(axis=1)
        output_dir = output_root / f'seed_{seed}'
        output_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({'video_name': test['video_name'], 'true_label': test['true_label'], 'predicted_label': [LABELS[index] for index in predictions]})
        for index, label in enumerate(LABELS):
            frame[f'prob_{label}'] = calibrated[:, index]
        frame.to_csv(output_dir / 'predictions.csv', index=False)
        result = {'model': model_name, 'seed': seed, 'line_weight': best[1], 'scrum_weight': best[2], 'validation_f1_macro': best[0], 'test_accuracy': accuracy_score(test_labels, predictions), 'f1_macro': f1_score(test_labels, predictions, average='macro', zero_division=0), 'f1_weighted': f1_score(test_labels, predictions, average='weighted', zero_division=0)}
        (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results.append(result)
        print(result)
    pd.DataFrame(results).to_csv(output_root / 'summary.csv', index=False)
    aggregate = {'model': model_name, 'runs': len(results)}
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
    row = {'model': model_name, 'baseline': 'm0', 'seeds': ','.join((str(seed) for seed in seeds)), 'bootstrap_samples_per_seed': args.bootstrap_samples}
    for metric, values in deltas.items():
        values = np.asarray(values, dtype=np.float64)
        row[f'{metric}_delta_mean'] = float(values.mean())
        row[f'{metric}_delta_low'] = float(np.quantile(values, 0.025))
        row[f'{metric}_delta_high'] = float(np.quantile(values, 0.975))
    pd.DataFrame([row]).to_csv(output_root / 'paired_bootstrap.csv', index=False)
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
