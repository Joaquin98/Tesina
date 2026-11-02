"""Combina pose same-RNN con la media geomÃ©trica RGB + RF."""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.M1.config import LABELS, RESULTS_DIR
from models.M1.fusion_methods import fuse, geometry_statistics, rgb_test_probabilities, rgb_validation_probabilities
from models.M1.protocol import build_partition
from models.M1.same_rnn import branch_model, load_arrays, inputs as same_inputs
OUTPUT_DIR = RESULTS_DIR / 'ensemble_same_rnn'

def load_same_rnn_probabilities(seed: int, frame: pd.DataFrame, variant: str, pose_model: str, same_results_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    arrays, labels, rows = load_arrays(frame, pose_model, variant)
    model = branch_model(variant, int(arrays['pose'].shape[-1]))
    checkpoint = same_results_root / variant / f'seed_{seed}' / 'best.weights.h5'
    model.load_weights(_resolve_model_artifact(checkpoint))
    probabilities = model.predict(same_inputs(arrays), verbose=0)
    return (probabilities, labels, rows['video_name'].astype(str).tolist())

def save_predictions(path: Path, names: list[str], labels: np.ndarray, probabilities: np.ndarray) -> None:
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame({'video_name': names, 'true_label': [LABELS[index] for index in labels], 'predicted_label': [LABELS[index] for index in predictions]})
    for index, label in enumerate(LABELS):
        frame[f'prob_{label}'] = probabilities[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def run_seed(seed: int, pose_model: str, trees: int, alpha_step: float, same_variant: str, same_results_root: Path, output_root: Path) -> dict:
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=pose_model)
    train_x, train_y, _ = geometry_statistics(train_df, pose_model, 'both')
    validation_x, validation_y, validation_names = geometry_statistics(validation_df, pose_model, 'both')
    test_x, test_y, test_names = geometry_statistics(test_df, pose_model, 'both')
    rf = RandomForestClassifier(n_estimators=trees, max_features='sqrt', class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(train_x, train_y)
    geometry_validation = rf.predict_proba(validation_x)
    geometry_test = rf.predict_proba(test_x)
    rgb_validation, rgb_validation_y, rgb_validation_names = rgb_validation_probabilities(seed, validation_df, pose_model)
    rgb_test = rgb_test_probabilities(seed, test_names)
    if validation_names != rgb_validation_names or not np.array_equal(validation_y, rgb_validation_y):
        raise AssertionError('El orden RGB y geomÃ©trico no coincide en validaciÃ³n')
    best_geometry = (-np.inf, 0.0)
    for alpha in np.arange(0.0, 1.0 + alpha_step * 0.5, alpha_step):
        candidate = fuse(rgb_validation, geometry_validation, 'geometric_mean', float(alpha))
        score = f1_score(validation_y, candidate.argmax(axis=1), average='macro', zero_division=0)
        if score > best_geometry[0]:
            best_geometry = (float(score), float(alpha))
    geometry_validation_fused = fuse(rgb_validation, geometry_validation, 'geometric_mean', best_geometry[1])
    geometry_test_fused = fuse(rgb_test, geometry_test, 'geometric_mean', best_geometry[1])
    same_validation, same_validation_y, same_validation_names = load_same_rnn_probabilities(seed, validation_df, same_variant, pose_model, same_results_root)
    same_test, same_test_y, same_test_names = load_same_rnn_probabilities(seed, test_df, same_variant, pose_model, same_results_root)
    if same_validation_names != validation_names or not np.array_equal(same_validation_y, validation_y):
        raise AssertionError('El orden same-RNN y geomÃ©trico no coincide en validaciÃ³n')
    if same_test_names != test_names or not np.array_equal(same_test_y, test_y):
        raise AssertionError('El orden same-RNN y geomÃ©trico no coincide en prueba')
    best_ensemble = (-np.inf, 0.0)
    for alpha in np.arange(0.0, 1.0 + alpha_step * 0.5, alpha_step):
        candidate = (1.0 - alpha) * same_validation + alpha * geometry_validation_fused
        score = f1_score(validation_y, candidate.argmax(axis=1), average='macro', zero_division=0)
        if score > best_ensemble[0]:
            best_ensemble = (float(score), float(alpha))
    validation_probabilities = (1.0 - best_ensemble[1]) * same_validation + best_ensemble[1] * geometry_validation_fused
    probabilities = (1.0 - best_ensemble[1]) * same_test + best_ensemble[1] * geometry_test_fused
    output_dir = output_root / f'seed_{seed}'
    save_predictions(output_dir / 'predictions.csv', test_names, test_y, probabilities)
    np.savez(output_dir / 'validation_probabilities.npz', probabilities=validation_probabilities, labels=validation_y)
    predictions = probabilities.argmax(axis=1)
    result = {'model': f'{same_variant}_plus_geometric_mean', 'seed': int(seed), 'alpha_geometry': best_geometry[1], 'alpha_same_rnn_ensemble': best_ensemble[1], 'validation_f1_macro': best_ensemble[0], 'test_accuracy': float(accuracy_score(test_y, predictions)), 'f1_macro': float(f1_score(test_y, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(test_y, predictions, average='weighted', zero_division=0)), 'pose_model': pose_model, 'protocol': protocol}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def bootstrap(results: list[dict], samples: int, output_root: Path) -> None:
    rng = np.random.default_rng(20260808)
    deltas = {'test_accuracy': [], 'f1_macro': [], 'f1_weighted': []}
    for result in results:
        seed = int(result['seed'])
        candidate = pd.read_csv(_resolve_model_artifact(output_root / f'seed_{seed}' / 'predictions.csv'))
        baseline = pd.read_csv(_resolve_model_artifact(RESULTS_DIR / 'm0' / f'seed_{seed}' / 'predictions.csv')).set_index('video_name')
        baseline = baseline.loc[candidate['video_name'].tolist()].reset_index()
        for _ in range(samples):
            indexes = rng.integers(0, len(candidate), len(candidate))
            c = candidate.iloc[indexes].reset_index(drop=True)
            b = baseline.iloc[indexes].reset_index(drop=True)
            deltas['test_accuracy'].append(float(np.mean(c['predicted_label'] == c['true_label']) - np.mean(b['predicted_label'] == b['true_label'])))
            deltas['f1_macro'].append(float(f1_score(c['true_label'], c['predicted_label'], average='macro', zero_division=0) - f1_score(b['true_label'], b['predicted_label'], average='macro', zero_division=0)))
            deltas['f1_weighted'].append(float(f1_score(c['true_label'], c['predicted_label'], average='weighted', zero_division=0) - f1_score(b['true_label'], b['predicted_label'], average='weighted', zero_division=0)))
    row = {'model': results[0]['model'], 'baseline': 'm0', 'seeds': ','.join((str(item['seed']) for item in results)), 'bootstrap_samples_per_seed': samples}
    for metric, values in deltas.items():
        values = np.asarray(values, dtype=np.float64)
        row[f'{metric}_delta_mean'] = float(values.mean())
        row[f'{metric}_delta_low'] = float(np.quantile(values, 0.025))
        row[f'{metric}_delta_high'] = float(np.quantile(values, 0.975))
    pd.DataFrame([row]).to_csv(output_root / 'paired_bootstrap.csv', index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--alpha-step', type=float, default=0.01)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--same-variant', default='pose_concat')
    parser.add_argument('--same-results-root', default=str(RESULTS_DIR / 'same_rnn'))
    parser.add_argument('--output-root', default=str(OUTPUT_DIR))
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    same_results_root = Path(args.same_results_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = [run_seed(seed, args.pose_model, args.trees, args.alpha_step, args.same_variant, same_results_root, output_root) for seed in seeds]
    pd.DataFrame(results).to_csv(output_root / 'summary.csv', index=False)
    aggregate = {'model': results[0]['model'], 'runs': len(results), 'test_accuracy_mean': np.mean([item['test_accuracy'] for item in results]), 'test_accuracy_std': np.std([item['test_accuracy'] for item in results]), 'f1_macro_mean': np.mean([item['f1_macro'] for item in results]), 'f1_macro_std': np.std([item['f1_macro'] for item in results]), 'f1_weighted_mean': np.mean([item['f1_weighted'] for item in results]), 'f1_weighted_std': np.std([item['f1_weighted'] for item in results])}
    pd.DataFrame([aggregate]).to_csv(output_root / 'aggregate_metrics.csv', index=False)
    bootstrap(results, args.bootstrap_samples, output_root)
    print(json.dumps(results, indent=2))
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
