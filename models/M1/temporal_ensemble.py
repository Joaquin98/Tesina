"""Evalua el ensamble ganador con una RNN de longitud temporal variable."""
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
import keras
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.M1.config import LABELS, RESULTS_DIR
from models.M1.fusion_methods import fuse, geometry_statistics, rgb_test_probabilities, rgb_validation_probabilities
from models.M1.protocol import build_partition
from models.M1.same_rnn import load_arrays
from models.M1.temporal_resolutions import build_model, model_inputs, resample_arrays
VARIANT = 'all_residual'

def save_predictions(path: Path, names: list[str], labels: np.ndarray, probabilities: np.ndarray) -> None:
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame({'video_name': names, 'true_label': [LABELS[index] for index in labels], 'predicted_label': [LABELS[index] for index in predictions]})
    for index, label in enumerate(LABELS):
        frame[f'prob_{label}'] = probabilities[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def temporal_probabilities(seed: int, frame: pd.DataFrame, target_length: int, pose_model: str, rnn_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    arrays, labels, rows = load_arrays(frame, pose_model, VARIANT)
    arrays = resample_arrays(arrays, target_length)
    model = build_model(target_length, int(arrays['pose'].shape[-1]))
    checkpoint = rnn_root / f'frames_{target_length}' / f'seed_{seed}' / 'best.weights.h5'
    if not _resolve_model_artifact(checkpoint).exists():
        raise FileNotFoundError(f'No existe checkpoint temporal: {checkpoint}')
    model.load_weights(_resolve_model_artifact(checkpoint))
    probabilities = model.predict(model_inputs(arrays), verbose=0)
    return (probabilities, labels, rows['video_name'].astype(str).tolist())

def run_seed(seed: int, target_length: int, pose_model: str, trees: int, alpha_step: float, rnn_root: Path, output_root: Path) -> dict:
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
        raise AssertionError('El orden RGB y geometrico no coincide en validacion')
    best_geometry = (-np.inf, 0.0)
    for alpha in np.arange(0.0, 1.0 + alpha_step * 0.5, alpha_step):
        candidate = fuse(rgb_validation, geometry_validation, 'geometric_mean', float(alpha))
        score = f1_score(validation_y, candidate.argmax(axis=1), average='macro', zero_division=0)
        if score > best_geometry[0]:
            best_geometry = (float(score), float(alpha))
    geometry_validation_fused = fuse(rgb_validation, geometry_validation, 'geometric_mean', best_geometry[1])
    geometry_test_fused = fuse(rgb_test, geometry_test, 'geometric_mean', best_geometry[1])
    temporal_validation, temporal_validation_y, temporal_validation_names = temporal_probabilities(seed, validation_df, target_length, pose_model, rnn_root)
    temporal_test, temporal_test_y, temporal_test_names = temporal_probabilities(seed, test_df, target_length, pose_model, rnn_root)
    if temporal_validation_names != validation_names or not np.array_equal(temporal_validation_y, validation_y):
        raise AssertionError('El orden RNN temporal y geometria no coincide en validacion')
    if temporal_test_names != test_names or not np.array_equal(temporal_test_y, test_y):
        raise AssertionError('El orden RNN temporal y geometria no coincide en prueba')
    best_ensemble = (-np.inf, 0.0)
    for alpha in np.arange(0.0, 1.0 + alpha_step * 0.5, alpha_step):
        candidate = (1.0 - alpha) * temporal_validation + alpha * geometry_validation_fused
        score = f1_score(validation_y, candidate.argmax(axis=1), average='macro', zero_division=0)
        if score > best_ensemble[0]:
            best_ensemble = (float(score), float(alpha))
    validation_probabilities = (1.0 - best_ensemble[1]) * temporal_validation + best_ensemble[1] * geometry_validation_fused
    test_probabilities = (1.0 - best_ensemble[1]) * temporal_test + best_ensemble[1] * geometry_test_fused
    output_dir = output_root / f'seed_{seed}'
    save_predictions(output_dir / 'predictions.csv', test_names, test_y, test_probabilities)
    np.savez_compressed(output_dir / 'validation_probabilities.npz', probabilities=validation_probabilities, labels=validation_y)
    predictions = test_probabilities.argmax(axis=1)
    result = {'model': f'all_residual_frames_{target_length}_plus_geometric_mean', 'seed': int(seed), 'target_length': int(target_length), 'alpha_geometry': best_geometry[1], 'alpha_temporal_ensemble': best_ensemble[1], 'validation_f1_macro': best_ensemble[0], 'test_accuracy': float(accuracy_score(test_y, predictions)), 'f1_macro': float(f1_score(test_y, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(test_y, predictions, average='weighted', zero_division=0)), 'pose_model': pose_model, 'protocol': protocol}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', default='50,100')
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--alpha-step', type=float, default=0.01)
    parser.add_argument('--rnn-root', default=str(RESULTS_DIR / 'temporal_resolutions_all_residual'))
    parser.add_argument('--output-root', default=str(RESULTS_DIR / 'temporal_ensemble'))
    args = parser.parse_args()
    lengths = [int(item.strip()) for item in args.lengths.split(',') if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    rnn_root = Path(args.rnn_root)
    output_base = Path(args.output_root)
    all_results = []
    for length in lengths:
        output_root = output_base / f'frames_{length}'
        output_root.mkdir(parents=True, exist_ok=True)
        results = [run_seed(seed, length, args.pose_model, args.trees, args.alpha_step, rnn_root, output_root) for seed in seeds]
        pd.DataFrame(results).to_csv(output_root / 'summary.csv', index=False)
        aggregate = {'model': results[0]['model'], 'runs': len(results), 'test_accuracy_mean': np.mean([item['test_accuracy'] for item in results]), 'test_accuracy_std': np.std([item['test_accuracy'] for item in results]), 'f1_macro_mean': np.mean([item['f1_macro'] for item in results]), 'f1_macro_std': np.std([item['f1_macro'] for item in results]), 'f1_weighted_mean': np.mean([item['f1_weighted'] for item in results]), 'f1_weighted_std': np.std([item['f1_weighted'] for item in results])}
        pd.DataFrame([aggregate]).to_csv(output_root / 'aggregate_metrics.csv', index=False)
        all_results.extend(results)
        print(json.dumps(aggregate, indent=2))
    pd.DataFrame(all_results).to_csv(output_base / 'summary.csv', index=False)
    print(f'Resultados: {output_base}')
if __name__ == '__main__':
    main()
