"""Entrenamiento RNN de los tres finalistas del screening YOLO--ResNet.

Los finalistas se eligieron por macro-F1 de validacion en ``run_suite.py``.
La rama RGB+pose conserva la adaptacion residual inicializada desde M0; el
vector estatico de cada experimento entra por una rama densa regularizada.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data' / 'player_geometry_fusion' / 'pose_cache' / 'frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PLAYER_DIR = REPO / 'data' / 'player_geometry_fusion'
if str(PLAYER_DIR) not in sys.path:
    sys.path.insert(0, str(source_directory(PLAYER_DIR)))
from models.M1.config import LABELS, LABEL_TO_ID
from models.M1.protocol import build_partition
from models.M1.same_rnn import MacroF1Checkpoint, load_arrays
from models.M1.stats_rnn import initialize_from_m0
from models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite import EXPERIMENTS, feature_matrix
from models.M1.targeted_stats_rnn import model_inputs
from models.M1.targeted_stats_rnn import build_model as build_targeted_model
FINALISTS = ('E04_tracking_smoothing', 'E03_roi_topk', 'E06_rugby_descriptors')
POSE_MODEL = 'yolo11n-pose.pt'
SEQUENCE_VARIANT = 'all_residual'

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def add_stats(arrays: dict[str, np.ndarray], values: np.ndarray) -> dict[str, np.ndarray]:
    result = dict(arrays)
    result['targeted_geometry_stats'] = values.astype(np.float32)
    return result

def load_rnn_arrays(df: pd.DataFrame, pose_model: str):
    arrays, labels, rows = load_arrays(df, pose_model, SEQUENCE_VARIANT)
    if rows['video_name'].astype(str).tolist() != df['video_name'].astype(str).tolist():
        raise AssertionError('El orden de RGB y de las features YOLO no coincide')
    return (arrays, labels, rows)

def score(labels: np.ndarray, probabilities: np.ndarray, split: str, name: str, seed: int) -> dict:
    predictions = probabilities.argmax(axis=1)
    report = classification_report(labels, predictions, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
    return {'experiment': name, 'seed': int(seed), 'split': split, 'accuracy': float(accuracy_score(labels, predictions)), 'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(labels, predictions, average='weighted', zero_division=0)), 'recall_line': float(report['line']['recall']), 'recall_other': float(report['other']['recall']), 'recall_scrum': float(report['scrum']['recall']), 'f1_line': float(report['line']['f1-score']), 'f1_other': float(report['other']['f1-score']), 'f1_scrum': float(report['scrum']['f1-score'])}

def train_one(experiment: str, seed: int, epochs: int, batch_size: int, output_root: Path, quick: bool) -> dict:
    keras.utils.set_random_seed(seed)
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=POSE_MODEL)
    train_arrays, y_train, train_rows = load_rnn_arrays(train_df, POSE_MODEL)
    validation_arrays, y_validation, validation_rows = load_rnn_arrays(validation_df, POSE_MODEL)
    test_arrays, y_test, test_rows = load_rnn_arrays(test_df, POSE_MODEL)
    train_stats, _, _ = feature_matrix(train_df, experiment)
    validation_stats, _, _ = feature_matrix(validation_df, experiment)
    test_stats, _, _ = feature_matrix(test_df, experiment)
    scaler = StandardScaler().fit(train_stats)
    train_stats = scaler.transform(train_stats).astype(np.float32)
    validation_stats = scaler.transform(validation_stats).astype(np.float32)
    test_stats = scaler.transform(test_stats).astype(np.float32)
    train_arrays = add_stats(train_arrays, train_stats)
    validation_arrays = add_stats(validation_arrays, validation_stats)
    test_arrays = add_stats(test_arrays, test_stats)
    root = output_root / experiment / f'seed_{seed}'
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / 'stats_scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    model = build_targeted_model(int(train_arrays['pose'].shape[-1]), int(train_stats.shape[-1]))
    initialize_from_m0(model, seed)
    checkpoint = root / 'best.weights.h5'
    callback = MacroF1Checkpoint(model_inputs(validation_arrays), y_validation, checkpoint, patience=5 if quick else 10)
    weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(LABELS)), y=y_train)
    class_weights = {int(i): float(value) for i, value in enumerate(weights)}
    started = time.perf_counter()
    history = model.fit(model_inputs(train_arrays), y_train, validation_data=(model_inputs(validation_arrays), y_validation), epochs=epochs, batch_size=batch_size, class_weight=class_weights, callbacks=[callback], verbose=1)
    elapsed = time.perf_counter() - started
    if _resolve_model_artifact(checkpoint).exists():
        model.load_weights(_resolve_model_artifact(checkpoint))
    validation_probabilities = model.predict(model_inputs(validation_arrays), verbose=0)
    test_probabilities = model.predict(model_inputs(test_arrays), verbose=0)
    np.savez_compressed(root / 'validation_probabilities.npz', probabilities=validation_probabilities, labels=y_validation)
    np.savez_compressed(root / 'test_probabilities.npz', probabilities=test_probabilities, labels=y_test)
    validation_result = score(y_validation, validation_probabilities, 'validation', experiment, seed)
    test_result = score(y_test, test_probabilities, 'test', experiment, seed)
    result = {'experiment': experiment, 'description': EXPERIMENTS[experiment], 'seed': int(seed), 'stats_dim': int(train_stats.shape[-1]), 'sequence_pose_dim': int(train_arrays['pose'].shape[-1]), 'epochs_requested': int(epochs), 'epochs_completed': int(len(history.history.get('loss', []))), 'training_seconds': float(elapsed), 'warm_start_m0': True, 'validation': validation_result, 'test': test_result, 'protocol': protocol}
    write_json(root / 'metrics.json', result)
    pd.DataFrame(callback.records).to_csv(root / 'history.csv', index=False)
    del model
    keras.backend.clear_session()
    gc.collect()
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-root', default=str(HERE / 'finalists_rnn'))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    results = []
    for seed in seeds:
        for experiment in FINALISTS:
            print(f'\n=== {experiment} seed={seed} ===', flush=True)
            result = train_one(experiment, seed, args.epochs, args.batch_size, output_root, args.quick)
            results.append(result)
            print(json.dumps({'experiment': experiment, 'seed': seed, 'validation_f1_macro': result['validation']['f1_macro'], 'test_f1_macro': result['test']['f1_macro'], 'test_accuracy': result['test']['accuracy']}, ensure_ascii=False), flush=True)
    flat = []
    for item in results:
        flat.extend([item['validation'], item['test']])
    pd.DataFrame(flat).to_csv(output_root / 'summary.csv', index=False)
    validation = pd.DataFrame([item['validation'] for item in results]).sort_values('f1_macro', ascending=False)
    validation.to_csv(output_root / 'validation_ranking.csv', index=False)
    pd.DataFrame([item['test'] for item in results]).to_csv(output_root / 'test_metrics_after_validation_selection.csv', index=False)
    write_json(output_root / 'protocol.json', {'finalists': FINALISTS, 'seeds': seeds, 'selection': 'screening validation macro F1; RNN checkpoint validation macro F1; test reported after selection', 'epochs': args.epochs, 'batch_size': args.batch_size, 'pose_model': POSE_MODEL, 'warm_start_m0': True})
    print(f'Resultados: {output_root}', flush=True)
if __name__ == '__main__':
    main()
