"""Train M1 with current CSV partitions. Comparison-model builders are retained for research modules."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
if __name__ == '__main__':
    from models.common.legacy_training import training_main
    training_main('M1')
    raise SystemExit(0)

from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import keras
from models.M1.protocol import build_partition, load_arrays, save_partition
from models.M1.config import DEFAULT_MODELS, DEFAULT_SEEDS, GEOMETRY_LENGTH, LABELS, POSE_MODEL_DEFAULT, RESULTS_DIR, PARTITIONS_DIR, RGB_DIM, RGB_LENGTH
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
MODEL_REQUIREMENTS = {'m0': {'rgb'}, 'm1': {'positions'}, 'm2': {'poses'}, 'm3': {'rgb', 'positions'}, 'm4': {'rgb', 'poses'}, 'm5': {'rgb', 'positions', 'poses'}}

def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(',') if item.strip()]

def parse_str_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(',') if item.strip()]

def branch_encoder(inputs, mask, units: int, prefix: str):
    x = keras.layers.Bidirectional(keras.layers.GRU(units, return_sequences=True), name=f'{prefix}_bigru_1')(inputs, mask=mask)
    x = keras.layers.Bidirectional(keras.layers.GRU(max(units // 2, 8)), name=f'{prefix}_bigru_2')(x, mask=mask)
    return x

def build_model(model_name: str, position_dim: int, pose_dim: int) -> tuple[keras.Model, dict[str, tuple[int, ...]]]:
    if model_name not in MODEL_REQUIREMENTS:
        raise ValueError(f'Modelo desconocido: {model_name}')
    requirements = MODEL_REQUIREMENTS[model_name]
    inputs = {}
    embeddings = []
    if 'rgb' in requirements:
        rgb = keras.Input((RGB_LENGTH, RGB_DIM), name='rgb')
        rgb_mask = keras.Input((RGB_LENGTH,), dtype='bool', name='rgb_mask')
        inputs.update(rgb=rgb, rgb_mask=rgb_mask)
        embeddings.append(branch_encoder(rgb, rgb_mask, 64, 'rgb'))
    if 'positions' in requirements:
        positions = keras.Input((GEOMETRY_LENGTH, position_dim), name='positions')
        geometry_mask = keras.Input((GEOMETRY_LENGTH,), dtype='bool', name='geometry_mask')
        inputs.update(positions=positions, geometry_mask=geometry_mask)
        embeddings.append(branch_encoder(positions, geometry_mask, 32, 'positions'))
    if 'poses' in requirements:
        poses = keras.Input((GEOMETRY_LENGTH, pose_dim), name='poses')
        if 'geometry_mask' not in inputs:
            geometry_mask = keras.Input((GEOMETRY_LENGTH,), dtype='bool', name='geometry_mask')
            inputs['geometry_mask'] = geometry_mask
        inputs['poses'] = poses
        embeddings.append(branch_encoder(poses, inputs['geometry_mask'], 32, 'poses'))
    x = embeddings[0] if len(embeddings) == 1 else keras.layers.Concatenate(name='branch_fusion')(embeddings)
    x = keras.layers.Dropout(0.3, name='fusion_dropout')(x)
    x = keras.layers.Dense(64, activation='relu', name='fusion_dense')(x)
    output = keras.layers.Dense(len(LABELS), activation='softmax', name='class_output')(x)
    model = keras.Model(inputs=inputs, outputs=output, name=f'player_geometry_{model_name}')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    shapes = {name: tuple((int(x) if x is not None else -1 for x in tensor.shape)) for name, tensor in inputs.items()}
    return (model, shapes)

class MacroF1Checkpoint(keras.callbacks.Callback):
    """Selecciona pesos por F1 macro de validación sin tocar el conjunto de prueba."""

    def __init__(self, validation_inputs, validation_labels, checkpoint_path: Path, patience: int=10):
        super().__init__()
        self.validation_inputs = validation_inputs
        self.validation_labels = validation_labels
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.best = -np.inf
        self.wait = 0
        self.records: list[dict] = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        probabilities = self.model.predict(self.validation_inputs, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        macro = float(f1_score(self.validation_labels, predictions, average='macro', zero_division=0))
        logs['val_f1_macro'] = macro
        row = {'epoch': int(epoch + 1), 'val_f1_macro': macro}
        row.update({key: float(value) for key, value in logs.items() if isinstance(value, (float, int, np.floating, np.integer))})
        self.records.append(row)
        if macro > self.best + 1e-08:
            self.best = macro
            self.wait = 0
            self.model.save_weights(self.checkpoint_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

def inputs_for_arrays(arrays: dict[str, np.ndarray], model_name: str) -> dict[str, np.ndarray]:
    names = MODEL_REQUIREMENTS[model_name]
    result: dict[str, np.ndarray] = {}
    if 'rgb' in names:
        result['rgb'] = arrays['rgb']
        result['rgb_mask'] = arrays['rgb_mask']
    if 'positions' in names:
        result['positions'] = arrays['positions']
    if 'poses' in names:
        result['poses'] = arrays['poses']
    if 'positions' in names or 'poses' in names:
        result['geometry_mask'] = arrays['geometry_mask']
    return result

def metrics_and_predictions(model, inputs, labels: np.ndarray, rows: pd.DataFrame, output_dir: Path, duration_seconds: float, model_name: str, seed: int) -> dict:
    started = time.perf_counter()
    probabilities = model.predict(inputs, verbose=0)
    inference_seconds = time.perf_counter() - started
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(labels, predictions, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
    result = {'model': model_name, 'seed': int(seed), 'test_rows': int(len(labels)), 'test_accuracy': float(accuracy_score(labels, predictions)), 'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(labels, predictions, average='weighted', zero_division=0)), 'recall_line': float(report['line']['recall']), 'recall_other': float(report['other']['recall']), 'recall_scrum': float(report['scrum']['recall']), 'f1_line': float(report['line']['f1-score']), 'f1_other': float(report['other']['f1-score']), 'f1_scrum': float(report['scrum']['f1-score']), 'training_seconds': float(duration_seconds), 'inference_seconds': float(inference_seconds), 'inference_ms_per_clip': float(1000.0 * inference_seconds / max(len(labels), 1))}
    predictions_frame = pd.DataFrame({'video_name': rows['video_name'].tolist(), 'true_label': [LABELS[index] for index in labels], 'predicted_label': [LABELS[index] for index in predictions]})
    for index, label in enumerate(LABELS):
        predictions_frame[f'prob_{label}'] = probabilities[:, index]
    predictions_frame.to_csv(output_dir / 'predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(labels, predictions, labels=list(range(len(LABELS))), normalize=None), index=LABELS, columns=LABELS).to_csv(output_dir / 'confusion_matrix.csv')
    pd.DataFrame(confusion_matrix(labels, predictions, labels=list(range(len(LABELS))), normalize='true'), index=LABELS, columns=LABELS).to_csv(output_dir / 'confusion_matrix_normalized.csv')
    (output_dir / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return result

def train_one(model_name: str, seed: int, epochs: int, batch_size: int, pose_model: str, overwrite: bool, quick: bool) -> dict:
    run_dir = RESULTS_DIR / model_name / f'seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / 'metrics.json'
    if _resolve_model_artifact(metrics_path).exists() and (not overwrite):
        cached_metrics = json.loads(_resolve_model_artifact(metrics_path).read_text(encoding='utf-8'))
        if int(cached_metrics.get('epochs_completed', 0)) >= int(epochs):
            return cached_metrics
    keras.utils.set_random_seed(seed)
    require_geometry = model_name != 'm0'
    train_df, validation_df, test_df, split_info = build_partition(seed, require_geometry=require_geometry, pose_model=pose_model)
    partition_dir = PARTITIONS_DIR / f'seed_{seed}'
    save_partition(train_df, validation_df, test_df, split_info, partition_dir)
    train_arrays, train_labels, train_rows = load_arrays(train_df, pose_model, model_name)
    validation_arrays, validation_labels, validation_rows = load_arrays(validation_df, pose_model, model_name)
    test_arrays, test_labels, test_rows = load_arrays(test_df, pose_model, model_name)
    position_dim = int(train_arrays.get('positions', np.zeros((1, 1, 31))).shape[-1])
    pose_dim = int(train_arrays.get('poses', np.zeros((1, 1, 30))).shape[-1])
    model, input_shapes = build_model(model_name, position_dim, pose_dim)
    (run_dir / 'model_summary.txt').write_text(model_summary(model), encoding='utf-8')
    checkpoint = run_dir / 'best.weights.h5'
    callback = MacroF1Checkpoint(inputs_for_arrays(validation_arrays, model_name), validation_labels, checkpoint, patience=5 if quick else 10)
    class_weights_values = compute_class_weight(class_weight='balanced', classes=np.arange(len(LABELS)), y=train_labels)
    class_weights = {int(index): float(value) for index, value in enumerate(class_weights_values)}
    started = time.perf_counter()
    history = model.fit(inputs_for_arrays(train_arrays, model_name), train_labels, validation_data=(inputs_for_arrays(validation_arrays, model_name), validation_labels), epochs=epochs, batch_size=batch_size, class_weight=class_weights, callbacks=[callback], verbose=1)
    duration = time.perf_counter() - started
    if _resolve_model_artifact(checkpoint).exists():
        model.load_weights(_resolve_model_artifact(checkpoint))
    history_rows = callback.records
    if history_rows:
        pd.DataFrame(history_rows).to_csv(run_dir / 'history.csv', index=False)
    result = metrics_and_predictions(model, inputs_for_arrays(test_arrays, model_name), test_labels, test_rows, run_dir, duration, model_name, seed)
    result.update({'protocol': split_info, 'pose_model': pose_model, 'epochs_requested': int(epochs), 'epochs_completed': int(len(history.history.get('loss', []))), 'batch_size': int(batch_size), 'input_shapes': input_shapes, 'position_dim': position_dim, 'pose_dim': pose_dim, 'checkpoint': str(checkpoint)})
    metrics_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def model_summary(model: keras.Model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return '\n'.join(lines)

def aggregate(results: list[dict]) -> None:
    if not results:
        return
    summary = pd.DataFrame(results)
    summary.to_csv(RESULTS_DIR / 'summary.csv', index=False)
    metric_names = ['test_accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum', 'f1_line', 'f1_other', 'f1_scrum']
    rows = []
    for model_name, group in summary.groupby('model'):
        row = {'model': model_name, 'runs': int(len(group))}
        for metric in metric_names:
            row[f'{metric}_mean'] = float(group[metric].mean())
            row[f'{metric}_std'] = float(group[metric].std(ddof=0))
        rows.append(row)
    pd.DataFrame(rows).sort_values('model').to_csv(RESULTS_DIR / 'aggregate_metrics.csv', index=False)

def comparison_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(DEFAULT_MODELS))
    parser.add_argument('--seeds', default=','.join((str(seed) for seed in DEFAULT_SEEDS)))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--pose-model', default=POSE_MODEL_DEFAULT)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--quick', action='store_true', help='Reduce patience, útil para smoke tests')
    parser.add_argument('--dry-run', action='store_true', help='Construye particiones y verifica formas, sin entrenar')
    args = parser.parse_args()
    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)
    unknown = set(models).difference(MODEL_REQUIREMENTS)
    if unknown:
        raise SystemExit(f'Modelos desconocidos: {sorted(unknown)}')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for model_name in models:
        for seed in seeds:
            print(f'\n=== {model_name} seed={seed} ===')
            if args.dry_run:
                train_df, validation_df, test_df, info = build_partition(seed, require_geometry=model_name != 'm0', pose_model=args.pose_model)
                for split_name, frame in [('train', train_df), ('validation', validation_df), ('test', test_df)]:
                    sample = frame.head(1)
                    arrays, labels, rows = load_arrays(sample, args.pose_model, model_name)
                    print(split_name, len(frame), {key: value.shape for key, value in arrays.items()})
                print(json.dumps(info, indent=2))
                continue
            results.append(train_one(model_name, seed, args.epochs, args.batch_size, args.pose_model, args.overwrite, args.quick))
            print(json.dumps({key: results[-1][key] for key in ('model', 'seed', 'test_accuracy', 'f1_macro', 'recall_line', 'recall_scrum')}, indent=2))
    aggregate(results)
    if results:
        print(f"Resultados: {RESULTS_DIR / 'summary.csv'}")
def main(argv=None):
    from models.common.legacy_training import training_main
    return training_main('M1', argv)


if __name__ == '__main__':
    main()
