"""Tres mejoras sobre la variante ``all_residual`` ganadora.

* ``attention_reliability``: atencion temporal condicionada por la geometria
  y la mascara de deteccion.
* ``trajectory_residual``: agrega velocidades y estabilidad de jugadores
  obtenidas al asociar cajas YOLO entre frames consecutivos.
* ``multiscale_50_200``: procesa una vista de 50 pasos y otra de 200 con
  BiGRU compartidas, y combina ambas representaciones.
"""
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
import time
from functools import lru_cache
from pathlib import Path
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from models.M1.config import GEOMETRY_LENGTH, LABELS, LABEL_TO_ID, RESULTS_DIR, RGB_DIM, RGB_LENGTH
from models.M1.protocol import build_partition, pose_cache_path
from models.M1.same_rnn import MacroF1Checkpoint, load_arrays
from models.M1.temporal_resolutions import resample_arrays
VARIANTS = ('attention_reliability', 'trajectory_residual', 'multiscale_50_200')
BASE_VARIANT = 'all_residual'

class LastValid(keras.layers.Layer):
    """Extrae el ultimo estado valido de una secuencia enmascarada."""

    def call(self, inputs):
        sequence, mask = inputs
        lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1) - 1
        lengths = tf.maximum(lengths, 0)
        batch = tf.range(tf.shape(sequence)[0], dtype=tf.int32)
        indexes = tf.stack([batch, lengths], axis=1)
        return tf.gather_nd(sequence, indexes)

class MaskedTemporalAttention(keras.layers.Layer):
    """Promedia estados temporales con pesos que ignoran frames invalidos."""

    def call(self, inputs):
        sequence, logits, mask = inputs
        scores = tf.squeeze(logits, axis=-1)
        scores = tf.where(mask, scores, tf.cast(-1000000000.0, scores.dtype))
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(sequence * weights[..., None], axis=1)

def model_inputs(arrays: dict[str, np.ndarray], variant: str) -> dict[str, np.ndarray]:
    if variant == 'multiscale_50_200':
        return {'rgb_50': arrays['rgb_50'], 'pose_50': arrays['pose_50'], 'mask_50': arrays['mask_50'], 'rgb_200': arrays['rgb_200'], 'pose_200': arrays['pose_200'], 'mask_200': arrays['mask_200']}
    return {'rgb': arrays['rgb'], 'pose': arrays['pose'], 'sequence_mask': arrays['rgb_mask']}

def build_residual_inputs(length: int, pose_dim: int):
    rgb = keras.Input((length, RGB_DIM), name='rgb')
    pose = keras.Input((length, pose_dim), name='pose')
    mask = keras.Input((length,), dtype='bool', name='sequence_mask')
    delta = keras.layers.Dense(RGB_DIM, activation='tanh', kernel_initializer='zeros', bias_initializer='zeros', name='pose_residual_projection')(pose)
    delta = keras.layers.Rescaling(0.1, name='pose_residual_scale')(delta)
    x = keras.layers.Add(name='pose_residual_add')([rgb, delta])
    return (rgb, pose, mask, x)

def build_attention_model(pose_dim: int) -> keras.Model:
    rgb = keras.Input((RGB_LENGTH, RGB_DIM), name='rgb')
    pose = keras.Input((RGB_LENGTH, pose_dim), name='pose')
    mask = keras.Input((RGB_LENGTH,), dtype='bool', name='sequence_mask')
    delta = keras.layers.Dense(RGB_DIM, activation='tanh', kernel_initializer='zeros', bias_initializer='zeros', name='pose_residual_projection')(pose)
    delta = keras.layers.Rescaling(0.1, name='pose_residual_scale')(delta)
    x = keras.layers.Add(name='pose_residual_add')([rgb, delta])
    first = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='same_bigru_1')(x, mask=mask)
    last = keras.layers.Bidirectional(keras.layers.GRU(32), name='same_bigru_2')(first, mask=mask)
    attention_input = keras.layers.Concatenate(name='attention_reliability_input')([first, pose])
    attention_hidden = keras.layers.Dense(32, activation='relu', name='attention_hidden')(attention_input)
    attention_logits = keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros', name='attention_score')(attention_hidden)
    context = MaskedTemporalAttention(name='temporal_attention')([first, attention_logits, mask])
    attention_delta = keras.layers.Dense(64, activation='linear', kernel_initializer='zeros', bias_initializer='zeros', name='attention_delta_projection')(context)
    fused = keras.layers.Add(name='attention_residual_add')([last, attention_delta])
    fused = keras.layers.Dropout(0.3, name='same_dropout')(fused)
    fused = keras.layers.Dense(64, activation='relu', name='same_fusion_dense')(fused)
    output = keras.layers.Dense(len(LABELS), activation='softmax', name='same_class_output')(fused)
    model = keras.Model(inputs={'rgb': rgb, 'pose': pose, 'sequence_mask': mask}, outputs=output, name='attention_reliability')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_trajectory_model(pose_dim: int) -> keras.Model:
    rgb, pose, mask, x = build_residual_inputs(RGB_LENGTH, pose_dim)
    first = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='same_bigru_1')(x, mask=mask)
    second = keras.layers.Bidirectional(keras.layers.GRU(32), name='same_bigru_2')(first, mask=mask)
    second = keras.layers.Dropout(0.3, name='same_dropout')(second)
    fused = keras.layers.Dense(64, activation='relu', name='same_fusion_dense')(second)
    output = keras.layers.Dense(len(LABELS), activation='softmax', name='same_class_output')(fused)
    model = keras.Model(inputs={'rgb': rgb, 'pose': pose, 'sequence_mask': mask}, outputs=output, name='trajectory_residual')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_multiscale_model(pose_dim: int) -> keras.Model:
    rgb_50 = keras.Input((50, RGB_DIM), name='rgb_50')
    pose_50 = keras.Input((50, pose_dim), name='pose_50')
    mask_50 = keras.Input((50,), dtype='bool', name='mask_50')
    rgb_200 = keras.Input((200, RGB_DIM), name='rgb_200')
    pose_200 = keras.Input((200, pose_dim), name='pose_200')
    mask_200 = keras.Input((200,), dtype='bool', name='mask_200')
    residual = keras.layers.Dense(RGB_DIM, activation='tanh', kernel_initializer='zeros', bias_initializer='zeros', name='pose_residual_projection')
    scale = keras.layers.Rescaling(0.1, name='pose_residual_scale')
    x_50 = keras.layers.Add(name='pose_residual_add_50')([rgb_50, scale(residual(pose_50))])
    x_200 = keras.layers.Add(name='pose_residual_add_200')([rgb_200, scale(residual(pose_200))])
    bigru_1 = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='same_bigru_1')
    bigru_2 = keras.layers.Bidirectional(keras.layers.GRU(32), name='same_bigru_2')
    h_50 = bigru_2(bigru_1(x_50, mask=mask_50), mask=mask_50)
    h_200 = bigru_2(bigru_1(x_200, mask=mask_200), mask=mask_200)
    fused = keras.layers.Concatenate(name='multiscale_concat')([h_50, h_200])
    fused = keras.layers.Dropout(0.3, name='same_dropout')(fused)
    fused = keras.layers.Dense(64, activation='relu', name='same_fusion_dense')(fused)
    output = keras.layers.Dense(len(LABELS), activation='softmax', name='same_class_output')(fused)
    model = keras.Model(inputs={'rgb_50': rgb_50, 'pose_50': pose_50, 'mask_50': mask_50, 'rgb_200': rgb_200, 'pose_200': pose_200, 'mask_200': mask_200}, outputs=output, name='multiscale_50_200')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model(variant: str, pose_dim: int) -> keras.Model:
    if variant == 'attention_reliability':
        return build_attention_model(pose_dim)
    if variant == 'trajectory_residual':
        return build_trajectory_model(pose_dim)
    if variant == 'multiscale_50_200':
        return build_multiscale_model(pose_dim)
    raise ValueError(f'Variante desconocida: {variant}')

def copy_compatible(source_layer: keras.layers.Layer, target_layer: keras.layers.Layer) -> None:
    source_weights = source_layer.get_weights()
    target_weights = target_layer.get_weights()
    copied = []
    for source_weight, target_weight in zip(source_weights, target_weights):
        if source_weight.shape == target_weight.shape:
            copied.append(source_weight)
        elif source_weight.ndim == 2 and target_weight.ndim == 2 and (source_weight.shape[1] == target_weight.shape[1]) and (target_weight.shape[0] > source_weight.shape[0]):
            expanded = np.zeros_like(target_weight)
            expanded[:source_weight.shape[0], :] = source_weight
            copied.append(expanded)
        else:
            copied.append(target_weight)
    target_layer.set_weights(copied)

def initialize_from_m0(model: keras.Model, seed: int) -> None:
    from models.M1.train import build_model as build_m0
    source, _ = build_m0('m0', 31, 30)
    checkpoint = RESULTS_DIR / 'm0' / f'seed_{seed}' / 'best.weights.h5'
    source.load_weights(_resolve_model_artifact(checkpoint))
    source_bigru = [layer for layer in source.layers if isinstance(layer, keras.layers.Bidirectional)]
    target_bigru = [layer for layer in model.layers if isinstance(layer, keras.layers.Bidirectional)]
    for source_layer, target_layer in zip(source_bigru, target_bigru):
        copy_compatible(source_layer, target_layer)
    copy_compatible(source.get_layer('fusion_dense'), model.get_layer('same_fusion_dense'))
    model.get_layer('same_class_output').set_weights(source.get_layer('class_output').get_weights())

@lru_cache(maxsize=10000)
def trajectory_sequence(video_name: str, pose_model: str) -> np.ndarray:
    path = pose_cache_path(video_name, pose_model)
    with _load_model_result_array(path, allow_pickle=False) as data:
        boxes = np.asarray(data['boxes'], dtype=np.float32)
        valid = np.asarray(data['valid'], dtype=bool)
        confidence = np.asarray(data['person_conf'], dtype=np.float32)
    output = np.zeros((GEOMETRY_LENGTH, 12), dtype=np.float32)
    previous_centers = None
    previous_mean_velocity = np.zeros(2, dtype=np.float32)
    for frame_index in range(GEOMETRY_LENGTH):
        keep = valid[frame_index] & (confidence[frame_index] >= 0.25)
        selected = boxes[frame_index, keep]
        current = ((selected[:, :2] + selected[:, 2:4]) * 0.5).astype(np.float32) if len(selected) else np.empty((0, 2), dtype=np.float32)
        if previous_centers is not None and len(previous_centers) and len(current):
            distances = np.sqrt(np.sum((previous_centers[:, None, :] - current[None, :, :]) ** 2, axis=-1))
            used = set()
            deltas = []
            for previous_index in np.argsort(np.min(distances, axis=1)):
                current_index = int(np.argmin(distances[previous_index]))
                if current_index in used or distances[previous_index, current_index] > 0.2:
                    continue
                used.add(current_index)
                deltas.append(current[current_index] - previous_centers[previous_index])
            if deltas:
                delta = np.asarray(deltas, dtype=np.float32)
                mean_velocity = delta.mean(axis=0)
                speed = np.sqrt(np.sum(delta * delta, axis=1))
                acceleration = mean_velocity - previous_mean_velocity
                output[frame_index] = np.array([len(deltas) / 30.0, mean_velocity[0], mean_velocity[1], delta[:, 0].std(), delta[:, 1].std(), np.abs(delta[:, 0]).mean(), np.abs(delta[:, 1]).mean(), np.abs(delta[:, 0]).max(), np.abs(delta[:, 1]).max(), speed.mean(), speed.std(), np.sqrt(np.sum(acceleration * acceleration))], dtype=np.float32)
                previous_mean_velocity = mean_velocity
        previous_centers = current
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

def load_variant_arrays(df: pd.DataFrame, pose_model: str, variant: str):
    arrays, labels, rows = load_arrays(df, pose_model, BASE_VARIANT)
    if variant == 'trajectory_residual':
        trajectories = np.asarray([trajectory_sequence(str(name), pose_model) for name in rows['video_name']], dtype=np.float32)
        base_values = arrays['pose'][..., :-1]
        mask = arrays['pose'][..., -1:]
        arrays = dict(arrays)
        arrays['pose'] = np.concatenate([base_values, trajectories, mask], axis=-1).astype(np.float32)
    if variant == 'multiscale_50_200':
        short = resample_arrays(arrays, 50)
        arrays = {'rgb_50': short['rgb'], 'pose_50': short['pose'], 'mask_50': short['rgb_mask'], 'rgb_200': arrays['rgb'], 'pose_200': arrays['pose'], 'mask_200': arrays['rgb_mask']}
    return (arrays, labels, rows)

def train_one(variant: str, seed: int, pose_model: str, epochs: int, batch_size: int, quick: bool, output_root: Path) -> dict:
    keras.utils.set_random_seed(seed)
    output_dir = output_root / variant / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=pose_model)
    train_arrays, train_labels, train_rows = load_variant_arrays(train_df, pose_model, variant)
    validation_arrays, validation_labels, validation_rows = load_variant_arrays(validation_df, pose_model, variant)
    test_arrays, test_labels, test_rows = load_variant_arrays(test_df, pose_model, variant)
    pose_key = 'pose_200' if variant == 'multiscale_50_200' else 'pose'
    model = build_model(variant, int(train_arrays[pose_key].shape[-1]))
    initialize_from_m0(model, seed)
    checkpoint = output_dir / 'best.weights.h5'
    callback = MacroF1Checkpoint(model_inputs(validation_arrays, variant), validation_labels, checkpoint, patience=5 if quick else 10)
    values = compute_class_weight(class_weight='balanced', classes=np.arange(len(LABELS)), y=train_labels)
    class_weights = {int(index): float(value) for index, value in enumerate(values)}
    started = time.perf_counter()
    history = model.fit(model_inputs(train_arrays, variant), train_labels, validation_data=(model_inputs(validation_arrays, variant), validation_labels), epochs=epochs, batch_size=batch_size, class_weight=class_weights, callbacks=[callback], verbose=1)
    duration = time.perf_counter() - started
    if _resolve_model_artifact(checkpoint).exists():
        model.load_weights(_resolve_model_artifact(checkpoint))
    validation_probabilities = model.predict(model_inputs(validation_arrays, variant), verbose=0)
    test_probabilities = model.predict(model_inputs(test_arrays, variant), verbose=0)
    np.savez_compressed(output_dir / 'validation_probabilities.npz', probabilities=validation_probabilities, labels=validation_labels)
    predictions = test_probabilities.argmax(axis=1)
    report = classification_report(test_labels, predictions, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
    frame = pd.DataFrame({'video_name': test_rows['video_name'].tolist(), 'true_label': [LABELS[index] for index in test_labels], 'predicted_label': [LABELS[index] for index in predictions]})
    for index, label in enumerate(LABELS):
        frame[f'prob_{label}'] = test_probabilities[:, index]
    frame.to_csv(output_dir / 'predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(test_labels, predictions, labels=list(range(len(LABELS)))), index=LABELS, columns=LABELS).to_csv(output_dir / 'confusion_matrix.csv')
    (output_dir / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(callback.records).to_csv(output_dir / 'history.csv', index=False)
    result = {'model': variant, 'seed': int(seed), 'test_accuracy': float(accuracy_score(test_labels, predictions)), 'f1_macro': float(f1_score(test_labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(test_labels, predictions, average='weighted', zero_division=0)), 'recall_line': float(report['line']['recall']), 'recall_other': float(report['other']['recall']), 'recall_scrum': float(report['scrum']['recall']), 'f1_line': float(report['line']['f1-score']), 'f1_other': float(report['other']['f1-score']), 'f1_scrum': float(report['scrum']['f1-score']), 'training_seconds': float(duration), 'epochs_requested': int(epochs), 'epochs_completed': int(len(history.history.get('loss', []))), 'batch_size': int(batch_size), 'pose_model': pose_model, 'pose_dim': int(train_arrays[pose_key].shape[-1]), 'protocol': protocol}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def aggregate(results: list[dict], output_root: Path) -> None:
    frame = pd.DataFrame(results)
    frame.to_csv(output_root / 'summary.csv', index=False)
    rows = []
    for model_name, group in frame.groupby('model'):
        row = {'model': model_name, 'runs': len(group)}
        for metric in ('test_accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum', 'f1_line', 'f1_other', 'f1_scrum'):
            row[f'{metric}_mean'] = group[metric].mean()
            row[f'{metric}_std'] = group[metric].std(ddof=0)
        rows.append(row)
    pd.DataFrame(rows).sort_values('model').to_csv(output_root / 'aggregate_metrics.csv', index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(VARIANTS))
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-root', default=str(RESULTS_DIR / 'improvement_experiments'))
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(',') if item.strip()]
    unknown = set(models).difference(VARIANTS)
    if unknown:
        raise SystemExit(f'Variantes desconocidas: {sorted(unknown)}')
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in models:
        for seed in seeds:
            print(f'\n=== improvement {variant} seed={seed} ===')
            result = train_one(variant, seed, args.pose_model, args.epochs, args.batch_size, args.quick, output_root)
            results.append(result)
            print(json.dumps({key: result[key] for key in ('model', 'seed', 'test_accuracy', 'f1_macro', 'recall_line', 'recall_scrum')}, indent=2))
    aggregate(results, output_root)
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
