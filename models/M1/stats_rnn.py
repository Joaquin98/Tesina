"""RNN residual con estadisticas geometricas integradas de extremo a extremo.

La secuencia recibe RGB, posiciones, pose y mascara como en ``all_residual``.
Ademas, el vector de 549 estadisticas usado por el Random Forest se proyecta
en una rama estatica y se concatena con la representacion temporal antes de
la clasificacion.
"""
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
import time
from pathlib import Path
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from models.M1.config import LABELS, RESULTS_DIR, RGB_DIM, RGB_LENGTH
from models.M1.late_fusion import geometry_statistics
from models.M1.protocol import build_partition
from models.M1.same_rnn import MacroF1Checkpoint, load_arrays
STATS_DIM = 549
SEQUENCE_VARIANT = 'all_residual'

def model_inputs(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {'rgb': arrays['rgb'], 'pose': arrays['pose'], 'sequence_mask': arrays['rgb_mask'], 'geometry_stats': arrays['geometry_stats']}

def build_model(pose_dim: int, stats_dim: int=STATS_DIM) -> keras.Model:
    rgb = keras.Input((RGB_LENGTH, RGB_DIM), name='rgb')
    pose = keras.Input((RGB_LENGTH, pose_dim), name='pose')
    sequence_mask = keras.Input((RGB_LENGTH,), dtype='bool', name='sequence_mask')
    geometry_stats = keras.Input((stats_dim,), name='geometry_stats')
    pose_delta = keras.layers.Dense(RGB_DIM, activation='tanh', kernel_initializer='zeros', bias_initializer='zeros', name='pose_residual_projection')(pose)
    pose_delta = keras.layers.Rescaling(0.1, name='pose_residual_scale')(pose_delta)
    x = keras.layers.Add(name='pose_residual_add')([rgb, pose_delta])
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='same_bigru_1')(x, mask=sequence_mask)
    temporal_embedding = keras.layers.Bidirectional(keras.layers.GRU(32), name='same_bigru_2')(x, mask=sequence_mask)
    stats_embedding = keras.layers.Dense(64, activation='relu', name='stats_projection')(geometry_stats)
    stats_embedding = keras.layers.LayerNormalization(name='stats_projection_norm')(stats_embedding)
    fused = keras.layers.Concatenate(name='stats_temporal_concat')([temporal_embedding, stats_embedding])
    fused = keras.layers.Dropout(0.3, name='same_dropout')(fused)
    fused = keras.layers.Dense(64, activation='relu', name='same_fusion_dense')(fused)
    output = keras.layers.Dense(len(LABELS), activation='softmax', name='same_class_output')(fused)
    model = keras.Model(inputs={'rgb': rgb, 'pose': pose, 'sequence_mask': sequence_mask, 'geometry_stats': geometry_stats}, outputs=output, name='stats_fused_rnn')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

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
    if not _resolve_model_artifact(checkpoint).exists():
        raise FileNotFoundError(f'No existe el checkpoint M0: {checkpoint}')
    source.load_weights(_resolve_model_artifact(checkpoint))
    source_bigru = [layer for layer in source.layers if isinstance(layer, keras.layers.Bidirectional)]
    target_bigru = [model.get_layer('same_bigru_1'), model.get_layer('same_bigru_2')]
    for source_layer, target_layer in zip(source_bigru, target_bigru):
        copy_compatible(source_layer, target_layer)
    copy_compatible(source.get_layer('fusion_dense'), model.get_layer('same_fusion_dense'))
    model.get_layer('same_class_output').set_weights(source.get_layer('class_output').get_weights())

def load_sequence(df: pd.DataFrame, pose_model: str) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    return load_arrays(df, pose_model, SEQUENCE_VARIANT)

def load_stats(df: pd.DataFrame, pose_model: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values, labels, names = geometry_statistics(df, pose_model, kind='both')
    if values.shape[1] != STATS_DIM:
        raise ValueError(f'Se esperaban {STATS_DIM} estadisticas y se obtuvieron {values.shape[1]}')
    return (values, labels, names)

def attach_stats(arrays: dict[str, np.ndarray], stats: np.ndarray) -> dict[str, np.ndarray]:
    output = dict(arrays)
    output['geometry_stats'] = np.asarray(stats, dtype=np.float32)
    return output

def train_one(seed: int, pose_model: str, epochs: int, batch_size: int, quick: bool, output_root: Path) -> dict:
    keras.utils.set_random_seed(seed)
    output_dir = output_root / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=pose_model)
    train_arrays, train_labels, train_rows = load_sequence(train_df, pose_model)
    validation_arrays, validation_labels, validation_rows = load_sequence(validation_df, pose_model)
    test_arrays, test_labels, test_rows = load_sequence(test_df, pose_model)
    train_stats, train_stats_labels, train_names = load_stats(train_df, pose_model)
    validation_stats, validation_stats_labels, validation_names = load_stats(validation_df, pose_model)
    test_stats, test_stats_labels, test_names = load_stats(test_df, pose_model)
    for expected, actual, name in ((train_labels, train_stats_labels, 'entrenamiento'), (validation_labels, validation_stats_labels, 'validacion'), (test_labels, test_stats_labels, 'prueba')):
        if not np.array_equal(expected, actual):
            raise AssertionError(f'Las etiquetas de secuencia y estadisticas no coinciden en {name}')
    for rows, names, split in ((train_rows, train_names, 'entrenamiento'), (validation_rows, validation_names, 'validacion'), (test_rows, test_names, 'prueba')):
        if rows['video_name'].astype(str).tolist() != names:
            raise AssertionError(f'El orden de secuencia y estadisticas no coincide en {split}')
    scaler = StandardScaler()
    train_stats = scaler.fit_transform(train_stats).astype(np.float32)
    validation_stats = scaler.transform(validation_stats).astype(np.float32)
    test_stats = scaler.transform(test_stats).astype(np.float32)
    np.savez_compressed(output_dir / 'stats_scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    train_arrays = attach_stats(train_arrays, train_stats)
    validation_arrays = attach_stats(validation_arrays, validation_stats)
    test_arrays = attach_stats(test_arrays, test_stats)
    model = build_model(int(train_arrays['pose'].shape[-1]), int(train_arrays['geometry_stats'].shape[-1]))
    initialize_from_m0(model, seed)
    checkpoint = output_dir / 'best.weights.h5'
    callback = MacroF1Checkpoint(model_inputs(validation_arrays), validation_labels, checkpoint, patience=5 if quick else 10)
    class_values = compute_class_weight(class_weight='balanced', classes=np.arange(len(LABELS)), y=train_labels)
    class_weights = {int(index): float(value) for index, value in enumerate(class_values)}
    started = time.perf_counter()
    history = model.fit(model_inputs(train_arrays), train_labels, validation_data=(model_inputs(validation_arrays), validation_labels), epochs=epochs, batch_size=batch_size, class_weight=class_weights, callbacks=[callback], verbose=1)
    duration = time.perf_counter() - started
    if _resolve_model_artifact(checkpoint).exists():
        model.load_weights(_resolve_model_artifact(checkpoint))
    validation_probabilities = model.predict(model_inputs(validation_arrays), verbose=0)
    test_probabilities = model.predict(model_inputs(test_arrays), verbose=0)
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
    result = {'model': 'stats_fused_rnn', 'seed': int(seed), 'test_accuracy': float(accuracy_score(test_labels, predictions)), 'f1_macro': float(f1_score(test_labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(test_labels, predictions, average='weighted', zero_division=0)), 'recall_line': float(report['line']['recall']), 'recall_other': float(report['other']['recall']), 'recall_scrum': float(report['scrum']['recall']), 'f1_line': float(report['line']['f1-score']), 'f1_other': float(report['other']['f1-score']), 'f1_scrum': float(report['scrum']['f1-score']), 'training_seconds': float(duration), 'epochs_requested': int(epochs), 'epochs_completed': int(len(history.history.get('loss', []))), 'batch_size': int(batch_size), 'pose_model': pose_model, 'sequence_pose_dim': int(train_arrays['pose'].shape[-1]), 'stats_dim': int(train_arrays['geometry_stats'].shape[-1]), 'protocol': protocol}
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
    pd.DataFrame(rows).to_csv(output_root / 'aggregate_metrics.csv', index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-root', default=str(RESULTS_DIR / 'stats_fused_rnn'))
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        print(f'\n=== stats_fused_rnn seed={seed} ===')
        result = train_one(seed, args.pose_model, args.epochs, args.batch_size, args.quick, output_root)
        results.append(result)
        print(json.dumps({key: result[key] for key in ('model', 'seed', 'test_accuracy', 'f1_macro', 'recall_line', 'recall_scrum')}, indent=2))
    aggregate(results, output_root)
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
