"""Fusion de la representacion YOLO dirigida con el ResNet-50 + BiGRU.

La variante ``y5_targeted_temporal`` se construye a partir de la cache YOLO
existente y produce un vector de 473 estadisticas por clip. Este programa lo
estandariza usando solamente entrenamiento, lo proyecta con una rama densa y
lo concatena con la representacion temporal RGB antes de clasificar. La rama
temporal conserva la secuencia de 200 pasos y la adaptacion residual de pose
usada en ``stats_rnn.py``.
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
from models.M1.config import LABELS, LABEL_TO_ID, RESULTS_DIR, RGB_DIM, RGB_LENGTH
from models.M1.protocol import build_partition
from models.M1.same_rnn import MacroF1Checkpoint, load_arrays
from models.M1.targeted_geometry_experiments import load_clip_variants
from models.M1.stats_rnn import initialize_from_m0
SEQUENCE_VARIANT = 'all_residual'
STATS_VARIANT = 'y5_targeted_temporal'
STATS_DIM = 473

def model_inputs(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {'rgb': arrays['rgb'], 'pose': arrays['pose'], 'sequence_mask': arrays['rgb_mask'], 'targeted_geometry_stats': arrays['targeted_geometry_stats']}

def build_model(pose_dim: int, stats_dim: int=STATS_DIM) -> keras.Model:
    rgb = keras.Input((RGB_LENGTH, RGB_DIM), name='rgb')
    pose = keras.Input((RGB_LENGTH, pose_dim), name='pose')
    sequence_mask = keras.Input((RGB_LENGTH,), dtype='bool', name='sequence_mask')
    targeted_stats = keras.Input((stats_dim,), name='targeted_geometry_stats')
    pose_delta = keras.layers.Dense(RGB_DIM, activation='tanh', kernel_initializer='zeros', bias_initializer='zeros', name='pose_residual_projection')(pose)
    pose_delta = keras.layers.Rescaling(0.1, name='pose_residual_scale')(pose_delta)
    x = keras.layers.Add(name='pose_residual_add')([rgb, pose_delta])
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='same_bigru_1')(x, mask=sequence_mask)
    temporal_embedding = keras.layers.Bidirectional(keras.layers.GRU(32), name='same_bigru_2')(x, mask=sequence_mask)
    stats_embedding = keras.layers.Dense(64, activation='relu', name='targeted_stats_projection')(targeted_stats)
    stats_embedding = keras.layers.LayerNormalization(name='targeted_stats_projection_norm')(stats_embedding)
    fused = keras.layers.Concatenate(name='targeted_stats_temporal_concat')([temporal_embedding, stats_embedding])
    fused = keras.layers.Dropout(0.3, name='same_dropout')(fused)
    fused = keras.layers.Dense(64, activation='relu', name='same_fusion_dense')(fused)
    output = keras.layers.Dense(len(LABELS), activation='softmax', name='same_class_output')(fused)
    model = keras.Model(inputs={'rgb': rgb, 'pose': pose, 'sequence_mask': sequence_mask, 'targeted_geometry_stats': targeted_stats}, outputs=output, name='targeted_stats_fused_rnn')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_targeted_stats(frame: pd.DataFrame, pose_model: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    for row in frame.to_dict('records'):
        name = str(row['video_name'])
        clip_variants = load_clip_variants(name, pose_model)
        vector = np.asarray(clip_variants[STATS_VARIANT], dtype=np.float32)
        if vector.shape != (STATS_DIM,):
            raise ValueError(f'Se esperaban {STATS_DIM} estadisticas para {name}, se obtuvieron {vector.shape}')
        values.append(vector)
        labels.append(LABEL_TO_ID[str(row['tag'])])
        names.append(name)
    return (np.asarray(values, dtype=np.float32), np.asarray(labels, dtype=np.int32), names)

def attach_stats(arrays: dict[str, np.ndarray], stats: np.ndarray) -> dict[str, np.ndarray]:
    output = dict(arrays)
    output['targeted_geometry_stats'] = np.asarray(stats, dtype=np.float32)
    return output

def train_one(seed: int, pose_model: str, epochs: int, batch_size: int, quick: bool, warm_start: bool, output_root: Path) -> dict:
    keras.utils.set_random_seed(seed)
    output_dir = output_root / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=pose_model)
    train_arrays, train_labels, train_rows = load_arrays(train_df, pose_model, SEQUENCE_VARIANT)
    validation_arrays, validation_labels, validation_rows = load_arrays(validation_df, pose_model, SEQUENCE_VARIANT)
    test_arrays, test_labels, test_rows = load_arrays(test_df, pose_model, SEQUENCE_VARIANT)
    train_stats, train_stats_labels, train_names = load_targeted_stats(train_df, pose_model)
    validation_stats, validation_stats_labels, validation_names = load_targeted_stats(validation_df, pose_model)
    test_stats, test_stats_labels, test_names = load_targeted_stats(test_df, pose_model)
    for expected, actual, split in ((train_labels, train_stats_labels, 'entrenamiento'), (validation_labels, validation_stats_labels, 'validacion'), (test_labels, test_stats_labels, 'prueba')):
        if not np.array_equal(expected, actual):
            raise AssertionError(f'Las etiquetas RGB y y5 no coinciden en {split}')
    for rows, names, split in ((train_rows, train_names, 'entrenamiento'), (validation_rows, validation_names, 'validacion'), (test_rows, test_names, 'prueba')):
        if rows['video_name'].astype(str).tolist() != names:
            raise AssertionError(f'El orden RGB y y5 no coincide en {split}')
    scaler = StandardScaler()
    train_stats = scaler.fit_transform(train_stats).astype(np.float32)
    validation_stats = scaler.transform(validation_stats).astype(np.float32)
    test_stats = scaler.transform(test_stats).astype(np.float32)
    np.savez_compressed(output_dir / 'targeted_stats_scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    train_arrays = attach_stats(train_arrays, train_stats)
    validation_arrays = attach_stats(validation_arrays, validation_stats)
    test_arrays = attach_stats(test_arrays, test_stats)
    model = build_model(int(train_arrays['pose'].shape[-1]), int(train_arrays['targeted_geometry_stats'].shape[-1]))
    if warm_start:
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
    validation_predictions = validation_probabilities.argmax(axis=1)
    report = classification_report(test_labels, predictions, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
    validation_report = classification_report(validation_labels, validation_predictions, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
    frame = pd.DataFrame({'video_name': test_rows['video_name'].tolist(), 'true_label': [LABELS[index] for index in test_labels], 'predicted_label': [LABELS[index] for index in predictions]})
    for index, label in enumerate(LABELS):
        frame[f'prob_{label}'] = test_probabilities[:, index]
    frame.to_csv(output_dir / 'predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(test_labels, predictions, labels=list(range(len(LABELS)))), index=LABELS, columns=LABELS).to_csv(output_dir / 'confusion_matrix.csv')
    (output_dir / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output_dir / 'validation_classification_report.json').write_text(json.dumps(validation_report, indent=2), encoding='utf-8')
    pd.DataFrame(callback.records).to_csv(output_dir / 'history.csv', index=False)
    result = {'model': 'targeted_stats_fused_rnn', 'seed': int(seed), 'test_accuracy': float(accuracy_score(test_labels, predictions)), 'f1_macro': float(f1_score(test_labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(test_labels, predictions, average='weighted', zero_division=0)), 'recall_line': float(report['line']['recall']), 'recall_other': float(report['other']['recall']), 'recall_scrum': float(report['scrum']['recall']), 'f1_line': float(report['line']['f1-score']), 'f1_other': float(report['other']['f1-score']), 'f1_scrum': float(report['scrum']['f1-score']), 'validation_accuracy': float(accuracy_score(validation_labels, validation_predictions)), 'validation_f1_macro': float(f1_score(validation_labels, validation_predictions, average='macro', zero_division=0)), 'training_seconds': float(duration), 'epochs_requested': int(epochs), 'epochs_completed': int(len(history.history.get('loss', []))), 'batch_size': int(batch_size), 'pose_model': pose_model, 'sequence_variant': SEQUENCE_VARIANT, 'stats_variant': STATS_VARIANT, 'sequence_pose_dim': int(train_arrays['pose'].shape[-1]), 'stats_dim': int(train_arrays['targeted_geometry_stats'].shape[-1]), 'warm_start_m0': bool(warm_start), 'protocol': protocol}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def aggregate(results: list[dict], output_root: Path) -> None:
    frame = pd.DataFrame(results)
    frame.to_csv(output_root / 'summary.csv', index=False)
    rows = []
    for model_name, group in frame.groupby('model'):
        row = {'model': model_name, 'runs': len(group)}
        metrics = ('test_accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum', 'f1_line', 'f1_other', 'f1_scrum', 'validation_accuracy', 'validation_f1_macro')
        for metric in metrics:
            row[f'{metric}_mean'] = float(group[metric].mean())
            row[f'{metric}_std'] = float(group[metric].std(ddof=0))
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_root / 'aggregate_metrics.csv', index=False)

def existing_results(output_root: Path) -> list[dict]:
    results: list[dict] = []
    for metrics_path in sorted(_model_artifact_glob(output_root, 'seed_*/metrics.json', recursive=False)):
        try:
            results.append(json.loads(_resolve_model_artifact(metrics_path).read_text(encoding='utf-8')))
        except (OSError, json.JSONDecodeError):
            continue
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-warm-start', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true', help='Agrega metrics.json ya existentes sin reentrenar')
    parser.add_argument('--output-root', default=str(RESULTS_DIR / 'targeted_stats_fused_rnn'))
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        results = existing_results(output_root)
        if not results:
            raise SystemExit(f'No hay metrics.json para agregar en {output_root}')
    else:
        results = []
        for seed in seeds:
            print(f'\n=== targeted_stats_fused_rnn seed={seed} ===')
            result = train_one(seed, args.pose_model, args.epochs, args.batch_size, args.quick, not args.no_warm_start, output_root)
            results.append(result)
            print(json.dumps({key: result[key] for key in ('model', 'seed', 'test_accuracy', 'f1_macro', 'validation_f1_macro', 'recall_line', 'recall_scrum')}, indent=2))
        merged = {int(result['seed']): result for result in existing_results(output_root)}
        merged.update({int(result['seed']): result for result in results})
        results = [merged[key] for key in sorted(merged)]
    aggregate(results, output_root)
    (output_root / 'protocol.json').write_text(json.dumps({'model': 'targeted_stats_fused_rnn', 'seeds': sorted((int(result['seed']) for result in results)), 'pose_model': args.pose_model, 'epochs': args.epochs, 'batch_size': args.batch_size, 'sequence_variant': SEQUENCE_VARIANT, 'stats_variant': STATS_VARIANT, 'stats_dim': STATS_DIM, 'warm_start_m0': not args.no_warm_start}, indent=2), encoding='utf-8')
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
