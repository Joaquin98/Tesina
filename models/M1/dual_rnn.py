"""Comparacion de ramas temporales separadas para RGB y geometria YOLO.

E1 (`dual_concat`) mantiene una BiGRU para RGB y otra para geometria y
concatena sus representaciones. E2 agrega una compuerta aprendida que decide
cuanto de la rama geometrica debe sumarse a la representacion visual. E3 usa
la misma compuerta, pero agrega diferencias temporales de los descriptores.
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
from sklearn.utils.class_weight import compute_class_weight
from models.M1.config import LABELS, LABEL_TO_ID, RESULTS_DIR, RGB_DIM, RGB_LENGTH
from models.M1.protocol import build_partition
from models.M1.same_rnn import MacroF1Checkpoint, inputs as base_inputs, load_arrays
VARIANTS = ('dual_concat', 'dual_gated', 'dual_gated_delta')
BASE_GEOMETRY_VARIANT = 'all_residual'

def add_temporal_differences(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Agrega primera diferencia de los 61 canales geometricos.

    El ultimo canal es la mascara de deteccion y se conserva sin derivar. La
    diferencia se calcula despues de alinear a los mismos 200 pasos de RGB.
    """
    output = dict(arrays)
    values = np.asarray(arrays['pose'], dtype=np.float32)
    geometry = values[..., :-1]
    mask = values[..., -1:]
    delta = np.diff(geometry, axis=1, prepend=np.zeros_like(geometry[:, :1]))
    output['pose'] = np.concatenate([geometry, delta, mask], axis=-1).astype(np.float32)
    return output

def build_model(variant: str, geometry_dim: int) -> keras.Model:
    if variant not in VARIANTS:
        raise ValueError(f'Variante desconocida: {variant}')
    rgb = keras.Input((RGB_LENGTH, RGB_DIM), name='rgb')
    geometry = keras.Input((RGB_LENGTH, geometry_dim), name='pose')
    mask = keras.Input((RGB_LENGTH,), dtype='bool', name='sequence_mask')
    rgb_sequence = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='rgb_bigru_1')(rgb, mask=mask)
    rgb_embedding = keras.layers.Bidirectional(keras.layers.GRU(32), name='rgb_bigru_2')(rgb_sequence, mask=mask)
    geometry_normalized = keras.layers.LayerNormalization(name='geometry_input_norm')(geometry)
    geometry_sequence = keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=True), name='geometry_bigru_1')(geometry_normalized, mask=mask)
    geometry_embedding = keras.layers.Bidirectional(keras.layers.GRU(32), name='geometry_bigru_2')(geometry_sequence, mask=mask)
    if variant == 'dual_concat':
        fused = keras.layers.Concatenate(name='dual_concat_embedding')([rgb_embedding, geometry_embedding])
    else:
        gate_input = keras.layers.Concatenate(name='geometry_gate_input')([rgb_embedding, geometry_embedding])
        gate = keras.layers.Dense(64, activation='sigmoid', kernel_initializer='zeros', bias_initializer=keras.initializers.Constant(-1.5), name='geometry_gate')(gate_input)
        gated_geometry = keras.layers.Multiply(name='gated_geometry')([gate, geometry_embedding])
        fused = keras.layers.Add(name='dual_gated_embedding')([rgb_embedding, gated_geometry])
    fused = keras.layers.Dropout(0.3, name='dual_dropout')(fused)
    fused = keras.layers.Dense(64, activation='relu', name='dual_fusion_dense')(fused)
    output = keras.layers.Dense(len(LABELS), activation='softmax', name='dual_class_output')(fused)
    model = keras.Model(inputs={'rgb': rgb, 'pose': geometry, 'sequence_mask': mask}, outputs=output, name=f'dual_rnn_{variant}')
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
    for source_layer, target_name in zip(source_bigru, ('rgb_bigru_1', 'rgb_bigru_2')):
        copy_compatible(source_layer, model.get_layer(target_name))
    source_dense = source.get_layer('fusion_dense')
    copy_compatible(source_dense, model.get_layer('dual_fusion_dense'))
    source_output = source.get_layer('class_output')
    model.get_layer('dual_class_output').set_weights(source_output.get_weights())

def model_inputs(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return base_inputs(arrays)

def prepare_arrays(df: pd.DataFrame, pose_model: str, variant: str):
    arrays, labels, rows = load_arrays(df, pose_model, BASE_GEOMETRY_VARIANT)
    if variant == 'dual_gated_delta':
        arrays = add_temporal_differences(arrays)
    return (arrays, labels, rows)

def train_one(variant: str, seed: int, pose_model: str, epochs: int, batch_size: int, quick: bool, output_root: Path) -> dict:
    keras.utils.set_random_seed(seed)
    output_dir = output_root / variant / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=pose_model)
    train_arrays, train_labels, train_rows = prepare_arrays(train_df, pose_model, variant)
    validation_arrays, validation_labels, validation_rows = prepare_arrays(validation_df, pose_model, variant)
    test_arrays, test_labels, test_rows = prepare_arrays(test_df, pose_model, variant)
    model = build_model(variant, int(train_arrays['pose'].shape[-1]))
    initialize_from_m0(model, seed)
    checkpoint = output_dir / 'best.weights.h5'
    callback = MacroF1Checkpoint(model_inputs(validation_arrays), validation_labels, checkpoint, patience=5 if quick else 10)
    classes = np.arange(len(LABELS))
    class_values = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
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
    result = {'model': variant, 'seed': int(seed), 'test_accuracy': float(accuracy_score(test_labels, predictions)), 'f1_macro': float(f1_score(test_labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(test_labels, predictions, average='weighted', zero_division=0)), 'recall_line': float(report['line']['recall']), 'recall_other': float(report['other']['recall']), 'recall_scrum': float(report['scrum']['recall']), 'training_seconds': float(duration), 'epochs_requested': int(epochs), 'epochs_completed': int(len(history.history.get('loss', []))), 'batch_size': int(batch_size), 'pose_model': pose_model, 'pose_dim_aligned': int(train_arrays['pose'].shape[-1]), 'protocol': protocol}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def aggregate(results: list[dict], output_root: Path) -> None:
    frame = pd.DataFrame(results)
    frame.to_csv(output_root / 'summary.csv', index=False)
    rows = []
    for model_name, group in frame.groupby('model'):
        rows.append({'model': model_name, 'runs': len(group), 'test_accuracy_mean': group['test_accuracy'].mean(), 'test_accuracy_std': group['test_accuracy'].std(ddof=0), 'f1_macro_mean': group['f1_macro'].mean(), 'f1_macro_std': group['f1_macro'].std(ddof=0), 'f1_weighted_mean': group['f1_weighted'].mean(), 'f1_weighted_std': group['f1_weighted'].std(ddof=0)})
    pd.DataFrame(rows).sort_values('model').to_csv(output_root / 'aggregate_metrics.csv', index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(VARIANTS))
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-root', default=str(RESULTS_DIR / 'dual_rnn'))
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
            print(f'\n=== dual_rnn {variant} seed={seed} ===')
            result = train_one(variant, seed, args.pose_model, args.epochs, args.batch_size, args.quick, output_root)
            results.append(result)
            print(json.dumps({key: result[key] for key in ('model', 'seed', 'test_accuracy', 'f1_macro', 'recall_line', 'recall_scrum')}, indent=2))
    aggregate(results, output_root)
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
