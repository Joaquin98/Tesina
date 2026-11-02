"""Variantes que incorporan geometria de jugadores en la RNN temporal RGB.

La rama RGB conserva sus 200 pasos y 2048 caracterÃ\xadsticas. La postura se
alinea de 32 a 200 pasos, se concatena en cada instante y entra a la misma
BiGRU. Los modelos se inicializan con el checkpoint M0 extendiendo con ceros
el kernel de entrada de la primera GRU, de modo que empiezan representando al
modelo histÃ³rico y aprenden el aporte de la postura.
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
from pathlib import Path
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from models.M1.config import GEOMETRY_LENGTH, LABELS, LABEL_TO_ID, RESULTS_DIR, RGB_DIM, RGB_LENGTH
from models.M1.protocol import build_partition, load_rgb_features, pose_cache_path
VARIANTS = ('pose_concat', 'pose_projected', 'raw4_projected', 'raw8_projected', 'pose_residual', 'all_concat', 'all_projected', 'all_residual')
RAW_FEATURES_PER_PERSON = 52

def align_to_rgb(sequence: np.ndarray, length: int=RGB_LENGTH) -> np.ndarray:
    old = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
    new = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.stack([np.interp(new, old, sequence[:, index]) for index in range(sequence.shape[1])], axis=1).astype(np.float32)

def raw_pose_sequence(path: Path, people: int) -> tuple[np.ndarray, np.ndarray]:
    with _load_model_result_array(path, allow_pickle=False) as data:
        boxes = np.asarray(data['boxes'], dtype=np.float32)
        keypoints = np.asarray(data['keypoints'], dtype=np.float32)
        keypoint_conf = np.asarray(data['keypoint_conf'], dtype=np.float32)
        person_conf = np.asarray(data['person_conf'], dtype=np.float32)
        valid = np.asarray(data['valid'], dtype=bool)
        frame_mask = np.asarray(data['frame_mask'], dtype=bool)
    values = np.zeros((GEOMETRY_LENGTH, people * RAW_FEATURES_PER_PERSON), dtype=np.float32)
    for frame_index in range(GEOMETRY_LENGTH):
        indexes = np.flatnonzero(valid[frame_index])
        if len(indexes):
            centers = (boxes[frame_index, indexes, 0] + boxes[frame_index, indexes, 2]) * 0.5
            indexes = indexes[np.argsort(centers)]
        for output_index, source_index in enumerate(indexes[:people]):
            start = output_index * RAW_FEATURES_PER_PERSON
            values[frame_index, start:start + 34] = keypoints[frame_index, source_index].reshape(-1)
            values[frame_index, start + 34:start + 51] = keypoint_conf[frame_index, source_index]
            values[frame_index, start + 51] = person_conf[frame_index, source_index]
    return (values, frame_mask)

def pose_sequence(video_name: str, pose_model: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    path = pose_cache_path(video_name, pose_model)
    with _load_model_result_array(path, allow_pickle=False) as data:
        descriptor = np.asarray(data['pose_features'], dtype=np.float32)
        positions = np.asarray(data['position_features'], dtype=np.float32)
        frame_mask = np.asarray(data['frame_mask'], dtype=bool)
    if variant in ('pose_concat', 'pose_projected', 'pose_residual'):
        values = descriptor
    elif variant == 'raw4_projected':
        values, frame_mask = raw_pose_sequence(path, 4)
    elif variant == 'raw8_projected':
        values, frame_mask = raw_pose_sequence(path, 8)
    elif variant in ('all_concat', 'all_projected', 'all_residual'):
        values = np.concatenate([positions, descriptor], axis=1)
    else:
        raise ValueError(f'Variante desconocida: {variant}')
    aligned = align_to_rgb(values)
    aligned_mask = np.interp(np.linspace(0.0, 1.0, RGB_LENGTH, dtype=np.float32), np.linspace(0.0, 1.0, len(frame_mask), dtype=np.float32), frame_mask.astype(np.float32)).astype(np.float32)
    return (np.concatenate([aligned, aligned_mask[:, None]], axis=1), aligned_mask > 0.5)

def load_arrays(df: pd.DataFrame, pose_model: str, variant: str) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    rgb_values = []
    rgb_masks = []
    pose_values = []
    pose_masks = []
    labels = []
    rows = []
    for row in df.to_dict('records'):
        name = str(row['video_name'])
        rgb, rgb_mask = load_rgb_features(name)
        pose, pose_mask = pose_sequence(name, pose_model, variant)
        rgb_values.append(rgb)
        rgb_masks.append(rgb_mask)
        pose_values.append(pose)
        pose_masks.append(pose_mask)
        labels.append(LABEL_TO_ID[str(row['tag'])])
        rows.append(row)
    arrays = {'rgb': np.asarray(rgb_values, dtype=np.float32), 'rgb_mask': np.asarray(rgb_masks, dtype=bool), 'pose': np.asarray(pose_values, dtype=np.float32), 'pose_mask': np.asarray(pose_masks, dtype=bool)}
    return (arrays, np.asarray(labels, dtype=np.int32), pd.DataFrame(rows))

def branch_model(variant: str, pose_dim: int) -> keras.Model:
    rgb = keras.Input((RGB_LENGTH, RGB_DIM), name='rgb')
    pose = keras.Input((RGB_LENGTH, pose_dim), name='pose')
    mask = keras.Input((RGB_LENGTH,), dtype='bool', name='sequence_mask')
    if variant in ('pose_residual', 'all_residual'):
        pose_delta = keras.layers.Dense(RGB_DIM, activation='tanh', kernel_initializer='zeros', bias_initializer='zeros', name='pose_residual_projection')(pose)
        pose_delta = keras.layers.Rescaling(0.1, name='pose_residual_scale')(pose_delta)
        x = keras.layers.Add(name='pose_residual_add')([rgb, pose_delta])
    elif variant in ('pose_concat', 'all_concat'):
        pose_representation = pose
        x = keras.layers.Concatenate(name='same_rnn_input')([rgb, pose_representation])
    else:
        pose_representation = keras.layers.Dense(64, activation='relu', name='pose_projection')(pose)
        pose_representation = keras.layers.LayerNormalization(name='pose_projection_norm')(pose_representation)
        x = keras.layers.Concatenate(name='same_rnn_input')([rgb, pose_representation])
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='same_bigru_1')(x, mask=mask)
    x = keras.layers.Bidirectional(keras.layers.GRU(32), name='same_bigru_2')(x, mask=mask)
    x = keras.layers.Dropout(0.3, name='same_dropout')(x)
    x = keras.layers.Dense(64, activation='relu', name='same_fusion_dense')(x)
    output = keras.layers.Dense(len(LABELS), activation='softmax', name='same_class_output')(x)
    model = keras.Model(inputs={'rgb': rgb, 'pose': pose, 'sequence_mask': mask}, outputs=output, name=f'same_rnn_{variant}')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def source_weight_layer(model: keras.Model, kind: type, position: int):
    layers = [layer for layer in model.layers if isinstance(layer, kind)]
    return layers[position]

def initialize_from_m0(model: keras.Model, seed: int) -> None:
    from models.M1.train import build_model
    source, _ = build_model('m0', 31, 30)
    checkpoint = RESULTS_DIR / 'm0' / f'seed_{seed}' / 'best.weights.h5'
    if not _resolve_model_artifact(checkpoint).exists():
        raise FileNotFoundError(f'No existe el checkpoint M0: {checkpoint}')
    source.load_weights(_resolve_model_artifact(checkpoint))
    source_bigru = [layer for layer in source.layers if isinstance(layer, keras.layers.Bidirectional)]
    target_bigru = [layer for layer in model.layers if isinstance(layer, keras.layers.Bidirectional)]
    for source_layer, target_layer in zip(source_bigru, target_bigru):
        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()
        copied = []
        for source_weight, target_weight in zip(source_weights, target_weights):
            if source_weight.shape == target_weight.shape:
                copied.append(source_weight)
            elif source_weight.ndim == 2 and source_weight.shape[0] == RGB_DIM and (target_weight.shape[0] > RGB_DIM):
                expanded = np.zeros_like(target_weight)
                expanded[:RGB_DIM, :source_weight.shape[1]] = source_weight
                copied.append(expanded)
            else:
                copied.append(target_weight)
        target_layer.set_weights(copied)
    source_dense = [layer for layer in source.layers if layer.name.startswith('fusion_dense')][0]
    target_dense = model.get_layer('same_fusion_dense')
    target_dense.set_weights(source_dense.get_weights())
    source_output = [layer for layer in source.layers if layer.name.startswith('class_output')][0]
    target_output = model.get_layer('same_class_output')
    target_output.set_weights(source_output.get_weights())

class MacroF1Checkpoint(keras.callbacks.Callback):

    def __init__(self, validation_inputs, validation_labels, checkpoint: Path, patience: int):
        super().__init__()
        self.validation_inputs = validation_inputs
        self.validation_labels = validation_labels
        self.checkpoint = checkpoint
        self.patience = patience
        self.best = -np.inf
        self.wait = 0
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        probabilities = self.model.predict(self.validation_inputs, verbose=0)
        predictions = probabilities.argmax(axis=1)
        score = float(f1_score(self.validation_labels, predictions, average='macro', zero_division=0))
        logs['val_f1_macro'] = score
        self.records.append({'epoch': epoch + 1, 'val_f1_macro': score, **{key: float(value) for key, value in logs.items() if isinstance(value, (float, int, np.floating, np.integer))}})
        if score > self.best + 1e-08:
            self.best = score
            self.wait = 0
            self.model.save_weights(self.checkpoint)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

def inputs(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {'rgb': arrays['rgb'], 'pose': arrays['pose'], 'sequence_mask': arrays['rgb_mask']}

def train_one(variant: str, seed: int, pose_model: str, epochs: int, batch_size: int, quick: bool, warm_start: bool, output_root: Path) -> dict:
    output_dir = output_root / variant / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=pose_model)
    train_arrays, train_labels, train_rows = load_arrays(train_df, pose_model, variant)
    validation_arrays, validation_labels, validation_rows = load_arrays(validation_df, pose_model, variant)
    test_arrays, test_labels, test_rows = load_arrays(test_df, pose_model, variant)
    pose_dim = int(train_arrays['pose'].shape[-1])
    model = branch_model(variant, pose_dim)
    if warm_start:
        initialize_from_m0(model, seed)
    checkpoint = output_dir / 'best.weights.h5'
    callback = MacroF1Checkpoint(inputs(validation_arrays), validation_labels, checkpoint, patience=5 if quick else 10)
    classes = np.arange(len(LABELS))
    values = compute_class_weight(class_weight='balanced', classes=classes, y=train_labels)
    class_weights = {int(index): float(value) for index, value in enumerate(values)}
    started = time.perf_counter()
    history = model.fit(inputs(train_arrays), train_labels, validation_data=(inputs(validation_arrays), validation_labels), epochs=epochs, batch_size=batch_size, class_weight=class_weights, callbacks=[callback], verbose=1)
    duration = time.perf_counter() - started
    if _resolve_model_artifact(checkpoint).exists():
        model.load_weights(_resolve_model_artifact(checkpoint))
    probabilities = model.predict(inputs(test_arrays), verbose=0)
    predictions = probabilities.argmax(axis=1)
    report = classification_report(test_labels, predictions, labels=list(range(len(LABELS))), target_names=LABELS, output_dict=True, zero_division=0)
    frame = pd.DataFrame({'video_name': test_rows['video_name'].tolist(), 'true_label': [LABELS[index] for index in test_labels], 'predicted_label': [LABELS[index] for index in predictions]})
    for index, label in enumerate(LABELS):
        frame[f'prob_{label}'] = probabilities[:, index]
    frame.to_csv(output_dir / 'predictions.csv', index=False)
    pd.DataFrame(confusion_matrix(test_labels, predictions, labels=list(range(len(LABELS)))), index=LABELS, columns=LABELS).to_csv(output_dir / 'confusion_matrix.csv')
    (output_dir / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame(callback.records).to_csv(output_dir / 'history.csv', index=False)
    result = {'model': variant, 'seed': int(seed), 'test_accuracy': float(accuracy_score(test_labels, predictions)), 'f1_macro': float(f1_score(test_labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(test_labels, predictions, average='weighted', zero_division=0)), 'recall_line': float(report['line']['recall']), 'recall_other': float(report['other']['recall']), 'recall_scrum': float(report['scrum']['recall']), 'training_seconds': float(duration), 'epochs_requested': int(epochs), 'epochs_completed': int(len(history.history.get('loss', []))), 'batch_size': int(batch_size), 'pose_model': pose_model, 'warm_start_m0': bool(warm_start), 'pose_dim_aligned': pose_dim, 'protocol': protocol}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def aggregate(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_dir / 'summary.csv', index=False)
    rows = []
    for model_name, group in pd.DataFrame(results).groupby('model'):
        rows.append({'model': model_name, 'runs': len(group), 'test_accuracy_mean': group['test_accuracy'].mean(), 'test_accuracy_std': group['test_accuracy'].std(ddof=0), 'f1_macro_mean': group['f1_macro'].mean(), 'f1_macro_std': group['f1_macro'].std(ddof=0), 'f1_weighted_mean': group['f1_weighted'].mean(), 'f1_weighted_std': group['f1_weighted'].std(ddof=0)})
    pd.DataFrame(rows).sort_values('model').to_csv(output_dir / 'aggregate_metrics.csv', index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(VARIANTS))
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-warm-start', action='store_true')
    parser.add_argument('--output-root', default=str(RESULTS_DIR / 'same_rnn'))
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(',') if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    unknown = set(models).difference(VARIANTS)
    if unknown:
        raise SystemExit(f'Variantes desconocidas: {sorted(unknown)}')
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in models:
        for seed in seeds:
            print(f'\n=== same_rnn {variant} seed={seed} ===')
            result = train_one(variant, seed, args.pose_model, args.epochs, args.batch_size, args.quick, not args.no_warm_start, output_root)
            results.append(result)
            print(json.dumps({key: result[key] for key in ('model', 'seed', 'test_accuracy', 'f1_macro', 'recall_line', 'recall_scrum')}, indent=2))
    aggregate(results, output_root)
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
