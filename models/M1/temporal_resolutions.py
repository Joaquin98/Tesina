"""Entrena la variante ganadora con distintas longitudes temporales.

Las cachés RGB y YOLO se mantienen en 200 pasos. Para 50 y 100 pasos se
seleccionan uniformemente los índices correspondientes, sin ejecutar de nuevo
los extractores. La arquitectura es la misma ``all_residual`` ganadora y
sólo cambia la cantidad de pasos que recibe la BiGRU.
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
from models.M1.config import LABELS, RESULTS_DIR, RGB_DIM
from models.M1.protocol import build_partition
from models.M1.same_rnn import MacroF1Checkpoint, initialize_from_m0, load_arrays
SOURCE_LENGTH = 200
VARIANT = 'all_residual'

def temporal_indexes(target_length: int) -> np.ndarray:
    if target_length <= 0 or target_length > SOURCE_LENGTH:
        raise ValueError(f'La longitud debe estar entre 1 y {SOURCE_LENGTH}: {target_length}')
    return np.rint(np.linspace(0, SOURCE_LENGTH - 1, target_length)).astype(np.int32)

def resample_arrays(arrays: dict[str, np.ndarray], target_length: int) -> dict[str, np.ndarray]:
    indexes = temporal_indexes(target_length)
    output = dict(arrays)
    for key in ('rgb', 'pose'):
        output[key] = arrays[key][:, indexes, :]
    for key in ('rgb_mask', 'pose_mask'):
        output[key] = arrays[key][:, indexes]
    return output

def model_inputs(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {'rgb': arrays['rgb'], 'pose': arrays['pose'], 'sequence_mask': arrays['rgb_mask']}

def build_model(length: int, pose_dim: int) -> keras.Model:
    rgb = keras.Input((length, RGB_DIM), name='rgb')
    pose = keras.Input((length, pose_dim), name='pose')
    sequence_mask = keras.Input((length,), dtype='bool', name='sequence_mask')
    pose_delta = keras.layers.Dense(RGB_DIM, activation='tanh', kernel_initializer='zeros', bias_initializer='zeros', name='pose_residual_projection')(pose)
    pose_delta = keras.layers.Rescaling(0.1, name='pose_residual_scale')(pose_delta)
    x = keras.layers.Add(name='pose_residual_add')([rgb, pose_delta])
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='same_bigru_1')(x, mask=sequence_mask)
    x = keras.layers.Bidirectional(keras.layers.GRU(32), name='same_bigru_2')(x, mask=sequence_mask)
    x = keras.layers.Dropout(0.3, name='same_dropout')(x)
    x = keras.layers.Dense(64, activation='relu', name='same_fusion_dense')(x)
    output = keras.layers.Dense(len(LABELS), activation='softmax', name='same_class_output')(x)
    model = keras.Model(inputs={'rgb': rgb, 'pose': pose, 'sequence_mask': sequence_mask}, outputs=output, name=f'all_residual_frames_{length}')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_one(target_length: int, seed: int, pose_model: str, epochs: int, batch_size: int, quick: bool, output_root: Path) -> dict:
    keras.utils.set_random_seed(seed)
    output_dir = output_root / f'frames_{target_length}' / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=pose_model)
    train_arrays, train_labels, train_rows = load_arrays(train_df, pose_model, VARIANT)
    validation_arrays, validation_labels, validation_rows = load_arrays(validation_df, pose_model, VARIANT)
    test_arrays, test_labels, test_rows = load_arrays(test_df, pose_model, VARIANT)
    train_arrays = resample_arrays(train_arrays, target_length)
    validation_arrays = resample_arrays(validation_arrays, target_length)
    test_arrays = resample_arrays(test_arrays, target_length)
    model = build_model(target_length, int(train_arrays['pose'].shape[-1]))
    initialize_from_m0(model, seed)
    checkpoint = output_dir / 'best.weights.h5'
    callback = MacroF1Checkpoint(model_inputs(validation_arrays), validation_labels, checkpoint, patience=5 if quick else 10)
    values = compute_class_weight(class_weight='balanced', classes=np.arange(len(LABELS)), y=train_labels)
    class_weights = {int(index): float(value) for index, value in enumerate(values)}
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
    result = {'model': f'{VARIANT}_frames_{target_length}', 'seed': int(seed), 'target_length': int(target_length), 'source_length': SOURCE_LENGTH, 'test_accuracy': float(accuracy_score(test_labels, predictions)), 'f1_macro': float(f1_score(test_labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(test_labels, predictions, average='weighted', zero_division=0)), 'recall_line': float(report['line']['recall']), 'recall_other': float(report['other']['recall']), 'recall_scrum': float(report['scrum']['recall']), 'f1_line': float(report['line']['f1-score']), 'f1_other': float(report['other']['f1-score']), 'f1_scrum': float(report['scrum']['f1-score']), 'training_seconds': float(duration), 'epochs_requested': int(epochs), 'epochs_completed': int(len(history.history.get('loss', []))), 'batch_size': int(batch_size), 'pose_model': pose_model, 'pose_dim_aligned': int(train_arrays['pose'].shape[-1]), 'protocol': protocol}
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
    parser.add_argument('--lengths', default='50,100,200')
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-root', default=str(RESULTS_DIR / 'temporal_resolutions_all_residual'))
    args = parser.parse_args()
    lengths = [int(item.strip()) for item in args.lengths.split(',') if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for length in lengths:
        for seed in seeds:
            print(f'\n=== temporal all_residual length={length} seed={seed} ===')
            result = train_one(length, seed, args.pose_model, args.epochs, args.batch_size, args.quick, output_root)
            results.append(result)
            print(json.dumps({key: result[key] for key in ('model', 'seed', 'test_accuracy', 'f1_macro', 'recall_line', 'recall_scrum')}, indent=2))
    aggregate(results, output_root)
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
