"""Train the temporal baseline with fixed current validation and training-only balancing."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
from models.common.legacy_training import rgb_dataset
import json
import os
import sys
import time
import gc
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
sys.path.insert(0, str(source_directory(experiment_directory(__file__))))
from models.M0.studies.independent_evaluation.protocol import BALANCED_TRAIN, FEATURE_ROOT, LABELS, MODEL_DIR, OUTPUT_DIR, SEED, read_manifest, write_json
RESULT_DIR = OUTPUT_DIR / 'simple_unique_validation'
MAX_SEQ_LENGTH = 200
NUM_FEATURES = 2048
VALIDATION_PER_CLASS = 120
TRAIN_TARGET_PER_CLASS = 480
EPOCHS = 100
BATCH_SIZE = 16

def create_controlled_split() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_source, validation_source, _test = current_frames()
    rng = np.random.default_rng(SEED)
    parts = []
    for label in LABELS:
        group = train_source[train_source.tag == label]
        indexes = rng.choice(len(group), TRAIN_TARGET_PER_CLASS, replace=len(group) < TRAIN_TARGET_PER_CLASS)
        parts.append(group.iloc[indexes])
    train = pd.concat(parts, ignore_index=True).sample(frac=1., random_state=SEED).reset_index(drop=True)
    validation = validation_source.reset_index(drop=True)
    audit = {'seed': SEED, 'partition_policy': 'current_fixed_csv',
             'train_rows': len(train), 'validation_rows': len(validation),
             'new_train_validation_overlap': len(set(train.video_name) & set(validation.video_name))}
    if audit['new_train_validation_overlap']:
        raise ValueError('Entrenamiento y validación comparten clips')
    return train, validation, audit

def load_arrays(frame: pd.DataFrame) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    return rgb_dataset(frame, LABELS)

def build_model() -> keras.Model:
    feature_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES), name='features')
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool', name='mask')
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True))(feature_input, mask=mask_input)
    x = keras.layers.Bidirectional(keras.layers.GRU(32))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(3, activation='softmax')(x)
    model = keras.Model([feature_input, mask_input], output, name='resnet50_bigru_unique_validation')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def bootstrap_intervals(actual: np.ndarray, predicted: np.ndarray, draws: int=2000) -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED)
    values = {'accuracy': [], 'f1_macro': [], 'f1_weighted': []}
    for _ in range(draws):
        indices = rng.integers(0, len(actual), size=len(actual))
        y_true = actual[indices]
        y_pred = predicted[indices]
        values['accuracy'].append(accuracy_score(y_true, y_pred))
        values['f1_macro'].append(f1_score(y_true, y_pred, average='macro', zero_division=0))
        values['f1_weighted'].append(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    return {key: [float(np.quantile(series, 0.025)), float(np.quantile(series, 0.975))] for key, series in values.items()}

def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    train, validation, split_audit = create_controlled_split()
    test = read_manifest(CURRENT_DATASET / 'test.csv')
    train.to_csv(RESULT_DIR / 'train_balanced.csv', index=False)
    validation.to_csv(RESULT_DIR / 'validation_unique.csv', index=False)
    test.to_csv(RESULT_DIR / 'test_historical_817.csv', index=False)
    write_json(RESULT_DIR / 'split_audit.json', split_audit)
    print(json.dumps(split_audit, ensure_ascii=False, indent=2), flush=True)
    print('Loading cached ResNet-50 features...', flush=True)
    train_inputs, train_labels = load_arrays(train)
    validation_inputs, validation_labels = load_arrays(validation)
    keras.utils.set_random_seed(SEED)
    model = build_model()
    checkpoint = RESULT_DIR / 'best_model_unique_validation.keras'
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True, verbose=1), keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-06, verbose=1)]
    started = time.perf_counter()
    history = model.fit(train_inputs, train_labels, validation_data=(validation_inputs, validation_labels), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, shuffle=True, verbose=2)
    duration = time.perf_counter() - started
    del train_inputs, train_labels, validation_inputs, validation_labels
    gc.collect()
    print('Loading historical test features after releasing training arrays...', flush=True)
    test_inputs, test_labels = load_arrays(test)
    probabilities = model.predict(test_inputs, batch_size=BATCH_SIZE, verbose=0)
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(test_labels, predicted, labels=np.arange(len(LABELS)), zero_division=0)
    matrix = confusion_matrix(test_labels, predicted, labels=np.arange(len(LABELS)))
    metrics = {'model': 'ResNet-50 + BiGRU', 'seed': SEED, 'epochs_executed': len(history.history['loss']), 'best_epoch_by_validation_accuracy': int(np.argmax(history.history['val_accuracy']) + 1), 'training_seconds': duration, 'test_n': len(test_labels), 'test_accuracy': float(accuracy_score(test_labels, predicted)), 'test_f1_macro': float(f1_score(test_labels, predicted, average='macro')), 'test_f1_weighted': float(f1_score(test_labels, predicted, average='weighted')), 'per_class': {label: {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i, label in enumerate(LABELS)}, 'confusion_matrix': matrix.tolist(), 'bootstrap_95': bootstrap_intervals(test_labels, predicted), 'historical_comparison': {'accuracy': 0.812, 'f1_macro': 0.787, 'recall_line': 0.853, 'recall_scrum': 0.844}}
    metrics['delta_vs_historical'] = {'accuracy': metrics['test_accuracy'] - 0.812, 'f1_macro': metrics['test_f1_macro'] - 0.787, 'recall_line': metrics['per_class']['line']['recall'] - 0.853, 'recall_scrum': metrics['per_class']['scrum']['recall'] - 0.844}
    write_json(RESULT_DIR / 'metrics.json', metrics)
    pd.DataFrame(history.history).to_csv(RESULT_DIR / 'history.csv', index=False)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(RESULT_DIR / 'confusion_matrix.csv')
    (RESULT_DIR / 'classification_report.txt').write_text(classification_report(test_labels, predicted, target_names=LABELS, zero_division=0), encoding='utf-8')
    predictions = pd.DataFrame({'video_name': test.video_name, 'actual_label': [LABELS[i] for i in test_labels], 'predicted_label': [LABELS[i] for i in predicted]})
    for index, label in enumerate(LABELS):
        predictions[f'probability_{label}'] = probabilities[:, index]
    predictions.to_csv(RESULT_DIR / 'predictions.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'], label='entrenamiento')
    axes[0].plot(history.history['val_loss'], label='validaciÃ³n')
    axes[0].set_title('PÃ©rdida')
    axes[1].plot(history.history['accuracy'], label='entrenamiento')
    axes[1].plot(history.history['val_accuracy'], label='validaciÃ³n')
    axes[1].set_title('Exactitud')
    for axis in axes:
        axis.set_xlabel('Ã‰poca')
        axis.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / 'history.png', dpi=140)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=axis, cmap='Blues', colorbar=False)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / 'confusion_matrix.png', dpi=140)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
