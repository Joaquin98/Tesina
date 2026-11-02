"""Train sparse temporal variants with the fixed current validation partition."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, auxiliary_directory, cached_rgb as current_rgb, current_frames, validation_data as current_validation
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from models.M0.config import TRAIN_CSV, TEST_CSV, FEATURE_CACHE_DIR, MAX_SEQ_LENGTH, NUM_FEATURES, LEARNING_RATE, WEIGHT_DECAY, OPTIMIZER_NAME, VALIDATION_SPLIT, CLASSES_TO_USE
from models.M0.config import BATCH_SIZE as _CFG_BATCH_SIZE
from models.M0.config import EPOCHS as _CFG_EPOCHS
from models.M0.model import SEQUENCE_MODELS, build_optimizer
try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    print('scikit-learn not found; class weights and detailed metrics disabled.')
    SKLEARN_AVAILABLE = False
for arg in sys.argv[1:]:
    if '=' in arg:
        key, value = arg.split('=', 1)
        os.environ[key.upper()] = value

def _env_bool(key: str, default: bool=False) -> bool:
    raw = os.environ.get(key, str(int(default))).strip().lower()
    return raw in ('1', 'true', 'yes', 'y')

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key, '')
    values = [item.strip() for item in raw.split(',') if item.strip()]
    return values or default

def _env_int_list(key: str, default: list[int]) -> list[int]:
    raw_values = _env_list(key, [str(value) for value in default])
    return [int(value) for value in raw_values]
BASE_DIR = Path(__file__).resolve().parent
BASE_RESULTS_DIR = auxiliary_directory('train_light_live') / 'results'
BASE_CHECKPOINTS_DIR = auxiliary_directory('train_light_live') / 'checkpoints'
SUMMARY_CSV_PATH = BASE_RESULTS_DIR / 'light_live_summary.csv'
TARGET_FPS = 25
WINDOW_SECONDS = 7
QUICK = _env_bool('QUICK', False)
EPOCHS = _env_int('EPOCHS', 3 if QUICK else _CFG_EPOCHS)
BATCH_SIZE = _env_int('BATCH_SIZE', _CFG_BATCH_SIZE)
FRAME_COUNTS = _env_int_list('FRAME_COUNTS', [25, 50, 75])
EXTRACTORS = _env_list('EXTRACTORS', ['resnet50', 'efficientnetb0', 'inceptionv3'])
RNN_MODELS = _env_list('RNN_MODELS', ['gru_small', 'gru_big', 'lstm'])
if QUICK:
    QUICK_TRAIN_ROWS = _env_int('QUICK_TRAIN_ROWS', 150)
    QUICK_TEST_ROWS = _env_int('QUICK_TEST_ROWS', 60)
else:
    QUICK_TRAIN_ROWS = 0
    QUICK_TEST_ROWS = 0

def build_sparse_temporal_view(frame_features: np.ndarray, frame_mask: np.ndarray, sampled_frame_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a sparse feature/mask view with selected valid steps packed first."""
    if sampled_frame_count <= 0:
        raise ValueError('sampled_frame_count must be positive for light-live training')
    if sampled_frame_count > MAX_SEQ_LENGTH:
        raise ValueError(f'sampled_frame_count cannot exceed MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}')
    valid_indices = np.flatnonzero(frame_mask.astype(bool))
    output_features = np.zeros((MAX_SEQ_LENGTH, NUM_FEATURES), dtype=np.float32)
    output_mask = np.zeros((MAX_SEQ_LENGTH,), dtype=bool)
    if len(valid_indices) == 0:
        return (output_features, output_mask, np.array([], dtype=np.int32))
    take_count = min(sampled_frame_count, len(valid_indices))
    positions = np.linspace(0, len(valid_indices) - 1, take_count, dtype=np.int32)
    selected_indices = valid_indices[positions]
    output_features[:take_count] = frame_features[selected_indices].astype(np.float32, copy=False)
    output_mask[:take_count] = True
    return (output_features, output_mask, selected_indices.astype(np.int32))

def _cache_path(extractor: str, video_name: str) -> Path:
    stem = Path(video_name).stem
    return Path(FEATURE_CACHE_DIR) / extractor / f'{stem}.npz'

def load_cached_features(video_name: str, extractor: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    return current_rgb(video_name, extractor)

def build_dataset(df: pd.DataFrame, label_processor, extractor: str, sampled_frame_count: int, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    features = []
    masks = []
    labels = []
    valid_counts = []
    missing = 0
    empty = 0
    for _, row in df.iterrows():
        video_name = row['video_name']
        frame_features, frame_mask = load_cached_features(video_name, extractor)
        if frame_features is None:
            missing += 1
            continue
        sparse_features, sparse_mask, _ = build_sparse_temporal_view(frame_features, frame_mask, sampled_frame_count)
        if not sparse_mask.any():
            empty += 1
            continue
        tag = str(row['tag']).strip().lower()
        encoded = int(keras.ops.convert_to_numpy(label_processor([tag]))[0])
        if encoded < 0:
            continue
        features.append(sparse_features)
        masks.append(sparse_mask)
        labels.append(encoded)
        valid_counts.append(int(sparse_mask.sum()))
    stats = {f'{split_name}_missing_cache': missing, f'{split_name}_empty_features': empty, f'{split_name}_samples': len(labels), f'{split_name}_min_valid_steps': min(valid_counts) if valid_counts else 0, f'{split_name}_max_valid_steps': max(valid_counts) if valid_counts else 0, f'{split_name}_mean_valid_steps': round(float(np.mean(valid_counts)), 3) if valid_counts else 0.0}
    return (np.asarray(features, dtype=np.float32), np.asarray(masks, dtype=bool), np.asarray(labels, dtype=np.int32), stats)

def save_history_plot(history, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get('loss', []), label='train')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history.history.get('accuracy', []), label='train')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='val')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

def train_one(extractor: str, rnn_name: str, sampled_frame_count: int, train_features, train_masks, train_labels, test_features, test_masks, test_labels, label_vocab: list[str], dataset_stats: dict) -> dict:
    run_name = f'{extractor}__{rnn_name}__frames{sampled_frame_count}'
    results_dir = BASE_RESULTS_DIR / run_name
    checkpoint_dir = BASE_CHECKPOINTS_DIR / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print('\n' + '=' * 72)
    print(f'Light live run: {run_name}')
    print(f'Train samples: {len(train_labels)} | Test samples: {len(test_labels)}')
    print('=' * 72)
    optimizer = build_optimizer(OPTIMIZER_NAME, LEARNING_RATE, WEIGHT_DECAY)
    model = SEQUENCE_MODELS[rnn_name](len(label_vocab), optimizer)
    class_weight = None
    if SKLEARN_AVAILABLE:
        try:
            weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            class_weight = dict(enumerate(weights))
            print(f'Class weights: {class_weight}')
        except Exception as exc:
            print(f'Could not compute class weights: {exc}')
    checkpoint_path = checkpoint_dir / 'model.keras'
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=8, restore_best_weights=True, verbose=1), keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', mode='max', save_best_only=True, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-06, verbose=1)]
    start = time.time()
    history = model.fit([train_features, train_masks], train_labels, validation_data=current_validation(label_vocab, extractor, sampled_frame_count), epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight, callbacks=callbacks, verbose=1)
    duration = time.time() - start
    test_loss, test_accuracy = model.evaluate([test_features, test_masks], test_labels, verbose=0)
    metrics = {'run_name': run_name, 'extractor': extractor, 'sequence_model': rnn_name, 'sampled_frame_count': sampled_frame_count, 'target_fps': TARGET_FPS, 'window_seconds': WINDOW_SECONDS, 'max_seq_length': MAX_SEQ_LENGTH, 'num_features': NUM_FEATURES, 'train_csv': str(TRAIN_CSV), 'test_csv': str(TEST_CSV), 'epochs_requested': EPOCHS, 'epochs_ran': len(history.history.get('loss', [])), 'batch_size': BATCH_SIZE, 'validation_csv': str(CURRENT_DATASET / 'validation.csv'), 'test_loss': round(float(test_loss), 6), 'test_accuracy': round(float(test_accuracy), 6), 'train_last_acc': round(float(history.history.get('accuracy', [0.0])[-1]), 6), 'val_last_acc': round(float(history.history.get('val_accuracy', [0.0])[-1]), 6), 'duration_s': round(duration, 1), 'checkpoint_path': str(checkpoint_path), **dataset_stats}
    if SKLEARN_AVAILABLE:
        probabilities = model.predict([test_features, test_masks], verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        metrics['f1_macro'] = round(float(f1_score(test_labels, predictions, average='macro', zero_division=0)), 4)
        metrics['f1_weighted'] = round(float(f1_score(test_labels, predictions, average='weighted', zero_division=0)), 4)
        report = classification_report(test_labels, predictions, target_names=label_vocab, zero_division=0)
        (results_dir / 'classification_report.txt').write_text(report, encoding='utf-8')
        cm = confusion_matrix(test_labels, predictions)
        row_sums = cm.sum(axis=1)
        recalls = np.divide(np.diag(cm).astype(float), row_sums, out=np.zeros(len(label_vocab)), where=row_sums != 0)
        for idx, label in enumerate(label_vocab):
            metrics[f'recall_{label}'] = round(float(recalls[idx]), 4)
        for normalize, suffix in [(None, 'raw'), ('true', 'normalized')]:
            matrix = confusion_matrix(test_labels, predictions, normalize=normalize)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(matrix, display_labels=label_vocab).plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'{run_name} ({suffix})')
            fig.tight_layout()
            fig.savefig(results_dir / f'confusion_matrix_{suffix}.png', dpi=100)
            plt.close(fig)
    save_history_plot(history, run_name, results_dir / 'history.png')
    np.save(checkpoint_dir / 'label_vocab.npy', np.asarray(label_vocab))
    metadata = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'purpose': 'Light live model trained with sparse temporal views matching live sampled-frame inference.', 'live_detection_alignment': {'target_fps': TARGET_FPS, 'window_seconds': WINDOW_SECONDS, 'sampled_frame_count': sampled_frame_count, 'ui_instruction': f'Use Target FPS={TARGET_FPS}, Window={WINDOW_SECONDS}s, Sampled Frames={sampled_frame_count}.'}, 'metrics': metrics}
    (checkpoint_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    row_path = results_dir / 'metrics.json'
    row_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics

def main() -> None:
    print('=' * 72)
    print('Light Live Training')
    print(f'Extractors: {EXTRACTORS}')
    print(f'RNN models: {RNN_MODELS}')
    print(f'Frame counts: {FRAME_COUNTS}')
    print(f'Epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | Quick: {QUICK}')
    print('=' * 72)
    train_df = pd.read_csv(_resolve_model_artifact(TRAIN_CSV))
    test_df = pd.read_csv(_resolve_model_artifact(TEST_CSV))
    if CLASSES_TO_USE is not None:
        clean_classes = [str(value).strip().lower() for value in CLASSES_TO_USE]
        train_df = train_df[train_df['tag'].astype(str).str.strip().str.lower().isin(clean_classes)].copy()
        test_df = test_df[test_df['tag'].astype(str).str.strip().str.lower().isin(clean_classes)].copy()
    if QUICK:
        train_df = train_df.head(QUICK_TRAIN_ROWS)
        test_df = test_df.head(QUICK_TEST_ROWS)
    unique_tags = sorted(set((str(tag).strip().lower() for tag in train_df['tag'].unique())))
    label_processor = keras.layers.StringLookup(num_oov_indices=0, mask_token=None, vocabulary=unique_tags)
    label_vocab = list(label_processor.get_vocabulary())
    print(f'Label vocabulary: {label_vocab}')
    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for extractor in EXTRACTORS:
        extractor_dir = Path(FEATURE_CACHE_DIR) / extractor
        for sampled_frame_count in FRAME_COUNTS:
            print(f'\nLoading {extractor} features as frames{sampled_frame_count}...')
            train_features, train_masks, train_labels, train_stats = build_dataset(train_df, label_processor, extractor, sampled_frame_count, 'train')
            test_features, test_masks, test_labels, test_stats = build_dataset(test_df, label_processor, extractor, sampled_frame_count, 'test')
            if len(train_labels) == 0 or len(test_labels) == 0:
                print(f'No data loaded for {extractor} frames{sampled_frame_count}; skipping.')
                continue
            dataset_stats = {**train_stats, **test_stats}
            print(f'Loaded {len(train_labels)} train and {len(test_labels)} test samples.')
            print(f"Train valid steps: {train_stats['train_min_valid_steps']}..{train_stats['train_max_valid_steps']}")
            for rnn_name in RNN_MODELS:
                if rnn_name not in SEQUENCE_MODELS:
                    print(f'Skipping unknown sequence model: {rnn_name}')
                    continue
                try:
                    metrics = train_one(extractor, rnn_name, sampled_frame_count, train_features, train_masks, train_labels, test_features, test_masks, test_labels, label_vocab, dataset_stats)
                    all_metrics.append(metrics)
                except Exception as exc:
                    print(f'Training failed for {extractor} {rnn_name} frames{sampled_frame_count}: {exc}')
                    import traceback
                    traceback.print_exc()
    if all_metrics:
        keys = []
        for metrics in all_metrics:
            for key in metrics:
                if key not in keys:
                    keys.append(key)
        with _resolve_model_artifact(SUMMARY_CSV_PATH).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            for metrics in all_metrics:
                writer.writerow(metrics)
        print(f'\nSummary saved to {SUMMARY_CSV_PATH}')
        summary = pd.DataFrame(all_metrics)
        display_cols = ['extractor', 'sequence_model', 'sampled_frame_count', 'test_accuracy', 'f1_macro', 'recall_line', 'recall_scrum', 'duration_s']
        existing_cols = [col for col in display_cols if col in summary.columns]
        print(summary[existing_cols].to_string(index=False))
    else:
        print('No light-live experiments completed.')
if __name__ == '__main__':
    main()
