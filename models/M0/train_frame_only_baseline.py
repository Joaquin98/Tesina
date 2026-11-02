"""Train a frame-level baseline with independent current train, validation and test partitions."""
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
from models.M0.config import CLASSES_TO_USE, FEATURE_CACHE_DIR, LEARNING_RATE, MAX_SEQ_LENGTH, NUM_FEATURES, TEST_CSV, TRAIN_CSV, VALIDATION_SPLIT, WEIGHT_DECAY
from models.M0.config import BATCH_SIZE as _CFG_BATCH_SIZE
from models.M0.config import EPOCHS as _CFG_EPOCHS
from models.M0.model import build_optimizer
try:
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    print('scikit-learn not found; detailed metrics and stratified validation disabled.')
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

def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))

def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = auxiliary_directory('train_frame_only_baseline') / 'results'
CHECKPOINT_DIR = auxiliary_directory('train_frame_only_baseline') / 'checkpoints'
QUICK = _env_bool('QUICK', False)
EXTRACTOR = _env_str('EXTRACTOR', 'resnet50')
EPOCHS = _env_int('EPOCHS', 3 if QUICK else _CFG_EPOCHS)
BATCH_SIZE = _env_int('BATCH_SIZE', max(_CFG_BATCH_SIZE, 16))
FRAME_BATCH_VIDEOS = _env_int('FRAME_BATCH_VIDEOS', BATCH_SIZE)
MAX_FRAMES_PER_VIDEO = _env_int('MAX_FRAMES_PER_VIDEO', 0)
QUICK_TRAIN_ROWS = _env_int('QUICK_TRAIN_ROWS', 150)
QUICK_TEST_ROWS = _env_int('QUICK_TEST_ROWS', 60)
RANDOM_SEED = _env_int('SEED', 42)
DROPOUT = _env_float('DROPOUT', 0.4)
DENSE_UNITS = _env_int('DENSE_UNITS', 256)
SMALL_DENSE_UNITS = _env_int('SMALL_DENSE_UNITS', 64)

def _cache_path(extractor: str, video_name: str) -> Path:
    return Path(FEATURE_CACHE_DIR) / extractor / f'{Path(video_name).stem}.npz'

def load_cached_video_frames(video_name: str, extractor: str, max_frames_per_video: int=0, random_sample: bool=False) -> np.ndarray | None:
    """Load valid frame features for one video from the feature cache."""
    frame_features, frame_mask = current_rgb(video_name, extractor)
    valid_indices = np.flatnonzero(frame_mask)
    if len(valid_indices) == 0:
        return np.empty((0, NUM_FEATURES), dtype=np.float32)
    if max_frames_per_video > 0 and len(valid_indices) > max_frames_per_video:
        if random_sample:
            rng = np.random.default_rng()
            valid_indices = np.sort(rng.choice(valid_indices, size=max_frames_per_video, replace=False))
        else:
            positions = np.linspace(0, len(valid_indices) - 1, max_frames_per_video, dtype=np.int32)
            valid_indices = valid_indices[positions]
    return frame_features[valid_indices].astype(np.float32, copy=False)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['tag'] = df['tag'].astype(str).str.strip().str.lower()
    if CLASSES_TO_USE is not None:
        classes = [str(value).strip().lower() for value in CLASSES_TO_USE]
        df = df[df['tag'].isin(classes)].copy()
    return df

def encode_label(label_processor, tag: str) -> int:
    return int(keras.ops.convert_to_numpy(label_processor([tag]))[0])

class FrameFeatureSequence(keras.utils.Sequence):
    """Yields batches of independent frame features grouped by source videos."""

    def __init__(self, df: pd.DataFrame, label_processor, extractor: str, batch_videos: int, max_frames_per_video: int, shuffle: bool, random_frame_sample: bool) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.label_processor = label_processor
        self.extractor = extractor
        self.batch_videos = batch_videos
        self.max_frames_per_video = max_frames_per_video
        self.shuffle = shuffle
        self.random_frame_sample = random_frame_sample
        self.indices = np.arange(len(self.df))
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_videos))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[idx * self.batch_videos:(idx + 1) * self.batch_videos]
        frame_batches = []
        label_batches = []
        for row_index in batch_indices:
            row = self.df.iloc[int(row_index)]
            frames = load_cached_video_frames(row['video_name'], self.extractor, self.max_frames_per_video, random_sample=self.random_frame_sample)
            if frames is None or len(frames) == 0:
                continue
            label = encode_label(self.label_processor, row['tag'])
            frame_batches.append(frames)
            label_batches.append(np.full((len(frames),), label, dtype=np.int32))
        if not frame_batches:
            return (np.zeros((0, NUM_FEATURES), dtype=np.float32), np.zeros((0,), dtype=np.int32))
        return (np.concatenate(frame_batches, axis=0), np.concatenate(label_batches, axis=0))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

def inspect_split(df: pd.DataFrame, extractor: str, split_name: str) -> dict:
    missing = 0
    empty = 0
    frame_counts = []
    for _, row in df.iterrows():
        frames = load_cached_video_frames(row['video_name'], extractor, MAX_FRAMES_PER_VIDEO)
        if frames is None:
            missing += 1
        elif len(frames) == 0:
            empty += 1
        else:
            frame_counts.append(len(frames))
    return {f'{split_name}_videos': len(df), f'{split_name}_missing_cache': missing, f'{split_name}_empty_videos': empty, f'{split_name}_valid_videos': len(frame_counts), f'{split_name}_frames': int(np.sum(frame_counts)) if frame_counts else 0, f'{split_name}_min_frames': int(np.min(frame_counts)) if frame_counts else 0, f'{split_name}_max_frames': int(np.max(frame_counts)) if frame_counts else 0, f'{split_name}_mean_frames': round(float(np.mean(frame_counts)), 3) if frame_counts else 0.0}

def build_frame_classifier(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(NUM_FEATURES,), name='frame_features')
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    if SMALL_DENSE_UNITS > 0:
        x = keras.layers.Dense(SMALL_DENSE_UNITS, activation='relu')(x)
        x = keras.layers.Dropout(DROPOUT / 2.0)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='frame_only_baseline')
    optimizer = build_optimizer('adam', LEARNING_RATE, WEIGHT_DECAY)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def collect_frame_arrays(df: pd.DataFrame, label_processor, extractor: str, max_frames_per_video: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    labels = []
    video_names = []
    for _, row in df.iterrows():
        frames = load_cached_video_frames(row['video_name'], extractor, max_frames_per_video)
        if frames is None or len(frames) == 0:
            continue
        label = encode_label(label_processor, row['tag'])
        features.append(frames)
        labels.append(np.full((len(frames),), label, dtype=np.int32))
        video_names.extend([row['video_name']] * len(frames))
    if not features:
        return (np.empty((0, NUM_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.int32), [])
    return (np.concatenate(features, axis=0), np.concatenate(labels, axis=0), video_names)

def evaluate_by_video(model: keras.Model, df: pd.DataFrame, label_processor, label_vocab: list[str], output_dir: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    y_true = []
    y_pred = []
    probabilities = []
    per_video_rows = []
    skipped = 0
    for _, row in df.iterrows():
        frames = load_cached_video_frames(row['video_name'], EXTRACTOR, MAX_FRAMES_PER_VIDEO)
        if frames is None or len(frames) == 0:
            skipped += 1
            continue
        probs = model.predict(frames, batch_size=1024, verbose=0)
        mean_probs = probs.mean(axis=0)
        true_idx = encode_label(label_processor, row['tag'])
        pred_idx = int(np.argmax(mean_probs))
        y_true.append(true_idx)
        y_pred.append(pred_idx)
        probabilities.append(mean_probs)
        per_video_rows.append({'video_name': row['video_name'], 'true_label': row['tag'], 'predicted_label': label_vocab[pred_idx], 'confidence': round(float(mean_probs[pred_idx]), 6), 'correct': int(pred_idx == true_idx), 'frame_count': len(frames)})
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred, dtype=np.int32)
    prob_arr = np.asarray(probabilities, dtype=np.float32)
    metrics = {'video_samples': int(len(y_true_arr)), 'video_skipped': int(skipped), 'video_accuracy': round(float(np.mean(y_true_arr == y_pred_arr)), 6) if len(y_true_arr) else 0.0}
    if SKLEARN_AVAILABLE and len(y_true_arr):
        metrics['video_f1_macro'] = round(float(f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)), 4)
        metrics['video_f1_weighted'] = round(float(f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)), 4)
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(label_vocab))))
        row_sums = cm.sum(axis=1)
        recalls = np.divide(np.diag(cm).astype(float), row_sums, out=np.zeros(len(label_vocab)), where=row_sums != 0)
        for idx, label in enumerate(label_vocab):
            metrics[f'video_recall_{label}'] = round(float(recalls[idx]), 4)
        report = classification_report(y_true_arr, y_pred_arr, target_names=label_vocab, zero_division=0)
        (output_dir / 'video_classification_report.txt').write_text(report, encoding='utf-8')
        for normalize, suffix in [(None, 'raw'), ('true', 'normalized')]:
            matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(label_vocab))), normalize=normalize)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(matrix, display_labels=label_vocab).plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'Frame-only video aggregation ({suffix})')
            fig.tight_layout()
            fig.savefig(output_dir / f'video_confusion_matrix_{suffix}.png', dpi=100)
            plt.close(fig)
    with _resolve_model_artifact(output_dir / 'per_video_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        fieldnames = ['video_name', 'true_label', 'predicted_label', 'confidence', 'correct', 'frame_count']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_video_rows)
    return (metrics, y_true_arr, y_pred_arr, prob_arr)

def save_history_plot(history, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get('loss', []), label='train')
    axes[0].plot(history.history.get('val_loss', []), label='val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history.history.get('accuracy', []), label='train')
    axes[1].plot(history.history.get('val_accuracy', []), label='val')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    fig.suptitle('Frame-only baseline')
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

def main() -> None:
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 72)
    print('Frame-only baseline')
    print(f'Extractor: {EXTRACTOR}')
    print(f'Epochs: {EPOCHS} | Video batch: {FRAME_BATCH_VIDEOS} | Quick: {QUICK}')
    print(f"Max frames per video: {('all' if MAX_FRAMES_PER_VIDEO <= 0 else MAX_FRAMES_PER_VIDEO)}")
    print('=' * 72)
    extractor_dir = Path(FEATURE_CACHE_DIR) / EXTRACTOR
    train_df = clean_dataframe(pd.read_csv(_resolve_model_artifact(TRAIN_CSV)))
    test_df = clean_dataframe(pd.read_csv(_resolve_model_artifact(TEST_CSV)))
    if QUICK:
        train_df = train_df.head(QUICK_TRAIN_ROWS)
        test_df = test_df.head(QUICK_TEST_ROWS)
    label_vocab = sorted(set(train_df['tag'].unique()))
    label_processor = keras.layers.StringLookup(num_oov_indices=0, mask_token=None, vocabulary=label_vocab)
    print(f'Labels: {label_vocab}')
    train_split = train_df
    val_split = clean_dataframe(current_frames()[1])
    train_stats = inspect_split(train_split, EXTRACTOR, 'train')
    val_stats = inspect_split(val_split, EXTRACTOR, 'val')
    test_stats = inspect_split(test_df, EXTRACTOR, 'test')
    print(f"Train videos: {train_stats['train_valid_videos']} | frames: {train_stats['train_frames']}")
    print(f"Val videos:   {val_stats['val_valid_videos']} | frames: {val_stats['val_frames']}")
    print(f"Test videos:  {test_stats['test_valid_videos']} | frames: {test_stats['test_frames']}")
    train_sequence = FrameFeatureSequence(train_split, label_processor, EXTRACTOR, FRAME_BATCH_VIDEOS, MAX_FRAMES_PER_VIDEO, shuffle=True, random_frame_sample=MAX_FRAMES_PER_VIDEO > 0)
    val_frames, val_labels, _ = collect_frame_arrays(val_split, label_processor, EXTRACTOR, MAX_FRAMES_PER_VIDEO)
    if len(val_labels) == 0:
        raise SystemExit('No validation frames were loaded.')
    class_weight = None
    if SKLEARN_AVAILABLE:
        train_video_labels = np.asarray([encode_label(label_processor, tag) for tag in train_split['tag']], dtype=np.int32)
        weights = compute_class_weight('balanced', classes=np.unique(train_video_labels), y=train_video_labels)
        class_weight = dict(enumerate(weights))
        print(f'Class weights: {class_weight}')
    model = build_frame_classifier(len(label_vocab))
    model.summary(print_fn=lambda line: print(f'  {line}'))
    checkpoint_path = CHECKPOINT_DIR / 'frame_only_baseline.keras'
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=8, restore_best_weights=True, verbose=1), keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', mode='max', save_best_only=True, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-06, verbose=1)]
    start = time.time()
    history = model.fit(train_sequence, validation_data=(val_frames, val_labels), epochs=EPOCHS, class_weight=class_weight, callbacks=callbacks, verbose=1)
    duration = time.time() - start
    print('\nEvaluating frame-level accuracy...')
    test_frames, test_frame_labels, _ = collect_frame_arrays(test_df, label_processor, EXTRACTOR, MAX_FRAMES_PER_VIDEO)
    frame_loss, frame_accuracy = model.evaluate(test_frames, test_frame_labels, batch_size=1024, verbose=0)
    frame_probabilities = model.predict(test_frames, batch_size=1024, verbose=0)
    frame_predictions = np.argmax(frame_probabilities, axis=1)
    metrics = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'experiment': 'frame_only_baseline', 'purpose': 'Test play classification without temporal sequence modeling.', 'extractor': EXTRACTOR, 'train_csv': str(TRAIN_CSV), 'test_csv': str(TEST_CSV), 'epochs_requested': EPOCHS, 'epochs_ran': len(history.history.get('loss', [])), 'frame_batch_videos': FRAME_BATCH_VIDEOS, 'max_frames_per_video': 'all' if MAX_FRAMES_PER_VIDEO <= 0 else MAX_FRAMES_PER_VIDEO, 'dense_units': DENSE_UNITS, 'small_dense_units': SMALL_DENSE_UNITS, 'dropout': DROPOUT, 'learning_rate': LEARNING_RATE, 'duration_s': round(duration, 1), 'frame_test_loss': round(float(frame_loss), 6), 'frame_accuracy': round(float(frame_accuracy), 6), 'checkpoint_path': str(checkpoint_path), **train_stats, **val_stats, **test_stats}
    if SKLEARN_AVAILABLE:
        metrics['frame_f1_macro'] = round(float(f1_score(test_frame_labels, frame_predictions, average='macro', zero_division=0)), 4)
        metrics['frame_f1_weighted'] = round(float(f1_score(test_frame_labels, frame_predictions, average='weighted', zero_division=0)), 4)
        frame_report = classification_report(test_frame_labels, frame_predictions, target_names=label_vocab, zero_division=0)
        (RESULTS_DIR / 'frame_classification_report.txt').write_text(frame_report, encoding='utf-8')
    print('Evaluating play/video accuracy by averaging frame probabilities...')
    video_metrics, _, _, _ = evaluate_by_video(model, test_df, label_processor, label_vocab, RESULTS_DIR)
    metrics.update(video_metrics)
    save_history_plot(history, RESULTS_DIR / 'history.png')
    np.save(CHECKPOINT_DIR / 'label_vocab.npy', np.asarray(label_vocab))
    (CHECKPOINT_DIR / 'metadata.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (RESULTS_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    csv_path = RESULTS_DIR / 'frame_only_results.csv'
    with _resolve_model_artifact(csv_path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print('\nDone.')
    print(f"Frame accuracy: {metrics['frame_accuracy'] * 100:.2f}%")
    print(f"Video accuracy: {metrics['video_accuracy'] * 100:.2f}%")
    if 'video_f1_macro' in metrics:
        print(f"Video F1 macro: {metrics['video_f1_macro']:.4f}")
    print(f'Results: {csv_path}')
if __name__ == '__main__':
    main()
