"""Train a one-vs-rest frame classifier from cached ResNet-50 features.

Examples from the repository root:
    python models/M0/studies/resnet_frame_play_detectors/train_binary_frame_resnet.py TARGET=scrum
    python models/M0/studies/resnet_frame_play_detectors/train_binary_frame_resnet.py TARGET=line QUICK=1
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
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
try:
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
REPO_ROOT = experiment_directory(__file__).parents[1]
MODEL_DIR = REPO_ROOT / 'models' / 'M0'
sys.path.insert(0, str(MODEL_DIR))
from config import FEATURE_CACHE_DIR, LEARNING_RATE, MAX_SEQ_LENGTH, NUM_FEATURES, TEST_CSV, TRAIN_CSV, VALIDATION_SPLIT, WEIGHT_DECAY
from config import BATCH_SIZE as CFG_BATCH_SIZE
from config import EPOCHS as CFG_EPOCHS
from model import build_optimizer
for arg in sys.argv[1:]:
    if '=' in arg:
        key, value = arg.split('=', 1)
        os.environ[key.upper()] = value

def env_bool(key: str, default: bool=False) -> bool:
    raw = os.environ.get(key, str(int(default))).strip().lower()
    return raw in {'1', 'true', 'yes', 'y'}

def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))

def env_str(key: str, default: str) -> str:
    return os.environ.get(key, default).strip() or default
BASE_DIR = experiment_directory(__file__)
TARGET = env_str('TARGET', 'scrum').lower()
EXTRACTOR = env_str('EXTRACTOR', 'resnet50').lower()
QUICK = env_bool('QUICK', False)
EPOCHS = env_int('EPOCHS', 3 if QUICK else CFG_EPOCHS)
BATCH_SIZE = env_int('BATCH_SIZE', max(CFG_BATCH_SIZE, 16))
FRAME_BATCH_VIDEOS = env_int('FRAME_BATCH_VIDEOS', BATCH_SIZE)
MAX_FRAMES_PER_VIDEO = env_int('MAX_FRAMES_PER_VIDEO', 0)
QUICK_TRAIN_ROWS = env_int('QUICK_TRAIN_ROWS', 180)
QUICK_TEST_ROWS = env_int('QUICK_TEST_ROWS', 90)
RANDOM_SEED = env_int('SEED', 42)
DENSE_UNITS = env_int('DENSE_UNITS', 256)
SMALL_DENSE_UNITS = env_int('SMALL_DENSE_UNITS', 64)
DROPOUT = env_float('DROPOUT', 0.4)
NEGATIVE_LABEL = f'not_{TARGET}'
LABEL_VOCAB = [NEGATIVE_LABEL, TARGET]
RESULTS_DIR = BASE_DIR / 'results' / TARGET
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / TARGET

def cache_path(video_name: str) -> Path:
    return Path(FEATURE_CACHE_DIR) / EXTRACTOR / f'{Path(video_name).stem}.npz'

def load_cached_video_frames(video_name: str, max_frames_per_video: int=0, random_sample: bool=False):
    path = cache_path(video_name)
    if not _resolve_model_artifact(path).exists():
        return None
    try:
        data = _load_model_result_array(path)
        frame_features = data['frame_features']
        frame_mask = data['frame_masks'].astype(bool)
    except Exception:
        return None
    if frame_features.shape != (MAX_SEQ_LENGTH, NUM_FEATURES):
        return None
    if frame_mask.shape != (MAX_SEQ_LENGTH,):
        return None
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
    df['binary_tag'] = np.where(df['tag'] == TARGET, TARGET, NEGATIVE_LABEL)
    return df

def encode_binary_label(binary_tag: str) -> int:
    return 1 if binary_tag == TARGET else 0

class BinaryFrameSequence(keras.utils.Sequence):

    def __init__(self, df: pd.DataFrame, batch_videos: int, max_frames_per_video: int, shuffle: bool, random_frame_sample: bool) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.batch_videos = batch_videos
        self.max_frames_per_video = max_frames_per_video
        self.shuffle = shuffle
        self.random_frame_sample = random_frame_sample
        self.indices = np.arange(len(self.df))
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_videos))

    def __getitem__(self, idx: int):
        batch_indices = self.indices[idx * self.batch_videos:(idx + 1) * self.batch_videos]
        frame_batches = []
        label_batches = []
        for row_index in batch_indices:
            row = self.df.iloc[int(row_index)]
            frames = load_cached_video_frames(row['video_name'], self.max_frames_per_video, random_sample=self.random_frame_sample)
            if frames is None or len(frames) == 0:
                continue
            label = encode_binary_label(row['binary_tag'])
            frame_batches.append(frames)
            label_batches.append(np.full((len(frames), 1), label, dtype=np.float32))
        if not frame_batches:
            return (np.zeros((0, NUM_FEATURES), dtype=np.float32), np.zeros((0, 1), dtype=np.float32))
        return (np.concatenate(frame_batches, axis=0), np.concatenate(label_batches, axis=0))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

def inspect_split(df: pd.DataFrame, split_name: str) -> dict:
    missing = 0
    empty = 0
    frame_counts = []
    for _, row in df.iterrows():
        frames = load_cached_video_frames(row['video_name'], MAX_FRAMES_PER_VIDEO)
        if frames is None:
            missing += 1
        elif len(frames) == 0:
            empty += 1
        else:
            frame_counts.append(len(frames))
    return {f'{split_name}_videos': int(len(df)), f'{split_name}_target_videos': int((df['binary_tag'] == TARGET).sum()), f'{split_name}_negative_videos': int((df['binary_tag'] != TARGET).sum()), f'{split_name}_missing_cache': int(missing), f'{split_name}_empty_videos': int(empty), f'{split_name}_valid_videos': int(len(frame_counts)), f'{split_name}_frames': int(np.sum(frame_counts)) if frame_counts else 0}

def build_binary_classifier() -> keras.Model:
    inputs = keras.Input(shape=(NUM_FEATURES,), name=f'{EXTRACTOR}_frame_features')
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Dense(DENSE_UNITS, activation='relu')(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    if SMALL_DENSE_UNITS > 0:
        x = keras.layers.Dense(SMALL_DENSE_UNITS, activation='relu')(x)
        x = keras.layers.Dropout(DROPOUT / 2.0)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid', name=f'{TARGET}_probability')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{EXTRACTOR}_{TARGET}_frame_detector')
    optimizer = build_optimizer('adam', LEARNING_RATE, WEIGHT_DECAY)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')])
    return model

def collect_frame_arrays(df: pd.DataFrame, max_frames_per_video: int):
    features = []
    labels = []
    video_names = []
    for _, row in df.iterrows():
        frames = load_cached_video_frames(row['video_name'], max_frames_per_video)
        if frames is None or len(frames) == 0:
            continue
        label = encode_binary_label(row['binary_tag'])
        features.append(frames)
        labels.append(np.full((len(frames), 1), label, dtype=np.float32))
        video_names.extend([row['video_name']] * len(frames))
    if not features:
        return (np.empty((0, NUM_FEATURES), dtype=np.float32), np.empty((0, 1), dtype=np.float32), [])
    return (np.concatenate(features, axis=0), np.concatenate(labels, axis=0), video_names)

def labels_from_probabilities(probabilities: np.ndarray, threshold: float=0.5) -> np.ndarray:
    return (probabilities.reshape(-1) >= threshold).astype(np.int32)

def write_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str, normalize=None) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize=normalize)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=LABEL_VOCAB).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

def evaluate_by_video(model: keras.Model, df: pd.DataFrame) -> dict:
    y_true = []
    y_pred = []
    rows = []
    skipped = 0
    for _, row in df.iterrows():
        frames = load_cached_video_frames(row['video_name'], MAX_FRAMES_PER_VIDEO)
        if frames is None or len(frames) == 0:
            skipped += 1
            continue
        probs = model.predict(frames, batch_size=1024, verbose=0).reshape(-1)
        mean_prob = float(np.mean(probs))
        true_idx = encode_binary_label(row['binary_tag'])
        pred_idx = int(mean_prob >= 0.5)
        y_true.append(true_idx)
        y_pred.append(pred_idx)
        rows.append({'video_name': row['video_name'], 'original_tag': row['tag'], 'true_label': LABEL_VOCAB[true_idx], 'predicted_label': LABEL_VOCAB[pred_idx], f'{TARGET}_probability': round(mean_prob, 6), 'correct': int(true_idx == pred_idx), 'frame_count': int(len(frames))})
    with _resolve_model_artifact(RESULTS_DIR / 'per_video_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        fieldnames = ['video_name', 'original_tag', 'true_label', 'predicted_label', f'{TARGET}_probability', 'correct', 'frame_count']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = np.asarray(y_pred, dtype=np.int32)
    metrics = {'video_samples': int(len(y_true_arr)), 'video_skipped': int(skipped), 'video_accuracy': round(float(np.mean(y_true_arr == y_pred_arr)), 6) if len(y_true_arr) else 0.0}
    if SKLEARN_AVAILABLE and len(y_true_arr):
        metrics.update(binary_metrics(y_true_arr, y_pred_arr, prefix='video'))
        report = classification_report(y_true_arr, y_pred_arr, target_names=LABEL_VOCAB, zero_division=0)
        (RESULTS_DIR / 'video_classification_report.txt').write_text(report, encoding='utf-8')
        write_confusion_matrix(y_true_arr, y_pred_arr, RESULTS_DIR / 'video_confusion_matrix_raw.png', 'Video predictions')
        write_confusion_matrix(y_true_arr, y_pred_arr, RESULTS_DIR / 'video_confusion_matrix_normalized.png', 'Video predictions normalized', normalize='true')
    return metrics

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    return {f'{prefix}_f1_macro': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4), f'{prefix}_f1_weighted': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4), f'{prefix}_target_precision': round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4), f'{prefix}_target_recall': round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4), f'{prefix}_target_f1': round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4)}

def save_history_plot(history) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history.get('loss', []), label='train')
    axes[0].plot(history.history.get('val_loss', []), label='val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history.history.get('accuracy', []), label='train')
    axes[1].plot(history.history.get('val_accuracy', []), label='val')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    fig.suptitle(f'{TARGET} vs {NEGATIVE_LABEL}')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'history.png', dpi=100)
    plt.close(fig)

def main() -> None:
    if TARGET not in {'scrum', 'line'}:
        raise SystemExit("TARGET must be either 'scrum' or 'line'.")
    if not SKLEARN_AVAILABLE:
        raise SystemExit('scikit-learn is required for this experiment.')
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    extractor_dir = Path(FEATURE_CACHE_DIR) / EXTRACTOR
    if not _resolve_model_artifact(extractor_dir).is_dir():
        raise SystemExit(f'Feature cache not found: {extractor_dir}')
    train_df = clean_dataframe(pd.read_csv(_resolve_model_artifact(TRAIN_CSV)))
    test_df = clean_dataframe(pd.read_csv(_resolve_model_artifact(TEST_CSV)))
    if QUICK:
        train_df = train_df.head(QUICK_TRAIN_ROWS).copy()
        test_df = test_df.head(QUICK_TEST_ROWS).copy()
    print('=' * 80)
    print(f'ResNet frame detector: {TARGET} vs {NEGATIVE_LABEL}')
    print(f'Train CSV: {TRAIN_CSV}')
    print(f'Test CSV: {TEST_CSV}')
    print(f'Extractor: {EXTRACTOR}')
    print(f'Epochs: {EPOCHS} | quick: {QUICK}')
    print(f"Max frames per video: {('all' if MAX_FRAMES_PER_VIDEO <= 0 else MAX_FRAMES_PER_VIDEO)}")
    print('=' * 80)
    train_split, val_split = train_test_split(train_df, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=train_df['binary_tag'])
    train_stats = inspect_split(train_split, 'train')
    val_stats = inspect_split(val_split, 'val')
    test_stats = inspect_split(test_df, 'test')
    print(f"Train videos: {train_stats['train_valid_videos']} | frames: {train_stats['train_frames']}")
    print(f"Val videos:   {val_stats['val_valid_videos']} | frames: {val_stats['val_frames']}")
    print(f"Test videos:  {test_stats['test_valid_videos']} | frames: {test_stats['test_frames']}")
    train_sequence = BinaryFrameSequence(train_split, FRAME_BATCH_VIDEOS, MAX_FRAMES_PER_VIDEO, shuffle=True, random_frame_sample=MAX_FRAMES_PER_VIDEO > 0)
    val_frames, val_labels, _ = collect_frame_arrays(val_split, MAX_FRAMES_PER_VIDEO)
    if len(val_labels) == 0:
        raise SystemExit('No validation frames were loaded.')
    train_video_labels = np.asarray([encode_binary_label(tag) for tag in train_split['binary_tag']], dtype=np.int32)
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_video_labels)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print(f'Class weights: {class_weight}')
    model = build_binary_classifier()
    model.summary(print_fn=lambda line: print(f'  {line}'))
    checkpoint_path = CHECKPOINT_DIR / 'model.keras'
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_recall', mode='max', patience=8, restore_best_weights=True, verbose=1), keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor='val_recall', mode='max', save_best_only=True, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-06, verbose=1)]
    start = time.time()
    history = model.fit(train_sequence, validation_data=(val_frames, val_labels), epochs=EPOCHS, class_weight=class_weight, callbacks=callbacks, verbose=1)
    duration = time.time() - start
    test_frames, test_labels, _ = collect_frame_arrays(test_df, MAX_FRAMES_PER_VIDEO)
    if len(test_labels) == 0:
        raise SystemExit('No test frames were loaded.')
    frame_loss, frame_accuracy, frame_precision, frame_recall = model.evaluate(test_frames, test_labels, batch_size=1024, verbose=0)
    frame_probs = model.predict(test_frames, batch_size=1024, verbose=0)
    frame_predictions = labels_from_probabilities(frame_probs)
    frame_true = test_labels.reshape(-1).astype(np.int32)
    metrics = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'experiment': 'resnet_frame_one_vs_rest_detector', 'target': TARGET, 'negative_label': NEGATIVE_LABEL, 'extractor': EXTRACTOR, 'train_csv': str(TRAIN_CSV), 'test_csv': str(TEST_CSV), 'epochs_requested': EPOCHS, 'epochs_ran': int(len(history.history.get('loss', []))), 'frame_batch_videos': FRAME_BATCH_VIDEOS, 'max_frames_per_video': 'all' if MAX_FRAMES_PER_VIDEO <= 0 else MAX_FRAMES_PER_VIDEO, 'dense_units': DENSE_UNITS, 'small_dense_units': SMALL_DENSE_UNITS, 'dropout': DROPOUT, 'learning_rate': LEARNING_RATE, 'duration_s': round(duration, 1), 'frame_test_loss': round(float(frame_loss), 6), 'frame_accuracy': round(float(frame_accuracy), 6), 'frame_target_precision_metric': round(float(frame_precision), 6), 'frame_target_recall_metric': round(float(frame_recall), 6), 'checkpoint_path': str(checkpoint_path), **train_stats, **val_stats, **test_stats}
    metrics.update(binary_metrics(frame_true, frame_predictions, prefix='frame'))
    frame_report = classification_report(frame_true, frame_predictions, target_names=LABEL_VOCAB, zero_division=0)
    (RESULTS_DIR / 'frame_classification_report.txt').write_text(frame_report, encoding='utf-8')
    write_confusion_matrix(frame_true, frame_predictions, RESULTS_DIR / 'frame_confusion_matrix_raw.png', 'Frame predictions')
    write_confusion_matrix(frame_true, frame_predictions, RESULTS_DIR / 'frame_confusion_matrix_normalized.png', 'Frame predictions normalized', normalize='true')
    video_metrics = evaluate_by_video(model, test_df)
    metrics.update(video_metrics)
    save_history_plot(history)
    np.save(CHECKPOINT_DIR / 'label_vocab.npy', np.asarray(LABEL_VOCAB))
    (CHECKPOINT_DIR / 'metadata.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (RESULTS_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    csv_path = RESULTS_DIR / 'summary.csv'
    with _resolve_model_artifact(csv_path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print('\nDone.')
    print(f'Target: {TARGET}')
    print(f"Frame accuracy: {metrics['frame_accuracy'] * 100:.2f}%")
    print(f"Frame target F1: {metrics['frame_target_f1']:.4f}")
    print(f"Video accuracy: {metrics['video_accuracy'] * 100:.2f}%")
    if 'video_target_f1' in metrics:
        print(f"Video target F1: {metrics['video_target_f1']:.4f}")
    print(f'Results: {csv_path}')
if __name__ == '__main__':
    main()
