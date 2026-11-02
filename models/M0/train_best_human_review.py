"""Train a sequence classifier with current partitions and optional manual annotations."""
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
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import keras
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.M0.data_utils import load_video
from models.M0.model import build_feature_extractor_resnet50, build_optimizer
try:
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
for arg in sys.argv[1:]:
    if '=' in arg:
        key, value = arg.split('=', 1)
        os.environ[key.upper()] = value

def _env_bool(key: str, default: bool=False) -> bool:
    raw = os.environ.get(key, str(int(default))).strip().lower()
    return raw in {'1', 'true', 'yes', 'y'}

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
BASELINE_TRAIN_CSV = CURRENT_DATASET / 'train.csv'
BASELINE_TEST_CSV = CURRENT_DATASET / 'test.csv'
_default_review_manifest = Path(os.environ.get('TESINA_WEB_RUNTIME', PROJECT_ROOT / '.runtime/web')) / 'active_learning_dataset/train_manifest.csv'
REVIEW_MANIFEST = Path(os.environ.get('REVIEW_MANIFEST', _default_review_manifest)).expanduser()
if not REVIEW_MANIFEST.is_absolute():
    REVIEW_MANIFEST = (Path.cwd() / REVIEW_MANIFEST).resolve()
EXPERIMENT_NAME = os.environ.get('EXPERIMENT_NAME', 'human_review_retrain').strip() or 'human_review_retrain'
BASELINE_CACHE_DIR = PROJECT_ROOT / '.runtime/cache/features'
EXPERIMENT_CACHE_DIR = PROJECT_ROOT / '.runtime/cache/review_features'
CHECKPOINT_DIR = auxiliary_directory('train_best_human_review') / 'checkpoints'
RESULTS_DIR = auxiliary_directory('train_best_human_review') / 'results'
QUICK = _env_bool('QUICK', False)
EPOCHS = _env_int('EPOCHS', 5 if QUICK else 100)
QUICK_BASELINE_ROWS = _env_int('QUICK_BASELINE_ROWS', 150)
QUICK_REVIEW_ROWS = _env_int('QUICK_REVIEW_ROWS', 40)
QUICK_TEST_ROWS = _env_int('QUICK_TEST_ROWS', 60)
REVIEW_ONLY_CORRECTED = _env_bool('REVIEW_ONLY_CORRECTED', False)
REVIEW_WEIGHT = _env_float('REVIEW_WEIGHT', 1.0)
SEED = _env_int('SEED', 42)
EXPECTED_REVIEW_ROWS = _env_int('EXPECTED_REVIEW_ROWS', 0)
EXPECTED_TEST_ROWS = _env_int('EXPECTED_TEST_ROWS', 817)
MAX_SEQ_LENGTH = 200
NUM_FEATURES = 2048
EXTRACTOR = 'resnet50'
RNN_TYPE = 'gru'
RNN_UNITS = 64
BIDIRECTIONAL = True
DROPOUT = 0.3
DENSE_UNITS = 64
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-05
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
TARGET_FPS = 25.0

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _resolve_model_artifact(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _safe_cache_key(value: str) -> str:
    text = str(value)
    for old, new in [('\\', '_'), ('/', '_'), (':', '_'), (' ', '_')]:
        text = text.replace(old, new)
    return Path(text).stem

def _load_npz(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not _resolve_model_artifact(path).exists():
        return (None, None)
    try:
        data = _load_model_result_array(path)
        frame_features = data['frame_features']
        frame_masks = data['frame_masks']
        if frame_features.shape == (MAX_SEQ_LENGTH, NUM_FEATURES) and frame_masks.shape == (MAX_SEQ_LENGTH,):
            return (frame_features, frame_masks)
    except Exception:
        return (None, None)
    return (None, None)

def _extract_review_features(video_path: Path, cache_path: Path, feature_extractor) -> tuple[np.ndarray, np.ndarray]:
    frames = load_video(str(video_path), target_fps=TARGET_FPS)[:MAX_SEQ_LENGTH]
    frame_features = np.zeros((MAX_SEQ_LENGTH, NUM_FEATURES), dtype=np.float32)
    frame_masks = np.zeros((MAX_SEQ_LENGTH,), dtype=bool)
    if len(frames) > 0:
        extracted = feature_extractor.predict(frames, batch_size=32, verbose=0)
        length = min(len(extracted), MAX_SEQ_LENGTH)
        frame_features[:length] = extracted[:length]
        frame_masks[:length] = True
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, frame_features=frame_features, frame_masks=frame_masks)
    return (frame_features, frame_masks)

def _load_or_extract_features(row: pd.Series, feature_extractor=None) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    source_kind = str(row.get('source_kind', 'baseline'))
    if source_kind.startswith('baseline'):
        features, masks = current_rgb(str(row['video_name']), 'resnet50')
        return (features, masks, 'baseline_cache')
    cache_key = row.get('cache_key') or row.get('dataset_id') or row.get('video_name')
    cache_path = EXPERIMENT_CACHE_DIR / f'{_safe_cache_key(cache_key)}.npz'
    features, masks = _load_npz(cache_path)
    if features is not None:
        return (features, masks, 'review_cache')
    if feature_extractor is None:
        return (None, None, 'missing_review_cache')
    clip_path = Path(str(row.get('clip_path', '')))
    if not _resolve_model_artifact(clip_path).exists():
        return (None, None, 'missing_review_clip')
    features, masks = _extract_review_features(clip_path, cache_path, feature_extractor)
    return (features, masks, 'review_extracted')

def build_model(num_classes: int, optimizer):

    def rnn_layer(units: int, return_sequences: bool):
        layer = keras.layers.GRU(units, return_sequences=return_sequences)
        if BIDIRECTIONAL:
            layer = keras.layers.Bidirectional(layer)
        return layer
    feat_input = keras.Input(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), name='features')
    mask_input = keras.Input(shape=(MAX_SEQ_LENGTH,), dtype='bool', name='mask')
    x = rnn_layer(RNN_UNITS, return_sequences=True)(feat_input, mask=mask_input)
    x = rnn_layer(max(RNN_UNITS // 2, 8), return_sequences=False)(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    x = keras.layers.Dense(DENSE_UNITS, activation='relu')(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model([feat_input, mask_input], output, name='best_model_human_review')
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_experiment_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    baseline_train = pd.read_csv(_resolve_model_artifact(BASELINE_TRAIN_CSV))
    baseline_test = pd.read_csv(_resolve_model_artifact(BASELINE_TEST_CSV))
    baseline_train = baseline_train[['video_name', 'tag']].copy()
    baseline_test = baseline_test[['video_name', 'tag']].copy()
    baseline_train['source_kind'] = 'baseline'
    baseline_test['source_kind'] = 'baseline_test'
    baseline_train['clip_path'] = ''
    baseline_test['clip_path'] = ''
    baseline_train['dataset_id'] = ''
    baseline_test['dataset_id'] = ''
    baseline_train['review_outcome'] = ''
    baseline_test['review_outcome'] = ''
    review = pd.DataFrame()
    if _resolve_model_artifact(REVIEW_MANIFEST).exists():
        review = pd.read_csv(_resolve_model_artifact(REVIEW_MANIFEST))
        if 'dataset_id' in review.columns and review['dataset_id'].astype(str).duplicated().any():
            duplicates = review.loc[review['dataset_id'].astype(str).duplicated(), 'dataset_id'].tolist()
            raise ValueError(f'Review manifest contains duplicate dataset_id values: {duplicates[:5]}')
        if REVIEW_ONLY_CORRECTED and 'prediction_is_correct' in review.columns:
            review = review[~review['prediction_is_correct'].astype(str).str.lower().isin({'true', '1', 'yes'})]
        review_rows = pd.DataFrame({'video_name': review.get('clip_name', review.get('video_name', '')).astype(str), 'tag': review.get('correct_label', review.get('tag', '')).astype(str).str.lower(), 'source_kind': 'human_review', 'clip_path': review.get('clip_path', '').astype(str), 'dataset_id': review.get('dataset_id', '').astype(str), 'review_outcome': review.get('review_outcome', '').astype(str)})
        review_rows['cache_key'] = review_rows['dataset_id'].where(review_rows['dataset_id'].astype(str).str.len() > 0, review_rows['video_name'])
    else:
        review_rows = pd.DataFrame(columns=list(baseline_train.columns) + ['cache_key'])
    if QUICK:
        baseline_train = baseline_train.head(QUICK_BASELINE_ROWS)
        baseline_test = baseline_test.head(QUICK_TEST_ROWS)
        review_rows = review_rows.head(QUICK_REVIEW_ROWS)
    held_out = set(current_frames()[1]['video_name']) | set(baseline_test['video_name'])
    if set(review_rows['video_name']) & held_out:
        raise ValueError('Las anotaciones añadidas contienen clips reservados para validación o test')
    train_df = pd.concat([baseline_train, review_rows], ignore_index=True, sort=False)
    test_df = baseline_test.reset_index(drop=True)
    summary = {'baseline_train_rows': int(len(baseline_train)), 'review_rows': int(len(review_rows)), 'test_rows': int(len(test_df)), 'review_only_corrected': REVIEW_ONLY_CORRECTED, 'review_label_counts': review_rows['tag'].value_counts().to_dict() if len(review_rows) else {}}
    if EXPECTED_REVIEW_ROWS and len(review_rows) != EXPECTED_REVIEW_ROWS:
        raise ValueError(f'Expected {EXPECTED_REVIEW_ROWS} review rows, found {len(review_rows)} in {REVIEW_MANIFEST}.')
    if EXPECTED_TEST_ROWS and len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(f'Expected {EXPECTED_TEST_ROWS} test rows, found {len(test_df)}.')
    return (train_df, test_df, summary)

def build_dataset(df: pd.DataFrame, label_processor, split_name: str, feature_extractor=None):
    features = []
    masks = []
    labels = []
    sample_weights = []
    load_counts = {}
    for _, row in df.iterrows():
        frame_features, frame_masks, load_status = _load_or_extract_features(row, feature_extractor)
        load_counts[load_status] = load_counts.get(load_status, 0) + 1
        if frame_features is None:
            continue
        if not frame_masks.any():
            load_counts[f'{split_name}_empty_features'] = load_counts.get(f'{split_name}_empty_features', 0) + 1
            continue
        tag = str(row['tag']).strip().lower()
        encoded = int(keras.ops.convert_to_numpy(label_processor([tag]))[0])
        features.append(frame_features)
        masks.append(frame_masks)
        labels.append(encoded)
        sample_weights.append(REVIEW_WEIGHT if row.get('source_kind') == 'human_review' else 1.0)
    return (np.asarray(features, dtype=np.float32), np.asarray(masks, dtype=bool), np.asarray(labels, dtype=np.int32), np.asarray(sample_weights, dtype=np.float32), load_counts)

def save_history_plot(history, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, val_metric, title in zip(axes, ['loss', 'accuracy'], ['val_loss', 'val_accuracy'], ['Loss', 'Accuracy']):
        ax.plot(history.history.get(metric, []), label='train')
        ax.plot(history.history.get(val_metric, []), label='val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
    fig.suptitle('Human-review retrain experiment')
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)

def main() -> int:
    keras.utils.set_random_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df, data_summary = load_experiment_rows()
    train_df.to_csv(RESULTS_DIR / 'merged_train_manifest.csv', index=False)
    test_df.to_csv(RESULTS_DIR / 'baseline_test_manifest.csv', index=False)
    unique_tags = sorted(set(train_df['tag'].astype(str).str.strip().str.lower()))
    label_processor = keras.layers.StringLookup(num_oov_indices=0, mask_token=None, vocabulary=unique_tags)
    label_vocab = list(label_processor.get_vocabulary())
    print('=' * 72)
    print('Human-review retrain experiment')
    print('=' * 72)
    print(f'Train rows: {len(train_df)}  Test rows: {len(test_df)}')
    print(f"Reviewed rows included: {data_summary['review_rows']}")
    print(f'Review only corrected: {REVIEW_ONLY_CORRECTED}')
    print(f'Review sample weight: {REVIEW_WEIGHT}')
    print(f'Seed: {SEED}')
    print(f'Labels: {label_vocab}')
    print('\nBuilding feature extractor for missing reviewed-clip caches...')
    from models.common.legacy_training import _rgb_source
    feature_extractor = _rgb_source('resnet50').feature_extractor
    print('\nLoading/extracting train features...')
    X_tr, M_tr, y_tr, w_tr, train_load_counts = build_dataset(train_df, label_processor, 'train', feature_extractor=feature_extractor)
    print(f'Train samples loaded: {len(y_tr)}')
    print(f'Train load counts: {train_load_counts}')
    print('\nLoading test features...')
    X_te, M_te, y_te, _, test_load_counts = build_dataset(test_df, label_processor, 'test', feature_extractor=None)
    print(f'Test samples loaded: {len(y_te)}')
    print(f'Test load counts: {test_load_counts}')
    if len(y_tr) == 0 or len(y_te) == 0:
        raise SystemExit('No train/test samples loaded; cannot train experiment.')
    class_weight = None
    if SKLEARN_OK:
        classes = np.unique(y_tr)
        weights = compute_class_weight('balanced', classes=classes, y=y_tr)
        class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        print(f'Class weights: {class_weight}')
    if class_weight:
        w_tr = np.asarray([float(weight) * float(class_weight.get(int(label), 1.0)) for weight, label in zip(w_tr, y_tr)], dtype=np.float32)
    optimizer = build_optimizer('adam', lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model = build_model(len(label_vocab), optimizer)
    ckpt_path = CHECKPOINT_DIR / 'best_model_human_review.keras'
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True, verbose=1), keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor='val_accuracy', mode='max', save_best_only=True, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-06, verbose=1)]
    start = time.time()
    history = model.fit([X_tr, M_tr], y_tr, validation_data=current_validation(label_vocab, 'resnet50'), epochs=EPOCHS, batch_size=BATCH_SIZE, sample_weight=w_tr, callbacks=callbacks, verbose=1)
    duration = time.time() - start
    test_loss, test_acc = model.evaluate([X_te, M_te], y_te, verbose=0)
    metrics = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'purpose': 'Best-model architecture retrained with human-reviewed clips as an isolated experiment.', 'checkpoint_path': str(ckpt_path), 'results_dir': str(RESULTS_DIR), 'baseline_train_csv': str(BASELINE_TRAIN_CSV), 'baseline_test_csv': str(BASELINE_TEST_CSV), 'review_manifest': str(REVIEW_MANIFEST), 'review_manifest_sha256': _sha256(REVIEW_MANIFEST), 'baseline_train_sha256': _sha256(BASELINE_TRAIN_CSV), 'baseline_test_sha256': _sha256(BASELINE_TEST_CSV), 'review_only_corrected': REVIEW_ONLY_CORRECTED, 'review_sample_weight': REVIEW_WEIGHT, 'seed': SEED, 'determinism_note': 'Python/NumPy/Keras RNGs share the recorded seed. Exact bitwise reproducibility is not guaranteed because backend kernels and oneDNN execution may be nondeterministic.', 'epochs_requested': EPOCHS, 'epochs_ran': len(history.history.get('loss', [])), 'duration_s': round(duration, 1), 'test_loss': round(float(test_loss), 6), 'test_accuracy': round(float(test_acc), 6), 'label_vocab': label_vocab, **data_summary, 'train_samples_loaded': int(len(y_tr)), 'test_samples_loaded': int(len(y_te)), 'train_load_counts': train_load_counts, 'test_load_counts': test_load_counts, 'hyperparameters': {'extractor': EXTRACTOR, 'rnn_type': RNN_TYPE, 'rnn_units': RNN_UNITS, 'bidirectional': BIDIRECTIONAL, 'dropout': DROPOUT, 'dense_units': DENSE_UNITS, 'optimizer': 'adam', 'learning_rate': LEARNING_RATE, 'weight_decay_requested': WEIGHT_DECAY, 'loss': 'sparse_categorical_crossentropy', 'batch_size': BATCH_SIZE, 'validation_csv': str(CURRENT_DATASET / 'validation.csv'), 'shuffle': True, 'max_seq_length': MAX_SEQ_LENGTH, 'num_features': NUM_FEATURES, 'target_fps': TARGET_FPS, 'frame_sampling': 'selection by timestamp via load_video', 'color_conversion': 'BGR to RGB in load_video', 'normalization': 'keras.applications.resnet50.preprocess_input inside extractor', 'manual_divide_by_255': False, 'class_weight': class_weight, 'early_stopping': {'monitor': 'val_accuracy', 'mode': 'max', 'patience': 10}, 'reduce_lr_on_plateau': {'monitor': 'val_loss', 'factor': 0.5, 'patience': 5, 'min_lr': 1e-06}}}
    if SKLEARN_OK:
        y_prob = model.predict([X_te, M_te], verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        metrics['f1_macro'] = round(float(f1_score(y_te, y_pred, average='macro', zero_division=0)), 4)
        metrics['f1_weighted'] = round(float(f1_score(y_te, y_pred, average='weighted', zero_division=0)), 4)
        cm = confusion_matrix(y_te, y_pred)
        row_sum = cm.sum(axis=1)
        recalls = np.where(row_sum > 0, np.diag(cm) / row_sum, 0.0)
        for idx, label in enumerate(label_vocab):
            metrics[f'recall_{label}'] = round(float(recalls[idx]), 4)
        report = classification_report(y_te, y_pred, target_names=label_vocab, zero_division=0)
        (RESULTS_DIR / 'classification_report.txt').write_text(report, encoding='utf-8')
        for normalize, suffix in [(None, 'raw'), ('true', 'normalized')]:
            matrix = confusion_matrix(y_te, y_pred, normalize=normalize)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(matrix, display_labels=label_vocab).plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'Human-review retrain confusion ({suffix})')
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / f'confusion_matrix_{suffix}.png', dpi=100)
            plt.close(fig)
    save_history_plot(history, RESULTS_DIR / 'history.png')
    np.save(CHECKPOINT_DIR / 'label_vocab.npy', np.array(label_vocab, dtype=object))
    with _resolve_model_artifact(CHECKPOINT_DIR / 'metadata.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    with _resolve_model_artifact(RESULTS_DIR / 'metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    with _resolve_model_artifact(RESULTS_DIR / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print('\nDone.')
    print(f'Checkpoint: {ckpt_path}')
    print(f"Metrics:    {RESULTS_DIR / 'metrics.json'}")
    print(f"Accuracy:   {metrics['test_accuracy']}")
    if 'f1_macro' in metrics:
        print(f"F1 macro:   {metrics['f1_macro']}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
