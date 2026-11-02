"""Train recurrent variants with current CSV partitions and a shared RGB cache."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, auxiliary_directory, cached_rgb as current_rgb, current_frames, validation_data as current_validation
import os
import sys
import csv
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from models.M0.config import TRAIN_CSV, TEST_CSV, FEATURE_CACHE_DIR, MAX_SEQ_LENGTH, NUM_FEATURES, LEARNING_RATE, WEIGHT_DECAY, OPTIMIZER_NAME, CLASSES_TO_USE
from models.M0.config import BATCH_SIZE as _CFG_BATCH_SIZE
from models.M0.config import EPOCHS as _CFG_EPOCHS
from models.M0.config import VALIDATION_SPLIT
from models.M0.model import SEQUENCE_MODELS, build_optimizer
try:
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    print('⚠  scikit-learn not found — class weights and detailed metrics disabled.')
    SKLEARN_AVAILABLE = False
ALL_EXTRACTORS = ['resnet50']
ALL_RNN_MODELS = ['gru_small', 'gru_big', 'lstm']
EXTRACTORS = [e.strip() for e in os.environ.get('EXTRACTORS', ','.join(ALL_EXTRACTORS)).split(',') if e.strip()]
RNN_MODELS = [m.strip() for m in os.environ.get('RNN_MODELS', ','.join(ALL_RNN_MODELS)).split(',') if m.strip()]
EPOCHS = int(os.environ.get('EPOCHS', _CFG_EPOCHS))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', _CFG_BATCH_SIZE))
BASE_RESULTS_DIR = str(auxiliary_directory('train_all_extractors') / 'results')
BASE_CHECKPOINTS_DIR = str(auxiliary_directory('train_all_extractors') / 'checkpoints')
COMBINED_CSV_PATH = os.path.join(BASE_RESULTS_DIR, 'all_experiments_results.csv')

def _sanitize(value: str) -> str:
    return str(value).replace('\\', '_').replace('/', '_').replace(':', '_').replace(' ', '_')

def _cache_path(cache_root: str, extractor: str, video_name: str) -> str:
    safe = _sanitize(extractor)
    stem = os.path.splitext(video_name)[0]
    return os.path.join(cache_root, safe, stem + '.npz')

def load_cached_features(video_name: str, extractor: str) -> tuple:
    return current_rgb(video_name, extractor)

def build_dataset(df: pd.DataFrame, label_processor, extractor: str, split_name: str):
    """Build (features, masks, labels) arrays from CSV rows + cache files."""
    features, masks, labels = ([], [], [])
    missing = []
    for _, row in df.iterrows():
        video_name = row['video_name']
        label_raw = row['tag']
        ff, fm = load_cached_features(video_name, extractor)
        if ff is None:
            missing.append(video_name)
            continue
        label_raw_clean = str(label_raw).strip().lower()
        label_enc = int(keras.ops.convert_to_numpy(label_processor([label_raw_clean]))[0])
        if label_enc < 0:
            print(f"  ⚠  OOV tag '{label_raw}' (cleaned: '{label_raw_clean}') for {video_name} — skipped.")
            continue
        features.append(ff)
        masks.append(fm)
        labels.append(label_enc)
    if missing:
        print(f"  ⚠  {len(missing)} {split_name} videos missing from cache for '{extractor}' — skipped.")
        if len(missing) <= 10:
            for v in missing:
                print(f'       {v}')
    return (np.array(features), np.array(masks), np.array(labels))

def save_confusion_matrices(y_true, y_pred, label_vocab, model_name, results_dir):
    for normalize, suffix in [(None, 'raw'), ('true', 'normalized')]:
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_vocab)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap='Blues', colorbar=True)
        title = f'{model_name} ({suffix})'
        ax.set_title(title)
        path = os.path.join(results_dir, f'confusion_matrix_{suffix}_{model_name}.png')
        fig.tight_layout()
        plt.savefig(path)
        plt.close(fig)
        print(f'    Saved {suffix} confusion matrix → {path}')

def save_history_plot(history, model_name, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, val_metric in zip(axes, ['loss', 'accuracy'], ['val_loss', 'val_accuracy']):
        ax.plot(history.history.get(metric, []), label=f'train_{metric}')
        if val_metric in history.history:
            ax.plot(history.history[val_metric], label=val_metric)
        ax.set_title(f'{metric} — {model_name}')
        ax.set_xlabel('Epoch')
        ax.legend()
    path = os.path.join(results_dir, f'history_{model_name}.png')
    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f'    Saved history plot → {path}')

def train_one(extractor: str, rnn_name: str, train_features, train_masks, train_labels, test_features, test_masks, test_labels, label_vocab: list, results_dir: str, checkpoints_dir: str) -> dict:
    """Train a single extractor+RNN combination and return metrics dict."""
    full_name = f'{extractor}__{rnn_name}'
    print(f"\n{'=' * 64}")
    print(f'  Extractor: {extractor}   RNN: {rnn_name}')
    print(f'  Train samples: {len(train_labels)}   Test samples: {len(test_labels)}')
    print(f"{'=' * 64}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    optimizer = build_optimizer(OPTIMIZER_NAME, LEARNING_RATE, WEIGHT_DECAY)
    model = SEQUENCE_MODELS[rnn_name](len(label_vocab), optimizer)
    class_weights_dict = None
    if SKLEARN_AVAILABLE:
        try:
            cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            class_weights_dict = dict(enumerate(cw))
            print('  Class weights:')
            for idx, w in class_weights_dict.items():
                print(f'    [{idx}] {label_vocab[idx]}: {w:.4f}')
        except Exception as e:
            print(f'  ⚠  Could not compute class weights: {e}')
    fit_kwargs = dict(validation_data=current_validation(label_vocab, extractor), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    if class_weights_dict:
        fit_kwargs['class_weight'] = class_weights_dict
    t0 = time.time()
    history = model.fit([train_features, train_masks], train_labels, **fit_kwargs)
    duration = time.time() - t0
    print(f'  Finished in {duration:.1f}s')
    test_loss, test_acc = model.evaluate([test_features, test_masks], test_labels, verbose=0)
    print(f'  Test loss: {test_loss:.4f}   Test accuracy: {test_acc * 100:.2f}%')
    metrics = {'extractor': extractor, 'rnn_model': rnn_name, 'test_loss': round(float(test_loss), 6), 'test_accuracy': round(float(test_acc), 6), 'train_last_acc': round(float(history.history.get('accuracy', [0])[-1]), 6), 'train_last_loss': round(float(history.history.get('loss', [0])[-1]), 6), 'val_last_acc': round(float(history.history.get('val_accuracy', [None])[-1] or 0), 6), 'val_last_loss': round(float(history.history.get('val_loss', [None])[-1] or 0), 6), 'duration_s': round(duration, 1), 'train_samples': len(train_labels), 'test_samples': len(test_labels)}
    if SKLEARN_AVAILABLE:
        y_prob = model.predict([test_features, test_masks], verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = test_labels
        print('\n  Classification Report:')
        report = classification_report(y_true, y_pred, target_names=label_vocab, zero_division=0)
        print(report)
        report_path = os.path.join(results_dir, f'classification_report_{full_name}.txt')
        with open(_resolve_model_artifact(report_path), 'w') as f:
            f.write(report)
        save_confusion_matrices(y_true, y_pred, label_vocab, full_name, results_dir)
        cm = confusion_matrix(y_true, y_pred)
        row_sums = cm.sum(axis=1)
        recalls = np.divide(np.diag(cm).astype(float), row_sums, out=np.zeros(len(label_vocab)), where=row_sums != 0)
        for idx, cls in enumerate(label_vocab):
            metrics[f'recall_{cls}'] = round(float(recalls[idx]), 4)
    save_history_plot(history, full_name, results_dir)
    model_path = os.path.join(checkpoints_dir, f'sequence_model_{full_name}.keras')
    model.save(model_path)
    print(f'  Model saved → {model_path}')
    return metrics

def main():
    print('=' * 64)
    print('  Multi-extractor RNN Training')
    print(f'  Extractors : {EXTRACTORS}')
    print(f'  RNN models : {RNN_MODELS}')
    print(f'  Epochs     : {EPOCHS}   Batch: {BATCH_SIZE}')
    print('=' * 64)
    train_df = pd.read_csv(_resolve_model_artifact(TRAIN_CSV))
    test_df = pd.read_csv(_resolve_model_artifact(TEST_CSV))
    if CLASSES_TO_USE is not None:
        train_df = train_df[train_df['tag'].isin(CLASSES_TO_USE)].copy()
        test_df = test_df[test_df['tag'].isin(CLASSES_TO_USE)].copy()
        print(f'\nFiltered to classes: {CLASSES_TO_USE}')
    print(f'\nTrain rows: {len(train_df)}   Test rows: {len(test_df)}')
    print(f"Class distribution (train): {train_df['tag'].value_counts().to_dict()}")
    unique_tags = sorted(set((str(t).strip().lower() for t in train_df['tag'].unique())))
    label_processor = keras.layers.StringLookup(num_oov_indices=0, mask_token=None, vocabulary=unique_tags)
    label_vocab = list(label_processor.get_vocabulary())
    print(f'\nLabel vocab (0-based, mask_token=None): {label_vocab}')
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    all_metrics = []
    for extractor in EXTRACTORS:
        print(f"\n{'#' * 64}")
        print(f'#  Loading cached features for extractor: {extractor}')
        print(f"{'#' * 64}")
        train_features, train_masks, train_labels = build_dataset(train_df, label_processor, extractor, 'train')
        test_features, test_masks, test_labels = build_dataset(test_df, label_processor, extractor, 'test')
        if len(train_labels) == 0 or len(test_labels) == 0:
            print(f"  ✗  No data loaded for '{extractor}' — skipping all RNN models for this extractor.")
            continue
        print(f'  ✓  Loaded {len(train_labels)} train and {len(test_labels)} test samples.')
        results_dir_ext = os.path.join(BASE_RESULTS_DIR, extractor)
        checkpoints_dir_ext = os.path.join(BASE_CHECKPOINTS_DIR, extractor)
        os.makedirs(results_dir_ext, exist_ok=True)
        os.makedirs(checkpoints_dir_ext, exist_ok=True)
        np.save(os.path.join(checkpoints_dir_ext, 'label_vocab.npy'), label_vocab)
        for rnn_name in RNN_MODELS:
            if rnn_name not in SEQUENCE_MODELS:
                print(f"  ⚠  '{rnn_name}' not in SEQUENCE_MODELS — skipping.")
                continue
            try:
                m = train_one(extractor, rnn_name, train_features, train_masks, train_labels, test_features, test_masks, test_labels, label_vocab, results_dir_ext, checkpoints_dir_ext)
                all_metrics.append(m)
            except Exception as e:
                print(f'  ✗  Training failed for {extractor} + {rnn_name}: {e}')
                import traceback
                traceback.print_exc()
    if all_metrics:
        all_keys = list(all_metrics[0].keys())
        for m in all_metrics:
            for k in m:
                if k not in all_keys:
                    all_keys.append(k)
        with open(_resolve_model_artifact(COMBINED_CSV_PATH), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            for m in all_metrics:
                writer.writerow(m)
        print(f"\n{'=' * 64}")
        print(f'  All experiments done.  Summary → {COMBINED_CSV_PATH}')
        print(f"{'=' * 64}")
        try:
            summary = pd.DataFrame(all_metrics)
            cols = ['extractor', 'rnn_model', 'test_accuracy', 'val_last_acc']
            recall_cols = [c for c in summary.columns if c.startswith('recall_')]
            print('\n  Results summary:')
            print(summary[cols + recall_cols].to_string(index=False))
        except Exception:
            pass
    else:
        print('\n  No experiments completed successfully.')
if __name__ == '__main__':
    main()
