"""Search recurrent heads on current validation data; test metrics are report-only."""
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
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from models.M0.config import TRAIN_CSV, TEST_CSV, FEATURE_CACHE_DIR, MAX_SEQ_LENGTH, NUM_FEATURES, CLASSES_TO_USE, VALIDATION_SPLIT, WEIGHT_DECAY, OPTIMIZER_NAME
from models.M0.model import build_optimizer
try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    print('⚠  scikit-learn not found — install it for F1 scoring: pip install scikit-learn')
    SKLEARN_AVAILABLE = False

def _env_list(key, default):
    raw = os.environ.get(key, '')
    return [x.strip() for x in raw.split(',') if x.strip()] or default

def _env_int(key, default):
    return int(os.environ.get(key, default))

def _env_bool(key, default=False):
    return os.environ.get(key, str(int(default))).strip() in ('1', 'true', 'yes')
for arg in sys.argv[1:]:
    if '=' in arg:
        k, v = arg.split('=', 1)
        os.environ[k.upper()] = v
ALL_EXTRACTORS = ['resnet50']
ALL_RNN_TYPES = ['gru', 'lstm']
EXTRACTORS = _env_list('EXTRACTORS', ALL_EXTRACTORS)
RNN_TYPES = _env_list('RNN_TYPES', ALL_RNN_TYPES)
EPOCHS = _env_int('EPOCHS', 30)
RESUME = _env_bool('RESUME')
QUICK = _env_bool('QUICK')
START_TRIAL = _env_int('START_TRIAL', 1)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_RESULTS_DIR = str(auxiliary_directory('grid_search') / 'results')
BASE_CKPT_DIR = str(auxiliary_directory('grid_search') / 'checkpoints')
RESULTS_CSV = os.path.join(BASE_RESULTS_DIR, 'grid_search_results.csv')
BEST_TXT = os.path.join(BASE_RESULTS_DIR, 'grid_search_best.txt')
os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
os.makedirs(BASE_CKPT_DIR, exist_ok=True)

if QUICK:
    PARAM_GRID = {'rnn_units': [32], 'bidirectional': [True], 'dropout': [0.4], 'dense_units': [64], 'learning_rate': [0.0001], 'batch_size': [16]}
else:
    PARAM_GRID = {'rnn_units': [32, 64, 128], 'bidirectional': [False, True], 'dropout': [0.3, 0.5], 'dense_units': [0, 64], 'learning_rate': [0.001, 0.0001], 'batch_size': [16, 32]}

def _sanitize(value):
    return str(value).replace('\\', '_').replace('/', '_').replace(':', '_').replace(' ', '_')

def _cache_path(extractor, video_name):
    return os.path.join(FEATURE_CACHE_DIR, _sanitize(extractor), os.path.splitext(video_name)[0] + '.npz')

def load_cached_features(video_name, extractor):
    return current_rgb(video_name, extractor)

def build_dataset(df, label_processor, extractor, split_name):
    features, masks, labels = ([], [], [])
    missing = []
    for _, row in df.iterrows():
        video_name = row['video_name']
        ff, fm = load_cached_features(video_name, extractor)
        if ff is None:
            missing.append(video_name)
            continue
        tag = str(row['tag']).strip().lower()
        enc = int(keras.ops.convert_to_numpy(label_processor([tag]))[0])
        if enc < 0:
            continue
        try:
            features.append(ff)
            masks.append(fm)
            labels.append(enc)
        except MemoryError:
            print(f"\n  ✗  OOM while accumulating {split_name} features for '{extractor}' — aborting this extractor.")
            return (None, None, None)
    if missing:
        print(f"  ⚠  {len(missing)} {split_name} cache misses for '{extractor}'")
    try:
        return (np.array(features), np.array(masks), np.array(labels))
    except MemoryError:
        print(f"\n  ✗  OOM converting {split_name} list→array for '{extractor}' ({len(features)} samples).")
        print(f'     Array would be shape ({len(features)}, {MAX_SEQ_LENGTH}, {NUM_FEATURES}) float32.')
        print(f'     Try: EXTRACTORS=inceptionv3,resnet50  to skip efficientnetb0, or reduce MAX_SEQ_LENGTH.')
        return (None, None, None)

def build_rnn_model(rnn_type, num_classes, optimizer, rnn_units=64, bidirectional=True, dropout=0.4, dense_units=64):
    """
    Build a GRU or LSTM sequence classifier with configurable hyperparameters.

    Architecture:
        Input (MAX_SEQ_LENGTH, NUM_FEATURES) + mask
        → RNN layer 1 (return_sequences=True)  [optionally Bidirectional]
        → RNN layer 2                           [optionally Bidirectional]
        → Dropout
        → Dense(dense_units, relu)              [skipped if dense_units=0]
        → Dense(num_classes, softmax)
    """
    rnn_cls = keras.layers.GRU if rnn_type == 'gru' else keras.layers.LSTM

    def rnn_layer(units, return_seq):
        layer = rnn_cls(units, return_sequences=return_seq)
        return keras.layers.Bidirectional(layer) if bidirectional else layer
    feat_in = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES), name='features')
    mask_in = keras.Input((MAX_SEQ_LENGTH,), dtype='bool', name='mask')
    x = rnn_layer(rnn_units, return_seq=True)(feat_in, mask=mask_in)
    x = rnn_layer(max(rnn_units // 2, 8), return_seq=False)(x)
    x = keras.layers.Dropout(dropout)(x)
    if dense_units > 0:
        x = keras.layers.Dense(dense_units, activation='relu')(x)
    out = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model([feat_in, mask_in], out, name=f"{rnn_type}_u{rnn_units}_{('bi' if bidirectional else 'uni')}")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def make_trial_name(extractor, rnn_type, params):
    bi = 'bi' if params['bidirectional'] else 'uni'
    lr = f"lr{params['learning_rate']:.0e}".replace('-0', '-')
    return f"{extractor}__{rnn_type}__u{params['rnn_units']}__{bi}__drop{int(params['dropout'] * 10)}__d{params['dense_units']}__{lr}__bs{params['batch_size']}"

def run_trial(extractor, rnn_type, params, train_features, train_masks, train_labels, test_features, test_masks, test_labels, label_vocab):
    """Run one hyperparameter trial. Returns a metrics dict."""
    trial_name = make_trial_name(extractor, rnn_type, params)
    ckpt_dir = os.path.join(BASE_CKPT_DIR, trial_name)
    ckpt_path = os.path.join(ckpt_dir, 'model.keras')
    results_dir = os.path.join(BASE_RESULTS_DIR, trial_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    if RESUME and os.path.exists(_resolve_model_artifact(ckpt_path)):
        print(f'  ↷  Skipping (checkpoint exists): {trial_name}')
        return None
    print(f"\n{'─' * 64}")
    print(f'  Trial: {trial_name}')
    print(f'  Params: {params}')
    print(f"{'─' * 64}")
    optimizer = build_optimizer(OPTIMIZER_NAME, lr=params['learning_rate'], weight_decay=WEIGHT_DECAY)
    model = build_rnn_model(rnn_type=rnn_type, num_classes=len(label_vocab), optimizer=optimizer, rnn_units=params['rnn_units'], bidirectional=params['bidirectional'], dropout=params['dropout'], dense_units=params['dense_units'])
    fit_kwargs = dict(validation_data=current_validation(label_vocab, extractor), epochs=EPOCHS, batch_size=params['batch_size'], verbose=0)
    if SKLEARN_AVAILABLE:
        try:
            cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
            fit_kwargs['class_weight'] = dict(enumerate(cw))
        except Exception:
            pass
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0)]
    t0 = time.time()
    history = model.fit([train_features, train_masks], train_labels, callbacks=callbacks, **fit_kwargs)
    duration = time.time() - t0
    best_epoch = int(np.argmin(history.history.get('val_loss', [0]))) + 1
    val_acc_best = float(max(history.history.get('val_accuracy', [0])))
    val_loss_best = float(min(history.history.get('val_loss', [9000000000.0])))
    test_loss, test_acc = model.evaluate([test_features, test_masks], test_labels, verbose=0)
    metrics = {'trial_name': trial_name, 'extractor': extractor, 'rnn_type': rnn_type, **{f'p_{k}': v for k, v in params.items()}, 'epochs_ran': len(history.history['loss']), 'best_epoch': best_epoch, 'val_acc_best': round(val_acc_best, 6), 'val_loss_best': round(val_loss_best, 6), 'test_loss': round(float(test_loss), 6), 'test_accuracy': round(float(test_acc), 6), 'duration_s': round(duration, 1)}
    if SKLEARN_AVAILABLE:
        y_prob = model.predict([test_features, test_masks], verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = test_labels
        f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        validation_x, validation_y = current_validation(label_vocab, extractor)
        validation_predictions = np.argmax(model.predict(validation_x, verbose=0), axis=1)
        metrics['val_f1_macro'] = round(float(f1_score(validation_y, validation_predictions, average='macro', zero_division=0)), 4)
        metrics['val_f1_weighted'] = round(float(f1_score(validation_y, validation_predictions, average='weighted', zero_division=0)), 4)
        cm = confusion_matrix(y_true, y_pred)
        row_sums = cm.sum(axis=1)
        recalls = np.divide(np.diag(cm).astype(float), row_sums, out=np.zeros(len(label_vocab)), where=row_sums != 0)
        for i, cls in enumerate(label_vocab):
            metrics[f'recall_{cls}'] = round(float(recalls[i]), 4)
        report = classification_report(y_true, y_pred, target_names=label_vocab, zero_division=0)
        with open(_resolve_model_artifact(os.path.join(results_dir, 'classification_report.txt')), 'w') as f:
            f.write(f'Trial: {trial_name}\n\n{report}')
        print(f'\n  Classification report:\n{report}')
    else:
        metrics['val_f1_macro'] = None
        metrics['val_f1_weighted'] = None
    if SKLEARN_AVAILABLE:
        try:
            from sklearn.metrics import ConfusionMatrixDisplay
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_vocab)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp.plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'Confusion (raw) — {trial_name[:35]}')
            fig.tight_layout()
            plt.savefig(os.path.join(results_dir, 'confusion_matrix_raw.png'), dpi=90)
            plt.close(fig)
            cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
            disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=label_vocab)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp2.plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'Confusion (norm.) — {trial_name[:35]}')
            fig.tight_layout()
            plt.savefig(os.path.join(results_dir, 'confusion_matrix_normalized.png'), dpi=90)
            plt.close(fig)
        except Exception as e:
            print(f'  Could not save confusion matrix PNGs: {e}')
    import json as _json
    history_data = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(_resolve_model_artifact(os.path.join(results_dir, 'history.json')), 'w') as f:
        _json.dump({'trial_name': trial_name, 'params': params, 'extractor': extractor, 'rnn_type': rnn_type, 'history': history_data}, f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m, vm in zip(axes, ['loss', 'accuracy'], ['val_loss', 'val_accuracy']):
        ax.plot(history.history.get(m, []), label=f'train_{m}')
        ax.plot(history.history.get(vm, []), label=vm)
        ax.set_title(f'{m} — {trial_name[:40]}')
        ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, 'history.png'))
    plt.close(fig)
    model.save(ckpt_path)
    print(f"  ✓  val_acc={val_acc_best:.4f}  test_acc={test_acc:.4f}  f1_macro={metrics.get('val_f1_macro', 'n/a')}  ({duration:.0f}s)")
    return metrics

def append_to_csv(row: dict, path: str):
    """Append one result row to the CSV (creates header on first write)."""
    write_header = not os.path.exists(_resolve_model_artifact(path))
    with open(_resolve_model_artifact(path), 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def main():
    param_keys = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    all_combos = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]
    total_trials = len(EXTRACTORS) * len(RNN_TYPES) * len(all_combos)
    print('=' * 64)
    print('  Grid Search')
    print(f'  Extractors : {EXTRACTORS}')
    print(f'  RNN types  : {RNN_TYPES}')
    print(f'  Param combos per extractor×RNN : {len(all_combos)}')
    print(f'  Total trials : {total_trials}')
    print(f'  Epochs per trial : {EPOCHS}  (early stop patience=7)')
    print(f'  Resume mode : {RESUME}')
    print(f'  Quick mode  : {QUICK}')
    print(f'  Start trial : {START_TRIAL}')
    print('=' * 64)
    completed_names = set()
    if RESUME and os.path.exists(_resolve_model_artifact(RESULTS_CSV)):
        try:
            done_df = pd.read_csv(_resolve_model_artifact(RESULTS_CSV))
            if 'trial_name' in done_df.columns:
                completed_names = set(done_df['trial_name'].dropna().tolist())
            print(f'  Resume: {len(completed_names)} trials already in CSV — will skip them.')
        except Exception as e:
            print(f'  ⚠  Could not read results CSV for resume: {e}')
    train_df = pd.read_csv(_resolve_model_artifact(TRAIN_CSV))
    test_df = pd.read_csv(_resolve_model_artifact(TEST_CSV))
    if CLASSES_TO_USE is not None:
        train_df = train_df[train_df['tag'].isin(CLASSES_TO_USE)].copy()
        test_df = test_df[test_df['tag'].isin(CLASSES_TO_USE)].copy()
    print(f'\nTrain: {len(train_df)} rows  |  Test: {len(test_df)} rows')
    print(f"Train distribution: {train_df['tag'].value_counts().to_dict()}")
    unique_tags = sorted(set((str(t).strip().lower() for t in train_df['tag'].unique())))
    label_processor = keras.layers.StringLookup(num_oov_indices=0, mask_token=None, vocabulary=unique_tags)
    label_vocab = list(label_processor.get_vocabulary())
    print(f'Label vocab: {label_vocab}')
    all_results = []
    trial_num = 0
    for extractor in EXTRACTORS:
        print(f"\n{'#' * 64}")
        print(f'#  Loading cache: {extractor}')
        print(f"{'#' * 64}")
        if RESUME and completed_names:
            extractor_trial_names = set()
            for rnn_type in RNN_TYPES:
                for params in all_combos:
                    extractor_trial_names.add(make_trial_name(extractor, rnn_type, params))
            if extractor_trial_names.issubset(completed_names):
                print(f"  ✓  All trials for '{extractor}' already completed — skipping cache load.")
                trial_num += len(RNN_TYPES) * len(all_combos)
                continue
        try:
            train_features, train_masks, train_labels = build_dataset(train_df, label_processor, extractor, 'train')
            test_features, test_masks, test_labels = build_dataset(test_df, label_processor, extractor, 'test')
        except MemoryError:
            print(f"\n  ✗  OOM loading cache for '{extractor}' — skipping entire extractor.")
            print(f'     Run separately with: EXTRACTORS={extractor}')
            trial_num += len(RNN_TYPES) * len(all_combos)
            continue
        if train_features is None or len(train_labels) == 0:
            print(f"  ✗  No training data found for '{extractor}' — skipping.")
            trial_num += len(RNN_TYPES) * len(all_combos)
            continue
        print(f'  ✓  {len(train_labels)} train  /  {len(test_labels)} test samples loaded.')
        for rnn_type in RNN_TYPES:
            for params in all_combos:
                trial_num += 1
                if trial_num < START_TRIAL:
                    continue
                name = make_trial_name(extractor, rnn_type, params)
                if RESUME and name in completed_names:
                    print(f'[Trial {trial_num}/{total_trials}]  ⏭  {name}  (already in CSV)')
                    continue
                print(f'\n[Trial {trial_num}/{total_trials}]', end='  ')
                result = run_trial(extractor, rnn_type, params, train_features, train_masks, train_labels, test_features, test_masks, test_labels, label_vocab)
                if result is None:
                    continue
                all_results.append(result)
                append_to_csv(result, RESULTS_CSV)
    if not all_results:
        print('\nNo new trials ran (all skipped or no data).')
        return
    results_df = pd.DataFrame(all_results)
    sort_col = 'val_f1_macro' if 'val_f1_macro' in results_df.columns and results_df['val_f1_macro'].notna().any() else 'val_acc_best'
    results_df_sorted = results_df.sort_values(sort_col, ascending=False)
    print(f"\n{'=' * 64}")
    print(f'  Grid search complete — {len(all_results)} trials run')
    print(f'  Results saved to: {RESULTS_CSV}')
    print(f"{'=' * 64}")
    top5 = results_df_sorted.head(5)
    display_cols = ['trial_name', 'test_accuracy', 'val_f1_macro'] + [c for c in results_df.columns if c.startswith('recall_')]
    print(f'\nTop-5 by {sort_col}:')
    print(top5[[c for c in display_cols if c in top5.columns]].to_string(index=False))
    best = results_df_sorted.iloc[0]
    summary = [f"Best trial: {best['trial_name']}", f"  extractor    : {best['extractor']}", f"  rnn_type     : {best['rnn_type']}", f"  rnn_units    : {best.get('p_rnn_units')}", f"  bidirectional: {best.get('p_bidirectional')}", f"  dropout      : {best.get('p_dropout')}", f"  dense_units  : {best.get('p_dense_units')}", f"  learning_rate: {best.get('p_learning_rate')}", f"  batch_size   : {best.get('p_batch_size')}", f"  val_f1_macro : {best.get('val_f1_macro')}", f"  test_accuracy: {best.get('test_accuracy')}", f"  recall_line  : {best.get('recall_line')}", f"  recall_scrum : {best.get('recall_scrum')}", f"  recall_other : {best.get('recall_other')}"]
    summary_text = '\n'.join(summary)
    print('\n' + summary_text)
    with open(_resolve_model_artifact(BEST_TXT), 'w', encoding='utf-8') as f:
        f.write(summary_text + '\n')
    print(f'\nBest config written to: {BEST_TXT}')
    try:
        fig, ax = plt.subplots(figsize=(max(10, len(results_df_sorted) * 0.4), 5))
        colors = ['#2196F3' if i == 0 else '#90CAF9' for i in range(len(results_df_sorted))]
        ax.bar(range(len(results_df_sorted)), results_df_sorted['test_accuracy'].values, color=colors)
        ax.set_xticks(range(len(results_df_sorted)))
        ax.set_xticklabels(results_df_sorted['trial_name'].values, rotation=90, fontsize=6)
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Grid Search — Test Accuracy per Trial (sorted)')
        fig.tight_layout()
        chart_path = os.path.join(BASE_DIR, 'results', 'grid_search_summary.png')
        plt.savefig(chart_path, dpi=100)
        plt.close(fig)
        print(f'Summary chart saved to: {chart_path}')
    except Exception as e:
        print(f'Could not save summary chart: {e}')
if __name__ == '__main__':
    main()
