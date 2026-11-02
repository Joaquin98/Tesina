"""Refit a selected current grid-search configuration. Set GRID_RESULTS_CSV to its results CSV."""
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
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from models.M0.model import build_optimizer
try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_OK = True
except ImportError:
    print('⚠ scikit-learn not found — install with: pip install scikit-learn')
    SKLEARN_OK = False
for arg in sys.argv[1:]:
    if '=' in arg:
        k, v = arg.split('=', 1)
        os.environ[k.upper()] = v

def _env_int(key, default):
    return int(os.environ.get(key, default))

def _env_bool(key, default=False):
    return os.environ.get(key, str(int(default))).strip().lower() in ('1', 'true', 'yes')

def _env_list(key, default):
    raw = os.environ.get(key, '')
    vals = [x.strip() for x in raw.split(',') if x.strip()]
    return vals or default

def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(int(value))
    s = str(value).strip().lower()
    return s in ('1', 'true', 'yes')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_grid_runs = sorted((_delivery_root / 'models/M0/runs').glob('auxiliar_*/grid_search/results/grid_search_results.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
GRID_RESULTS_CSV = os.environ.get('GRID_RESULTS_CSV', str(_grid_runs[0] if _grid_runs else auxiliary_directory('grid_search') / 'results' / 'grid_search_results.csv'))
TRAIN_CSV = str(CURRENT_DATASET / 'train.csv')
TEST_CSV = str(CURRENT_DATASET / 'test.csv')
FEATURE_CACHE_BASE = str(_delivery_root / '.runtime/cache/features')
CHECKPOINT_DIR = str(auxiliary_directory('train_best_focused') / 'checkpoints')
RESULTS_DIR = str(auxiliary_directory('train_best_focused') / 'results')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
MAX_SEQ_LENGTH = 200
NUM_FEATURES = 2048
EXTRACTORS = _env_list('EXTRACTORS', ['resnet50'])
RNN_TYPES = _env_list('RNN_TYPES', ['gru', 'lstm'])
SELECT_BY = os.environ.get('SELECT_BY', 'val_f1_macro')
REQUIRE_CHECKPOINT = _env_bool('REQUIRE_CHECKPOINT', True)
QUICK = _env_bool('QUICK', False)
EPOCHS = _env_int('EPOCHS', 5 if QUICK else 100)
VALIDATION_SPLIT = float(os.environ.get('VALIDATION_SPLIT', '0.2'))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', '1e-5'))
FOCUSED_GRID = {'p_rnn_units': [64, 128], 'p_bidirectional': [True], 'p_dropout': [0.2, 0.3, 0.4, 0.5], 'p_dense_units': [0, 64], 'p_learning_rate': [0.0001, 0.0003, 0.001], 'p_batch_size': [16, 24, 32]}

def _float_close(x, y, tol=1e-12):
    return abs(float(x) - float(y)) <= tol

def _in_float_list(x, values):
    return any((_float_close(x, v) for v in values))

def load_and_select_best_trial():
    if not os.path.exists(_resolve_model_artifact(GRID_RESULTS_CSV)):
        raise FileNotFoundError(f'grid search results not found: {GRID_RESULTS_CSV}')
    df = pd.read_csv(_resolve_model_artifact(GRID_RESULTS_CSV))
    if df.empty:
        raise RuntimeError('grid_search_results.csv is empty')
    required_cols = ['trial_name', 'extractor', 'rnn_type', 'p_rnn_units', 'p_bidirectional', 'p_dropout', 'p_dense_units', 'p_learning_rate', 'p_batch_size', 'val_f1_macro', 'test_accuracy']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing required columns in CSV: {missing}')
    df['p_rnn_units'] = pd.to_numeric(df['p_rnn_units'], errors='coerce')
    df['p_dense_units'] = pd.to_numeric(df['p_dense_units'], errors='coerce')
    df['p_batch_size'] = pd.to_numeric(df['p_batch_size'], errors='coerce')
    df['p_dropout'] = pd.to_numeric(df['p_dropout'], errors='coerce')
    df['p_learning_rate'] = pd.to_numeric(df['p_learning_rate'], errors='coerce')
    df['val_f1_macro'] = pd.to_numeric(df['val_f1_macro'], errors='coerce')
    df['test_accuracy'] = pd.to_numeric(df['test_accuracy'], errors='coerce')

    def row_in_focused_space(row):
        if row['extractor'] not in EXTRACTORS:
            return False
        if row['rnn_type'] not in RNN_TYPES:
            return False
        if int(row['p_rnn_units']) not in FOCUSED_GRID['p_rnn_units']:
            return False
        if _to_bool(row['p_bidirectional']) not in FOCUSED_GRID['p_bidirectional']:
            return False
        if not _in_float_list(row['p_dropout'], FOCUSED_GRID['p_dropout']):
            return False
        if int(row['p_dense_units']) not in FOCUSED_GRID['p_dense_units']:
            return False
        if not _in_float_list(row['p_learning_rate'], FOCUSED_GRID['p_learning_rate']):
            return False
        if int(row['p_batch_size']) not in FOCUSED_GRID['p_batch_size']:
            return False
        return True
    focused = df[df.apply(row_in_focused_space, axis=1)].copy()
    if focused.empty:
        raise RuntimeError('No rows match focused search space in grid_search_results.csv')
    is_new_signature = focused['p_batch_size'].eq(24) | focused['p_learning_rate'].apply(lambda x: _float_close(x, 0.0003)) | focused['p_dropout'].apply(lambda x: _float_close(x, 0.2) or _float_close(x, 0.4))
    focused_new = focused[is_new_signature].copy()
    candidates = focused_new if not focused_new.empty else focused
    if REQUIRE_CHECKPOINT:

        def has_ckpt(trial_name):
            p = str(_DeliveryPath(GRID_RESULTS_CSV).resolve().parent.parent / 'checkpoints' / trial_name / 'model.keras')
            return os.path.exists(_resolve_model_artifact(p))
        candidates = candidates[candidates['trial_name'].apply(has_ckpt)].copy()
    if candidates.empty:
        raise RuntimeError('No candidate focused trials with checkpoints found. Try REQUIRE_CHECKPOINT=0 or run/finish grid_search_focused first.')
    if SELECT_BY not in candidates.columns:
        print(f"⚠ SELECT_BY='{SELECT_BY}' not found; falling back to val_f1_macro")
        sort_col = 'val_f1_macro'
    else:
        sort_col = SELECT_BY
    candidates = candidates.sort_values(by=[sort_col, 'trial_name'], ascending=[False, True], na_position='last')
    top_preview = candidates.head(5)[['trial_name', sort_col, 'test_accuracy', 'val_f1_macro']]
    print('\nTop focused candidates:')
    print(top_preview.to_string(index=False))
    best = candidates.iloc[0].to_dict()
    return best

def load_features(cache_dir, video_name):
    return current_rgb(video_name, _DeliveryPath(cache_dir).name)

def build_dataset(df, label_processor, cache_dir, split_name):
    features, masks, labels = ([], [], [])
    misses = 0
    for _, row in df.iterrows():
        ff, fm = load_features(cache_dir, row['video_name'])
        if ff is None:
            misses += 1
            continue
        tag = str(row['tag']).strip().lower()
        enc = int(keras.ops.convert_to_numpy(label_processor([tag]))[0])
        if enc < 0:
            continue
        features.append(ff)
        masks.append(fm)
        labels.append(enc)
    if misses:
        print(f'⚠ [{split_name}] cache misses: {misses}')
    return (np.array(features), np.array(masks), np.array(labels, dtype=np.int32))

def build_model(num_classes, rnn_type, rnn_units, bidirectional, dropout, dense_units, learning_rate):
    rnn_cls = keras.layers.GRU if rnn_type == 'gru' else keras.layers.LSTM

    def rnn_layer(units, return_seq):
        layer = rnn_cls(units, return_sequences=return_seq)
        return keras.layers.Bidirectional(layer) if bidirectional else layer
    feat_input = keras.Input(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), name='features')
    mask_input = keras.Input(shape=(MAX_SEQ_LENGTH,), dtype='bool', name='mask')
    x = rnn_layer(rnn_units, return_seq=True)(feat_input, mask=mask_input)
    x = rnn_layer(max(rnn_units // 2, 8), return_seq=False)(x)
    x = keras.layers.Dropout(dropout)(x)
    if dense_units > 0:
        x = keras.layers.Dense(dense_units, activation='relu')(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=[feat_input, mask_input], outputs=x, name='best_model_focused')
    optimizer = build_optimizer('adam', lr=learning_rate, weight_decay=WEIGHT_DECAY)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def save_results_json(path, payload):
    with open(_resolve_model_artifact(path), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def main():
    print('=' * 72)
    print(' Train NEW best model from focused grid search')
    print('=' * 72)
    best = load_and_select_best_trial()
    trial_name = str(best['trial_name'])
    extractor = str(best['extractor'])
    rnn_type = str(best['rnn_type'])
    rnn_units = int(best['p_rnn_units'])
    bidirectional = _to_bool(best['p_bidirectional'])
    dropout = float(best['p_dropout'])
    dense_units = int(best['p_dense_units'])
    learning_rate = float(best['p_learning_rate'])
    batch_size = int(best['p_batch_size'])
    print(f'\nSelected focused-best trial: {trial_name}')
    print(f' extractor={extractor} rnn={rnn_type} units={rnn_units} bi={bidirectional}')
    print(f' dropout={dropout} dense={dense_units} lr={learning_rate} bs={batch_size}')
    print(f" val_f1_macro={best.get('val_f1_macro')} test_acc={best.get('test_accuracy')}")
    cache_dir = os.path.join(FEATURE_CACHE_BASE, extractor)
    train_df = pd.read_csv(_resolve_model_artifact(TRAIN_CSV))
    test_df = pd.read_csv(_resolve_model_artifact(TEST_CSV))
    if QUICK:
        train_df = train_df.head(150)
        test_df = test_df.head(60)
    unique_tags = sorted(set((str(t).strip().lower() for t in train_df['tag'].unique())))
    label_processor = keras.layers.StringLookup(num_oov_indices=0, mask_token=None, vocabulary=unique_tags)
    label_vocab = list(label_processor.get_vocabulary())
    print(f'\nLoading features from: {cache_dir}')
    x_tr, m_tr, y_tr = build_dataset(train_df, label_processor, cache_dir, 'train')
    x_te, m_te, y_te = build_dataset(test_df, label_processor, cache_dir, 'test')
    print(f' train samples={len(y_tr)} | test samples={len(y_te)}')
    if len(y_tr) == 0:
        raise RuntimeError('No training samples loaded from cache')
    class_weight = None
    if SKLEARN_OK:
        try:
            cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
            class_weight = dict(enumerate(cw))
        except Exception as e:
            print(f'⚠ class-weight compute error: {e}')
    model = build_model(num_classes=len(label_vocab), rnn_type=rnn_type, rnn_units=rnn_units, bidirectional=bidirectional, dropout=dropout, dense_units=dense_units, learning_rate=learning_rate)
    model_name = f'best_focused__{trial_name}.keras'
    model_path = os.path.join(CHECKPOINT_DIR, model_name)
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True, verbose=1), keras.callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-06, verbose=1)]
    print(f'\nTraining up to {EPOCHS} epochs...')
    t0 = time.time()
    history = model.fit([x_tr, m_tr], y_tr, validation_data=current_validation(label_vocab, extractor), epochs=EPOCHS, batch_size=batch_size, class_weight=class_weight, callbacks=callbacks, verbose=1)
    duration = time.time() - t0
    test_loss, test_acc = model.evaluate([x_te, m_te], y_te, verbose=0)
    result = {'source_trial': trial_name, 'extractor': extractor, 'rnn_type': rnn_type, 'rnn_units': rnn_units, 'bidirectional': bidirectional, 'dropout': dropout, 'dense_units': dense_units, 'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs_ran': len(history.history.get('loss', [])), 'duration_s': round(duration, 1), 'test_loss': round(float(test_loss), 6), 'test_accuracy': round(float(test_acc), 6), 'saved_model': model_path}
    if SKLEARN_OK and len(y_te) > 0:
        y_prob = model.predict([x_te, m_te], verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        result['f1_macro'] = round(float(f1_score(y_te, y_pred, average='macro', zero_division=0)), 4)
        result['f1_weighted'] = round(float(f1_score(y_te, y_pred, average='weighted', zero_division=0)), 4)
        cm = confusion_matrix(y_te, y_pred)
        row_sum = cm.sum(axis=1)
        recalls = np.where(row_sum > 0, np.diag(cm) / row_sum, 0.0)
        for i, cls in enumerate(label_vocab):
            result[f'recall_{cls}'] = round(float(recalls[i]), 4)
        report = classification_report(y_te, y_pred, target_names=label_vocab, zero_division=0)
        with open(_resolve_model_artifact(os.path.join(RESULTS_DIR, 'classification_report_focused.txt')), 'w', encoding='utf-8') as f:
            f.write(report)
        for norm, suffix in [(None, 'raw'), ('true', 'normalized')]:
            cm_plot = confusion_matrix(y_te, y_pred, normalize=norm)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm_plot, display_labels=label_vocab).plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'Best Focused Model — Confusion ({suffix})')
            fig.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_focused_{suffix}.png'), dpi=100)
            plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m, vm, title in zip(axes, ['loss', 'accuracy'], ['val_loss', 'val_accuracy'], ['Loss', 'Accuracy']):
        ax.plot(history.history.get(m, []), label='train')
        ax.plot(history.history.get(vm, []), label='val')
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'history_focused.png'), dpi=100)
    plt.close(fig)
    save_results_json(os.path.join(RESULTS_DIR, 'best_focused_metrics.json'), result)
    save_results_json(os.path.join(RESULTS_DIR, 'best_focused_source_trial.json'), best)
    np.save(os.path.join(CHECKPOINT_DIR, 'label_vocab_focused.npy'), np.array(label_vocab, dtype=object))
    csv_path = os.path.join(RESULTS_DIR, 'best_focused_results.csv')
    write_header = not os.path.exists(_resolve_model_artifact(csv_path))
    with open(_resolve_model_artifact(csv_path), 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            w.writeheader()
        w.writerow(result)
    print('\n' + '=' * 72)
    print('DONE')
    print(f'Saved model   : {model_path}')
    print(f"Metrics JSON  : {os.path.join(RESULTS_DIR, 'best_focused_metrics.json')}")
    print(f"Test accuracy : {result['test_accuracy']:.4f}")
    if 'f1_macro' in result:
        print(f"F1 macro      : {result['f1_macro']:.4f}")
    print('=' * 72)
if __name__ == '__main__':
    main()
