"""Optimize recurrent heads using the current CSV partitions and shared feature cache."""
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
from models.M0.config import TRAIN_CSV, TEST_CSV, FEATURE_CACHE_DIR, MAX_SEQ_LENGTH, NUM_FEATURES, CLASSES_TO_USE, VALIDATION_SPLIT, WEIGHT_DECAY, OPTIMIZER_NAME
from models.M0.model import build_optimizer
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    print('❌  Optuna not installed. Run:  pip install optuna')
    sys.exit(1)
try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    print('⚠  scikit-learn not found — install it: pip install scikit-learn')
    SKLEARN_AVAILABLE = False
for arg in sys.argv[1:]:
    if '=' in arg:
        k, v = arg.split('=', 1)
        os.environ[k.upper()] = v

def _env(key, default):
    return os.environ.get(key, str(default))

def _env_int(key, default):
    return int(_env(key, default))

def _env_bool(key, default=False):
    return _env(key, str(int(default))).strip() in ('1', 'true', 'yes')
EXTRACTOR = _env('EXTRACTOR', 'inceptionv3')
RNN_TYPE = _env('RNN_TYPE', 'gru')
N_TRIALS = _env_int('N_TRIALS', 50)
EPOCHS = _env_int('EPOCHS', 30)
QUICK = _env_bool('QUICK')
RESUME = _env_bool('RESUME')
if QUICK:
    EPOCHS = min(EPOCHS, 5)
    N_TRIALS = min(N_TRIALS, 5)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR_BASE = str(auxiliary_directory('optuna_search') / 'results')
OPTUNA_DIR = str(auxiliary_directory('optuna_search') / 'results')
RESULTS_CSV = os.path.join(RESULTS_DIR_BASE, 'grid_search_results.csv')
BEST_TXT = os.path.join(RESULTS_DIR_BASE, 'optuna_best.txt')
STUDY_NAME = f'optuna__{EXTRACTOR}__{RNN_TYPE}'
STORAGE_PATH = f'sqlite:///{os.path.join(OPTUNA_DIR, STUDY_NAME)}.db'
os.makedirs(OPTUNA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR_BASE, exist_ok=True)

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
        features.append(ff)
        masks.append(fm)
        labels.append(enc)
    if missing:
        print(f"  ⚠  {len(missing)} {split_name} cache misses for '{extractor}'")
    return (np.array(features), np.array(masks), np.array(labels))

def build_rnn_model(rnn_type, num_classes, optimizer, rnn_units=64, bidirectional=True, dropout=0.4, dense_units=64):
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
    model = keras.Model([feat_in, mask_in], out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

class OptunaPruningCallback(keras.callbacks.Callback):

    def __init__(self, trial, monitor='val_loss'):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        self.trial.report(float(value), step=epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned(f'Pruned at epoch {epoch + 1} ({self.monitor}={value:.4f})')

def append_to_csv(row: dict, path: str):
    write_header = not os.path.exists(_resolve_model_artifact(path))
    with open(_resolve_model_artifact(path), 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def make_objective(train_features, train_masks, train_labels, test_features, test_masks, test_labels, label_vocab, completed_trials):

    def objective(trial):
        rnn_units = trial.suggest_categorical('rnn_units', [32, 64, 128, 256])
        bidirectional = trial.suggest_categorical('bidirectional', [False, True])
        dropout = trial.suggest_float('dropout', 0.2, 0.6, step=0.05)
        dense_units = trial.suggest_categorical('dense_units', [0, 64, 128, 256])
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        params = dict(rnn_units=rnn_units, bidirectional=bidirectional, dropout=dropout, dense_units=dense_units, learning_rate=learning_rate, batch_size=batch_size)
        bi = 'bi' if bidirectional else 'uni'
        lr_str = f'lr{learning_rate:.1e}'.replace('-0', '-')
        trial_name = f'optuna__{EXTRACTOR}__{RNN_TYPE}__u{rnn_units}__{bi}__drop{int(dropout * 100)}__d{dense_units}__{lr_str}__bs{batch_size}__t{trial.number}'
        if trial_name in completed_trials:
            raise optuna.exceptions.TrialPruned(f'Already completed: {trial_name}')
        trial_dir = os.path.join(OPTUNA_DIR, STUDY_NAME, trial_name)
        os.makedirs(trial_dir, exist_ok=True)
        print(f'\n  Trial #{trial.number}: {trial_name}')
        optimizer = build_optimizer(OPTIMIZER_NAME, lr=learning_rate, weight_decay=WEIGHT_DECAY)
        model = build_rnn_model(rnn_type=RNN_TYPE, num_classes=len(label_vocab), optimizer=optimizer, rnn_units=rnn_units, bidirectional=bidirectional, dropout=dropout, dense_units=dense_units)
        fit_kwargs = dict(validation_data=current_validation(label_vocab, EXTRACTOR), epochs=EPOCHS, batch_size=batch_size, verbose=0)
        if SKLEARN_AVAILABLE:
            try:
                cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
                fit_kwargs['class_weight'] = dict(enumerate(cw))
            except Exception:
                pass
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0), OptunaPruningCallback(trial, monitor='val_loss')]
        t0 = time.time()
        try:
            history = model.fit([train_features, train_masks], train_labels, callbacks=callbacks, **fit_kwargs)
        except optuna.exceptions.TrialPruned:
            raise
        duration = time.time() - t0
        best_epoch = int(np.argmin(history.history.get('val_loss', [0]))) + 1
        val_acc_best = float(max(history.history.get('val_accuracy', [0])))
        val_loss_best = float(min(history.history.get('val_loss', [9000000000.0])))
        epochs_ran = len(history.history['loss'])
        test_loss, test_acc = model.evaluate([test_features, test_masks], test_labels, verbose=0)
        metrics = {'trial_name': trial_name, 'extractor': EXTRACTOR, 'rnn_type': RNN_TYPE, 'p_rnn_units': rnn_units, 'p_bidirectional': bidirectional, 'p_dropout': round(dropout, 4), 'p_dense_units': dense_units, 'p_learning_rate': learning_rate, 'p_batch_size': batch_size, 'epochs_ran': epochs_ran, 'best_epoch': best_epoch, 'val_acc_best': round(val_acc_best, 6), 'val_loss_best': round(val_loss_best, 6), 'test_loss': round(float(test_loss), 6), 'test_accuracy': round(float(test_acc), 6), 'duration_s': round(duration, 1)}
        f1_macro = 0.0
        if SKLEARN_AVAILABLE:
            y_prob = model.predict([test_features, test_masks], verbose=0)
            y_pred = np.argmax(y_prob, axis=1)
            y_true = test_labels
            validation_x, validation_y = current_validation(label_vocab, EXTRACTOR)
            validation_predictions = np.argmax(model.predict(validation_x, verbose=0), axis=1)
            f1_macro = float(f1_score(validation_y, validation_predictions, average='macro', zero_division=0))
            f1_weighted = float(f1_score(validation_y, validation_predictions, average='weighted', zero_division=0))
            metrics['val_f1_macro'] = round(f1_macro, 4)
            metrics['val_f1_weighted'] = round(f1_weighted, 4)
            cm = confusion_matrix(y_true, y_pred)
            row_sums = cm.sum(axis=1)
            recalls = np.divide(np.diag(cm).astype(float), row_sums, out=np.zeros(len(label_vocab)), where=row_sums != 0)
            for i, cls in enumerate(label_vocab):
                metrics[f'recall_{cls}'] = round(float(recalls[i]), 4)
            report = classification_report(y_true, y_pred, target_names=label_vocab, zero_division=0)
            with open(_resolve_model_artifact(os.path.join(trial_dir, 'classification_report.txt')), 'w') as f:
                f.write(f'Trial: {trial_name}\n\n{report}')
            print(f'  F1_macro={f1_macro:.4f}  test_acc={test_acc:.4f}')
            try:
                from sklearn.metrics import ConfusionMatrixDisplay
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay(cm, display_labels=label_vocab).plot(ax=ax, cmap='Blues', colorbar=False)
                ax.set_title(f'Confusion (raw)')
                fig.tight_layout()
                plt.savefig(os.path.join(trial_dir, 'confusion_matrix_raw.png'), dpi=90)
                plt.close(fig)
                cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay(cm_norm, display_labels=label_vocab).plot(ax=ax, cmap='Blues', colorbar=False)
                ax.set_title('Confusion (norm.)')
                fig.tight_layout()
                plt.savefig(os.path.join(trial_dir, 'confusion_matrix_normalized.png'), dpi=90)
                plt.close(fig)
            except Exception as e:
                print(f'  ⚠  CM PNG error: {e}')
        else:
            metrics['val_f1_macro'] = None
            metrics['val_f1_weighted'] = None
        import json as _json
        hdata = {k: [float(x) for x in v] for k, v in history.history.items()}
        with open(_resolve_model_artifact(os.path.join(trial_dir, 'history.json')), 'w') as f:
            _json.dump({'trial_name': trial_name, 'params': params, 'extractor': EXTRACTOR, 'rnn_type': RNN_TYPE, 'history': hdata}, f)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, m, vm in zip(axes, ['loss', 'accuracy'], ['val_loss', 'val_accuracy']):
            ax.plot(history.history.get(m, []), label=f'train_{m}')
            ax.plot(history.history.get(vm, []), label=vm)
            ax.set_title(m)
            ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(trial_dir, 'history.png'))
        plt.close(fig)
        append_to_csv(metrics, RESULTS_CSV)
        completed_trials.add(trial_name)
        return f1_macro
    return objective

def main():
    print('=' * 64)
    print(f'  Optuna Hyperparameter Search')
    print(f'  Extractor : {EXTRACTOR}   RNN : {RNN_TYPE}')
    print(f'  Trials    : {N_TRIALS}    Epochs/trial : {EPOCHS}')
    print(f'  Study     : {STUDY_NAME}')
    print(f'  Storage   : {STORAGE_PATH}')
    print('=' * 64)
    train_df = pd.read_csv(_resolve_model_artifact(TRAIN_CSV))
    test_df = pd.read_csv(_resolve_model_artifact(TEST_CSV))
    if CLASSES_TO_USE is not None:
        train_df = train_df[train_df['tag'].isin(CLASSES_TO_USE)].copy()
        test_df = test_df[test_df['tag'].isin(CLASSES_TO_USE)].copy()
    unique_tags = sorted(set((str(t).strip().lower() for t in train_df['tag'].unique())))
    label_processor = keras.layers.StringLookup(num_oov_indices=0, mask_token=None, vocabulary=unique_tags)
    label_vocab = list(label_processor.get_vocabulary())
    print(f'Label vocab: {label_vocab}')
    print(f'\nLoading {EXTRACTOR} features...')
    train_features, train_masks, train_labels = build_dataset(train_df, label_processor, EXTRACTOR, 'train')
    test_features, test_masks, test_labels = build_dataset(test_df, label_processor, EXTRACTOR, 'test')
    if len(train_labels) == 0:
        print(f"❌  No training data found for '{EXTRACTOR}'. Check feature_cache/.")
        return
    print(f'  ✓  {len(train_labels)} train  /  {len(test_labels)} test samples loaded.')
    completed_trials = set()
    if RESUME and os.path.exists(_resolve_model_artifact(RESULTS_CSV)):
        df_existing = pd.read_csv(_resolve_model_artifact(RESULTS_CSV))
        completed_trials = set(df_existing['trial_name'].astype(str).tolist())
        print(f'  Resume mode: {len(completed_trials)} trials already in CSV.')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE_PATH, direction='maximize', sampler=TPESampler(seed=42), pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5), load_if_exists=True)
    objective = make_objective(train_features, train_masks, train_labels, test_features, test_masks, test_labels, label_vocab, completed_trials)
    print(f'\nStarting Optuna optimisation ({N_TRIALS} trials)...\n')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, catch=(Exception,))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"\n{'=' * 64}")
    print(f'  Optuna study complete')
    print(f'  Completed : {len(completed)}  |  Pruned : {len(pruned)}  |  Failed : {len(failed)}')
    print(f"{'=' * 64}")
    if not completed:
        print('No completed trials — nothing to summarise.')
        return
    best = study.best_trial
    print(f'\nBest trial #{best.number}:')
    print(f'  Val F1 macro : {best.value:.4f}')
    for k, v in best.params.items():
        print(f'  {k:20s}: {v}')
    summary = f'Best Optuna trial for {EXTRACTOR} + {RNN_TYPE}\nVal F1 macro: {best.value:.4f}\n' + '\n'.join((f'  {k}: {v}' for k, v in best.params.items()))
    with open(_resolve_model_artifact(BEST_TXT), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'\nBest config saved to: {BEST_TXT}')
    try:
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, 4))
        names = list(importances.keys())
        vals = list(importances.values())
        ax.barh(names[::-1], vals[::-1], color='#1565C0')
        ax.set_xlabel('Relative Importance')
        ax.set_title(f'Hyperparameter Importance — {STUDY_NAME}')
        fig.tight_layout()
        chart_path = os.path.join(OPTUNA_DIR, f'{STUDY_NAME}_importance.png')
        plt.savefig(chart_path, dpi=100)
        plt.close(fig)
        print(f'Importance chart saved: {chart_path}')
    except Exception as e:
        print(f'  ⚠  Could not generate importance chart: {e}')
if __name__ == '__main__':
    main()
