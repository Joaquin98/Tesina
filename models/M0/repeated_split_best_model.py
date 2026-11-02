"""Repeat M0 initialization seeds using fixed current partitions. Experimental split helpers remain callable separately."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, auxiliary_directory, cached_rgb as current_rgb, validation_data as current_validation
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from models.M0.model import build_optimizer
try:
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_OK = True
except ImportError:
    print('scikit-learn not found. Install it with: pip install scikit-learn')
    SKLEARN_OK = False
for arg in sys.argv[1:]:
    if '=' in arg:
        key, value = arg.split('=', 1)
        os.environ[key.upper()] = value

def _env_int(key, default):
    return int(os.environ.get(key, default))

def _env_float(key, default):
    return float(os.environ.get(key, default))

def _env_bool(key, default=False):
    return os.environ.get(key, str(int(default))).strip().lower() in ('1', 'true', 'yes')
BASE_DIR = Path(__file__).resolve().parent
SOURCE_TRAIN_CSV = Path(os.environ.get('SOURCE_TRAIN_CSV', CURRENT_DATASET / 'train.csv'))
SOURCE_TEST_CSV = Path(os.environ.get('SOURCE_TEST_CSV', CURRENT_DATASET / 'test.csv'))
PARTITIONS_DIR = Path(os.environ.get('PARTITIONS_DIR', auxiliary_directory('repeated') / 'partitions'))
RESULTS_DIR = Path(os.environ.get('RESULTS_DIR', auxiliary_directory('repeated') / 'results'))
CHECKPOINT_DIR = Path(os.environ.get('CHECKPOINT_DIR', auxiliary_directory('repeated') / 'checkpoints'))
EXTRACTOR = os.environ.get('EXTRACTOR', 'resnet50')
CACHE_DIR = BASE_DIR / 'feature_cache' / EXTRACTOR
MAX_SEQ_LENGTH = 200
NUM_FEATURES = 2048
RNN_UNITS = _env_int('RNN_UNITS', 64)
BIDIRECTIONAL = _env_bool('BIDIRECTIONAL', True)
DROPOUT = _env_float('DROPOUT', 0.3)
DENSE_UNITS = _env_int('DENSE_UNITS', 64)
LEARNING_RATE = _env_float('LEARNING_RATE', 0.0001)
WEIGHT_DECAY = _env_float('WEIGHT_DECAY', 1e-05)
BATCH_SIZE = _env_int('BATCH_SIZE', 16)
VALIDATION_SPLIT = _env_float('VALIDATION_SPLIT', 0.2)
LOSS_MODE = os.environ.get('LOSS_MODE', 'sparse_ce').strip().lower()
LABEL_SMOOTHING = _env_float('LABEL_SMOOTHING', 0.0)
FOCAL_GAMMA = _env_float('FOCAL_GAMMA', 2.0)
RUNS = _env_int('RUNS', 5)
RUN_START = _env_int('RUN_START', 1)
SEED = _env_int('SEED', 20260607)
TEST_SIZE = _env_float('TEST_SIZE', 0.2)
TRAIN_TARGET_PER_CLASS = _env_int('TRAIN_TARGET_PER_CLASS', 600)
BALANCE_TRAIN = _env_bool('BALANCE_TRAIN', True)
PARTITIONS_ONLY = _env_bool('PARTITIONS_ONLY', False)
QUICK = _env_bool('QUICK', False)
EPOCHS = _env_int('EPOCHS', 3 if QUICK else 80)
SPLIT_MODE = os.environ.get('SPLIT_MODE', 'stratified').strip().lower()

def ensure_ready():
    if not SKLEARN_OK:
        raise RuntimeError('scikit-learn is required for repeated stratified splits.')
    if not _resolve_model_artifact(SOURCE_TRAIN_CSV).exists():
        raise FileNotFoundError(f'Source train CSV not found: {SOURCE_TRAIN_CSV}')
    if not _resolve_model_artifact(SOURCE_TEST_CSV).exists():
        raise FileNotFoundError(f'Source test CSV not found: {SOURCE_TEST_CSV}')
    if not _resolve_model_artifact(CACHE_DIR).exists():
        raise FileNotFoundError(f'Feature cache not found: {CACHE_DIR}')
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_manifest(df):
    required = {'video_name', 'tag'}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f'Missing columns in manifest: {sorted(missing)}')
    out = df[['video_name', 'tag']].copy()
    out['video_name'] = out['video_name'].astype(str).str.strip()
    out['tag'] = out['tag'].astype(str).str.strip().str.lower()
    out = out[(out['video_name'] != '') & (out['tag'] != '')]
    out = out.drop_duplicates(subset=['video_name'], keep='first')
    return out.reset_index(drop=True)

def cache_path(video_name):
    return CACHE_DIR / (Path(video_name).stem + '.npz')

def filter_cached(df):
    has_cache = df['video_name'].map(lambda name: _resolve_model_artifact(cache_path(name)).exists())
    missing = int((~has_cache).sum())
    if missing:
        print(f'Cache filter: skipping {missing} rows without {EXTRACTOR} features')
    return df[has_cache].reset_index(drop=True)

def describe_counts(df):
    return {str(k): int(v) for k, v in df['tag'].value_counts().sort_index().items()}

def infer_source_group(video_name):
    """Infer a coarse source group from historical clip filenames."""
    stem = Path(str(video_name)).stem
    if stem.endswith('_cut'):
        stem = stem[:-4]
    parts = stem.split('_')
    lower_parts = [p.lower() for p in parts]
    for part in lower_parts:
        if len(part) >= 2 and part[0] in ('f', 'r') and part[1:].isdigit():
            return part.upper()
    team_tokens = [p.upper() for p in parts if p.isalpha() and len(p) >= 3 and (p.lower() not in {'todo', 'otros', 'line', 'scrum', 'juego', 'salidas', 'salida', 'ataque', 'defensa', 'recepcion', 'recepciones', 'penales', 'palos', 'pie', 'cap'})]
    if len(team_tokens) >= 2:
        return '_'.join(team_tokens[-2:])
    if team_tokens:
        return team_tokens[-1]
    if 'otros' in lower_parts:
        return 'OTROS_MISC'
    if 'scrum' in lower_parts:
        return 'SCRUM_MISC'
    if 'pie' in lower_parts:
        return 'PIE_MISC'
    if 'recepciones' in lower_parts or 'recepcion' in lower_parts:
        return 'RECEPCION_MISC'
    return parts[0].upper() if parts else 'UNKNOWN'

def load_source_manifest():
    train_df = normalize_manifest(pd.read_csv(_resolve_model_artifact(SOURCE_TRAIN_CSV)))
    test_df = normalize_manifest(pd.read_csv(_resolve_model_artifact(SOURCE_TEST_CSV)))
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=['video_name'], keep='first')
    all_df = filter_cached(all_df)
    if all_df['tag'].nunique() < 2:
        raise RuntimeError('Need at least two classes for stratified splitting.')
    all_df['source_group'] = all_df['video_name'].map(infer_source_group)
    return all_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

def balance_mixed(df, target, rng):
    if not BALANCE_TRAIN:
        return df.sample(frac=1.0, random_state=int(rng.integers(0, 2 ** 31))).reset_index(drop=True)
    parts = []
    for _, group in df.groupby('tag'):
        if len(group) >= target:
            part = group.sample(n=target, replace=False, random_state=int(rng.integers(0, 2 ** 31)))
        else:
            extra = group.sample(n=target - len(group), replace=True, random_state=int(rng.integers(0, 2 ** 31)))
            part = pd.concat([group, extra], ignore_index=True)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(0, 2 ** 31))).reset_index(drop=True)

def create_partition(all_df, run_index):
    run_seed = SEED + run_index
    if SPLIT_MODE == 'grouped':
        splitter = GroupShuffleSplit(n_splits=64, test_size=TEST_SIZE, random_state=run_seed)
        raw_train = test_df = None
        required_tags = set(all_df['tag'].unique())
        for train_idx, test_idx in splitter.split(all_df, all_df['tag'], groups=all_df['source_group']):
            candidate_train = all_df.iloc[train_idx].copy()
            candidate_test = all_df.iloc[test_idx].copy()
            if set(candidate_test['tag'].unique()) == required_tags:
                raw_train, test_df = (candidate_train, candidate_test)
                break
        if raw_train is None or test_df is None:
            raise RuntimeError(f'Could not create grouped split with all classes for run {run_index}')
    else:
        raw_train, test_df = train_test_split(all_df, test_size=TEST_SIZE, stratify=all_df['tag'], random_state=run_seed)
    rng = np.random.default_rng(run_seed)
    train_df = balance_mixed(raw_train.reset_index(drop=True), TRAIN_TARGET_PER_CLASS, rng)
    test_df = test_df.sample(frac=1.0, random_state=run_seed).reset_index(drop=True)
    run_name = f'run_{run_index:02d}_seed_{run_seed}'
    split_dir = PARTITIONS_DIR / run_name
    split_dir.mkdir(parents=True, exist_ok=True)
    train_path = split_dir / 'train.csv'
    test_path = split_dir / 'test.csv'
    info_path = split_dir / 'split_info.json'
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    info = {'run': run_index, 'seed': run_seed, 'test_size': TEST_SIZE, 'split_mode': SPLIT_MODE, 'source_rows': int(len(all_df)), 'source_groups': int(all_df['source_group'].nunique()), 'raw_train_rows': int(len(raw_train)), 'balanced_train_rows': int(len(train_df)), 'test_rows': int(len(test_df)), 'balance_train': BALANCE_TRAIN, 'train_target_per_class': TRAIN_TARGET_PER_CLASS if BALANCE_TRAIN else None, 'raw_train_counts': describe_counts(raw_train), 'balanced_train_counts': describe_counts(train_df), 'test_counts': describe_counts(test_df), 'train_groups': sorted(raw_train['source_group'].unique().tolist()), 'test_groups': sorted(test_df['source_group'].unique().tolist()), 'train_csv': str(train_path), 'test_csv': str(test_path)}
    info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
    return (run_name, train_df, test_df, info)

def load_features(video_name):
    return current_rgb(video_name, EXTRACTOR)

def build_dataset(df, label_processor, split_name):
    features, masks, labels = ([], [], [])
    misses = 0
    for _, row in df.iterrows():
        frame_features, frame_masks = load_features(row['video_name'])
        if frame_features is None:
            misses += 1
            continue
        tag = str(row['tag']).strip().lower()
        encoded = int(keras.ops.convert_to_numpy(label_processor([tag]))[0])
        features.append(frame_features)
        masks.append(frame_masks)
        labels.append(encoded)
    if misses:
        print(f'  {split_name}: skipped {misses} rows with missing or invalid cache')
    return (np.array(features), np.array(masks), np.array(labels, dtype=np.int32))

def build_model(num_classes):

    def sparse_label_smoothing_loss(y_true, y_pred):
        y_true = keras.ops.cast(y_true, 'int32')
        y_true_one_hot = keras.ops.one_hot(y_true, num_classes)
        smooth = LABEL_SMOOTHING
        y_smooth = y_true_one_hot * (1.0 - smooth) + smooth / float(num_classes)
        y_pred = keras.ops.clip(y_pred, 1e-07, 1.0)
        return -keras.ops.sum(y_smooth * keras.ops.log(y_pred), axis=-1)

    def sparse_focal_loss(y_true, y_pred):
        y_true = keras.ops.cast(y_true, 'int32')
        y_true_one_hot = keras.ops.one_hot(y_true, num_classes)
        y_pred = keras.ops.clip(y_pred, 1e-07, 1.0)
        pt = keras.ops.sum(y_true_one_hot * y_pred, axis=-1)
        ce = -keras.ops.log(pt)
        return keras.ops.power(1.0 - pt, FOCAL_GAMMA) * ce

    def rnn_layer(units, return_sequences):
        layer = keras.layers.GRU(units, return_sequences=return_sequences)
        return keras.layers.Bidirectional(layer) if BIDIRECTIONAL else layer
    feat_input = keras.Input(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), name='features')
    mask_input = keras.Input(shape=(MAX_SEQ_LENGTH,), dtype='bool', name='mask')
    x = rnn_layer(RNN_UNITS, return_sequences=True)(feat_input, mask=mask_input)
    x = rnn_layer(max(RNN_UNITS // 2, 8), return_sequences=False)(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    if DENSE_UNITS > 0:
        x = keras.layers.Dense(DENSE_UNITS, activation='relu')(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model([feat_input, mask_input], output, name='best_model_repeated_split')
    optimizer = build_optimizer('adam', lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if LOSS_MODE == 'label_smoothing':
        loss = sparse_label_smoothing_loss
    elif LOSS_MODE == 'focal':
        loss = sparse_focal_loss
    else:
        loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def save_history_plot(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, val_metric, title in zip(axes, ['loss', 'accuracy'], ['val_loss', 'val_accuracy'], ['Loss', 'Accuracy']):
        ax.plot(history.history.get(metric, []), label='train')
        ax.plot(history.history.get(val_metric, []), label='val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
    fig.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)

def evaluate_and_save(model, x_te, m_te, y_te, test_df, label_vocab, run_dir, run_name):
    test_loss, test_acc = model.evaluate([x_te, m_te], y_te, verbose=0)
    y_prob = model.predict([x_te, m_te], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    result = {'run': run_name, 'test_loss': round(float(test_loss), 6), 'test_accuracy': round(float(test_acc), 6), 'f1_macro': round(float(f1_score(y_te, y_pred, average='macro', zero_division=0)), 6), 'f1_weighted': round(float(f1_score(y_te, y_pred, average='weighted', zero_division=0)), 6)}
    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(label_vocab))))
    row_sum = cm.sum(axis=1)
    recalls = np.divide(np.diag(cm), row_sum, out=np.zeros(len(label_vocab)), where=row_sum != 0)
    for idx, label in enumerate(label_vocab):
        result[f'recall_{label}'] = round(float(recalls[idx]), 6)
    report = classification_report(y_te, y_pred, target_names=label_vocab, zero_division=0)
    (run_dir / 'classification_report.txt').write_text(report, encoding='utf-8')
    predictions = pd.DataFrame({'video_name': list(test_df['video_name'].iloc[:len(y_te)]), 'true_label': [label_vocab[i] for i in y_te], 'predicted_label': [label_vocab[i] for i in y_pred], 'confidence': np.max(y_prob, axis=1)})
    predictions.to_csv(run_dir / 'predictions.csv', index=False)
    for norm, suffix in [(None, 'raw'), ('true', 'normalized')]:
        matrix = confusion_matrix(y_te, y_pred, labels=list(range(len(label_vocab))), normalize=norm)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(matrix, display_labels=label_vocab).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{run_name} confusion ({suffix})')
        fig.tight_layout()
        plt.savefig(run_dir / f'confusion_matrix_{suffix}.png', dpi=100)
        plt.close(fig)
    return result

def train_one_run(run_name, train_df, test_df):
    print(f'\n{run_name}: loading cached features')
    unique_tags = sorted(set(train_df['tag'].unique()).union(set(test_df['tag'].unique())))
    label_processor = keras.layers.StringLookup(num_oov_indices=0, mask_token=None, vocabulary=unique_tags)
    label_vocab = list(label_processor.get_vocabulary())
    x_tr, m_tr, y_tr = build_dataset(train_df, label_processor, 'train')
    x_te, m_te, y_te = build_dataset(test_df, label_processor, 'test')
    print(f'{run_name}: train={len(y_tr)} test={len(y_te)} labels={label_vocab}')
    if len(y_tr) == 0 or len(y_te) == 0:
        raise RuntimeError(f'{run_name}: no data loaded from feature cache')
    class_weight = None
    try:
        weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
        class_weight = dict(enumerate(weights))
    except Exception as exc:
        print(f'{run_name}: class weight warning: {exc}')
    run_dir = RESULTS_DIR / run_name
    run_ckpt_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(len(label_vocab))
    ckpt_path = run_ckpt_dir / 'model.keras'
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True, verbose=1), keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-06, verbose=1)]
    print(f'{run_name}: training up to {EPOCHS} epochs')
    start = time.time()
    history = model.fit([x_tr, m_tr], y_tr, validation_data=current_validation(label_vocab, EXTRACTOR), epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=class_weight, callbacks=callbacks, verbose=1)
    duration = time.time() - start
    result = evaluate_and_save(model, x_te, m_te, y_te, test_df, label_vocab, run_dir, run_name)
    result.update({'extractor': EXTRACTOR, 'rnn_type': 'gru', 'bidirectional': BIDIRECTIONAL, 'rnn_units': RNN_UNITS, 'dropout': DROPOUT, 'dense_units': DENSE_UNITS, 'learning_rate': LEARNING_RATE, 'loss_mode': LOSS_MODE, 'label_smoothing': LABEL_SMOOTHING, 'focal_gamma': FOCAL_GAMMA, 'batch_size': BATCH_SIZE, 'epochs_ran': len(history.history.get('loss', [])), 'duration_s': round(duration, 1), 'checkpoint': str(ckpt_path)})
    save_history_plot(history, run_dir / 'history.png')
    (run_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.save(run_ckpt_dir / 'label_vocab.npy', np.array(label_vocab, dtype=object))
    return result

def write_summary(results, split_infos):
    if RUN_START == 1:
        summary_csv = RESULTS_DIR / 'summary.csv'
    else:
        run_end = RUN_START + RUNS - 1
        summary_csv = RESULTS_DIR / f'summary_runs_{RUN_START:02d}_{run_end:02d}.csv'
    if results:
        keys = list(results[0].keys())
        with _resolve_model_artifact(summary_csv).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    metric_keys = ['test_accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum']
    aggregate = {'runs_requested': RUNS, 'run_start': RUN_START, 'runs_completed': len(results), 'split_mode': SPLIT_MODE, 'partitions_dir': str(PARTITIONS_DIR), 'results_dir': str(RESULTS_DIR), 'checkpoints_dir': str(CHECKPOINT_DIR), 'source_train_csv': str(SOURCE_TRAIN_CSV), 'source_test_csv': str(SOURCE_TEST_CSV), 'split_info': split_infos, 'metrics': {}}
    for key in metric_keys:
        values = [float(row[key]) for row in results if key in row]
        if values:
            aggregate['metrics'][key] = {'mean': round(float(np.mean(values)), 6), 'std': round(float(np.std(values, ddof=0)), 6), 'min': round(float(np.min(values)), 6), 'max': round(float(np.max(values)), 6)}
    (RESULTS_DIR / 'aggregate_metrics.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    return aggregate

def experimental_main():
    ensure_ready()
    print('=' * 72)
    print('Repeated split validation for best CNN-RNN model')
    print('=' * 72)
    print(f'Source train: {SOURCE_TRAIN_CSV}')
    print(f'Source test : {SOURCE_TEST_CSV}')
    print(f'Extractor   : {EXTRACTOR}')
    print(f'Runs        : {RUNS}')
    print(f'Run start   : {RUN_START}')
    print(f'Epochs      : {EPOCHS}')
    print(f'Split mode  : {SPLIT_MODE}')
    print(f'Loss mode   : {LOSS_MODE}')
    print(f'Partitions  : {PARTITIONS_DIR}')
    print(f'Results     : {RESULTS_DIR}')
    all_df = load_source_manifest()
    print(f'\nSource rows after dedupe/cache filter: {len(all_df)}')
    print(f'Source counts: {describe_counts(all_df)}')
    results = []
    split_infos = []
    for run_index in range(RUN_START, RUN_START + RUNS):
        run_name, train_df, test_df, info = create_partition(all_df, run_index)
        split_infos.append(info)
        print(f"\n{run_name}: train {info['balanced_train_counts']} | test {info['test_counts']}")
        if not PARTITIONS_ONLY:
            result = train_one_run(run_name, train_df, test_df)
            results.append(result)
            print(f"{run_name}: acc={result['test_accuracy']:.4f} f1_macro={result['f1_macro']:.4f}")
    aggregate = write_summary(results, split_infos)
    print('\nDone.')
    if PARTITIONS_ONLY:
        print(f'Created partitions only: {PARTITIONS_DIR}')
    else:
        print(f"Summary CSV: {RESULTS_DIR / 'summary.csv'}")
        print('Aggregate metrics:')
        print(json.dumps(aggregate['metrics'], indent=2))
def main():
    """Repeat initialization seeds while keeping the current partitions fixed."""
    from models.common.workflow import train_run
    records = []
    for run_index in range(RUN_START, RUN_START + RUNS):
        records.append(train_run('M0', dataset=CURRENT_DATASET, epochs=EPOCHS,
                                 batch_size=BATCH_SIZE, seed=SEED + run_index))
    print(json.dumps(records, ensure_ascii=False))


if __name__ == '__main__':
    main()
