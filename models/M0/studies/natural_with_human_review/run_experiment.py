"""Retrain the natural-distribution ResNet50-BiGRU with human-reviewed clips.

Uses historical data, trained models and feature caches as inputs.
Writes experimental artifacts under data/natural_with_human_review.
"""
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
from models.common.study_data import study_csv_path, RGBFeatureStore
import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import keras
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = experiment_directory(__file__)
REPO = HERE.parents[1]
SOURCE_MODEL_DIR = REPO / 'models' / 'M0'
SOURCE_EXPERIMENT = REPO / 'data' / 'class_imbalance_comparison' / 'full'
SOURCE_SPLITS = CURRENT_DATASET
REVIEW_MANIFEST = Path(os.environ.get('REVIEW_MANIFEST', Path(os.environ.get('TESINA_WEB_RUNTIME', REPO / '.runtime/web')) / 'active_learning_dataset/train_manifest.csv')).expanduser().resolve()
REFERENCE_RUN = SOURCE_EXPERIMENT / 'runs' / 'natural_unweighted'
LEGACY_REVIEW_CACHES = (SOURCE_MODEL_DIR / 'feature_cache_human_review_retrain_860' / 'resnet50', SOURCE_MODEL_DIR / 'feature_cache_human_review_retrain' / 'resnet50', HERE / 'smoke' / 'cache' / 'new_review_npz')
_configured_review_caches = os.environ.get('REVIEW_CACHE_DIRS', '').strip()
REVIEW_CACHES = tuple((Path(item).expanduser().resolve() for item in _configured_review_caches.split(os.pathsep) if item.strip())) or LEGACY_REVIEW_CACHES
SEED = 42
LABELS = ('line', 'other', 'scrum')
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
MAX_SEQ_LENGTH = 200
NUM_FEATURES = 2048
BATCH_SIZE = 16
MAX_EPOCHS = 100
BOOTSTRAP_REPLICATES = 2000
VARIANT_SPECS = {'all_reviews': {'selection': 'all', 'review_weight': 1.0, 'class_weights': False, 'source_balance': False}, 'nonoverlap_reviews': {'selection': 'nonoverlap', 'review_weight': 1.0, 'class_weights': False, 'source_balance': False}, 'aligned_unweighted_rw025': {'selection': 'all', 'review_weight': 0.25, 'class_weights': False, 'source_balance': False}, 'aligned_unweighted_rw050': {'selection': 'all', 'review_weight': 0.5, 'class_weights': False, 'source_balance': False}, 'aligned_unweighted_rw100': {'selection': 'all', 'review_weight': 1.0, 'class_weights': False, 'source_balance': False}, 'aligned_classweighted_rw025': {'selection': 'all', 'review_weight': 0.25, 'class_weights': True, 'source_balance': False}, 'aligned_classweighted_rw050': {'selection': 'all', 'review_weight': 0.5, 'class_weights': True, 'source_balance': False}, 'aligned_classweighted_rw100': {'selection': 'all', 'review_weight': 1.0, 'class_weights': True, 'source_balance': False}, 'aligned_sourcebalanced': {'selection': 'all', 'review_weight': 1.0, 'class_weights': False, 'source_balance': True}, 'aligned_nonoverlap_rw100': {'selection': 'nonoverlap', 'review_weight': 1.0, 'class_weights': False, 'source_balance': False}, 'balanced_control': {'selection': 'none', 'review_weight': 0.0, 'class_weights': False, 'source_balance': False}, 'balanced_aligned_rw025': {'selection': 'all', 'review_weight': 0.25, 'class_weights': False, 'source_balance': False}, 'balanced_aligned_rw050': {'selection': 'all', 'review_weight': 0.5, 'class_weights': False, 'source_balance': False}, 'balanced_aligned_rw100': {'selection': 'all', 'review_weight': 1.0, 'class_weights': False, 'source_balance': False}}
VARIANTS = tuple(VARIANT_SPECS)

def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def label_counts(frame: pd.DataFrame, column: str='tag') -> dict[str, int]:
    counts = frame[column].value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}

def stratified_limit(frame: pd.DataFrame, limit: int, label_column: str, seed: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.reset_index(drop=True)
    pieces = []
    fractions = frame[label_column].value_counts(normalize=True)
    allocation = {label: max(1, int(round(limit * fractions.get(label, 0)))) for label in LABELS}
    while sum(allocation.values()) > limit:
        allocation[max(allocation, key=allocation.get)] -= 1
    while sum(allocation.values()) < limit:
        label = max(LABELS, key=lambda item: fractions.get(item, 0) - allocation[item] / limit)
        allocation[label] += 1
    rng = np.random.default_rng(seed)
    for label in LABELS:
        group = frame[frame[label_column] == label]
        pieces.append(group.sample(min(allocation[label], len(group)), random_state=int(rng.integers(2 ** 31))))
    return pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)

def balanced_reference_rows(train: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Reproduce the mixed 600-per-class reference schedule."""
    y = train['tag'].map(LABEL_TO_ID).to_numpy(dtype=np.int32)
    rng = np.random.default_rng(seed)
    parts = []
    for class_id in range(len(LABELS)):
        ids = np.flatnonzero(y == class_id)
        if len(ids) >= target:
            part = rng.choice(ids, size=target, replace=False)
        else:
            extra = rng.choice(ids, size=target - len(ids), replace=True)
            part = np.concatenate([ids, extra])
        parts.append(part)
    selected = np.concatenate(parts)
    rng.shuffle(selected)
    return train.iloc[selected].reset_index(drop=True)

def load_base_splits(quick: bool, base_distribution: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = [frame[['video_name', 'tag']].copy() for frame in current_frames()]
    train, validation, test = frames
    if set(train.video_name) & set(validation.video_name) or set(train.video_name) & set(test.video_name):
        raise ValueError('Base split leakage detected')
    if set(validation.video_name) & set(test.video_name):
        raise ValueError('Validation/test leakage detected')
    if quick:
        train = stratified_limit(train, 96, 'tag', SEED)
        validation = stratified_limit(validation, 48, 'tag', SEED + 1)
        test = stratified_limit(test, 60, 'tag', SEED + 2)
    if base_distribution == 'balanced_reference':
        train = balanced_reference_rows(train, 30 if quick else 600, SEED)
    elif base_distribution != 'natural':
        raise ValueError(f'Unsupported base distribution: {base_distribution}')
    return (train, validation, test)

def load_reviews() -> pd.DataFrame:
    review = pd.read_csv(_resolve_model_artifact(study_csv_path(REVIEW_MANIFEST))).copy()
    required = {'dataset_id', 'source_video', 'start_time', 'end_time', 'duration', 'correct_label', 'review_outcome', 'clip_path', 'clip_name', 'session_id'}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f'Review manifest lacks columns: {sorted(missing)}')
    review['dataset_id'] = review['dataset_id'].astype(str)
    review['tag'] = review['correct_label'].astype(str).str.strip().str.lower()
    review['clip_path'] = review['clip_path'].astype(str)
    review['clip_name'] = review['clip_name'].astype(str)
    review['start_time'] = pd.to_numeric(review['start_time'], errors='raise')
    review['end_time'] = pd.to_numeric(review['end_time'], errors='raise')
    if set(review.tag) - set(LABELS):
        raise ValueError(f'Unexpected review labels: {sorted(set(review.tag) - set(LABELS))}')
    if review.dataset_id.duplicated().any() or review.clip_path.duplicated().any():
        raise ValueError('Exact duplicate review rows detected')
    missing_clips = [path for path in review.clip_path if not _resolve_model_artifact(study_csv_path(Path(path))).exists()]
    if missing_clips:
        raise FileNotFoundError(f'{len(missing_clips)} reviewed clips are missing; first={missing_clips[:3]}')
    return review.reset_index(drop=True)

def nonoverlap_reviews(review: pd.DataFrame) -> pd.DataFrame:
    """Greedily retain temporally non-overlapping windows within each source video."""
    selected = []
    ordered = review.sort_values(['source_video', 'start_time', 'end_time', 'dataset_id'])
    for _, group in ordered.groupby('source_video', sort=True):
        previous_end = -np.inf
        for index, row in group.iterrows():
            if float(row.start_time) >= previous_end:
                selected.append(index)
                previous_end = float(row.end_time)
    return review.loc[selected].reset_index(drop=True)

def prepare_manifests(output_root: Path, quick: bool, base_distribution: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    train, validation, test = load_base_splits(quick, base_distribution)
    all_review = load_reviews()
    nonoverlap = nonoverlap_reviews(all_review)
    if quick:
        all_review = stratified_limit(all_review, 30, 'tag', SEED + 3)
        nonoverlap = stratified_limit(nonoverlap, 24, 'tag', SEED + 4)
    variants = {'all_reviews': all_review, 'nonoverlap_reviews': nonoverlap, 'none_reviews': all_review.iloc[0:0].copy()}
    split_dir = output_root / 'manifests'
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / 'train_natural.csv', index=False)
    validation.to_csv(split_dir / 'validation_fixed.csv', index=False)
    test.to_csv(split_dir / 'test_natural_fixed.csv', index=False)
    for name, frame in variants.items():
        frame.to_csv(split_dir / f'{name}.csv', index=False)
    exact_name_overlap = {split: sorted(set(all_review.clip_name) & set(frame.video_name)) for split, frame in (('train', train), ('validation', validation), ('test', test))}
    audit = {'created_at': pd.Timestamp.now().isoformat(), 'seed': SEED, 'quick': quick, 'base_distribution': base_distribution, 'protocol': {'architecture': 'ResNet50 cached features -> BiGRU(64, seq) -> BiGRU(32) -> Dropout(0.3) -> Dense(64, relu) -> softmax', 'optimizer': 'Adam learning_rate=1e-4', 'loss': 'sparse_categorical_crossentropy', 'class_weights': None, 'resampling': None, 'batch_size': BATCH_SIZE, 'maximum_epochs': 1 if quick else MAX_EPOCHS, 'early_stopping': 'val_accuracy max, patience=10, restore_best_weights', 'lr_reduction': 'val_loss, factor=0.5, patience=5, min_lr=1e-6', 'validation_and_test': 'fixed current data/clips_dataset partitions'}, 'base': {'train': {'rows': len(train), 'counts': label_counts(train)}, 'validation': {'rows': len(validation), 'counts': label_counts(validation)}, 'test': {'rows': len(test), 'counts': label_counts(test)}}, 'reviews': {'manifest': str(REVIEW_MANIFEST), 'all_rows': len(all_review), 'all_counts': label_counts(all_review), 'nonoverlap_rows': len(nonoverlap), 'nonoverlap_counts': label_counts(nonoverlap), 'source_video_count': int(all_review.source_video.nunique()), 'source_counts': {str(k): int(v) for k, v in all_review.source_video.value_counts().items()}, 'session_count': int(all_review.session_id.nunique()), 'exact_clip_name_overlap': exact_name_overlap, 'missing_clip_count': 0}, 'software': {'python': sys.version.split()[0], 'platform': platform.platform(), 'keras': keras.__version__, 'tensorflow': tf.__version__, 'numpy': np.__version__}}
    json_dump(output_root / 'audit.json', audit)
    return (train, validation, test, variants)

class FeatureStore:

    def __init__(self, features_path: Path, masks_path: Path, names: list[str]):
        self.features = _load_model_result_array(features_path, mmap_mode='r')
        self.masks = _load_model_result_array(masks_path, mmap_mode='r')
        self.names = list(map(str, names))
        self.index = {name: i for i, name in enumerate(self.names)}
        if self.features.shape != (len(self.names), MAX_SEQ_LENGTH, NUM_FEATURES):
            raise ValueError(f'Unexpected feature store shape: {self.features.shape}')
        if self.masks.shape != (len(self.names), MAX_SEQ_LENGTH):
            raise ValueError(f'Unexpected mask store shape: {self.masks.shape}')

def load_base_store() -> FeatureStore:
    return RGBFeatureStore()

def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not _resolve_model_artifact(study_csv_path(path)).exists():
        return None
    try:
        with _load_model_result_array(path) as data:
            features = np.asarray(data['frame_features'], dtype=np.float32)
            masks = np.asarray(data['frame_masks'], dtype=bool)
        if features.shape == (MAX_SEQ_LENGTH, NUM_FEATURES) and masks.shape == (MAX_SEQ_LENGTH,):
            return (features, masks)
    except Exception:
        return None
    return None

def _review_cache_path(row: pd.Series, new_cache: Path) -> Path | None:
    name = f'{row.dataset_id}.npz'
    for directory in (*REVIEW_CACHES, new_cache):
        candidate = directory / name
        if _load_npz(candidate) is not None:
            return candidate
    return None

def _extract_review_features(row: pd.Series, cache_path: Path, extractor) -> tuple[np.ndarray, np.ndarray]:
    if str(SOURCE_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(source_directory(SOURCE_MODEL_DIR)))
    from data_utils import load_video
    frames = load_video(str(row.clip_path), max_frames=MAX_SEQ_LENGTH, target_fps=25.0)
    features = np.zeros((MAX_SEQ_LENGTH, NUM_FEATURES), dtype=np.float32)
    masks = np.zeros((MAX_SEQ_LENGTH,), dtype=bool)
    if len(frames):
        extracted = extractor.predict(frames, batch_size=32, verbose=0)
        length = min(len(extracted), MAX_SEQ_LENGTH)
        features[:length] = extracted[:length]
        masks[:length] = True
    if not masks.any():
        raise ValueError(f'No frames extracted from reviewed clip: {row.clip_path}')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, frame_features=features, frame_masks=masks)
    return (features, masks)

def build_review_store(output_root: Path, review: pd.DataFrame) -> tuple[FeatureStore, dict]:
    cache_dir = output_root / 'cache'
    new_npz = cache_dir / 'new_review_npz'
    features_path = cache_dir / 'review_features.npy'
    masks_path = cache_dir / 'review_masks.npy'
    index_path = cache_dir / 'review_index.json'
    ids = review.dataset_id.astype(str).tolist()
    if _resolve_model_artifact(study_csv_path(features_path)).exists() and _resolve_model_artifact(study_csv_path(masks_path)).exists() and _resolve_model_artifact(study_csv_path(index_path)).exists():
        old_index = json.loads(_resolve_model_artifact(study_csv_path(index_path)).read_text(encoding='utf-8'))
        if old_index.get('dataset_ids') == ids:
            store = FeatureStore(features_path, masks_path, ids)
            return (store, old_index.get('build_audit', {}))
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_features = cache_dir / 'review_features.tmp.npy'
    temp_masks = cache_dir / 'review_masks.tmp.npy'
    feature_map = np.lib.format.open_memmap(temp_features, mode='w+', dtype=np.float32, shape=(len(review), MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_map = np.lib.format.open_memmap(temp_masks, mode='w+', dtype=bool, shape=(len(review), MAX_SEQ_LENGTH))
    extractor = None
    reused = 0
    extracted_count = 0
    started = time.perf_counter()
    try:
        for position, (_, row) in enumerate(review.iterrows()):
            cache_path = _review_cache_path(row, new_npz)
            loaded = _load_npz(cache_path) if cache_path else None
            if loaded is None:
                if extractor is None:
                    if str(SOURCE_MODEL_DIR) not in sys.path:
                        sys.path.insert(0, str(source_directory(SOURCE_MODEL_DIR)))
                    from model import build_feature_extractor_resnet50
                    print('Building ResNet50 extractor for missing reviewed features', flush=True)
                    extractor = build_feature_extractor_resnet50()
                loaded = _extract_review_features(row, new_npz / f'{row.dataset_id}.npz', extractor)
                extracted_count += 1
            else:
                reused += 1
            feature_map[position] = loaded[0]
            mask_map[position] = loaded[1]
            if (position + 1) % 25 == 0 or position + 1 == len(review):
                print(f'Review feature store: {position + 1}/{len(review)}', flush=True)
        feature_map.flush()
        mask_map.flush()
    finally:
        del feature_map
        del mask_map
        keras.backend.clear_session()
        gc.collect()
    temp_features.replace(features_path)
    temp_masks.replace(masks_path)
    build_audit = {'rows': len(review), 'legacy_or_existing_cache_rows': reused, 'newly_extracted_rows': extracted_count, 'seconds': time.perf_counter() - started}
    json_dump(index_path, {'dataset_ids': ids, 'build_audit': build_audit})
    return (FeatureStore(features_path, masks_path, ids), build_audit)

def indexed_rows(base: pd.DataFrame, review: pd.DataFrame, base_store: FeatureStore, review_store: FeatureStore, review_weight: float=1.0, source_balance: bool=False) -> pd.DataFrame:
    rows = []
    for row in base.itertuples(index=False):
        if row.video_name not in base_store.index:
            raise KeyError(f'Base feature missing: {row.video_name}')
        rows.append({'name': row.video_name, 'tag': row.tag, 'y': LABEL_TO_ID[row.tag], 'store': 'base', 'feature_index': base_store.index[row.video_name], 'source_video': '', 'review_outcome': '', 'sample_weight': 1.0})
    source_factors = {}
    if len(review) and source_balance:
        source_counts = review.source_video.astype(str).value_counts()
        target_count = float(source_counts.min())
        source_factors = {str(source): target_count / float(count) for source, count in source_counts.items()}
    for row in review.itertuples(index=False):
        dataset_id = str(row.dataset_id)
        if dataset_id not in review_store.index:
            raise KeyError(f'Review feature missing: {dataset_id}')
        rows.append({'name': dataset_id, 'tag': row.tag, 'y': LABEL_TO_ID[row.tag], 'store': 'review', 'feature_index': review_store.index[dataset_id], 'source_video': str(row.source_video), 'review_outcome': str(row.review_outcome), 'sample_weight': float(review_weight) * float(source_factors.get(str(row.source_video), 1.0))})
    return pd.DataFrame(rows)

class MixedFeatureBatches(keras.utils.PyDataset):

    def __init__(self, rows: pd.DataFrame, base_store: FeatureStore, review_store: FeatureStore, shuffle: bool, seed: int):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
        self.rows = rows.reset_index(drop=True)
        self.base_store = base_store
        self.review_store = review_store
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.schedule = self._schedule()

    def _schedule(self) -> np.ndarray:
        indices = np.arange(len(self.rows))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        return indices

    def __len__(self) -> int:
        return int(np.ceil(len(self.rows) / BATCH_SIZE))

    def __getitem__(self, batch_index: int):
        ids = self.schedule[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
        batch = self.rows.iloc[ids]
        features = np.empty((len(batch), MAX_SEQ_LENGTH, NUM_FEATURES), dtype=np.float32)
        masks = np.empty((len(batch), MAX_SEQ_LENGTH), dtype=bool)
        for store_name, store in (('base', self.base_store), ('review', self.review_store)):
            positions = np.flatnonzero(batch.store.to_numpy() == store_name)
            if len(positions):
                source_ids = batch.iloc[positions].feature_index.to_numpy(dtype=np.int64)
                features[positions] = store.features[source_ids]
                masks[positions] = store.masks[source_ids]
        labels = batch.y.to_numpy(dtype=np.int32)
        sample_weights = batch.sample_weight.to_numpy(dtype=np.float32)
        return ((features, masks), labels, sample_weights)

    def on_epoch_end(self):
        self.epoch += 1
        self.schedule = self._schedule()

def build_model() -> keras.Model:
    feature_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES), name='features')
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool', name='mask')
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True))(feature_input, mask=mask_input)
    x = keras.layers.Bidirectional(keras.layers.GRU(32))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(len(LABELS), activation='softmax')(x)
    model = keras.Model([feature_input, mask_input], output, name='resnet50_bigru_natural_human_review')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(len(LABELS)), zero_division=0)
    return {'accuracy': float(accuracy_score(y_true, y_pred)), 'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 'per_class': {label: {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i, label in enumerate(LABELS)}}

def save_plots(run_dir: Path, history: pd.DataFrame, matrix: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric in zip(axes, ('loss', 'accuracy')):
        axis.plot(history[metric], label='train')
        axis.plot(history[f'val_{metric}'], label='validation')
        axis.set_title(metric)
        axis.set_xlabel('Epoch')
        axis.legend()
    fig.tight_layout()
    fig.savefig(run_dir / 'history.png', dpi=120)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap='Blues')
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            axis.text(j, i, str(matrix[i, j]), ha='center', va='center')
    axis.set_xticks(range(len(LABELS)), LABELS)
    axis.set_yticks(range(len(LABELS)), LABELS)
    axis.set_xlabel('Predicted')
    axis.set_ylabel('True')
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(run_dir / 'confusion_matrix.png', dpi=120)
    plt.close(fig)

def train_variant(name: str, output_root: Path, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, reviews: pd.DataFrame, base_store: FeatureStore, review_store: FeatureStore, quick: bool, force: bool, review_weight: float, use_class_weights: bool, source_balance: bool) -> dict:
    run_dir = output_root / 'runs' / name
    metrics_path = run_dir / 'metrics.json'
    if _resolve_model_artifact(study_csv_path(metrics_path)).exists() and (not force):
        print(f'Skipping completed variant: {name}', flush=True)
        return json.loads(_resolve_model_artifact(study_csv_path(metrics_path)).read_text(encoding='utf-8'))
    run_dir.mkdir(parents=True, exist_ok=True)
    train_rows = indexed_rows(train, reviews, base_store, review_store, review_weight=review_weight, source_balance=source_balance)
    validation_rows = indexed_rows(validation, reviews.iloc[0:0], base_store, review_store)
    test_rows = indexed_rows(test, reviews.iloc[0:0], base_store, review_store)
    class_weights = None
    if use_class_weights:
        counts = train_rows.y.value_counts().to_dict()
        class_weights = {int(label_id): float(len(train_rows) / (len(LABELS) * counts[label_id])) for label_id in range(len(LABELS))}
        train_rows['sample_weight'] = train_rows.apply(lambda row: float(row.sample_weight) * class_weights[int(row.y)], axis=1)
    train_rows.to_csv(run_dir / 'train_manifest.csv', index=False)
    validation_rows.to_csv(run_dir / 'validation_manifest.csv', index=False)
    test_rows.to_csv(run_dir / 'test_manifest.csv', index=False)
    configuration = {'variant': name, 'seed': SEED, 'architecture': 'natural_unweighted exact architecture', 'resampling': False, 'class_weights': class_weights, 'review_sample_weight': review_weight, 'source_balance': source_balance, 'loss': 'sparse_categorical_crossentropy', 'batch_size': BATCH_SIZE, 'maximum_epochs': 1 if quick else MAX_EPOCHS, 'base_train_rows': len(train), 'review_rows': len(reviews), 'train_rows': len(train_rows), 'train_counts': label_counts(train_rows), 'review_counts': label_counts(reviews), 'validation_rows': len(validation_rows), 'test_rows': len(test_rows), 'review_source_video_count': int(reviews.source_video.nunique()), 'effective_review_weight_sum': float(train_rows.loc[train_rows.store == 'review', 'sample_weight'].sum()), 'review_cache_dirs': [str(path) for path in REVIEW_CACHES]}
    json_dump(run_dir / 'configuration.json', configuration)
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model = build_model()
    train_data = MixedFeatureBatches(train_rows, base_store, review_store, True, SEED)
    validation_data = MixedFeatureBatches(validation_rows, base_store, review_store, False, SEED)
    test_data = MixedFeatureBatches(test_rows, base_store, review_store, False, SEED)
    checkpoint = run_dir / 'best_model.keras'
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True, verbose=1), keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-06, verbose=1)]
    print(f'Training {name}: base={len(train)} reviews={len(reviews)} total={len(train_rows)}', flush=True)
    started = time.perf_counter()
    history_object = model.fit(train_data, validation_data=validation_data, epochs=1 if quick else MAX_EPOCHS, callbacks=callbacks, verbose=2, shuffle=False)
    training_seconds = time.perf_counter() - started
    probabilities = model.predict(test_data, verbose=0)
    y_true = test_rows.y.to_numpy(dtype=np.int32)
    y_pred = np.argmax(probabilities, axis=1)
    metrics = {'status': 'completed', 'epochs_ran': len(history_object.history.get('loss', [])), 'training_time_seconds': training_seconds, **metric_bundle(y_true, y_pred)}
    json_dump(metrics_path, metrics)
    history = pd.DataFrame(history_object.history)
    history.to_csv(run_dir / 'history.csv', index=False)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(LABELS)))
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(run_dir / 'confusion_matrix.csv')
    predictions = pd.DataFrame({'video_name': test_rows.name, 'true_label': [LABELS[i] for i in y_true], 'predicted_label': [LABELS[i] for i in y_pred], 'correct': y_true == y_pred})
    for i, label in enumerate(LABELS):
        predictions[f'prob_{label}'] = probabilities[:, i]
    predictions.to_csv(run_dir / 'predictions.csv', index=False)
    save_plots(run_dir, history, matrix)
    del model, train_data, validation_data, test_data
    keras.backend.clear_session()
    gc.collect()
    return metrics

def bootstrap_variant(output_root: Path, name: str, quick: bool) -> None:
    current = pd.read_csv(_resolve_model_artifact(study_csv_path(output_root / 'runs' / name / 'predictions.csv')))
    reference = pd.read_csv(_resolve_model_artifact(study_csv_path(REFERENCE_RUN / 'predictions.csv')))
    if current.video_name.tolist() != reference.video_name.tolist():
        if quick:
            reference = reference.set_index('video_name').loc[current.video_name].reset_index()
        else:
            raise ValueError('Test prediction order differs from natural_unweighted reference')
    y_true = np.array([LABEL_TO_ID[x] for x in current.true_label])
    y_pred = np.array([LABEL_TO_ID[x] for x in current.predicted_label])
    y_ref = np.array([LABEL_TO_ID[x] for x in reference.predicted_label])
    rng = np.random.default_rng(SEED + 7000)
    replicates = 100 if quick else BOOTSTRAP_REPLICATES
    rows = []
    for metric in ('accuracy', 'f1_macro', 'f1_weighted'):
        values, differences = ([], [])
        for _ in range(replicates):
            indices = rng.integers(0, len(y_true), size=len(y_true))
            value = metric_bundle(y_true[indices], y_pred[indices])[metric]
            ref_value = metric_bundle(y_true[indices], y_ref[indices])[metric]
            values.append(value)
            differences.append(value - ref_value)
        low, high = np.percentile(values, [2.5, 97.5])
        diff_low, diff_high = np.percentile(differences, [2.5, 97.5])
        p_value = min(1.0, 2 * min(np.mean(np.asarray(differences) <= 0), np.mean(np.asarray(differences) >= 0)))
        rows.append({'variant': name, 'metric': metric, 'bootstrap_replicates': replicates, 'ci95_low': low, 'ci95_high': high, 'paired_difference_vs_natural_unweighted': float(np.mean(differences)), 'paired_difference_ci95_low': diff_low, 'paired_difference_ci95_high': diff_high, 'paired_two_sided_p': p_value})
    pd.DataFrame(rows).to_csv(output_root / 'runs' / name / 'bootstrap_vs_reference.csv', index=False)

def write_summary(output_root: Path, variant_names: list[str]) -> None:
    rows = []
    reference_metrics = json.loads(_resolve_model_artifact(study_csv_path(REFERENCE_RUN / 'metrics.json')).read_text(encoding='utf-8'))
    rows.append({'experiment': 'natural_unweighted_reference', 'review_rows': 0, 'accuracy': reference_metrics['accuracy'], 'f1_macro': reference_metrics['f1_macro'], 'f1_weighted': reference_metrics['f1_weighted'], 'epochs_ran': reference_metrics['epochs_ran'], 'training_time_seconds': reference_metrics['training_time_seconds'], 'status': 'historical_reference'})
    for name in variant_names:
        metrics_path = output_root / 'runs' / name / 'metrics.json'
        config_path = output_root / 'runs' / name / 'configuration.json'
        if not _resolve_model_artifact(study_csv_path(metrics_path)).exists():
            rows.append({'experiment': name, 'status': 'not_run'})
            continue
        metrics = json.loads(_resolve_model_artifact(study_csv_path(metrics_path)).read_text(encoding='utf-8'))
        config = json.loads(_resolve_model_artifact(study_csv_path(config_path)).read_text(encoding='utf-8'))
        rows.append({'experiment': name, 'review_rows': config['review_rows'], 'accuracy': metrics['accuracy'], 'f1_macro': metrics['f1_macro'], 'f1_weighted': metrics['f1_weighted'], 'epochs_ran': metrics['epochs_ran'], 'training_time_seconds': metrics['training_time_seconds'], 'status': metrics['status']})
    pd.DataFrame(rows).to_csv(output_root / 'comparison_summary.csv', index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--base-distribution', choices=('natural', 'balanced_reference'), default='natural')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve() if args.output_root else HERE / ('smoke' if args.quick else 'full')
    output_root.mkdir(parents=True, exist_ok=True)
    train, validation, test, review_variants = prepare_manifests(output_root, args.quick, args.base_distribution)
    if args.audit_only:
        print(f'Audit completed in {output_root}')
        return
    base_store = load_base_store()
    requested_ids = set()
    for name in args.variants:
        selection = VARIANT_SPECS[name]['selection']
        requested_ids.update(review_variants[f'{selection}_reviews'].dataset_id.astype(str))
    all_reviews = load_reviews()
    review_union = all_reviews[all_reviews.dataset_id.astype(str).isin(requested_ids)].copy()
    review_union = review_union.sort_values('dataset_id').reset_index(drop=True)
    review_store, build_audit = build_review_store(output_root, review_union)
    json_dump(output_root / 'feature_store_audit.json', build_audit)
    for name in args.variants:
        spec = VARIANT_SPECS[name]
        reviews = review_variants[f"{spec['selection']}_reviews"]
        metrics = train_variant(name, output_root, train, validation, test, reviews, base_store, review_store, args.quick, args.force, review_weight=float(spec['review_weight']), use_class_weights=bool(spec['class_weights']), source_balance=bool(spec['source_balance']))
        print(f"{name}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['f1_macro']:.4f}", flush=True)
        bootstrap_variant(output_root, name, args.quick)
        write_summary(output_root, args.variants)
    write_summary(output_root, args.variants)
    print(f'Completed outputs in {output_root}', flush=True)
if __name__ == '__main__':
    main()
