"""Unified manifests and cache readers for historical clips and final matches."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from .protocol import CACHE_SIGNATURE, LABEL_TO_ID, RunSpec
ROOT = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
OUTPUTS = HERE / 'outputs'
MASTER_MANIFEST = OUTPUTS / 'master_examples.csv'
FINAL_MANIFEST = ROOT / 'data/final_tests_feature_cache_20260826/outputs/final_test_feature_manifest.csv'
HISTORICAL_MANIFEST = ROOT / 'data/independent_evaluation/outputs/nested_source_manifest.csv'
HISTORICAL_RESNET = ROOT / 'models/M0/feature_cache/resnet50'
HISTORICAL_YOLO = ROOT / 'data/player_geometry_highres_pose/outputs/cache/yolo11n-pose_pt'
HISTORICAL_FUSION = ROOT / 'data/joint_multimodal_20260813/rgb_dynamics_seed42/fusion_features.npz'
HISTORICAL_DESCRIPTORS = ROOT / 'data/player_geometry_highres_pose/outputs/quantile_100frames/e04_e06_highres_100frames.npy'
HISTORICAL_PAIRED = ROOT / 'data/player_geometry_highres_pose/outputs/quantile_100frames/paired_predictions.npz'
HISTORICAL_TEMPORAL = ROOT / 'data/player_geometry_highres_pose/outputs/temporal_attention_sequence_cache_20260814/features_e04_e06_quality.npy'
HISTORICAL_PRIMITIVES = ROOT / 'data/player_geometry_highres_pose/outputs/cached_descriptor_grid_20260814/primitive_cache/current'
DYNAMIC_PCA = ROOT / 'data/joint_multimodal_20260813/rgb_dynamics_seed42/dynamic_pca.joblib'
PREPARED_FINAL = OUTPUTS / 'prepared_final'
M1_FINAL_POSE32 = OUTPUTS / 'm1_pose32'
M1_HISTORICAL_POSE32 = ROOT / 'data/player_geometry_fusion/pose_cache/yolo11n-pose_pt'

def historical_outer_fold_map() -> dict[str, str]:
    paths = sorted(_model_artifact_glob(ROOT / 'data/alternative_split_robustness', 'final_fold_*_stream/outer_*/evaluation_predictions.csv', recursive=False))
    if len(paths) != 5:
        raise RuntimeError(f'Expected five historical outer folds, found {len(paths)}')
    result = {}
    for index, path in enumerate(paths, 1):
        names = pd.read_csv(_resolve_model_artifact(path), usecols=['video_name'])['video_name'].astype(str)
        for name in names:
            if name in result:
                raise ValueError(f'Historical clip appears in two outer folds: {name}')
            result[name] = f'G{index:02d}'
    return result

def build_master_manifest(*, write: bool=True) -> pd.DataFrame:
    outer = historical_outer_fold_map()
    historical = pd.read_csv(_resolve_model_artifact(HISTORICAL_MANIFEST))
    historical['video_name'] = historical['video_name'].astype(str)
    if set(historical['video_name']) != set(outer):
        missing = set(historical['video_name']) - set(outer)
        extra = set(outer) - set(historical['video_name'])
        raise ValueError(f'Historical outer-fold coverage mismatch: missing={len(missing)} extra={len(extra)}')
    rows = []
    for row in historical.to_dict('records'):
        name = str(row['video_name'])
        stem = Path(name).stem
        resnet = HISTORICAL_RESNET / f'{stem}.npz'
        yolo = HISTORICAL_YOLO / f'{stem}.npz'
        m1_pose32 = M1_HISTORICAL_POSE32 / f'{stem}.npz'
        rows.append({'block_id': f'CH::{stem}', 'domain': 'CH', 'group_id': outer[name], 'source_group': str(row['source_group']), 'section': stem, 'block_index': 0, 'start_s': 0.0, 'end_s': 7.0, 'duration': 7.0, 'label': str(row['tag']), 'resnet_path': str(resnet.resolve()), 'yolo_path': str(yolo.resolve()), 'resnet_status': 'ready' if _resolve_model_artifact(resnet).is_file() else 'missing', 'yolo_status': 'ready' if _resolve_model_artifact(yolo).is_file() else 'missing', 'm1_pose32_path': str(m1_pose32.resolve()), 'm1_pose32_status': 'ready' if _resolve_model_artifact(m1_pose32).is_file() else 'missing', 'cache_signature': 'historical_highres_100frames', 'video_name': name})
    with _resolve_model_artifact(FINAL_MANIFEST).open(encoding='utf-8-sig', newline='') as handle:
        final_rows = list(csv.DictReader(handle))
    for row in final_rows:
        m1_pose32 = M1_FINAL_POSE32 / f"{row['dataset_id']}.npz"
        rows.append({'block_id': str(row['dataset_id']), 'domain': 'CV', 'group_id': str(row['match']), 'source_group': str(row['match']), 'section': str(row['section']), 'block_index': int(row['block_index']), 'start_s': float(row['start_time']), 'end_s': float(row['end_time']), 'duration': float(row['duration']), 'label': str(row['true_label']), 'resnet_path': str(row['resnet_cache_path']), 'yolo_path': str(row['yolo_cache_path']), 'resnet_status': str(row['resnet_status']), 'yolo_status': str(row['yolo_status']), 'm1_pose32_path': str(m1_pose32.resolve()), 'm1_pose32_status': 'ready' if _resolve_model_artifact(m1_pose32).is_file() else 'pending', 'cache_signature': str(row['protocol_signature']), 'video_name': str(row['dataset_id'])})
    frame = pd.DataFrame(rows)
    if frame['block_id'].duplicated().any():
        raise ValueError('Master manifest contains duplicate block_id values')
    if set(frame['label']) != set(LABEL_TO_ID):
        raise ValueError(f"Unexpected labels: {sorted(set(frame['label']))}")
    if write:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        frame.to_csv(MASTER_MANIFEST, index=False)
    return frame

def load_master_manifest(*, refresh: bool=False) -> pd.DataFrame:
    if refresh or not _resolve_model_artifact(MASTER_MANIFEST).is_file():
        return build_master_manifest(write=True)
    return pd.read_csv(_resolve_model_artifact(MASTER_MANIFEST))

def split_for_run(frame: pd.DataFrame, spec: RunSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if spec.panel == 'PV':
        development = frame[(frame.domain == 'CV') & (frame.group_id == spec.development_group)]
        test = frame[(frame.domain == 'CV') & (frame.group_id == spec.test_group)]
        if spec.data_regime == 'D0':
            train = frame[frame.domain == 'CH']
        elif spec.data_regime == 'D1':
            train = frame[(frame.domain == 'CV') & frame.group_id.isin(spec.train_groups)]
        elif spec.data_regime == 'D2':
            train = frame[(frame.domain == 'CH') | (frame.domain == 'CV') & frame.group_id.isin(spec.train_groups)]
        else:
            train = frame.iloc[0:0]
    elif spec.panel == 'PH':
        development = frame[(frame.domain == 'CH') & (frame.group_id == spec.development_group)]
        test = frame[(frame.domain == 'CH') & (frame.group_id == spec.test_group)]
        if spec.data_regime == 'D0':
            train = frame[(frame.domain == 'CH') & frame.group_id.isin(spec.train_groups)]
        elif spec.data_regime == 'D1':
            train = frame[frame.domain == 'CV']
        elif spec.data_regime == 'D2':
            train = frame[(frame.domain == 'CH') & frame.group_id.isin(spec.train_groups) | (frame.domain == 'CV')]
        else:
            train = frame.iloc[0:0]
    else:
        raise ValueError(f'Unknown panel: {spec.panel}')
    sets = [set(part.block_id.astype(str)) for part in (train, development, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError(f'Data leakage in {spec.run_id}')
    return tuple((part.reset_index(drop=True) for part in (train, development, test)))

def assert_cache_ready(frame: pd.DataFrame) -> None:
    bad = frame[~frame.resnet_status.astype(str).str.startswith('ready') | (frame.yolo_status.astype(str) != 'ready')]
    missing = frame[~frame.resnet_path.map(lambda value: _resolve_model_artifact(Path(str(value))).is_file()) | ~frame.yolo_path.map(lambda value: _resolve_model_artifact(Path(str(value))).is_file())]
    if len(bad) or len(missing):
        raise RuntimeError(f'Cache is not ready for requested rows: statuses={len(bad)} missing_paths={len(missing)}')

def labels_for(frame: pd.DataFrame) -> np.ndarray:
    return frame.label.map(LABEL_TO_ID).to_numpy(np.int64)

@lru_cache(maxsize=8192)
def _load_resnet_path(path_value: str, block_id: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load an immutable cached sequence once per experiment process.

    A training recipe revisits the same frozen ResNet tensors on every epoch.
    Keeping the decoded arrays in an LRU cache avoids repeated NPZ
    decompression without changing their values or the training protocol.
    """
    with _load_model_result_array(path_value, allow_pickle=False) as data:
        if 'features' in data.files:
            features = data['features'].astype(np.float32)
            mask = data['mask'].astype(bool)
            length = int(data['length'])
        else:
            features = data['frame_features'].astype(np.float32)
            mask = data['frame_masks'].astype(bool)
            length = int(mask.sum())
    if features.shape != (200, 2048) or mask.shape != (200,):
        raise ValueError(f'Invalid ResNet cache for {block_id}: {features.shape}, {mask.shape}')
    features.setflags(write=False)
    mask.setflags(write=False)
    return (features, mask, length)

def load_resnet(row) -> tuple[np.ndarray, np.ndarray, int]:
    return _load_resnet_path(str(row.resnet_path), str(row.block_id))

def load_yolo(row) -> dict[str, np.ndarray]:
    with _load_model_result_array(str(row.yolo_path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}

def load_m1_pose32(row) -> dict[str, np.ndarray]:
    with _load_model_result_array(str(row.m1_pose32_path), allow_pickle=False) as data:
        values = {key: np.asarray(data[key]) for key in data.files}
    required = {'position_features': (32, 31), 'pose_features': (32, 30), 'frame_mask': (32,)}
    for key, shape in required.items():
        if key not in values or values[key].shape != shape:
            raise ValueError(f'Invalid M1 32-frame cache for {row.block_id}: {key}')
    return values

def align_sequence(sequence: np.ndarray, length: int=200) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) == length:
        return sequence
    old = np.linspace(0.0, 1.0, len(sequence), dtype=np.float32)
    new = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.stack([np.interp(new, old, sequence[:, index]) for index in range(sequence.shape[1])], axis=1).astype(np.float32)

def geometry_sequence(raw: dict[str, np.ndarray], length: int=200) -> np.ndarray:
    values = np.concatenate([raw['position_features'], raw['pose_features']], axis=1).astype(np.float32)
    mask = align_sequence(raw['frame_mask'].astype(np.float32)[:, None], length)
    return np.concatenate([align_sequence(values, length), mask], axis=1)

def geometry_statistics(raw: dict[str, np.ndarray]) -> np.ndarray:
    sequence = np.concatenate([raw['position_features'], raw['pose_features']], axis=1).astype(np.float32)
    mask = raw['frame_mask'].astype(bool)
    valid = sequence[mask] if mask.any() else sequence
    delta = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, 61), np.float32)
    return np.concatenate([valid.mean(0), valid.std(0), valid.min(0), valid.max(0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), delta.mean(0), delta.std(0), np.abs(delta).mean(0)]).astype(np.float32)

def dynamic_components(features: np.ndarray, length: int) -> dict[str, np.ndarray]:
    rgb = align_sequence(np.asarray(features[:length], dtype=np.float32), 200)
    segments = rgb.reshape(10, 20, 2048)
    return {'segment': segments.mean(axis=1), 'delta': np.abs(np.diff(segments, axis=1)).mean(axis=1), 'mean': rgb.mean(axis=0, keepdims=True), 'std': rgb.std(axis=0, keepdims=True)}

def transform_dynamic(components: dict[str, np.ndarray], pca: dict) -> np.ndarray:
    return np.concatenate([pca['segment'].transform(components['segment']).reshape(-1), pca['delta'].transform(components['delta']).reshape(-1), pca['mean'].transform(components['mean']).reshape(-1), pca['std'].transform(components['std']).reshape(-1)]).astype(np.float32)

def fit_dynamic_pca(frame: pd.DataFrame, *, output: Path, components: int=32, sample_batch: int=128) -> dict[str, IncrementalPCA]:
    """Fit the four dynamic PCAs from training-only ResNet caches.

    The implementation is incremental so a D2 fold does not materialize the
    full 10 x 2048 segment matrix in memory.  A small tail is retained until it
    contains enough rows for a valid ``partial_fit`` call.  Every ResNet cache
    is loaded once per training batch and feeds all four estimators; fitting
    the estimators in separate passes would repeat the dominant cache loading
    and component extraction work four times without changing their samples.
    """
    keys = ('segment', 'delta', 'mean', 'std')
    result = {key: IncrementalPCA(n_components=components, batch_size=sample_batch) for key in keys}
    pending = {key: np.empty((0, 2048), dtype=np.float32) for key in keys}
    target = max(components, sample_batch)
    for start in range(0, len(frame), sample_batch):
        rows = {key: [] for key in keys}
        for row in frame.iloc[start:start + sample_batch].itertuples(index=False):
            features, _mask, length = load_resnet(row)
            values = dynamic_components(features, length)
            for key in keys:
                rows[key].append(values[key])
        for key in keys:
            pending[key] = np.concatenate([pending[key], *rows[key]], axis=0)
            while len(pending[key]) >= target + components:
                result[key].partial_fit(pending[key][:target])
                pending[key] = pending[key][target:]
    for key in keys:
        if len(pending[key]) < components:
            raise ValueError(f"Not enough training observations for dynamic PCA '{key}': {len(pending[key])}")
        result[key].partial_fit(pending[key])
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, output)
    return result

def dynamic_matrix(frame: pd.DataFrame, pca: dict | None=None) -> np.ndarray:
    if pca is None:
        return compact_matrix(frame, ('dynamic',))
    rows = []
    for row in frame.itertuples(index=False):
        features, _mask, length = load_resnet(row)
        rows.append(transform_dynamic(dynamic_components(features, length), pca))
    return np.asarray(rows, dtype=np.float32)

def _temporal_helpers():
    player_dir = ROOT / 'data/player_geometry_fusion'
    saved = sys.modules.pop('config', None)
    sys.path.insert(0, str(source_directory(player_dir)))
    try:
        from models.M2.grid_search_cached_descriptors import primitive_base
        from models.M3.run_temporal_attention_experiment import quality_score, smooth_three, usable_mask
    finally:
        sys.path.remove(str(source_directory(player_dir)))
        if saved is not None:
            sys.modules['config'] = saved
    return (primitive_base, quality_score, smooth_three, usable_mask)

def temporal_from_raw(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primitive_base, quality_score, smooth_three, usable_mask = _temporal_helpers()
    primitive = primitive_base(raw)
    frame_mask = primitive['mask'].astype(bool)
    quality = primitive['quality'].astype(np.float32)
    current = smooth_three(primitive['current'][None], frame_mask[None])[0]
    rugby = smooth_three(primitive['rugby'][None], frame_mask[None])[0]
    features = np.nan_to_num(np.concatenate([current, quality, rugby], axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    score = quality_score(quality[None])[0].astype(np.float32)
    usable = usable_mask(frame_mask[None], score[None])[0].astype(bool)
    return (features, usable, score)

@lru_cache(maxsize=1)
def _historical_indexes():
    with _load_model_result_array(HISTORICAL_FUSION, allow_pickle=True) as data:
        dynamic_names = data['names'].astype(str)
    with _load_model_result_array(HISTORICAL_PAIRED, allow_pickle=True) as data:
        paired_names = data['names'].astype(str)
    return ({name: index for index, name in enumerate(dynamic_names)}, {name: index for index, name in enumerate(paired_names)})

@lru_cache(maxsize=1)
def _historical_arrays():
    with _load_model_result_array(HISTORICAL_FUSION, allow_pickle=True) as data:
        dynamic = data['features'][:, :704].astype(np.float32)
    descriptors = _load_model_result_array(HISTORICAL_DESCRIPTORS, mmap_mode='r')
    temporal = _load_model_result_array(HISTORICAL_TEMPORAL, mmap_mode='r')
    mask = _load_model_result_array(HISTORICAL_PRIMITIVES / 'mask.npy', mmap_mode='r')
    quality = _load_model_result_array(HISTORICAL_PRIMITIVES / 'quality.npy', mmap_mode='r')
    _primitive, quality_score, _smooth, usable_mask = _temporal_helpers()
    score = quality_score(np.asarray(quality)).astype(np.float32)
    usable = usable_mask(np.asarray(mask, dtype=bool), score).astype(bool)
    return (dynamic, descriptors, temporal, usable, score)

def prepared_final_path(block_id: str) -> Path:
    return PREPARED_FINAL / f'{block_id}.npz'

def prepare_final_record(row, *, overwrite: bool=False) -> Path:
    path = prepared_final_path(str(row.block_id))
    if _resolve_model_artifact(path).is_file() and (not overwrite):
        return path
    raw = load_yolo(row)
    features, _mask, length = load_resnet(row)
    pca = joblib.load(_resolve_model_artifact(DYNAMIC_PCA))
    dynamic = transform_dynamic(dynamic_components(features, length), pca)
    descriptor = raw.get('descriptor')
    if descriptor is None:
        raise ValueError(f'Final YOLO cache lacks descriptor: {row.block_id}')
    temporal, temporal_mask, quality = temporal_from_raw(raw)
    geometry = geometry_statistics(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dynamic=dynamic.astype(np.float32), descriptor=np.asarray(descriptor, dtype=np.float32), geometry=geometry, temporal=temporal, temporal_mask=temporal_mask, quality=quality, cache_signature=np.asarray(CACHE_SIGNATURE))
    return path

def compact_features(row, keys: tuple[str, ...] | None=None) -> dict[str, np.ndarray]:
    if str(row.domain) == 'CH':
        dynamic_index, paired_index = _historical_indexes()
        dynamic, descriptors, temporal, usable, score = _historical_arrays()
        name = str(row.video_name)
        values = {'dynamic': np.asarray(dynamic[dynamic_index[name]], dtype=np.float32), 'descriptor': np.asarray(descriptors[paired_index[name]], dtype=np.float32), 'temporal': np.asarray(temporal[paired_index[name]], dtype=np.float32), 'temporal_mask': np.asarray(usable[paired_index[name]], dtype=bool), 'quality': np.asarray(score[paired_index[name]], dtype=np.float32)}
        if keys is None or 'geometry' in keys:
            values['geometry'] = geometry_statistics(load_yolo(row))
        return values if keys is None else {key: values[key] for key in keys}
    path = prepare_final_record(row)
    with _load_model_result_array(path, allow_pickle=False) as data:
        selected = [key for key in data.files if key != 'cache_signature' and (keys is None or key in keys)]
        return {key: np.asarray(data[key]) for key in selected}

def compact_matrix(frame: pd.DataFrame, keys: tuple[str, ...]) -> np.ndarray:
    rows = []
    for row in frame.itertuples(index=False):
        values = compact_features(row, keys)
        rows.append(np.concatenate([np.asarray(values[key]).reshape(-1) for key in keys]))
    return np.asarray(rows, dtype=np.float32)

def preflight_report(frame: pd.DataFrame) -> dict:
    ready = frame.resnet_status.astype(str).str.startswith('ready') & (frame.yolo_status.astype(str) == 'ready')
    return {'rows': int(len(frame)), 'historical_rows': int((frame.domain == 'CH').sum()), 'final_rows': int((frame.domain == 'CV').sum()), 'historical_groups': sorted(frame.loc[frame.domain == 'CH', 'group_id'].unique()), 'final_groups': sorted(frame.loc[frame.domain == 'CV', 'group_id'].unique()), 'ready_rows': int(ready.sum()), 'pending_rows': int((~ready).sum()), 'labels': frame.label.value_counts().to_dict(), 'cache_signature': CACHE_SIGNATURE}
