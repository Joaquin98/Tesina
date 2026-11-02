"""Particiones y carga de datos para la comparación de geometría.

Las particiones se leen de los CSV actuales y se balancea solamente el entrenamiento. El conjunto de prueba permanece fijo y
no participa en decisiones de selección.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json
from pathlib import Path
import numpy as np
import pandas as pd
from models.common.legacy_training import current_frames, cached_rgb, cached_geometry
from models.M1.config import DATA_DIR, GEOMETRY_LENGTH, LABELS, LABEL_TO_ID, NATURAL_TRAIN_CSV, RGB_CACHE_DIR, RGB_DIM, RGB_LENGTH, POSE_CACHE_DIR, TEST_CSV, TRAIN_TARGET_PER_CLASS, VALIDATION_FRACTION

def normalize_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve_model_artifact(path))
    required = {'video_name', 'tag'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} no contiene las columnas requeridas: {sorted(missing)}')
    out = df[['video_name', 'tag']].copy()
    out['video_name'] = out['video_name'].astype(str).str.strip()
    out['tag'] = out['tag'].astype(str).str.strip().str.lower()
    out = out[(out['video_name'] != '') & out['tag'].isin(LABELS)]
    return out.reset_index(drop=True)

def cache_path(video_name: str) -> Path:
    return RGB_CACHE_DIR / f'{Path(video_name).stem}.npz'

def pose_cache_path(video_name: str, pose_model: str) -> Path:
    safe = pose_model.replace('/', '_').replace('\\', '_').replace('.', '_')
    return POSE_CACHE_DIR / safe / f'{Path(video_name).stem}.npz'

def load_rgb_features(video_name: str) -> tuple[np.ndarray, np.ndarray]:
    return cached_rgb(video_name)

def load_geometry_features(video_name: str, pose_model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return cached_geometry(video_name, pose_model, GEOMETRY_LENGTH)

def _counts(df: pd.DataFrame) -> dict[str, int]:
    return {label: int((df['tag'] == label).sum()) for label in LABELS}

def _unique(df: pd.DataFrame) -> pd.DataFrame:
    conflicts = df.groupby('video_name')['tag'].nunique()
    if (conflicts > 1).any():
        names = conflicts[conflicts > 1].index.tolist()[:5]
        raise ValueError(f'Clips con etiquetas contradictorias: {names}')
    return df.drop_duplicates('video_name', keep='first').reset_index(drop=True)

def _balance_train(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chunks: list[pd.DataFrame] = []
    for label in LABELS:
        group = df[df['tag'] == label].reset_index(drop=True)
        if len(group) == 0:
            raise ValueError(f'No hay clips de clase {label} para balancear')
        if len(group) >= TRAIN_TARGET_PER_CLASS:
            indices = rng.choice(len(group), TRAIN_TARGET_PER_CLASS, replace=False)
            chunk = group.iloc[indices].copy()
        else:
            extra = rng.choice(len(group), TRAIN_TARGET_PER_CLASS - len(group), replace=True)
            chunk = pd.concat([group, group.iloc[extra]], ignore_index=True)
        chunks.append(chunk)
    out = pd.concat(chunks, ignore_index=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)

def build_partition(seed: int, require_geometry: bool=True, pose_model: str='yolo11n-pose.pt') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    natural, validation, test = current_frames()
    train = natural
    train = _balance_train(train.reset_index(drop=True), seed)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_names = set(train['video_name'])
    validation_names = set(validation['video_name'])
    test_names = set(test['video_name'])
    overlap = {'train_validation': sorted(train_names & validation_names), 'train_test': sorted(train_names & test_names), 'validation_test': sorted(validation_names & test_names)}
    if any(overlap.values()):
        raise AssertionError(f'La partición tiene solapamiento: {overlap}')
    info = {'seed': int(seed), 'protocol': 'saved_disjoint_partitions', 'train_rows': int(len(train)), 'validation_rows': int(len(validation)), 'test_rows': int(len(test)), 'train_unique_clips': int(train['video_name'].nunique()), 'validation_unique_clips': int(validation['video_name'].nunique()), 'test_unique_clips': int(test['video_name'].nunique()), 'train_counts': _counts(train), 'validation_counts': _counts(validation), 'test_counts': _counts(test), 'overlap': {key: len(value) for key, value in overlap.items()}, 'geometry_required': bool(require_geometry), 'pose_model': pose_model}
    return (train, validation, test, info)

def save_partition(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, info: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / 'train.csv', index=False)
    validation.to_csv(output_dir / 'validation.csv', index=False)
    test.to_csv(output_dir / 'test.csv', index=False)
    (output_dir / 'split_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

def load_arrays(df: pd.DataFrame, pose_model: str, mode: str) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """Carga matrices alineadas. ``mode`` es uno de m0..m5."""
    rgb_values: list[np.ndarray] = []
    rgb_masks: list[np.ndarray] = []
    position_values: list[np.ndarray] = []
    pose_values: list[np.ndarray] = []
    geometry_masks: list[np.ndarray] = []
    labels: list[int] = []
    kept_rows: list[dict] = []
    needs_rgb = mode in {'m0', 'm3', 'm4', 'm5'}
    needs_position = mode in {'m1', 'm3', 'm5'}
    needs_pose = mode in {'m2', 'm4', 'm5'}
    for row in df.to_dict('records'):
        name = str(row['video_name'])
        try:
            if needs_rgb:
                rgb, rgb_mask = load_rgb_features(name)
            if needs_position or needs_pose:
                positions, poses, geometry_mask = load_geometry_features(name, pose_model)
        except (FileNotFoundError, OSError, ValueError, KeyError):
            continue
        if needs_rgb:
            rgb_values.append(rgb)
            rgb_masks.append(rgb_mask)
        if needs_position:
            position_values.append(positions)
        if needs_pose:
            pose_values.append(poses)
        if needs_position or needs_pose:
            geometry_masks.append(geometry_mask)
        labels.append(LABEL_TO_ID[str(row['tag'])])
        kept_rows.append(row)
    if not kept_rows:
        raise RuntimeError(f'No se pudieron cargar datos para {mode}')
    arrays: dict[str, np.ndarray] = {'labels': np.asarray(labels, dtype=np.int32)}
    if needs_rgb:
        arrays['rgb'] = np.asarray(rgb_values, dtype=np.float32)
        arrays['rgb_mask'] = np.asarray(rgb_masks, dtype=bool)
    if needs_position:
        arrays['positions'] = np.asarray(position_values, dtype=np.float32)
    if needs_pose:
        arrays['poses'] = np.asarray(pose_values, dtype=np.float32)
    if needs_position or needs_pose:
        arrays['geometry_mask'] = np.asarray(geometry_masks, dtype=bool)
    return (arrays, arrays['labels'], pd.DataFrame(kept_rows))
