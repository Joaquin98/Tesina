"""Configuración central del experimento de geometría de jugadores.

El módulo no modifica la configuración histórica de ``models/M0``.
Todos los artefactos nuevos se escriben debajo de este experimento.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / 'models' / 'M0'
DATA_DIR = ROOT / 'data' / 'videos' / 'clips_224'
NATURAL_TRAIN_CSV = ROOT / 'data/clips_dataset/train.csv'
VALIDATION_CSV = ROOT / 'data/clips_dataset/validation.csv'
HISTORICAL_BALANCED_TRAIN_CSV = ROOT / 'data/clips_dataset/balanced_train.csv'
TEST_CSV = ROOT / 'data/clips_dataset/test.csv'
RGB_CACHE_DIR = ROOT / '.runtime/cache/features'
EXPERIMENT_DIR = experiment_directory(__file__)
_geometry_length_env = os.environ.get('PLAYER_GEOMETRY_LENGTH', '32')
try:
    GEOMETRY_LENGTH = int(_geometry_length_env)
except ValueError as exc:
    raise ValueError('PLAYER_GEOMETRY_LENGTH must be an integer') from exc
if GEOMETRY_LENGTH <= 0:
    raise ValueError('PLAYER_GEOMETRY_LENGTH must be positive')
_pose_cache_root_env = os.environ.get('PLAYER_GEOMETRY_POSE_CACHE_ROOT')
if _pose_cache_root_env:
    POSE_CACHE_DIR = Path(_pose_cache_root_env)
elif GEOMETRY_LENGTH == 32:
    POSE_CACHE_DIR = ROOT / '.runtime/cache/pose_cache'
else:
    POSE_CACHE_DIR = ROOT / '.runtime/cache/pose_cache' / f'frames_{GEOMETRY_LENGTH}'
from models.common.legacy_training import auxiliary_directory
RESULTS_DIR = auxiliary_directory('geometry', 'M1') / 'results'
PARTITIONS_DIR = auxiliary_directory('geometry', 'M1') / 'partitions'
AUDIT_DIR = auxiliary_directory('geometry', 'M1') / 'audit'
LABELS = ('line', 'other', 'scrum')
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
RGB_LENGTH = 200
RGB_DIM = 2048
MAX_PEOPLE = 30
TRAIN_TARGET_PER_CLASS = 420
VALIDATION_FRACTION = 0.3
DEFAULT_SEEDS = (17, 42, 2026)
DEFAULT_MODELS = ('m0', 'm1', 'm2', 'm3', 'm4', 'm5')
POSE_MODEL_DEFAULT = 'yolo11n-pose.pt'
POSE_FRAME_COUNT_DEFAULT = GEOMETRY_LENGTH
POSE_CONFIDENCE_DEFAULT = 0.25
POSE_KEYPOINT_CONFIDENCE_DEFAULT = 0.2
POSE_IMAGE_SIZE_DEFAULT = 640
