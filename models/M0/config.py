"""Configuración de datos actuales y entrenamientos auxiliares RGB."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os
from models.common.legacy_training import DATASET, auxiliary_directory
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(CONFIG_DIR, '../../data/videos/clips_224')
TRAIN_CSV = str(DATASET / 'train.csv')
VALIDATION_CSV = str(DATASET / 'validation.csv')
TEST_CSV = str(DATASET / 'test.csv')
OUTPUT_DIR = str(_delivery_root / 'models/M0/runs/auxiliary')
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PROCESSED_DATA_DIR = os.path.join(CONFIG_DIR, 'all_videos_features')
TRAIN_PROCESSED_DIR = os.path.join(CONFIG_DIR, 'train_processed_data')
TEST_PROCESSED_DIR = os.path.join(CONFIG_DIR, 'test_processed_data')
FEATURE_CACHE_DIR = str(_delivery_root / '.runtime' / 'cache' / 'features')
IMG_SIZE = 224
MAX_SEQ_LENGTH = 200
NUM_FEATURES = 2048
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-05
VALIDATION_SPLIT = 0.3
OPTIMIZER_NAME = 'adam'
FEATURE_EXTRACTOR = 'resnet50'
SEQUENCE_MODEL = 'lstm'
RUN_ALL_MODELS = False
MODEL_VARIANTS = ['gru_small', 'gru_big', 'lstm']
PLAY_CATEGORIES = ['scrum', 'line', 'other']
CLASSES_TO_USE = ['line', 'scrum', 'other']
