"""Configuration for Two-Stream Network model."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
IMG_SIZE = 224
NUM_FLOW_FRAMES = 10
NUM_RGB_FRAMES = 1
NUM_CLASSES = 3
BACKBONE = 'resnet50'
FUSION_TYPE = 'late'
PRETRAINED = True
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.5
SPATIAL_LR = 0.001
TEMPORAL_LR = 0.001
COMBINED_LR = 0.0001
ENABLE_AUGMENTATION = True
AUGMENTATION_PROB = 0.5
BRIGHTNESS_FACTOR = 0.2
CONTRAST_FACTOR = 0.2
SATURATION_FACTOR = 0.2
FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERATIONS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2
FLOW_MAX = 20
FLOW_NORMALIZE = True
TRAIN_STAGES = [{'name': 'stage1_spatial', 'duration_epochs': 10, 'train_spatial': True, 'train_temporal': False, 'learning_rate': 0.001}, {'name': 'stage2_temporal', 'duration_epochs': 10, 'train_spatial': False, 'train_temporal': True, 'learning_rate': 0.001}, {'name': 'stage3_joint', 'duration_epochs': 30, 'train_spatial': True, 'train_temporal': True, 'learning_rate': 0.0001}]
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
MODEL_NAME = 'two_stream'
VALIDATION_SPLIT = 0.2
SHUFFLE_DATA = True
NUM_WORKERS = 4
DEVICE = 'cuda'
USE_CLASS_WEIGHTS = True
CLASS_WEIGHTS = None
LOSS_TYPE = 'cross_entropy'
LABEL_SMOOTHING = 0.1
