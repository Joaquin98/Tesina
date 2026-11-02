"""
Two-Stream Network — Keras training script (true Simonyan-style)
================================================================
Architecture
------------
  Spatial stream  : sample 1 RGB frame per clip
                    → MobileNetV2 (ImageNet pretrained, top layers fine-tuned)
                    → GlobalAveragePooling2D
                    → Dense(256, relu) → Dropout → Dense(3, softmax)

  Temporal stream : stack 10 Farneback optical flow fields (u, v) → 20 channels
                    → MobileNetV2 with first Conv2D replaced (3→20 channels)
                      weights averaged from pretrained RGB conv, then repeated
                    → GlobalAveragePooling2D
                    → Dense(256, relu) → Dropout → Dense(3, softmax)

  Fusion          : learnable weighted average of both softmax outputs
                    alpha = sigmoid(Dense(1)([sp, fl]))
                    final = alpha * sp + (1-alpha) * fl

This follows the original Simonyan & Zisserman (2014) design:
  - Two independent CNNs trained on different modalities
  - No feature cache reuse — both streams process raw data
  - Flow cached to disk after first extraction

Usage (from models/two_stream/ directory)
-----------------------------------------
  python train.py                          # full training
  python train.py QUICK=1                  # 50 samples, 5 epochs smoke-test
  python train.py BACKBONE=mobilenetv2     # default backbone
  python train.py UNFREEZE=30              # unfreeze last N backbone layers
  python train.py LR=1e-4 EPOCHS=80

Outputs
-------
  results/two_stream_results.csv
  results/confusion_matrix_raw.png
  results/confusion_matrix_normalized.png
  results/history.png
  results/classification_report.txt
  checkpoints/two_stream_spatial.weights.h5
  checkpoints/two_stream_temporal.weights.h5
  checkpoints/two_stream_best.weights.h5
  flow_cache/<video>.npz
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import os
import sys
import csv
import json
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
BACKBONE = _env('BACKBONE', 'mobilenetv2')
IMG_SIZE = _env_int('IMG_SIZE', 224)
UNFREEZE = _env_int('UNFREEZE', 30)
TEMPORAL_UNFREEZE_FIRST = _env_int('TEMPORAL_UNFREEZE_FIRST', 20)
DENSE_UNITS = _env_int('DENSE_UNITS', 256)
DROPOUT = float(_env('DROPOUT', '0.5'))
LR = float(_env('LR', '1e-4'))
BATCH_SIZE = _env_int('BATCH_SIZE', 16)
EPOCHS = _env_int('EPOCHS', 50)
QUICK = _env_bool('QUICK')
SPLIT_SEED = _env_int('SPLIT_SEED', 42)
VAL_FRACTION = float(_env('VAL_FRACTION', '0.2'))
TRAIN_SAMPLES_PER_CLASS = _env_int('TRAIN_SAMPLES_PER_CLASS', 600)
STRICT_DATA = _env_bool('STRICT_DATA', True)
TRAIN_MODE = _env('TRAIN_MODE', 'natural_weighted').strip().lower()
USE_CLASS_WEIGHTS = _env_bool('USE_CLASS_WEIGHTS', True)
EVALUATE_TEST = _env_bool('EVALUATE_TEST', True)
REUSE_VERIFIED_IMPROVED_CACHE = _env_bool('REUSE_VERIFIED_IMPROVED_CACHE', True)
if QUICK:
    EPOCHS = min(EPOCHS, 5)
NUM_FLOW_FRAMES = _env_int('NUM_FLOW_FRAMES', 10)
FLOW_IMG_SIZE = _env_int('FLOW_IMG_SIZE', 224)
FLOW_MAX = 20.0
TARGET_FPS = float(_env('TARGET_FPS', '25.0'))
FLOW_SAMPLING = _env('FLOW_SAMPLING', 'adjacent_center').strip().lower()
FLOW_CENTERING = _env('FLOW_CENTERING', 'none').strip().lower()
PRE_CROP_SCALE = float(_env('PRE_CROP_SCALE', '0.5'))
PREPROCESS_MODE = _env('PREPROCESS_MODE', 'half_center224').strip().lower()
CLIPNORM = float(_env('CLIPNORM', '1.0'))
EXPERIMENT_NAME = _env('EXPERIMENT_NAME', 'corrected')
FLOW_CACHE_NAME = _env('FLOW_CACHE_NAME', EXPERIMENT_NAME)
if FLOW_CENTERING not in {'none', 'per_field_mean'}:
    raise ValueError(f"FLOW_CENTERING must be 'none' or 'per_field_mean', got {FLOW_CENTERING!r}")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
VIDEO_DIR = os.path.join(REPO_ROOT, 'data', 'clips_224')
FLOW_CACHE_DIR = os.path.join(BASE_DIR, 'flow_cache', FLOW_CACHE_NAME)
TRAIN_CSV = os.path.join(BASE_DIR, '..', 'M0', 'train.csv')
TEST_CSV = os.path.join(BASE_DIR, '..', 'M0', 'test.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', EXPERIMENT_NAME)
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints', EXPERIMENT_NAME)
for _d in [FLOW_CACHE_DIR, RESULTS_DIR, CKPT_DIR]:
    os.makedirs(_d, exist_ok=True)
try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
    from sklearn.utils.class_weight import compute_class_weight
    SKLEARN_OK = True
except ImportError:
    print('WARNING: scikit-learn not found -- pip install scikit-learn')
    SKLEARN_OK = False
try:
    import cv2
    CV2_OK = True
except ImportError:
    print('WARNING: OpenCV not found -- pip install opencv-python')
    CV2_OK = False
import keras
import tensorflow as tf
keras.utils.set_random_seed(SPLIT_SEED)
CLASS_NAMES = ('line', 'other', 'scrum')
LABEL_ALIASES = {'line': 'line', 'lineout': 'line', 'other': 'other', 'scrum': 'scrum'}

def canonical_tag(value):
    """Map historical label spellings to the single model vocabulary."""
    return LABEL_ALIASES.get(str(value).strip().lower())

class SparseMacroF1(keras.metrics.Metric):
    """Macro F1 for sparse integer labels, accumulated over a full epoch."""

    def __init__(self, num_classes, name='f1_macro', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = int(num_classes)
        self.confusion = self.add_weight(name='confusion', shape=(self.num_classes, self.num_classes), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        weights = None
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)
        batch_confusion = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, weights=weights, dtype=self.dtype)
        self.confusion.assign_add(batch_confusion)

    def result(self):
        true_positives = tf.linalg.diag_part(self.confusion)
        false_positives = tf.reduce_sum(self.confusion, axis=0) - true_positives
        false_negatives = tf.reduce_sum(self.confusion, axis=1) - true_positives
        per_class = tf.math.divide_no_nan(2.0 * true_positives, 2.0 * true_positives + false_positives + false_negatives)
        return tf.reduce_mean(per_class)

    def reset_state(self):
        self.confusion.assign(tf.zeros_like(self.confusion))

def normalize_manifest(df, source_name):
    """Validate and canonicalize a video manifest without changing its file."""
    required = {'video_name', 'tag'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{source_name} is missing columns: {sorted(missing)}')
    out = df[['video_name', 'tag']].copy()
    out['video_name'] = out['video_name'].astype(str).str.strip()
    out['tag'] = out['tag'].map(canonical_tag)
    unknown = int(out['tag'].isna().sum())
    if unknown:
        print(f'  [{source_name}] WARNING: dropping {unknown} rows with unknown labels')
        out = out.dropna(subset=['tag'])
    return out[out['video_name'] != ''].reset_index(drop=True)

def split_unique_stratified(df):
    """Split unique videos by class before any training oversampling."""
    if not 0.0 < VAL_FRACTION < 1.0:
        raise ValueError('VAL_FRACTION must be between 0 and 1')
    unique_df = df.drop_duplicates('video_name', keep='first')
    train_parts, val_parts = ([], [])
    for offset, class_name in enumerate(CLASS_NAMES):
        group = unique_df[unique_df['tag'] == class_name]
        if len(group) < 2:
            raise ValueError(f'Not enough unique samples for class {class_name!r}')
        val_count = max(1, int(round(len(group) * VAL_FRACTION)))
        val_part = group.sample(n=val_count, random_state=SPLIT_SEED + offset, replace=False)
        train_parts.append(group.drop(val_part.index))
        val_parts.append(val_part)
    return (pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True))

def balance_training_manifest(df):
    """Balance only training rows after the leakage-safe split."""
    if TRAIN_SAMPLES_PER_CLASS <= 0:
        raise ValueError('TRAIN_SAMPLES_PER_CLASS must be positive')
    parts = []
    for offset, class_name in enumerate(CLASS_NAMES):
        group = df[df['tag'] == class_name]
        if group.empty:
            raise ValueError(f'Training partition has no samples for {class_name!r}')
        parts.append(group.sample(n=TRAIN_SAMPLES_PER_CLASS, replace=len(group) < TRAIN_SAMPLES_PER_CLASS, random_state=SPLIT_SEED + 100 + offset))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SPLIT_SEED + 200).reset_index(drop=True)

def _find_video(video_name):
    for candidate_dir in [VIDEO_DIR, os.path.join(REPO_ROOT, 'data', 'raw'), os.path.join(REPO_ROOT, 'data', 'processed'), os.path.join(REPO_ROOT, 'data', '7seg_unique_cropped')]:
        p = os.path.join(candidate_dir, video_name)
        if os.path.exists(_resolve_model_artifact(p)):
            return p
    return None

def _sample_frame_indices(total_frames, source_fps, target_fps):
    if total_frames <= 0:
        return []
    if not target_fps or target_fps <= 0 or (not source_fps) or (source_fps <= 0):
        return list(range(total_frames))
    duration = total_frames / source_fps
    sample_count = max(1, int(np.floor(duration * target_fps)))
    sample_times = np.arange(sample_count, dtype=np.float32) / float(target_fps)
    indices = np.rint(sample_times * float(source_fps)).astype(np.int32)
    indices = np.clip(indices, 0, total_frames - 1)
    return indices.tolist()

def _center_crop_fixed(frame, crop_size):
    h, w = frame.shape[:2]
    if h < crop_size or w < crop_size:
        return None
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return frame[y0:y0 + crop_size, x0:x0 + crop_size]

def _crop_center_square(frame):
    h, w = frame.shape[:2]
    m = min(h, w)
    y0, x0 = ((h - m) // 2, (w - m) // 2)
    return frame[y0:y0 + m, x0:x0 + m]

def _preprocess_frame_bgr(frame, output_size):
    """Match the project preprocessing contract before model-specific normalization.

    The main prepared corpus was generated by downscaling large source frames and
    center-cropping a 224x224 region. Already-cropped 224x224 clips are preserved.
    """
    if PREPROCESS_MODE == 'half_center224':
        h, w = frame.shape[:2]
        if min(h, w) >= output_size * 2 and PRE_CROP_SCALE > 0:
            frame = cv2.resize(frame, (int(w * PRE_CROP_SCALE), int(h * PRE_CROP_SCALE)), interpolation=cv2.INTER_AREA)
        cropped = _center_crop_fixed(frame, output_size)
        if cropped is not None:
            return cropped
    frame = _crop_center_square(frame)
    return cv2.resize(frame, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

def _read_frame(cap, frame_idx, output_size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        return None
    return _preprocess_frame_bgr(frame, output_size)

def _has_minimum_flow_input(video_name):
    """Check that a sample can provide at least two frames or a verified cache."""
    for candidate in [_flow_cache_path(video_name), _verified_legacy_flow_cache_path(video_name)]:
        if candidate and os.path.exists(_resolve_model_artifact(candidate)):
            return True
    video_path = _find_video(video_name)
    if video_path is None or not CV2_OK:
        return False
    cap = cv2.VideoCapture(video_path)
    try:
        return cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 2
    finally:
        cap.release()

def filter_manifest_inputs(df, source_name):
    """Exclude invalid inputs explicitly and return both valid and rejected rows."""
    valid_rows = []
    rejected_rows = []
    for row in df.itertuples(index=False):
        video_name = str(row.video_name)
        if _find_video(video_name) is None:
            rejected_rows.append({'video_name': video_name, 'tag': row.tag, 'reason': 'missing_rgb_video'})
        elif not _has_minimum_flow_input(video_name):
            rejected_rows.append({'video_name': video_name, 'tag': row.tag, 'reason': 'fewer_than_two_frames_for_flow'})
        else:
            valid_rows.append({'video_name': video_name, 'tag': row.tag})
    valid_df = pd.DataFrame(valid_rows, columns=['video_name', 'tag'])
    rejected_df = pd.DataFrame(rejected_rows, columns=['video_name', 'tag', 'reason'])
    if len(rejected_df):
        print(f"  [{source_name}] excluding {len(rejected_df)} invalid inputs: {rejected_df['reason'].value_counts().to_dict()}")
    return (valid_df, rejected_df)

def load_rgb_frame(video_name):
    if not CV2_OK:
        return None
    video_path = _find_video(video_name)
    if video_path is None:
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = max(0, total // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        frame = _preprocess_frame_bgr(frame, IMG_SIZE)
        frame = frame[:, :, ::-1].astype(np.float32) / 255.0
        return frame
    except Exception:
        return None

def _flow_cache_path(video_name):
    stem = os.path.splitext(video_name)[0]
    sampling = FLOW_SAMPLING.replace(os.sep, '_')
    scale = str(PRE_CROP_SCALE).replace('.', 'p')
    fps = f'{TARGET_FPS:g}'.replace('.', 'p')
    flow_max = f'{FLOW_MAX:g}'.replace('.', 'p')
    return os.path.join(FLOW_CACHE_DIR, stem + f'_{FLOW_IMG_SIZE}px_{NUM_FLOW_FRAMES}f_{sampling}_fps{fps}_s{scale}_{PREPROCESS_MODE}_m{flow_max}.npz')

def _verified_legacy_flow_cache_path(video_name):
    """Return the May-2026 cache path only for an exactly matching recipe."""
    if not REUSE_VERIFIED_IMPROVED_CACHE:
        return None
    if not (NUM_FLOW_FRAMES == 10 and FLOW_IMG_SIZE == 224 and (FLOW_SAMPLING == 'adjacent_center') and (abs(TARGET_FPS - 25.0) < 1e-09) and (PREPROCESS_MODE == 'half_center224') and (abs(PRE_CROP_SCALE - 0.5) < 1e-09) and (abs(FLOW_MAX - 20.0) < 1e-09)):
        return None
    stem = os.path.splitext(video_name)[0]
    return os.path.join(BASE_DIR, 'flow_cache', 'improved', stem + '_224px_adjacent_center_fps25_s0p5.npz')

def _compute_farneback(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open: {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if FLOW_SAMPLING == 'uniform_clip':
        indices = np.linspace(0, max(total - 1, 1), NUM_FLOW_FRAMES + 1, dtype=int).tolist()
    else:
        sampled = _sample_frame_indices(total, source_fps, TARGET_FPS)
        if not sampled:
            sampled = list(range(total))
        needed = NUM_FLOW_FRAMES + 1
        center = len(sampled) // 2
        start = max(0, center - needed // 2)
        end = start + needed
        if end > len(sampled):
            end = len(sampled)
            start = max(0, end - needed)
        indices = sampled[start:end]
        while len(indices) < needed and indices:
            indices.append(indices[-1])
    flows, prev_gray = ([], None)
    for fi in indices:
        frame = _read_frame(cap, fi, FLOW_IMG_SIZE)
        if frame is None:
            if flows:
                flows.append(flows[-1].copy())
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            flows.append(np.clip(flow, -FLOW_MAX, FLOW_MAX) / FLOW_MAX)
        prev_gray = gray
    cap.release()
    if not flows:
        raise RuntimeError('No flow frames extracted')
    while len(flows) < NUM_FLOW_FRAMES:
        flows.append(flows[-1].copy())
    flows = flows[:NUM_FLOW_FRAMES]
    return np.concatenate(flows, axis=-1).astype(np.float32)

def _postprocess_flow(stacked):
    """Apply experiment-specific flow normalization without changing caches.

    ``per_field_mean`` subtracts the spatial mean displacement independently
    from the horizontal and vertical channels of every flow field. This is a
    cheap global-camera-motion compensation control inspired by the original
    Two-Stream input preprocessing. Cached tensors remain immutable so that
    experiments can share the audited base cache safely.
    """
    arr = np.asarray(stacked, dtype=np.float32)
    if FLOW_CENTERING == 'none':
        return arr
    fields = arr.reshape(FLOW_IMG_SIZE, FLOW_IMG_SIZE, NUM_FLOW_FRAMES, 2).copy()
    fields -= fields.mean(axis=(0, 1), keepdims=True)
    return np.clip(fields, -1.0, 1.0).reshape(arr.shape).astype(np.float32)

def load_or_compute_flow(video_name):
    cache_path = _flow_cache_path(video_name)
    cache_candidates = [cache_path]
    verified_legacy = _verified_legacy_flow_cache_path(video_name)
    if verified_legacy:
        cache_candidates.append(verified_legacy)
    for candidate in cache_candidates:
        if not os.path.exists(_resolve_model_artifact(candidate)):
            continue
        try:
            with _load_model_result_array(candidate) as data:
                arr = data['flow']
            if arr.shape == (FLOW_IMG_SIZE, FLOW_IMG_SIZE, NUM_FLOW_FRAMES * 2):
                if np.isfinite(arr).all() and float(np.max(np.abs(arr))) <= 1.00001:
                    return _postprocess_flow(arr)
        except Exception:
            pass
    if not CV2_OK:
        return None
    video_path = _find_video(video_name)
    if video_path is None:
        return None
    try:
        stacked = _compute_farneback(video_path)
        np.savez_compressed(cache_path, flow=stacked)
        return _postprocess_flow(stacked)
    except Exception as e:
        print(f'  WARNING: Flow failed for {video_name}: {e}')
        return None

class TwoStreamSequence(keras.utils.PyDataset):
    """Batch generator that loads RGB frames and flow stacks on demand.

    Keeps only BATCH_SIZE samples in RAM at once instead of the full dataset.
    Inherits from keras.utils.PyDataset so Keras handles shuffling/ordering.
    """

    def __init__(self, samples, label_map, batch_size, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.samples = samples
        self.label_map = label_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self._indices)

    def __len__(self):
        return max(1, int(np.ceil(len(self.samples) / self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self._indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        rgb_batch, flow_batch, label_batch = ([], [], [])
        for i in batch_idx:
            video_name, label = self.samples[i]
            rgb = load_rgb_frame(video_name)
            flow = load_or_compute_flow(video_name)
            if rgb is None or flow is None:
                if STRICT_DATA:
                    missing = []
                    if rgb is None:
                        missing.append('RGB')
                    if flow is None:
                        missing.append('flow')
                    raise RuntimeError(f"Invalid sample {video_name!r}: missing {', '.join(missing)}")
                rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                flow = np.zeros((FLOW_IMG_SIZE, FLOW_IMG_SIZE, NUM_FLOW_FRAMES * 2), dtype=np.float32)
            rgb_batch.append(rgb)
            flow_batch.append(flow)
            label_batch.append(label)
        return ((np.array(rgb_batch, dtype=np.float32), np.array(flow_batch, dtype=np.float32)), np.array(label_batch, dtype=np.int32))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._indices)

    @property
    def num_samples(self):
        return len(self.samples)

def build_sequence(df, label_map, split_name, shuffle=True):
    """Validate every row once (cheap: just check the cache / video exists),
    then return a TwoStreamSequence over the valid rows.

    This does NOT load any pixel data — validation is just a path check.
    """
    if QUICK and len(df) > 50:
        df = df.sample(n=50, random_state=SPLIT_SEED).reset_index(drop=True)
    samples = []
    skipped = {'label': 0, 'rgb': 0, 'flow': 0}
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f'  [{split_name}] validating {i}/{total} ...', flush=True)
        video_name = str(row['video_name'])
        tag = canonical_tag(row['tag'])
        if tag not in label_map:
            skipped['label'] += 1
            continue
        if _find_video(video_name) is None:
            skipped['rgb'] += 1
            continue
        cache_ok = os.path.exists(_resolve_model_artifact(_flow_cache_path(video_name)))
        if not cache_ok and _find_video(video_name) is None:
            skipped['flow'] += 1
            continue
        samples.append((video_name, label_map[tag]))
    for reason, n in skipped.items():
        if n:
            print(f'  [{split_name}] WARNING: {n} skipped ({reason})')
    print(f'  [{split_name}] {len(samples)} valid samples  ({len(samples) // BATCH_SIZE + 1} batches of {BATCH_SIZE})')
    return TwoStreamSequence(samples, label_map, BATCH_SIZE, shuffle=shuffle)

def build_spatial_cnn(num_classes):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='rgb_input')
    x = keras.layers.Lambda(lambda t: tf.keras.applications.mobilenet_v2.preprocess_input(t * 255.0), name='sp_preprocess')(inputs)
    backbone = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=x, pooling=None)
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-UNFREEZE:]:
        layer.trainable = True
    features = backbone.output
    features = keras.layers.GlobalAveragePooling2D(name='sp_gap')(features)
    features = keras.layers.Dense(DENSE_UNITS, activation='relu', name='sp_dense')(features)
    features = keras.layers.Dropout(DROPOUT, name='sp_dropout')(features)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='sp_head')(features)
    return keras.Model(inputs=backbone.input, outputs=outputs, name='spatial_cnn')

def build_temporal_cnn(num_classes):
    flow_channels = NUM_FLOW_FRAMES * 2
    base_rgb = keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(FLOW_IMG_SIZE, FLOW_IMG_SIZE, 3), pooling=None)
    inputs = keras.Input(shape=(FLOW_IMG_SIZE, FLOW_IMG_SIZE, flow_channels), name='flow_input')
    base_flow = keras.applications.MobileNetV2(include_top=False, weights=None, input_tensor=inputs, pooling=None)
    first_conv_name = None
    for layer in base_flow.layers:
        if isinstance(layer, keras.layers.Conv2D):
            first_conv_name = layer.name
            break
    for lf in base_flow.layers:
        if lf.name == first_conv_name:
            continue
        try:
            lr = base_rgb.get_layer(lf.name)
            lf.set_weights(lr.get_weights())
        except Exception:
            pass
    rgb_weights = base_rgb.get_layer(first_conv_name).get_weights()
    rgb_kernel = rgb_weights[0]
    rgb_biases = rgb_weights[1:]
    avg_kernel = rgb_kernel.mean(axis=2, keepdims=True)
    new_kernel = np.repeat(avg_kernel, flow_channels, axis=2).astype(np.float32)
    base_flow.get_layer(first_conv_name).set_weights([new_kernel] + rgb_biases)
    for layer in base_flow.layers:
        layer.trainable = False
    for layer in base_flow.layers[:TEMPORAL_UNFREEZE_FIRST]:
        layer.trainable = True
    for layer in base_flow.layers[-UNFREEZE:]:
        layer.trainable = True
    features = base_flow.output
    features = keras.layers.GlobalAveragePooling2D(name='fl_gap')(features)
    features = keras.layers.Dense(DENSE_UNITS, activation='relu', name='fl_dense')(features)
    features = keras.layers.Dropout(DROPOUT, name='fl_dropout')(features)
    outputs = keras.layers.Dense(num_classes, activation='softmax', name='fl_head')(features)
    return keras.Model(inputs=base_flow.input, outputs=outputs, name='temporal_cnn')

class TwoStreamModel(keras.Model):

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.spatial_cnn = build_spatial_cnn(num_classes)
        self.temporal_cnn = build_temporal_cnn(num_classes)
        self.concat = keras.layers.Concatenate(name='fusion_concat')
        self.alpha_fc = keras.layers.Dense(1, activation='sigmoid', kernel_initializer='zeros', name='fusion_alpha')
        self.aux_weight = 0.1
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.accuracy_tracker = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        self.f1_tracker = SparseMacroF1(num_classes, name='f1_macro')

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker, self.f1_tracker]

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config

    def call(self, inputs, training=False):
        rgb, flow = inputs
        sp_out = self.spatial_cnn(rgb, training=training)
        fl_out = self.temporal_cnn(flow, training=training)
        alpha = self.alpha_fc(self.concat([sp_out, fl_out]))
        return alpha * sp_out + (1.0 - alpha) * fl_out

    def train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        rgb, flow = x
        with tf.GradientTape() as tape:
            sp_out = self.spatial_cnn(rgb, training=True)
            fl_out = self.temporal_cnn(flow, training=True)
            alpha = self.alpha_fc(self.concat([sp_out, fl_out]))
            fused = alpha * sp_out + (1.0 - alpha) * fl_out
            loss_fn = keras.losses.SparseCategoricalCrossentropy()
            loss_main = loss_fn(y, fused, sample_weight=sample_weight)
            loss_sp = loss_fn(y, sp_out, sample_weight=sample_weight)
            loss_fl = loss_fn(y, fl_out, sample_weight=sample_weight)
            total = loss_main + self.aux_weight * (loss_sp + loss_fl)
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(total)
        self.accuracy_tracker.update_state(y, fused)
        self.f1_tracker.update_state(y, fused)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Evaluate the same main+auxiliary objective used for training."""
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        rgb, flow = x
        sp_out = self.spatial_cnn(rgb, training=False)
        fl_out = self.temporal_cnn(flow, training=False)
        alpha = self.alpha_fc(self.concat([sp_out, fl_out]))
        fused = alpha * sp_out + (1.0 - alpha) * fl_out
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        loss_main = loss_fn(y, fused, sample_weight=sample_weight)
        loss_sp = loss_fn(y, sp_out, sample_weight=sample_weight)
        loss_fl = loss_fn(y, fl_out, sample_weight=sample_weight)
        total = loss_main + self.aux_weight * (loss_sp + loss_fl)
        self.loss_tracker.update_state(total)
        self.accuracy_tracker.update_state(y, fused, sample_weight=sample_weight)
        self.f1_tracker.update_state(y, fused, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}
RESULTS_CSV = os.path.join(RESULTS_DIR, 'two_stream_results.csv')

def save_to_csv(row):
    write_header = not os.path.exists(_resolve_model_artifact(RESULTS_CSV))
    with open(_resolve_model_artifact(RESULTS_CSV), 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def main():
    print('=' * 64)
    print('  Two-Stream Network -- Keras (true Simonyan-style)')
    print(f'  Backbone       : {BACKBONE}')
    print(f'  RGB size       : {IMG_SIZE}px   Unfreeze last: {UNFREEZE} layers')
    print(f'  Flow           : {NUM_FLOW_FRAMES} frames x {FLOW_IMG_SIZE}px -> {NUM_FLOW_FRAMES * 2} channels')
    print(f'  Flow sampling  : {FLOW_SAMPLING} @ target_fps={TARGET_FPS}')
    print(f'  Preprocess     : {PREPROCESS_MODE} scale={PRE_CROP_SCALE} before center crop')
    print(f'  Temporal tune  : first {TEMPORAL_UNFREEZE_FIRST} layers + last {UNFREEZE} layers')
    print(f'  Train mode     : {TRAIN_MODE}  class_weights={USE_CLASS_WEIGHTS}')
    print(f'  Evaluate test  : {EVALUATE_TEST}')
    print(f'  Split seed     : {SPLIT_SEED}  validation={VAL_FRACTION:.2f}')
    print(f'  Flow centering : {FLOW_CENTERING}')
    print(f'  Dense units    : {DENSE_UNITS}  Dropout: {DROPOUT}')
    print(f'  LR={LR}  Clipnorm={CLIPNORM}  Batch={BATCH_SIZE}  Epochs={EPOCHS}  Quick={QUICK}')
    print('=' * 64)
    if not os.path.exists(_resolve_model_artifact(TRAIN_CSV)):
        print(f'ERROR: Train CSV not found: {TRAIN_CSV}')
        sys.exit(1)
    if not os.path.isdir(_resolve_model_artifact(VIDEO_DIR)):
        print(f'WARNING: VIDEO_DIR not found: {VIDEO_DIR} -- will try fallback dirs.')
    train_raw = normalize_manifest(pd.read_csv(_resolve_model_artifact(TRAIN_CSV)), 'train')
    test_df = normalize_manifest(pd.read_csv(_resolve_model_artifact(TEST_CSV)), 'test')
    train_raw, rejected_train = filter_manifest_inputs(train_raw, 'train')
    test_df, rejected_test = filter_manifest_inputs(test_df, 'test')
    pd.concat([rejected_train, rejected_test], ignore_index=True).to_csv(os.path.join(RESULTS_DIR, 'excluded_inputs.csv'), index=False)
    print(f'\nTrain manifest: {len(train_raw)} rows  |  Test: {len(test_df)} rows')
    label_map = {tag: i for i, tag in enumerate(CLASS_NAMES)}
    label_vocab = list(CLASS_NAMES)
    num_classes = len(label_vocab)
    print(f'Classes (sorted, 0-based): {label_vocab}')
    print('\nScanning train set (path validation, no pixel loading)...')
    t0 = time.time()
    tr_unique_df, val_df = split_unique_stratified(train_raw)
    if TRAIN_MODE == 'natural_weighted':
        tr_df = tr_unique_df.sample(frac=1.0, random_state=SPLIT_SEED + 200).reset_index(drop=True)
    elif TRAIN_MODE == 'balanced':
        tr_df = balance_training_manifest(tr_unique_df)
    else:
        raise ValueError("TRAIN_MODE must be 'natural_weighted' or 'balanced'")
    overlap = set(tr_df['video_name']).intersection(val_df['video_name'])
    if overlap:
        raise RuntimeError(f'Train/validation leakage detected for {len(overlap)} videos')
    test_overlap = set(train_raw['video_name']).intersection(test_df['video_name'])
    if test_overlap:
        raise RuntimeError(f'Train/test leakage detected for {len(test_overlap)} videos')
    print(f'  Unique split: {len(tr_unique_df)} train / {len(val_df)} val; training manifest: {len(tr_df)} rows ({TRAIN_MODE})')
    tr_df.to_csv(os.path.join(RESULTS_DIR, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(RESULTS_DIR, 'validation_split.csv'), index=False)
    test_df.to_csv(os.path.join(RESULTS_DIR, 'test_split.csv'), index=False)
    run_config = {'experiment_name': EXPERIMENT_NAME, 'flow_cache_name': FLOW_CACHE_NAME, 'backbone': BACKBONE, 'img_size': IMG_SIZE, 'unfreeze': UNFREEZE, 'temporal_unfreeze_first': TEMPORAL_UNFREEZE_FIRST, 'dense_units': DENSE_UNITS, 'dropout': DROPOUT, 'learning_rate': LR, 'batch_size': BATCH_SIZE, 'max_epochs': EPOCHS, 'split_seed': SPLIT_SEED, 'validation_fraction': VAL_FRACTION, 'train_mode': TRAIN_MODE, 'use_class_weights': USE_CLASS_WEIGHTS, 'evaluate_test': EVALUATE_TEST, 'flow_frames': NUM_FLOW_FRAMES, 'flow_img_size': FLOW_IMG_SIZE, 'flow_max': FLOW_MAX, 'target_fps': TARGET_FPS, 'flow_sampling': FLOW_SAMPLING, 'flow_centering': FLOW_CENTERING, 'preprocess_mode': PREPROCESS_MODE, 'pre_crop_scale': PRE_CROP_SCALE, 'strict_data': STRICT_DATA, 'reuse_verified_improved_cache': REUSE_VERIFIED_IMPROVED_CACHE, 'train_rows': len(tr_df), 'validation_rows': len(val_df), 'test_rows': len(test_df), 'train_validation_overlap': 0, 'train_test_overlap': 0}
    with open(_resolve_model_artifact(os.path.join(RESULTS_DIR, 'run_config.json')), 'w', encoding='utf-8') as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    train_seq = build_sequence(tr_df, label_map, 'train', shuffle=True)
    val_seq = build_sequence(val_df, label_map, 'val', shuffle=False)
    test_seq = build_sequence(test_df, label_map, 'test', shuffle=False)
    print(f'  Scan done in {time.time() - t0:.0f}s')
    if train_seq.num_samples == 0:
        print('ERROR: No valid training samples. Check VIDEO_DIR and CSV paths.')
        sys.exit(1)
    class_weight = None
    if SKLEARN_OK and USE_CLASS_WEIGHTS:
        try:
            y_tr_labels = [lbl for _, lbl in train_seq.samples]
            classes = np.unique(y_tr_labels)
            cw = compute_class_weight('balanced', classes=classes, y=y_tr_labels)
            class_weight = {int(class_id): float(weight) for class_id, weight in zip(classes, cw)}
            print(f'Class weights: {class_weight}')
        except Exception as e:
            print(f'  WARNING: Class weight error: {e}')
    print('\nBuilding two-stream model...')
    optimizer = keras.optimizers.Adam(learning_rate=LR, weight_decay=1e-05, clipnorm=CLIPNORM)
    model = TwoStreamModel(num_classes=num_classes, name='two_stream')
    dummy_rgb = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    dummy_flow = np.zeros((1, FLOW_IMG_SIZE, FLOW_IMG_SIZE, NUM_FLOW_FRAMES * 2), dtype=np.float32)
    _ = model([dummy_rgb, dummy_flow], training=False)
    model.compile(optimizer=optimizer)
    print(f'  Params: {model.count_params():,}')
    ckpt_path = os.path.join(CKPT_DIR, 'two_stream_best.weights.h5')
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_f1_macro', mode='max', patience=10, restore_best_weights=True, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_f1_macro', mode='max', factor=0.5, patience=5, min_lr=1e-06, verbose=1), keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_f1_macro', mode='max', save_best_only=True, save_weights_only=True, verbose=1)]
    print(f'\nTraining for up to {EPOCHS} epochs...')
    t_start = time.time()
    history = model.fit(train_seq, validation_data=val_seq, epochs=EPOCHS, class_weight=class_weight, callbacks=callbacks, verbose=1)
    duration = time.time() - t_start
    print(f'\nTraining done in {duration:.0f}s')
    sp_ckpt = os.path.join(CKPT_DIR, 'two_stream_spatial.weights.h5')
    fl_ckpt = os.path.join(CKPT_DIR, 'two_stream_temporal.weights.h5')
    model.spatial_cnn.save_weights(sp_ckpt)
    model.temporal_cnn.save_weights(fl_ckpt)
    test_loss = test_acc = test_f1 = None
    if EVALUATE_TEST:
        print('\nEvaluating on test set...')
        eval_out = model.evaluate(test_seq, verbose=0, return_dict=True)
        test_loss = float(eval_out.get('loss', 0.0))
        test_acc = float(eval_out.get('accuracy', 0.0))
        test_f1 = float(eval_out.get('f1_macro', 0.0))
        print(f'  Test acc={test_acc:.4f}  F1 macro={test_f1:.4f}  loss={test_loss:.4f}')
    else:
        print('\nSkipping test evaluation (validation-only experiment).')
    history_df = pd.DataFrame(history.history)
    history_df.index = np.arange(1, len(history_df) + 1)
    history_df.index.name = 'epoch'
    history_df.to_csv(os.path.join(RESULTS_DIR, 'history.csv'))
    val_f1_history = history.history.get('val_f1_macro', [])
    best_epoch = int(np.argmax(val_f1_history) + 1) if val_f1_history else None
    metrics = {'experiment_name': EXPERIMENT_NAME, 'flow_cache_name': FLOW_CACHE_NAME, 'backbone': BACKBONE, 'unfreeze': UNFREEZE, 'flow_frames': NUM_FLOW_FRAMES, 'flow_img_size': FLOW_IMG_SIZE, 'flow_sampling': FLOW_SAMPLING, 'flow_centering': FLOW_CENTERING, 'target_fps': TARGET_FPS, 'preprocess_mode': PREPROCESS_MODE, 'pre_crop_scale': PRE_CROP_SCALE, 'temporal_unfreeze_first': TEMPORAL_UNFREEZE_FIRST, 'dense_units': DENSE_UNITS, 'dropout': DROPOUT, 'lr': LR, 'batch_size': BATCH_SIZE, 'train_mode': TRAIN_MODE, 'use_class_weights': USE_CLASS_WEIGHTS, 'split_seed': SPLIT_SEED, 'val_samples': val_seq.num_samples, 'test_samples': test_seq.num_samples, 'test_accuracy': round(float(test_acc), 6) if test_acc is not None else None, 'test_f1_macro_metric': round(float(test_f1), 6) if test_f1 is not None else None, 'test_loss': round(float(test_loss), 6) if test_loss is not None else None, 'duration_s': round(duration, 1), 'train_samples': train_seq.num_samples, 'epochs_ran': len(history.history.get('loss', [])), 'best_epoch': best_epoch, 'best_val_f1_macro': round(float(max(val_f1_history)), 6) if val_f1_history else None}
    if EVALUATE_TEST and SKLEARN_OK and (test_seq.num_samples > 0):
        y_prob_list, y_true_list = ([], [])
        sp_prob_list, fl_prob_list, alpha_list = ([], [], [])
        for (rgb_b, flow_b), lbl_b in test_seq:
            sp_probs = model.spatial_cnn.predict(rgb_b, verbose=0)
            fl_probs = model.temporal_cnn.predict(flow_b, verbose=0)
            alphas = model.alpha_fc(np.concatenate([sp_probs, fl_probs], axis=1), training=False).numpy()
            fused_probs = alphas * sp_probs + (1.0 - alphas) * fl_probs
            y_prob_list.append(fused_probs)
            sp_prob_list.append(sp_probs)
            fl_prob_list.append(fl_probs)
            alpha_list.append(alphas)
            y_true_list.append(lbl_b)
        y_prob = np.concatenate(y_prob_list, axis=0)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.concatenate(y_true_list, axis=0)
        prediction_rows = []
        sp_all = np.concatenate(sp_prob_list, axis=0)
        fl_all = np.concatenate(fl_prob_list, axis=0)
        alpha_all = np.concatenate(alpha_list, axis=0).reshape(-1)
        for idx, ((video_name, _), true_id, pred_id) in enumerate(zip(test_seq.samples, y_true, y_pred)):
            prediction_rows.append({'video_name': video_name, 'true_label': label_vocab[int(true_id)], 'predicted_label': label_vocab[int(pred_id)], 'prob_line': float(y_prob[idx, 0]), 'prob_other': float(y_prob[idx, 1]), 'prob_scrum': float(y_prob[idx, 2]), 'spatial_prob_line': float(sp_all[idx, 0]), 'spatial_prob_other': float(sp_all[idx, 1]), 'spatial_prob_scrum': float(sp_all[idx, 2]), 'temporal_prob_line': float(fl_all[idx, 0]), 'temporal_prob_other': float(fl_all[idx, 1]), 'temporal_prob_scrum': float(fl_all[idx, 2]), 'fusion_alpha': float(alpha_all[idx])})
        pd.DataFrame(prediction_rows).to_csv(os.path.join(RESULTS_DIR, 'test_predictions.csv'), index=False)
        f1m = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        f1w = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_macro'] = round(f1m, 4)
        metrics['f1_weighted'] = round(f1w, 4)
        cm = confusion_matrix(y_true, y_pred)
        row_sum = cm.sum(axis=1)
        recalls = np.where(row_sum > 0, np.diag(cm) / row_sum, 0.0)
        for i, cls in enumerate(label_vocab):
            metrics[f'recall_{cls}'] = round(float(recalls[i]), 4)
        alpha_values = alpha_all
        metrics['alpha_mean'] = round(float(alpha_values.mean()), 6)
        metrics['alpha_min'] = round(float(alpha_values.min()), 6)
        metrics['alpha_max'] = round(float(alpha_values.max()), 6)
        for prefix, probs in [('spatial', sp_all), ('temporal', fl_all)]:
            pred = np.argmax(probs, axis=1)
            metrics[f'{prefix}_accuracy'] = round(float(np.mean(pred == y_true)), 6)
            metrics[f'{prefix}_f1_macro'] = round(float(f1_score(y_true, pred, average='macro', zero_division=0)), 4)
        report = classification_report(y_true, y_pred, target_names=label_vocab, zero_division=0)
        print(f'\nClassification Report:\n{report}')
        with open(_resolve_model_artifact(os.path.join(RESULTS_DIR, 'classification_report.txt')), 'w', encoding='utf-8') as f:
            f.write(report)
        for norm, suffix in [(None, 'raw'), ('true', 'normalized')]:
            cm_p = confusion_matrix(y_true, y_pred, normalize=norm)
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay(cm_p, display_labels=label_vocab).plot(ax=ax, cmap='Blues', colorbar=False)
            ax.set_title(f'Two-Stream -- Confusion ({suffix})')
            fig.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{suffix}.png'), dpi=90)
            plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, m, vm in zip(axes, ['loss', 'accuracy', 'f1_macro'], ['val_loss', 'val_accuracy', 'val_f1_macro']):
        ax.plot(history.history.get(m, []), label='train')
        ax.plot(history.history.get(vm, []), label='val')
        ax.set_title(m)
        ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'history.png'))
    plt.close(fig)
    save_to_csv(metrics)
    print(f'\nDone.')
    print(f'   Results CSV   : {RESULTS_CSV}')
    print(f'   Best weights  : {ckpt_path}')
    print(f'   Spatial ckpt  : {sp_ckpt}')
    print(f'   Temporal ckpt : {fl_ckpt}')
if __name__ == '__main__':
    main()
