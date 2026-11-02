"""Data preprocessing utilities for video classification."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import keras
from tqdm import tqdm
from models.M0.config import IMG_SIZE, MAX_SEQ_LENGTH, NUM_FEATURES
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from models.common.spatial import normalize_training_frame

def crop_center_square(frame):
    """Crop the center square from a frame."""
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = x // 2 - min_dim // 2
    start_y = y // 2 - min_dim // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def training_center_crop(frame, target_size=IMG_SIZE, scale=None):
    """Normalize raw resolution and apply the fixed 224x224 training crop.

    ``scale`` remains accepted for compatibility with older callers, but web
    inference now derives the scale from the 360 px intermediate height used
    to construct the initial 1280x720 training collection.
    """
    return normalize_training_frame(frame, target_size=target_size)

def preprocess_raw_frame(frame, target_size=IMG_SIZE):
    """Return one raw BGR frame in the spatial format used for training."""
    if frame is not None and frame.shape[0] == target_size and (frame.shape[1] == target_size):
        return frame.copy()
    return training_center_crop(frame, target_size=target_size)

def apply_clahe(frame):
    """Apply the CLAHE transform used by the experimental CLAHE clips.

    The input and output remain BGR uint8 frames.  Keeping this operation
    before the BGR-to-RGB conversion makes web inference match the offline
    filter experiment exactly.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

def _sample_frame_indices(total_frames, source_fps, target_fps):
    """Return frame indices sampled at a fixed temporal FPS."""
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

def load_raw_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE), target_fps=25.0, frame_filter=None):
    """Load raw video using the training crop and the training FPS cadence.

    ``load_video`` remains the loader for the already-preprocessed 224x224
    dataset.  Web inference receives raw videos, so it must use this separate
    loader instead of applying a square crop directly to the source frame.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    target_size = int(resize[0])
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        sample_indices = _sample_frame_indices(total_frames, source_fps, target_fps)
        sample_counts = {}
        for index in sample_indices:
            sample_counts[index] = sample_counts.get(index, 0) + 1
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            repeat_count = sample_counts.get(frame_index, 0) if sample_counts else 1
            if repeat_count <= 0:
                frame_index += 1
                continue
            frame = preprocess_raw_frame(frame, target_size=target_size)
            if frame is None:
                raise ValueError(f'Raw frame could not be normalized for {target_size}x{target_size} training input')
            if frame_filter is not None:
                frame = frame_filter(frame)
            frame = frame[:, :, [2, 1, 0]]
            for _ in range(repeat_count):
                frames.append(frame.copy())
                if max_frames > 0 and len(frames) == max_frames:
                    break
            frame_index += 1
            if max_frames > 0 and len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE), target_fps=25.0):
    """Load a video file and return frames sampled to the training FPS."""
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        sample_indices = _sample_frame_indices(total_frames, source_fps, target_fps)
        sample_counts = {}
        for index in sample_indices:
            sample_counts[index] = sample_counts.get(index, 0) + 1
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            repeat_count = sample_counts.get(frame_index, 0) if sample_counts else 1
            if repeat_count <= 0:
                frame_index += 1
                continue
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            for _ in range(repeat_count):
                frames.append(frame.copy())
                if max_frames > 0 and len(frames) == max_frames:
                    break
            frame_index += 1
            if max_frames > 0 and len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def save_individual_data(frame_features, frame_masks, labels, file_path):
    """Save preprocessed video data to npz file."""
    np.savez(file_path, frame_features=frame_features, frame_masks=frame_masks, labels=labels)

def load_individual_data(file_path):
    """Load preprocessed video data from npz file."""
    data = _load_model_result_array(file_path)
    frame_features = data['frame_features']
    frame_masks = data['frame_masks']
    labels = data['labels']
    return ((frame_features, frame_masks), labels)

def _sanitize_cache_key(value):
    """Return a filesystem-safe cache key."""
    return str(value).replace('\\', '_').replace('/', '_').replace(':', '_').replace(' ', '_')

def _build_cache_path(cache_root_dir, feature_extractor_name, video_name):
    """Build extractor-isolated cache file path for one video."""
    safe_extractor = _sanitize_cache_key(feature_extractor_name)
    cache_dir = os.path.join(cache_root_dir, safe_extractor)
    output_filename = os.path.splitext(video_name)[0] + '.npz'
    return (cache_dir, os.path.join(cache_dir, output_filename))

def prepare_single_video_data(video_path, label, output_path, feature_extractor_model):
    """Extract features from a single video and save to disk.

    Added diagnostics/logging to help trace why load_video may return zero frames.
    """
    print(f'[prepare_single_video_data] start -- video_path={video_path!r}, output_path={output_path!r}, label={label!r}')
    if not os.path.exists(_resolve_model_artifact(video_path)):
        print(f'[prepare_single_video_data] ERROR: input file not found: {video_path}')
        raise FileNotFoundError(video_path)
    try:
        size = os.path.getsize(video_path)
        print(f'[prepare_single_video_data] file exists, size={size} bytes')
    except Exception as e:
        print(f'[prepare_single_video_data] could not determine file size: {e}')
    try:
        probe_cap = cv2.VideoCapture(video_path)
        opened = probe_cap.isOpened()
        frame_count_prop = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else -1
        fps_prop = probe_cap.get(cv2.CAP_PROP_FPS) if opened else -1
        width_prop = probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH) if opened else -1
        height_prop = probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if opened else -1
        print(f'[prepare_single_video_data] cv2 probe -> opened={opened}, frame_count={frame_count_prop}, fps={fps_prop}, width={width_prop}, height={height_prop}')
        if opened:
            probe_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = probe_cap.read()
            print(f'[prepare_single_video_data] cv2 single-frame read -> ret={ret}, frame_type={(type(frame) if ret else None)}')
            if ret and frame is not None:
                try:
                    print(f'[prepare_single_video_data] probe frame shape={frame.shape}, dtype={frame.dtype}')
                except Exception:
                    pass
        probe_cap.release()
    except Exception as e:
        print(f'[prepare_single_video_data] cv2 probe failed: {e}')
    try:
        frames = load_video(video_path)
    except Exception as e:
        print(f'[prepare_single_video_data] ERROR while calling load_video: {e}')
        raise
    print(f'[prepare_single_video_data] load_video returned {len(frames)} frames')
    if len(frames) == 0:
        print(f'[prepare_single_video_data] WARNING: no frames extracted from {video_path} — saving zero-filled features')
    frames = frames[:MAX_SEQ_LENGTH]
    video_length = frames.shape[0]
    if video_length > 0:
        try:
            print(f'[prepare_single_video_data] frames.shape={frames.shape}, dtype={frames.dtype}, nbytes={frames.nbytes}')
        except Exception:
            pass
    temp_frame_mask = np.zeros(shape=(MAX_SEQ_LENGTH,), dtype='bool')
    temp_frame_features = np.zeros(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')
    if video_length > 0:
        try:
            frame_features_batch = feature_extractor_model.predict(frames, verbose=0)
            print(f"[prepare_single_video_data] feature_extractor produced features with shape {getattr(frame_features_batch, 'shape', None)}")
            temp_frame_features[:video_length, :] = frame_features_batch
            temp_frame_mask[:video_length] = 1
        except Exception as e:
            print(f'[prepare_single_video_data] ERROR during feature extraction: {e}')
            raise
    save_individual_data(temp_frame_features, temp_frame_mask, np.array([label]), output_path)
    print(f'[prepare_single_video_data] saved npz -> {output_path} (video_length={video_length})')

def prepare_all_videos_data(video_root_dir, feature_extractor_model, output_dir='all_videos_features'):
    """Process all videos in a directory using the given feature extractor."""
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(video_root_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f"Found {len(video_files)} videos in '{video_root_dir}'")
    all_features = []
    all_masks = []
    all_names = []
    for video_name in tqdm(video_files, desc='Extracting features'):
        video_path = os.path.join(video_root_dir, video_name)
        output_path = os.path.join(output_dir, os.path.splitext(video_name)[0] + '.npz')
        try:
            if not os.path.exists(_resolve_model_artifact(output_path)):
                prepare_single_video_data(video_path, 'scrum', output_path, feature_extractor_model)
            data = _load_model_result_array(output_path)
            frame_features = data['frame_features']
            frame_masks = data['frame_masks']
            all_features.append(frame_features)
            all_masks.append(frame_masks)
            all_names.append(video_name)
        except Exception as e:
            print(f'⚠️ Error processing {video_name}: {e}')
            continue
    return (np.array(all_features), np.array(all_masks), all_names)

def load_or_prepare_data(train_root_dir, test_root_dir, cache_root_dir, feature_extractor_model, train_df, test_df, label_processor, feature_extractor_name='feature_extractor'):
    from models.common.legacy_training import rgb_dataset
    labels = list(label_processor.get_vocabulary())
    train_inputs, train_labels = rgb_dataset(train_df, labels, feature_extractor_name)
    test_inputs, test_labels = rgb_dataset(test_df, labels, feature_extractor_name)
    return train_inputs, train_labels, test_inputs, test_labels
