"""
Flask web application for rugby play detection.

This is the main application that provides a web interface for:
- Selecting trained models
- Uploading videos or selecting from library
- Adjusting detection parameters
- Running inference
- Visualizing results with timeline
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from pathlib import Path
import csv
import json
import logging
import math
import time
import re
import shutil
import subprocess
from datetime import datetime
from functools import lru_cache
from urllib.parse import quote
import threading
import uuid
from collections import Counter, deque
import sys
import os
import cv2
import numpy as np
try:
    from .video_metadata import read_video_metadata, read_video_metadata_many
except ImportError:
    from video_metadata import read_video_metadata, read_video_metadata_many
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    if __package__:
        from .inference_engine import InferenceEngine
    else:
        from inference_engine import InferenceEngine
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning('PyTorch inference engine not available')
try:
    if __package__:
        from .keras_inference_engine import KerasInferenceEngine, RugbyPlayDetector
    else:
        from keras_inference_engine import KerasInferenceEngine, RugbyPlayDetector
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    logger.warning('Keras inference engine not available')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'rugby-classification-thesis-2026'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.common.spatial import MODEL_INPUT_SIZE, SPATIAL_PREPROCESSING_VERSION, TRAINING_REFERENCE_HEIGHT, YOLO_SOURCE_ROI_VERSION, normalize_training_frame, training_normalization_geometry
from models.common.event_verifier import LineEventVerifier
from src.web.manual_clip_reference import consolidate_clip_sessions
WEB_RUNTIME = Path(os.environ.get('TESINA_WEB_RUNTIME', PROJECT_ROOT / '.runtime' / 'web')).resolve()
app.config['UPLOAD_FOLDER'] = WEB_RUNTIME / 'uploads'
app.config['RESULTS_FOLDER'] = WEB_RUNTIME / 'results'
app.config['VIDEO_LIBRARY'] = PROJECT_ROOT / 'data' / 'videos' / 'partidos_evaluacion'
app.config['REVIEW_FOLDER'] = WEB_RUNTIME / 'active_learning_review'
app.config['TRAINING_DATASET_FOLDER'] = WEB_RUNTIME / 'active_learning_dataset'
app.config['PREPROCESSED_VIDEO_FOLDER'] = app.config['RESULTS_FOLDER'] / 'preprocessed'
app.config['COMPARISON_RESULTS_FOLDER'] = app.config['RESULTS_FOLDER'] / 'comparisons'
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['PREPROCESSED_VIDEO_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['COMPARISON_RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['REVIEW_FOLDER'].mkdir(parents=True, exist_ok=True)
(app.config['REVIEW_FOLDER'] / 'sessions').mkdir(parents=True, exist_ok=True)
(app.config['REVIEW_FOLDER'] / 'exports').mkdir(parents=True, exist_ok=True)
(app.config['REVIEW_FOLDER'] / 'temp').mkdir(parents=True, exist_ok=True)
app.config['TRAINING_DATASET_FOLDER'].mkdir(parents=True, exist_ok=True)
(app.config['TRAINING_DATASET_FOLDER'] / 'sessions').mkdir(parents=True, exist_ok=True)
(app.config['TRAINING_DATASET_FOLDER'] / 'clips').mkdir(parents=True, exist_ok=True)
(app.config['TRAINING_DATASET_FOLDER'] / 'temp').mkdir(parents=True, exist_ok=True)
PROCESS_JOBS: dict = {}
LIVE_DETECTION_JOBS: dict = {}
ANALYSIS_JOBS: dict = {}
PREPROCESSED_VIDEO_LOCK = threading.Lock()
RESULTS_HISTORY_CACHE = {'signature': None, 'entries': None}
RESULTS_HISTORY_CACHE_LOCK = threading.Lock()
TRAINING_INPUT_SIZE = MODEL_INPUT_SIZE
INITIAL_TRAINING_ROOT = PROJECT_ROOT / 'data' / 'videos' / 'clips_224'
INITIAL_TRAIN_CSV = PROJECT_ROOT / 'data' / 'clips_dataset' / 'train.csv'
INITIAL_VALIDATION_CSV = PROJECT_ROOT / 'data' / 'clips_dataset' / 'validation.csv'
INITIAL_TEST_CSV = PROJECT_ROOT / 'data' / 'clips_dataset' / 'test.csv'
INITIAL_BALANCED_TRAIN_CSV = PROJECT_ROOT / 'data' / 'clips_dataset' / 'balanced_train.csv'
INITIAL_TRAINING_VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
INITIAL_TRAINING_CATALOG_CACHE = {'signature': None, 'clips': None, 'summary': None}
INITIAL_TRAINING_CATALOG_LOCK = threading.Lock()
INITIAL_TRAINING_PREVIEW_CACHE = WEB_RUNTIME / 'training_clip_previews'
INITIAL_TRAINING_PREVIEW_LOCK = threading.Lock()
TARGET_REVIEW_LABELS = {'scrum', 'line'}
VALID_REVIEW_FINAL_LABELS = {'scrum', 'line', 'other', 'no_event'}
REVIEW_OVERLAP_MERGE_SECONDS = 0.0
REVIEW_MODE_OVERLAP = 'overlap'
REVIEW_MODE_NON_OVERLAPPING = 'non_overlapping'
REVIEW_MODE_FULL_MATCH = 'full_match'
REVIEW_MODE_CLIP_GENERATOR = 'clip_generator'
VALID_REVIEW_MODES = {REVIEW_MODE_OVERLAP, REVIEW_MODE_NON_OVERLAPPING, REVIEW_MODE_FULL_MATCH, REVIEW_MODE_CLIP_GENERATOR}
REVIEW_BLOCK_DURATION_SECONDS = 7.0
if KERAS_AVAILABLE:
    inference_engine = KerasInferenceEngine()
    logger.info('Using Keras inference engine')
elif PYTORCH_AVAILABLE:
    inference_engine = InferenceEngine()
    logger.info('Using PyTorch inference engine')
else:
    raise RuntimeError('No inference engine available! Install either TensorFlow or PyTorch')

def _crop_center_square(frame):
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = x // 2 - min_dim // 2
    start_y = y // 2 - min_dim // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def _training_center_crop(frame, target_size: int=224):
    """Normalize source resolution, then apply the fixed training crop."""
    return normalize_training_frame(frame, target_size=target_size)

def _now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')

def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)

def _to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)

def _sequence_length_from_model(model) -> int:
    """Infer expected sequence length from model input shape; fallback to 200."""
    try:
        input_shape = model.sequence_model.input_shape
        if isinstance(input_shape, list) and input_shape:
            seq_len = input_shape[0][1]
            if seq_len:
                return int(seq_len)
    except Exception:
        pass
    return 200

def _build_live_job(job_id: str, video_path: Path, model_name: str, params: dict) -> dict:
    return {'job_id': job_id, 'status': 'queued', 'created_at': _now_iso(), 'updated_at': _now_iso(), 'video_path': str(video_path), 'model_name': model_name, 'parameters': params, 'progress': {'current_time': 0.0, 'duration': 0.0, 'processed_frames': 0, 'total_frames': 0, 'percent': 0.0}, 'latest_prediction': None, 'detections': [], 'playback_time': 0.0, 'playback_playing': False, 'playback_updated_at': None, 'stop_requested': False, 'error': None, 'result_file': None}

def _persist_live_job_result(job: dict) -> None:
    result_payload = {'status': job['status'], 'job_id': job['job_id'], 'created_at': job['created_at'], 'updated_at': job['updated_at'], 'video_path': job['video_path'], 'model_name': job['model_name'], 'parameters': job['parameters'], 'progress': job['progress'], 'latest_prediction': job['latest_prediction'], 'detections': job['detections'], 'error': job['error']}
    result_path = app.config['RESULTS_FOLDER'] / f"live_detection_{job['job_id']}.json"
    with _resolve_model_artifact(result_path).open('w', encoding='utf-8') as handle:
        json.dump(result_payload, handle, indent=2)
    job['result_file'] = str(result_path)

def _predict_live_window(model, window_frames: list[np.ndarray], sequence_len: int, sampled_frame_count: int | None=None, raw_window_frames: list[np.ndarray] | None=None) -> tuple[str, float, dict, dict]:
    if not window_frames:
        return ('other', 0.0, {}, {'primary_play_type': 'other', 'primary_confidence': 0.0, 'primary_probabilities': {}, 'meta_corrector_applied': False})
    target_count = sampled_frame_count if sampled_frame_count and sampled_frame_count > 0 else sequence_len
    target_count = min(sequence_len, max(1, int(target_count)))
    if len(window_frames) > target_count:
        indices = np.linspace(0, len(window_frames) - 1, num=target_count, dtype=np.int32)
        sampled = [window_frames[idx] for idx in indices]
    else:
        sampled = window_frames
    if hasattr(model, 'predict_live_window'):
        live_kwargs = {'sampled_frame_count': sampled_frame_count}
        if getattr(model, 'requires_raw_yolo_frames', False):
            live_kwargs['raw_window_frames'] = raw_window_frames
        predictions = model.predict_live_window(sampled, sequence_len, **live_kwargs)
    else:
        frame_array = np.array(sampled, dtype=np.uint8)
        features = model.feature_extractor.predict(frame_array, batch_size=32, verbose=0)
        feature_dim = int(features.shape[1])
        padded_features = np.zeros((sequence_len, feature_dim), dtype=np.float32)
        mask = np.zeros(sequence_len, dtype=np.float32)
        length = min(len(features), sequence_len)
        padded_features[:length] = features[:length]
        mask[:length] = 1.0
        predictions = model.sequence_model.predict([np.expand_dims(padded_features, axis=0), np.expand_dims(mask, axis=0)], verbose=0)
    pred_index = int(np.argmax(predictions[0]))
    primary_confidence = float(predictions[0][pred_index])
    primary_label = str(model.label_vocab[pred_index])
    primary_probabilities = {str(lbl): float(prob) for lbl, prob in zip(model.label_vocab, predictions[0])}
    if hasattr(model, 'correct_probabilities'):
        label, confidence, probabilities, correction_audit = model.correct_probabilities(primary_probabilities)
        return (label, confidence, probabilities, {**correction_audit, 'primary_play_type': primary_label, 'primary_confidence': primary_confidence, 'primary_probabilities': primary_probabilities})
    return (primary_label, primary_confidence, primary_probabilities, {'primary_play_type': primary_label, 'primary_confidence': primary_confidence, 'primary_probabilities': primary_probabilities, 'meta_corrector_applied': False})

def _live_detection_worker(job_id: str) -> None:
    job = LIVE_DETECTION_JOBS[job_id]
    params = job['parameters']
    video_path = Path(job['video_path'])
    try:
        job['status'] = 'running'
        job['updated_at'] = _now_iso()
        if job.get('model_name'):
            inference_engine.load_model(job['model_name'])
        model = getattr(inference_engine, 'current_model', None)
        if model is None:
            inference_engine.load_model(job.get('model_name'))
            model = getattr(inference_engine, 'current_model', None)
        required_attrs = ['feature_extractor', 'sequence_model', 'label_vocab']
        missing = [attr for attr in required_attrs if not hasattr(model, attr)]
        if missing:
            raise RuntimeError(f"Current model type does not support live job inference ({', '.join(missing)} missing).")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video_path}')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0
        sequence_len = _sequence_length_from_model(model)
        window_seconds = _to_float(params.get('window_seconds'), 7.0)
        prediction_interval_seconds = _to_float(params.get('prediction_interval_seconds', params.get('stride_seconds')), 1.0)
        confidence_threshold = _to_float(params.get('confidence_threshold'), 0.7)
        line_confidence_threshold = _to_float(params.get('line_confidence_threshold'), 0.95)
        line_confidence_threshold = max(0.0, min(1.0, line_confidence_threshold))
        target_fps = _to_float(params.get('target_fps'), 25.0)
        sampled_frame_count = _to_int(params.get('sampled_frame_count'), 0)
        simulate_realtime = bool(params.get('simulate_realtime', True))
        sleep_scale = _to_float(params.get('sleep_scale'), 1.0)
        frame_interval = 1.0 / max(fps, 0.1)
        sample_interval = 1.0 / max(target_fps, 0.1)
        next_sample_time = 0.0
        job['progress'].update({'duration': float(duration), 'total_frames': int(total_frames)})
        buffer = deque()
        next_inference_time = window_seconds
        processed = 0
        stream_started_wall = time.perf_counter()
        while True:
            if job.get('stop_requested'):
                job['status'] = 'stopped'
                break
            current_time = processed / fps if fps > 0 else 0.0
            if simulate_realtime:
                target_wall_time = stream_started_wall + current_time * sleep_scale
                sleep_seconds = target_wall_time - time.perf_counter()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                while not job.get('stop_requested') and current_time > float(job.get('playback_time') or 0.0) + 0.08:
                    time.sleep(0.02)
            ok, frame = cap.read()
            if not ok:
                if job['status'] != 'stopped':
                    job['status'] = 'completed'
                break
            processed += 1
            if current_time + 0.5 / max(fps, 0.1) >= next_sample_time:
                pre = _training_center_crop(frame, target_size=224)
                if pre is None:
                    cap.release()
                    raise RuntimeError('Raw frame is too small for the 224x224 training crop')
                pre = pre[:, :, [2, 1, 0]]
                while current_time + 0.5 / max(fps, 0.1) >= next_sample_time:
                    buffer.append((next_sample_time, pre.copy(), frame.copy()))
                    next_sample_time += sample_interval
            min_keep_time = max(0.0, current_time - window_seconds - prediction_interval_seconds)
            while buffer and buffer[0][0] < min_keep_time:
                buffer.popleft()
            while current_time >= next_inference_time:
                window_start = max(0.0, next_inference_time - window_seconds)
                window_frames = [frm for t, frm, _ in buffer if t >= window_start]
                raw_window_frames = [raw for t, _, raw in buffer if t >= window_start]
                if window_frames:
                    label, confidence, probs, prediction_audit = _predict_live_window(model, window_frames, sequence_len, sampled_frame_count=sampled_frame_count, raw_window_frames=raw_window_frames)
                    prediction = {'time': float(next_inference_time), 'window_start': float(window_start), 'window_end': float(next_inference_time), 'play_type': label, 'confidence': float(confidence), 'probabilities': probs, 'primary_play_type': prediction_audit.get('primary_play_type'), 'primary_confidence': prediction_audit.get('primary_confidence'), 'primary_probabilities': prediction_audit.get('primary_probabilities'), 'meta_corrector_applied': bool(prediction_audit.get('meta_corrector_applied', False))}
                    job['latest_prediction'] = prediction
                    effective_threshold = line_confidence_threshold if label == 'line' else confidence_threshold
                    if confidence >= effective_threshold:
                        job['detections'].append(prediction)
                next_inference_time += prediction_interval_seconds
            percent = processed / total_frames * 100.0 if total_frames > 0 else 0.0
            job['progress'].update({'current_time': float(current_time), 'processed_frames': int(processed), 'percent': float(min(100.0, percent))})
            job['updated_at'] = _now_iso()
        cap.release()
        job['updated_at'] = _now_iso()
        _persist_live_job_result(job)
    except Exception as exc:
        logger.error(f'Live detection job failed ({job_id}): {exc}', exc_info=True)
        job['status'] = 'error'
        job['error'] = str(exc)
        job['updated_at'] = _now_iso()
        _persist_live_job_result(job)

def _review_sessions_dir() -> Path:
    return app.config['REVIEW_FOLDER'] / 'sessions'

def _review_exports_dir() -> Path:
    return app.config['REVIEW_FOLDER'] / 'exports'

def _review_temp_dir() -> Path:
    return app.config['REVIEW_FOLDER'] / 'temp'

def _full_match_review_root() -> Path:
    """Return the isolated export root for complete-match review sessions."""
    return PROJECT_ROOT / 'data' / 'full_match_detection' / 'review_sessions'

def _training_dataset_root() -> Path:
    return app.config['TRAINING_DATASET_FOLDER']

def _training_dataset_sessions_dir() -> Path:
    return _training_dataset_root() / 'sessions'

def _training_dataset_clips_dir() -> Path:
    return _training_dataset_root() / 'clips'

def _training_dataset_temp_dir() -> Path:
    return _training_dataset_root() / 'temp'

def _training_dataset_manifest_path() -> Path:
    return _training_dataset_root() / 'train_manifest.csv'

def _training_dataset_test_manifest_path() -> Path:
    return _training_dataset_root() / 'test_manifest.csv'

def _training_dataset_review_log_path() -> Path:
    return _training_dataset_root() / 'review_log.csv'

def _training_dataset_test_review_log_path() -> Path:
    return _training_dataset_root() / 'test_review_log.csv'

def _source_origin(source_video: str | None) -> dict:
    if not source_video:
        return {'creator': 'Unknown creator', 'video_name': 'Unknown video', 'source_video': ''}
    source_path = Path(source_video)
    creator = 'Unknown creator'
    parts = list(source_path.parts)
    lowered_parts = [part.lower() for part in parts]
    if 'youtube' in lowered_parts:
        youtube_index = lowered_parts.index('youtube')
        if youtube_index + 1 < len(parts):
            creator = parts[youtube_index + 1]
    return {'creator': creator, 'video_name': source_path.name, 'source_video': source_video}

def _training_dataset_session_file(session_id: str) -> Path:
    return _training_dataset_sessions_dir() / f'training_session_{session_id}.json'

def _result_file(result_id: str) -> Path:
    return app.config['RESULTS_FOLDER'] / f'result_{result_id}.json'

def _comparison_file(comparison_id: str) -> Path:
    if not re.fullmatch('[A-Za-z0-9_-]+', comparison_id or ''):
        raise ValueError('Invalid comparison id')
    return app.config['COMPARISON_RESULTS_FOLDER'] / f'comparison_{comparison_id}.json'

def _comparison_summary(payload: dict) -> dict:
    results = payload.get('results') or {}
    total_detections = sum((len(result.get('detections') or []) for result in results.values()))
    clip_generator = None
    video_path = str(payload.get('video_path') or '')
    reference_result = next(iter(results.values()), {})
    embedded_manual = reference_result.get('manual_comparison') or {}
    if embedded_manual.get('composite_reference') and embedded_manual.get('labels'):
        manual_summary = embedded_manual.get('summary') or {}
        manual_counts = manual_summary.get('final_counts') or {}
        clip_generator = {'available': True, 'line': int(manual_counts.get('line') or 0), 'scrum': int(manual_counts.get('scrum') or 0), 'reviewed': int(manual_summary.get('reviewed') or 0), 'total': int(manual_summary.get('total_candidates') or 0)}
    if video_path and reference_result and (clip_generator is None):
        try:
            try:
                resolved_video_path = _resolve_original_result_source(video_path).resolve()
            except (FileNotFoundError, ValueError):
                resolved_video_path = _resolve_video_path(video_path).resolve()
            manual_comparison = _latest_labeled_clip_generator_comparison(resolved_video_path, reference_result)
            if manual_comparison is not None:
                manual_summary = manual_comparison.get('summary') or {}
                manual_counts = manual_summary.get('final_counts') or {}
                clip_generator = {'available': True, 'line': int(manual_counts.get('line') or 0), 'scrum': int(manual_counts.get('scrum') or 0), 'reviewed': int(manual_summary.get('reviewed') or 0), 'total': int(manual_summary.get('total_candidates') or 0)}
        except Exception as exc:
            logger.warning('Could not add manual labels to comparison %s: %s', payload.get('id'), exc)
    return {'id': payload.get('id'), 'created_at': payload.get('created_at'), 'video_path': video_path, 'video_name': payload.get('video_name') or Path(video_path or 'Video').name, 'model_names': payload.get('model_names') or list(results), 'parameters': payload.get('parameters') or {}, 'total_detections': total_detections, 'clip_generator': clip_generator}

def _review_session_file(session_id: str) -> Path:
    return _review_sessions_dir() / f'review_session_{session_id}.json'

def _load_json_file(path: Path) -> dict:
    with _resolve_model_artifact(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)

def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _resolve_model_artifact(path).open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    with RESULTS_HISTORY_CACHE_LOCK:
        RESULTS_HISTORY_CACHE['signature'] = None
        RESULTS_HISTORY_CACHE['entries'] = None

def _read_csv_rows(path: Path) -> list[dict]:
    if not _resolve_model_artifact(path).exists():
        return []
    with _resolve_model_artifact(path).open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))

def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _resolve_model_artifact(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _upsert_csv_rows(path: Path, fieldnames: list[str], new_rows: list[dict], *, key_name: str, key_value: str) -> None:
    existing_rows = [row for row in _read_csv_rows(path) if row.get(key_name) != key_value]
    _write_csv_rows(path, fieldnames, existing_rows + new_rows)

def _csv_bool(value) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}

def _resolve_video_path(video_path_str: str) -> Path:
    video_path = Path(video_path_str)
    if video_path.is_absolute() and _resolve_model_artifact(video_path).exists():
        return video_path
    if _resolve_model_artifact(app.config['UPLOAD_FOLDER'] / video_path.name).exists():
        return app.config['UPLOAD_FOLDER'] / video_path.name
    if _resolve_model_artifact(app.config['VIDEO_LIBRARY'] / video_path.name).exists():
        return app.config['VIDEO_LIBRARY'] / video_path.name
    project_relative = PROJECT_ROOT / video_path
    if _resolve_model_artifact(project_relative).exists():
        return project_relative
    return video_path

def _load_result_payload(result_id: str) -> dict:
    result_path = _result_file(result_id)
    if not _resolve_model_artifact(result_path).exists():
        raise FileNotFoundError(f'Result not found: {result_id}')
    return _load_json_file(result_path)

def _review_summary(candidates: list[dict], labels=None) -> dict:
    pending = 0
    reviewed = 0
    accepted = 0
    relabeled = 0
    summary_labels = labels if labels is not None else sorted(VALID_REVIEW_FINAL_LABELS)
    final_counts = {label: 0 for label in summary_labels}
    for candidate in candidates:
        final_label = candidate.get('final_label')
        if final_label:
            reviewed += 1
            final_counts[final_label] = final_counts.get(final_label, 0) + 1
            if final_label == candidate.get('predicted_label'):
                accepted += 1
            else:
                relabeled += 1
        else:
            pending += 1
    return {'total_candidates': len(candidates), 'pending': pending, 'reviewed': reviewed, 'accepted': accepted, 'relabeled': relabeled, 'final_counts': final_counts}

def _review_summary_for_session(session: dict) -> dict:
    """Keep blind clip-generator sessions strictly three-class."""
    labels = None
    if session.get('review_mode') == REVIEW_MODE_CLIP_GENERATOR:
        labels = ('line', 'other', 'scrum')
    return _review_summary(session.get('candidates', []), labels=labels)

def _review_candidate_from_detection(detection: dict, detection_index: int) -> dict:
    start_time = float(detection.get('start_time', 0.0))
    end_time = float(detection.get('end_time', start_time))
    predicted_label = str(detection.get('play_type', 'other'))
    probabilities = detection.get('probabilities') or detection.get('all_probabilities') or {}
    return {'candidate_index': detection_index, 'detection_index': detection_index, 'start_time': start_time, 'end_time': end_time, 'duration': float(detection.get('duration', end_time - start_time)), 'predicted_label': predicted_label, 'confidence': float(detection.get('confidence', 0.0)), 'probabilities': probabilities, 'final_label': None, 'reviewed_at': None, 'exported_clip_path': None}

def _review_source_detections(result_payload: dict) -> tuple[list[dict], str]:
    review_detections = result_payload.get('review_detections_7s') or []
    if review_detections:
        return (review_detections, 'review_detections_7s')
    window_predictions = result_payload.get('window_predictions') or []
    if window_predictions:
        return (window_predictions, 'window_predictions')
    return (result_payload.get('detections', []), 'detections')

def _temporal_overlap_seconds(left: dict, right: dict) -> float:
    start_time = max(float(left.get('start_time', 0.0)), float(right.get('start_time', 0.0)))
    end_time = min(float(left.get('end_time', 0.0)), float(right.get('end_time', 0.0)))
    return max(0.0, end_time - start_time)

def _merge_review_candidates(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (item.get('start_time', 0.0), item.get('end_time', 0.0)))
    merged_candidates: list[dict] = []
    current_group = None
    for candidate in ordered:
        candidate = dict(candidate)
        candidate['source_detection_indices'] = [candidate['detection_index']]
        candidate['merged_detection_count'] = 1
        candidate['_merged_probabilities'] = [candidate.get('probabilities', {})] if candidate.get('probabilities') else []
        if current_group is None:
            current_group = candidate
            continue
        same_label = candidate.get('predicted_label') == current_group.get('predicted_label')
        overlapping = _temporal_overlap_seconds(current_group, candidate) > REVIEW_OVERLAP_MERGE_SECONDS
        if same_label and overlapping:
            current_group['start_time'] = min(float(current_group['start_time']), float(candidate['start_time']))
            current_group['end_time'] = max(float(current_group['end_time']), float(candidate['end_time']))
            current_group['duration'] = float(current_group['end_time']) - float(current_group['start_time'])
            current_group['confidence'] = max(float(current_group.get('confidence', 0.0)), float(candidate.get('confidence', 0.0)))
            if candidate.get('probabilities'):
                current_group.setdefault('_merged_probabilities', []).append(candidate['probabilities'])
            current_group['source_detection_indices'].append(candidate['detection_index'])
            current_group['merged_detection_count'] += 1
            continue
        merged_candidates.append(current_group)
        current_group = candidate
    if current_group is not None:
        merged_candidates.append(current_group)
    for candidate in merged_candidates:
        merged_probabilities = candidate.pop('_merged_probabilities', [])
        if merged_probabilities:
            combined: dict[str, float] = {}
            for probabilities in merged_probabilities:
                for label, probability in probabilities.items():
                    combined[label] = combined.get(label, 0.0) + float(probability)
            count = float(len(merged_probabilities))
            candidate['probabilities'] = {label: value / count for label, value in combined.items()}
        candidate['primary_detection_index'] = candidate['source_detection_indices'][0]
        candidate['candidate_index'] = candidate['primary_detection_index']
        candidate['detection_index'] = candidate['primary_detection_index']
    return merged_candidates

def _non_overlapping_review_candidates(result_payload: dict, block_duration: float=REVIEW_BLOCK_DURATION_SECONDS) -> list[dict]:
    """Build fixed, consecutive review blocks without temporal overlap.

    The detector still produces overlapping evidence windows (for inference),
    but the reviewer sees blocks anchored at 0, ``block_duration``,
    ``2 * block_duration``, ... .  Probabilities from evidence windows whose
    start belongs to each block are averaged before choosing the displayed
    label.  This keeps the old overlapping review queue untouched and gives a
    separate, deterministic review protocol for efficient whole-video review.
    Unlike the active-learning queue, this mode includes ``other`` blocks too,
    so the reviewer sees the complete sequence starting at the beginning of
    the video.
    """
    windows = result_payload.get('window_predictions') or result_payload.get('review_detections_7s') or []
    if not windows or block_duration <= 0:
        return []
    metadata = result_payload.get('metadata') or {}
    duration = float(metadata.get('duration') or 0.0)
    if duration <= 0.0:
        duration = max((float(item.get('end_time', 0.0)) for item in windows))
    grouped: dict[int, list[tuple[int, dict]]] = {}
    for source_index, window in enumerate(windows):
        start_time = max(0.0, float(window.get('start_time', 0.0)))
        block_index = int(start_time // block_duration)
        grouped.setdefault(block_index, []).append((source_index, window))
    candidates: list[dict] = []
    for block_index in sorted(grouped):
        members = grouped[block_index]
        block_start = block_index * block_duration
        block_end = min(duration, block_start + block_duration)
        if block_end <= block_start:
            continue
        probability_values: dict[str, list[float]] = {}
        primary_probability_values: dict[str, list[float]] = {}
        labels: list[str] = []
        primary_labels: list[str] = []
        for _, member in members:
            probabilities = member.get('probabilities') or {}
            for label, value in probabilities.items():
                probability_values.setdefault(str(label), []).append(float(value))
            if probabilities:
                labels.append(max(probabilities, key=probabilities.get))
            primary_probabilities = member.get('primary_probabilities') or member.get('original_probabilities') or {}
            for label, value in primary_probabilities.items():
                primary_probability_values.setdefault(str(label), []).append(float(value))
            if primary_probabilities:
                primary_labels.append(max(primary_probabilities, key=primary_probabilities.get))
        averaged_probabilities = {label: sum(values) / len(values) for label, values in probability_values.items() if values}
        if averaged_probabilities:
            predicted_label = max(averaged_probabilities, key=averaged_probabilities.get)
            confidence = float(averaged_probabilities[predicted_label])
        else:
            predicted_label = Counter(labels).most_common(1)[0][0] if labels else 'other'
            confidence = 0.0
        averaged_primary = {label: sum(values) / len(values) for label, values in primary_probability_values.items() if values}
        primary_label = max(averaged_primary, key=averaged_primary.get) if averaged_primary else Counter(primary_labels).most_common(1)[0][0] if primary_labels else None
        primary_confidence = float(averaged_primary.get(primary_label, 0.0)) if primary_label else None
        source_indices = [source_index for source_index, _ in members]
        candidates.append({'candidate_index': block_index, 'detection_index': block_index, 'primary_detection_index': source_indices[0], 'source_detection_indices': source_indices, 'merged_detection_count': len(source_indices), 'start_time': float(block_start), 'end_time': float(block_end), 'duration': float(block_end - block_start), 'predicted_label': str(predicted_label), 'is_event_target': predicted_label in TARGET_REVIEW_LABELS, 'confidence': confidence, 'probabilities': averaged_probabilities, 'primary_play_type': primary_label, 'primary_confidence': primary_confidence, 'primary_probabilities': averaged_primary, 'aggregation': 'mean_overlapping_windows_in_fixed_non_overlapping_block', 'review_block_index': block_index, 'final_label': None, 'reviewed_at': None, 'exported_clip_path': None})
    for queue_index, candidate in enumerate(candidates):
        candidate['queue_index'] = queue_index
    return candidates

def _find_existing_review_session(result_id: str, review_mode: str=REVIEW_MODE_OVERLAP) -> dict | None:
    latest_session = None
    latest_updated_at = ''
    for session_file in sorted(_model_artifact_glob(_review_sessions_dir(), 'review_session_*.json', recursive=False)):
        try:
            session = _load_json_file(session_file)
        except Exception:
            continue
        if session.get('result_id') != result_id:
            continue
        stored_mode = session.get('review_mode', REVIEW_MODE_OVERLAP)
        if stored_mode != review_mode:
            continue
        updated_at = session.get('updated_at', '')
        if updated_at >= latest_updated_at:
            latest_session = session
            latest_updated_at = updated_at
    return latest_session

def _persist_review_session(session: dict) -> dict:
    session['updated_at'] = datetime.now().isoformat(timespec='seconds')
    session['summary'] = _review_summary_for_session(session)
    _write_json_file(_review_session_file(session['session_id']), session)
    return session

def _load_review_session(session_id: str) -> dict:
    session_path = _review_session_file(session_id)
    if not _resolve_model_artifact(session_path).exists():
        raise FileNotFoundError(f'Review session not found: {session_id}')
    return _load_json_file(session_path)

def _create_review_session(result_id: str, force_new: bool=False, review_mode: str=REVIEW_MODE_OVERLAP) -> dict:
    if review_mode not in VALID_REVIEW_MODES:
        raise ValueError(f'review_mode must be one of {sorted(VALID_REVIEW_MODES)}')
    if not force_new:
        existing = _find_existing_review_session(result_id, review_mode=review_mode)
        if existing:
            existing['summary'] = _review_summary(existing.get('candidates', []))
            return existing
    result_payload = _load_result_payload(result_id)
    if review_mode in {REVIEW_MODE_NON_OVERLAPPING, REVIEW_MODE_FULL_MATCH}:
        candidates = _non_overlapping_review_candidates(result_payload)
        review_source = 'full_match_fixed_blocks' if review_mode == REVIEW_MODE_FULL_MATCH else 'non_overlapping_fixed_blocks'
    else:
        detections, review_source = _review_source_detections(result_payload)
        raw_candidates = [_review_candidate_from_detection(detection, detection_index) for detection_index, detection in enumerate(detections) if str(detection.get('play_type', 'other')) in TARGET_REVIEW_LABELS]
        candidates = raw_candidates if review_source in {'review_detections_7s', 'window_predictions'} else _merge_review_candidates(raw_candidates)
    if not candidates:
        raise ValueError('No scrum/line detections are available for review in this result.')
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    video_path = _resolve_original_result_source(result_payload.get('video_path', ''))
    session = {'session_id': session_id, 'result_id': result_id, 'created_at': datetime.now().isoformat(timespec='seconds'), 'updated_at': datetime.now().isoformat(timespec='seconds'), 'status': 'in_progress', 'video_path': str(video_path), 'video_name': video_path.name, 'model_name': result_payload.get('model_name'), 'source_statistics': result_payload.get('statistics', {}), 'review_mode': review_mode, 'review_mode_label': 'Partido completo: bloques fijos de 7 segundos' if review_mode == REVIEW_MODE_FULL_MATCH else 'Bloques fijos sin solapamiento' if review_mode == REVIEW_MODE_NON_OVERLAPPING else 'Ventanas superpuestas originales', 'review_source': review_source, 'review_clip_duration_seconds': REVIEW_BLOCK_DURATION_SECONDS, 'review_block_duration_seconds': REVIEW_BLOCK_DURATION_SECONDS if review_mode in {REVIEW_MODE_NON_OVERLAPPING, REVIEW_MODE_FULL_MATCH} else None, 'candidates': candidates, 'exports': {'last_export_at': None, 'export_dir': None, 'manifest_path': None, 'review_log_path': None, 'exported_count': 0}, 'summary': _review_summary(candidates)}
    _write_json_file(_review_session_file(session_id), session)
    return session

def _manual_clip_candidates(duration: float, block_duration: float=REVIEW_BLOCK_DURATION_SECONDS) -> list[dict]:
    """Build consecutive manual-labeling clips that cover the whole video."""
    if duration <= 0 or block_duration <= 0:
        return []
    candidates = []
    block_count = int(math.ceil(duration / block_duration))
    for block_index in range(block_count):
        start_time = block_index * block_duration
        end_time = min(duration, start_time + block_duration)
        if end_time <= start_time:
            continue
        candidates.append({'candidate_index': block_index, 'queue_index': block_index, 'detection_index': block_index, 'primary_detection_index': block_index, 'source_detection_indices': [], 'merged_detection_count': 0, 'start_time': float(start_time), 'end_time': float(end_time), 'duration': float(end_time - start_time), 'predicted_label': 'other', 'confidence': 0.0, 'probabilities': {}, 'manual_labeling': True, 'final_label': None, 'reviewed_at': None, 'exported_clip_path': None})
    return candidates

def _analysis_block_predictions(result_payload: dict, candidates: list[dict]) -> list[dict]:
    """Align one saved analysis with the manual seven-second review blocks."""
    fixed_candidates = _non_overlapping_review_candidates(result_payload)
    fixed_by_index = {int(item.get('review_block_index', item.get('candidate_index', -1))): item for item in fixed_candidates}
    detections = result_payload.get('detections') or []
    predictions = []
    for block_index, candidate in enumerate(candidates):
        fixed = fixed_by_index.get(block_index)
        if fixed:
            label = str(fixed.get('predicted_label') or 'other')
            confidence = float(fixed.get('confidence') or 0.0)
            probabilities = fixed.get('probabilities') or {}
            source = 'fixed_block_average'
        else:
            block_start = float(candidate['start_time'])
            block_end = float(candidate['end_time'])
            overlapping = []
            for detection in detections:
                start_time = float(detection.get('start_time', 0.0))
                end_time = float(detection.get('end_time', start_time))
                overlap = max(0.0, min(block_end, end_time) - max(block_start, start_time))
                if overlap > 0:
                    overlapping.append((overlap, float(detection.get('confidence', 0.0)), detection))
            if overlapping:
                _, confidence, detection = max(overlapping, key=lambda item: (item[0], item[1]))
                label = str(detection.get('play_type') or 'other')
                probabilities = detection.get('probabilities') or detection.get('all_probabilities') or {}
                source = 'overlapping_detection'
            else:
                label = 'other'
                confidence = 0.0
                probabilities = {}
                source = 'no_detection'
        predictions.append({'block_index': block_index, 'start_time': float(candidate['start_time']), 'end_time': float(candidate['end_time']), 'predicted_label': label, 'confidence': confidence, 'probabilities': probabilities, 'source': source})
    return predictions

def _result_detection_statistics(detections: list[dict]) -> dict:
    """Build the result statistics consumed by the web summary cards."""
    if not detections:
        return {'avg_confidence': 0.0, 'avg_duration': 0.0, 'by_type': {}}
    by_type = {}
    for detection in detections:
        label = str(detection.get('play_type') or 'other')
        duration = float(detection.get('duration') or float(detection.get('end_time', 0.0)) - float(detection.get('start_time', 0.0)))
        bucket = by_type.setdefault(label, {'count': 0, 'total_duration': 0.0})
        bucket['count'] += 1
        bucket['total_duration'] += duration
    return {'avg_confidence': sum((float(item.get('confidence', 0.0)) for item in detections)) / len(detections), 'avg_duration': sum((float(item.get('duration') or float(item.get('end_time', 0.0)) - float(item.get('start_time', 0.0))) for item in detections)) / len(detections), 'by_type': by_type}

def _rethreshold_result_payload(result_payload: dict, confidence_threshold: float, line_confidence_threshold: float) -> dict:
    """Rebuild event detections from cached window probabilities without inference."""
    if (result_payload.get('metadata') or {}).get('saved_decisions_locked'):
        raise ValueError('Este resultado conserva las decisiones del experimento. Elegí otra variante guardada para comparar umbrales.')
    confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
    line_confidence_threshold = max(0.0, min(1.0, float(line_confidence_threshold)))
    windows = result_payload.get('window_predictions') or []
    used_cached_windows = bool(windows)
    line_threshold_controlled_by_model = False
    fixed_line_threshold = None
    if windows and KERAS_AVAILABLE:
        policy = result_payload.get('event_post_processing') or (result_payload.get('metadata') or {}).get('event_post_processing') or {}
        detector = object.__new__(RugbyPlayDetector)
        detector.EVENT_TARGET_LABELS = tuple(policy.get('target_labels') or ('scrum', 'line'))
        detector.EVENT_CONTEXT_RADIUS_SECONDS = float(policy.get('context_radius_seconds', 0.0))
        detector.EVENT_MERGE_GAP_SECONDS = float(policy.get('merge_gap_seconds', 1.5))
        detector.EVENT_ANCHOR_CONFIDENCE = float(policy.get('anchor_confidence', 0.85))
        detector.EVENT_NEIGHBOR_BOOST = float(policy.get('neighbor_boost', 0.0))
        detector.EVENT_REQUIRE_TARGET_WINNER = bool(policy.get('require_target_winner', True))
        detector.EVENT_MERGE_BY_START_TIME = bool(policy.get('merge_by_start_time', True))
        detector.EVENT_MIN_TARGET_CONFIDENCE = float(policy.get('min_target_confidence', 0.0))
        detector.EVENT_MIN_SUPPORT_WINDOWS = int(policy.get('min_support_windows', 1))
        detector.EVENT_MAX_DURATION_SECONDS = policy.get('max_duration_seconds', 15.0)
        metadata = result_payload.get('metadata') or {}
        duration = float(metadata.get('duration') or 0.0)
        if duration <= 0.0:
            duration = max((float(item.get('end_time', 0.0)) for item in windows), default=0.0)
        source_fps = float(result_payload.get('source_fps') or metadata.get('source_fps') or 25.0)
        detections = detector._post_process_event_detections(windows, confidence_threshold=confidence_threshold, video_duration=duration, source_fps=source_fps, class_confidence_thresholds={'line': line_confidence_threshold})
        if policy.get('line_mode') == 'event_logistic_verifier':
            artifact_relative = policy.get('line_verifier_artifact')
            if not artifact_relative:
                raise ValueError('The saved result does not identify its line verifier artifact.')
            fixed_line_threshold = float(policy.get('line_verifier_threshold', 0.11))
            verifier = LineEventVerifier(PROJECT_ROOT / str(artifact_relative), threshold=fixed_line_threshold, historical_weight=float(policy.get('line_historical_weight', 0.4)), static_weight=float(policy.get('line_static_weight', 0.6)))
            line_events, _ = verifier.verify(windows, video_duration=duration, source_fps=source_fps)
            detections = [detection for detection in detections if str(detection.get('play_type')) != 'line'] + line_events
            detections.sort(key=lambda item: float(item.get('start_time', 0.0)))
            line_threshold_controlled_by_model = True
    else:
        detections = [dict(item) for item in result_payload.get('detections', []) if float(item.get('confidence', 0.0)) >= (line_confidence_threshold if str(item.get('play_type')) == 'line' else confidence_threshold)]
    statistics = _result_detection_statistics(detections)
    plays_by_type = {label: int(values.get('count', 0)) for label, values in statistics['by_type'].items()}
    summary = {'total_detections': len(detections), 'plays_by_type': plays_by_type, 'average_confidence': statistics['avg_confidence'], 'predicted_class': 'unknown', 'overall_confidence': 0.0}
    policy_minimum = float((result_payload.get('event_post_processing') or {}).get('min_target_confidence', 0.0))
    return {'detections': detections, 'statistics': statistics, 'summary': summary, 'thresholds': {'confidence_threshold': confidence_threshold, 'line_confidence_threshold': line_confidence_threshold, 'effective_scrum_threshold': max(confidence_threshold, policy_minimum), 'effective_line_threshold': fixed_line_threshold if line_threshold_controlled_by_model else max(confidence_threshold, line_confidence_threshold, policy_minimum), 'line_threshold_controlled_by_model': line_threshold_controlled_by_model}, 'used_cached_windows': used_cached_windows}

def _analysis_comparisons(video_path: Path, *, limit: int | None=None) -> list[dict]:
    """Return saved analyses whose resolved source is ``video_path``, newest first."""
    target_path = video_path.resolve()
    duration = float(read_video_metadata(video_path).get('duration') or 0.0)
    candidates = _manual_clip_candidates(duration)
    comparisons = []
    for result_file in sorted(_model_artifact_glob(app.config['RESULTS_FOLDER'], 'result_*.json', recursive=False), reverse=True):
        try:
            result_payload = _load_json_file(result_file)
            raw_path = str(result_payload.get('video_path') or '')
            try:
                result_video = _resolve_original_result_source(raw_path).resolve()
            except (FileNotFoundError, ValueError):
                result_video = _resolve_video_path(raw_path).resolve()
            if result_video != target_path:
                continue
            result_id = result_file.stem[len('result_'):]
            detection_counts = Counter((str(item.get('play_type') or 'other') for item in result_payload.get('detections') or []))
            comparisons.append({'result_id': result_id, 'model_name': result_payload.get('model_name') or 'Modelo analizado', 'predictions': _analysis_block_predictions(result_payload, candidates), 'total_detections': len(result_payload.get('detections') or []), 'detection_counts': dict(detection_counts), 'block_duration_seconds': REVIEW_BLOCK_DURATION_SECONDS, 'video_duration': duration, 'analysis_url': f"/viewer?result_id={quote(result_id, safe='')}"})
            if limit is not None and len(comparisons) >= limit:
                break
        except Exception as exc:
            logger.warning('Could not build clip comparison from %s: %s', result_file.name, exc)
    return comparisons

def _latest_analysis_comparison(video_path: Path) -> dict | None:
    """Return the newest saved analysis whose resolved source is ``video_path``."""
    comparisons = _analysis_comparisons(video_path, limit=1)
    return comparisons[0] if comparisons else None

def _latest_labeled_clip_generator_comparison(video_path: Path, result_payload: dict | None=None) -> dict | None:
    """Return accumulated manual labels with newer explicit corrections applied."""
    target_identity = _video_file_identity_key(video_path)
    labeled_sessions = []
    for session_file in _model_artifact_glob(_review_sessions_dir(), 'review_session_*.json', recursive=False):
        try:
            session = _load_json_file(session_file)
            if session.get('review_mode') != REVIEW_MODE_CLIP_GENERATOR:
                continue
            session_path = _resolve_video_path(str(session.get('video_path') or '')).resolve()
            if _video_file_identity_key(session_path) != target_identity:
                continue
            summary = _review_summary_for_session(session)
            if int(summary.get('reviewed') or 0) <= 0:
                continue
            labeled_sessions.append(session)
        except Exception as exc:
            logger.warning('Could not inspect clip generator session %s: %s', session_file.name, exc)
    latest_session = consolidate_clip_sessions(labeled_sessions)
    if latest_session is None:
        return None
    latest_session['summary'] = _review_summary_for_session(latest_session)
    candidates = latest_session.get('candidates', [])
    predictions = _analysis_block_predictions(result_payload or {}, candidates)
    predictions_by_index = {int(item.get('block_index', index)): item for index, item in enumerate(predictions)}
    labels = []
    matches = 0
    reviewed = 0
    for index, candidate in enumerate(candidates):
        block_index = int(candidate.get('candidate_index', index))
        final_label = candidate.get('final_label')
        predicted_label = predictions_by_index.get(block_index, {}).get('predicted_label')
        if final_label:
            reviewed += 1
            if final_label == predicted_label:
                matches += 1
        labels.append({'block_index': block_index, 'start_time': float(candidate.get('start_time', block_index * REVIEW_BLOCK_DURATION_SECONDS)), 'end_time': float(candidate.get('end_time', (block_index + 1) * REVIEW_BLOCK_DURATION_SECONDS)), 'final_label': final_label, 'predicted_label': predicted_label, 'source_session_id': candidate.get('source_session_id')})
    return {'session_id': latest_session['session_id'], 'composite_reference': latest_session['composite_reference'], 'source_session_ids': latest_session['source_session_ids'], 'reference_id': latest_session['reference_id'], 'reference_sha256': latest_session['reference_sha256'], 'dataset_split': latest_session.get('dataset_split', 'train'), 'updated_at': latest_session.get('updated_at'), 'labels': labels, 'summary': latest_session['summary'], 'agreement': {'matches': matches, 'reviewed': reviewed, 'percentage': round(matches / reviewed * 100, 1) if reviewed else None}, 'clip_generator_url': f"/clip-generator?session_id={quote(latest_session['session_id'], safe='')}"}

def _video_file_identity_key(video_path: Path) -> str:
    """Identify hard-linked video aliases as the same underlying source file."""
    resolved_video = video_path.resolve()
    try:
        stat = resolved_video.stat()
    except OSError:
        return f'path:{str(resolved_video).casefold()}'
    if stat.st_ino:
        return f'file:{stat.st_dev}:{stat.st_ino}'
    return f'path:{str(resolved_video).casefold()}'

def _find_existing_clip_generator_session(video_path: Path, dataset_split: str) -> dict | None:
    video_identity = _video_file_identity_key(video_path)
    latest_session = None
    latest_updated_at = ''
    for session_file in sorted(_model_artifact_glob(_review_sessions_dir(), 'review_session_*.json', recursive=False)):
        try:
            session = _load_json_file(session_file)
        except Exception:
            continue
        if session.get('review_mode') != REVIEW_MODE_CLIP_GENERATOR:
            continue
        if session.get('dataset_split', 'train') != dataset_split:
            continue
        if _video_file_identity_key(Path(session.get('video_path', ''))) != video_identity:
            continue
        summary = _review_summary_for_session(session)
        if summary.get('pending', 0) <= 0:
            continue
        updated_at = session.get('updated_at', '')
        if updated_at >= latest_updated_at:
            session['summary'] = summary
            latest_session = session
            latest_updated_at = updated_at
    return latest_session

def _clip_generator_video_summaries() -> dict[str, dict]:
    """Return the most useful manual-label session summary for each source video."""
    summaries = {}
    for session_file in _model_artifact_glob(_review_sessions_dir(), 'review_session_*.json', recursive=False):
        try:
            session = _load_json_file(session_file)
            if session.get('review_mode') != REVIEW_MODE_CLIP_GENERATOR:
                continue
            video_path = _resolve_video_path(str(session.get('video_path') or '')).resolve()
            summary = _review_summary_for_session(session)
            reviewed = int(summary.get('reviewed') or 0)
            total = int(summary.get('total_candidates') or 0)
            pending = int(summary.get('pending') or 0)
            final_counts = summary.get('final_counts') or {}
            session_id = str(session.get('session_id') or '')
            item = {'session_id': session_id, 'dataset_split': session.get('dataset_split', 'train'), 'updated_at': session.get('updated_at'), 'status': 'complete' if total > 0 and pending <= 0 else 'in_progress' if reviewed > 0 else 'empty', 'reviewed': reviewed, 'total': total, 'pending': pending, 'line': int(final_counts.get('line') or 0), 'scrum': int(final_counts.get('scrum') or 0), 'clip_generator_url': f"/clip-generator?session_id={quote(session_id, safe='')}"}
            key = _video_file_identity_key(video_path)
            current = summaries.get(key)
            item_rank = (reviewed, str(session.get('updated_at') or ''))
            current_rank = (int(current.get('reviewed') or 0), str(current.get('updated_at') or '')) if current else (-1, '')
            if item_rank >= current_rank:
                summaries[key] = item
        except Exception as exc:
            logger.warning('Could not summarize clip generator session %s: %s', session_file.name, exc)
    return summaries

def _create_clip_generator_session(video_path_str: str, dataset_split: str='train', force_new: bool=False) -> dict:
    if dataset_split not in {'train', 'test'}:
        raise ValueError('dataset_split must be train or test')
    video_path = _resolve_video_path(video_path_str).resolve()
    if not _resolve_model_artifact(video_path).exists() or not _resolve_model_artifact(video_path).is_file():
        raise FileNotFoundError(f'Video not found: {video_path_str}')
    allowed_roots = [app.config['VIDEO_LIBRARY'].resolve(), app.config['UPLOAD_FOLDER'].resolve()]
    if not any((root == video_path.parent or root in video_path.parents for root in allowed_roots)):
        raise ValueError('The selected video is outside the available video library')
    if not force_new:
        existing = _find_existing_clip_generator_session(video_path, dataset_split)
        if existing:
            return existing
    duration = float(read_video_metadata(video_path).get('duration') or 0.0)
    candidates = _manual_clip_candidates(duration)
    if not candidates:
        raise ValueError('The selected video does not contain a complete 7-second block')
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    session = {'session_id': session_id, 'result_id': f'manual_{session_id}', 'created_at': datetime.now().isoformat(timespec='seconds'), 'updated_at': datetime.now().isoformat(timespec='seconds'), 'status': 'in_progress', 'video_path': str(video_path), 'video_name': video_path.name, 'video_duration': duration, 'ignored_tail_seconds': max(0.0, duration - len(candidates) * REVIEW_BLOCK_DURATION_SECONDS), 'model_name': 'Etiquetado manual', 'source_statistics': {}, 'review_mode': REVIEW_MODE_CLIP_GENERATOR, 'review_mode_label': 'Generador manual: bloques consecutivos de 7 segundos', 'review_source': 'manual_fixed_blocks', 'dataset_split': dataset_split, 'clip_export_mode': 'source_resolution_25fps', 'rgb_preprocessing_version': SPATIAL_PREPROCESSING_VERSION, 'rgb_input_size': MODEL_INPUT_SIZE, 'yolo_preprocessing_version': YOLO_SOURCE_ROI_VERSION, 'yolo_input_size': 640, 'review_clip_duration_seconds': REVIEW_BLOCK_DURATION_SECONDS, 'review_block_duration_seconds': REVIEW_BLOCK_DURATION_SECONDS, 'candidates': candidates, 'exports': {'last_export_at': None, 'export_dir': None, 'manifest_path': None, 'review_log_path': None, 'exported_count': 0}, 'summary': _review_summary(candidates, labels=('line', 'other', 'scrum'))}
    _write_json_file(_review_session_file(session_id), session)
    return session

def _review_session_video_url(session: dict) -> str:
    if session.get('review_mode') == REVIEW_MODE_CLIP_GENERATOR:
        encoded_path = quote(str(session.get('video_path', '')), safe='')
        return f'/api/video-by-path?video_path={encoded_path}'
    return f"/api/video-for-result/{session['result_id']}"

def _transcode_video_for_browser(raw_path: Path, out_path: Path) -> None:
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        cmd = [ffmpeg_path, '-y', '-i', str(raw_path), '-c:v', 'libx264', '-preset', 'fast', '-crf', '28', '-pix_fmt', 'yuv420p', str(out_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            raw_path.unlink(missing_ok=True)
            return
        logger.warning('ffmpeg re-encode failed for %s: %s', out_path.name, result.stderr)
    if _resolve_model_artifact(out_path).exists():
        out_path.unlink()
    raw_path.replace(out_path)

def _write_segment_clip(video_path: Path, start_sec: float, end_sec: float, out_path: Path, *, crop_for_training: bool, target_size: int=224) -> tuple[bool, str | None]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (False, f'Cannot open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame + 1, int(end_sec * fps))
    frame_size = (target_size, target_size) if crop_for_training else (width, height)
    output_fps = target_fps
    clip_duration = max(0.0, float(end_sec) - float(start_sec))
    output_frame_count = max(1, int(np.floor(clip_duration * target_fps)))
    output_indices = np.rint(start_frame + np.arange(output_frame_count, dtype=np.float64) * fps / target_fps).astype(np.int64)
    output_indices = np.clip(output_indices, start_frame, end_frame - 1)
    output_counts = Counter((int(index) for index in output_indices.tolist()))
    if crop_for_training:
        try:
            training_normalization_geometry(width, height, target_size=target_size)
        except ValueError as exc:
            cap.release()
            return (False, str(exc))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix('.raw.mp4')
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*'mp4v'), output_fps, frame_size)
    if not writer.isOpened():
        cap.release()
        return (False, f'Cannot create writer for {out_path}')
    wrote_frames = 0
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        repeat_count = output_counts.get(frame_idx, 0)
        if repeat_count <= 0:
            frame_idx += 1
            continue
        if crop_for_training:
            frame = _training_center_crop(frame, target_size=target_size)
            if frame is None:
                writer.release()
                cap.release()
                raw_path.unlink(missing_ok=True)
                return (False, 'Frame is too small for the training center crop.')
        for _ in range(repeat_count):
            writer.write(frame)
            wrote_frames += 1
        frame_idx += 1
    writer.release()
    cap.release()
    if wrote_frames == 0:
        raw_path.unlink(missing_ok=True)
        return (False, 'No frames were written to the clip.')
    _transcode_video_for_browser(raw_path, out_path)
    return (True, None)

def _export_review_session(session: dict) -> dict:
    isolated_full_match = session.get('review_mode') == REVIEW_MODE_FULL_MATCH
    dataset_split = session.get('dataset_split', 'train')
    is_test_split = session.get('review_mode') == REVIEW_MODE_CLIP_GENERATOR and dataset_split == 'test'
    central_manifest_path = _training_dataset_test_manifest_path() if is_test_split else _training_dataset_manifest_path()
    central_review_log_path = _training_dataset_test_review_log_path() if is_test_split else _training_dataset_review_log_path()
    export_root = _full_match_review_root() / session['session_id'] if isolated_full_match else _training_dataset_sessions_dir() / session['session_id']
    clips_dir = export_root / 'clips'
    export_root.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    source_video = _resolve_original_result_source(session.get('video_path', ''))
    if not _resolve_model_artifact(source_video).exists():
        raise FileNotFoundError(f'Source video not found for export: {source_video}')
    source_metadata = read_video_metadata(source_video)
    preserve_source_resolution = session.get('review_mode') == REVIEW_MODE_CLIP_GENERATOR
    clip_width = int(source_metadata.get('width') or 0) if preserve_source_resolution else MODEL_INPUT_SIZE
    clip_height = int(source_metadata.get('height') or 0) if preserve_source_resolution else MODEL_INPUT_SIZE
    manifest_rows = []
    review_rows = []
    exported_count = 0
    correct_count = 0
    corrected_count = 0
    manifest_columns = ['dataset_id', 'session_id', 'result_id', 'source_video', 'source_model', 'candidate_index', 'detection_index', 'start_time', 'end_time', 'duration', 'predicted_label', 'correct_label', 'prediction_is_correct', 'review_outcome', 'confidence', 'reviewed_at', 'clip_path', 'clip_name', 'clip_spatial_mode', 'clip_width', 'clip_height', 'clip_fps', 'rgb_preprocessing_version', 'rgb_input_size', 'yolo_preprocessing_version', 'yolo_input_size']
    log_columns = manifest_columns + ['predicted_probabilities']
    for queue_index, candidate in enumerate(session.get('candidates', [])):
        final_label = candidate.get('final_label')
        if final_label not in VALID_REVIEW_FINAL_LABELS:
            continue
        predicted_label = str(candidate.get('predicted_label', 'other'))
        is_correct = predicted_label == final_label
        if is_correct:
            correct_count += 1
        else:
            corrected_count += 1
        start_ms = int(float(candidate['start_time']) * 1000)
        end_ms = int(float(candidate['end_time']) * 1000)
        review_outcome = 'correct' if is_correct else 'corrected'
        clip_name = f"{source_video.stem}_det{candidate['detection_index']:04d}_{start_ms:08d}_{end_ms:08d}_{final_label}_{review_outcome}.mp4"
        clip_path = clips_dir / clip_name
        ok, error_message = _write_segment_clip(source_video, float(candidate['start_time']), float(candidate['end_time']), clip_path, crop_for_training=not preserve_source_resolution, target_size=MODEL_INPUT_SIZE)
        if not ok:
            raise RuntimeError(error_message or f'Failed to export clip for candidate {queue_index}')
        candidate['exported_clip_path'] = str(clip_path)
        dataset_row = {'dataset_id': f"{session['session_id']}_{candidate['detection_index']:04d}", 'session_id': session['session_id'], 'result_id': session['result_id'], 'source_video': str(source_video), 'source_model': session.get('model_name'), 'candidate_index': queue_index, 'detection_index': candidate['detection_index'], 'start_time': round(float(candidate['start_time']), 3), 'end_time': round(float(candidate['end_time']), 3), 'duration': round(float(candidate.get('duration', 0.0)), 3), 'predicted_label': predicted_label, 'correct_label': final_label, 'prediction_is_correct': str(is_correct).lower(), 'review_outcome': review_outcome, 'confidence': round(float(candidate['confidence']), 6), 'reviewed_at': candidate.get('reviewed_at'), 'clip_path': str(clip_path), 'clip_name': clip_name, 'clip_spatial_mode': 'source_resolution' if preserve_source_resolution else 'rgb_training_crop', 'clip_width': clip_width, 'clip_height': clip_height, 'clip_fps': 25.0, 'rgb_preprocessing_version': SPATIAL_PREPROCESSING_VERSION, 'rgb_input_size': MODEL_INPUT_SIZE, 'yolo_preprocessing_version': YOLO_SOURCE_ROI_VERSION if preserve_source_resolution else 'upsampled_224', 'yolo_input_size': 640}
        manifest_rows.append(dataset_row)
        review_rows.append({**dataset_row, 'predicted_probabilities': json.dumps(candidate.get('probabilities', {}), ensure_ascii=False)})
        exported_count += 1
    session_manifest_path = export_root / f'{dataset_split}_manifest.csv'
    session_review_log_path = export_root / f'{dataset_split}_review_log.csv'
    _write_csv_rows(session_manifest_path, manifest_columns, manifest_rows)
    _write_csv_rows(session_review_log_path, log_columns, review_rows)
    if not isolated_full_match:
        _upsert_csv_rows(central_manifest_path, manifest_columns, manifest_rows, key_name='session_id', key_value=session['session_id'])
        _upsert_csv_rows(central_review_log_path, log_columns, review_rows, key_name='session_id', key_value=session['session_id'])
    session_snapshot = {'session_id': session['session_id'], 'result_id': session['result_id'], 'review_mode': session.get('review_mode'), 'dataset_split': dataset_split, 'created_at': session.get('created_at'), 'last_export_at': datetime.now().isoformat(timespec='seconds'), 'exported_count': exported_count, 'correct_count': correct_count, 'corrected_count': corrected_count, 'source_video': str(source_video), 'model_name': session.get('model_name'), 'clip_spatial_mode': 'source_resolution' if preserve_source_resolution else 'rgb_training_crop', 'clip_resolution': [clip_width, clip_height], 'clip_fps': 25.0, 'rgb_preprocessing_version': SPATIAL_PREPROCESSING_VERSION, 'rgb_input_size': MODEL_INPUT_SIZE, 'yolo_preprocessing_version': YOLO_SOURCE_ROI_VERSION if preserve_source_resolution else 'upsampled_224', 'yolo_input_size': 640, 'export_dir': str(export_root), 'manifest_path': str(session_manifest_path), 'review_log_path': str(session_review_log_path)}
    if isolated_full_match:
        _write_json_file(export_root / 'session_snapshot.json', session_snapshot)
    else:
        _write_json_file(_training_dataset_session_file(session['session_id']), session_snapshot)
    session['exports'] = {'last_export_at': datetime.now().isoformat(timespec='seconds'), 'export_dir': str(export_root), 'manifest_path': str(session_manifest_path), 'review_log_path': str(session_review_log_path), 'central_manifest_path': None if isolated_full_match else str(central_manifest_path), 'central_review_log_path': None if isolated_full_match else str(central_review_log_path), 'isolated_experiment': isolated_full_match, 'exported_count': exported_count, 'correct_count': correct_count, 'corrected_count': corrected_count}
    session['status'] = 'exported' if session['summary']['pending'] == 0 else 'in_progress'
    _persist_review_session(session)
    return session

def _training_dataset_summary() -> dict:
    manifest_path = _training_dataset_manifest_path()
    rows = _read_csv_rows(manifest_path)
    total_clips = len(rows)
    correct_clips = sum((1 for row in rows if _csv_bool(row.get('prediction_is_correct'))))
    corrected_clips = total_clips - correct_clips
    session_ids = {row.get('session_id') for row in rows if row.get('session_id')}
    result_ids = {row.get('result_id') for row in rows if row.get('result_id')}
    label_counts = {'scrum': 0, 'line': 0, 'other': 0}
    review_outcome_counts = {'correct': 0, 'corrected': 0}
    origin_counts = {}
    for row in rows:
        correct_label = row.get('correct_label') or row.get('tag') or 'other'
        label_counts[correct_label] = label_counts.get(correct_label, 0) + 1
        review_outcome = row.get('review_outcome') or ('correct' if _csv_bool(row.get('prediction_is_correct')) else 'corrected')
        review_outcome_counts[review_outcome] = review_outcome_counts.get(review_outcome, 0) + 1
        origin = _source_origin(row.get('source_video'))
        origin_key = f"{origin['creator']}|{origin['video_name']}|{origin['source_video']}"
        if origin_key not in origin_counts:
            origin_counts[origin_key] = {**origin, 'clip_count': 0, 'label_counts': {'scrum': 0, 'line': 0, 'other': 0}}
        origin_counts[origin_key]['clip_count'] += 1
        origin_counts[origin_key]['label_counts'][correct_label] = origin_counts[origin_key]['label_counts'].get(correct_label, 0) + 1
    recent_sessions = []
    if _resolve_model_artifact(_training_dataset_sessions_dir()).exists():
        session_files = sorted(_model_artifact_glob(_training_dataset_sessions_dir(), 'training_session_*.json', recursive=False), key=lambda item: item.stat().st_mtime, reverse=True)
        for session_file in session_files[:5]:
            try:
                snapshot = _load_json_file(session_file)
            except Exception:
                continue
            recent_sessions.append({'session_id': snapshot.get('session_id'), 'result_id': snapshot.get('result_id'), 'last_export_at': snapshot.get('last_export_at'), 'exported_count': snapshot.get('exported_count', 0), 'correct_count': snapshot.get('correct_count', 0), 'corrected_count': snapshot.get('corrected_count', 0), 'model_name': snapshot.get('model_name'), **_source_origin(snapshot.get('source_video'))})
    origins = sorted(origin_counts.values(), key=lambda item: (-item['clip_count'], item['creator'], item['video_name']))
    recent_clips = []
    for row in sorted(rows, key=lambda item: item.get('reviewed_at') or '', reverse=True)[:20]:
        origin = _source_origin(row.get('source_video'))
        recent_clips.append({'dataset_id': row.get('dataset_id'), 'clip_name': row.get('clip_name'), 'start_time': row.get('start_time'), 'end_time': row.get('end_time'), 'correct_label': row.get('correct_label') or row.get('tag') or 'other', 'review_outcome': row.get('review_outcome'), **origin})
    return {'dataset_root': str(_training_dataset_root()), 'manifest_path': str(manifest_path), 'review_log_path': str(_training_dataset_review_log_path()), 'total_clips': total_clips, 'session_count': len(session_ids), 'result_count': len(result_ids), 'correct_clips': correct_clips, 'corrected_clips': corrected_clips, 'correct_rate': correct_clips / total_clips if total_clips else 0.0, 'label_counts': label_counts, 'review_outcome_counts': review_outcome_counts, 'origins': origins, 'recent_clips': recent_clips, 'recent_sessions': recent_sessions}

def _initial_clip_label(filename: str) -> str:
    lowered = filename.lower()
    if 'scrum' in lowered:
        return 'scrum'
    if 'line' in lowered:
        return 'line'
    return 'other'

def _initial_training_catalog() -> tuple[list[dict], dict]:
    """Describe the current collection and its three independent partitions."""
    signature = tuple((path.stat().st_mtime_ns if path.exists() else None for path in (INITIAL_TRAINING_ROOT, INITIAL_TRAIN_CSV, INITIAL_VALIDATION_CSV, INITIAL_TEST_CSV, INITIAL_BALANCED_TRAIN_CSV)))
    if INITIAL_TRAINING_CATALOG_CACHE['signature'] == signature:
        return (INITIAL_TRAINING_CATALOG_CACHE['clips'], INITIAL_TRAINING_CATALOG_CACHE['summary'])
    with INITIAL_TRAINING_CATALOG_LOCK:
        if INITIAL_TRAINING_CATALOG_CACHE['signature'] == signature:
            return (INITIAL_TRAINING_CATALOG_CACHE['clips'], INITIAL_TRAINING_CATALOG_CACHE['summary'])
        clips, summary = _build_initial_training_catalog()
        INITIAL_TRAINING_CATALOG_CACHE.update({'signature': signature, 'clips': clips, 'summary': summary})
        return (clips, summary)

def _build_initial_training_catalog() -> tuple[list[dict], dict]:
    partitions = {'train': INITIAL_TRAIN_CSV, 'validation': INITIAL_VALIDATION_CSV, 'test': INITIAL_TEST_CSV}
    assignments = {}
    for split, csv_path in partitions.items():
        if not csv_path.is_file():
            raise FileNotFoundError(f'Dataset partition not found: {csv_path.name}')
        for row in _read_csv_rows(csv_path):
            name, label = row.get('video_name'), row.get('tag')
            if not name or label not in {'line', 'other', 'scrum'}:
                raise ValueError(f'Invalid dataset row in {csv_path.name}')
            if name in assignments:
                raise ValueError(f'Clip assigned more than once: {name}')
            assignments[name] = (split, label)
    balanced_rows = _read_csv_rows(INITIAL_BALANCED_TRAIN_CSV)
    train_occurrences = Counter()
    for row in balanced_rows:
        name = row.get('video_name')
        if assignments.get(name) != ('train', row.get('tag')):
            raise ValueError(f'Balanced training clip outside training partition: {name}')
        train_occurrences[name] += 1
    clips = []
    if _resolve_model_artifact(INITIAL_TRAINING_ROOT).exists():
        for path in INITIAL_TRAINING_ROOT.iterdir():
            if not _resolve_model_artifact(path).is_file() or path.suffix.lower() not in INITIAL_TRAINING_VIDEO_EXTENSIONS:
                continue
            if path.name not in assignments:
                raise ValueError(f'Video missing from dataset CSV: {path.name}')
            split, label = assignments[path.name]
            clips.append({'filename': path.name, 'label': label, 'split': split, 'training_occurrences': int(train_occurrences.get(path.name, 0)), 'size_bytes': path.stat().st_size})
    if len(clips) != len(assignments):
        raise ValueError('Dataset CSV references missing clip files')
    clips.sort(key=lambda item: (item['label'], item['filename'].lower()))
    split_counts = Counter((item['split'] for item in clips))
    label_counts = Counter((item['label'] for item in clips))
    summary = {'collection_clips': len(clips), 'training_examples': split_counts.get('train', 0), 'unique_training_clips': split_counts.get('train', 0), 'validation_clips': split_counts.get('validation', 0), 'test_clips': split_counts.get('test', 0), 'unassigned_clips': 0, 'balanced_training_examples': len(balanced_rows), 'balanced_unique_training_clips': len(train_occurrences), 'oversampled_examples': sum(count - 1 for count in train_occurrences.values()), 'label_counts': dict(label_counts)}
    return (clips, summary)

def _browser_compatible_training_clip(source_path: Path) -> Path:
    """Create an H.264 preview on demand without modifying the training clip."""
    INITIAL_TRAINING_PREVIEW_CACHE.mkdir(parents=True, exist_ok=True)
    preview_path = INITIAL_TRAINING_PREVIEW_CACHE / f'{source_path.stem}_h264.mp4'
    if _resolve_model_artifact(preview_path).exists() and preview_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
        return preview_path
    with INITIAL_TRAINING_PREVIEW_LOCK:
        if _resolve_model_artifact(preview_path).exists() and preview_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
            return preview_path
        temp_path = INITIAL_TRAINING_PREVIEW_CACHE / f'{source_path.stem}_h264.tmp.mp4'
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source_path), '-map', '0:v:0', '-map', '0:a?', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '96k', '-movflags', '+faststart', str(temp_path)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if completed.returncode != 0 or not _resolve_model_artifact(temp_path).exists():
            temp_path.unlink(missing_ok=True)
            detail = (completed.stderr or 'ffmpeg did not create the preview').strip()
            raise RuntimeError(f'Could not create browser-compatible preview: {detail}')
        temp_path.replace(preview_path)
    return preview_path

@app.route('/')
def index():
    """Main page - model and video selection."""
    return render_template('index.html')

@app.route('/live')
def live_detection_page():
    """Live detection workflow."""
    return render_template('live.html')

@app.route('/training-dataset')
def training_dataset_page():
    """Human-reviewed training dataset dashboard."""
    return render_template('training_dataset.html')

@app.route('/training-clips')
def training_clips_page():
    """Browser for the dataset clips."""
    return render_template('training_clips.html')

@app.route('/clip-generator')
def clip_generator_page():
    """Manual workflow for labeling consecutive seven-second clips."""
    return render_template('clip_generator.html')

@app.route('/api/training-clips')
def list_initial_training_clips():
    try:
        clips, summary = _initial_training_catalog()
        query = (request.args.get('q') or '').strip().lower()
        label = (request.args.get('label') or 'all').strip().lower()
        split = (request.args.get('split') or 'train').strip().lower()
        page = max(1, _to_int(request.args.get('page'), 1))
        page_size = max(8, min(48, _to_int(request.args.get('page_size'), 24)))
        if label not in {'all', 'scrum', 'line', 'other'}:
            return (jsonify({'success': False, 'error': 'Invalid label filter'}), 400)
        if split not in {'all', 'train', 'validation', 'test', 'unassigned'}:
            return (jsonify({'success': False, 'error': 'Invalid split filter'}), 400)
        filtered = [clip for clip in clips if (label == 'all' or clip['label'] == label) and (split == 'all' or clip['split'] == split) and (not query or query in clip['filename'].lower())]
        total = len(filtered)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, pages)
        start = (page - 1) * page_size
        return jsonify({'success': True, 'clips': filtered[start:start + page_size], 'summary': summary, 'pagination': {'page': page, 'page_size': page_size, 'pages': pages, 'total': total}})
    except Exception as exc:
        logger.error(f'Error listing initial training clips: {exc}', exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/training-clips/video/<path:filename>')
def serve_initial_training_clip(filename):
    """Serve only files located directly inside the initial training collection."""
    try:
        if Path(filename).name != filename:
            return (jsonify({'success': False, 'error': 'Invalid clip name'}), 400)
        clip_path = (INITIAL_TRAINING_ROOT / filename).resolve()
        root = INITIAL_TRAINING_ROOT.resolve()
        if root not in clip_path.parents or clip_path.suffix.lower() not in INITIAL_TRAINING_VIDEO_EXTENSIONS:
            return (jsonify({'success': False, 'error': 'Invalid clip path'}), 400)
        if not _resolve_model_artifact(clip_path).exists() or not _resolve_model_artifact(clip_path).is_file():
            return (jsonify({'success': False, 'error': 'Training clip not found'}), 404)
        preview_path = _browser_compatible_training_clip(clip_path)
        return send_file(preview_path, mimetype='video/mp4', conditional=True)
    except Exception as exc:
        logger.error(f'Error serving initial training clip: {exc}', exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/models')
def get_models():
    """Get list of available trained models."""
    try:
        models = inference_engine.list_available_models()
        return jsonify({'success': True, 'models': models})
    except Exception as e:
        logger.error(f'Error listing models: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/videos')
def get_videos():
    """Get list of available videos.

    Includes the current evaluation match segments and uploaded videos.
    """
    try:
        video_library = app.config['VIDEO_LIBRARY']
        videos = []
        seen = set()
        VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        library_candidates = [video_file for video_file in sorted(_model_artifact_glob(video_library, '*', recursive=True)) if _resolve_model_artifact(video_file).is_file() and video_file.suffix.lower() in VIDEO_EXTS] if _resolve_model_artifact(video_library).exists() else []
        canonical_name = re.compile('^V_\\d+_(?:S\\d{2}(?:_[A-Z0-9]+)?|FULL)_', re.IGNORECASE)
        candidates_by_identity = {}
        for video_file in library_candidates:
            stat = video_file.stat()
            identity = (stat.st_dev, stat.st_ino) if stat.st_ino else str(video_file.resolve()).casefold()
            rank = (0 if canonical_name.match(video_file.name) else 1, str(video_file).casefold())
            current = candidates_by_identity.get(identity)
            if current is None or rank < current[0]:
                candidates_by_identity[identity] = (rank, video_file)
        library_files = sorted((item[1] for item in candidates_by_identity.values()), key=lambda path: str(path).casefold())
        uploads = app.config['UPLOAD_FOLDER']
        upload_files = [video_file for video_file in sorted(_model_artifact_glob(uploads, '*', recursive=False)) if _resolve_model_artifact(video_file).is_file() and video_file.suffix.lower() in VIDEO_EXTS] if _resolve_model_artifact(uploads).exists() else []
        metadata_by_path = read_video_metadata_many(library_files + upload_files)
        clip_generator_by_video = _clip_generator_video_summaries()
        if library_files:
            for video_file in library_files:
                key = str(video_file)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    rel = video_file.relative_to(video_library)
                    display_name = str(rel)
                except ValueError:
                    display_name = video_file.name
                metadata = metadata_by_path[video_file]
                videos.append({'name': display_name, 'path': str(video_file), 'size': video_file.stat().st_size, 'duration': metadata['duration'], 'clip_generator': clip_generator_by_video.get(_video_file_identity_key(video_file))})
        if upload_files:
            for video_file in upload_files:
                key = str(video_file)
                if key in seen:
                    continue
                seen.add(key)
                metadata = metadata_by_path[video_file]
                videos.append({'name': video_file.name, 'path': str(video_file), 'size': video_file.stat().st_size, 'duration': metadata['duration'], 'clip_generator': clip_generator_by_video.get(_video_file_identity_key(video_file))})
        return jsonify({'success': True, 'videos': videos, 'video_library_path': str(video_library), 'uploads_path': str(uploads)})
    except Exception as e:
        logger.error(f'Error listing videos: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload a new video file."""
    try:
        if 'video' not in request.files:
            return (jsonify({'success': False, 'error': 'No video file provided'}), 400)
        file = request.files['video']
        if file.filename == '':
            return (jsonify({'success': False, 'error': 'No file selected'}), 400)
        allowed_extensions = {'.mp4', '.avi', '.mov'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return (jsonify({'success': False, 'error': f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"}), 400)
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{timestamp}_{filename}'
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        logger.info(f'Uploaded video: {filename}')
        return jsonify({'success': True, 'filename': filename, 'filepath': str(filepath), 'message': 'Video uploaded successfully'})
    except Exception as e:
        logger.error(f'Error uploading video: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

def _execute_analysis(data: dict, progress_callback=None) -> dict:
    """Run an analysis and return the same payload as the synchronous API."""
    total_started = time.perf_counter()

    def report(percent, stage, message):
        if progress_callback:
            progress_callback(float(percent), stage, message)
    report(2, 'validating', 'Validando modelo y video')
    model_name = data.get('model_name')
    video_path_str = data.get('video_path')
    parameters = data.get('parameters') or {}
    if not model_name or not video_path_str:
        raise ValueError('Model name and video path are required')
    video_path = _resolve_video_path(video_path_str)
    if not _resolve_model_artifact(video_path).exists():
        raise FileNotFoundError(f'Video not found: {video_path_str}. Looked in: {video_path}')
    _validate_inference_source(video_path)
    logger.info(f'Analyzing {video_path.name} with {model_name}')
    logger.info(f'Full path: {video_path}')
    logger.info(f'Parameters: {parameters}')
    report(8, 'model', 'Cargando el modelo seleccionado')
    model_load_started = time.perf_counter()
    try:
        inference_engine.load_model(model_name)
    except Exception as exc:
        logger.error('Could not load selected model %s: %s', model_name, exc)
        raise RuntimeError(f"No se pudo cargar el modelo seleccionado '{model_name}'. El análisis fue cancelado para evitar ejecutar otro modelo.") from exc
    model_load_seconds = time.perf_counter() - model_load_started
    report(15, 'preprocessing', 'Preparando el video para el análisis')
    inference_started = time.perf_counter()
    results = inference_engine.analyze_video(video_path, confidence_threshold=parameters.get('confidence_threshold', 0.7), line_confidence_threshold=max(0.0, min(1.0, _to_float(parameters.get('line_confidence_threshold'), 0.95))), min_duration=parameters.get('min_play_duration', 3.0), max_duration=parameters.get('max_play_duration', 15.0), overlap_threshold=parameters.get('overlap_threshold', 0.5), batch_size=parameters.get('batch_size', 8), segment_duration=parameters.get('segment_duration', 7), use_segments=parameters.get('use_segments', True), temporal_fps_mode=parameters.get('temporal_fps_mode', 'training_match_25'), custom_effective_fps=parameters.get('custom_effective_fps'), feature_batch_size=parameters.get('feature_batch_size', 256), feature_predict_batch_size=parameters.get('feature_predict_batch_size', 8), progress_callback=progress_callback)
    if results.get('status') != 'success':
        raise RuntimeError(results.get('error') or 'El modelo no pudo completar el análisis del video.')
    inference_seconds = time.perf_counter() - inference_started
    video_seconds = float((results.get('metadata') or {}).get('duration') or 0.0)
    timing = {'clock': 'time.perf_counter_wall_time', 'model_load_seconds': float(model_load_seconds), 'inference_seconds': float(inference_seconds), 'total_seconds': float(time.perf_counter() - total_started), 'video_seconds': video_seconds, 'inference_realtime_factor': float(inference_seconds / video_seconds) if video_seconds > 0 else None, 'processed_video_seconds_per_wall_second': float(video_seconds / inference_seconds) if inference_seconds > 0 else None}
    results['timing'] = timing
    results.setdefault('metadata', {})['timing'] = timing
    report(97, 'saving', 'Guardando las detecciones')
    result_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = app.config['RESULTS_FOLDER'] / f'result_{result_id}.json'
    with _resolve_model_artifact(result_file).open('w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    logger.info(f"Analysis complete. Found {len(results.get('detections', []))} plays")
    report(100, 'completed', 'Análisis completo')
    return {'success': True, 'result_id': result_id, 'video_url': f'/api/video-for-result/{result_id}', 'results': results, 'message': 'Analysis complete'}

def _analysis_worker(job_id: str, data: dict) -> None:

    def update_progress(percent, stage, message):
        job = ANALYSIS_JOBS.get(job_id)
        if not job:
            return
        job['progress'] = {'percent': round(max(0.0, min(100.0, float(percent))), 1), 'stage': str(stage), 'message': str(message)}
        job['updated_at'] = _now_iso()
    job = ANALYSIS_JOBS[job_id]
    job['status'] = 'running'
    job['started_at'] = _now_iso()
    try:
        job['response'] = _execute_analysis(data, update_progress)
        job['status'] = 'completed'
    except Exception as exc:
        logger.error(f'Analysis job {job_id} failed: {exc}', exc_info=True)
        job['status'] = 'error'
        job['error'] = str(exc)
    finally:
        job['updated_at'] = _now_iso()

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """Backward-compatible synchronous analysis endpoint."""
    try:
        return jsonify(_execute_analysis(request.get_json(silent=True) or {}))
    except ValueError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 400)
    except FileNotFoundError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 404)
    except Exception as exc:
        logger.error(f'Error analyzing video: {exc}', exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/analyze/start', methods=['POST'])
def start_analysis():
    """Start analysis in the background so the UI can poll real progress."""
    data = request.get_json(silent=True) or {}
    if not data.get('model_name') or not data.get('video_path'):
        return (jsonify({'success': False, 'error': 'Model name and video path are required'}), 400)
    active = next((job_id for job_id, job in ANALYSIS_JOBS.items() if job.get('status') in {'queued', 'running'}), None)
    if active:
        return (jsonify({'success': False, 'error': 'Ya hay un análisis en curso. Esperá a que finalice.', 'job_id': active}), 409)
    job_id = uuid.uuid4().hex[:12]
    ANALYSIS_JOBS[job_id] = {'status': 'queued', 'progress': {'percent': 0.0, 'stage': 'queued', 'message': 'Análisis en cola'}, 'created_at': _now_iso(), 'updated_at': _now_iso(), 'started_at': None, 'response': None, 'error': None}
    threading.Thread(target=_analysis_worker, args=(job_id, data), daemon=True).start()
    return (jsonify({'success': True, 'job_id': job_id}), 202)

@app.route('/api/analyze/status/<job_id>')
def analysis_status(job_id):
    job = ANALYSIS_JOBS.get(job_id)
    if not job:
        return (jsonify({'success': False, 'error': 'Analysis job not found'}), 404)
    return jsonify({'success': True, 'job_id': job_id, **job})

@app.route('/api/processed_videos')
def list_processed_videos():
    """List prepared evaluation videos and locally generated analysis chunks."""
    try:
        found = []
        for raw_dir in (app.config['VIDEO_LIBRARY'], WEB_RUNTIME / 'source_chunks'):
            if not raw_dir.exists():
                continue
            processed_files = [p for p in sorted(raw_dir.rglob('*'), reverse=True) if p.is_file() and p.suffix.lower() in ['.mp4', '.mkv', '.webm']]
            metadata_by_path = read_video_metadata_many(processed_files)
            for p in processed_files:
                metadata = metadata_by_path[p]
                width = int(metadata['width'])
                height = int(metadata['height'])
                analysis_eligible = _supports_training_preprocessing(p)
                found.append({'name': p.name, 'path': str(p), 'size': p.stat().st_size, 'modified': p.stat().st_mtime, 'width': width, 'height': height, 'duration': metadata['duration'], 'analysis_eligible': analysis_eligible, 'analysis_warning': None if analysis_eligible else 'No se pudo leer la resolución de este video.'})
        return jsonify({'success': True, 'processed_videos': found})
    except Exception as e:
        logger.error(f'Error listing processed videos: {e}', exc_info=True)
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/process_video', methods=['POST'])
def process_video():
    """Start server-side split+preprocess for a selected video (runs in background).

    JSON body:
    {
      "video_path": "relative/or/absolute/path/to/video.mp4",
      "chunk_minutes": 20,
      "scale": 0.5,            # optional; if omitted auto-scale will be chosen
      "preprocess": false,
      "overwrite": false
    }
    """
    try:
        data = request.get_json() or {}
        video_path_str = data.get('video_path')
        if not video_path_str:
            return (jsonify({'success': False, 'error': 'video_path is required'}), 400)
        video_path = Path(video_path_str)
        if not video_path.is_absolute():
            if _resolve_model_artifact(app.config['UPLOAD_FOLDER'] / video_path.name).exists():
                video_path = app.config['UPLOAD_FOLDER'] / video_path.name
            elif _resolve_model_artifact(app.config['VIDEO_LIBRARY'] / video_path.name).exists():
                video_path = app.config['VIDEO_LIBRARY'] / video_path.name
            else:
                video_path = PROJECT_ROOT / video_path
        if not _resolve_model_artifact(video_path).exists():
            return (jsonify({'success': False, 'error': f'Video not found: {video_path_str}'}), 404)
        chunk_minutes = float(data.get('chunk_minutes', 20.0))
        if data.get('preprocess'):
            logger.warning('Ignoring preprocess=true: web analysis chunks must retain source resolution')
        preprocess = False
        overwrite = bool(data.get('overwrite', False))
        scale = data.get('scale', None)
        if scale is None:
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                max_dim = max(w, h)
                scale = 0.5 if max_dim > 1000 else 1.0
                logger.info(f'Auto-scale for {video_path.name}: {scale} (resolution {w}x{h})')
            except Exception:
                scale = 0.5
        out_dir = WEB_RUNTIME / 'source_chunks'
        job_id = uuid.uuid4().hex[:8]
        PROCESS_JOBS[job_id] = {'status': 'running', 'out_files': [], 'error': None}

        def _worker(job_id_inner, input_path, out_dir_inner, chunk_minutes_inner, scale_inner, preprocess_inner, overwrite_inner):
            try:
                import sys as _sys
                _scripts_dir = str(PROJECT_ROOT)
                if _scripts_dir not in _sys.path:
                    _sys.path.insert(0, _scripts_dir)
                from scripts.data_prep.split_and_preprocess_video import split_and_preprocess
                written = split_and_preprocess(input_path, out_dir_inner, chunk_minutes=chunk_minutes_inner, scale=scale_inner, crop_size=224, preprocess=preprocess_inner, overwrite=overwrite_inner)
                PROCESS_JOBS[job_id_inner]['status'] = 'done'
                PROCESS_JOBS[job_id_inner]['out_files'] = [str(p) for p in written]
            except Exception as e:
                logger.error(f'Processing job {job_id_inner} failed: {e}', exc_info=True)
                PROCESS_JOBS[job_id_inner]['status'] = 'error'
                PROCESS_JOBS[job_id_inner]['error'] = str(e)
        t = threading.Thread(target=_worker, args=(job_id, video_path, out_dir, chunk_minutes, float(scale), preprocess, overwrite), daemon=True)
        t.start()
        return jsonify({'success': True, 'job_id': job_id, 'message': 'Processing started'})
    except Exception as e:
        logger.error(f'Error starting processing job: {e}', exc_info=True)
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/process_status/<job_id>')
def process_status(job_id):
    job = PROCESS_JOBS.get(job_id)
    if job is None:
        return (jsonify({'success': False, 'error': 'Job not found'}), 404)
    return jsonify({'success': True, 'job': job})

@app.route('/api/live_detection/start', methods=['POST'])
def start_live_detection():
    """Start a live-style sliding inference job over a saved video."""
    try:
        data = request.get_json() or {}
        model_name = data.get('model_name')
        video_path_str = data.get('video_path')
        params = data.get('parameters', {})
        if not video_path_str:
            return (jsonify({'success': False, 'error': 'video_path is required'}), 400)
        video_path = _resolve_video_path(video_path_str)
        if not _resolve_model_artifact(video_path).exists():
            return (jsonify({'success': False, 'error': f'Video not found: {video_path_str}'}), 404)
        try:
            _validate_inference_source(video_path)
        except ValueError as exc:
            return (jsonify({'success': False, 'error': str(exc)}), 422)
        job_id = uuid.uuid4().hex[:12]
        safe_params = {'window_seconds': _to_float(params.get('window_seconds'), 7.0), 'stride_seconds': _to_float(params.get('stride_seconds'), 1.0), 'prediction_interval_seconds': _to_float(params.get('prediction_interval_seconds', params.get('stride_seconds')), 1.0), 'confidence_threshold': _to_float(params.get('confidence_threshold'), 0.7), 'target_fps': _to_float(params.get('target_fps'), 25.0), 'sampled_frame_count': _to_int(params.get('sampled_frame_count'), 0), 'simulate_realtime': bool(params.get('simulate_realtime', True)), 'sleep_scale': _to_float(params.get('sleep_scale'), 1.0)}
        LIVE_DETECTION_JOBS[job_id] = _build_live_job(job_id=job_id, video_path=video_path, model_name=model_name, params=safe_params)
        thread = threading.Thread(target=_live_detection_worker, args=(job_id,), daemon=True)
        thread.start()
        return jsonify({'success': True, 'job_id': job_id, 'status': 'queued', 'video_path': str(video_path), 'model_name': model_name, 'parameters': safe_params})
    except Exception as exc:
        logger.error(f'Error starting live detection: {exc}', exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/live_detection/status/<job_id>')
def live_detection_status(job_id):
    """Get progress and latest prediction for a live-style detection job."""
    job = LIVE_DETECTION_JOBS.get(job_id)
    if not job:
        return (jsonify({'success': False, 'error': 'Job not found'}), 404)
    return jsonify({'success': True, 'job': {'job_id': job['job_id'], 'status': job['status'], 'created_at': job['created_at'], 'updated_at': job['updated_at'], 'video_path': job['video_path'], 'model_name': job['model_name'], 'parameters': job['parameters'], 'progress': job['progress'], 'playback_time': job.get('playback_time', 0.0), 'playback_playing': job.get('playback_playing', False), 'latest_prediction': job['latest_prediction'], 'detections_count': len(job.get('detections', [])), 'error': job.get('error'), 'result_file': job.get('result_file')}})

@app.route('/api/live_detection/playback/<job_id>', methods=['POST'])
def live_detection_playback(job_id):
    """Update the browser preview time that gates live-style processing."""
    job = LIVE_DETECTION_JOBS.get(job_id)
    if not job:
        return (jsonify({'success': False, 'error': 'Job not found'}), 404)
    payload = request.get_json(silent=True) or {}
    playback_time = max(0.0, _to_float(payload.get('current_time'), 0.0))
    job['playback_time'] = playback_time
    job['playback_playing'] = bool(payload.get('playing', False))
    job['playback_updated_at'] = _now_iso()
    job['updated_at'] = _now_iso()
    return jsonify({'success': True})

@app.route('/api/live_detection/result/<job_id>')
def live_detection_result(job_id):
    """Return full detections for a live-style detection job."""
    job = LIVE_DETECTION_JOBS.get(job_id)
    if not job:
        return (jsonify({'success': False, 'error': 'Job not found'}), 404)
    return jsonify({'success': True, 'job_id': job_id, 'status': job['status'], 'video_path': job['video_path'], 'model_name': job['model_name'], 'parameters': job['parameters'], 'progress': job['progress'], 'latest_prediction': job['latest_prediction'], 'detections': job.get('detections', []), 'result_file': job.get('result_file'), 'error': job.get('error')})

@app.route('/api/live_detection/stop/<job_id>', methods=['POST'])
def stop_live_detection(job_id):
    """Request stop for a running live-style detection job."""
    job = LIVE_DETECTION_JOBS.get(job_id)
    if not job:
        return (jsonify({'success': False, 'error': 'Job not found'}), 404)
    if job['status'] in {'completed', 'error', 'stopped'}:
        return jsonify({'success': True, 'status': job['status'], 'message': 'Job already finished'})
    job['stop_requested'] = True
    job['updated_at'] = _now_iso()
    return jsonify({'success': True, 'status': 'stopping', 'job_id': job_id})

@app.route('/api/results')
def list_results():
    """List all saved analysis results, newest first."""
    try:
        results_folder = app.config['RESULTS_FOLDER']
        result_files = sorted(_model_artifact_glob(results_folder, 'result_*.json', recursive=False), reverse=True)
        history_signature = (results_folder.stat().st_mtime_ns, _review_sessions_dir().stat().st_mtime_ns)
        with RESULTS_HISTORY_CACHE_LOCK:
            if RESULTS_HISTORY_CACHE['signature'] == history_signature:
                response = jsonify({'success': True, 'results': RESULTS_HISTORY_CACHE['entries']})
                response.headers['X-History-Cache'] = 'hit'
                return response
            entries = []
            clip_comparison_cache = {}
            clip_generator_by_video = _clip_generator_video_summaries()
            for f in result_files:
                result_id = f.stem[len('result_'):]
                try:
                    data = _load_json_file(f)
                    detections = data.get('detections', [])
                    counts = {}
                    for d in detections:
                        pt = d.get('play_type', 'unknown')
                        counts[pt] = counts.get(pt, 0) + 1
                    try:
                        dt = datetime.strptime(result_id, '%Y%m%d_%H%M%S')
                        date_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        date_str = result_id
                    video_path = data.get('video_path', '')
                    video_name = Path(video_path).name if video_path else '—'
                    clip_generator = None
                    if video_path:
                        try:
                            try:
                                resolved_video_path = _resolve_original_result_source(str(video_path)).resolve()
                            except (FileNotFoundError, ValueError):
                                resolved_video_path = _resolve_video_path(str(video_path)).resolve()
                            cache_key = str(resolved_video_path)
                            if cache_key not in clip_comparison_cache:
                                clip_comparison_cache[cache_key] = clip_generator_by_video.get(_video_file_identity_key(resolved_video_path))
                            manual_comparison = clip_comparison_cache[cache_key]
                            if manual_comparison is not None:
                                clip_generator = {'available': True, 'line': int(manual_comparison.get('line') or 0), 'scrum': int(manual_comparison.get('scrum') or 0), 'reviewed': int(manual_comparison.get('reviewed') or 0), 'total': int(manual_comparison.get('total') or 0)}
                        except Exception as exc:
                            logger.warning('Could not add clip labels to history entry %s: %s', result_id, exc)
                    entries.append({'result_id': result_id, 'date': date_str, 'video': video_name, 'model': data.get('model_name', '—'), 'total_detections': len(detections), 'counts': counts, 'clip_generator': clip_generator})
                except Exception as e:
                    logger.warning(f'Could not parse result {f.name}: {e}')
            RESULTS_HISTORY_CACHE['signature'] = history_signature
            RESULTS_HISTORY_CACHE['entries'] = entries
        response = jsonify({'success': True, 'results': entries})
        response.headers['X-History-Cache'] = 'miss'
        return response
    except Exception as e:
        logger.error(f'Error listing results: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/result/<result_id>')
def get_result(result_id):
    """Retrieve saved analysis result."""
    try:
        result_file = app.config['RESULTS_FOLDER'] / f'result_{result_id}.json'
        if not _resolve_model_artifact(result_file).exists():
            return (jsonify({'success': False, 'error': 'Result not found'}), 404)
        results = _load_json_file(result_file)
        return jsonify({'success': True, 'video_url': f'/api/video-for-result/{result_id}', 'results': results})
    except Exception as e:
        logger.error(f'Error retrieving result: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/result/<result_id>/clip-labels')
def get_result_clip_labels(result_id):
    """Return manual clip-generator labels for the analyzed source video."""
    try:
        result_payload = _load_result_payload(result_id)
        raw_path = str(result_payload.get('video_path') or '')
        try:
            video_path = _resolve_original_result_source(raw_path).resolve()
        except (FileNotFoundError, ValueError):
            video_path = _resolve_video_path(raw_path).resolve()
        comparison = _latest_labeled_clip_generator_comparison(video_path, result_payload)
        return jsonify({'success': True, 'available': comparison is not None, 'comparison': comparison})
    except FileNotFoundError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 404)
    except Exception as exc:
        logger.error('Error loading clip labels for result %s: %s', result_id, exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/result/<result_id>/threshold-preview', methods=['POST'])
def preview_result_thresholds(result_id):
    """Recalculate saved detections from cached classification probabilities."""
    try:
        data = request.get_json() or {}
        result_payload = _load_result_payload(result_id)
        metadata = result_payload.get('metadata') or {}
        confidence_threshold = _to_float(data.get('confidence_threshold'), metadata.get('confidence_threshold', 0.7))
        line_confidence_threshold = _to_float(data.get('line_confidence_threshold'), metadata.get('line_confidence_threshold', 0.95))
        preview = _rethreshold_result_payload(result_payload, confidence_threshold=confidence_threshold, line_confidence_threshold=line_confidence_threshold)
        return jsonify({'success': True, **preview})
    except FileNotFoundError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 404)
    except (TypeError, ValueError) as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 400)
    except Exception as exc:
        logger.error('Error previewing thresholds for %s: %s', result_id, exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/review/start', methods=['POST'])
def start_review_session():
    """Create or resume a review session for an existing analysis result."""
    try:
        data = request.get_json() or {}
        result_id = data.get('result_id')
        force_new = bool(data.get('force_new', False))
        review_mode = str(data.get('review_mode') or REVIEW_MODE_OVERLAP)
        if not result_id:
            return (jsonify({'success': False, 'error': 'result_id is required'}), 400)
        session = _create_review_session(result_id, force_new=force_new, review_mode=review_mode)
        session['video_url'] = _review_session_video_url(session)
        return jsonify({'success': True, 'session': session})
    except FileNotFoundError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 404)
    except ValueError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 400)
    except Exception as exc:
        logger.error('Error starting review session: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/clip-generator/start', methods=['POST'])
def start_clip_generator_session():
    """Create or resume a manual fixed-block labeling session."""
    try:
        data = request.get_json() or {}
        video_path = str(data.get('video_path') or '')
        dataset_split = str(data.get('dataset_split') or 'train').lower()
        force_new = bool(data.get('force_new', False))
        if not video_path:
            return (jsonify({'success': False, 'error': 'video_path is required'}), 400)
        session = _create_clip_generator_session(video_path, dataset_split=dataset_split, force_new=force_new)
        session['video_url'] = _review_session_video_url(session)
        return jsonify({'success': True, 'session': session})
    except FileNotFoundError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 404)
    except ValueError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 400)
    except Exception as exc:
        logger.error('Error starting clip generator session: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/clip-generator/analysis')
def clip_generator_analysis():
    """Return saved analyses aligned to manual seven-second blocks."""
    try:
        video_path_str = str(request.args.get('video_path') or '')
        if not video_path_str:
            return (jsonify({'success': False, 'error': 'video_path is required'}), 400)
        video_path = _resolve_video_path(video_path_str).resolve()
        if not _resolve_model_artifact(video_path).exists() or not _resolve_model_artifact(video_path).is_file():
            return (jsonify({'success': False, 'error': f'Video not found: {video_path_str}'}), 404)
        allowed_roots = [app.config['VIDEO_LIBRARY'].resolve(), app.config['UPLOAD_FOLDER'].resolve()]
        if not any((root == video_path.parent or root in video_path.parents for root in allowed_roots)):
            return (jsonify({'success': False, 'error': 'Video is outside the available library'}), 400)
        include_all = str(request.args.get('include_all') or '').lower() in {'1', 'true', 'yes'}
        comparisons = _analysis_comparisons(video_path, limit=None if include_all else 1)
        comparison = comparisons[0] if comparisons else None
        return jsonify({'success': True, 'available': comparison is not None, 'analysis': comparison, 'analyses': comparisons})
    except Exception as exc:
        logger.error('Error loading clip generator analysis: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/clip-generator/labels')
def clip_generator_labels():
    """Return manual seven-second labels directly from a source video.

    A comparison can be created without first saving an individual analysis.
    Keeping this lookup video-based lets that comparison show its ground truth
    without depending on an unrelated result file.
    """
    try:
        video_path_str = str(request.args.get('video_path') or '')
        if not video_path_str:
            return (jsonify({'success': False, 'error': 'video_path is required'}), 400)
        video_path = _resolve_video_path(video_path_str).resolve()
        if not _resolve_model_artifact(video_path).exists() or not _resolve_model_artifact(video_path).is_file():
            return (jsonify({'success': False, 'error': f'Video not found: {video_path_str}'}), 404)
        allowed_roots = [app.config['VIDEO_LIBRARY'].resolve(), app.config['UPLOAD_FOLDER'].resolve()]
        if not any((root == video_path.parent or root in video_path.parents for root in allowed_roots)):
            return (jsonify({'success': False, 'error': 'Video is outside the available library'}), 400)
        comparison = _latest_labeled_clip_generator_comparison(video_path)
        return jsonify({'success': True, 'available': comparison is not None, 'comparison': comparison})
    except Exception as exc:
        logger.error('Error loading clip-generator labels: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/review/session/<session_id>')
def get_review_session(session_id):
    """Fetch a persisted review session with source video metadata."""
    try:
        session = _load_review_session(session_id)
        session['summary'] = _review_summary_for_session(session)
        session['video_url'] = _review_session_video_url(session)
        return jsonify({'success': True, 'session': session})
    except FileNotFoundError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 404)
    except Exception as exc:
        logger.error('Error loading review session: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/review/session/<session_id>/decision', methods=['POST'])
def save_review_decision(session_id):
    """Persist a review decision for a candidate clip."""
    try:
        data = request.get_json() or {}
        candidate_index = data.get('candidate_index')
        final_label = data.get('final_label')
        if candidate_index is None:
            return (jsonify({'success': False, 'error': 'candidate_index is required'}), 400)
        session = _load_review_session(session_id)
        valid_labels = VALID_REVIEW_FINAL_LABELS
        if session.get('review_mode') != REVIEW_MODE_FULL_MATCH:
            valid_labels = {'scrum', 'line', 'other'}
        if final_label not in valid_labels:
            return (jsonify({'success': False, 'error': f'final_label must be one of {sorted(valid_labels)}'}), 400)
        candidates = session.get('candidates', [])
        if candidate_index < 0 or candidate_index >= len(candidates):
            return (jsonify({'success': False, 'error': 'candidate_index out of range'}), 404)
        candidate = candidates[candidate_index]
        candidate['final_label'] = final_label
        candidate['reviewed_at'] = datetime.now().isoformat(timespec='seconds')
        session = _persist_review_session(session)
        auto_exported = False
        auto_export_error = None
        reviewed_count = int(session.get('summary', {}).get('reviewed', 0))
        pending_count = int(session.get('summary', {}).get('pending', 0))
        should_auto_export = reviewed_count > 0 and pending_count == 0 and (session.get('review_mode') != REVIEW_MODE_CLIP_GENERATOR)
        if should_auto_export:
            try:
                session = _export_review_session(session)
                auto_exported = True
            except Exception as export_exc:
                auto_export_error = str(export_exc)
                logger.error('Auto-export after review decision failed: %s', export_exc, exc_info=True)
        session['video_url'] = _review_session_video_url(session)
        dataset_synced = bool(session.get('exports', {}).get('last_export_at'))
        response_payload = {'success': True, 'session': session, 'dataset_synced': dataset_synced, 'auto_exported': auto_exported}
        if auto_export_error:
            response_payload['sync_warning'] = auto_export_error
        return jsonify(response_payload)
    except FileNotFoundError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 404)
    except Exception as exc:
        logger.error('Error saving review decision: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/review/session/<session_id>/cropped-preview/<int:candidate_index>')
def review_cropped_preview(session_id, candidate_index):
    """Generate and serve a cropped 224x224 preview clip for a review candidate."""
    try:
        session = _load_review_session(session_id)
        candidates = session.get('candidates', [])
        if candidate_index < 0 or candidate_index >= len(candidates):
            return (jsonify({'success': False, 'error': 'candidate_index out of range'}), 404)
        candidate = candidates[candidate_index]
        source_video = _resolve_original_result_source(session.get('video_path', ''))
        if not _resolve_model_artifact(source_video).exists():
            return (jsonify({'success': False, 'error': 'Source video not found'}), 404)
        preview_path = _review_temp_dir() / f'{session_id}_candidate_{candidate_index:04d}.mp4'
        if not _resolve_model_artifact(preview_path).exists():
            ok, error_message = _write_segment_clip(source_video, float(candidate['start_time']), float(candidate['end_time']), preview_path, crop_for_training=True, target_size=224)
            if not ok:
                raise RuntimeError(error_message or 'Failed to generate cropped preview')
        return send_file(preview_path, mimetype='video/mp4', conditional=True)
    except FileNotFoundError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 404)
    except Exception as exc:
        logger.error('Error serving cropped preview: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/review/session/<session_id>/exported-clip/<int:candidate_index>')
def review_exported_clip(session_id, candidate_index):
    """Serve one clip after it has been exported from a review session."""
    try:
        session = _load_review_session(session_id)
        candidates = session.get('candidates', [])
        if candidate_index < 0 or candidate_index >= len(candidates):
            return (jsonify({'success': False, 'error': 'candidate_index out of range'}), 404)
        clip_path_value = candidates[candidate_index].get('exported_clip_path')
        if not clip_path_value:
            return (jsonify({'success': False, 'error': 'This candidate has not been exported yet'}), 404)
        clip_path = Path(clip_path_value).resolve()
        allowed_roots = [_training_dataset_sessions_dir().resolve(), _review_exports_dir().resolve(), _full_match_review_root().resolve()]
        if not any((root == clip_path.parent or root in clip_path.parents for root in allowed_roots)):
            return (jsonify({'success': False, 'error': 'Invalid exported clip path'}), 400)
        if clip_path.suffix.lower() not in INITIAL_TRAINING_VIDEO_EXTENSIONS:
            return (jsonify({'success': False, 'error': 'Invalid exported clip type'}), 400)
        if not _resolve_model_artifact(clip_path).exists() or not _resolve_model_artifact(clip_path).is_file():
            return (jsonify({'success': False, 'error': 'Exported clip not found'}), 404)
        return send_file(clip_path, mimetype='video/mp4', conditional=True)
    except FileNotFoundError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 404)
    except Exception as exc:
        logger.error('Error serving exported review clip: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/review/session/<session_id>/export', methods=['POST'])
def export_review_session(session_id):
    """Export reviewed detections as training-ready cropped clips plus manifests."""
    try:
        session = _load_review_session(session_id)
        if session['summary']['reviewed'] == 0:
            return (jsonify({'success': False, 'error': 'Review at least one detection before exporting'}), 400)
        session = _export_review_session(session)
        session['video_url'] = _review_session_video_url(session)
        return jsonify({'success': True, 'session': session, 'exports': session['exports']})
    except FileNotFoundError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 404)
    except Exception as exc:
        logger.error('Error exporting review session: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/preview_preprocessing', methods=['POST'])
def preview_preprocessing():
    """
    Preview preprocessed video.
    Creates a temporary video file showing the preprocessed frames.
    
    Expected JSON body:
    {
        "video_path": "path/to/video.mp4",
        "max_frames": 200  // Maximum frames to include
    }
    """
    try:
        import sys
        import tempfile
        import uuid
        models_path = PROJECT_ROOT / 'models' / 'M0'
        if str(models_path) not in sys.path:
            sys.path.insert(0, str(models_path))
        from data_utils import load_raw_video
        import cv2
        import numpy as np
        data = request.get_json()
        video_path_str = data.get('video_path')
        max_frames = data.get('max_frames', 200)
        if not video_path_str:
            return (jsonify({'success': False, 'error': 'Video path is required'}), 400)
        video_path = Path(video_path_str)
        if not video_path.is_absolute():
            if _resolve_model_artifact(app.config['UPLOAD_FOLDER'] / video_path.name).exists():
                video_path = app.config['UPLOAD_FOLDER'] / video_path.name
            elif _resolve_model_artifact(app.config['VIDEO_LIBRARY'] / video_path.name).exists():
                video_path = app.config['VIDEO_LIBRARY'] / video_path.name
            else:
                video_path = PROJECT_ROOT / video_path
        if not _resolve_model_artifact(video_path).exists():
            return (jsonify({'success': False, 'error': f'Video not found: {video_path_str}'}), 404)
        logger.info(f'Generating preprocessing preview video for: {video_path.name}')
        preview_fps = 25.0
        frames = load_raw_video(str(video_path), max_frames=max_frames, resize=(224, 224), target_fps=preview_fps)
        if len(frames) == 0:
            return (jsonify({'success': False, 'error': 'No frames could be extracted from video'}), 400)
        temp_dir = app.config['RESULTS_FOLDER'] / 'temp'
        temp_dir.mkdir(exist_ok=True)
        video_id = str(uuid.uuid4())[:8]
        output_path = temp_dir / f'preprocessed_{video_id}.mp4'
        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        logger.info(f'Creating video with {len(frames)} frames at {preview_fps} fps')
        import subprocess
        raw_path = output_path.with_suffix('.raw.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(raw_path), fourcc, preview_fps, (224, 224))
        for frame in frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        out.release()
        ffmpeg_cmd = ['ffmpeg', '-y', '-i', str(raw_path), '-c:v', 'libx264', '-preset', 'fast', '-crf', '28', '-pix_fmt', 'yuv420p', str(output_path)]
        result_ff = subprocess.run(ffmpeg_cmd, capture_output=True)
        raw_path.unlink(missing_ok=True)
        if result_ff.returncode != 0:
            logger.error(f'ffmpeg re-encode failed: {result_ff.stderr.decode()}')
            raise RuntimeError('ffmpeg failed')
        logger.info('Preview video created with H.264 (ffmpeg)')
        video_url = f'/api/temp_video/{video_id}'
        logger.info(f'Preprocessed video created: {output_path}')
        return jsonify({'success': True, 'video_url': video_url, 'total_frames': len(frames), 'preprocessing_steps': ['1. Normalize resolution to the 360 px training reference height', '2. Preserve aspect ratio and center crop to 224×224', '3. BGR to RGB conversion', f'4. Sampled at {preview_fps:g} FPS; sequence limited to {max_frames} frames'], 'original_video': {'width': original_width, 'height': original_height, 'fps': original_fps, 'total_frames': original_frames}, 'processed_video': {'width': 224, 'height': 224, 'fps': preview_fps, 'total_frames': len(frames)}})
    except Exception as e:
        logger.error(f'Error generating preprocessing preview: {e}', exc_info=True)
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/training_dataset/summary')
def training_dataset_summary():
    """Return summary statistics for the centralized human-annotated training dataset."""
    try:
        return jsonify({'success': True, 'summary': _training_dataset_summary()})
    except Exception as exc:
        logger.error('Error loading training dataset summary: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/temp_video/<video_id>')
def serve_temp_video(video_id):
    """Serve temporary preprocessed video file."""
    try:
        temp_dir = app.config['RESULTS_FOLDER'] / 'temp'
        video_file = temp_dir / f'preprocessed_{video_id}.mp4'
        if not _resolve_model_artifact(video_file).exists():
            logger.error(f'Video file not found: {video_file}')
            return ('Video not found', 404)
        file_size = video_file.stat().st_size
        logger.info(f'Serving video: {video_file.name} (size: {file_size} bytes)')
        if file_size == 0:
            logger.error(f'Video file is empty: {video_file}')
            return ('Video file is empty', 500)
        return send_file(video_file, mimetype='video/mp4', as_attachment=False, conditional=True)
    except Exception as e:
        logger.error(f'Error serving temp video: {e}', exc_info=True)
        return (str(e), 500)

@app.route('/api/export/<result_id>')
def export_result(result_id):
    """Export results as JSON or CSV."""
    try:
        format_type = request.args.get('format', 'json')
        result_file = app.config['RESULTS_FOLDER'] / f'result_{result_id}.json'
        if not _resolve_model_artifact(result_file).exists():
            return (jsonify({'success': False, 'error': 'Result not found'}), 404)
        if format_type == 'json':
            return send_file(result_file, mimetype='application/json', as_attachment=True, download_name=f'rugby_analysis_{result_id}.json')
        elif format_type == 'csv':
            import pandas as pd
            with open(_resolve_model_artifact(result_file), 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data['detections'])
            csv_file = app.config['RESULTS_FOLDER'] / f'result_{result_id}.csv'
            df.to_csv(csv_file, index=False)
            return send_file(csv_file, mimetype='text/csv', as_attachment=True, download_name=f'rugby_analysis_{result_id}.csv')
        else:
            return (jsonify({'success': False, 'error': 'Invalid format. Use "json" or "csv"'}), 400)
    except Exception as e:
        logger.error(f'Error exporting result: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/viewer')
def viewer():
    """Video viewer with timeline and detections."""
    result_id = request.args.get('result_id')
    return render_template('viewer.html', result_id=result_id)

@app.route('/review')
def review():
    """Active-learning review interface."""
    session_id = request.args.get('session_id')
    result_id = request.args.get('result_id')
    return render_template('review.html', session_id=session_id, result_id=result_id)

@app.route('/compare')
def compare():
    """Compare multiple models on the same video."""
    return render_template('compare.html')

@app.route('/api/comparison-history', methods=['GET', 'POST'])
def comparison_history():
    """List saved comparisons or persist a browser-side comparison."""
    try:
        if request.method == 'GET':
            history = []
            for comparison_file in _model_artifact_glob(app.config['COMPARISON_RESULTS_FOLDER'], 'comparison_*.json', recursive=False):
                try:
                    history.append(_comparison_summary(_load_json_file(comparison_file)))
                except Exception as exc:
                    logger.warning('Could not read comparison history file %s: %s', comparison_file.name, exc)
            history.sort(key=lambda item: str(item.get('created_at') or ''), reverse=True)
            return jsonify({'success': True, 'comparisons': history})
        data = request.get_json() or {}
        results = data.get('results') or {}
        model_names = data.get('model_names') or list(results)
        video_path = data.get('video_path')
        if not results or len(model_names) < 2 or (not video_path):
            return (jsonify({'success': False, 'error': 'Complete comparison data is required'}), 400)
        requested_id = str(data.get('id') or '')
        comparison_id = requested_id if re.fullmatch('[A-Za-z0-9_-]+', requested_id) else datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:6]
        payload = {'id': comparison_id, 'created_at': data.get('created_at') or _now_iso(), 'video_path': str(video_path), 'video_name': data.get('video_name') or Path(str(video_path)).name, 'model_names': model_names, 'parameters': data.get('parameters') or {}, 'results': results, 'comparison': data.get('comparison') or {}}
        _write_json_file(_comparison_file(comparison_id), payload)
        return (jsonify({'success': True, 'comparison': _comparison_summary(payload)}), 201)
    except Exception as exc:
        logger.error('Error handling comparison history: %s', exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/comparison-history/<comparison_id>', methods=['GET', 'DELETE'])
def comparison_history_item(comparison_id):
    """Retrieve or delete one saved comparison."""
    try:
        comparison_file = _comparison_file(comparison_id)
        if not _resolve_model_artifact(comparison_file).exists():
            return (jsonify({'success': False, 'error': 'Comparison not found'}), 404)
        if request.method == 'DELETE':
            comparison_file.unlink()
            return jsonify({'success': True})
        return jsonify({'success': True, 'comparison': _load_json_file(comparison_file)})
    except ValueError as exc:
        return (jsonify({'success': False, 'error': str(exc)}), 400)
    except Exception as exc:
        logger.error('Error handling comparison %s: %s', comparison_id, exc, exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """Compare multiple models on the same video."""
    try:
        data = request.get_json()
        model_names = data.get('model_names', [])
        video_path_str = data.get('video_path')
        parameters = data.get('parameters', {})
        if not model_names or not video_path_str:
            return (jsonify({'success': False, 'error': 'Model names and video path are required'}), 400)
        video_path = Path(video_path_str)
        if not video_path.is_absolute():
            if _resolve_model_artifact(app.config['UPLOAD_FOLDER'] / video_path.name).exists():
                video_path = app.config['UPLOAD_FOLDER'] / video_path.name
            elif _resolve_model_artifact(app.config['VIDEO_LIBRARY'] / video_path.name).exists():
                video_path = app.config['VIDEO_LIBRARY'] / video_path.name
            else:
                video_path = PROJECT_ROOT / video_path
        if not _resolve_model_artifact(video_path).exists():
            return (jsonify({'success': False, 'error': f'Video not found: {video_path_str}'}), 404)
        try:
            _validate_inference_source(video_path)
        except ValueError as exc:
            return (jsonify({'success': False, 'error': str(exc)}), 422)
        engine_kwargs = dict(confidence_threshold=parameters.get('confidence_threshold', 0.7), line_confidence_threshold=max(0.0, min(1.0, _to_float(parameters.get('line_confidence_threshold'), 0.95))), batch_size=parameters.get('batch_size', 8))
        results = {}
        for model_name in model_names:
            logger.info(f'Running comparison with {model_name}')
            inference_engine.load_model(model_name)
            model_results = inference_engine.analyze_video(video_path, **engine_kwargs)
            results[model_name] = model_results
        comparison = {}
        for model_name, res in results.items():
            dets = res.get('detections', [])
            by_type = {}
            for d in dets:
                pt = d.get('play_type', 'unknown')
                by_type[pt] = by_type.get(pt, 0) + 1
            avg_conf = sum((d['confidence'] for d in dets)) / len(dets) if dets else 0.0
            comparison[model_name] = {'total': len(dets), 'by_type': by_type, 'avg_confidence': avg_conf}
        comparison_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:6]
        comparison_payload = {'id': comparison_id, 'created_at': _now_iso(), 'video_path': str(video_path), 'video_name': video_path.name, 'model_names': model_names, 'parameters': {'confidence_threshold': engine_kwargs['confidence_threshold'], 'line_confidence_threshold': engine_kwargs['line_confidence_threshold']}, 'results': results, 'comparison': comparison}
        _write_json_file(_comparison_file(comparison_id), comparison_payload)
        return jsonify({'success': True, 'results': results, 'comparison': comparison, 'comparison_id': comparison_id})
    except Exception as e:
        logger.error(f'Error comparing models: {e}', exc_info=True)
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/model-info/<model_name>')
def get_model_info(model_name):
    """Get detailed information about a model."""
    try:
        info = inference_engine.get_model_info(model_name)
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        logger.error(f'Error getting model info: {e}')
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/play-segment/<result_id>/<int:detection_index>')
def play_segment(result_id, detection_index):
    """Return segment metadata. video_url points to the safe serve endpoint."""
    try:
        result_file = app.config['RESULTS_FOLDER'] / f'result_{result_id}.json'
        if not _resolve_model_artifact(result_file).exists():
            return (jsonify({'success': False, 'error': 'Result not found'}), 404)
        with open(_resolve_model_artifact(result_file), 'r') as f:
            result_data = json.load(f)
        if detection_index >= len(result_data['detections']):
            return (jsonify({'success': False, 'error': 'Detection index out of range'}), 404)
        detection = result_data['detections'][detection_index]
        return jsonify({'success': True, 'video_url': f'/api/video-for-result/{result_id}', 'start_time': detection['start_time'], 'end_time': detection['end_time'], 'play_type': detection['play_type'], 'confidence': detection['confidence']})
    except Exception as e:
        logger.error(f'Error getting segment info: {e}', exc_info=True)
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/video-for-result/<result_id>')
def video_for_result(result_id):
    """Serve the source video for a given result ID (Range-request aware)."""
    try:
        result_file = app.config['RESULTS_FOLDER'] / f'result_{result_id}.json'
        if not _resolve_model_artifact(result_file).exists():
            return (jsonify({'success': False, 'error': 'Result not found'}), 404)
        with open(_resolve_model_artifact(result_file), 'r') as f:
            result_data = json.load(f)
        raw_path = result_data.get('video_path', '')
        video_file = _resolve_result_playback_file(raw_path)
        if not _resolve_model_artifact(video_file).exists():
            logger.error(f'video_for_result: file not found: {raw_path}')
            return (jsonify({'success': False, 'error': 'Video file not found'}), 404)
        logger.info(f'Serving video for result {result_id}: {video_file.name}')
        return send_file(video_file, conditional=True)
    except Exception as e:
        logger.error(f'Error serving video for result: {e}', exc_info=True)
        return (jsonify({'success': False, 'error': str(e)}), 500)

def _resolve_result_source_file(raw_path: str) -> Path:
    """Resolve a result's source video using the same library fallbacks."""
    video_file = Path(raw_path)
    if _resolve_model_artifact(video_file).exists():
        return video_file
    filename = video_file.name
    for folder in [app.config['VIDEO_LIBRARY'], app.config['UPLOAD_FOLDER']]:
        candidate = folder / filename
        if _resolve_model_artifact(candidate).exists():
            return candidate
        nested_candidate = next(_model_artifact_glob(folder, filename, recursive=True), None)
        if nested_candidate is not None:
            return nested_candidate
    return video_file

def _video_dimensions(video_file: Path) -> tuple[int, int]:
    """Read the encoded video dimensions without decoding the complete file."""
    metadata = read_video_metadata(video_file)
    return (int(metadata['width']), int(metadata['height']))

def _path_is_within(path: Path, folder: Path) -> bool:
    """Return whether ``path`` belongs to ``folder``, including the folder itself."""
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except (OSError, ValueError):
        return False

def _supports_training_preprocessing(video_file: Path) -> bool:
    """Accept decodable raw videos and prepared clips at the model input size."""
    if not _resolve_model_artifact(video_file).exists() or not _resolve_model_artifact(video_file).is_file():
        return False
    width, height = _video_dimensions(video_file)
    return width > 0 and height > 0

def _validate_inference_source(video_file: Path) -> None:
    """Validate raw or prepared media before web inference."""
    if _supports_training_preprocessing(video_file):
        return
    raise ValueError('No se pudo leer la resolución del video seleccionado. Verificá que el archivo no esté dañado y que el formato sea compatible.')

@lru_cache(maxsize=256)
def _resolve_result_playback_file(raw_path: str) -> Path:
    """Serve the same available video used for inference."""
    return _resolve_result_source_file(raw_path)


def _resolve_original_result_source(raw_path: str) -> Path:
    """Resolve decodable raw or prepared input without requiring another video."""
    source_file = _resolve_result_playback_file(raw_path)
    if _supports_training_preprocessing(source_file):
        return source_file
    raise ValueError(f'No se puede leer el video de este resultado: {source_file}')


def _effective_result_fps(result_data: dict, source_fps: float) -> float:
    """Match the temporal FPS cap used by sliding-window inference."""
    metadata = result_data.get('metadata') or {}
    mode = str(metadata.get('temporal_fps_mode') or 'training_match_25').strip().lower()
    custom_fps = metadata.get('custom_effective_fps')
    if mode == 'fast_12':
        requested_fps = 12.0
    elif mode == 'custom':
        try:
            requested_fps = max(1.0, float(custom_fps))
        except (TypeError, ValueError):
            requested_fps = 25.0
    else:
        requested_fps = 25.0
    model_budget_fps = 200.0 / 7.0
    return max(1.0, min(float(source_fps or requested_fps), model_budget_fps, requested_fps))

def _sample_result_indices(total_frames: int, source_fps: float, target_fps: float) -> list[int]:
    """Sample source indices with the same timestamp cadence as inference."""
    if total_frames <= 0:
        return []
    if target_fps <= 0 or source_fps <= 0:
        return list(range(total_frames))
    duration = total_frames / source_fps
    sample_count = max(1, int(np.floor(duration * target_fps)))
    sample_times = np.arange(sample_count, dtype=np.float32) / float(target_fps)
    indices = np.rint(sample_times * float(source_fps)).astype(np.int32)
    indices = np.clip(indices, 0, total_frames - 1)
    return indices.tolist()

def _preprocessed_cache_is_current(output_path: Path, metadata_path: Path, source_file: Path, target_fps: float) -> bool:
    if not _resolve_model_artifact(output_path).exists() or not _resolve_model_artifact(metadata_path).exists():
        return False
    try:
        cache_metadata = _load_json_file(metadata_path)
        source_stat = source_file.stat()
        return cache_metadata.get('source_path') == str(source_file.resolve()) and int(cache_metadata.get('source_size', -1)) == int(source_stat.st_size) and (int(cache_metadata.get('source_mtime_ns', -1)) == int(source_stat.st_mtime_ns)) and (abs(float(cache_metadata.get('effective_fps', 0.0)) - float(target_fps)) < 1e-06) and (cache_metadata.get('spatial_preprocessing_version') == SPATIAL_PREPROCESSING_VERSION)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False

def _build_preprocessed_result_video(result_id: str, result_data: dict, source_file: Path) -> Path:
    """Create and cache the exact 224x224 video representation used by inference."""
    cache_folder = app.config['PREPROCESSED_VIDEO_FOLDER']
    output_path = cache_folder / f'preprocessed_{result_id}.mp4'
    metadata_path = cache_folder / f'preprocessed_{result_id}.json'
    cap = cv2.VideoCapture(str(source_file))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open source video: {source_file}')
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    target_fps = _effective_result_fps(result_data, source_fps)
    source_stat = source_file.stat()
    if _preprocessed_cache_is_current(output_path, metadata_path, source_file, target_fps):
        cap.release()
        return output_path
    try:
        spatial_geometry = training_normalization_geometry(width, height, target_size=224)
    except ValueError as exc:
        cap.release()
        raise ValueError(f'Invalid source video geometry {width}x{height}: {exc}') from exc
    sample_indices = _sample_result_indices(total_frames, source_fps, target_fps)
    sample_counts = Counter(sample_indices)
    raw_output_path = output_path.with_suffix('.raw.mp4')
    output_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    raw_output_path.unlink(missing_ok=True)
    cache_folder.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(raw_output_path), cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (224, 224))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot create preprocessed video: {raw_output_path}')
    written_frames = 0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            repeat_count = sample_counts.get(frame_index, 0)
            if repeat_count:
                cropped = _training_center_crop(frame, target_size=224)
                if cropped is None:
                    raise ValueError('A source frame is too small for the 224x224 training crop.')
                for _ in range(repeat_count):
                    writer.write(cropped)
                    written_frames += 1
            frame_index += 1
    finally:
        writer.release()
        cap.release()
    if written_frames == 0:
        raw_output_path.unlink(missing_ok=True)
        raise RuntimeError('No frames were written to the preprocessed video.')
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        transcode = subprocess.run([ffmpeg_path, '-y', '-i', str(raw_output_path), '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output_path)], capture_output=True, text=True)
        raw_output_path.unlink(missing_ok=True)
        if transcode.returncode != 0:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(transcode.stderr.strip() or 'ffmpeg failed while encoding the preprocessed video.')
    else:
        raw_output_path.replace(output_path)
    metadata_path.write_text(json.dumps({'result_id': result_id, 'source_path': str(source_file.resolve()), 'source_size': int(source_stat.st_size), 'source_mtime_ns': int(source_stat.st_mtime_ns), 'source_fps': source_fps, 'source_total_frames': total_frames, 'effective_fps': target_fps, 'sampled_total_frames': len(sample_indices), 'width': 224, 'height': 224, 'spatial_preprocessing_version': SPATIAL_PREPROCESSING_VERSION, 'spatial_transform': 'preserve aspect ratio, normalize to the 360 px training reference height, then fixed center crop 224x224', 'spatial_geometry': spatial_geometry}, indent=2), encoding='utf-8')
    return output_path

@app.route('/api/preprocessed-video/<result_id>')
def preprocessed_video_for_result(result_id):
    """Serve the cached video representation consumed by web inference."""
    try:
        result_file = app.config['RESULTS_FOLDER'] / f'result_{result_id}.json'
        if not _resolve_model_artifact(result_file).exists():
            return (jsonify({'success': False, 'error': 'Result not found'}), 404)
        result_data = _load_json_file(result_file)
        source_file = _resolve_original_result_source(result_data.get('video_path', ''))
        if not _resolve_model_artifact(source_file).exists() or not _resolve_model_artifact(source_file).is_file():
            return (jsonify({'success': False, 'error': 'Source video file not found'}), 404)
        with PREPROCESSED_VIDEO_LOCK:
            output_path = _build_preprocessed_result_video(result_id, result_data, source_file)
        return send_file(output_path, mimetype='video/mp4', conditional=True)
    except ValueError as exc:
        logger.warning(f'Preprocessed video unavailable for {result_id}: {exc}')
        return (jsonify({'success': False, 'error': str(exc)}), 422)
    except Exception as exc:
        logger.error(f'Error serving preprocessed video for {result_id}: {exc}', exc_info=True)
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/video-by-path')
def video_by_path():
    """Serve a video resolved from a query path parameter."""
    try:
        video_path_str = request.args.get('video_path', '')
        if not video_path_str:
            return (jsonify({'success': False, 'error': 'video_path query parameter is required'}), 400)
        video_file = _resolve_result_playback_file(video_path_str)
        if not _resolve_model_artifact(video_file).exists() or not _resolve_model_artifact(video_file).is_file():
            return (jsonify({'success': False, 'error': f'Video file not found: {video_path_str}'}), 404)
        return send_file(video_file, conditional=True)
    except Exception as e:
        logger.error(f'Error serving video by path: {e}', exc_info=True)
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/api/serve-video/<path:video_path>')
def serve_video(video_path):
    """Serve video file with support for range requests (seeking).

    Flask's <path:> converter strips the leading slash, so Windows absolute
    paths like 'D:\\Tesina\\...' arrive as 'D:Tesina\\...'.  We reconstruct
    the real path by trying several strategies before giving up.
    """
    try:
        candidates = [Path(video_path)]
        if len(video_path) >= 2 and video_path[1] in ('/', '\\', ':'):
            drive = video_path[0]
            rest = video_path[2:].replace('/', '\\')
            candidates.append(Path(f'{drive}:\\{rest}'))
        candidates.append(Path(video_path.replace('/', '\\')))
        filename = Path(video_path).name
        for folder in [app.config['VIDEO_LIBRARY'], app.config['UPLOAD_FOLDER']]:
            candidates.append(folder / filename)
        video_file = None
        for c in candidates:
            if _resolve_model_artifact(c).exists() and _resolve_model_artifact(c).is_file():
                video_file = c
                break
        if video_file is None:
            logger.error(f'serve_video: file not found. Tried: {candidates}')
            return (jsonify({'success': False, 'error': f'Video file not found: {filename}'}), 404)
        logger.info(f'Serving video: {video_file}')
        return send_file(video_file, mimetype='video/mp4', conditional=True)
    except Exception as e:
        logger.error(f'Error serving video: {e}', exc_info=True)
        return (jsonify({'success': False, 'error': str(e)}), 500)

@app.route('/training-results')
def training_results_page():
    return render_template('training_results.html')

@app.route('/api/training-results')
def api_training_results():
    """Return available training records and model metadata, with their scope."""
    from src.web.training_history import read_training_history
    _, dataset = _initial_training_catalog()
    published = []
    for model in inference_engine.list_available_models():
        metrics = model.get('metrics') or {}
        published.append({'model': model['display_name'], 'architecture': model.get('architecture'),
                          'reference_accuracy': metrics.get('test_accuracy', metrics.get('accuracy')),
                          'reference_f1_macro': metrics.get('f1_macro')})
    return jsonify({'success': True, 'dataset': dataset, 'published_models': published,
                    **read_training_history(PROJECT_ROOT)})

@app.route('/api/training-results/image/<path:rel_path>')
def serve_training_image(rel_path):
    """Serve a training result image (PNG) by its path relative to PROJECT_ROOT."""
    abs_path = PROJECT_ROOT / rel_path
    try:
        abs_path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return (jsonify({'error': 'Forbidden'}), 403)
    if not _resolve_model_artifact(abs_path).exists():
        return (jsonify({'error': 'Not found'}), 404)
    return send_file(abs_path, mimetype='image/png')
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
