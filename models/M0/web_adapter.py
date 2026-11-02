"""
Adapter to use InceptionV3-GRU Keras model with the web interface.

This script provides inference capabilities for rugby video classification
compatible with the web interface structure.

IMPORTANT: Uses the SAME preprocessing pipeline as training for consistency.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os
import numpy as np
import keras
import cv2
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from models.M0.config import *
from models.M0.data_utils import load_video, load_raw_video, preprocess_raw_frame, apply_clahe, _sample_frame_indices
from models.inference_cache import cached_features, load_with_revision

class RugbyPlayDetector:
    """Adapter class for using Keras model in web interface."""
    _KNOWN_EXTRACTORS = ('efficientnetb0', 'inceptionv3', 'resnet50')
    WEB_MAX_EFFECTIVE_FPS = 25.0
    WEB_FAST_EFFECTIVE_FPS = 12.0
    WEB_FEATURE_EXTRACTION_BATCH = 128
    WEB_FEATURE_PREDICT_BATCH = 8
    EVENT_TARGET_LABELS = ('scrum', 'line')
    EVENT_CONTEXT_RADIUS_SECONDS = 3.0
    EVENT_MERGE_GAP_SECONDS = 2.0
    EVENT_ANCHOR_CONFIDENCE = 0.85
    EVENT_NEIGHBOR_BOOST = 0.45
    EVENT_MIN_TARGET_CONFIDENCE = 0.0
    EVENT_REQUIRE_TARGET_WINNER = False
    EVENT_MERGE_BY_START_TIME = False
    EVENT_MIN_SUPPORT_WINDOWS = 1
    EVENT_MAX_DURATION_SECONDS = None

    def __init__(self, model_dir: str=None, model_name: str=None, extractor_name: str=None, preprocessing_filter: str=None):
        """
        Initialize the detector with trained models.

        Args:
            model_dir: Directory containing the trained models (default: current directory)
            model_name: Specific sequence model filename to load (e.g.,
                        'grid_search/resnet50__gru__u64__bi__drop3__d64__lr1e-4__bs16/model.keras').
                        If None, loads the first sequence model found.
            extractor_name: Override which feature extractor to use
                            ('resnet50', 'inceptionv3', 'efficientnetb0').
                            When None, the extractor is inferred from *model_name*
                            (grid search trial names encode the backbone as the
                            first double-underscore-delimited segment).
                            Falls back to the global FEATURE_EXTRACTOR constant
                            when neither is specified.
            preprocessing_filter: Optional web preprocessing variant.  ``clahe``
                                  matches the isolated CLAHE training experiment.
        """
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = model_dir
        self.checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        self.requested_model_name = model_name
        self.preprocessing_filter = (preprocessing_filter or 'none').strip().lower()
        if self.preprocessing_filter not in {'none', 'clahe'}:
            raise ValueError(f'Unsupported preprocessing filter: {preprocessing_filter}')
        self.extractor_name = extractor_name or self._detect_extractor_from_model_name(model_name)
        print('Loading models...')
        print(f'  Feature extractor: {self.extractor_name}')
        self.feature_extractor = self._load_feature_extractor()
        self.sequence_model = self._load_sequence_model()
        self.label_vocab = self._load_label_vocab()
        print(f'Models loaded successfully!')
        print(f'Labels: {self.label_vocab}')

    def _apply_frame_filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply the selected BGR filter after the training spatial transform."""
        if self.preprocessing_filter == 'clahe':
            return apply_clahe(frame)
        return frame

    def _detect_extractor_from_model_name(self, model_name: Optional[str]) -> str:
        """
        Infer the feature extractor backbone from the model path.

        Grid search trial directories encode the backbone as the first
        double-underscore-delimited segment, e.g.:
            grid_search/resnet50__gru__u64__bi__drop3__d64__lr1e-4__bs16/model.keras
                         ^^^^^^^  → 'resnet50'

        If the backbone cannot be determined, falls back to the global
        FEATURE_EXTRACTOR constant defined in config.py.
        """
        if not model_name:
            return FEATURE_EXTRACTOR
        for part in Path(model_name).parts:
            for extractor in self._KNOWN_EXTRACTORS:
                if part == extractor or part.startswith(extractor + '__'):
                    return extractor
        return FEATURE_EXTRACTOR

    def _load_feature_extractor(self):
        """Load the feature extractor model that matches this trial's backbone."""
        model_path = os.path.join(self.checkpoints_dir, f'feature_extractor_{self.extractor_name}.keras')
        if not os.path.exists(_resolve_model_artifact(model_path)):
            raise FileNotFoundError(f'Feature extractor not found: {model_path}')
        model, self._feature_cache_rgb_revision = load_with_revision(_resolve_model_artifact(model_path), keras.models.load_model)
        return model

    def _load_sequence_model(self):
        """Load the sequence model (GRU/LSTM)."""
        if self.requested_model_name:
            model_path = os.path.join(self.checkpoints_dir, self.requested_model_name)
            if not os.path.exists(_resolve_model_artifact(model_path)):
                raise FileNotFoundError(f'Requested sequence model not found: {model_path}')
            print(f'Loading sequence model: {self.requested_model_name}')
            return keras.models.load_model(_resolve_model_artifact(model_path))
        checkpoint_files = os.listdir(self.checkpoints_dir)
        sequence_models = [f for f in checkpoint_files if f.startswith('sequence_model_') and f.endswith('.keras')]
        if not sequence_models:
            raise FileNotFoundError(f'No sequence model found in {self.checkpoints_dir}')
        model_name = sequence_models[0]
        model_path = os.path.join(self.checkpoints_dir, model_name)
        print(f'Loading sequence model: {model_name}')
        return keras.models.load_model(_resolve_model_artifact(model_path))

    def _load_label_vocab(self):
        return np.asarray(['line', 'other', 'scrum'])

    def _resolve_temporal_sampling_fps(self, source_fps: float, window_duration: int, temporal_fps_mode: str='training_match_25', custom_effective_fps: Optional[float]=None) -> float:
        """Resolve effective temporal FPS for web sliding-window analysis."""
        model_budget_fps = MAX_SEQ_LENGTH / max(window_duration, 1)
        source_budget_fps = source_fps if source_fps > 0 else model_budget_fps
        mode = (temporal_fps_mode or 'training_match_25').strip().lower()
        if mode == 'fast_12':
            requested_cap = self.WEB_FAST_EFFECTIVE_FPS
        elif mode == 'custom':
            try:
                requested_cap = float(custom_effective_fps)
            except (TypeError, ValueError):
                requested_cap = self.WEB_MAX_EFFECTIVE_FPS
            requested_cap = max(1.0, requested_cap)
        else:
            requested_cap = self.WEB_MAX_EFFECTIVE_FPS
        return min(source_budget_fps, model_budget_fps, requested_cap)

    def _build_temporal_sample_indices(self, source_length: int, target_length: int) -> np.ndarray:
        """Return uniformly spaced indices for temporal standardization."""
        if source_length <= 0 or target_length <= 0:
            return np.array([], dtype=np.int32)
        if target_length >= source_length:
            return np.arange(source_length, dtype=np.int32)
        return np.linspace(0, source_length - 1, num=target_length, dtype=np.int32)

    def _predict_feature_batch(self, frames: List[np.ndarray], initial_batch_size: int) -> np.ndarray:
        """Extract features with adaptive micro-batching to prevent memory failures."""
        if not frames:
            return np.empty((0, 0), dtype=np.float32)
        frame_array = np.array(frames, dtype=np.uint8)
        batch_size = max(1, int(initial_batch_size))
        last_error = None
        while batch_size >= 1:
            try:
                return self.feature_extractor.predict(frame_array, batch_size=batch_size, verbose=0)
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError, tf.errors.AbortedError, MemoryError) as exc:
                last_error = exc
                if batch_size == 1:
                    break
                next_batch_size = max(1, batch_size // 2)
                print(f'Feature extraction batch_size={batch_size} failed ({type(exc).__name__}); retrying with {next_batch_size}')
                batch_size = next_batch_size
        raise RuntimeError(f'Feature extraction failed even with batch_size=1: {last_error}')

    def _begin_auxiliary_sequence(self, sampled_total_frames: int) -> None:
        return None

    def _on_sampled_frame(self, frame_bgr: np.ndarray) -> None:
        return None

    def _on_raw_sampled_frame(self, frame_bgr: np.ndarray) -> None:
        """Observe a sampled frame before the 224x224 RGB transform.

        Auxiliary detectors that need the source-resolution equivalent ROI
        can override this hook.  Existing detectors continue to use
        ``_on_sampled_frame`` and therefore retain their historical input.
        """
        return None

    def _finish_auxiliary_sequence(self) -> None:
        return None

    def _predict_sequence_window(self, padded_features: np.ndarray, mask: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
        return self.sequence_model.predict([np.expand_dims(padded_features, axis=0), np.expand_dims(mask, axis=0)], verbose=0)

    def _post_process_event_detections(self, window_predictions: List[Dict], confidence_threshold: float, video_duration: float, source_fps: float, class_confidence_thresholds: Optional[Dict[str, float]]=None) -> List[Dict]:
        """
        Convert overlapping positive windows into longer event detections.

        Sliding windows are evidence, not final events. The model-specific
        policy can require the target class to be the window winner, disable
        neighbour promotion, and cap event duration before producing an
        event-level detection with a real start/end span.
        """
        self._last_review_window_detections = []
        if not window_predictions:
            return []
        target_labels = [label for label in self.EVENT_TARGET_LABELS if any((label in item.get('probabilities', {}) for item in window_predictions))]
        if not target_labels:
            return []
        evidence_threshold = max(float(confidence_threshold), float(getattr(self, 'EVENT_MIN_TARGET_CONFIDENCE', 0.0)))
        configured_thresholds = class_confidence_thresholds or {}
        evidence_thresholds = {label: max(evidence_threshold, float(configured_thresholds.get(label, 0.0))) for label in target_labels}
        anchors_by_label = {label: [item for item in window_predictions if item.get('probabilities', {}).get(label, 0.0) >= self.EVENT_ANCHOR_CONFIDENCE] for label in target_labels}
        evidence = []
        for item in window_predictions:
            boosted_probs = dict(item.get('probabilities', {}))
            raw_label = item.get('play_type')
            if self.EVENT_REQUIRE_TARGET_WINNER and raw_label not in target_labels:
                continue
            for label in target_labels:
                base_prob = float(boosted_probs.get(label, 0.0))
                best_influence = 0.0
                item_center = (item['start_time'] + item['end_time']) / 2.0
                for anchor in anchors_by_label[label]:
                    if anchor is item:
                        continue
                    gap = max(0.0, max(item['start_time'], anchor['start_time']) - min(item['end_time'], anchor['end_time']))
                    center_distance = abs(item_center - (anchor['start_time'] + anchor['end_time']) / 2.0)
                    if gap <= self.EVENT_MERGE_GAP_SECONDS and center_distance <= self.EVENT_CONTEXT_RADIUS_SECONDS + item['duration']:
                        temporal_weight = max(0.0, 1.0 - center_distance / max(self.EVENT_CONTEXT_RADIUS_SECONDS + item['duration'], 1e-06))
                        anchor_prob = float(anchor.get('probabilities', {}).get(label, 0.0))
                        best_influence = max(best_influence, anchor_prob * temporal_weight)
                if best_influence > 0.0:
                    boosted_probs[label] = min(1.0, base_prob + self.EVENT_NEIGHBOR_BOOST * best_influence * (1.0 - base_prob))
            best_target = max(target_labels, key=lambda label: boosted_probs.get(label, 0.0))
            best_target_prob = float(boosted_probs.get(best_target, 0.0))
            original_target_prob = float(item.get('probabilities', {}).get(best_target, 0.0))
            if best_target_prob >= evidence_thresholds[best_target] and (not self.EVENT_REQUIRE_TARGET_WINNER or best_target == raw_label):
                evidence_item = dict(item)
                evidence_item['play_type'] = best_target
                evidence_item['confidence'] = best_target_prob
                evidence_item['original_confidence'] = original_target_prob
                evidence_item['probabilities'] = boosted_probs
                evidence.append(evidence_item)
        if not evidence:
            return []
        self._last_review_window_detections = sorted([dict(item) for item in evidence], key=lambda item: (item['start_time'], item['end_time'], item['play_type']))
        evidence.sort(key=lambda item: (item['play_type'], item['start_time'], item['end_time']))
        merged_events = []
        current = None
        for item in evidence:
            merge_reference = None if current is None else current.get('last_start_time', current['end_time']) if self.EVENT_MERGE_BY_START_TIME else current['end_time']
            if current is None or item['play_type'] != current['play_type'] or item['start_time'] - merge_reference > self.EVENT_MERGE_GAP_SECONDS:
                if current is not None:
                    if self._event_has_support(current):
                        merged_events.append(self._finalize_event_detection(current, video_duration, source_fps))
                current = {'play_type': item['play_type'], 'start_time': item['start_time'], 'end_time': item['end_time'], 'last_start_time': item['start_time'], 'windows': [item]}
                continue
            max_duration = getattr(self, 'EVENT_MAX_DURATION_SECONDS', None)
            if max_duration is not None and item['end_time'] - current['start_time'] > float(max_duration):
                if self._event_has_support(current):
                    merged_events.append(self._finalize_event_detection(current, video_duration, source_fps))
                current = {'play_type': item['play_type'], 'start_time': max(float(item['start_time']), float(current['end_time'])), 'end_time': item['end_time'], 'last_start_time': item['start_time'], 'windows': [item]}
                continue
            current['start_time'] = min(current['start_time'], item['start_time'])
            current['end_time'] = max(current['end_time'], item['end_time'])
            current['last_start_time'] = item['start_time']
            current['windows'].append(item)
        if current is not None:
            if self._event_has_support(current):
                merged_events.append(self._finalize_event_detection(current, video_duration, source_fps))
        merged_events.sort(key=lambda item: item['start_time'])
        return merged_events

    def _event_has_support(self, event: Dict) -> bool:
        """Reject isolated low-confidence windows from final event output."""
        windows = event.get('windows', [])
        if len(windows) >= int(getattr(self, 'EVENT_MIN_SUPPORT_WINDOWS', 1)):
            return True
        peak = max((float(item.get('confidence', 0.0)) for item in windows), default=0.0)
        return peak >= float(self.EVENT_ANCHOR_CONFIDENCE)

    def _finalize_event_detection(self, event: Dict, video_duration: float, source_fps: float) -> Dict:
        """Build the public detection object for a merged event."""
        windows = event['windows']
        start_time = max(0.0, float(event['start_time']))
        end_time = min(float(video_duration), float(event['end_time']))
        confidences = [float(item['confidence']) for item in windows]
        peak_window = max(windows, key=lambda item: item['confidence'])
        probability_sums: Dict[str, float] = {}
        for item in windows:
            for label, value in item.get('probabilities', {}).items():
                probability_sums[label] = probability_sums.get(label, 0.0) + float(value)
        count = max(1, len(windows))
        averaged_probabilities = {label: value / count for label, value in probability_sums.items()}
        return {'play_type': event['play_type'], 'start_time': start_time, 'end_time': end_time, 'duration': max(0.0, end_time - start_time), 'confidence': max(confidences), 'mean_confidence': float(np.mean(confidences)), 'frame_start': int(round(start_time * source_fps)) if source_fps > 0 else int(peak_window['frame_start']), 'frame_end': int(round(end_time * source_fps)) if source_fps > 0 else int(peak_window['frame_end']), 'probabilities': averaged_probabilities, 'source_window_count': count, 'peak_window': {'start_time': float(peak_window['start_time']), 'end_time': float(peak_window['end_time']), 'confidence': float(peak_window['confidence']), 'original_confidence': float(peak_window.get('original_confidence', peak_window['confidence']))}, 'contributing_windows': [{'start_time': float(item['start_time']), 'end_time': float(item['end_time']), 'confidence': float(item['confidence']), 'original_confidence': float(item.get('original_confidence', item['confidence']))} for item in windows]}

    def analyze_video(self, video_path: str, confidence_threshold: float=0.5, segment_duration: int=5, use_cache: bool=True, cache_dir=None) -> Dict:
        """
        Analyze a video and detect rugby plays.
        
        IMPORTANT: Uses the SAME preprocessing as training:
        - Center crop to square
        - Resize to IMG_SIZE x IMG_SIZE (224x224)
        - BGR to RGB conversion
        - Normalization via feature extractor
        
        Args:
            video_path: Path to the video file
            confidence_threshold: Minimum confidence for predictions (0.0-1.0)
            segment_duration: Duration of segments in seconds (not used, kept for compatibility)
            use_cache: Generate and reuse local RGB features (default True).
            cache_dir: Optional local cache folder; defaults to .runtime/cache/features.
            
        Returns:
            Dictionary with detection results
        """
        print(f'\nAnalyzing video: {video_path}')
        try:
            frames = load_raw_video(video_path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE), target_fps=25.0, frame_filter=self._apply_frame_filter if self.preprocessing_filter != 'none' else None)
            print(f'Loaded {len(frames)} frames from video')
        except Exception as e:
            print(f'Error loading video: {e}')
            return {'status': 'error', 'error': str(e), 'detections': []}
        print('Extracting features...')
        features, cache_info = cached_features(self, video_path, {'mode': 'clip', 'target_fps': 25.0, 'max_frames': MAX_SEQ_LENGTH}, len(frames), lambda: self.feature_extractor.predict(frames, verbose=0), use_cache=use_cache, cache_dir=cache_dir, auxiliary=False)
        if len(features) < MAX_SEQ_LENGTH:
            padded_features = np.zeros((MAX_SEQ_LENGTH, features.shape[1]), dtype=np.float32)
            padded_features[:len(features)] = features
            mask = np.zeros(MAX_SEQ_LENGTH, dtype=np.float32)
            mask[:len(features)] = 1.0
        else:
            padded_features = features[:MAX_SEQ_LENGTH]
            mask = np.ones(MAX_SEQ_LENGTH, dtype=np.float32)
        features_batch = np.expand_dims(padded_features, axis=0)
        mask_batch = np.expand_dims(mask, axis=0)
        print('Running inference...')
        predictions = self.sequence_model.predict([features_batch, mask_batch], verbose=0)
        pred_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_class_idx]
        predicted_label = self.label_vocab[pred_class_idx]
        print(f'Prediction: {predicted_label} (confidence: {confidence:.2%})')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        results = {'status': 'success', 'feature_cache': cache_info, 'video_path': video_path, 'preprocessing_filter': self.preprocessing_filter, 'duration': duration, 'predicted_class': predicted_label, 'confidence': float(confidence), 'all_probabilities': {label: float(prob) for label, prob in zip(self.label_vocab, predictions[0])}, 'detections': [{'play_type': predicted_label, 'start_time': 0.0, 'end_time': duration, 'confidence': float(confidence), 'frame_start': 0, 'frame_end': total_frames}] if confidence >= confidence_threshold else []}
        return results

    def _extract_sampled_video_features(self, video_path, sample_index_set, total_frames, feature_batch_size, feature_predict_batch_size, progress_callback):
        feature_chunks = []
        frame_batch = []
        sampled_count = 0
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count in sample_index_set:
                    self._on_raw_sampled_frame(frame)
                    frame = preprocess_raw_frame(frame, target_size=IMG_SIZE)
                    if frame is None:
                        raise ValueError('Raw frame could not be normalized for the 224x224 training crop')
                    self._on_sampled_frame(frame)
                    frame = self._apply_frame_filter(frame)
                    frame = frame[:, :, [2, 1, 0]]
                    frame_batch.append(frame)
                    sampled_count += 1
                    if len(frame_batch) >= feature_batch_size:
                        batch_features = self._predict_feature_batch(frame_batch, initial_batch_size=min(feature_predict_batch_size, len(frame_batch)))
                        feature_chunks.append(batch_features)
                        frame_batch.clear()
                        if progress_callback and total_frames > 0:
                            extraction_ratio = min(1.0, frame_count / total_frames)
                            progress_callback(20 + extraction_ratio * 50, 'features', f'Extrayendo características: {sampled_count} cuadros muestreados')
                frame_count += 1
        finally:
            cap.release()
        if frame_batch:
            batch_features = self._predict_feature_batch(frame_batch, initial_batch_size=min(feature_predict_batch_size, len(frame_batch)))
            feature_chunks.append(batch_features)
        if not feature_chunks:
            raise ValueError('No frames available after temporal standardization')
        self._finish_auxiliary_sequence()
        return np.vstack(feature_chunks)

    def analyze_video_segments(self, video_path: str, segment_duration: int=5, confidence_threshold: float=0.5, line_confidence_threshold: Optional[float]=None, temporal_fps_mode: str='training_match_25', custom_effective_fps: Optional[float]=None, feature_batch_size: int=WEB_FEATURE_EXTRACTION_BATCH, feature_predict_batch_size: int=WEB_FEATURE_PREDICT_BATCH, progress_callback=None, use_cache: bool=True, cache_dir=None, feature_provider=None) -> Dict:
        """
        Analyze video using sliding window approach for better temporal localization.
        
        Uniform temporal sampling and feature extraction precede classification
        in 7-second windows with a 1-second stride.
        
        IMPORTANT: Each window uses the SAME preprocessing as training.
        
        Args:
            video_path: Path to the video file
            segment_duration: Compatibility argument; sliding windows use 7 seconds
            confidence_threshold: Minimum confidence for detections
            use_cache: Generate and reuse RGB/pose features without caching decisions.
            cache_dir: Optional local cache folder; defaults to .runtime/cache/features.
            
        Returns:
            Dictionary with sliding-window detection results
        """
        window_duration = 7
        stride_duration = 1
        print(f'\nAnalyzing video with sliding windows ({window_duration}s windows, {stride_duration}s stride): {video_path}')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        print(f'Video: {duration:.1f}s, {total_frames} frames at {fps:.1f} fps')
        if progress_callback:
            progress_callback(18, 'preprocessing', 'Preparando el muestreo temporal')
        effective_fps = self._resolve_temporal_sampling_fps(fps, window_duration, temporal_fps_mode=temporal_fps_mode, custom_effective_fps=custom_effective_fps)
        self._web_effective_fps = float(effective_fps)
        self._web_window_duration = float(window_duration)
        frames_per_window = max(1, int(round(effective_fps * window_duration)))
        frames_per_stride = max(1, int(round(effective_fps * stride_duration)))
        sample_indices = _sample_frame_indices(total_frames, fps, effective_fps)
        sampled_total_frames = len(sample_indices)
        feature_batch_size = max(1, int(feature_batch_size))
        feature_predict_batch_size = max(1, int(feature_predict_batch_size))
        print(f'Applying temporal standardization: effective_fps={effective_fps:.2f}, sampled_total_frames={sampled_total_frames}, frames_per_window={frames_per_window}, frames_per_stride={frames_per_stride}, mode={temporal_fps_mode}, batch={feature_batch_size}, predict_batch={feature_predict_batch_size}')
        print('Streaming sampled frame preprocessing and feature extraction...')
        sample_index_set = set(sample_indices)
        if not sample_index_set:
            return {'status': 'error', 'error': 'No frames available after temporal standardization', 'detections': []}
        self._begin_auxiliary_sequence(sampled_total_frames)
        feature_provider = feature_provider or getattr(self, 'feature_provider', None)
        parameters = {'mode': 'segments', 'source_fps': float(fps), 'total_frames': total_frames, 'effective_fps': float(effective_fps), 'sample_indices': [int(i) for i in sample_indices], 'feature_batch_size': feature_batch_size, 'feature_predict_batch_size': feature_predict_batch_size}
        try:
            if feature_provider is None:
                all_features, cache_info = cached_features(self, video_path, parameters, len(sample_index_set), lambda: self._extract_sampled_video_features(video_path, sample_index_set, total_frames, feature_batch_size, feature_predict_batch_size, progress_callback), use_cache=use_cache, cache_dir=cache_dir)
            else:
                all_features, cache_info = feature_provider(self, video_path, parameters, use_cache=use_cache, cache_dir=cache_dir)
        except ValueError as error:
            return {'status': 'error', 'error': str(error), 'detections': []}
        if cache_info['hit'] and progress_callback:
            progress_callback(70, 'features', 'Características cargadas de la caché local')
        print(f"Features: {all_features.shape}; cache={cache_info['status']}")
        window_predictions = []
        if len(all_features) <= frames_per_window:
            window_starts = [0]
        else:
            window_starts = list(range(0, len(all_features) - frames_per_window + 1, frames_per_stride))
        num_windows = len(window_starts)
        if progress_callback:
            progress_callback(72, 'inference', f'Analizando {num_windows} ventanas temporales')
        print(f'Processing {num_windows} sliding windows...')
        print(f"{'Window':<8} {'Time Range':<20} {'Predicted':<10} {'Confidence':<12} {'Status':<10}")
        print('-' * 70)
        for window_idx, start_frame in enumerate(window_starts):
            end_frame = min(start_frame + frames_per_window, len(all_features))
            start_time = start_frame / effective_fps if effective_fps > 0 else 0.0
            end_time = end_frame / effective_fps if effective_fps > 0 else duration
            raw_frame_start = int(round(start_time * fps)) if fps > 0 else start_frame
            raw_frame_end = int(round(end_time * fps)) if fps > 0 else end_frame
            window_features = all_features[start_frame:end_frame]
            if len(window_features) < MAX_SEQ_LENGTH:
                padded_features = np.zeros((MAX_SEQ_LENGTH, window_features.shape[1]), dtype=np.float32)
                padded_features[:len(window_features)] = window_features
                mask = np.zeros(MAX_SEQ_LENGTH, dtype=np.float32)
                mask[:len(window_features)] = 1.0
            else:
                padded_features = window_features[:MAX_SEQ_LENGTH]
                mask = np.ones(MAX_SEQ_LENGTH, dtype=np.float32)
            features_batch = np.expand_dims(padded_features, axis=0)
            mask_batch = np.expand_dims(mask, axis=0)
            predictions = self._predict_sequence_window(padded_features, mask, start_frame, end_frame)
            pred_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_class_idx]
            predicted_label = self.label_vocab[pred_class_idx]
            time_range = f'{start_time:.1f}s-{end_time:.1f}s'
            conf_str = f'{confidence:.2%}'
            probabilities = {label: float(prob) for label, prob in zip(self.label_vocab, predictions[0])}
            window_predictions.append({'play_type': predicted_label, 'start_time': float(start_time), 'end_time': float(end_time), 'duration': float(end_time - start_time), 'confidence': float(confidence), 'frame_start': raw_frame_start, 'frame_end': raw_frame_end, 'probabilities': probabilities})
            target_confidence = max((probabilities.get(label, 0.0) for label in self.EVENT_TARGET_LABELS), default=0.0)
            status = 'EVENT EVIDENCE' if target_confidence >= confidence_threshold else 'Below threshold'
            print(f'{window_idx + 1:<8} {time_range:<20} {predicted_label:<10} {conf_str:<12} {status:<10}')
            if progress_callback:
                window_ratio = (window_idx + 1) / max(1, num_windows)
                progress_callback(74 + window_ratio * 21, 'inference', f'Ventana {window_idx + 1} de {num_windows}')
        detections = self._post_process_event_detections(window_predictions, confidence_threshold=confidence_threshold, video_duration=duration, source_fps=fps, class_confidence_thresholds={'line': float(line_confidence_threshold)} if line_confidence_threshold is not None else None)
        review_detections_7s = getattr(self, '_last_review_window_detections', [])
        if progress_callback:
            progress_callback(96, 'postprocessing', 'Agrupando detecciones superpuestas')
        print('-' * 70)
        print(f'Summary: Found {len(detections)} merged event detections above threshold {confidence_threshold}')
        print('\nDetailed Statistics:')
        print(f'  Total windows analyzed: {num_windows}')
        print(f'  Event detections after temporal merge: {len(detections)}')
        print('\nPredictions by play type (all windows):')
        for label in self.label_vocab:
            count = sum((1 for det in window_predictions if det['play_type'] == label))
            if count > 0:
                avg_conf = np.mean([det['confidence'] for det in window_predictions if det['play_type'] == label])
                print(f'  {label}: {count} windows (avg confidence: {avg_conf:.2%})')
        if len(detections) == 0:
            print('\n⚠️  No detections found. Consider:')
            print('   - Lowering confidence_threshold (current: {:.2f})'.format(confidence_threshold))
            print('   - Checking if video contains expected play types')
            print('   - Verifying video preprocessing is correct')
        return {'status': 'success', 'feature_cache': cache_info, 'video_path': video_path, 'preprocessing_filter': self.preprocessing_filter, 'duration': duration, 'window_duration': window_duration, 'stride_duration': stride_duration, 'source_fps': float(fps), 'effective_fps': float(effective_fps), 'source_total_frames': int(total_frames), 'sampled_total_frames': int(len(all_features)), 'frames_per_window': int(frames_per_window), 'frames_per_stride': int(frames_per_stride), 'temporal_fps_mode': str(temporal_fps_mode), 'custom_effective_fps': float(custom_effective_fps) if custom_effective_fps is not None else None, 'feature_batch_size': int(feature_batch_size), 'feature_predict_batch_size': int(feature_predict_batch_size), 'temporal_standardization': 'uniform_global_resampling_before_feature_extraction', 'total_windows': num_windows, 'event_post_processing': {'target_labels': list(self.EVENT_TARGET_LABELS), 'context_radius_seconds': float(self.EVENT_CONTEXT_RADIUS_SECONDS), 'merge_gap_seconds': float(self.EVENT_MERGE_GAP_SECONDS), 'anchor_confidence': float(self.EVENT_ANCHOR_CONFIDENCE), 'neighbor_boost': float(self.EVENT_NEIGHBOR_BOOST), 'require_target_winner': bool(self.EVENT_REQUIRE_TARGET_WINNER), 'merge_by_start_time': bool(self.EVENT_MERGE_BY_START_TIME), 'min_target_confidence': float(getattr(self, 'EVENT_MIN_TARGET_CONFIDENCE', 0.0)), 'threshold_mode': 'per_class_pre_merge' if line_confidence_threshold is not None else 'global', 'per_class_min_target_confidence': {'line': max(float(confidence_threshold), float(getattr(self, 'EVENT_MIN_TARGET_CONFIDENCE', 0.0)), float(line_confidence_threshold)), 'scrum': max(float(confidence_threshold), float(getattr(self, 'EVENT_MIN_TARGET_CONFIDENCE', 0.0)))} if line_confidence_threshold is not None else None, 'min_support_windows': int(getattr(self, 'EVENT_MIN_SUPPORT_WINDOWS', 1)), 'max_duration_seconds': getattr(self, 'EVENT_MAX_DURATION_SECONDS', None)}, 'window_predictions': window_predictions, 'review_detections_7s': review_detections_7s, 'detections': detections}

def main():
    """Test the detector with a sample video."""
    import sys
    if len(sys.argv) < 2:
        print('Usage: python web_adapter.py <video_path> [confidence_threshold]')
        print('\nExample:')
        print('  python web_adapter.py ../../data/raw/sample_video.mp4 0.5')
        return
    video_path = sys.argv[1]
    confidence_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    detector = RugbyPlayDetector()
    results = detector.analyze_video(video_path, confidence_threshold=confidence_threshold)
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    print(f"Status: {results['status']}")
    if results['status'] == 'success':
        print(f"Predicted class: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print(f'\nAll probabilities:')
        for label, prob in results['all_probabilities'].items():
            print(f'  {label}: {prob:.2%}')
        print(f"\nDetections: {len(results['detections'])}")
    else:
        print(f"Error: {results.get('error', 'Unknown error')}")
if __name__ == '__main__':
    main()
