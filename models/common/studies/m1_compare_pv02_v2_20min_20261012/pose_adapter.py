"""Remove M1's appearance and dual branches, retain its trained YOLO forests."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import runtime as rt
from runtime import HERE, ROOT
import json, sys, time
from collections import defaultdict
import numpy as np, joblib, cv2
sys.path.insert(0, str(ROOT))
from src.web.keras_inference_engine import PlayerGeometryFusionDetector, M1H2D2CrossFoldDetector
from models.M0.web_adapter import RugbyPlayDetector

class PoseOnlyDetector(RugbyPlayDetector):
    POSE_FRAMES_PER_WINDOW = 100
    MAX_PEOPLE = 30
    POSE_PERSON_CONFIDENCE = 0.25
    POSE_KEYPOINT_CONFIDENCE = 0.2
    EVENT_NEIGHBOR_BOOST = 0.0

    def __init__(self, manifest, thresholds):
        self.manifest = manifest
        self.thresholds = thresholds
        self.preprocessing_filter = 'none'
        self.extractor_name = 'none_YOLO_only'
        self.label_vocab = np.array(['line', 'other', 'scrum'], dtype=object)
        self.stats = defaultdict(float)
        self.counts = defaultdict(int)
        self.fold_models = [{'fold': f['fold'], 'geometry': joblib.load(_resolve_model_artifact(ROOT / f['geometry_rf']))} for f in manifest['folds']]
        for f in self.fold_models:
            assert f['geometry'].n_features_in_ == 549
        previous = sys.modules.pop('config', None)
        geometry_root = ROOT / 'data/player_geometry_fusion'
        sys.path.insert(0, str(source_directory(geometry_root)))
        try:
            from models.M1.pose_cache import detect_frame
            from models.M1.geometry_features import build_geometry_features
        finally:
            sys.path.remove(str(source_directory(geometry_root)))
            if previous is not None:
                sys.modules['config'] = previous
        self._detect_frame = detect_frame
        self._build_geometry_features = build_geometry_features
        from ultralytics import YOLO
        self.pose_detector = YOLO(_resolve_model_artifact(str(ROOT / manifest['pose_model'])))
        self._geometry_frames = []
        self._geometry_positions = np.zeros((0, 31), np.float32)
        self._geometry_poses = np.zeros((0, 30), np.float32)
        self._geometry_mask = np.zeros(0, bool)
        self.fusion_metadata = {'source_configuration': 'M1-H2-D2', 'variant': 'YOLO_geometry_RF_only', 'uses_RGB_appearance': False, 'ResNet_loaded': False, 'dual_BiGRU_loaded': False, 'fold_count': len(self.fold_models), 'geometry_features': 549, 'pose_frames_per_window': 100, 'ensemble_policy': 'equal_soft_vote_geometry_probabilities', 'thresholds': thresholds, 'threshold_source': 'PV02_geometry_development_only_V3', 'validation_scope': 'quality on held-out per-fold geometry branch; new video analysis with only the PV02 checkpoint'}

    @property
    def display_name(self):
        return 'M1-H2-D2 · sólo YOLO + geometría'
    _begin_auxiliary_sequence = PlayerGeometryFusionDetector._begin_auxiliary_sequence
    _finish_auxiliary_sequence = PlayerGeometryFusionDetector._finish_auxiliary_sequence
    _ordered_forest_probability = staticmethod(M1H2D2CrossFoldDetector._ordered_forest_probability)

    def _on_sampled_frame(self, frame_bgr):
        t = time.perf_counter()
        PlayerGeometryFusionDetector._on_sampled_frame(self, frame_bgr)
        self.stats['YOLO_seconds'] += time.perf_counter() - t
        self.counts['YOLO_calls'] += 1

    def _predict_feature_batch(self, frames, initial_batch_size):
        self.counts['placeholder_rows'] += len(frames)
        return np.zeros((len(frames), 1), np.float32)

    def geometry_vector(self, start_frame, end_frame):
        positions = self._geometry_positions[start_frame:end_frame]
        poses = self._geometry_poses[start_frame:end_frame]
        mask = self._geometry_mask[start_frame:end_frame].astype(bool)
        if len(positions) > self.POSE_FRAMES_PER_WINDOW:
            indices = np.linspace(0, len(positions) - 1, self.POSE_FRAMES_PER_WINDOW, dtype=np.int32)
            positions = positions[indices]
            poses = poses[indices]
            mask = mask[indices]
        values = np.concatenate([positions, poses], axis=1)
        valid = values[mask] if np.any(mask) else values
        if len(valid) == 0:
            valid = np.zeros((1, 61), np.float32)
        deltas = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, 61), np.float32)
        return np.concatenate([valid.mean(0), valid.std(0), valid.min(0), valid.max(0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), deltas.mean(0), deltas.std(0), np.abs(deltas).mean(0)]).astype(np.float32).reshape(1, -1)

    def _predict_sequence_window(self, features, mask, start_frame, end_frame):
        t = time.perf_counter()
        vector = self.geometry_vector(start_frame, end_frame)
        self.stats['geometry_statistics_seconds'] += time.perf_counter() - t
        t = time.perf_counter()
        probabilities = [self._ordered_forest_probability(m['geometry'], vector) for m in self.fold_models]
        self.stats['forest_seconds'] += time.perf_counter() - t
        self.counts['forest_calls'] += len(self.fold_models)
        self.counts['window_calls'] += 1
        p = np.clip(np.mean(probabilities, axis=0), 1e-07, None)
        p = p / p.sum()
        return p[None].astype(np.float32)

    def _post_process_event_detections(self, window_predictions, confidence_threshold, video_duration, source_fps, class_confidence_thresholds=None):
        original = self.EVENT_TARGET_LABELS
        events = []
        reviews = []
        try:
            for label in original:
                self.EVENT_TARGET_LABELS = (label,)
                events += RugbyPlayDetector._post_process_event_detections(self, window_predictions, self.thresholds[label], video_duration, source_fps)
                reviews += self._last_review_window_detections
        finally:
            self.EVENT_TARGET_LABELS = original
        self._last_review_window_detections = sorted(reviews, key=lambda w: (w['start_time'], w['play_type']))
        return sorted(events, key=lambda w: (w['start_time'], w['play_type']))

def validate_geometry_extraction(detector):
    """Check the isolated statistic vector against original M1 on synthetic pose."""
    rng = np.random.default_rng(42)
    records = []

    class Output:

        def __getitem__(self, index):
            return self

        def numpy(self):
            return np.array([0.2, 0.5, 0.3])

    class Dual:

        def __call__(self, *args, **kwargs):
            return Output()

    class CaptureForest:
        classes_ = np.arange(3)

        def predict_proba(self, vector):
            self.vector = vector.copy()
            return np.array([[0.2, 0.5, 0.3]])
    for length in [0, 1, 7, 100, 175, 200]:
        fake = type('OriginalM1Probe', (), {})()
        fake.POSE_FRAMES_PER_WINDOW = 100
        fake.RGB_LENGTH = 200
        fake._geometry_positions = rng.normal(size=(length, 31)).astype(np.float32)
        fake._geometry_poses = rng.normal(size=(length, 30)).astype(np.float32)
        fake._geometry_mask = rng.random(length) > 0.2
        fake._align = PlayerGeometryFusionDetector._align
        fake._normalize = PlayerGeometryFusionDetector._normalize
        fake._ordered_forest_probability = M1H2D2CrossFoldDetector._ordered_forest_probability
        forest = CaptureForest()
        fake.fold_models = [{'dual': Dual(), 'geometry': forest, 'geometry_alpha': 0.5}]
        M1H2D2CrossFoldDetector._predict_sequence_window(fake, np.zeros((200, 2048), np.float32), np.ones(200, bool), 0, length)
        current = PoseOnlyDetector.geometry_vector(fake, 0, length)
        assert np.array_equal(current, forest.vector), length
        records.append({'length': length, 'exactly_equal': True})
    return {'all_pass': True, 'synthetic_geometry_vectors_exact': records, 'no_real_RGB_model_loaded_or_evaluated': True}
