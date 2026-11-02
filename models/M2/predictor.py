"""Web adapter for the frozen RGB-dynamics + YOLO quantile fusion.

The production artifact is a late-fusion model: the historical ResNet50+
BiGRU remains frozen and produces ``p_hist``; this adapter extracts the same
200-frame RGB/YOLO representations used by the experiment, obtains ``p_aux``
from the persisted quantile scaler and logistic head, and returns
``0.4 * p_hist + 0.6 * p_aux`` for every web window.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import cv2
import joblib
import numpy as np
from models.common.spatial import normalize_yolo_source_roi
from models.common.event_verifier import LineEventVerifier
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT_RELATIVE = 'data/joint_multimodal_20260813'
DEPLOYMENT_RELATIVE = MODEL_ROOT_RELATIVE + '/deployment_fusion_quantile_100frames_seed42'
FEATURE_RELATIVE = MODEL_ROOT_RELATIVE + '/rgb_dynamics_seed42'
POSE_MODEL_RELATIVE = 'yolo11n-pose.pt'
MODEL_NAME = 'rgb_yolo_quantile_fusion_all'
HIGHRES_MODEL_NAME = 'rgb_yolo_quantile_highres_roi'
HIGHRES_DEPLOYMENT_RELATIVE = 'data/player_geometry_highres_pose/outputs/deployment_quantile_100frames'
EVENT_VERIFIER_MODEL_NAME = 'rgb_yolo_highres_temporal_event_verifier'
EVENT_VERIFIER_RELATIVE = 'data/line_cached_improvement_suite_20260818/outputs/stable_event_logreg_c0_1_pending_external_match.joblib'
TEMPORAL_DEPLOYMENT_RELATIVE = 'data/player_geometry_highres_pose/outputs/temporal_external_primera_20260815/temporal_models'
HISTORICAL_MODEL_RELATIVE = 'models/M0/checkpoints/best_model/best_model.keras'
PRIMERA_CALIBRATED_MODEL_NAME = 'rgb_yolo_historical_primera_calibrated'
PRIMERA_HISTORICAL_RELATIVE = 'data/historical_primera_grouped_20260818/outputs/historical_primera.keras'
PRIMERA_VERIFIER_RELATIVE = 'data/historical_primera_calibrated_20260818/outputs/line_verifier_historical_primera.joblib'
LABELS = np.asarray(['line', 'other', 'scrum'], dtype=object)
MODELS_DIR = PROJECT_ROOT / 'models' / 'M0'
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))
from models.M0.web_adapter import RugbyPlayDetector

class RGBYoloQuantileFusionDetector(RugbyPlayDetector):
    """Deploy the audited 1,686-feature late-fusion model in the web app."""
    RGB_LENGTH = 200
    YOLO_FRAME_BUDGET = 100
    YOLO_WINDOW_DURATION = 7.0
    EVENT_REQUIRE_TARGET_WINNER = True
    EVENT_MERGE_BY_START_TIME = True
    EVENT_NEIGHBOR_BOOST = 0.0
    EVENT_CONTEXT_RADIUS_SECONDS = 0.0
    EVENT_MERGE_GAP_SECONDS = 1.5
    EVENT_MIN_SUPPORT_WINDOWS = 2
    EVENT_MIN_TARGET_CONFIDENCE = 0.6
    EVENT_MAX_DURATION_SECONDS = 15.0
    MAX_PEOPLE = 30
    POSE_PERSON_CONFIDENCE = 0.25
    POSE_KEYPOINT_CONFIDENCE = 0.2

    def __init__(self, project_root: Path=PROJECT_ROOT, *, deployment_relative: str=DEPLOYMENT_RELATIVE, pose_input_mode: str='upsampled_224', model_name: str=MODEL_NAME, display_name: str | None=None, historical_model_relative: str=HISTORICAL_MODEL_RELATIVE):
        self.project_root = Path(project_root).resolve()
        if pose_input_mode not in {'upsampled_224', 'source_equivalent_roi'}:
            raise ValueError(f'Unsupported YOLO pose input mode: {pose_input_mode}')
        self.pose_input_mode = pose_input_mode
        self.requires_raw_yolo_frames = pose_input_mode == 'source_equivalent_roi'
        self._model_name = str(model_name)
        self._display_name = display_name
        self.deployment_root = self.project_root / deployment_relative
        self.feature_root = self.project_root / FEATURE_RELATIVE
        self.historical_model_path = self.project_root / historical_model_relative
        required = [self.deployment_root / 'quantile_scaler.joblib', self.deployment_root / 'logistic.joblib', self.feature_root / 'dynamic_pca.joblib', self.project_root / POSE_MODEL_RELATIVE, self.historical_model_path]
        missing = [str(path) for path in required if not _resolve_model_artifact(path).exists()]
        if missing:
            raise FileNotFoundError(f'RGB+YOLO web artifacts missing: {missing}')
        super().__init__(str(MODELS_DIR), model_name=str(self.historical_model_path), extractor_name='resnet50')
        self.quantile_scaler = joblib.load(_resolve_model_artifact(self.deployment_root / 'quantile_scaler.joblib'))
        self.auxiliary_model = joblib.load(_resolve_model_artifact(self.deployment_root / 'logistic.joblib'))
        self.dynamic_pca = joblib.load(_resolve_model_artifact(self.feature_root / 'dynamic_pca.joblib'))
        self.label_vocab = LABELS.copy()
        self.historical_weight = 0.4
        self.auxiliary_weight = 0.6
        self._raw_geometry = None
        self._geometry_frames: list[tuple[np.ndarray, ...]] = []
        self._yolo_source_indices = np.empty(0, dtype=np.int32)
        self._yolo_index_set: set[int] = set()
        self._sampled_frame_index = 0
        self._window_branch_predictions: list[dict[str, np.ndarray]] = []
        player_dir = self.project_root / 'data' / 'player_geometry_fusion'
        saved_config = sys.modules.pop('config', None)
        sys.path.insert(0, str(source_directory(player_dir)))
        try:
            from models.M1.pose_cache import detect_frame
            from models.M1.geometry_features import build_geometry_features
            from models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite import make_variant
        finally:
            sys.path.remove(str(source_directory(player_dir)))
            if saved_config is not None:
                sys.modules['config'] = saved_config
        self._detect_frame = detect_frame
        self._build_geometry_features = build_geometry_features
        self._make_variant = make_variant
        try:
            yolo_config_dir = _temporary_model_settings()
            yolo_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault('YOLO_CONFIG_DIR', str(yolo_config_dir))
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError('Instalar ultralytics para usar la fusión RGB + YOLO en la web.') from exc
        from models.inference_cache import load_with_revision
        self.pose_detector, self._feature_cache_pose_revision = load_with_revision(_resolve_model_artifact(str(self.project_root / POSE_MODEL_RELATIVE)), YOLO)
        self.fusion_metadata = {'model_name': self._model_name, 'architecture': 'frozen ResNet50+BiGRU + RGB dynamics + YOLO11n-pose (100 frames/window) E04/E06 + QuantileTransformer + logistic late fusion', 'feature_dim': 1686, 'rgb_dynamics_dim': 704, 'yolo_e04_dim': 707, 'yolo_e06_dim': 275, 'sequence_frames': 200, 'yolo_frames_per_window': self.YOLO_FRAME_BUDGET, 'yolo_temporal_budget': '100 uniform YOLO frames per 7-second window', 'pose_input_mode': self.pose_input_mode, 'historical_weight': self.historical_weight, 'auxiliary_weight': self.auxiliary_weight, 'classifier': 'LogisticRegression(C=0.003, moderate class weights)', 'external_grouped_accuracy': 0.8392944634982852, 'external_grouped_f1_macro': 0.8123015138995129, 'external_grouped_delta_accuracy': 0.0906418422341989, 'reference_200_accuracy': 0.8407643312101911, 'reference_200_f1_macro': 0.8140962271208876, 'training_note': 'The 100-frame deployment head was fit on the natural feature cache; external grouped metrics are reported by the budget audit.'}

    @property
    def display_name(self) -> str:
        return self._display_name or 'Nuevo modelo RGB + YOLO (fusión cuantílica)'

    def _begin_auxiliary_sequence(self, sampled_total_frames: int) -> None:
        self._geometry_frames = []
        self._raw_geometry = None
        self._sampled_frame_index = 0
        self._window_branch_predictions = []
        total = max(0, int(sampled_total_frames))
        effective_fps = getattr(self, '_web_effective_fps', None)
        if effective_fps is not None and float(effective_fps) > 0:
            yolo_fps = self.YOLO_FRAME_BUDGET / self.YOLO_WINDOW_DURATION
            step = max(1.0, float(effective_fps) / yolo_fps)
            indices = np.rint(np.arange(0.0, max(1, total), step)).astype(np.int32)
            indices = np.unique(np.clip(indices, 0, max(0, total - 1)))
        else:
            target = min(total, self.YOLO_FRAME_BUDGET)
            indices = self._build_temporal_sample_indices(total, target)
        self._yolo_source_indices = indices
        self._yolo_index_set = set((int(index) for index in indices.tolist()))

    def _consume_yolo_frame(self, frame_bgr: np.ndarray, *, source_resolution: bool) -> None:
        source_index = self._sampled_frame_index
        self._sampled_frame_index += 1
        if source_index not in self._yolo_index_set:
            return
        if source_resolution:
            pose_frame = normalize_yolo_source_roi(frame_bgr, yolo_size=640)
            if pose_frame is None:
                raise ValueError('Source frame could not be mapped to the high-resolution YOLO ROI')
        else:
            pose_frame = cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        self._geometry_frames.append(self._detect_frame(self.pose_detector, pose_frame, image_size=640, person_threshold=self.POSE_PERSON_CONFIDENCE, max_people=self.MAX_PEOPLE))

    def _on_raw_sampled_frame(self, frame_bgr: np.ndarray) -> None:
        if self.pose_input_mode == 'source_equivalent_roi':
            self._consume_yolo_frame(frame_bgr, source_resolution=True)

    def _on_sampled_frame(self, frame_bgr: np.ndarray) -> None:
        if self.pose_input_mode == 'upsampled_224':
            self._consume_yolo_frame(frame_bgr, source_resolution=False)

    def _finish_auxiliary_sequence(self) -> None:
        if not self._geometry_frames:
            self._raw_geometry = None
            self._yolo_source_indices = np.empty(0, dtype=np.int32)
            return
        boxes = np.asarray([item[0] for item in self._geometry_frames], dtype=np.float32)
        person_conf = np.asarray([item[1] for item in self._geometry_frames], dtype=np.float32)
        keypoints = np.asarray([item[2] for item in self._geometry_frames], dtype=np.float32)
        keypoint_conf = np.asarray([item[3] for item in self._geometry_frames], dtype=np.float32)
        valid = np.asarray([item[4] for item in self._geometry_frames], dtype=bool)
        positions, poses, frame_mask = self._build_geometry_features(boxes, person_conf, keypoints, keypoint_conf, valid, person_threshold=self.POSE_PERSON_CONFIDENCE, keypoint_threshold=self.POSE_KEYPOINT_CONFIDENCE)
        self._raw_geometry = {'boxes': boxes, 'person_conf': person_conf, 'keypoints': keypoints, 'keypoint_conf': keypoint_conf, 'valid': valid, 'position_features': positions, 'pose_features': poses, 'frame_mask': frame_mask}
        if len(self._yolo_source_indices) != len(self._geometry_frames):
            self._yolo_source_indices = self._yolo_source_indices[:len(self._geometry_frames)]

    @staticmethod
    def _align(sequence: np.ndarray, length: int) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(f'Expected a 2-D temporal sequence, got {sequence.shape}')
        if sequence.shape[0] == length:
            return sequence
        if sequence.shape[0] == 0:
            return np.zeros((length, sequence.shape[1]), dtype=np.float32)
        old = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
        new = np.linspace(0.0, 1.0, length, dtype=np.float32)
        return np.stack([np.interp(new, old, sequence[:, index]) for index in range(sequence.shape[1])], axis=1).astype(np.float32)

    def _rgb_dynamic_vector(self, padded_features: np.ndarray, length: int) -> np.ndarray:
        rgb = np.asarray(padded_features[:length], dtype=np.float32)
        if rgb.shape[1] != 2048:
            raise ValueError(f'The production fusion expects 2,048 ResNet features, got {rgb.shape[1]}')
        rgb = self._align(rgb, self.RGB_LENGTH)
        segments = rgb.reshape(10, 20, 2048)
        segment_mean = segments.mean(axis=1)
        segment_delta = np.abs(np.diff(segments, axis=1)).mean(axis=1)
        global_mean = rgb.mean(axis=0)
        global_std = rgb.std(axis=0)
        pca = self.dynamic_pca
        return np.concatenate([pca['segment'].transform(segment_mean).reshape(-1), pca['delta'].transform(segment_delta).reshape(-1), pca['mean'].transform(global_mean.reshape(1, -1)).reshape(-1), pca['std'].transform(global_std.reshape(1, -1)).reshape(-1)]).astype(np.float32)

    def _yolo_vector(self, start_frame: int, end_frame: int) -> np.ndarray:
        if self._raw_geometry is None:
            return np.zeros(982, dtype=np.float32)
        source = np.asarray(self._yolo_source_indices, dtype=np.int32)
        selected = (source >= int(start_frame)) & (source < int(end_frame))
        if not np.any(selected):
            return np.zeros(982, dtype=np.float32)
        data = {key: value[selected] for key, value in self._raw_geometry.items()}
        e04 = self._make_variant(data, 'E04_tracking_smoothing')
        e06 = self._make_variant(data, 'E06_rugby_descriptors')
        vector = np.concatenate([e04, e06]).astype(np.float32)
        if vector.shape != (982,):
            raise ValueError(f'Unexpected YOLO representation shape: {vector.shape}')
        return vector

    @staticmethod
    def _normalize(probabilities: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-07, None)
        total = values.sum(axis=-1, keepdims=True)
        total[total <= 0.0] = 1.0
        return values / total

    def _predict_sequence_window(self, padded_features: np.ndarray, mask: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
        historical = self.sequence_model.predict([np.expand_dims(padded_features, axis=0), np.expand_dims(mask, axis=0)], verbose=0)[0]
        historical = self._normalize(historical)
        length = max(0, min(int(end_frame - start_frame), int(np.sum(mask))))
        dynamic = self._rgb_dynamic_vector(padded_features, length)
        yolo = self._yolo_vector(start_frame, end_frame)
        auxiliary_features = np.concatenate([dynamic, yolo]).reshape(1, -1)
        if auxiliary_features.shape[1] != 1686:
            raise ValueError(f'Unexpected RGB+YOLO feature shape: {auxiliary_features.shape}')
        auxiliary = self.auxiliary_model.predict_proba(self.quantile_scaler.transform(auxiliary_features))[0]
        auxiliary = self._normalize(auxiliary)
        fused = self.historical_weight * historical + self.auxiliary_weight * auxiliary
        fused = self._normalize(fused)
        self._window_branch_predictions.append({'historical': historical.astype(np.float32), 'static': auxiliary.astype(np.float32), 'official': fused.astype(np.float32)})
        return fused[None, :].astype(np.float32)

    def predict_live_window(self, window_frames: list[np.ndarray], sequence_len: int, sampled_frame_count: int | None=None, raw_window_frames: list[np.ndarray] | None=None) -> np.ndarray:
        """Run the same fusion for the live-preview route."""
        target_count = sampled_frame_count if sampled_frame_count and sampled_frame_count > 0 else sequence_len
        target_count = min(sequence_len, max(1, int(target_count)))
        if len(window_frames) > target_count:
            indices = np.linspace(0, len(window_frames) - 1, num=target_count, dtype=np.int32)
            sampled = [window_frames[index] for index in indices]
        else:
            sampled = window_frames
        frame_array = np.asarray(sampled, dtype=np.uint8)
        features = self._predict_feature_batch(frame_array.tolist(), min(32, max(1, len(sampled))))
        padded = np.zeros((sequence_len, features.shape[1]), dtype=np.float32)
        mask = np.zeros(sequence_len, dtype=np.float32)
        length = min(len(features), sequence_len)
        padded[:length] = features[:length]
        mask[:length] = 1.0
        self._web_effective_fps = None
        self._begin_auxiliary_sequence(length)
        if self.pose_input_mode == 'source_equivalent_roi':
            if raw_window_frames is None:
                raw_sampled = [frame_rgb[:, :, ::-1] for frame_rgb in sampled]
            elif len(raw_window_frames) > target_count:
                raw_indices = np.linspace(0, len(raw_window_frames) - 1, num=target_count, dtype=np.int32)
                raw_sampled = [raw_window_frames[index] for index in raw_indices]
            else:
                raw_sampled = raw_window_frames
            for frame_bgr in raw_sampled[:length]:
                self._on_raw_sampled_frame(frame_bgr)
        else:
            for frame_rgb in sampled[:length]:
                self._on_sampled_frame(frame_rgb[:, :, ::-1])
        self._finish_auxiliary_sequence()
        return self._predict_sequence_window(padded, mask, 0, length)

class RGBYoloHighresQuantileFusionDetector(RGBYoloQuantileFusionDetector):
    """Candidate using source-resolution pixels for the equivalent YOLO ROI."""

    def __init__(self, project_root: Path=PROJECT_ROOT, *, historical_model_relative: str=HISTORICAL_MODEL_RELATIVE, model_name: str=HIGHRES_MODEL_NAME, display_name: str='RGB + YOLO con ROI de alta resolución'):
        super().__init__(project_root, deployment_relative=HIGHRES_DEPLOYMENT_RELATIVE, pose_input_mode='source_equivalent_roi', model_name=model_name, display_name=display_name, historical_model_relative=historical_model_relative)
        summary_path = self.deployment_root / 'summary.json'
        try:
            deployment_summary = json.loads(_resolve_model_artifact(summary_path).read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            deployment_summary = {}
        metrics = deployment_summary.get('external_metrics') or {}
        delta = deployment_summary.get('external_delta_vs_lowres') or {}
        if metrics:
            historical_accuracy = 0.7486526212640863
            self.fusion_metadata.update({'external_grouped_accuracy': metrics.get('accuracy'), 'external_grouped_balanced_accuracy': metrics.get('balanced_accuracy'), 'external_grouped_f1_macro': metrics.get('f1_macro'), 'external_grouped_delta_accuracy': float(metrics['accuracy']) - historical_accuracy if metrics.get('accuracy') is not None else None, 'external_delta_vs_lowres_accuracy': delta.get('accuracy'), 'external_delta_vs_lowres_balanced_accuracy': delta.get('balanced_accuracy'), 'external_delta_vs_lowres_f1_macro': delta.get('f1_macro'), 'promoted': bool(deployment_summary.get('promoted')), 'training_note': 'High-resolution ROI deployment fitted on all 4,082 clips after a frozen five-fold external audit.'})
            self.fusion_metadata.pop('reference_200_accuracy', None)
            self.fusion_metadata.pop('reference_200_f1_macro', None)

class RGBYoloTemporalEventVerifierDetector(RGBYoloHighresQuantileFusionDetector):
    """Best web-ready candidate: high-res fusion plus a complete-event line veto.

    The temporal TCN+BiGRU branch is not mixed into the public three-class
    prediction.  It is additional evidence for the binary line verifier after
    candidate blocks have been grouped, matching the experiment that preserved
    every annotated line while reducing false positives.
    """
    TEMPORAL_SEEDS = (42, 17, 2026)
    LINE_VERIFIER_THRESHOLD = 0.11

    def __init__(self, project_root: Path=PROJECT_ROOT, *, historical_model_relative: str=HISTORICAL_MODEL_RELATIVE, model_name: str=EVENT_VERIFIER_MODEL_NAME, display_name: str='RGB + YOLO temporal + verificador de line', verifier_relative: str=EVENT_VERIFIER_RELATIVE, line_historical_weight: float=0.4, line_static_weight: float=0.6, line_verifier_threshold: float | None=None, validation_status: str='pending_different_match_validation'):
        super().__init__(project_root, historical_model_relative=historical_model_relative, model_name=model_name, display_name=display_name)
        self.line_verifier_threshold = float(self.LINE_VERIFIER_THRESHOLD if line_verifier_threshold is None else line_verifier_threshold)
        self.event_validation_status = str(validation_status)
        self.line_verifier_path = self.project_root / verifier_relative
        self.temporal_root = self.project_root / TEMPORAL_DEPLOYMENT_RELATIVE
        required = [self.line_verifier_path] + [self.temporal_root / f'seed_{seed}.pt' for seed in self.TEMPORAL_SEEDS]
        missing = [str(path) for path in required if not _resolve_model_artifact(path).exists()]
        if missing:
            raise FileNotFoundError(f'Temporal event-verifier artifacts missing: {missing}')
        self.line_event_verifier = LineEventVerifier(self.line_verifier_path, threshold=self.line_verifier_threshold, historical_weight=line_historical_weight, static_weight=line_static_weight)
        try:
            import torch
            from models.M2.grid_search_cached_descriptors import primitive_base
            from models.M3.run_temporal_architecture_suite import build_model
            from models.M3.run_temporal_attention_experiment import quality_score, smooth_three, usable_mask
        except ImportError as exc:
            raise ImportError('La variante temporal necesita PyTorch y los módulos de geometría YOLO.') from exc
        self._torch = torch
        self._primitive_base = primitive_base
        self._smooth_three = smooth_three
        self._quality_score = quality_score
        self._usable_mask = usable_mask
        self._temporal_models = []
        for seed in self.TEMPORAL_SEEDS:
            checkpoint = torch.load(_resolve_model_artifact(self.temporal_root / f'seed_{seed}.pt'), map_location='cpu', weights_only=False)
            model = build_model(str(checkpoint['architecture']))
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            self._temporal_models.append({'model': model, 'mean': np.asarray(checkpoint['mean'], dtype=np.float32), 'std': np.asarray(checkpoint['std'], dtype=np.float32), 'temperature': float(checkpoint['temperature']), 'seed': int(checkpoint['seed'])})
        self.fusion_metadata.update({'model_name': self._model_name, 'architecture': 'high-resolution 40/60 historical+RGB/YOLO fusion + three-seed TCN+BiGRU pose evidence + event logistic line verifier', 'line_verifier': 'StandardScaler + balanced LogisticRegression(C=0.1)', 'line_verifier_threshold': self.line_verifier_threshold, 'line_historical_weight': self.line_event_verifier.historical_weight, 'line_static_weight': self.line_event_verifier.static_weight, 'line_verifier_feature_dim': 208, 'line_verifier_grid_seconds': 7.0, 'temporal_architecture': 'TCN+BiGRU with quality-prior attention', 'temporal_seeds': list(self.TEMPORAL_SEEDS), 'temporal_inference_stride_seconds': 7.0, 'development_event_recall': 1.0, 'development_event_precision': 0.5, 'development_false_positives_per_hour': 23.1, 'development_status': 'pending different-match validation'})

    @staticmethod
    def _resample_raw_window(raw: dict[str, np.ndarray], length: int=100) -> dict[str, np.ndarray]:
        available = len(raw['frame_mask'])
        if available <= 0:
            raise ValueError('Temporal YOLO window has no observations')
        indices = np.rint(np.linspace(0, available - 1, num=length)).astype(np.int32)
        return {key: np.asarray(value)[indices] for key, value in raw.items()}

    def _temporal_probability(self, start_frame: int, end_frame: int) -> np.ndarray:
        if self._raw_geometry is None:
            return np.full(3, 1.0 / 3.0, dtype=np.float32)
        source = np.asarray(self._yolo_source_indices, dtype=np.int32)
        selected = (source >= int(start_frame)) & (source < int(end_frame))
        if not np.any(selected):
            return np.full(3, 1.0 / 3.0, dtype=np.float32)
        raw = self._resample_raw_window({key: np.asarray(value)[selected] for key, value in self._raw_geometry.items()})
        primitive = self._primitive_base(raw)
        frame_mask = primitive['mask'].astype(bool)
        quality = primitive['quality'].astype(np.float32)
        current = self._smooth_three(primitive['current'][None], frame_mask[None])[0]
        rugby = self._smooth_three(primitive['rugby'][None], frame_mask[None])[0]
        features = np.nan_to_num(np.concatenate([current, quality, rugby], axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        score = self._quality_score(quality[None])[0]
        usable = self._usable_mask(frame_mask[None], score[None])[0]
        probabilities = []
        with self._torch.no_grad():
            for member in self._temporal_models:
                normalized = np.clip((features - member['mean']) / member['std'], -8.0, 8.0).astype(np.float32)
                normalized[~usable] = 0.0
                logits = member['model'](self._torch.from_numpy(normalized[None]), self._torch.from_numpy(usable[None]), self._torch.from_numpy(score[None]))
                probability = self._torch.softmax(logits / member['temperature'], dim=1)[0].cpu().numpy()
                probabilities.append(probability)
        return self._normalize(np.mean(probabilities, axis=0)).astype(np.float32)

    def _predict_sequence_window(self, padded_features: np.ndarray, mask: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
        official = super()._predict_sequence_window(padded_features, mask, start_frame, end_frame)
        effective_fps = float(getattr(self, '_web_effective_fps', 0.0) or 0.0)
        grid_frames = effective_fps * self.line_event_verifier.GRID_SECONDS
        on_verifier_grid = grid_frames <= 0.0 or abs(float(start_frame) / grid_frames - round(float(start_frame) / grid_frames)) <= 0.075
        self._window_branch_predictions[-1]['temporal'] = self._temporal_probability(start_frame, end_frame) if on_verifier_grid else np.full(3, 1.0 / 3.0, dtype=np.float32)
        return official

    def _post_process_event_detections(self, window_predictions: List[Dict], confidence_threshold: float, video_duration: float, source_fps: float, class_confidence_thresholds: Dict[str, float] | None=None) -> List[Dict]:
        if len(window_predictions) != len(self._window_branch_predictions):
            raise RuntimeError(f'Window/branch alignment failed for the temporal event verifier: {len(window_predictions)} != {len(self._window_branch_predictions)}')
        for item, branches in zip(window_predictions, self._window_branch_predictions):
            item['branch_probabilities'] = {name: {label: float(value) for label, value in zip(LABELS, np.asarray(probability).reshape(-1))} for name, probability in branches.items()}
        base_events = super()._post_process_event_detections(window_predictions, confidence_threshold, video_duration, source_fps, class_confidence_thresholds=class_confidence_thresholds)
        base_review = list(getattr(self, '_last_review_window_detections', []))
        line_events, line_review = self.line_event_verifier.verify(window_predictions, video_duration=video_duration, source_fps=source_fps)
        self._last_review_window_detections = sorted([item for item in base_review if item.get('play_type') != 'line'] + [dict(item) for item in line_review], key=lambda item: (float(item['start_time']), str(item.get('play_type', ''))))
        result = [event for event in base_events if event.get('play_type') != 'line']
        result.extend(line_events)
        result.sort(key=lambda item: float(item['start_time']))
        return result

    def _annotate_event_policy(self, result: Dict) -> Dict:
        if result.get('status') == 'success':
            result.setdefault('event_post_processing', {}).update({'line_mode': 'event_logistic_verifier', 'line_verifier_threshold': self.line_verifier_threshold, 'line_verifier_grid_seconds': 7.0, 'line_verifier_feature_dim': 208, 'line_verifier_artifact': str(self.line_verifier_path.relative_to(self.project_root)).replace('\\', '/'), 'line_historical_weight': self.line_event_verifier.historical_weight, 'line_static_weight': self.line_event_verifier.static_weight, 'line_ui_threshold_used': False, 'validation_status': self.event_validation_status})
        return result

    def analyze_video_segments(self, *args, **kwargs) -> Dict:
        return self._annotate_event_policy(super().analyze_video_segments(*args, **kwargs))

    def analyze_video(self, *args, **kwargs) -> Dict:
        return self._annotate_event_policy(super().analyze_video(*args, **kwargs))

class RGBYoloHistoricalPrimeraCalibratedDetector(RGBYoloTemporalEventVerifierDetector):
    """Fine-tuned historical branch plus the externally held-out line calibration."""
    LINE_HISTORICAL_WEIGHT = 0.55
    LINE_STATIC_WEIGHT = 0.45
    LINE_VERIFIER_THRESHOLD = 0.4101333272622148

    def __init__(self, project_root: Path=PROJECT_ROOT):
        super().__init__(project_root, historical_model_relative=PRIMERA_HISTORICAL_RELATIVE, model_name=PRIMERA_CALIBRATED_MODEL_NAME, display_name='Mejor modelo · Histórico Primera + RGB/YOLO temporal calibrado', verifier_relative=PRIMERA_VERIFIER_RELATIVE, line_historical_weight=self.LINE_HISTORICAL_WEIGHT, line_static_weight=self.LINE_STATIC_WEIGHT, line_verifier_threshold=self.LINE_VERIFIER_THRESHOLD, validation_status='untouched_same_match_40_60_holdout')
        self.fusion_metadata.update({'model_name': PRIMERA_CALIBRATED_MODEL_NAME, 'architecture': 'historical Primera fine-tune + high-resolution RGB/YOLO static + three-seed TCN+BiGRU pose evidence + calibrated event line verifier', 'historical_model': PRIMERA_HISTORICAL_RELATIVE, 'line_verifier': 'StandardScaler + balanced LogisticRegression(C=1.0)', 'line_verifier_threshold': self.LINE_VERIFIER_THRESHOLD, 'line_historical_weight': self.LINE_HISTORICAL_WEIGHT, 'line_static_weight': self.LINE_STATIC_WEIGHT, 'scrum_historical_weight': 0.4, 'scrum_static_weight': 0.6, 'holdout_combined_precision': 0.631578947368421, 'holdout_combined_recall': 0.9230769230769231, 'holdout_combined_f1': 0.75, 'holdout_false_positives': 7, 'validation_status': 'untouched_same_match_40_60_holdout'})
