"""M1 RGB, pose and geometry inference implementations."""
from pathlib import Path
from typing import Dict
import contextlib
import io
import json
import os
import sys
import cv2
import joblib
import keras
import numpy as np
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, temporary_settings as _temporary_model_settings
from models.M0.web_adapter import RugbyPlayDetector
ROOT=Path(__file__).resolve().parents[2]
player_geometry_path=ROOT/'data/player_geometry_fusion'
PLAYER_GEOMETRY_FUSION_EXPORT_RELATIVE='data/player_geometry_fusion/results/web_export'
PLAYER_GEOMETRY_FUSION_POSE_MODEL='yolo11n-pose.pt'
M1_H2_D2_WEB_MODEL_NAME='m1_h2_d2_cross_fold_ensemble'
M1_H2_D2_WEB_EXPORT_RELATIVE='data/final_model_comparison_20260827/outputs/web_export_m1_h2_d2'

class PlayerGeometryFusionDetector(RugbyPlayDetector):
    """Deployable RGB + pose + geometry ensemble used by the thesis result."""
    RGB_LENGTH = 200
    GEOMETRY_FRAMES = 32
    MAX_PEOPLE = 30
    POSE_PERSON_CONFIDENCE = 0.25
    POSE_KEYPOINT_CONFIDENCE = 0.2

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.export_root = self.project_root / PLAYER_GEOMETRY_FUSION_EXPORT_RELATIVE
        self.manifest_path = self.export_root / 'manifest.json'
        if not _resolve_model_artifact(self.manifest_path).exists():
            raise FileNotFoundError(f'Geometry-fusion manifest not found: {self.manifest_path}')
        self.manifest = json.loads(_resolve_model_artifact(self.manifest_path).read_text(encoding='utf-8'))
        self.pose_model_path = self.project_root / PLAYER_GEOMETRY_FUSION_POSE_MODEL
        if not _resolve_model_artifact(self.pose_model_path).exists():
            raise FileNotFoundError(f'Pose checkpoint not found: {self.pose_model_path}')
        model_dir = self.project_root / 'models' / 'M0'
        super().__init__(str(model_dir), model_name='best_model/best_model.keras', extractor_name='resnet50')
        saved_config = sys.modules.pop('config', None)
        sys.path.insert(0, str(source_directory(player_geometry_path)))
        try:
            from models.M1.same_rnn import branch_model
            from models.M1.train import build_model
            from models.M1.pose_cache import detect_frame
            from models.M1.geometry_features import build_geometry_features
        finally:
            sys.path.remove(str(source_directory(player_geometry_path)))
            if saved_config is not None:
                sys.modules['config'] = saved_config
        self._detect_frame = detect_frame
        self._build_geometry_features = build_geometry_features
        self.seed_models = []
        for run in self.manifest['runs']:
            seed = int(run['seed'])
            rgb_model, _ = build_model('m0', 31, 30)
            rgb_checkpoint = self.project_root / 'data' / 'player_geometry_fusion' / 'results' / 'm0' / f'seed_{seed}' / 'best.weights.h5'
            if not _resolve_model_artifact(rgb_checkpoint).exists():
                raise FileNotFoundError(f'RGB checkpoint not found: {rgb_checkpoint}')
            rgb_model.load_weights(_resolve_model_artifact(rgb_checkpoint))
            pose_model = branch_model('pose_concat', 31)
            pose_checkpoint = self.project_root / 'data' / 'player_geometry_fusion' / 'results' / 'same_rnn' / 'pose_concat' / f'seed_{seed}' / 'best.weights.h5'
            if not _resolve_model_artifact(pose_checkpoint).exists():
                raise FileNotFoundError(f'Pose-RNN checkpoint not found: {pose_checkpoint}')
            pose_model.load_weights(_resolve_model_artifact(pose_checkpoint))
            rf_path = self.export_root / run['rf_path']
            if not _resolve_model_artifact(rf_path).exists():
                raise FileNotFoundError(f'Geometry RF not found: {rf_path}')
            self.seed_models.append({'seed': seed, 'rgb': rgb_model, 'same_rnn': pose_model, 'geometry': joblib.load(_resolve_model_artifact(rf_path)), 'geometry_alpha': float(run['geometry_alpha']), 'same_rnn_alpha': float(run['same_rnn_alpha'])})
        try:
            yolo_config_dir = _temporary_model_settings()
            yolo_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault('YOLO_CONFIG_DIR', str(yolo_config_dir))
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError('El modelo de pose web requiere ultralytics. Instalar las dependencias de data/player_geometry_fusion.') from exc
        from models.inference_cache import load_with_revision
        self.pose_detector, self._feature_cache_pose_revision = load_with_revision(_resolve_model_artifact(str(self.pose_model_path)), YOLO)
        self.label_vocab = np.asarray(['line', 'other', 'scrum'], dtype=object)
        self._geometry_frames = []
        self._geometry_positions = np.zeros((0, 31), dtype=np.float32)
        self._geometry_poses = np.zeros((0, 30), dtype=np.float32)
        self._geometry_mask = np.zeros((0,), dtype=bool)

    @property
    def display_name(self) -> str:
        return '⭐ RGB + Pose + Geometría (ensemble calibrado)'

    def _begin_auxiliary_sequence(self, sampled_total_frames: int) -> None:
        self._geometry_frames = []

    def _on_sampled_frame(self, frame_bgr: np.ndarray) -> None:
        pose_frame = cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        self._geometry_frames.append(self._detect_frame(self.pose_detector, pose_frame, image_size=640, person_threshold=self.POSE_PERSON_CONFIDENCE, max_people=self.MAX_PEOPLE))

    def _finish_auxiliary_sequence(self) -> None:
        if not self._geometry_frames:
            self._geometry_positions = np.zeros((0, 31), dtype=np.float32)
            self._geometry_poses = np.zeros((0, 30), dtype=np.float32)
            self._geometry_mask = np.zeros((0,), dtype=bool)
            return
        boxes = np.asarray([item[0] for item in self._geometry_frames], dtype=np.float32)
        person_conf = np.asarray([item[1] for item in self._geometry_frames], dtype=np.float32)
        keypoints = np.asarray([item[2] for item in self._geometry_frames], dtype=np.float32)
        keypoint_conf = np.asarray([item[3] for item in self._geometry_frames], dtype=np.float32)
        valid = np.asarray([item[4] for item in self._geometry_frames], dtype=bool)
        self._geometry_positions, self._geometry_poses, self._geometry_mask = self._build_geometry_features(boxes, person_conf, keypoints, keypoint_conf, valid, person_threshold=self.POSE_PERSON_CONFIDENCE, keypoint_threshold=self.POSE_KEYPOINT_CONFIDENCE)

    @staticmethod
    def _align(sequence: np.ndarray, length: int) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.shape[0] == length:
            return sequence
        if sequence.shape[0] == 0:
            return np.zeros((length, sequence.shape[1]), dtype=np.float32)
        old = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
        new = np.linspace(0.0, 1.0, length, dtype=np.float32)
        return np.stack([np.interp(new, old, sequence[:, index]) for index in range(sequence.shape[1])], axis=1).astype(np.float32)

    def _geometry_vector(self, start_frame: int, end_frame: int) -> np.ndarray:
        positions = self._geometry_positions[start_frame:end_frame]
        poses = self._geometry_poses[start_frame:end_frame]
        mask = self._geometry_mask[start_frame:end_frame]
        sequence = np.concatenate([positions, poses], axis=-1)
        valid = sequence[mask] if np.any(mask) else sequence
        if len(valid) == 0:
            valid = np.zeros((1, sequence.shape[-1]), dtype=np.float32)
        deltas = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, sequence.shape[-1]), dtype=np.float32)
        return np.concatenate([valid.mean(axis=0), valid.std(axis=0), valid.min(axis=0), valid.max(axis=0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), deltas.mean(axis=0), deltas.std(axis=0), np.abs(deltas).mean(axis=0)]).astype(np.float32)

    @staticmethod
    def _normalize(probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64)
        values = np.clip(values, 1e-07, None)
        return values / values.sum(axis=-1, keepdims=True)

    def _predict_sequence_window(self, padded_features: np.ndarray, mask: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
        pose_values = self._geometry_poses[start_frame:end_frame]
        pose_mask = self._geometry_mask[start_frame:end_frame].astype(np.float32)
        aligned_pose = self._align(pose_values, self.RGB_LENGTH)
        aligned_mask = self._align(pose_mask[:, None], self.RGB_LENGTH)[:, 0]
        pose_input = np.concatenate([aligned_pose, aligned_mask[:, None]], axis=1)
        rgb_input = np.asarray(padded_features, dtype=np.float32)
        rgb_mask = np.asarray(mask, dtype=bool)
        geometry_vector = self._geometry_vector(start_frame, end_frame).reshape(1, -1)
        seed_probabilities = []
        class_weights = np.asarray([0.8, 1.0, 0.9], dtype=np.float64)
        for item in self.seed_models:
            rgb_probability = item['rgb'].predict({'rgb': rgb_input[None, ...], 'rgb_mask': rgb_mask[None, ...]}, verbose=0)[0]
            same_probability = item['same_rnn'].predict({'rgb': rgb_input[None, ...], 'pose': pose_input[None, ...], 'sequence_mask': rgb_mask[None, ...]}, verbose=0)[0]
            geometry_probability = item['geometry'].predict_proba(geometry_vector)[0]
            geometry_fused = np.exp((1.0 - item['geometry_alpha']) * np.log(np.clip(rgb_probability, 1e-07, 1.0)) + item['geometry_alpha'] * np.log(np.clip(geometry_probability, 1e-07, 1.0)))
            geometry_fused = self._normalize(geometry_fused)
            combined = (1.0 - item['same_rnn_alpha']) * same_probability + item['same_rnn_alpha'] * geometry_fused
            seed_probabilities.append(self._normalize(combined * class_weights))
        return np.mean(np.asarray(seed_probabilities, dtype=np.float64), axis=0, keepdims=True).astype(np.float32)

    def analyze_video(self, video_path: str, confidence_threshold: float=0.5, **kwargs) -> Dict:
        segment_duration = kwargs.pop('segment_duration', 7)
        allowed = {'temporal_fps_mode', 'custom_effective_fps', 'feature_batch_size', 'feature_predict_batch_size', 'progress_callback', 'use_cache', 'cache_dir'}
        kwargs = {key: value for key, value in kwargs.items() if key in allowed}
        return self.analyze_video_segments(video_path, segment_duration=segment_duration, confidence_threshold=confidence_threshold, **kwargs)

    def predict_live_window(self, window_frames: list[np.ndarray], sequence_len: int, sampled_frame_count: int | None=None) -> np.ndarray:
        """Predict a live RGB window while also extracting its pose online."""
        target_count = sampled_frame_count if sampled_frame_count and sampled_frame_count > 0 else sequence_len
        target_count = min(sequence_len, max(1, int(target_count)))
        if len(window_frames) > target_count:
            indices = np.linspace(0, len(window_frames) - 1, num=target_count, dtype=np.int32)
            sampled = [window_frames[index] for index in indices]
        else:
            sampled = window_frames
        frame_array = np.asarray(sampled, dtype=np.uint8)
        features = self.feature_extractor.predict(frame_array, batch_size=32, verbose=0)
        padded_features = np.zeros((sequence_len, features.shape[1]), dtype=np.float32)
        mask = np.zeros(sequence_len, dtype=np.float32)
        length = min(len(features), sequence_len)
        padded_features[:length] = features[:length]
        mask[:length] = 1.0
        self._begin_auxiliary_sequence(length)
        for frame_rgb in sampled[:length]:
            self._on_sampled_frame(frame_rgb[:, :, ::-1])
        self._finish_auxiliary_sequence()
        return self._predict_sequence_window(padded_features, mask, 0, length)

class M1H2D2CrossFoldDetector(PlayerGeometryFusionDetector):
    """Web ensemble built from the six completed M1-H2-D2 outer folds.

    ResNet-50 and YOLO are evaluated once per sampled frame.  Each temporal
    window is then passed through the six fold-specific dual BiGRU models and
    geometry forests.  Their already-fused probabilities receive equal-weight
    soft voting.  The web ensemble itself is intentionally marked exploratory:
    the reported metrics belong to the outer-fold protocol, not to this new
    cross-fold voting policy.
    """
    RGB_LENGTH = 200
    GEOMETRY_FRAMES = 200
    POSE_FRAMES_PER_WINDOW = 100
    EVENT_NEIGHBOR_BOOST = 0.0

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.export_root = self.project_root / M1_H2_D2_WEB_EXPORT_RELATIVE
        self.manifest_path = self.export_root / 'manifest.json'
        if not _resolve_model_artifact(self.manifest_path).exists():
            raise FileNotFoundError(f'M1-H2-D2 web manifest not found: {self.manifest_path}')
        self.manifest = json.loads(_resolve_model_artifact(self.manifest_path).read_text(encoding='utf-8'))
        if self.manifest.get('model_name') != M1_H2_D2_WEB_MODEL_NAME:
            raise ValueError('Unexpected M1-H2-D2 web manifest model name')
        if self.manifest.get('label_order') != ['line', 'other', 'scrum']:
            raise ValueError('Unexpected M1-H2-D2 label order')
        if len(self.manifest.get('folds', [])) != 6:
            raise ValueError('M1-H2-D2 web ensemble requires exactly six folds')
        self.pose_model_path = self.project_root / self.manifest.get('pose_model', PLAYER_GEOMETRY_FUSION_POSE_MODEL)
        if not _resolve_model_artifact(self.pose_model_path).exists():
            raise FileNotFoundError(f'Pose checkpoint not found: {self.pose_model_path}')
        model_dir = self.project_root / 'models' / 'M0'
        RugbyPlayDetector.__init__(self, str(model_dir), model_name='best_model/best_model.keras', extractor_name='resnet50')
        saved_config = sys.modules.pop('config', None)
        sys.path.insert(0, str(source_directory(player_geometry_path)))
        try:
            from models.M1.pose_cache import detect_frame
            from models.M1.geometry_features import build_geometry_features
        finally:
            sys.path.remove(str(source_directory(player_geometry_path)))
            if saved_config is not None:
                sys.modules['config'] = saved_config
        self._detect_frame = detect_frame
        self._build_geometry_features = build_geometry_features
        self.fold_models = []
        for fold in self.manifest['folds']:
            dual_path = self.project_root / fold['dual_model']
            forest_path = self.project_root / fold['geometry_rf']
            missing = [path for path in (dual_path, forest_path) if not _resolve_model_artifact(path).exists()]
            if missing:
                raise FileNotFoundError(f'Missing M1-H2-D2 fold artifacts: {missing}')
            dual_model = keras.models.load_model(_resolve_model_artifact(dual_path), compile=False)
            forest = joblib.load(_resolve_model_artifact(forest_path))
            if int(getattr(forest, 'n_features_in_', -1)) != 549:
                raise ValueError(f'Unexpected geometry input for {forest_path}')
            self.fold_models.append({'fold': fold['fold'], 'dual': dual_model, 'geometry': forest, 'geometry_alpha': float(fold['geometry_alpha']), 'line_threshold': float(fold['line_threshold']), 'scrum_threshold': float(fold['scrum_threshold'])})
        try:
            yolo_config_dir = _temporary_model_settings()
            yolo_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault('YOLO_CONFIG_DIR', str(yolo_config_dir))
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError('M1-H2-D2 web inference requires ultralytics') from exc
        from models.inference_cache import load_with_revision
        self.pose_detector, self._feature_cache_pose_revision = load_with_revision(_resolve_model_artifact(str(self.pose_model_path)), YOLO)
        self.label_vocab = np.asarray(self.manifest['label_order'], dtype=object)
        self._geometry_frames = []
        self._geometry_positions = np.zeros((0, 31), dtype=np.float32)
        self._geometry_poses = np.zeros((0, 30), dtype=np.float32)
        self._geometry_mask = np.zeros((0,), dtype=bool)
        self.fusion_metadata = {'deployment_policy': self.manifest['deployment_policy'], 'fold_count': len(self.fold_models), 'validation_status': self.manifest['validation_status'], 'recommended_web_thresholds': self.manifest['recommended_web_thresholds'], 'cross_validation_metrics': self.manifest['cross_validation_metrics']}

    @property
    def display_name(self) -> str:
        return 'M1-H2-D2 · ensemble de 6 folds'

    @staticmethod
    def _ordered_forest_probability(forest, geometry_vector: np.ndarray) -> np.ndarray:
        raw = np.asarray(forest.predict_proba(geometry_vector)[0], dtype=np.float64)
        ordered = np.zeros(3, dtype=np.float64)
        for index, label_index in enumerate(np.asarray(forest.classes_).astype(int)):
            ordered[int(label_index)] = raw[index]
        return ordered

    def _predict_sequence_window(self, padded_features: np.ndarray, mask: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
        positions = self._geometry_positions[start_frame:end_frame]
        poses = self._geometry_poses[start_frame:end_frame]
        pose_mask = self._geometry_mask[start_frame:end_frame].astype(bool)
        if len(positions) > self.POSE_FRAMES_PER_WINDOW:
            selected = np.linspace(0, len(positions) - 1, self.POSE_FRAMES_PER_WINDOW, dtype=np.int32)
            positions = positions[selected]
            poses = poses[selected]
            pose_mask = pose_mask[selected]
        pose_values = np.concatenate([positions, poses], axis=1)
        aligned_pose = self._align(pose_values, self.RGB_LENGTH)
        aligned_pose_mask = self._align(pose_mask.astype(np.float32)[:, None], self.RGB_LENGTH)[:, 0]
        pose_input = np.concatenate([aligned_pose, aligned_pose_mask[:, None]], axis=1)
        rgb_input = np.asarray(padded_features, dtype=np.float32)
        rgb_mask = np.asarray(mask, dtype=bool)
        valid_geometry = pose_values[pose_mask] if np.any(pose_mask) else pose_values
        if len(valid_geometry) == 0:
            valid_geometry = np.zeros((1, 61), dtype=np.float32)
        deltas = np.diff(valid_geometry, axis=0) if len(valid_geometry) > 1 else np.zeros((1, 61), dtype=np.float32)
        geometry_vector = np.concatenate([valid_geometry.mean(axis=0), valid_geometry.std(axis=0), valid_geometry.min(axis=0), valid_geometry.max(axis=0), np.quantile(valid_geometry, 0.25, axis=0), np.quantile(valid_geometry, 0.75, axis=0), deltas.mean(axis=0), deltas.std(axis=0), np.abs(deltas).mean(axis=0)]).astype(np.float32).reshape(1, -1)
        fold_probabilities = []
        for item in self.fold_models:
            dual_probability = np.asarray(item['dual']({'rgb': rgb_input[None, ...], 'pose': pose_input[None, ...], 'sequence_mask': rgb_mask[None, ...]}, training=False)[0].numpy(), dtype=np.float64)
            geometry_probability = self._ordered_forest_probability(item['geometry'], geometry_vector)
            alpha = float(item['geometry_alpha'])
            fused = np.exp((1.0 - alpha) * np.log(np.clip(dual_probability, 1e-07, 1.0)) + alpha * np.log(np.clip(geometry_probability, 1e-07, 1.0)))
            fold_probabilities.append(self._normalize(fused))
        averaged = self._normalize(np.mean(fold_probabilities, axis=0))
        return averaged[None, :].astype(np.float32)

