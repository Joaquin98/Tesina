"""Web adapters for the selected M3 Panel-V cross-fold configurations."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json
from pathlib import Path
from typing import Dict, List
import joblib
import keras
import numpy as np
from models.common.event_verifier import LineEventVerifier, _runs
from models.M2.predictor import LABELS, RGBYoloHighresQuantileFusionDetector
PROJECT_ROOT = Path(__file__).resolve().parents[2]
M3_H2_D2_MODEL_NAME = 'm3_h2_d2_cross_fold_ensemble'
M3_H1_D2_MODEL_NAME = 'm3_h1_d2_cross_fold_ensemble'
MANIFEST_RELATIVE = {M3_H2_D2_MODEL_NAME: 'data/final_model_comparison_20260827/outputs/web_export_m3_h2_d2/manifest.json', M3_H1_D2_MODEL_NAME: 'data/final_model_comparison_20260827/outputs/web_export_m3_h1_d2/manifest.json'}

class M3CrossFoldDetector(RGBYoloHighresQuantileFusionDetector):
    """Six-fold soft-vote deployment for a selected M3 configuration.

    RGB ResNet features and high-resolution YOLO observations are extracted
    once.  Each fold then applies its own historical/static/temporal heads and
    its own line-event verifier.  Three-class probabilities use equal soft
    voting; a line event needs at least three positive fold-verifier votes.
    """
    EVENT_NEIGHBOR_BOOST = 0.0

    def __init__(self, project_root: Path=PROJECT_ROOT, *, model_name: str):
        if model_name not in MANIFEST_RELATIVE:
            raise ValueError(f'Unsupported final-comparison model: {model_name}')
        self.project_root = Path(project_root).resolve()
        self.manifest_path = self.project_root / MANIFEST_RELATIVE[model_name]
        if not _resolve_model_artifact(self.manifest_path).is_file():
            raise FileNotFoundError(f'M3 web manifest not found: {self.manifest_path}')
        self.manifest = json.loads(_resolve_model_artifact(self.manifest_path).read_text(encoding='utf-8'))
        if self.manifest.get('model_name') != model_name:
            raise ValueError('Unexpected M3 web manifest model name')
        if self.manifest.get('label_order') != ['line', 'other', 'scrum']:
            raise ValueError('Unexpected M3 label order')
        if len(self.manifest.get('folds', [])) != 6:
            raise ValueError('M3 cross-fold web ensemble requires exactly six folds')
        first_historical = self.manifest['folds'][0]['historical_model']
        super().__init__(self.project_root, historical_model_relative=first_historical, model_name=model_name, display_name='M3-H2-D2 · ensemble de 6 folds' if model_name == M3_H2_D2_MODEL_NAME else 'M3-H1-D2 · ensemble de 6 folds')
        try:
            import torch
            from models.M2.grid_search_cached_descriptors import primitive_base
            from models.M3.run_temporal_architecture_suite import build_model
            from models.M3.run_temporal_attention_experiment import quality_score, smooth_three, usable_mask
        except ImportError as exc:
            raise ImportError('M3 web inference requires PyTorch and the YOLO geometry modules') from exc
        self._torch = torch
        self._primitive_base = primitive_base
        self._smooth_three = smooth_three
        self._quality_score = quality_score
        self._usable_mask = usable_mask
        object_cache: dict[tuple[str, str], object] = {('keras', first_historical): self.sequence_model}
        self.fold_models = []
        for fold in self.manifest['folds']:
            paths = {key: self.project_root / fold[key] for key in ('dynamic_pca', 'quantile_scaler', 'logistic', 'historical_model', 'temporal_model', 'line_verifier')}
            missing = [str(path) for path in paths.values() if not _resolve_model_artifact(path).is_file()]
            if missing:
                raise FileNotFoundError(f'Missing M3 fold artifacts: {missing}')

            def cached(kind: str, key: str, loader):
                cache_key = (kind, str(paths[key].resolve()))
                if cache_key not in object_cache:
                    object_cache[cache_key] = loader(_resolve_model_artifact(paths[key]))
                return object_cache[cache_key]
            checkpoint = torch.load(_resolve_model_artifact(paths['temporal_model']), map_location='cpu', weights_only=False)
            temporal_model = build_model(str(checkpoint['architecture']))
            temporal_model.load_state_dict(checkpoint['state_dict'])
            temporal_model.eval()
            verifier = cached('joblib', 'line_verifier', joblib.load)
            expected = getattr(verifier, 'n_features_in_', None)
            if expected is not None and int(expected) != 208:
                raise ValueError(f'Fold verifier expects {expected} features instead of 208')
            self.fold_models.append({'fold': fold['fold'], 'dynamic_pca': cached('joblib', 'dynamic_pca', joblib.load), 'scaler': cached('joblib', 'quantile_scaler', joblib.load), 'logistic': cached('joblib', 'logistic', joblib.load), 'historical': cached('keras', 'historical_model', lambda path: keras.models.load_model(_resolve_model_artifact(path), compile=False)), 'temporal': temporal_model, 'temporal_mean': np.asarray(checkpoint['mean'], dtype=np.float32), 'temporal_std': np.asarray(checkpoint['std'], dtype=np.float32), 'temporal_temperature': float(checkpoint.get('temperature', 1.0)), 'verifier': verifier, 'verifier_threshold': float(fold['line_verifier_threshold']), 'line_event_verifier': LineEventVerifier(verifier, threshold=float(fold['line_verifier_threshold']), historical_weight=0.4, static_weight=0.6), 'static_alpha': float(fold['static_alpha'])})
        self._window_fold_predictions: list[list[dict[str, np.ndarray]]] = []
        self.label_vocab = LABELS.copy()
        self.fusion_metadata = {'model_name': model_name, 'source_configuration': self.manifest['source_configuration'], 'comparison_rank': self.manifest['comparison_rank'], 'deployment_policy': self.manifest['deployment_policy'], 'event_verifier_policy': self.manifest['event_verifier_policy'], 'fold_count': 6, 'validation_status': self.manifest['validation_status'], 'recommended_web_thresholds': self.manifest['recommended_web_thresholds'], 'cross_validation_metrics': self.manifest['cross_validation_metrics'], 'pose_input_mode': 'source_equivalent_roi'}

    def _begin_auxiliary_sequence(self, sampled_total_frames: int) -> None:
        super()._begin_auxiliary_sequence(sampled_total_frames)
        self._window_fold_predictions = []

    def _dynamic_vector(self, padded_features: np.ndarray, length: int, pca) -> np.ndarray:
        rgb = np.asarray(padded_features[:length], dtype=np.float32)
        if rgb.shape[1] != 2048:
            raise ValueError(f'M3 expects 2,048 ResNet features, got {rgb.shape[1]}')
        rgb = self._align(rgb, self.RGB_LENGTH)
        segments = rgb.reshape(10, 20, 2048)
        segment_mean = segments.mean(axis=1)
        segment_delta = np.abs(np.diff(segments, axis=1)).mean(axis=1)
        global_mean = rgb.mean(axis=0)
        global_std = rgb.std(axis=0)
        return np.concatenate([pca['segment'].transform(segment_mean).reshape(-1), pca['delta'].transform(segment_delta).reshape(-1), pca['mean'].transform(global_mean.reshape(1, -1)).reshape(-1), pca['std'].transform(global_std.reshape(1, -1)).reshape(-1)]).astype(np.float32)

    @staticmethod
    def _ordered_probability(classifier, transformed: np.ndarray) -> np.ndarray:
        raw = np.asarray(classifier.predict_proba(transformed)[0], dtype=np.float64)
        ordered = np.zeros(3, dtype=np.float64)
        for index, label_index in enumerate(np.asarray(classifier.classes_).astype(int)):
            ordered[int(label_index)] = raw[index]
        return ordered

    @staticmethod
    def _resample_raw_window(raw: dict[str, np.ndarray], length: int=100) -> dict[str, np.ndarray]:
        available = len(raw['frame_mask'])
        if available <= 0:
            raise ValueError('Temporal YOLO window has no observations')
        indices = np.rint(np.linspace(0, available - 1, num=length)).astype(np.int32)
        return {key: np.asarray(value)[indices] for key, value in raw.items()}

    def _temporal_input(self, start_frame: int, end_frame: int):
        if self._raw_geometry is None:
            return None
        source = np.asarray(self._yolo_source_indices, dtype=np.int32)
        selected = (source >= int(start_frame)) & (source < int(end_frame))
        if not np.any(selected):
            return None
        raw = self._resample_raw_window({key: np.asarray(value)[selected] for key, value in self._raw_geometry.items()})
        primitive = self._primitive_base(raw)
        frame_mask = primitive['mask'].astype(bool)
        quality = primitive['quality'].astype(np.float32)
        current = self._smooth_three(primitive['current'][None], frame_mask[None])[0]
        rugby = self._smooth_three(primitive['rugby'][None], frame_mask[None])[0]
        features = np.nan_to_num(np.concatenate([current, quality, rugby], axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        score = self._quality_score(quality[None])[0]
        usable = self._usable_mask(frame_mask[None], score[None])[0]
        return (features, usable, score)

    def _temporal_probability(self, member: dict, temporal_input) -> np.ndarray:
        if temporal_input is None:
            return np.full(3, 1.0 / 3.0, dtype=np.float32)
        features, usable, score = temporal_input
        normalized = np.clip((features - member['temporal_mean']) / member['temporal_std'], -8.0, 8.0).astype(np.float32)
        normalized[~usable] = 0.0
        with self._torch.no_grad():
            logits = member['temporal'](self._torch.from_numpy(normalized[None]), self._torch.from_numpy(usable[None]), self._torch.from_numpy(score[None]))
            probability = self._torch.softmax(logits / member['temporal_temperature'], dim=1)[0].cpu().numpy()
        return self._normalize(probability).astype(np.float32)

    def _predict_sequence_window(self, padded_features: np.ndarray, mask: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
        length = max(0, min(int(end_frame - start_frame), int(np.sum(mask))))
        yolo = self._yolo_vector(start_frame, end_frame)
        effective_fps = float(getattr(self, '_web_effective_fps', 0.0) or 0.0)
        grid_frames = effective_fps * 7.0
        on_grid = grid_frames <= 0.0 or abs(float(start_frame) / grid_frames - round(float(start_frame) / grid_frames)) <= 0.075
        temporal_input = self._temporal_input(start_frame, end_frame) if on_grid else None
        fold_predictions = []
        for member in self.fold_models:
            historical = member['historical'].predict([np.expand_dims(padded_features, axis=0), np.expand_dims(mask, axis=0)], verbose=0)[0]
            historical = self._normalize(historical)
            dynamic = self._dynamic_vector(padded_features, length, member['dynamic_pca'])
            vector = np.concatenate([dynamic, yolo]).reshape(1, -1)
            if vector.shape[1] != 1686:
                raise ValueError(f'Unexpected M3 static feature shape: {vector.shape}')
            static = self._ordered_probability(member['logistic'], member['scaler'].transform(vector))
            static = self._normalize(static)
            alpha = member['static_alpha']
            official = self._normalize((1.0 - alpha) * historical + alpha * static)
            temporal = self._temporal_probability(member, temporal_input) if on_grid else np.full(3, 1.0 / 3.0, dtype=np.float32)
            fold_predictions.append({'historical': historical.astype(np.float32), 'static': static.astype(np.float32), 'official': official.astype(np.float32), 'temporal': temporal.astype(np.float32)})
        self._window_fold_predictions.append(fold_predictions)
        averaged = self._normalize(np.mean([item['official'] for item in fold_predictions], axis=0))
        return averaged[None, :].astype(np.float32)

    @staticmethod
    def _audited_grid(windows: list[dict]) -> list[int]:
        if not windows:
            return []
        origin = float(windows[0]['start_time'])
        return [index for index, item in enumerate(windows) if abs((float(item['start_time']) - origin) / 7.0 - round((float(item['start_time']) - origin) / 7.0)) <= 0.075]

    def _verify_line_ensemble(self, windows: list[dict], *, video_duration: float, source_fps: float) -> tuple[list[dict], list[dict]]:
        grid_indices = self._audited_grid(windows)
        if not grid_indices:
            return ([], [])
        grid = [windows[index] for index in grid_indices]
        per_fold = [[self._window_fold_predictions[index][fold_index] for index in grid_indices] for fold_index in range(len(self.fold_models))]
        ensemble_verifier_official = self._normalize(np.mean([[0.4 * window['historical'] + 0.6 * window['static'] for window in fold_windows] for fold_windows in per_fold], axis=0))
        votes_by_start = {round(float(item['start_time']), 6): 0 for item in grid}
        fold_event_scores: list[dict[float, float]] = []
        for fold_index, member in enumerate(self.fold_models):
            fold_windows = []
            for window_index, item in enumerate(windows):
                branches = self._window_fold_predictions[window_index][fold_index]
                fold_windows.append({**item, 'branch_probabilities': {name: {label: float(value) for label, value in zip(LABELS, probability)} for name, probability in branches.items()}})
            fold_events, fold_accepted = member['line_event_verifier'].verify(fold_windows, video_duration=video_duration, source_fps=source_fps)
            event_score_map = {}
            for event in fold_events:
                score = float(event['line_verifier_score'])
                for contributing in event.get('contributing_windows', []):
                    event_score_map[round(float(contributing['start_time']), 6)] = score
            fold_event_scores.append(event_score_map)
            for item in fold_accepted:
                key = round(float(item['start_time']), 6)
                if key in votes_by_start:
                    votes_by_start[key] += 1
        vote_counts = np.asarray([votes_by_start[round(float(item['start_time']), 6)] for item in grid], dtype=np.int32)
        candidates = list(_runs(vote_counts >= getattr(self, 'line_min_votes', 3)))
        accepted_events = []
        accepted_windows = []
        for start, stop in candidates:
            source = grid[start:stop]
            accepted_windows.extend([dict(item, play_type='line') for item in source])
            start_time = max(0.0, float(source[0]['start_time']))
            end_time = min(float(video_duration), float(source[-1]['end_time']))
            average = ensemble_verifier_official[start:stop].mean(axis=0)
            contributing_keys = {round(float(item['start_time']), 6) for item in source}
            scores = [max((score_map.get(key, 0.0) for key in contributing_keys), default=0.0) for score_map in fold_event_scores]
            accepted_events.append({'play_type': 'line', 'start_time': start_time, 'end_time': end_time, 'duration': max(0.0, end_time - start_time), 'confidence': float(np.max(ensemble_verifier_official[start:stop, 0])), 'mean_confidence': float(np.mean(ensemble_verifier_official[start:stop, 0])), 'frame_start': int(round(start_time * source_fps)) if source_fps > 0 else 0, 'frame_end': int(round(end_time * source_fps)) if source_fps > 0 else 0, 'probabilities': {label: float(value) for label, value in zip(LABELS, average)}, 'source_window_count': len(source), 'line_verifier_votes': int(np.max(vote_counts[start:stop])), 'line_verifier_window_votes': vote_counts[start:stop].tolist(), 'line_verifier_fold_scores': scores, 'line_verifier_policy': f"at_least_{getattr(self, 'line_min_votes', 3)}_of_{len(self.fold_models)}"})
        return (accepted_events, accepted_windows)

    def _post_process_event_detections(self, window_predictions: List[Dict], confidence_threshold: float, video_duration: float, source_fps: float, class_confidence_thresholds: Dict[str, float] | None=None) -> List[Dict]:
        if len(window_predictions) != len(self._window_fold_predictions):
            raise RuntimeError('M3 window/fold prediction alignment failed')
        base_events = super()._post_process_event_detections(window_predictions, confidence_threshold, video_duration, source_fps, class_confidence_thresholds=class_confidence_thresholds)
        base_review = list(getattr(self, '_last_review_window_detections', []))
        line_events, line_review = self._verify_line_ensemble(window_predictions, video_duration=video_duration, source_fps=source_fps)
        self._last_review_window_detections = sorted([item for item in base_review if item.get('play_type') != 'line'] + line_review, key=lambda item: (float(item['start_time']), str(item.get('play_type', ''))))
        result = [event for event in base_events if event.get('play_type') != 'line']
        result.extend(line_events)
        result.sort(key=lambda item: float(item['start_time']))
        return result

    def analyze_video_segments(self, *args, **kwargs) -> Dict:
        result = super().analyze_video_segments(*args, **kwargs)
        if result.get('status') == 'success':
            result.setdefault('event_post_processing', {}).update({'line_mode': 'six_fold_verifier_majority_vote' if len(self.fold_models) == 6 else 'single_run_verifier', 'line_verifier_fold_count': len(self.fold_models), 'line_verifier_min_votes': getattr(self, 'line_min_votes', 3), 'line_ui_threshold_used': False, 'validation_status': self.manifest['validation_status']})
        return result

class M3H2D2CrossFoldDetector(M3CrossFoldDetector):

    def __init__(self, project_root: Path=PROJECT_ROOT):
        super().__init__(project_root, model_name=M3_H2_D2_MODEL_NAME)

class M3H1D2CrossFoldDetector(M3CrossFoldDetector):

    def __init__(self, project_root: Path=PROJECT_ROOT):
        super().__init__(project_root, model_name=M3_H1_D2_MODEL_NAME)
