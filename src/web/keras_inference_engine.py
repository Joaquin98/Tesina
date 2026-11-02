"""
Keras Inference Engine for CNN-RNN Rugby Play Detection.

Adapter to use the trained Keras models with the Flask web interface.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
import contextlib
import io
import os
import numpy as np
import cv2
import sys
import keras
import joblib
web_module_path = Path(__file__).parent
if str(web_module_path) not in sys.path:
    sys.path.insert(0, str(web_module_path))
models_path = Path(__file__).parent.parent.parent / 'models' / 'M0'
sys.path.insert(0, str(models_path))
two_stream_path = Path(__file__).parent.parent.parent / 'data' / 'two_stream_feature_fusion'
sys.path.insert(0, str(two_stream_path))
player_geometry_path = Path(__file__).parent.parent.parent / 'data' / 'player_geometry_fusion'
try:
    from models.M0.web_adapter import RugbyPlayDetector
    ADAPTER_AVAILABLE = True
except ImportError as e:
    ADAPTER_AVAILABLE = False
    import_error = str(e)
try:
    from models.M0.ensemble_detector import EnsembleDetector
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
logger = logging.getLogger(__name__)
TWO_STREAM_FULL_MODEL_NAME = 'two_stream_full_classifier'
TWO_STREAM_PRECISION_WORKFLOW_NAME = 'two_stream_precision_exact_line1_scrum7'
META_CORRECTION_MODEL_NAME = 'meta_correction/recommended_same_video_50pct'
META_CORRECTION_ARTIFACT_RELATIVE = 'data/same_video_few_shot_suite_20260721/recommended_model/meta_correction_balanced_logreg.joblib'
META_CORRECTION_PRIMARY_RELATIVE = 'data/line_recall_training/full/training/line_weight_1_25/best_accuracy.keras'
META_CORRECTION_METRICS_RELATIVE = 'data/same_video_few_shot_suite_20260721/recommended_model/metrics.json'
META_CORRECTION_LABELS = ('line', 'other', 'scrum')
CLAHE_WEB_MODEL_NAME = 'experimental/clahe_line_weight_1_25_best_macro.keras'
CLAHE_WEB_MODEL_PROJECT_RELATIVE = 'data/filter_sweep/clahe_weight_sweep/line_weight_1_25/training/line_weight_1_25/best_macro.keras'
CLAHE_WEB_MODEL_CHECKPOINT_RELATIVE = 'M0/checkpoints/' + CLAHE_WEB_MODEL_NAME
PLAYER_GEOMETRY_FUSION_MODEL_NAME = 'player_geometry_fusion_calibrated'
PLAYER_GEOMETRY_FUSION_EXPORT_RELATIVE = 'data/player_geometry_fusion/results/web_export'
PLAYER_GEOMETRY_FUSION_POSE_MODEL = 'yolo11n-pose.pt'
M1_H2_D2_WEB_MODEL_NAME = 'm1_h2_d2_cross_fold_ensemble'
M1_H2_D2_WEB_EXPORT_RELATIVE = 'data/final_model_comparison_20260827/outputs/web_export_m1_h2_d2'
M3_H2_D2_WEB_MODEL_NAME = 'm3_h2_d2_cross_fold_ensemble'
M3_H1_D2_WEB_MODEL_NAME = 'm3_h1_d2_cross_fold_ensemble'
M3_WEB_EXPORTS = {M3_H2_D2_WEB_MODEL_NAME: 'data/final_model_comparison_20260827/outputs/web_export_m3_h2_d2', M3_H1_D2_WEB_MODEL_NAME: 'data/final_model_comparison_20260827/outputs/web_export_m3_h1_d2'}
HISTORICAL_YOLO_RESIDUAL_MODEL_NAME = 'historical_yolo_residual_ensemble'
HISTORICAL_YOLO_RESIDUAL_ROOT_RELATIVE = 'data/historical_locked_yolo_residual_20260811'
HISTORICAL_YOLO_RESIDUAL_CHECKPOINT_RELATIVE = 'data/line_recall_training/full/training/line_weight_1_25/best_accuracy.keras'
RGB_YOLO_QUANTILE_FUSION_MODEL_NAME = 'rgb_yolo_quantile_fusion_all'
RGB_YOLO_QUANTILE_FUSION_ROOT_RELATIVE = 'data/joint_multimodal_20260813'
RGB_YOLO_QUANTILE_DEPLOYMENT_RELATIVE = RGB_YOLO_QUANTILE_FUSION_ROOT_RELATIVE + '/deployment_fusion_quantile_100frames_seed42'
RGB_YOLO_QUANTILE_FEATURE_RELATIVE = RGB_YOLO_QUANTILE_FUSION_ROOT_RELATIVE + '/rgb_dynamics_seed42'
RGB_YOLO_HIGHRES_MODEL_NAME = 'rgb_yolo_quantile_highres_roi'
RGB_YOLO_HIGHRES_DEPLOYMENT_RELATIVE = 'data/player_geometry_highres_pose/outputs/deployment_quantile_100frames'
RGB_YOLO_EVENT_VERIFIER_MODEL_NAME = 'rgb_yolo_highres_temporal_event_verifier'
RGB_YOLO_EVENT_VERIFIER_RELATIVE = 'data/line_cached_improvement_suite_20260818/outputs/stable_event_logreg_c0_1_pending_external_match.joblib'
RGB_YOLO_TEMPORAL_MODELS_RELATIVE = 'data/player_geometry_highres_pose/outputs/temporal_external_primera_20260815/temporal_models'
RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME = 'rgb_yolo_historical_primera_calibrated'
RGB_YOLO_PRIMERA_HISTORICAL_RELATIVE = 'data/historical_primera_grouped_20260818/outputs/historical_primera.keras'
RGB_YOLO_PRIMERA_VERIFIER_RELATIVE = 'data/historical_primera_calibrated_20260818/outputs/line_verifier_historical_primera.joblib'

class MetaCorrectionDetector:
    """Primary CNN-RNN followed by the persisted few-shot meta-corrector.

    The joblib artifact was fitted on the primary model's three probabilities
    and deterministic confidence features.  This adapter deliberately keeps
    both outputs: ``play_type`` is the corrected label and
    ``primary_play_type`` is the unmodified CNN-RNN decision.  The latter is
    useful for audit/review and prevents the second stage from becoming a
    unannounced changes to historical model selection.
    """
    label_vocab = np.asarray(META_CORRECTION_LABELS, dtype=object)

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.artifact_path = self.project_root / META_CORRECTION_ARTIFACT_RELATIVE
        self.primary_model_path = self.project_root / META_CORRECTION_PRIMARY_RELATIVE
        if not _resolve_model_artifact(self.artifact_path).exists():
            raise FileNotFoundError(f'Meta-corrector artifact not found: {self.artifact_path}')
        if not _resolve_model_artifact(self.primary_model_path).exists():
            raise FileNotFoundError(f'Primary CNN-RNN checkpoint not found: {self.primary_model_path}')
        self.corrector = joblib.load(_resolve_model_artifact(self.artifact_path))
        from models.M0.web_adapter import RugbyPlayDetector
        model_dir = self.project_root / 'models' / 'M0'
        self.primary = RugbyPlayDetector(str(model_dir), model_name=str(self.primary_model_path), extractor_name='resnet50')
        self.feature_extractor = self.primary.feature_extractor
        self.sequence_model = self.primary.sequence_model
        self.label_vocab = np.asarray(self.primary.label_vocab, dtype=object)

    @property
    def display_name(self) -> str:
        return 'CNN-RNN + Corrector few-shot (mismo video)'

    @staticmethod
    def _features(probability_rows: np.ndarray) -> np.ndarray:
        """Match the 11-feature representation used during fitting exactly."""
        p = np.asarray(probability_rows, dtype=float)
        p = np.clip(p, 1e-07, 1.0)
        entropy = -np.sum(p * np.log(p), axis=1, keepdims=True)
        margins = np.column_stack([p[:, 0] - p[:, 1], p[:, 0] - p[:, 2], p[:, 1] - p[:, 2]])
        sorted_p = np.sort(p, axis=1)
        top2_margin = (sorted_p[:, -1] - sorted_p[:, -2]).reshape(-1, 1)
        return np.column_stack([p, np.log(p), margins, entropy, top2_margin])

    def correct_probabilities(self, probabilities: dict) -> tuple[str, float, dict, dict]:
        """Correct one primary probability vector.

        Returns ``(label, confidence, corrected_probabilities, audit)``.  The
        method is also used by the live-window path in ``src/web/app.py``.
        """
        primary_probs = np.asarray([float(probabilities.get(label, 0.0)) for label in META_CORRECTION_LABELS], dtype=float)
        primary_probs = np.clip(primary_probs, 0.0, 1.0)
        total = float(primary_probs.sum())
        if total <= 0.0:
            primary_probs[:] = 1.0 / len(META_CORRECTION_LABELS)
        else:
            primary_probs /= total
        features = self._features(primary_probs.reshape(1, -1))
        primary_index = int(np.argmax(primary_probs))
        corrected_index = int(self.corrector.predict(features)[0])
        corrected_probs = np.zeros(len(META_CORRECTION_LABELS), dtype=float)
        if hasattr(self.corrector, 'predict_proba'):
            predicted_probabilities = np.asarray(self.corrector.predict_proba(features)[0], dtype=float)
            classes = getattr(self.corrector, 'classes_', None)
            if classes is None and hasattr(self.corrector, 'named_steps'):
                final_step = list(self.corrector.named_steps.values())[-1]
                classes = getattr(final_step, 'classes_', None)
            if classes is not None:
                for class_id, value in zip(classes, predicted_probabilities):
                    class_index = int(class_id)
                    if 0 <= class_index < len(corrected_probs):
                        corrected_probs[class_index] = float(value)
        if corrected_probs.sum() <= 0.0:
            corrected_probs[corrected_index] = 1.0
        else:
            corrected_probs /= corrected_probs.sum()
        corrected_label = META_CORRECTION_LABELS[corrected_index]
        corrected_by_label = {label: float(corrected_probs[idx]) for idx, label in enumerate(META_CORRECTION_LABELS)}
        primary_by_label = {label: float(primary_probs[idx]) for idx, label in enumerate(META_CORRECTION_LABELS)}
        audit = {'primary_play_type': META_CORRECTION_LABELS[primary_index], 'primary_confidence': float(primary_probs[primary_index]), 'primary_probabilities': primary_by_label, 'meta_corrector_applied': True, 'meta_corrector_scope': 'same_video_support_required'}
        return (corrected_label, float(corrected_probs[corrected_index]), corrected_by_label, audit)

    def _correct_windows(self, window_predictions: list[dict]) -> list[dict]:
        corrected = []
        for original in window_predictions or []:
            item = dict(original)
            primary_label = str(item.get('play_type', 'other'))
            primary_confidence = float(item.get('confidence', 0.0))
            label, confidence, probabilities, audit = self.correct_probabilities(item.get('probabilities') or {})
            item.update(audit)
            item['primary_play_type'] = primary_label
            item['primary_confidence'] = primary_confidence
            item['original_probabilities'] = dict(audit['primary_probabilities'])
            item['play_type'] = label
            item['confidence'] = confidence
            item['probabilities'] = probabilities
            corrected.append(item)
        return corrected

    def analyze_video_segments(self, video_path: str, **kwargs) -> Dict:
        with contextlib.redirect_stdout(io.StringIO()):
            primary_results = self.primary.analyze_video_segments(video_path, **kwargs)
        result = dict(primary_results)
        primary_windows = list(primary_results.get('window_predictions') or [])
        corrected_windows = self._correct_windows(primary_windows)
        confidence_threshold = float(kwargs.get('confidence_threshold', 0.5))
        duration = float(primary_results.get('duration', 0.0))
        source_fps = float(primary_results.get('source_fps', 0.0))
        corrected_detections = self.primary._post_process_event_detections(corrected_windows, confidence_threshold=confidence_threshold, video_duration=duration, source_fps=source_fps, class_confidence_thresholds={'line': float(kwargs['line_confidence_threshold'])} if kwargs.get('line_confidence_threshold') is not None else None)
        for detection in corrected_detections:
            peak = detection.get('peak_window') or {}
            candidates = [item for item in corrected_windows if abs(float(item.get('start_time', 0.0)) - float(peak.get('start_time', 0.0))) < 1e-06 and abs(float(item.get('end_time', 0.0)) - float(peak.get('end_time', 0.0))) < 1e-06]
            if candidates:
                source_window = candidates[0]
                detection['primary_play_type'] = source_window.get('primary_play_type')
                detection['primary_confidence'] = source_window.get('primary_confidence')
                detection['original_probabilities'] = source_window.get('primary_probabilities')
                detection['meta_corrector_applied'] = True
        review_windows = getattr(self.primary, '_last_review_window_detections', [])
        result.update({'window_predictions': corrected_windows, 'review_detections_7s': review_windows, 'detections': corrected_detections, 'primary_detections': primary_results.get('detections', []), 'meta_correction': {'applied': True, 'artifact': str(self.artifact_path), 'primary_checkpoint': str(self.primary_model_path), 'scope': 'same_video_support_required', 'support_rows': 401, 'support_fraction': 0.5, 'warning': 'Recalibración entrenada para un partido específico; para otro video se necesita soporte etiquetado nuevo.'}})
        return result

    def analyze_video(self, video_path: str, confidence_threshold: float=0.5, **kwargs) -> Dict:
        with contextlib.redirect_stdout(io.StringIO()):
            primary_results = dict(self.primary.analyze_video(video_path, confidence_threshold=confidence_threshold, **kwargs))
        probabilities = primary_results.get('all_probabilities') or {}
        if not probabilities and primary_results.get('detections'):
            probabilities = primary_results['detections'][0].get('probabilities') or {}
        label, confidence, corrected_probs, audit = self.correct_probabilities(probabilities)
        primary_results.update({'predicted_class': label, 'confidence': confidence, 'all_probabilities': corrected_probs, 'primary_predicted_class': audit['primary_play_type'], 'primary_confidence': audit['primary_confidence'], 'primary_all_probabilities': audit['primary_probabilities'], 'meta_correction': {'applied': True, 'artifact': str(self.artifact_path), 'primary_checkpoint': str(self.primary_model_path), 'scope': 'same_video_support_required', 'support_rows': 401, 'support_fraction': 0.5}})
        for detection in primary_results.get('detections', []):
            detection['primary_play_type'] = audit['primary_play_type']
            detection['primary_confidence'] = audit['primary_confidence']
            detection['original_probabilities'] = dict(audit['primary_probabilities'])
            detection['play_type'] = label
            detection['confidence'] = confidence
            detection['probabilities'] = dict(corrected_probs)
            detection['meta_corrector_applied'] = True
        return primary_results
CLASS_IMBALANCE_EXPERIMENT_MODELS = {'line_recall/calibrated_alpha_2_95.keras': {'project_relative_path': 'data/line_recall_training/artifacts/line_recall_calibrated_alpha_2_95/model.keras', 'display_name': 'Line recall: Natural calibrado x2,95 (83,1% recall)', 'description': 'ResNet-50 + BiGRU natural con calibración de salida para line. No fue reentrenado: incorpora el multiplicador 2,95 en el sesgo softmax para priorizar recall de line.', 'metrics': {'test_accuracy': 0.8616891064871481, 'f1_macro': 0.8361043340214156, 'f1_weighted': 0.8639230373745928, 'precision_line': 0.7272727272727273, 'recall_line': 0.8306010928961749, 'f1_line': 0.7755102040816326, 'f1_other': 0.8985781990521327, 'f1_scrum': 0.8342245989304813}}, 'class_imbalance/natural_unweighted.keras': {'filename': 'natural_unweighted.keras', 'display_name': 'Class Imbalance: Natural sin pesos (86.3%)', 'description': 'ResNet-50 + BiGRU entrenada con la distribución natural, sin remuestreo ni pesos. Mejor F1 macro del experimento aislado.', 'metrics': {'test_accuracy': 0.8629130966952264, 'f1_macro': 0.83402970973795, 'f1_weighted': 0.8626954512752284, 'f1_line': 0.7611111111111111, 'f1_other': 0.9005524861878453, 'f1_scrum': 0.8404255319148937}}, 'class_imbalance/natural_class_weights.keras': {'filename': 'natural_class_weights.keras', 'display_name': 'Class Imbalance: Natural + pesos (85.6%)', 'description': 'ResNet-50 + BiGRU entrenada con la distribución natural y pesos de clase balanced. Segunda por F1 macro en el experimento aislado.', 'metrics': {'test_accuracy': 0.8555691554467564, 'f1_macro': 0.8265474040060049, 'f1_weighted': 0.8558944607611353, 'f1_line': 0.7548209366391184, 'f1_other': 0.8940520446096655, 'f1_scrum': 0.8307692307692308}}}
NO_PLAY_TUNED_ENSEMBLES = {'no_play_tuned_macro_ensemble': {'display_name': 'No-play Tuned Ensemble (best macro-F1)', 'description': 'Soft-voting ensemble trained with curated Granitto no-play negatives and human-reviewed fragments. Tuned for the best mixed macro-F1.', 'members': ['curated_np273_line125_aug', 'mixed_np273_human860_line125_aug', 'mixed_np273_human860_aug', 'curated_np209_aug'], 'class_bias': {'line': 0.975, 'other': 1.0, 'scrum': 2.0}, 'metrics': {'mixed_accuracy': 0.8363228699551569, 'mixed_f1_macro': 0.786602265539647, 'curated_accuracy': 0.8298397040690506, 'curated_f1_macro': 0.7932099057919969, 'heldout_no_play_as_other': 73 / 81}}, 'no_play_tuned_accuracy_ensemble': {'display_name': 'No-play Tuned Ensemble (best accuracy)', 'description': 'Two-model soft-voting ensemble trained with curated Granitto no-play negatives and human-reviewed fragments. Tuned for the best mixed accuracy.', 'members': ['curated_np273_aug', 'mixed_np273_human860_line125_aug'], 'class_bias': {'line': 1.045, 'other': 1.0, 'scrum': 1.22}, 'metrics': {'mixed_accuracy': 0.8396860986547086, 'mixed_f1_macro': 0.782172000852556, 'curated_accuracy': 0.8323057953144266, 'curated_f1_macro': 0.7874867543879739, 'heldout_no_play_as_other': 74 / 81}}}

class TwoStreamProductionDetector:
    """Web adapter for the production Two-Stream descriptor pipeline."""
    label_vocab = np.array(['line', 'other', 'scrum'], dtype=object)

    def __init__(self, mode: str, project_root: Path):
        self.mode = mode
        self.project_root = Path(project_root)
        self.experiments_dir = self.project_root / 'data' / 'two_stream_feature_fusion'
        self.results_dir = self.experiments_dir / 'results'
        self.cache_dir = self.results_dir / 'production_candidate' / 'video_descriptor_cache'
        self.feature_extractor = None
        from production_candidate_inference import load_candidate, load_target_triage_policy
        self.bundle = load_candidate()
        self.target_policy = load_target_triage_policy(self.results_dir / 'production_target_triage_policy.json')

    @property
    def display_name(self) -> str:
        if self.mode == TWO_STREAM_PRECISION_WORKFLOW_NAME:
            return 'Two-Stream precision review workflow'
        return 'Two-Stream full classifier'

    def _duration_seconds(self, video_path: Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            return float(frames / fps) if fps > 0 else 0.0
        finally:
            cap.release()

    def _descriptor(self, video_path: Path) -> tuple[np.ndarray, str]:
        from production_video_predict import build_resnet50_extractor, expanded_descriptor_cached, load_cached_descriptor
        cached = load_cached_descriptor(Path(video_path), self.cache_dir)
        if cached is not None:
            return (cached, 'cache')
        if self.feature_extractor is None:
            self.feature_extractor = build_resnet50_extractor()
        return expanded_descriptor_cached(Path(video_path), self.feature_extractor, cache_dir=self.cache_dir, overwrite_cache=False)

    def analyze_video(self, video_path: str, confidence_threshold: float=0.5, **_kwargs) -> Dict:
        from production_candidate_inference import line_review_rank_score, predict_from_features, score_meta_router_probability, scrum_review_rank_score, target_review_rank_score, target_specialist_probability, target_triage_decisions
        progress_callback = _kwargs.get('progress_callback')
        if progress_callback:
            progress_callback(18, 'preprocessing', 'Preparando el descriptor del video')
        path = Path(video_path)
        descriptor, descriptor_source = self._descriptor(path)
        if progress_callback:
            progress_callback(70, 'features', 'Descriptor visual extraído')
        labels, probability = predict_from_features(descriptor, self.bundle, profile='balanced')
        specialist = target_specialist_probability(descriptor, self.bundle)
        score_meta = score_meta_router_probability(probability, specialist, self.bundle)
        triage_actions, triage_predictions, triage_scores = target_triage_decisions(probability, self.bundle, self.target_policy, 'deploy_score_meta_scrum_other_ultra', profile='balanced', specialist_target_probability=specialist, score_meta_probability=score_meta)
        if progress_callback:
            progress_callback(92, 'inference', 'Clasificación terminada')
        label = str(labels[0])
        probs = probability[0].astype(float)
        prob_by_label = {'line': float(probs[0]), 'other': float(probs[1]), 'scrum': float(probs[2])}
        confidence = float(prob_by_label[label])
        duration = self._duration_seconds(path)
        specialist_score = float(specialist[0]) if specialist is not None else 0.0
        line_score = float(line_review_rank_score(probability, specialist, profile='high_precision')[0])
        scrum_score = float(scrum_review_rank_score(probability, specialist)[0])
        target_score = float(target_review_rank_score(probability)[0])
        triage_action = str(triage_actions[0])
        triage_prediction = str(triage_predictions[0])
        triage_score = float(triage_scores[0])
        base_detection = {'start_time': 0.0, 'end_time': duration, 'duration': duration, 'play_type': label, 'confidence': confidence, 'probabilities': prob_by_label, 'all_probabilities': prob_by_label, 'two_stream': {'descriptor_source': descriptor_source, 'target_specialist_probability': specialist_score, 'target_review_score': target_score, 'line_precision_review_score': line_score, 'scrum_precision_review_score': scrum_score, 'target_triage_action': triage_action, 'target_triage_prediction': triage_prediction, 'target_triage_score': triage_score}}
        if self.mode == TWO_STREAM_PRECISION_WORKFLOW_NAME:
            exact_task = 'line' if line_score >= scrum_score else 'scrum'
            exact_score = max(line_score, scrum_score)
            detection = {**base_detection, 'play_type': exact_task, 'confidence': exact_score, 'review_workflow': 'precision_exact_line1_scrum7', 'review_note': 'Single-video score shown for review. The validated 0.90+ precision workflow is a batch-ranked queue: line top 1% plus scrum top 7%.'}
            detections = [detection] if exact_score >= confidence_threshold else []
            summary = {'total_detections': len(detections), 'plays_by_type': {exact_task: len(detections)} if detections else {}, 'average_confidence': exact_score if detections else 0.0, 'predicted_class': exact_task, 'overall_confidence': exact_score, 'review_workflow': 'precision_exact_line1_scrum7', 'validated_scope': 'batch-ranked review queue, not full automatic classification'}
        else:
            detections = [base_detection] if confidence >= confidence_threshold else []
            summary = {'total_detections': len(detections), 'plays_by_type': {label: len(detections)} if detections else {}, 'average_confidence': confidence if detections else 0.0, 'predicted_class': label, 'overall_confidence': confidence, 'validated_scope': 'full 3-class classifier on one clip/video'}
        return {'status': 'success', 'video_path': str(path), 'model_name': self.display_name, 'detections': detections, 'review_detections_7s': detections if self.mode == TWO_STREAM_PRECISION_WORKFLOW_NAME else [], 'window_predictions': [], 'statistics': self._statistics(detections), 'summary': summary, 'metadata': {'duration': duration, 'confidence_threshold': confidence_threshold, 'analysis_mode': 'two_stream_descriptor', 'descriptor_source': descriptor_source, 'model_key': self.mode}}

    def analyze_video_segments(self, video_path: str, **kwargs) -> Dict:
        return self.analyze_video(video_path, **kwargs)

    def _statistics(self, detections: List[Dict]) -> Dict:
        if not detections:
            return {'avg_confidence': 0.0, 'avg_duration': 0.0, 'by_type': {}}
        by_type = {}
        for detection in detections:
            play_type = detection['play_type']
            by_type.setdefault(play_type, {'count': 0, 'total_duration': 0.0})
            by_type[play_type]['count'] += 1
            by_type[play_type]['total_duration'] += float(detection.get('duration', 0.0))
        return {'avg_confidence': float(sum((det['confidence'] for det in detections)) / len(detections)), 'avg_duration': float(sum((det.get('duration', 0.0) for det in detections)) / len(detections)), 'by_type': by_type}

class TunedEnsembleSequenceModel:
    """Keras-like sequence model wrapper that averages tuned no-play members."""

    def __init__(self, member_paths: List[Path], label_vocab: np.ndarray, class_bias: Dict[str, float]):
        if not member_paths:
            raise ValueError('At least one ensemble member is required.')
        self.member_paths = [Path(path) for path in member_paths]
        self.label_vocab = np.array(label_vocab)
        self.models = [keras.models.load_model(_resolve_model_artifact(path)) for path in self.member_paths]
        self.input_shape = self.models[0].input_shape
        self.output_shape = self.models[0].output_shape
        self.class_bias = np.array([float(class_bias.get(str(label), 1.0)) for label in self.label_vocab], dtype=np.float32)

    def predict(self, inputs, verbose: int=0, **kwargs):
        predictions = [np.asarray(model.predict(inputs, verbose=verbose, **kwargs), dtype=np.float32) for model in self.models]
        averaged = np.mean(predictions, axis=0)
        tuned = averaged * self.class_bias.reshape((1, -1))
        row_sums = tuned.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        return tuned / row_sums

class TunedNoPlayEnsembleDetector:
    """
    Adapter that reuses RugbyPlayDetector preprocessing while swapping in a
    tuned ensemble sequence model.
    """

    def __init__(self, model_dir: Path, ensemble_root: Path, config: Dict):
        self.model_dir = Path(model_dir)
        self.ensemble_root = Path(ensemble_root)
        self.config = config
        member_paths = [self.ensemble_root / member / 'run_01_seed_20260630' / 'model.keras' for member in config['members']]
        missing = [str(path) for path in member_paths if not _resolve_model_artifact(path).exists()]
        if missing:
            raise FileNotFoundError(f'Missing no-play ensemble members: {missing}')
        label_vocab_path = member_paths[0].parent / 'label_vocab.npy'
        if not _resolve_model_artifact(label_vocab_path).exists():
            raise FileNotFoundError(f'Missing label vocabulary: {label_vocab_path}')
        label_vocab = _load_model_result_array(label_vocab_path, allow_pickle=True)
        base_model_name = 'best_model/best_model.keras'
        if not _resolve_model_artifact(self.model_dir / 'checkpoints' / base_model_name).exists():
            base_model_name = None
        self.base_detector = RugbyPlayDetector(str(self.model_dir), model_name=base_model_name, extractor_name='resnet50')
        self.sequence_model = TunedEnsembleSequenceModel(member_paths=member_paths, label_vocab=label_vocab, class_bias=config['class_bias'])
        self.label_vocab = label_vocab
        self.feature_extractor = self.base_detector.feature_extractor
        self.base_detector.sequence_model = self.sequence_model
        self.base_detector.label_vocab = self.label_vocab

    def __getattr__(self, name):
        return getattr(self.base_detector, name)

from models.M1.inference import PlayerGeometryFusionDetector, M1H2D2CrossFoldDetector


class HistoricalYoloResidualDetector(PlayerGeometryFusionDetector):
    """Historical ResNet-50 + BiGRU with gated 200-frame YOLO corrections.

    The historical prediction is returned unchanged when its maximum
    probability is at least 0.85. Less certain windows receive a 50% residual
    correction from three validation-selected geometry experts.
    """
    GEOMETRY_FRAMES = 200

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.experiment_root = self.project_root / HISTORICAL_YOLO_RESIDUAL_ROOT_RELATIVE
        self.checkpoint_path = self.project_root / HISTORICAL_YOLO_RESIDUAL_CHECKPOINT_RELATIVE
        self.pose_model_path = self.project_root / PLAYER_GEOMETRY_FUSION_POSE_MODEL
        result_path = self.experiment_root / 'ensemble_three_seeds' / 'ensemble_result.json'
        required = [self.checkpoint_path, self.pose_model_path, result_path]
        required.extend((self.experiment_root / f'results_seed{seed}' / 'selected_model.joblib' for seed in (17, 42, 2026)))
        missing = [path for path in required if not _resolve_model_artifact(path).exists()]
        if missing:
            raise FileNotFoundError(f'Missing historical-YOLO residual artifacts: {missing}')
        model_dir = self.project_root / 'models' / 'M0'
        RugbyPlayDetector.__init__(self, str(model_dir), model_name=str(self.checkpoint_path), extractor_name='resnet50')
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
        ensemble_result = json.loads(_resolve_model_artifact(result_path).read_text(encoding='utf-8'))
        self.ensemble_weights = np.asarray(ensemble_result['selected_weights_by_validation'], dtype=np.float64)
        self.residual_models = []
        for seed in (17, 42, 2026):
            root = self.experiment_root / f'results_seed{seed}'
            result = json.loads(_resolve_model_artifact(root / 'selected_result.json').read_text(encoding='utf-8'))
            configuration = result['selected_configuration']
            artifact = joblib.load(_resolve_model_artifact(root / 'selected_model.joblib'))
            self.residual_models.append({'seed': seed, 'configuration': configuration, 'scaler': artifact['scaler'], 'pca': artifact['pca'], 'model': artifact['model']})
        try:
            yolo_config_dir = _temporary_model_settings()
            yolo_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault('YOLO_CONFIG_DIR', str(yolo_config_dir))
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError('El modelo residual YOLO requiere ultralytics.') from exc
        self.pose_detector = YOLO(_resolve_model_artifact(str(self.pose_model_path)))
        self.label_vocab = np.asarray(['line', 'other', 'scrum'], dtype=object)
        self._geometry_frames = []
        self._raw_geometry = None

    @property
    def display_name(self) -> str:
        return 'ResNet-50 + BiGRU histórico + corrección YOLO residual'

    def _begin_auxiliary_sequence(self, sampled_total_frames: int) -> None:
        self._geometry_frames = []
        self._raw_geometry = None

    def _on_sampled_frame(self, frame_bgr: np.ndarray) -> None:
        pose_frame = cv2.resize(frame_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        self._geometry_frames.append(self._detect_frame(self.pose_detector, pose_frame, image_size=640, person_threshold=self.POSE_PERSON_CONFIDENCE, max_people=self.MAX_PEOPLE))

    def _finish_auxiliary_sequence(self) -> None:
        if not self._geometry_frames:
            self._raw_geometry = None
            return
        boxes = np.asarray([item[0] for item in self._geometry_frames], dtype=np.float32)
        person_conf = np.asarray([item[1] for item in self._geometry_frames], dtype=np.float32)
        keypoints = np.asarray([item[2] for item in self._geometry_frames], dtype=np.float32)
        keypoint_conf = np.asarray([item[3] for item in self._geometry_frames], dtype=np.float32)
        valid = np.asarray([item[4] for item in self._geometry_frames], dtype=bool)
        positions, poses, frame_mask = self._build_geometry_features(boxes, person_conf, keypoints, keypoint_conf, valid, person_threshold=self.POSE_PERSON_CONFIDENCE, keypoint_threshold=self.POSE_KEYPOINT_CONFIDENCE)
        self._raw_geometry = {'boxes': boxes, 'person_conf': person_conf, 'keypoints': keypoints, 'keypoint_conf': keypoint_conf, 'valid': valid, 'position_features': positions, 'pose_features': poses, 'frame_mask': frame_mask}

    @staticmethod
    def _quality_channels(data: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        valid = data['valid'].astype(bool)
        person_conf = data['person_conf'].astype(np.float32)
        keypoint_conf = data['keypoint_conf'].astype(np.float32)
        count = valid.sum(axis=1).astype(np.float32)
        person_mean = np.divide(np.where(valid, person_conf, 0.0).sum(axis=1), np.maximum(count, 1.0))
        keypoint_mean = np.divide(np.where(valid[..., None], keypoint_conf, 0.0).sum(axis=(1, 2)), np.maximum(count * keypoint_conf.shape[-1], 1.0))
        coverage = np.clip(count / 30.0, 0.0, 1.0)
        quality = np.column_stack([count / 30.0, person_mean, keypoint_mean, coverage]).astype(np.float32)
        frame_ok = (person_mean >= 0.25) & (keypoint_mean >= 0.2) & (count >= 1) & data['frame_mask'].astype(bool)
        return (quality, frame_ok)

    @staticmethod
    def _summarize(sequence: np.ndarray, mask: np.ndarray, temporal_extra: bool=False) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        valid = sequence[mask] if np.any(mask) else sequence
        if len(valid) == 0:
            valid = np.zeros((1, sequence.shape[-1]), dtype=np.float32)
        valid = np.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
        deltas = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, sequence.shape[-1]), dtype=np.float32)
        parts = [valid.mean(axis=0), valid.std(axis=0), valid.min(axis=0), valid.max(axis=0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), deltas.mean(axis=0), deltas.std(axis=0), np.abs(deltas).mean(axis=0)]
        if temporal_extra:
            if len(valid) > 1:
                time_axis = np.linspace(-1.0, 1.0, len(valid), dtype=np.float32)
                centered_time = time_axis - time_axis.mean()
                denominator = float(np.sum(centered_time * centered_time)) + 1e-06
                slope = np.sum(centered_time[:, None] * (valid - valid.mean(axis=0)[None, :]), axis=0) / denominator
                first_last = valid[-1] - valid[0]
            else:
                slope = np.zeros(sequence.shape[-1], dtype=np.float32)
                first_last = np.zeros(sequence.shape[-1], dtype=np.float32)
            parts.extend([slope, first_last])
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _robust_smooth(sequence: np.ndarray, valid: np.ndarray) -> np.ndarray:
        values = np.asarray(sequence, dtype=np.float32).copy()
        if values.shape[0] < 3:
            return values
        padded = np.pad(values, ((1, 1), (0, 0)), mode='edge')
        smoothed = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]], axis=0), axis=0)
        smoothed[~np.asarray(valid, dtype=bool)] = values[~np.asarray(valid, dtype=bool)]
        return smoothed.astype(np.float32)

    def _representation_vector(self, representation: str, start_frame: int, end_frame: int) -> np.ndarray:
        if self._raw_geometry is None:
            dimensions = 585 if representation == 'E02_quality_mask' else 707
            return np.zeros(dimensions, dtype=np.float32)
        data = {key: value[start_frame:end_frame] for key, value in self._raw_geometry.items()}
        positions = data['position_features'].astype(np.float32)
        poses = data['pose_features'].astype(np.float32)
        mask = data['frame_mask'].astype(bool)
        quality, quality_mask = self._quality_channels(data)
        current = np.concatenate([positions, poses], axis=1)
        if representation == 'E02_quality_mask':
            return np.concatenate([self._summarize(current, mask & quality_mask), self._summarize(quality, mask)])
        if representation == 'E04_tracking_smoothing':
            smoothed = self._robust_smooth(current, mask)
            return np.concatenate([self._summarize(smoothed, mask, temporal_extra=True), self._summarize(quality, mask)])
        raise ValueError(f'Unsupported residual representation: {representation}')

    def _predict_sequence_window(self, padded_features: np.ndarray, mask: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
        historical = RugbyPlayDetector._predict_sequence_window(self, padded_features, mask, start_frame, end_frame)[0]
        historical = self._normalize(historical)
        corrected = []
        for item in self.residual_models:
            configuration = item['configuration']
            vector = self._representation_vector(configuration['representation'], start_frame, end_frame).reshape(1, -1)
            transformed = item['pca'].transform(item['scaler'].transform(vector))
            geometry = item['model'].predict_proba(transformed)[0]
            if float(np.max(historical)) < float(configuration['threshold']):
                alpha = float(configuration['alpha'])
                probability = (1.0 - alpha) * historical + alpha * geometry
            else:
                probability = historical
            corrected.append(self._normalize(probability))
        ensemble = np.average(np.asarray(corrected, dtype=np.float64), axis=0, weights=self.ensemble_weights)
        return self._normalize(ensemble)[None, :].astype(np.float32)

class KerasInferenceEngine:
    """Inference engine for Keras-based rugby play detection models."""

    def __init__(self, models_dir: Path=None):
        """
        Initialize the Keras inference engine.
        
        Args:
            models_dir: Base directory containing model subdirectories
        """
        if not ADAPTER_AVAILABLE:
            raise ImportError(f'RugbyPlayDetector not available. Error: {import_error}')
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / 'models'
        self.models_dir = models_dir
        self.current_model = None
        self.current_model_name = None
        self.current_model_type = None
        logger.info(f'KerasInferenceEngine initialized. Models dir: {self.models_dir}')

    def clear_model(self) -> None:
        """Clear the active model after a failed or invalid load.

        The web worker is long-lived and the engine is shared by requests.  A
        failed load must therefore not leave the previously selected model
        available for a subsequent inference request, otherwise the result
        could be produced by a different model than the one shown in the UI.
        """
        self.current_model = None
        self.current_model_name = None
        self.current_model_type = None

    def list_available_models(self) -> List[Dict]:
        """
        List all available trained Keras models.
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        meta_artifact = self.models_dir.parent / META_CORRECTION_ARTIFACT_RELATIVE
        meta_primary = self.models_dir.parent / META_CORRECTION_PRIMARY_RELATIVE
        meta_metrics = self.models_dir.parent / META_CORRECTION_METRICS_RELATIVE
        if _resolve_model_artifact(meta_artifact).exists() and _resolve_model_artifact(meta_primary).exists():
            metrics = {}
            try:
                if _resolve_model_artifact(meta_metrics).exists():
                    with _resolve_model_artifact(meta_metrics).open('r', encoding='utf-8') as handle:
                        metrics = json.load(handle).get('metrics', {})
            except Exception as exc:
                logger.warning(f'Could not parse meta-corrector metrics: {exc}')
            models.append({'name': META_CORRECTION_MODEL_NAME, 'display_name': 'CNN-RNN + Corrector few-shot (mismo video)', 'path': str(meta_artifact), 'model_type': 'meta_correction', 'architecture': 'primary_cnn_rnn_plus_balanced_logistic_regression', 'size_mb': round(_resolve_model_artifact(meta_artifact).stat().st_size / (1024 * 1024), 2), 'framework': 'Keras/TensorFlow + scikit-learn', 'description': 'Ejecuta el CNN-RNN histórico y corrige sus probabilidades con un modelo calibrado sobre 401 clips etiquetados del mismo video. Requiere recalibración humana para otro partido.', 'metrics': {'test_accuracy': metrics.get('accuracy'), 'f1_macro': metrics.get('f1_macro'), 'f1_weighted': metrics.get('f1_weighted'), 'recall_line': metrics.get('recall_line'), 'precision_line': metrics.get('precision_line'), 'f1_line': metrics.get('f1_line')}, 'experimental': True, 'requires_same_video_support': True, 'support_rows': 401})
        clahe_model = self.models_dir / CLAHE_WEB_MODEL_CHECKPOINT_RELATIVE
        if _resolve_model_artifact(clahe_model).exists():
            models.insert(0, {'name': CLAHE_WEB_MODEL_NAME, 'display_name': 'CLAHE · Recall experimental (macro-F1)', 'path': str(clahe_model), 'model_type': 'cnn_rnn_clahe', 'architecture': 'resnet50_bigru_clahe_line_weight_1_25', 'size_mb': round(_resolve_model_artifact(clahe_model).stat().st_size / (1024 * 1024), 2), 'framework': 'Keras/TensorFlow', 'description': 'Variante experimental: CLAHE después del recorte central y la escala 0.50, 25 FPS, line_weight=1.25, checkpoint seleccionado por macro-F1. Accuracy=85.43%, F1 macro=83.20%, recall line=83.06%, scrum=91.11%.', 'metrics': {'test_accuracy': 0.8543451652386781, 'f1_macro': 0.8320385867251314, 'recall_line': 0.8306010928961749, 'recall_scrum': 0.9111111111111111}, 'experimental': True, 'preprocessing_filter': 'clahe'})
        quantile_root = self.models_dir.parent / RGB_YOLO_QUANTILE_FUSION_ROOT_RELATIVE
        quantile_deployment = self.models_dir.parent / RGB_YOLO_QUANTILE_DEPLOYMENT_RELATIVE
        quantile_feature_root = self.models_dir.parent / RGB_YOLO_QUANTILE_FEATURE_RELATIVE
        quantile_required = [quantile_deployment / 'quantile_scaler.joblib', quantile_deployment / 'logistic.joblib', quantile_feature_root / 'dynamic_pca.joblib', self.models_dir.parent / PLAYER_GEOMETRY_FUSION_POSE_MODEL, self.models_dir / 'M0' / 'checkpoints' / 'best_model' / 'best_model.keras']
        if all((_resolve_model_artifact(path).exists() for path in quantile_required)):
            total_size = sum((_resolve_model_artifact(path).stat().st_size for path in quantile_required))
            models.insert(0, {'name': RGB_YOLO_QUANTILE_FUSION_MODEL_NAME, 'display_name': 'Nuevo modelo RGB + YOLO (fusion cuantica)', 'path': str(quantile_deployment), 'model_type': 'rgb_yolo_quantile_fusion', 'architecture': 'frozen_resnet50_bigru_plus_rgb_dynamics_yolo100_e04_e06_quantile_logreg', 'size_mb': round(total_size / (1024 * 1024), 2), 'framework': 'Keras/TensorFlow + Ultralytics + scikit-learn', 'description': 'Modelo recomendado para analizar videos completos. Mantiene el historico ResNet-50 + BiGRU y agrega 704 variables RGB, 707 YOLO E04 y 275 YOLO E06; mezcla 40% historico + 60% correccion YOLO. Usa 100 muestras YOLO por ventana de 7 s.', 'metrics': {'test_accuracy': 0.8392944634982852, 'accuracy': 0.8392944634982852, 'f1_macro': 0.8123015138995129, 'external_grouped_delta_accuracy': 0.0906418422341989, 'natural_test_accuracy': 0.8788249694002448, 'reference_200_accuracy': 0.8407643312101911, 'reference_200_f1_macro': 0.8140962271208876}, 'experimental': False, 'recommended': True, 'requires_pose_model': True, 'pose_model': PLAYER_GEOMETRY_FUSION_POSE_MODEL, 'feature_dim': 1686, 'yolo_frames_per_window': 100, 'yolo_sampling': 'uniforme, 100 frames por ventana de 7 segundos', 'historical_weight': 0.4, 'yolo_weight': 0.6})
        highres_deployment = self.models_dir.parent / RGB_YOLO_HIGHRES_DEPLOYMENT_RELATIVE
        highres_summary_path = highres_deployment / 'summary.json'
        highres_required = [highres_deployment / 'quantile_scaler.joblib', highres_deployment / 'logistic.joblib', quantile_feature_root / 'dynamic_pca.joblib', self.models_dir.parent / PLAYER_GEOMETRY_FUSION_POSE_MODEL, self.models_dir / 'M0' / 'checkpoints' / 'best_model' / 'best_model.keras']
        highres_summary = {}
        if all((_resolve_model_artifact(path).exists() for path in highres_required)):
            try:
                highres_summary = json.loads(_resolve_model_artifact(highres_summary_path).read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                highres_summary = {}
            if bool(highres_summary.get('promoted')):
                for listed_model in models:
                    if listed_model.get('name') == RGB_YOLO_QUANTILE_FUSION_MODEL_NAME:
                        listed_model['recommended'] = False
            metrics = highres_summary.get('external_metrics') or {}
            delta = highres_summary.get('external_delta_vs_lowres') or {}
            total_size = sum((_resolve_model_artifact(path).stat().st_size for path in highres_required if _resolve_model_artifact(path).is_file()))
            models.insert(0, {'name': RGB_YOLO_HIGHRES_MODEL_NAME, 'display_name': 'M2 · RGB + YOLO (ROI equivalente)', 'path': str(highres_deployment), 'model_type': 'rgb_yolo_quantile_highres', 'architecture': 'resnet50_bigru_rgb_dynamics_yolo100_highres_roi_quantile_logreg', 'size_mb': round(total_size / (1024 * 1024), 2), 'framework': 'Keras/TensorFlow + Ultralytics + scikit-learn', 'description': 'Admite los clips 224×224 actuales y videos sin recortar; usa el mismo campo visual para RGB y YOLO.', 'metrics': {'accuracy': metrics.get('accuracy'), 'f1_macro': metrics.get('f1_macro'), 'delta_accuracy_vs_upsampled_224': delta.get('accuracy'), 'delta_f1_macro_vs_upsampled_224': delta.get('f1_macro')}, 'experimental': False, 'recommended': bool(highres_summary.get('promoted')), 'requires_pose_model': True, 'pose_model': PLAYER_GEOMETRY_FUSION_POSE_MODEL, 'feature_dim': 1686, 'yolo_frames_per_window': 100, 'pose_input_mode': 'source_equivalent_roi', 'historical_weight': 0.4, 'yolo_weight': 0.6})
        event_verifier_path = self.models_dir.parent / RGB_YOLO_EVENT_VERIFIER_RELATIVE
        temporal_models_root = self.models_dir.parent / RGB_YOLO_TEMPORAL_MODELS_RELATIVE
        event_verifier_required = highres_required + [event_verifier_path, *(temporal_models_root / f'seed_{seed}.pt' for seed in (42, 17, 2026))]
        if all((_resolve_model_artifact(path).exists() for path in event_verifier_required)) and bool(highres_summary.get('promoted')):
            total_size = sum((_resolve_model_artifact(path).stat().st_size for path in event_verifier_required if _resolve_model_artifact(path).is_file()))
            models.insert(0, {'name': RGB_YOLO_EVENT_VERIFIER_MODEL_NAME, 'display_name': 'Mejor candidato · RGB + YOLO temporal + verificador de line', 'path': str(event_verifier_path), 'model_type': 'rgb_yolo_temporal_event_verifier', 'architecture': 'highres_40_60_plus_tcn_bigru_attention_event_logreg_c0_1', 'size_mb': round(total_size / (1024 * 1024), 2), 'framework': 'Keras/TensorFlow + Ultralytics + PyTorch + scikit-learn', 'description': 'Mantiene el modelo RGB + YOLO high-res para clasificar y agrega tres modelos temporales TCN+BiGRU con atención. Después verifica cada evento completo de line con una cabeza logística de 208 variables. En desarrollo conservó 23/23 lines y redujo los falsos positivos de 54 a 23.', 'metrics': {'development_event_recall': 1.0, 'development_event_precision': 0.5, 'development_false_positives_per_hour': 23.1, 'development_false_positive_reduction': 0.574}, 'experimental': True, 'recommended': False, 'validation_status': 'pending_different_match_validation', 'requires_pose_model': True, 'compatible_live': False, 'pose_model': PLAYER_GEOMETRY_FUSION_POSE_MODEL, 'feature_dim': 1686, 'line_verifier_feature_dim': 208, 'line_verifier_threshold': 0.11, 'line_verifier_grid_seconds': 7.0, 'yolo_frames_per_window': 100, 'pose_input_mode': 'source_equivalent_roi', 'historical_weight': 0.4, 'static_yolo_weight': 0.6, 'temporal_role': 'event verification only'})
        primera_historical_path = self.models_dir.parent / RGB_YOLO_PRIMERA_HISTORICAL_RELATIVE
        primera_verifier_path = self.models_dir.parent / RGB_YOLO_PRIMERA_VERIFIER_RELATIVE
        primera_required = [*event_verifier_required, primera_historical_path, primera_verifier_path]
        if all((_resolve_model_artifact(path).exists() for path in primera_required)) and bool(highres_summary.get('promoted')):
            for listed_model in models:
                if listed_model.get('name') in {RGB_YOLO_QUANTILE_FUSION_MODEL_NAME, RGB_YOLO_HIGHRES_MODEL_NAME, RGB_YOLO_EVENT_VERIFIER_MODEL_NAME}:
                    listed_model['recommended'] = False
            total_size = sum((_resolve_model_artifact(path).stat().st_size for path in primera_required if _resolve_model_artifact(path).is_file()))
            models.insert(0, {'name': RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME, 'display_name': 'Mejor modelo · Histórico Primera + RGB/YOLO temporal calibrado', 'path': str(primera_verifier_path), 'model_type': 'rgb_yolo_historical_primera_calibrated', 'architecture': 'primera_finetune_highres_40_60_tcn_bigru_line55_45_event_logreg', 'size_mb': round(total_size / (1024 * 1024), 2), 'framework': 'Keras/TensorFlow + Ultralytics + PyTorch + scikit-learn', 'description': 'Usa el Histórico afinado con las revisiones de Primera, mantiene 40/60 para scrum y calibra line con 55% Histórico, 45% RGB/YOLO y evidencia temporal. En el tramo 40–60 no usado para ajuste conservó 12/13 eventos y redujo los falsos positivos de 12 a 7.', 'metrics': {'holdout_event_precision': 0.631578947368421, 'holdout_event_recall': 0.9230769230769231, 'holdout_event_f1': 0.75, 'holdout_false_positives': 7, 'baseline_event_f1': 0.6486486486486487, 'baseline_false_positives': 12}, 'experimental': False, 'recommended': True, 'validation_status': 'untouched_same_match_40_60_holdout', 'requires_pose_model': True, 'compatible_live': False, 'pose_model': PLAYER_GEOMETRY_FUSION_POSE_MODEL, 'feature_dim': 1686, 'line_verifier_feature_dim': 208, 'line_verifier_threshold': 0.4101333272622148, 'line_verifier_grid_seconds': 7.0, 'yolo_frames_per_window': 100, 'pose_input_mode': 'source_equivalent_roi', 'historical_model': RGB_YOLO_PRIMERA_HISTORICAL_RELATIVE, 'scrum_historical_weight': 0.4, 'scrum_static_weight': 0.6, 'line_historical_weight': 0.55, 'line_static_weight': 0.45, 'temporal_role': 'line event verification only'})
        geometry_export = self.models_dir.parent / PLAYER_GEOMETRY_FUSION_EXPORT_RELATIVE
        geometry_manifest = geometry_export / 'manifest.json'
        geometry_pose_checkpoint = self.models_dir.parent / PLAYER_GEOMETRY_FUSION_POSE_MODEL
        if _resolve_model_artifact(geometry_manifest).exists() and _resolve_model_artifact(geometry_pose_checkpoint).exists():
            try:
                manifest = json.loads(_resolve_model_artifact(geometry_manifest).read_text(encoding='utf-8'))
                seed_checkpoints = []
                for run in manifest.get('runs', []):
                    seed = int(run['seed'])
                    rgb_checkpoint = self.models_dir.parent / 'data' / 'player_geometry_fusion' / 'results' / 'm0' / f'seed_{seed}' / 'best.weights.h5'
                    pose_checkpoint = self.models_dir.parent / 'data' / 'player_geometry_fusion' / 'results' / 'same_rnn' / 'pose_concat' / f'seed_{seed}' / 'best.weights.h5'
                    rf_checkpoint = geometry_export / run['rf_path']
                    seed_checkpoints.extend([rgb_checkpoint, pose_checkpoint, rf_checkpoint])
                if seed_checkpoints and all((_resolve_model_artifact(path).exists() for path in seed_checkpoints)):
                    total_size = sum((_resolve_model_artifact(path).stat().st_size for path in seed_checkpoints)) + _resolve_model_artifact(geometry_pose_checkpoint).stat().st_size
                    models.insert(0, {'name': PLAYER_GEOMETRY_FUSION_MODEL_NAME, 'display_name': '⭐ RGB + Pose + Geometría (ensemble calibrado)', 'path': str(geometry_manifest), 'model_type': 'player_geometry_fusion', 'architecture': 'resnet50_bigru_pose_rf_geometric_mean', 'size_mb': round(total_size / (1024 * 1024), 2), 'framework': 'Keras/TensorFlow + Ultralytics + scikit-learn', 'description': 'Modelo recomendado de la tesina. Combina ResNet-50 + BiGRU, descriptores de postura YOLO11n-pose y estadísticas temporales de posiciones con media geométrica y calibración fija de clases.', 'metrics': {'test_accuracy': 0.8368013055895552, 'f1_macro': 0.8091761667843139, 'f1_weighted': 0.8396390568353388, 'bootstrap_accuracy_delta': 0.03840677274581804, 'bootstrap_f1_macro_delta': 0.025245557222127533}, 'experimental': False, 'requires_pose_model': True, 'pose_model': PLAYER_GEOMETRY_FUSION_POSE_MODEL})
            except Exception as exc:
                logger.warning(f'Could not list geometry-fusion model: {exc}')
        m1_h2_export = self.models_dir.parent / M1_H2_D2_WEB_EXPORT_RELATIVE
        m1_h2_manifest_path = m1_h2_export / 'manifest.json'
        if _resolve_model_artifact(m1_h2_manifest_path).exists() and _resolve_model_artifact(geometry_pose_checkpoint).exists():
            try:
                manifest = json.loads(_resolve_model_artifact(m1_h2_manifest_path).read_text(encoding='utf-8'))
                fold_artifacts = []
                for fold in manifest.get('folds', []):
                    fold_artifacts.extend([self.models_dir.parent / fold['dual_model'], self.models_dir.parent / fold['geometry_rf']])
                if len(manifest.get('folds', [])) == 6 and all((_resolve_model_artifact(path).exists() for path in fold_artifacts)):
                    metrics = manifest['cross_validation_metrics']
                    thresholds = manifest['recommended_web_thresholds']
                    total_size = sum((_resolve_model_artifact(path).stat().st_size for path in fold_artifacts))
                    total_size += _resolve_model_artifact(geometry_pose_checkpoint).stat().st_size
                    models.insert(0, {'name': M1_H2_D2_WEB_MODEL_NAME, 'display_name': 'M1-H2-D2 · ensemble de 6 folds', 'path': str(m1_h2_manifest_path), 'model_type': 'm1_h2_d2_cross_fold', 'architecture': 'resnet50_dual_bigru_pose_geometry_rf_six_fold_soft_vote', 'size_mb': round(total_size / (1024 * 1024), 2), 'framework': 'Keras/TensorFlow + Ultralytics + scikit-learn', 'description': 'Combina ResNet-50, una BiGRU RGB, una BiGRU de pose y un Random Forest geométrico. Para videos nuevos promedia los seis modelos externos de M1-H2-D2.', 'metrics': {'f1w_temporal': metrics['f1w_macro'], 'f1w_line': metrics['f1w_line'], 'f1w_scrum': metrics['f1w_scrum'], 'fp_far_hour': metrics['fp_far_hour']}, 'experimental': True, 'validation_status': manifest['validation_status'], 'requires_pose_model': True, 'compatible_live': False, 'pose_model': PLAYER_GEOMETRY_FUSION_POSE_MODEL, 'fold_count': 6, 'comparison_rank': 1, 'analysis_defaults': {'confidence_threshold': thresholds['scrum'], 'line_confidence_threshold': thresholds['line']}})
            except Exception as exc:
                logger.warning(f'Could not list M1-H2-D2 web ensemble: {exc}')
        for m3_name, export_relative in M3_WEB_EXPORTS.items():
            manifest_path = self.models_dir.parent / export_relative / 'manifest.json'
            if not _resolve_model_artifact(manifest_path).exists() or not _resolve_model_artifact(geometry_pose_checkpoint).exists():
                continue
            try:
                manifest = json.loads(_resolve_model_artifact(manifest_path).read_text(encoding='utf-8'))
                artifact_keys = ('dynamic_pca', 'quantile_scaler', 'logistic', 'historical_model', 'temporal_model', 'line_verifier')
                artifact_paths = {self.models_dir.parent / fold[key] for fold in manifest.get('folds', []) for key in artifact_keys}
                if len(manifest.get('folds', [])) != 6 or not all((_resolve_model_artifact(path).exists() for path in artifact_paths)):
                    continue
                metrics = manifest['cross_validation_metrics']
                thresholds = manifest['recommended_web_thresholds']
                source = manifest['source_configuration']
                models.insert(0, {'name': m3_name, 'display_name': f"{source.replace('PV-', '')} · ensemble de 6 folds", 'path': str(manifest_path), 'model_type': 'm3_cross_fold', 'architecture': 'resnet50_bigru_rgb_yolo_highres_tcn_bigru_attention_fold_verifier_six_fold_soft_vote', 'size_mb': round((sum((_resolve_model_artifact(path).stat().st_size for path in artifact_paths)) + _resolve_model_artifact(geometry_pose_checkpoint).stat().st_size) / (1024 * 1024), 2), 'framework': 'Keras/TensorFlow + PyTorch + Ultralytics + scikit-learn', 'description': f"Puesto {manifest['comparison_rank']} de Panel V. Extrae RGB y pose una vez y promedia los seis folds; los eventos de line requieren mayoría de sus verificadores.", 'metrics': {'f1w_temporal': metrics['f1w_macro'], 'f1w_line': metrics['f1w_line'], 'f1w_scrum': metrics['f1w_scrum'], 'fp_far_hour': metrics['fp_far_hour']}, 'experimental': True, 'validation_status': manifest['validation_status'], 'requires_pose_model': True, 'compatible_live': False, 'pose_model': PLAYER_GEOMETRY_FUSION_POSE_MODEL, 'fold_count': 6, 'comparison_rank': manifest['comparison_rank'], 'analysis_defaults': {'confidence_threshold': thresholds['scrum'], 'line_confidence_threshold': thresholds['line']}})
            except Exception as exc:
                logger.warning(f'Could not list {m3_name} web ensemble: {exc}')
        residual_root = self.models_dir.parent / HISTORICAL_YOLO_RESIDUAL_ROOT_RELATIVE
        residual_result_path = residual_root / 'ensemble_three_seeds' / 'ensemble_result.json'
        residual_artifacts = [residual_root / f'results_seed{seed}' / 'selected_model.joblib' for seed in (17, 42, 2026)]
        residual_checkpoint = self.models_dir.parent / HISTORICAL_YOLO_RESIDUAL_CHECKPOINT_RELATIVE
        if _resolve_model_artifact(residual_result_path).exists() and _resolve_model_artifact(residual_checkpoint).exists() and _resolve_model_artifact(geometry_pose_checkpoint).exists() and all((_resolve_model_artifact(path).exists() for path in residual_artifacts)):
            try:
                result = json.loads(_resolve_model_artifact(residual_result_path).read_text(encoding='utf-8'))
                metrics = result['ensemble_test']
                bootstrap = result['paired_bootstrap']
                total_size = _resolve_model_artifact(residual_checkpoint).stat().st_size + _resolve_model_artifact(geometry_pose_checkpoint).stat().st_size
                total_size += sum((_resolve_model_artifact(path).stat().st_size for path in residual_artifacts))
                models.insert(0, {'name': HISTORICAL_YOLO_RESIDUAL_MODEL_NAME, 'display_name': 'Histórico + corrección YOLO residual (200 frames)', 'path': str(residual_result_path), 'model_type': 'historical_yolo_residual', 'architecture': 'resnet50_bigru_frozen_yolo_pca_logreg_confidence_gate', 'size_mb': round(total_size / (1024 * 1024), 2), 'framework': 'Keras/TensorFlow + Ultralytics + scikit-learn', 'description': 'Conserva el mejor ResNet-50 + BiGRU histórico y aplica una corrección YOLO de 200 frames únicamente cuando la confianza histórica es menor a 0.85. Los pesos del ensamble se eligieron sólo con validación.', 'metrics': {'test_accuracy': metrics['accuracy'], 'f1_macro': metrics['f1_macro'], 'f1_weighted': metrics['f1_weighted'], 'recall_line': metrics['recall_line'], 'recall_scrum': metrics['recall_scrum'], 'bootstrap_accuracy_probability_positive': bootstrap['accuracy_probability_positive'], 'bootstrap_macro_f1_probability_positive': bootstrap['macro_f1_probability_positive']}, 'experimental': True, 'requires_pose_model': True, 'pose_model': PLAYER_GEOMETRY_FUSION_POSE_MODEL})
            except Exception as exc:
                logger.warning(f'Could not list historical-YOLO residual model: {exc}')
        two_stream_bundle = self.models_dir.parent / 'data' / 'two_stream_feature_fusion' / 'results' / 'production_candidate' / 'two_stream_expanded_fusion_tree_ensemble_deduplicated.joblib'
        if _resolve_model_artifact(two_stream_bundle).exists():
            size_mb = _resolve_model_artifact(two_stream_bundle).stat().st_size / (1024 * 1024)
            models.extend([{'name': TWO_STREAM_PRECISION_WORKFLOW_NAME, 'display_name': 'Two-Stream Precision Review (line1 + scrum7)', 'path': str(two_stream_bundle), 'model_type': 'two_stream_review_workflow', 'architecture': 'two_stream_precision_exact_line1_scrum7', 'size_mb': round(size_mb, 2), 'framework': 'Two-Stream descriptors + sklearn ensemble', 'description': 'Review-first play detector. Validated as line top 1% plus scrum top 7%; 0.904 strict and 0.908 source-family review precision. Single videos show scores, while the validated mode is batch-ranked review.', 'metrics': {'accuracy': 0.9083, 'strict_review_precision': 0.9043, 'source_family_review_precision': 0.9083}}, {'name': TWO_STREAM_FULL_MODEL_NAME, 'display_name': 'Two-Stream Full Classifier (strict production)', 'path': str(two_stream_bundle), 'model_type': 'two_stream_full_classifier', 'architecture': 'expanded_fusion_tree_ensemble_deduplicated', 'size_mb': round(size_mb, 2), 'framework': 'Two-Stream descriptors + sklearn ensemble', 'description': 'Full 3-class line/other/scrum classifier using spatial CNN descriptors plus optical-flow statistics. Strict de-duplicated grouped accuracy is 76.1%.', 'metrics': {'accuracy': 0.7611, 'f1_macro': 0.7749, 'target_f1': 0.7851}}])
        imbalance_export_dir = self.models_dir.parent / 'data' / 'class_imbalance_comparison' / 'exports' / 'top2_models'
        for public_name, model_info in reversed(CLASS_IMBALANCE_EXPERIMENT_MODELS.items()):
            if model_info.get('project_relative_path'):
                model_file = self.models_dir.parent / model_info['project_relative_path']
            else:
                model_file = imbalance_export_dir / model_info['filename']
            if not _resolve_model_artifact(model_file).exists():
                continue
            try:
                size_mb = _resolve_model_artifact(model_file).stat().st_size / (1024 * 1024)
                models.insert(0, {'name': public_name, 'display_name': model_info['display_name'], 'path': str(model_file), 'model_type': 'cnn_rnn', 'architecture': 'resnet50_bigru_class_imbalance_experiment', 'size_mb': round(size_mb, 2), 'framework': 'Keras/TensorFlow', 'description': model_info['description'], 'metrics': model_info['metrics'], 'experimental': True})
            except Exception as e:
                logger.warning(f'Could not list class-imbalance model {public_name}: {e}')
        inception_gru_dir = self.models_dir / 'M0' / 'checkpoints'
        if _resolve_model_artifact(inception_gru_dir).exists():
            for model_file in _model_artifact_glob(inception_gru_dir, 'sequence_model_*.keras', recursive=False):
                try:
                    size_mb = _resolve_model_artifact(model_file).stat().st_size / (1024 * 1024)
                    model_name = model_file.stem.replace('sequence_model_', '')
                    models.append({'name': model_file.name, 'display_name': f'CNN-RNN ({model_name})', 'path': str(model_file), 'model_type': 'cnn_rnn', 'architecture': model_name, 'size_mb': round(size_mb, 2), 'framework': 'Keras/TensorFlow', 'description': f'Video classification with ResNet50 features + {model_name.upper()} sequence model'})
                except Exception as e:
                    logger.warning(f'Could not load metadata for {model_file.name}: {e}')
        best_model_file = self.models_dir / 'M0' / 'checkpoints' / 'best_model' / 'best_model.keras'
        if _resolve_model_artifact(best_model_file).exists():
            try:
                size_mb = _resolve_model_artifact(best_model_file).stat().st_size / (1024 * 1024)
                models.insert(0, {'name': 'best_model/best_model.keras', 'display_name': '⭐ Best Model (resnet50 · BiGRU · 81.3%)', 'path': str(best_model_file), 'model_type': 'cnn_rnn', 'architecture': 'best_bigru', 'size_mb': round(size_mb, 2), 'framework': 'Keras/TensorFlow', 'description': 'Best grid-search config: resnet50 features + Bidirectional GRU (64u) · dropout=0.3 · dense=64 · lr=1e-4 | Test acc=81.27% | Recall: line=89.1% other=77.6% scrum=87.8%', 'metrics': {'test_accuracy': 0.8127, 'f1_macro': 0.7973, 'recall_line': 0.891, 'recall_other': 0.776, 'recall_scrum': 0.878}})
            except Exception as e:
                logger.warning(f'Could not list best model: {e}')
        human_860_file = self.models_dir / 'M0' / 'checkpoints' / 'human_review_retrain_860' / 'best_model_human_review.keras'
        human_860_meta = human_860_file.parent / 'metadata.json'
        if _resolve_model_artifact(human_860_file).exists():
            try:
                metadata = {}
                if _resolve_model_artifact(human_860_meta).exists():
                    with open(_resolve_model_artifact(human_860_meta), 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                size_mb = _resolve_model_artifact(human_860_file).stat().st_size / (1024 * 1024)
                models.insert(0, {'name': 'human_review_retrain_860/best_model_human_review.keras', 'display_name': 'Human Review 860 (experimental)', 'path': str(human_860_file), 'model_type': 'cnn_rnn', 'architecture': 'best_bigru_human_review_860', 'size_mb': round(size_mb, 2), 'framework': 'Keras/TensorFlow', 'description': 'Best architecture retrained with 860 human-reviewed clips. More sensitive to line/scrum, lower other recall.', 'metrics': {'test_accuracy': metadata.get('test_accuracy'), 'f1_macro': metadata.get('f1_macro'), 'recall_line': metadata.get('recall_line'), 'recall_other': metadata.get('recall_other'), 'recall_scrum': metadata.get('recall_scrum')}})
            except Exception as e:
                logger.warning(f'Could not list human-review 860 model: {e}')
        focused_ckpt_dir = self.models_dir / 'M0' / 'checkpoints' / 'best_model_focused'
        focused_results_dir = self.models_dir / 'M0' / 'results' / 'best_model_focused'
        focused_metrics_path = focused_results_dir / 'best_focused_metrics.json'
        focused_metrics = {}
        if _resolve_model_artifact(focused_metrics_path).exists():
            try:
                with open(_resolve_model_artifact(focused_metrics_path), 'r', encoding='utf-8') as f:
                    focused_metrics = json.load(f)
            except Exception as e:
                logger.warning(f'Could not parse focused metrics JSON: {e}')
        if _resolve_model_artifact(focused_ckpt_dir).exists():
            for model_file in sorted(_model_artifact_glob(focused_ckpt_dir, 'best_focused__*.keras', recursive=False)):
                try:
                    size_mb = _resolve_model_artifact(model_file).stat().st_size / (1024 * 1024)
                    rel_name = f'best_model_focused/{model_file.name}'
                    test_acc = focused_metrics.get('test_accuracy')
                    f1_macro = focused_metrics.get('f1_macro')
                    source_trial = focused_metrics.get('source_trial')
                    metric_text = []
                    if test_acc is not None:
                        metric_text.append(f'test acc={float(test_acc) * 100:.2f}%')
                    if f1_macro is not None:
                        metric_text.append(f'f1_macro={float(f1_macro):.4f}')
                    if source_trial:
                        metric_text.append(f'source={source_trial}')
                    models.insert(0, {'name': rel_name, 'display_name': '🧪 Focused Best Model (stage-2)', 'path': str(model_file), 'model_type': 'cnn_rnn', 'architecture': 'best_focused', 'size_mb': round(size_mb, 2), 'framework': 'Keras/TensorFlow', 'description': 'Best retrained model from focused grid search stage-2' + (f" | {' · '.join(metric_text)}" if metric_text else ''), 'metrics': {'test_accuracy': test_acc, 'f1_macro': f1_macro, 'source_trial': source_trial, 'recall_line': focused_metrics.get('recall_line'), 'recall_other': focused_metrics.get('recall_other'), 'recall_scrum': focused_metrics.get('recall_scrum')}})
                except Exception as e:
                    logger.warning(f'Could not list focused model {model_file.name}: {e}')
        light_live_run = 'resnet50__lstm__frames20'
        light_live_ckpt = self.models_dir / 'M0' / 'checkpoints' / 'light_live' / light_live_run / 'model.keras'
        light_live_meta = light_live_ckpt.parent / 'metadata.json'
        if _resolve_model_artifact(light_live_ckpt).exists():
            try:
                metadata = {}
                if _resolve_model_artifact(light_live_meta).exists():
                    with open(_resolve_model_artifact(light_live_meta), 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                metrics = metadata.get('metrics', {})
                live_defaults = metadata.get('live_detection_alignment', {})
                size_mb = _resolve_model_artifact(light_live_ckpt).stat().st_size / (1024 * 1024)
                rel_name = f'light_live/{light_live_run}/model.keras'
                models.insert(0, {'name': rel_name, 'display_name': 'Light Live Best (resnet50 + LSTM + 20 frames)', 'path': str(light_live_ckpt), 'model_type': 'cnn_rnn', 'architecture': 'light_live_resnet50_lstm_frames20', 'size_mb': round(size_mb, 2), 'framework': 'Keras/TensorFlow', 'description': 'Best completed light-live model. For live detection use Target FPS=25, Window=7s, Sampled Frames=20.', 'metrics': {'test_accuracy': metrics.get('test_accuracy'), 'f1_macro': metrics.get('f1_macro'), 'recall_line': metrics.get('recall_line'), 'recall_other': metrics.get('recall_other'), 'recall_scrum': metrics.get('recall_scrum')}, 'live_defaults': {'target_fps': live_defaults.get('target_fps', 25), 'window_seconds': live_defaults.get('window_seconds', 7), 'sampled_frame_count': live_defaults.get('sampled_frame_count', 20)}})
            except Exception as e:
                logger.warning(f'Could not list light-live model: {e}')
        if ENSEMBLE_AVAILABLE:
            results_csv = self.models_dir / 'M0' / 'results' / 'grid_search_results.csv'
            grid_search_dir = self.models_dir / 'M0' / 'checkpoints' / 'grid_search'
            if _resolve_model_artifact(results_csv).exists() and _resolve_model_artifact(grid_search_dir).exists():
                for top_n in [3, 5]:
                    models.insert(0, {'name': f'ensemble_top{top_n}', 'display_name': f'Ensemble Top-{top_n} (soft voting)', 'path': f'ensemble_top{top_n}', 'model_type': 'ensemble', 'architecture': f'ensemble_top{top_n}', 'size_mb': 0, 'framework': 'Keras/TensorFlow (Ensemble)', 'description': f'Soft-voting ensemble of the top-{top_n} grid search models ranked by val_f1_macro. Averages softmax outputs across all members for more robust predictions.', 'metrics': {}})
        no_play_root = self.models_dir.parent / 'data' / 'no_play_best_model' / 'checkpoints' / 'folded_no_play_other'
        for ensemble_name, ensemble_config in reversed(NO_PLAY_TUNED_ENSEMBLES.items()):
            member_paths = [no_play_root / member / 'run_01_seed_20260630' / 'model.keras' for member in ensemble_config['members']]
            if not all((_resolve_model_artifact(path).exists() for path in member_paths)):
                continue
            size_mb = sum((_resolve_model_artifact(path).stat().st_size for path in member_paths)) / (1024 * 1024)
            models.insert(0, {'name': ensemble_name, 'display_name': ensemble_config['display_name'], 'path': ensemble_name, 'model_type': 'tuned_no_play_ensemble', 'architecture': ensemble_name, 'size_mb': round(size_mb, 2), 'framework': 'Keras/TensorFlow (Ensemble)', 'description': ensemble_config['description'], 'metrics': ensemble_config['metrics']})
        recommendation_profiles = {'best_model/best_model.keras': ('base_estable', 'Referencia estable ResNet-50 + BiGRU y menor costo que los ensembles.'), RGB_YOLO_HIGHRES_MODEL_NAME: ('ensemble_general', 'Ensemble general con YOLO sobre el ROI de resolución fuente.'), RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME: ('mejor_calibrado', 'Mejor combinación calibrada para eventos, con validación holdout.'), PLAYER_GEOMETRY_FUSION_MODEL_NAME: ('multimodal', 'Alternativa multimodal que combina RGB, pose y geometría temporal.'), 'light_live/resnet50__lstm__frames20/model.keras': ('liviano_cpu', 'Opción liviana para CPU y detección en vivo con 20 cuadros.')}
        for listed_model in models:
            profile = recommendation_profiles.get(listed_model.get('name'))
            if listed_model.get('name') == RGB_YOLO_HIGHRES_MODEL_NAME and not bool(highres_summary.get('promoted')):
                profile = None
            listed_model['recommended'] = profile is not None
            if profile is not None:
                listed_model['recommendation_category'] = profile[0]
                listed_model['recommendation_reason'] = profile[1]
        return models

    def load_model(self, model_path: str=None) -> None:
        """
        Load a trained Keras model for inference.
        
        Args:
            model_path: Path to model directory or checkpoint file.
                       If None, loads the default cnn_rnn model.
        """
        requested_name = str(model_path) if model_path else 'default'
        _norm = requested_name.lower().replace('_', '').replace('-', '').replace(' ', '')
        _cur_norm = (self.current_model_name or '').lower().replace('_', '').replace('-', '').replace(' ', '')
        if self.current_model is not None and self.current_model_name and (requested_name in self.current_model_name or _norm in _cur_norm or (requested_name == RGB_YOLO_QUANTILE_FUSION_MODEL_NAME and self.current_model_type == 'rgb_yolo_quantile_fusion') or (requested_name == RGB_YOLO_HIGHRES_MODEL_NAME and self.current_model_type == 'rgb_yolo_quantile_highres') or (requested_name == RGB_YOLO_EVENT_VERIFIER_MODEL_NAME and self.current_model_type == 'rgb_yolo_temporal_event_verifier') or (requested_name == RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME and self.current_model_type == 'rgb_yolo_historical_primera_calibrated') or (requested_name == PLAYER_GEOMETRY_FUSION_MODEL_NAME and self.current_model_type == 'player_geometry_fusion') or (requested_name == M1_H2_D2_WEB_MODEL_NAME and self.current_model_type == 'm1_h2_d2_cross_fold') or (requested_name in M3_WEB_EXPORTS and self.current_model_type == requested_name) or (requested_name == HISTORICAL_YOLO_RESIDUAL_MODEL_NAME and self.current_model_type == 'historical_yolo_residual')):
            logger.info(f'Model already loaded: {self.current_model_name}')
            return
        try:
            if model_path and str(model_path) == RGB_YOLO_QUANTILE_FUSION_MODEL_NAME:
                logger.info('Loading audited RGB + YOLO quantile fusion model')
                from models.M2.predictor import RGBYoloQuantileFusionDetector
                self.current_model = RGBYoloQuantileFusionDetector(self.models_dir.parent)
                self.current_model_name = self.current_model.display_name
                self.current_model_type = 'rgb_yolo_quantile_fusion'
                logger.info(f'RGB+YOLO quantile fusion loaded: {self.current_model_name}')
                return
            if model_path and str(model_path) == RGB_YOLO_HIGHRES_MODEL_NAME:
                logger.info('Loading promoted RGB + YOLO high-resolution ROI model')
                from models.M2.predictor import RGBYoloHighresQuantileFusionDetector
                self.current_model = RGBYoloHighresQuantileFusionDetector(self.models_dir.parent)
                self.current_model_name = self.current_model.display_name
                self.current_model_type = 'rgb_yolo_quantile_highres'
                logger.info(f'High-resolution RGB+YOLO model loaded: {self.current_model_name}')
                return
            if model_path and str(model_path) == RGB_YOLO_EVENT_VERIFIER_MODEL_NAME:
                logger.info('Loading high-resolution RGB+YOLO temporal event verifier')
                from models.M2.predictor import RGBYoloTemporalEventVerifierDetector
                self.current_model = RGBYoloTemporalEventVerifierDetector(self.models_dir.parent)
                self.current_model_name = self.current_model.display_name
                self.current_model_type = 'rgb_yolo_temporal_event_verifier'
                logger.info(f'Temporal event verifier loaded: {self.current_model_name}')
                return
            if model_path and str(model_path) == RGB_YOLO_PRIMERA_CALIBRATED_MODEL_NAME:
                logger.info('Loading Historical Primera calibrated RGB+YOLO temporal model')
                from models.M2.predictor import RGBYoloHistoricalPrimeraCalibratedDetector
                self.current_model = RGBYoloHistoricalPrimeraCalibratedDetector(self.models_dir.parent)
                self.current_model_name = self.current_model.display_name
                self.current_model_type = 'rgb_yolo_historical_primera_calibrated'
                logger.info(f'Historical Primera calibrated model loaded: {self.current_model_name}')
                return
            if model_path and str(model_path) == HISTORICAL_YOLO_RESIDUAL_MODEL_NAME:
                logger.info('Loading historical model with gated YOLO residual ensemble')
                self.current_model = HistoricalYoloResidualDetector(self.models_dir.parent)
                self.current_model_name = self.current_model.display_name
                self.current_model_type = 'historical_yolo_residual'
                logger.info(f'Historical-YOLO residual ensemble loaded: {self.current_model_name}')
                return
            if model_path and str(model_path) == PLAYER_GEOMETRY_FUSION_MODEL_NAME:
                logger.info('Loading thesis RGB + pose + geometry fusion ensemble')
                self.current_model = PlayerGeometryFusionDetector(self.models_dir.parent)
                self.current_model_name = self.current_model.display_name
                self.current_model_type = 'player_geometry_fusion'
                logger.info(f'Geometry-fusion ensemble loaded: {self.current_model_name}')
                return
            if model_path and str(model_path) == M1_H2_D2_WEB_MODEL_NAME:
                logger.info('Loading M1-H2-D2 six-fold web ensemble')
                self.current_model = M1H2D2CrossFoldDetector(self.models_dir.parent)
                self.current_model_name = self.current_model.display_name
                self.current_model_type = 'm1_h2_d2_cross_fold'
                logger.info(f'M1-H2-D2 web ensemble loaded: {self.current_model_name}')
                return
            if model_path and str(model_path) in M3_WEB_EXPORTS:
                from models.M3.predictor import M3H1D2CrossFoldDetector, M3H2D2CrossFoldDetector
                detector_class = M3H2D2CrossFoldDetector if str(model_path) == M3_H2_D2_WEB_MODEL_NAME else M3H1D2CrossFoldDetector
                logger.info('Loading %s six-fold web ensemble', model_path)
                self.current_model = detector_class(self.models_dir.parent)
                self.current_model_name = self.current_model.display_name
                self.current_model_type = str(model_path)
                logger.info('M3 web ensemble loaded: %s', self.current_model_name)
                return
            if model_path and str(model_path) == CLAHE_WEB_MODEL_NAME:
                model_dir = self.models_dir / 'M0'
                model_file = model_dir / 'checkpoints' / CLAHE_WEB_MODEL_NAME
                if not _resolve_model_artifact(model_file).exists():
                    raise FileNotFoundError(f'CLAHE web checkpoint not found: {model_file}')
                logger.info(f'Loading isolated CLAHE web model: {model_file}')
                self.current_model = RugbyPlayDetector(str(model_dir), model_name=CLAHE_WEB_MODEL_NAME, extractor_name='resnet50', preprocessing_filter='clahe')
                self.current_model_name = 'CLAHE · Recall experimental (macro-F1)'
                self.current_model_type = 'cnn_rnn_clahe'
                logger.info('CLAHE web model loaded successfully')
                return
            if model_path and str(model_path) in {TWO_STREAM_FULL_MODEL_NAME, TWO_STREAM_PRECISION_WORKFLOW_NAME}:
                mode = str(model_path)
                logger.info(f'Loading Two-Stream production adapter: {mode}')
                self.current_model = TwoStreamProductionDetector(mode=mode, project_root=self.models_dir.parent)
                self.current_model_name = self.current_model.display_name
                self.current_model_type = 'two_stream_review_workflow' if mode == TWO_STREAM_PRECISION_WORKFLOW_NAME else 'two_stream_full_classifier'
                logger.info(f'Two-Stream adapter loaded: {self.current_model_name}')
                return
            if model_path and str(model_path) == META_CORRECTION_MODEL_NAME:
                logger.info('Loading CNN-RNN + few-shot meta-corrector')
                self.current_model = MetaCorrectionDetector(self.models_dir.parent)
                self.current_model_name = self.current_model.display_name
                self.current_model_type = 'meta_correction'
                logger.info(f'Meta-corrector loaded: {self.current_model_name}')
                return
            if model_path and str(model_path) in CLASS_IMBALANCE_EXPERIMENT_MODELS:
                public_name = str(model_path)
                model_info = CLASS_IMBALANCE_EXPERIMENT_MODELS[public_name]
                model_dir = self.models_dir / 'M0'
                if model_info.get('project_relative_path'):
                    model_file = (self.models_dir.parent / model_info['project_relative_path']).resolve()
                else:
                    model_file = (self.models_dir.parent / 'data' / 'class_imbalance_comparison' / 'exports' / 'top2_models' / model_info['filename']).resolve()
                if not _resolve_model_artifact(model_file).exists():
                    raise FileNotFoundError(f'Class-imbalance model not found: {model_file}')
                logger.info(f'Loading isolated class-imbalance model: {model_file}')
                self.current_model = RugbyPlayDetector(str(model_dir), model_name=str(model_file), extractor_name='resnet50')
                self.current_model_name = model_info['display_name']
                self.current_model_type = 'cnn_rnn'
                logger.info(f'Class-imbalance model loaded: {self.current_model_name}')
                return
            if model_path and str(model_path) in NO_PLAY_TUNED_ENSEMBLES:
                model_dir = self.models_dir / 'M0'
                ensemble_root = self.models_dir.parent / 'data' / 'no_play_best_model' / 'checkpoints' / 'folded_no_play_other'
                ensemble_config = NO_PLAY_TUNED_ENSEMBLES[str(model_path)]
                logger.info(f'Loading tuned no-play ensemble: {model_path}')
                self.current_model = TunedNoPlayEnsembleDetector(model_dir=model_dir, ensemble_root=ensemble_root, config=ensemble_config)
                self.current_model_name = ensemble_config['display_name']
                self.current_model_type = 'tuned_no_play_ensemble'
                logger.info(f'Tuned no-play ensemble loaded: {self.current_model_name}')
                return
            if model_path and str(model_path).startswith('ensemble_top'):
                if not ENSEMBLE_AVAILABLE:
                    raise ImportError('EnsembleDetector not available. Check ensemble_detector.py.')
                try:
                    top_n = int(str(model_path).replace('ensemble_top', ''))
                except ValueError:
                    top_n = 3
                model_dir = self.models_dir / 'M0'
                results_csv = model_dir / 'results' / 'grid_search_results.csv'
                logger.info(f'Loading ensemble of top-{top_n} grid search models...')
                self.current_model = EnsembleDetector(top_n=top_n, results_csv=results_csv)
                self.current_model_name = f'Ensemble Top-{top_n} (soft voting)'
                self.current_model_type = 'ensemble'
                logger.info(f'Ensemble loaded: {self.current_model}')
                return
            if model_path is None or 'cnn_rnn' in str(model_path).lower() or 'gru' in str(model_path).lower() or ('lstm' in str(model_path).lower()) or str(model_path).endswith('.keras'):
                model_dir = self.models_dir / 'M0'
                logger.info(f'Loading CNN-RNN model from {model_dir}')
                if model_path and str(model_path).endswith('.keras'):
                    full_path = Path(model_path)
                    checkpoints_dir = model_dir / 'checkpoints'
                    if full_path.is_absolute():
                        try:
                            model_filename = str(full_path.relative_to(checkpoints_dir))
                        except ValueError:
                            model_filename = str(full_path)
                    else:
                        model_filename = str(full_path)
                    model_filename = model_filename.replace('\\', '/')
                    self.current_model = RugbyPlayDetector(str(model_dir), model_name=model_filename)
                    self.current_model_name = f'CNN-RNN ({model_filename})'
                else:
                    self.current_model = RugbyPlayDetector(str(model_dir))
                    self.current_model_name = 'CNN-RNN'
                self.current_model_type = 'cnn_rnn'
                logger.info(f'Model loaded successfully: {self.current_model_name}')
            else:
                raise ValueError(f'Unsupported model path: {model_path}')
        except Exception as e:
            self.clear_model()
            logger.error(f'Failed to load model: {e}')
            raise

    def analyze_video(self, video_path: Path, confidence_threshold: float=0.5, line_confidence_threshold: float=None, min_duration: float=3.0, max_duration: float=15.0, overlap_threshold: float=0.5, batch_size: int=16, segment_duration: int=7, use_segments: bool=True, temporal_fps_mode: str='training_match_25', custom_effective_fps: float=None, feature_batch_size: int=256, feature_predict_batch_size: int=8, progress_callback=None) -> Dict:
        """
        Analyze a video and detect rugby plays.
        
        Args:
            video_path: Path to video file
            confidence_threshold: Minimum confidence for detection (0.0-1.0)
            min_duration: Minimum play duration in seconds (not used for full video)
            max_duration: Maximum play duration in seconds (not used for full video)
            overlap_threshold: IoU threshold for NMS (not used currently)
            batch_size: Batch size for inference (not used, kept for compatibility)
            segment_duration: Duration of each window in seconds (default 7 for sliding window)
            use_segments: Whether to analyze video with sliding windows (default True)
        
        Returns:
            Dictionary with detection results
        """
        if self.current_model is None:
            raise ValueError('No model loaded. Call load_model() first.')
        logger.info(f'Analyzing video: {video_path.name}')
        logger.info(f'Parameters: confidence={confidence_threshold}, sliding_window={use_segments}')
        if self.current_model_type in {'two_stream_full_classifier', 'two_stream_review_workflow'}:
            return self.current_model.analyze_video(str(video_path), confidence_threshold=confidence_threshold, line_confidence_threshold=line_confidence_threshold, min_duration=min_duration, max_duration=max_duration, overlap_threshold=overlap_threshold, batch_size=batch_size, segment_duration=segment_duration, use_segments=use_segments, temporal_fps_mode=temporal_fps_mode, custom_effective_fps=custom_effective_fps, feature_batch_size=feature_batch_size, feature_predict_batch_size=feature_predict_batch_size, progress_callback=progress_callback)
        try:
            if use_segments:
                results = self.current_model.analyze_video_segments(str(video_path), segment_duration=segment_duration, confidence_threshold=confidence_threshold, line_confidence_threshold=line_confidence_threshold, temporal_fps_mode=temporal_fps_mode, custom_effective_fps=custom_effective_fps, feature_batch_size=feature_batch_size, feature_predict_batch_size=feature_predict_batch_size, progress_callback=progress_callback)
            else:
                results = self.current_model.analyze_video(str(video_path), confidence_threshold=confidence_threshold)
            detections = results.get('detections', [])
            for det in detections:
                if 'duration' not in det:
                    det['duration'] = det.get('end_time', 0) - det.get('start_time', 0)
            statistics = self._generate_statistics(detections)
            formatted_results = {'status': results['status'], 'video_path': str(video_path), 'model_name': self.current_model_name, 'detections': detections, 'primary_detections': results.get('primary_detections', []), 'review_detections_7s': results.get('review_detections_7s', []), 'window_predictions': results.get('window_predictions', []), 'event_post_processing': results.get('event_post_processing'), 'meta_correction': results.get('meta_correction'), 'fusion_metadata': getattr(self.current_model, 'fusion_metadata', None), 'statistics': statistics, 'summary': self._generate_summary(results), 'metadata': {'duration': results.get('duration', 0), 'confidence_threshold': confidence_threshold, 'line_confidence_threshold': line_confidence_threshold, 'analysis_mode': 'segments' if use_segments else 'full_video', 'temporal_fps_mode': temporal_fps_mode, 'custom_effective_fps': custom_effective_fps, 'event_post_processing': results.get('event_post_processing'), 'meta_correction': results.get('meta_correction'), 'fusion_metadata': getattr(self.current_model, 'fusion_metadata', None)}}
            return formatted_results
        except Exception as e:
            import traceback
            logger.error(f'Error analyzing video: {e}')
            logger.error(f'Full traceback:\n{traceback.format_exc()}')
            return {'status': 'error', 'error': str(e), 'detections': []}

    def _generate_statistics(self, detections: List[Dict]) -> Dict:
        """
        Generate statistics in the format expected by the frontend.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Statistics dictionary with avg_confidence, avg_duration, by_type
        """
        if not detections:
            return {'avg_confidence': 0.0, 'avg_duration': 0.0, 'by_type': {}}
        total_confidence = sum((det['confidence'] for det in detections))
        total_duration = sum((det.get('duration', det.get('end_time', 0) - det.get('start_time', 0)) for det in detections))
        by_type = {}
        for det in detections:
            play_type = det['play_type']
            if play_type not in by_type:
                by_type[play_type] = {'count': 0, 'total_duration': 0.0}
            by_type[play_type]['count'] += 1
            by_type[play_type]['total_duration'] += det.get('duration', det.get('end_time', 0) - det.get('start_time', 0))
        return {'avg_confidence': total_confidence / len(detections), 'avg_duration': total_duration / len(detections), 'by_type': by_type}

    def _generate_summary(self, results: Dict) -> Dict:
        """
        Generate summary statistics from detection results.
        
        Args:
            results: Raw results from model
            
        Returns:
            Summary dictionary
        """
        detections = results.get('detections', [])
        if not detections:
            return {'total_detections': 0, 'plays_by_type': {}, 'average_confidence': 0.0}
        plays_by_type = {}
        total_confidence = 0.0
        for det in detections:
            play_type = det['play_type']
            plays_by_type[play_type] = plays_by_type.get(play_type, 0) + 1
            total_confidence += det['confidence']
        return {'total_detections': len(detections), 'plays_by_type': plays_by_type, 'average_confidence': total_confidence / len(detections) if detections else 0.0, 'predicted_class': results.get('predicted_class', 'unknown'), 'overall_confidence': results.get('confidence', 0.0)}

    def get_model_info(self, model_name: str=None) -> Dict:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.current_model is None:
            return {'status': 'no_model_loaded'}
        return {'name': self.current_model_name, 'type': self.current_model_type, 'framework': 'Keras/TensorFlow', 'labels': self.current_model.label_vocab.tolist() if hasattr(self.current_model, 'label_vocab') else [], 'architecture': getattr(self.current_model, 'fusion_metadata', {}).get('architecture') if hasattr(self.current_model, 'fusion_metadata') else None, 'fusion_metadata': getattr(self.current_model, 'fusion_metadata', None), 'status': 'loaded'}
