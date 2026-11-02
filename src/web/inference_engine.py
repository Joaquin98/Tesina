"""
Inference engine for rugby play detection.

Handles model loading, video analysis, and result generation.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm
import sys
sys.path.append(str(source_directory(Path(__file__).parent.parent)))
from src.preprocessing.video_utils import extract_frames
from models.common.spatial import normalize_training_frame
from src.evaluation.metrics import PLAY_CATEGORIES
logger = logging.getLogger(__name__)

class InferenceEngine:
    """Engine for running inference on videos with trained models."""

    def __init__(self, models_dir: Path=None):
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / 'models'
        self.models_dir = models_dir
        self.current_model = None
        self.current_model_name = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'InferenceEngine initialized. Device: {self.device}')

    def list_available_models(self) -> List[Dict]:
        """
        List all available trained models.
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        for model_dir in self.models_dir.iterdir():
            if not _resolve_model_artifact(model_dir).is_dir():
                continue
            checkpoints = list(_model_artifact_glob(model_dir, '*.pth', recursive=False)) + list(_model_artifact_glob(model_dir, '*.pt', recursive=False))
            for checkpoint_file in checkpoints:
                try:
                    checkpoint = torch.load(_resolve_model_artifact(checkpoint_file), map_location='cpu')
                    models.append({'name': checkpoint_file.name, 'path': str(checkpoint_file), 'model_type': model_dir.name, 'size_mb': checkpoint_file.stat().st_size / (1024 * 1024), 'epoch': checkpoint.get('epoch', 'unknown'), 'metrics': checkpoint.get('metrics', {})})
                except Exception as e:
                    logger.warning(f'Could not load metadata for {checkpoint_file.name}: {e}')
                    models.append({'name': checkpoint_file.name, 'path': str(checkpoint_file), 'model_type': model_dir.name, 'size_mb': checkpoint_file.stat().st_size / (1024 * 1024)})
        return models

    def load_model(self, model_name: str) -> None:
        """
        Load a trained model for inference.
        
        Args:
            model_name: Name of the model checkpoint file
        """
        model_path = None
        for model_dir in self.models_dir.iterdir():
            if _resolve_model_artifact(model_dir).is_dir():
                candidate = model_dir / model_name
                if _resolve_model_artifact(candidate).exists():
                    model_path = candidate
                    break
        if model_path is None:
            raise ValueError(f'Model not found: {model_name}')
        logger.info(f'Loading model: {model_path}')
        checkpoint = torch.load(_resolve_model_artifact(model_path), map_location=self.device)
        from models.model_loader import load_model_from_checkpoint
        self.current_model = load_model_from_checkpoint(checkpoint, self.device)
        self.current_model.eval()
        self.current_model_name = model_name
        logger.info(f'Model loaded successfully: {model_name}')

    def analyze_video(self, video_path: Path, confidence_threshold: float=0.7, min_duration: float=3.0, max_duration: float=15.0, overlap_threshold: float=0.5, batch_size: int=8) -> Dict:
        """
        Analyze a video and detect rugby plays.
        
        Args:
            video_path: Path to video file
            confidence_threshold: Minimum confidence for detection
            min_duration: Minimum play duration in seconds
            max_duration: Maximum play duration in seconds
            overlap_threshold: IoU threshold for NMS
            batch_size: Batch size for inference
        
        Returns:
            Dictionary with detection results
        """
        if self.current_model is None:
            raise ValueError('No model loaded. Call load_model() first.')
        logger.info(f'Analyzing video: {video_path.name}')
        frames = extract_frames(video_path, target_fps=25)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = len(frames) / fps
        cap.release()
        detections = self._sliding_window_detection(frames, fps, batch_size, confidence_threshold, min_duration, max_duration)
        detections = self._non_maximum_suppression(detections, overlap_threshold)
        results = {'video_path': str(video_path), 'video_name': video_path.name, 'model_name': self.current_model_name, 'duration': duration, 'fps': fps, 'num_frames': len(frames), 'parameters': {'confidence_threshold': confidence_threshold, 'min_duration': min_duration, 'max_duration': max_duration, 'overlap_threshold': overlap_threshold}, 'detections': detections, 'statistics': self._calculate_statistics(detections)}
        return results

    def _sliding_window_detection(self, frames: List[np.ndarray], fps: float, batch_size: int, confidence_threshold: float, min_duration: float, max_duration: float) -> List[Dict]:
        """Run sliding window detection on frames."""
        detections = []
        min_frames = int(min_duration * fps)
        max_frames = int(max_duration * fps)
        window_sizes = [min_frames, (min_frames + max_frames) // 2, max_frames]
        stride = min_frames // 2
        total_windows = sum(((len(frames) - ws) // stride + 1 for ws in window_sizes))
        with torch.no_grad():
            pbar = tqdm(total=total_windows, desc='Detecting plays')
            for window_size in window_sizes:
                for start_idx in range(0, len(frames) - window_size + 1, stride):
                    end_idx = start_idx + window_size
                    window_frames = frames[start_idx:end_idx]
                    inputs = self._prepare_input(window_frames)
                    outputs = self.current_model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                    confidence = confidence.item()
                    predicted_class = predicted.item()
                    if confidence >= confidence_threshold:
                        detections.append({'start_frame': start_idx, 'end_frame': end_idx, 'start_time': start_idx / fps, 'end_time': end_idx / fps, 'duration': (end_idx - start_idx) / fps, 'play_type': PLAY_CATEGORIES[predicted_class], 'confidence': float(confidence), 'probabilities': {PLAY_CATEGORIES[i]: float(probs[0, i]) for i in range(len(PLAY_CATEGORIES))}})
                    pbar.update(1)
            pbar.close()
        return detections

    def _prepare_input(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Prepare frames for model input."""
        processed = []
        for frame in frames:
            resized = normalize_training_frame(frame, target_size=224)
            if resized is None:
                raise ValueError('Frame could not be normalized to the 224x224 training geometry')
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            processed.append(normalized)
        video = np.stack(processed)
        video = np.transpose(video, (3, 0, 1, 2))
        video = np.expand_dims(video, 0)
        tensor = torch.from_numpy(video).float().to(self.device)
        return tensor

    def _non_maximum_suppression(self, detections: List[Dict], overlap_threshold: float) -> List[Dict]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [det for det in detections if self._calculate_iou(best, det) < overlap_threshold]
        keep = sorted(keep, key=lambda x: x['start_time'])
        return keep

    def _calculate_iou(self, det1: Dict, det2: Dict) -> float:
        """Calculate Intersection over Union for two detections."""
        start1, end1 = (det1['start_frame'], det1['end_frame'])
        start2, end2 = (det2['start_frame'], det2['end_frame'])
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)
        return intersection / union if union > 0 else 0

    def _calculate_statistics(self, detections: List[Dict]) -> Dict:
        """Calculate statistics about detections."""
        if not detections:
            return {'total_plays': 0, 'by_type': {}, 'avg_confidence': 0.0, 'avg_duration': 0.0}
        by_type = {}
        for det in detections:
            play_type = det['play_type']
            if play_type not in by_type:
                by_type[play_type] = []
            by_type[play_type].append(det)
        return {'total_plays': len(detections), 'by_type': {pt: {'count': len(dets), 'avg_confidence': np.mean([d['confidence'] for d in dets]), 'avg_duration': np.mean([d['duration'] for d in dets])} for pt, dets in by_type.items()}, 'avg_confidence': np.mean([d['confidence'] for d in detections]), 'avg_duration': np.mean([d['duration'] for d in detections])}

    def compare_results(self, results: Dict[str, Dict]) -> Dict:
        """Compare results from multiple models."""
        comparison = {'models': list(results.keys()), 'total_detections': {model: len(res['detections']) for model, res in results.items()}, 'avg_confidence': {model: res['statistics']['avg_confidence'] for model, res in results.items()}, 'detections_by_type': {}}
        for play_type in PLAY_CATEGORIES.values():
            comparison['detections_by_type'][play_type] = {model: len([d for d in res['detections'] if d['play_type'] == play_type]) for model, res in results.items()}
        return comparison

    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed information about a model."""
        model_path = None
        for model_dir in self.models_dir.iterdir():
            if _resolve_model_artifact(model_dir).is_dir():
                candidate = model_dir / model_name
                if _resolve_model_artifact(candidate).exists():
                    model_path = candidate
                    break
        if model_path is None:
            raise ValueError(f'Model not found: {model_name}')
        checkpoint = torch.load(_resolve_model_artifact(model_path), map_location='cpu')
        return {'name': model_name, 'path': str(model_path), 'epoch': checkpoint.get('epoch', 'unknown'), 'metrics': checkpoint.get('metrics', {}), 'config': checkpoint.get('config', {}), 'size_mb': model_path.stat().st_size / (1024 * 1024)}
