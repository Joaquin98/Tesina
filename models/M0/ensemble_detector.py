"""
EnsembleDetector — RugbyPlayDetector-compatible ensemble inference using
soft-voting across the top-N grid search models.

The class exposes exactly the same public API as RugbyPlayDetector
(analyze_video + analyze_video_segments) so it can be used in
keras_inference_engine.py without changing any calling code.

Usage (standalone):
    from models.M0.ensemble_detector import EnsembleDetector
    detector = EnsembleDetector(top_n=3)
    result = detector.analyze_video("path/to/video.mp4")
    result = detector.analyze_video_segments("path/to/video.mp4")
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
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
_DIR = Path(__file__).parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(source_directory(_DIR)))
from models.M0.ensemble import GridSearchEnsemble, select_top_models
logger = logging.getLogger(__name__)
_DEFAULT_RESULTS_CSV = _DIR / 'results' / 'grid_search_results.csv'
_DEFAULT_TOP_N = 3

class EnsembleDetector:
    """
    Wraps GridSearchEnsemble with the same interface as RugbyPlayDetector.

    Parameters
    ----------
    top_n : int
        Number of top grid-search models to include (ranked by val_f1_macro).
    extractor_filter : str | None
        If given, only consider models trained with this feature extractor.
    results_csv : Path | str | None
        Path to grid_search_results.csv.
        Defaults to models/M0/results/grid_search_results.csv.
    model_configs : list[dict] | None
        If provided, skip CSV lookup and use this explicit list of model configs.
        Each dict must have: trial_name, extractor, val_f1_macro.
    """

    def __init__(self, top_n: int=_DEFAULT_TOP_N, extractor_filter: Optional[str]=None, results_csv: Optional[Path]=None, model_configs: Optional[List[Dict]]=None):
        if results_csv is None:
            results_csv = _DEFAULT_RESULTS_CSV
        if model_configs is None:
            if not _resolve_model_artifact(Path(results_csv)).exists():
                raise FileNotFoundError(f'Grid search results CSV not found: {results_csv}\nRun grid_search.py first, or pass model_configs explicitly.')
            model_configs = select_top_models(Path(results_csv), top_n=top_n, extractor_filter=extractor_filter)
            if not model_configs:
                raise ValueError(f'No models found in {results_csv} ' + (f'for extractor={extractor_filter}' if extractor_filter else ''))
        logger.info(f'EnsembleDetector: using top-{len(model_configs)} models (extractor_filter={extractor_filter})')
        for i, cfg in enumerate(model_configs, 1):
            logger.info(f"  [{i}] {cfg['trial_name']}  extractor={cfg.get('extractor', '')}  val_f1={cfg.get('val_f1_macro', 0):.4f}")
        self._ensemble = GridSearchEnsemble(model_configs)
        self.label_vocab = self._ensemble.label_vocab
        self._top_n = len(model_configs)
        self._model_configs = model_configs

    def analyze_video(self, video_path: str, confidence_threshold: float=0.5, segment_duration: int=5) -> Dict:
        """
        Classify an entire video as a single rugby play type.

        Mirrors RugbyPlayDetector.analyze_video().

        Args:
            video_path: Path to the video file.
            confidence_threshold: Minimum confidence for a detection (0–1).
            segment_duration: Unused — kept for API compatibility.

        Returns:
            Dict with keys: status, video_path, duration, predicted_class,
            confidence, all_probabilities, detections.
        """
        return self._ensemble.predict_video(video_path=video_path, confidence_threshold=confidence_threshold)

    def analyze_video_segments(self, video_path: str, segment_duration: int=7, confidence_threshold: float=0.5, temporal_fps_mode: str='training_match_25', custom_effective_fps: Optional[float]=None, feature_batch_size: int=256, feature_predict_batch_size: int=8, progress_callback=None, **_compat_kwargs) -> Dict:
        """
        Detect rugby plays with a sliding-window approach.

        Mirrors RugbyPlayDetector.analyze_video_segments().

        Args:
            video_path: Path to the video file.
            segment_duration: Window size in seconds (default 7).
            confidence_threshold: Minimum confidence for detections.

        Returns:
            Dict with keys: status, video_path, duration, window_duration,
            stride_duration, total_windows, detections.
        """
        return self._ensemble.predict_video_sliding_window(video_path=video_path, window_duration=segment_duration, stride_duration=1, confidence_threshold=confidence_threshold, temporal_fps_mode=temporal_fps_mode, custom_effective_fps=custom_effective_fps, feature_batch_size=feature_batch_size, feature_predict_batch_size=feature_predict_batch_size, progress_callback=progress_callback)

    @property
    def ensemble_size(self) -> int:
        """Number of models in the ensemble."""
        return self._top_n

    @property
    def member_names(self) -> List[str]:
        """Trial names of the ensemble members."""
        return [cfg['trial_name'] for cfg in self._model_configs]

    def __repr__(self) -> str:
        members = ', '.join(self.member_names)
        return f'EnsembleDetector(top_n={self._top_n}, members=[{members}])'
