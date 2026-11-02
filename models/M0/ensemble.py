"""
Ensemble of top-N grid search models via soft voting.

Strategy: average softmax probability outputs from the top-N models
(ranked by val_f1_macro from grid_search_results.csv).
This typically adds +2-4% accuracy with zero additional training.

Usage:
    # Evaluate top-3 ensemble on test set:
    python ensemble.py

    # Evaluate top-5 ensemble:
    python ensemble.py --top 5

    # Specify a custom CSV and pick top models from a specific extractor:
    python ensemble.py --top 3 --extractor resnet50

    # Evaluate ensemble on a specific video:
    python ensemble.py --video ../../data/raw/myvideo.mp4
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import os
import sys
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import keras
import cv2
_DIR = Path(__file__).parent
sys.path.insert(0, str(source_directory(_DIR)))
from models.M0.config import MAX_SEQ_LENGTH, IMG_SIZE, CHECKPOINTS_DIR, TEST_CSV, RESULTS_DIR, FEATURE_CACHE_DIR
from models.M0.data_utils import load_video, load_raw_video, preprocess_raw_frame, _sample_frame_indices
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def _load_results_csv(csv_path: Path) -> List[Dict]:
    """Return list of dicts from grid_search_results.csv, sorted by val_f1_macro desc."""
    rows = []
    with open(_resolve_model_artifact(csv_path), newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['val_f1_macro'] = float(row.get('val_f1_macro', 0) or 0)
                rows.append(row)
            except (ValueError, TypeError):
                pass
    rows.sort(key=lambda r: r['val_f1_macro'], reverse=True)
    return rows

def select_top_models(results_csv: Path, top_n: int=3, extractor_filter: str=None) -> List[Dict]:
    """
    Select the top-N models from grid search results.

    Args:
        results_csv: Path to grid_search_results.csv
        top_n: How many models to include in the ensemble
        extractor_filter: If given, only consider models with this extractor
                          (e.g. 'resnet50', 'efficientnetb0', 'inceptionv3')

    Returns:
        List of dicts with keys: trial_name, extractor, val_f1_macro, ...
    """
    rows = _load_results_csv(results_csv)
    if extractor_filter:
        rows = [r for r in rows if r.get('extractor', '') == extractor_filter]
    seen = set()
    unique_rows = []
    for row in rows:
        name = row.get('trial_name', '')
        if name not in seen:
            seen.add(name)
            unique_rows.append(row)
    selected = unique_rows[:top_n]
    return selected

def _checkpoint_path(trial_name: str) -> Path:
    """Resolve the .keras file for a given trial name."""
    grid_dir = Path(CHECKPOINTS_DIR) / 'grid_search' / trial_name
    model_file = grid_dir / 'model.keras'
    if not _resolve_model_artifact(model_file).exists():
        raise FileNotFoundError(f'Checkpoint not found: {model_file}\n(trial: {trial_name})')
    return model_file

def _extractor_path(extractor_name: str) -> Path:
    """Resolve the shared feature extractor .keras file."""
    path = Path(CHECKPOINTS_DIR) / f'feature_extractor_{extractor_name}.keras'
    if not _resolve_model_artifact(path).exists():
        raise FileNotFoundError(f'Feature extractor not found: {path}')
    return path

class GridSearchEnsemble:
    """
    Ensemble of K trained sequence models from grid search.

    All models that share the same feature extractor reuse it.
    For models with different extractors, each extractor is loaded once.
    Inference runs one extractor pass per unique extractor, then feeds the
    resulting features to all sequence models that use it.
    """

    def __init__(self, model_configs: List[Dict]):
        """
        Args:
            model_configs: List of dicts, each with at least:
                - 'trial_name': str  (matches checkpoints/grid_search/<name>/)
                - 'extractor':  str  ('resnet50' | 'efficientnetb0' | 'inceptionv3')
                - 'val_f1_macro': float
        """
        self.model_configs = model_configs
        self.label_vocab: np.ndarray = None
        self._extractors: Dict[str, keras.Model] = {}
        self._seq_models: List[keras.Model] = []
        self._load_label_vocab()
        self._load_models()

    def _load_label_vocab(self):
        return np.asarray(['line', 'other', 'scrum'])

    def _load_models(self):
        logger.info(f'Loading {len(self.model_configs)} ensemble member(s)...')
        for cfg in self.model_configs:
            extractor_name = cfg['extractor']
            trial_name = cfg['trial_name']
            if extractor_name not in self._extractors:
                ext_path = _extractor_path(extractor_name)
                logger.info(f'  Loading feature extractor: {extractor_name} from {ext_path}')
                self._extractors[extractor_name] = keras.models.load_model(_resolve_model_artifact(str(ext_path)))
            ckpt_path = _checkpoint_path(trial_name)
            logger.info(f"  Loading sequence model: {trial_name} (val_f1={cfg.get('val_f1_macro', '?'):.4f}) from {ckpt_path}")
            self._seq_models.append(keras.models.load_model(_resolve_model_artifact(str(ckpt_path))))
        logger.info('All ensemble models loaded.')

    def _extract_features(self, frames: np.ndarray, extractor_name: str) -> np.ndarray:
        """
        Extract features from an array of frames using the named extractor.

        Args:
            frames: uint8 array of shape (N, H, W, 3) — already center-cropped and resized
            extractor_name: which extractor to use

        Returns:
            float32 array of shape (N, feature_dim)
        """
        extractor = self._extractors[extractor_name]
        features = extractor.predict(frames, batch_size=32, verbose=0)
        return features.astype(np.float32)

    def _pad_features(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pad / truncate features to MAX_SEQ_LENGTH; return (features_batch, mask_batch)."""
        if len(features) < MAX_SEQ_LENGTH:
            padded = np.zeros((MAX_SEQ_LENGTH, features.shape[1]), dtype=np.float32)
            padded[:len(features)] = features
            mask = np.zeros(MAX_SEQ_LENGTH, dtype=np.float32)
            mask[:len(features)] = 1.0
        else:
            padded = features[:MAX_SEQ_LENGTH]
            mask = np.ones(MAX_SEQ_LENGTH, dtype=np.float32)
        features_batch = np.expand_dims(padded, axis=0)
        mask_batch = np.expand_dims(mask, axis=0)
        return (features_batch, mask_batch)

    def _avg_probs_from_features(self, features_by_extractor: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run all sequence models on pre-extracted features and return averaged probs.

        Args:
            features_by_extractor: {extractor_name: feature_array (seq_len, feat_dim)}

        Returns:
            Averaged probability vector (num_classes,)
        """
        all_probs = []
        for cfg, seq_model in zip(self.model_configs, self._seq_models):
            extractor_name = cfg['extractor']
            features = features_by_extractor[extractor_name]
            feat_batch, mask_batch = self._pad_features(features)
            probs = seq_model.predict([feat_batch, mask_batch], verbose=0)[0]
            all_probs.append(probs)
        return np.mean(all_probs, axis=0)

    def predict_video(self, video_path: str, confidence_threshold: float=0.5) -> Dict:
        """
        Predict the class of an entire video (single-label classification).

        Args:
            video_path: Path to video file
            confidence_threshold: Minimum confidence for a detection to be reported

        Returns:
            Result dict compatible with RugbyPlayDetector.analyze_video()
        """
        logger.info(f'Predicting video: {video_path}')
        try:
            frames = load_raw_video(video_path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE), target_fps=25.0)
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'detections': []}
        logger.info(f'Loaded {len(frames)} frames')
        unique_extractors = {cfg['extractor'] for cfg in self.model_configs}
        features_by_extractor = {}
        for ext_name in unique_extractors:
            features_by_extractor[ext_name] = self._extract_features(frames, ext_name)
        avg_probs = self._avg_probs_from_features(features_by_extractor)
        pred_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx])
        predicted_label = str(self.label_vocab[pred_idx])
        logger.info(f'Ensemble prediction: {predicted_label} (confidence={confidence:.2%})')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        detections = []
        if confidence >= confidence_threshold:
            detections.append({'play_type': predicted_label, 'start_time': 0.0, 'end_time': duration, 'confidence': confidence, 'frame_start': 0, 'frame_end': total_frames})
        return {'status': 'success', 'video_path': video_path, 'duration': duration, 'predicted_class': predicted_label, 'confidence': confidence, 'all_probabilities': {str(label): float(prob) for label, prob in zip(self.label_vocab, avg_probs)}, 'detections': detections}

    def predict_video_sliding_window(self, video_path: str, window_duration: int=7, stride_duration: int=1, confidence_threshold: float=0.5, temporal_fps_mode: str='training_match_25', custom_effective_fps: float=None, feature_batch_size: int=256, feature_predict_batch_size: int=8, progress_callback=None) -> Dict:
        """
        Predict plays in a video using a sliding-window approach.

        Each window's softmax outputs from all ensemble members are averaged
        before taking argmax (soft voting per window).

        Args:
            video_path: Path to video file
            window_duration: Window size in seconds (default 7)
            stride_duration: Stride in seconds (default 1)
            confidence_threshold: Minimum confidence for a detection

        Returns:
            Result dict compatible with RugbyPlayDetector.analyze_video_segments()
        """
        logger.info(f'Sliding-window ensemble ({window_duration}s window, {stride_duration}s stride): {video_path}')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        mode = (temporal_fps_mode or 'training_match_25').strip().lower()
        if mode == 'fast_12':
            requested_fps = 12.0
        elif mode == 'custom':
            try:
                requested_fps = max(1.0, float(custom_effective_fps))
            except (TypeError, ValueError):
                requested_fps = 25.0
        else:
            requested_fps = 25.0
        effective_fps = min(float(fps), requested_fps)
        frames_per_window = max(1, int(round(effective_fps * window_duration)))
        frames_per_stride = max(1, int(round(effective_fps * stride_duration)))
        sample_indices = set(_sample_frame_indices(total_frames, fps, effective_fps))
        if progress_callback:
            progress_callback(18, 'preprocessing', 'Preparando el muestreo temporal')
        logger.info('Extracting all frames...')
        all_frames = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx not in sample_indices:
                frame_idx += 1
                continue
            frame = preprocess_raw_frame(frame, target_size=IMG_SIZE)
            if frame is None:
                cap.release()
                return {'status': 'error', 'error': 'Raw frame could not be normalized for the 224x224 training crop', 'detections': []}
            frame = frame[:, :, [2, 1, 0]]
            all_frames.append(frame)
            frame_idx += 1
        cap.release()
        if not all_frames:
            return {'status': 'error', 'error': 'No frames extracted', 'detections': []}
        all_frames_array = np.array(all_frames, dtype=np.uint8)
        logger.info(f'Extracted {len(all_frames_array)} frames')
        unique_extractors = {cfg['extractor'] for cfg in self.model_configs}
        all_feats: Dict[str, np.ndarray] = {}
        for ext_name in unique_extractors:
            extractor = self._extractors[ext_name]
            logger.info(f'Extracting {ext_name} features for all frames...')
            predict_batch = max(1, int(feature_predict_batch_size or 8))
            all_feats[ext_name] = extractor.predict(all_frames_array, batch_size=predict_batch, verbose=0).astype(np.float32)
        if progress_callback:
            progress_callback(70, 'features', 'Características extraídas')
        if len(all_frames_array) <= frames_per_window:
            window_starts = [0]
        else:
            window_starts = list(range(0, len(all_frames_array) - frames_per_window + 1, frames_per_stride))
        num_windows = len(window_starts)
        logger.info(f'Processing {num_windows} windows...')
        detections = []
        for w_idx, start_f in enumerate(window_starts):
            end_f = start_f + frames_per_window
            if end_f > len(all_frames_array):
                break
            start_time = start_f / effective_fps
            end_time = end_f / effective_fps
            window_feats = {ext: all_feats[ext][start_f:end_f] for ext in unique_extractors}
            avg_probs = self._avg_probs_from_features(window_feats)
            pred_idx = int(np.argmax(avg_probs))
            confidence = float(avg_probs[pred_idx])
            predicted_label = str(self.label_vocab[pred_idx])
            if confidence >= confidence_threshold:
                detections.append({'play_type': predicted_label, 'start_time': float(start_time), 'end_time': float(end_time), 'duration': float(end_time - start_time), 'confidence': confidence, 'frame_start': int(start_f), 'frame_end': int(end_f), 'probabilities': {str(label): float(prob) for label, prob in zip(self.label_vocab, avg_probs)}})
            if progress_callback:
                progress_callback(72 + 22 * (w_idx + 1) / max(1, num_windows), 'inference', f'Ventana {w_idx + 1} de {num_windows}')
        logger.info(f'Found {len(detections)} detections above threshold {confidence_threshold}')
        return {'status': 'success', 'video_path': video_path, 'duration': duration, 'source_fps': float(fps), 'effective_fps': float(effective_fps), 'sampled_total_frames': int(len(all_frames_array)), 'window_duration': window_duration, 'stride_duration': stride_duration, 'temporal_fps_mode': str(temporal_fps_mode), 'custom_effective_fps': float(custom_effective_fps) if custom_effective_fps is not None else None, 'total_windows': num_windows, 'detections': detections}

    def evaluate_test_set(self, test_csv: Path=None, batch_size: int=32) -> Dict:
        """
        Evaluate the ensemble on the test set using cached features.

        Loads pre-extracted features from FEATURE_CACHE_DIR (one .npz per video,
        per extractor) if available, otherwise extracts on the fly.

        Args:
            test_csv: Path to test CSV (columns: clip_path, label OR video_path, tag)
            batch_size: Feature extraction batch size (used when cache is missing)

        Returns:
            Dict with accuracy, macro-F1, and per-class recall
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
        if test_csv is None:
            test_csv = Path(TEST_CSV)
        logger.info(f'Evaluating ensemble on test set: {test_csv}')
        df = pd.read_csv(_resolve_model_artifact(test_csv))
        if 'clip_path' in df.columns:
            path_col, label_col = ('clip_path', 'label')
        elif 'video_path' in df.columns:
            path_col, label_col = ('video_path', 'tag')
        else:
            path_col, label_col = (df.columns[0], df.columns[1])
        label_to_idx = {str(lbl): i for i, lbl in enumerate(self.label_vocab)}
        y_true = []
        y_pred = []
        all_avg_probs = []
        cache_root = Path(FEATURE_CACHE_DIR)
        unique_extractors = {cfg['extractor'] for cfg in self.model_configs}
        logger.info(f'Running on {len(df)} test samples...')
        for row_idx, row in df.iterrows():
            video_path = str(row[path_col])
            true_label = str(row[label_col])
            if true_label not in label_to_idx:
                logger.warning(f"Unknown label '{true_label}' for {video_path}, skipping")
                continue
            features_by_extractor: Dict[str, np.ndarray] = {}
            need_video_load = False
            video_name = Path(video_path).stem
            for ext_name in unique_extractors:
                safe_ext = ext_name.replace('/', '_').replace('\\', '_')
                cache_dir = cache_root / safe_ext
                cache_file = cache_dir / f'{video_name}.npz'
                if _resolve_model_artifact(cache_file).exists():
                    data = _load_model_result_array(str(cache_file), allow_pickle=True)
                    features_by_extractor[ext_name] = data['frame_features'].astype(np.float32)
                else:
                    need_video_load = True
                    break
            if need_video_load:
                if not os.path.exists(_resolve_model_artifact(video_path)):
                    logger.warning(f'Video not found: {video_path}, skipping')
                    continue
                frames = load_video(video_path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE))
                if len(frames) == 0:
                    logger.warning(f'No frames from {video_path}, skipping')
                    continue
                for ext_name in unique_extractors:
                    features_by_extractor[ext_name] = self._extract_features(frames, ext_name)
            avg_probs = self._avg_probs_from_features(features_by_extractor)
            pred_idx = int(np.argmax(avg_probs))
            predicted_label = str(self.label_vocab[pred_idx])
            y_true.append(label_to_idx[true_label])
            y_pred.append(pred_idx)
            all_avg_probs.append(avg_probs)
            if (row_idx + 1) % 50 == 0:
                logger.info(f'  Processed {row_idx + 1}/{len(df)} samples')
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        class_names = [str(l) for l in self.label_vocab]
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        results = {'num_samples': len(y_true), 'accuracy': float(acc), 'f1_macro': float(f1_macro), 'f1_weighted': float(f1_weighted), 'per_class_recall': {class_names[i]: float(r) for i, r in enumerate(recalls)}, 'classification_report': report, 'confusion_matrix': cm.tolist(), 'ensemble_members': [{'trial_name': cfg.get('trial_name', ''), 'extractor': cfg.get('extractor', ''), 'val_f1_macro': float(cfg.get('val_f1_macro', 0))} for cfg in self.model_configs]}
        return results

def _parse_args():
    parser = argparse.ArgumentParser(description='Ensemble of top-N grid search models (soft voting).', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--top', type=int, default=3, help='Number of top models to include in the ensemble (default: 3)')
    parser.add_argument('--extractor', type=str, default=None, help='Only use models with this feature extractor (e.g. resnet50, efficientnetb0, inceptionv3). Default: any.')
    parser.add_argument('--results-csv', type=str, default=None, help='Path to grid_search_results.csv. Default: models/M0/results/grid_search_results.csv')
    parser.add_argument('--test-csv', type=str, default=None, help='Path to test CSV. Default: value from config.py')
    parser.add_argument('--video', type=str, default=None, help='If given, run sliding-window inference on this video instead of evaluating the test set.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--window', type=int, default=7, help='Sliding-window duration in seconds (default: 7)')
    parser.add_argument('--stride', type=int, default=1, help='Sliding-window stride in seconds (default: 1)')
    parser.add_argument('--output', type=str, default=None, help='Save evaluation results to this JSON file.')
    return parser.parse_args()

def main():
    args = _parse_args()
    if args.results_csv:
        results_csv = Path(args.results_csv)
    else:
        results_csv = _DIR / 'results' / 'grid_search_results.csv'
    if not _resolve_model_artifact(results_csv).exists():
        logger.error(f'Grid search results CSV not found: {results_csv}')
        sys.exit(1)
    logger.info(f'Selecting top-{args.top} models from {results_csv}' + (f' (extractor={args.extractor})' if args.extractor else ''))
    top_models = select_top_models(results_csv, top_n=args.top, extractor_filter=args.extractor)
    if not top_models:
        logger.error('No models found matching the criteria.')
        sys.exit(1)
    logger.info(f'Selected {len(top_models)} ensemble member(s):')
    for i, cfg in enumerate(top_models, 1):
        logger.info(f"  {i}. {cfg['trial_name']}  extractor={cfg.get('extractor', '')}  val_f1={cfg.get('val_f1_macro', 0):.4f}")
    missing = []
    for cfg in top_models:
        try:
            _checkpoint_path(cfg['trial_name'])
        except FileNotFoundError as e:
            missing.append(str(e))
    if missing:
        logger.error('Some checkpoints are missing:')
        for m in missing:
            logger.error(f'  {m}')
        logger.error('Only models that have been trained and saved by grid_search.py can be used. Run grid_search.py first.')
        sys.exit(1)
    ensemble = GridSearchEnsemble(top_models)
    if args.video:
        logger.info(f'Running sliding-window inference on: {args.video}')
        result = ensemble.predict_video_sliding_window(args.video, window_duration=args.window, stride_duration=args.stride, confidence_threshold=args.threshold)
        print('\n' + '=' * 60)
        print('ENSEMBLE INFERENCE RESULTS')
        print('=' * 60)
        print(f"Status    : {result['status']}")
        print(f"Duration  : {result.get('duration', 0):.1f}s")
        print(f"Windows   : {result.get('total_windows', 0)}")
        print(f"Detections: {len(result.get('detections', []))}")
        for det in result.get('detections', []):
            print(f"  [{det['start_time']:.1f}s - {det['end_time']:.1f}s]  {det['play_type']}  conf={det['confidence']:.2%}")
        if args.output:
            with open(_resolve_model_artifact(args.output), 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f'Results saved to: {args.output}')
    else:
        logger.info('Evaluating ensemble on test set...')
        results = ensemble.evaluate_test_set(test_csv=Path(args.test_csv) if args.test_csv else None)
        print('\n' + '=' * 60)
        print('ENSEMBLE EVALUATION RESULTS')
        print('=' * 60)
        print(f"Samples   : {results['num_samples']}")
        print(f"Accuracy  : {results['accuracy']:.4f}  ({results['accuracy'] * 100:.2f}%)")
        print(f"F1 Macro  : {results['f1_macro']:.4f}")
        print(f"F1 Weighted: {results['f1_weighted']:.4f}")
        print('\nPer-class Recall:')
        for cls, rec in results['per_class_recall'].items():
            print(f'  {cls:<10}: {rec:.4f}  ({rec * 100:.2f}%)')
        print('\nClassification Report:')
        print(results['classification_report'])
        if args.output:
            out_path = Path(args.output)
        else:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            out_path = Path(RESULTS_DIR) / f'ensemble_top{args.top}_results.json'
        save_data = {k: v for k, v in results.items() if k != 'classification_report'}
        save_data['classification_report_text'] = results['classification_report']
        with open(_resolve_model_artifact(out_path), 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f'Results saved to: {out_path}')
if __name__ == '__main__':
    main()
