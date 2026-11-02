"""Fast external evaluation of the frozen high-resolution RGB+YOLO model.

This is an execution-optimized equivalent of the production adapter.  It
keeps the same RGB preprocessing, 200-frame historical sequence, source ROI
mapping, YOLO11n-pose thresholds, E04/E06 descriptors, quantile scaler,
classifier, and 40/60 fusion.  Only inference batching is changed.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
REPO = experiment_directory(__file__).parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_MANIFEST = REPO / 'data' / 'active_learning_dataset' / 'test_manifest.csv'
DEFAULT_OUTPUT = experiment_directory(__file__) / 'outputs_optimized'
LABELS = ['line', 'other', 'scrum']
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
RGB_LENGTH = 200
YOLO_FRAMES = 100
DEFAULT_YOLO_BATCH = 100
MAX_PEOPLE = 30
PERSON_CONFIDENCE = 0.25
KEYPOINT_CONFIDENCE = 0.2

def _decode_result(result, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.zeros((MAX_PEOPLE, 4), dtype=np.float32)
    person_conf = np.zeros(MAX_PEOPLE, dtype=np.float32)
    keypoints = np.zeros((MAX_PEOPLE, 17, 2), dtype=np.float32)
    keypoint_conf = np.zeros((MAX_PEOPLE, 17), dtype=np.float32)
    valid = np.zeros(MAX_PEOPLE, dtype=bool)
    if result.boxes is None or len(result.boxes) == 0:
        return (boxes, person_conf, keypoints, keypoint_conf, valid)
    raw_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    raw_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    raw_keypoints = result.keypoints.xy.detach().cpu().numpy().astype(np.float32) if result.keypoints is not None and result.keypoints.xy is not None else None
    raw_keypoint_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32) if result.keypoints is not None and result.keypoints.conf is not None else None
    order = np.argsort(raw_scores)[::-1][:MAX_PEOPLE]
    width = max(float(frame.shape[1]), 1.0)
    height = max(float(frame.shape[0]), 1.0)
    for output_index, source_index in enumerate(order):
        x1, y1, x2, y2 = raw_boxes[source_index]
        boxes[output_index] = np.clip([x1 / width, y1 / height, x2 / width, y2 / height], 0.0, 1.0)
        person_conf[output_index] = raw_scores[source_index]
        valid[output_index] = True
        if raw_keypoints is not None and source_index < len(raw_keypoints):
            keypoints[output_index] = np.clip(raw_keypoints[source_index] / np.array([width, height], dtype=np.float32), 0.0, 1.0)
        if raw_keypoint_conf is not None and source_index < len(raw_keypoint_conf):
            keypoint_conf[output_index] = raw_keypoint_conf[source_index]
    return (boxes, person_conf, keypoints, keypoint_conf, valid)

def _metrics(y_true: list[int], y_pred: list[int]) -> dict:
    return {'count': int(len(y_true)), 'accuracy': float(accuracy_score(y_true, y_pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'classification_report': classification_report(y_true, y_pred, labels=list(range(3)), target_names=LABELS, output_dict=True, zero_division=0), 'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(range(3))).tolist()}

def _manifest(path: Path, limit: int | None, offset: int) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_model_artifact(path))
    required = {'clip_path', 'clip_name', 'correct_label', 'start_time', 'end_time'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'Manifest missing columns: {missing}')
    frame['correct_label'] = frame['correct_label'].astype(str).str.strip().str.lower()
    if set(frame['correct_label']) - set(LABELS):
        raise ValueError(f"Unexpected labels: {sorted(set(frame['correct_label']) - set(LABELS))}")
    frame['clip_path'] = frame['clip_path'].map(lambda value: Path(str(value)))
    missing_clips = [str(path) for path in frame['clip_path'] if not _resolve_model_artifact(path).exists()]
    if missing_clips:
        raise FileNotFoundError(f'{len(missing_clips)} clips are missing; first={missing_clips[:3]}')
    if offset or limit is not None:
        frame = frame.iloc[offset:None if limit is None else offset + limit]
    return frame.reset_index(drop=True)

def _read_rgb_clip(path: Path, load_raw_video) -> np.ndarray:
    frames = load_raw_video(str(path), max_frames=0, resize=(224, 224), target_fps=25.0)
    if len(frames) == 0:
        raise RuntimeError(f'No RGB frames decoded from {path}')
    return np.asarray(frames, dtype=np.uint8)

def _extract_rgb_features(frame_arrays: list[np.ndarray], detector, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    feature_lists: list[list[np.ndarray]] = [[] for _ in frame_arrays]
    pending_frames: list[np.ndarray] = []
    pending_owner: list[int] = []
    for clip_index, frames in enumerate(frame_arrays):
        for frame in frames:
            pending_frames.append(frame)
            pending_owner.append(clip_index)
            if len(pending_frames) >= batch_size:
                values = detector.feature_extractor.predict(np.asarray(pending_frames, dtype=np.uint8), batch_size=min(64, len(pending_frames)), verbose=0)
                for value, owner in zip(values, pending_owner):
                    feature_lists[owner].append(np.asarray(value, dtype=np.float32))
                pending_frames.clear()
                pending_owner.clear()
    if pending_frames:
        values = detector.feature_extractor.predict(np.asarray(pending_frames, dtype=np.uint8), batch_size=min(64, len(pending_frames)), verbose=0)
        for value, owner in zip(values, pending_owner):
            feature_lists[owner].append(np.asarray(value, dtype=np.float32))
    lengths = np.asarray([len(items) for items in feature_lists], dtype=np.int32)
    padded = np.zeros((len(frame_arrays), RGB_LENGTH, 2048), dtype=np.float32)
    masks = np.zeros((len(frame_arrays), RGB_LENGTH), dtype=np.float32)
    for index, values in enumerate(feature_lists):
        count = min(len(values), RGB_LENGTH)
        if count:
            padded[index, :count] = np.asarray(values[:count], dtype=np.float32)
            masks[index, :count] = 1.0
    return (padded, lengths)

def _extract_yolo_features(paths: list[Path], detector, batch_size: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from models.common.spatial import normalize_yolo_source_roi
    yolo_model = detector.pose_detector
    boxes = np.zeros((len(paths), YOLO_FRAMES, MAX_PEOPLE, 4), dtype=np.float32)
    person_conf = np.zeros((len(paths), YOLO_FRAMES, MAX_PEOPLE), dtype=np.float32)
    keypoints = np.zeros((len(paths), YOLO_FRAMES, MAX_PEOPLE, 17, 2), dtype=np.float32)
    keypoint_conf = np.zeros((len(paths), YOLO_FRAMES, MAX_PEOPLE, 17), dtype=np.float32)
    valid = np.zeros((len(paths), YOLO_FRAMES, MAX_PEOPLE), dtype=bool)
    position_features = np.zeros((len(paths), YOLO_FRAMES, 31), dtype=np.float32)
    pose_features = np.zeros((len(paths), YOLO_FRAMES, 30), dtype=np.float32)
    frame_masks = np.zeros((len(paths), YOLO_FRAMES), dtype=bool)
    pending_rois: list[np.ndarray] = []
    pending_keys: list[tuple[int, int]] = []
    for clip_index, path in enumerate(paths):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f'No se pudo abrir el video fuente para YOLO: {path}')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = np.unique(np.rint(np.linspace(0, max(total - 1, 0), YOLO_FRAMES)).astype(np.int32))
        selected = set((int(value) for value in indices.tolist()))
        selected_order = [int(value) for value in indices.tolist()]
        while len(selected_order) < YOLO_FRAMES:
            selected_order.append(selected_order[-1])
        rois: list[np.ndarray] = []
        frame_index = 0
        try:
            while frame_index < total:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index in selected:
                    roi = normalize_yolo_source_roi(frame, yolo_size=640)
                    if roi is None:
                        raise RuntimeError(f'High-resolution ROI extraction failed for {path}')
                    rois.append(roi)
                frame_index += 1
        finally:
            cap.release()
        if not rois:
            raise RuntimeError(f'No source frames extracted for YOLO: {path}')
        roi_by_source_index = {}
        source_index_order = [int(value) for value in indices.tolist()]
        for source_index, roi in zip(source_index_order, rois):
            roi_by_source_index[source_index] = roi
        for frame_index in range(YOLO_FRAMES):
            roi = roi_by_source_index[selected_order[frame_index]]
            pending_rois.append(roi)
            pending_keys.append((clip_index, frame_index))
            if len(pending_rois) >= batch_size:
                results = yolo_model.predict(pending_rois, imgsz=640, conf=PERSON_CONFIDENCE, device='cpu', verbose=False, batch=batch_size)
                for result, roi, key in zip(results, pending_rois, pending_keys):
                    c, f = key
                    boxes[c, f], person_conf[c, f], keypoints[c, f], keypoint_conf[c, f], valid[c, f] = _decode_result(result, roi)
                pending_rois.clear()
                pending_keys.clear()
    if pending_rois:
        results = yolo_model.predict(pending_rois, imgsz=640, conf=PERSON_CONFIDENCE, device='cpu', verbose=False, batch=len(pending_rois))
        for result, roi, key in zip(results, pending_rois, pending_keys):
            c, f = key
            boxes[c, f], person_conf[c, f], keypoints[c, f], keypoint_conf[c, f], valid[c, f] = _decode_result(result, roi)
    features = np.zeros((len(paths), 982), dtype=np.float32)
    for clip_index in range(len(paths)):
        data = {'boxes': boxes[clip_index], 'person_conf': person_conf[clip_index], 'keypoints': keypoints[clip_index], 'keypoint_conf': keypoint_conf[clip_index], 'valid': valid[clip_index]}
        positions, poses, current_frame_mask = detector._build_geometry_features(boxes[clip_index], person_conf[clip_index], keypoints[clip_index], keypoint_conf[clip_index], valid[clip_index], person_threshold=PERSON_CONFIDENCE, keypoint_threshold=KEYPOINT_CONFIDENCE)
        position_features[clip_index] = positions
        pose_features[clip_index] = poses
        frame_masks[clip_index] = current_frame_mask
        data.update({'position_features': positions, 'pose_features': poses, 'frame_mask': current_frame_mask})
        e04 = detector._make_variant(data, 'E04_tracking_smoothing')
        e06 = detector._make_variant(data, 'E06_rugby_descriptors')
        features[clip_index] = np.concatenate([e04, e06]).astype(np.float32)
    raw = {'boxes': boxes, 'person_conf': person_conf, 'keypoints': keypoints, 'keypoint_conf': keypoint_conf, 'valid': valid, 'position_features': position_features, 'pose_features': pose_features, 'frame_mask': frame_masks}
    return (features, raw)

def _run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args.manifest, args.limit, args.offset)
    from models.M2.predictor import RGBYoloHighresQuantileFusionDetector
    detector = RGBYoloHighresQuantileFusionDetector(REPO)
    if args.pose_model:
        from ultralytics import YOLO
        detector.pose_detector = YOLO(_resolve_model_artifact(str(args.pose_model)))
    from data_utils import load_raw_video
    started = time.perf_counter()
    paths = manifest['clip_path'].tolist()
    rgb_cache_path = args.output / 'rgb_feature_cache.npz'
    if _resolve_model_artifact(rgb_cache_path).exists():
        print(f'Loading cached RGB features from {rgb_cache_path}...', flush=True)
        with _load_model_result_array(rgb_cache_path) as rgb_cache:
            padded = np.asarray(rgb_cache['padded'], dtype=np.float32)
            lengths = np.asarray(rgb_cache['lengths'], dtype=np.int32)
    else:
        print(f'Loading RGB frames for {len(paths)} clips...', flush=True)
        frame_arrays = [_read_rgb_clip(path, load_raw_video) for path in paths]
        print(f'Extracting RGB features in batches...', flush=True)
        padded, lengths = _extract_rgb_features(frame_arrays, detector, args.rgb_batch)
        np.savez(rgb_cache_path, padded=padded, lengths=lengths)
        del frame_arrays
    print(f'Running frozen historical sequence model...', flush=True)
    masks = (np.arange(RGB_LENGTH)[None, :] < np.minimum(lengths, RGB_LENGTH)[:, None]).astype(np.float32)
    historical = detector.sequence_model.predict([padded, masks], batch_size=16, verbose=0)
    historical = detector._normalize(historical)
    dynamic = np.stack([detector._rgb_dynamic_vector(padded[index], int(min(lengths[index], RGB_LENGTH))) for index in range(len(paths))])
    print(f'Extracting YOLO ROI features from 640x360 source frames in batches of {args.yolo_batch}...', flush=True)
    yolo, yolo_raw = _extract_yolo_features(paths, detector, args.yolo_batch)
    np.savez_compressed(args.output / 'yolo_temporal_raw.npz', **yolo_raw)
    auxiliary_features = np.concatenate([dynamic, yolo], axis=1)
    auxiliary = detector.auxiliary_model.predict_proba(detector.quantile_scaler.transform(auxiliary_features))
    auxiliary = detector._normalize(auxiliary)
    fused = detector._normalize(0.4 * historical + 0.6 * auxiliary)
    predicted_ids = np.argmax(fused, axis=1)
    predicted = [LABELS[int(index)] for index in predicted_ids]
    true = manifest['correct_label'].tolist()
    records = manifest.copy()
    records['predicted_label'] = predicted
    records['p_line'] = fused[:, 0]
    records['p_other'] = fused[:, 1]
    records['p_scrum'] = fused[:, 2]
    records['confidence'] = fused.max(axis=1)
    records['correct'] = records['correct_label'] == records['predicted_label']
    records['rgb_frames'] = lengths
    records['yolo_frames'] = YOLO_FRAMES
    records['inference_backend'] = 'batched_equivalent_production_preprocessing'
    records['clip_path'] = records['clip_path'].astype(str)
    records.to_csv(args.output / 'clip_predictions.csv', index=False)
    y_true = [LABEL_TO_ID[label] for label in true]
    y_pred = [LABEL_TO_ID[label] for label in predicted]
    summary = {'experiment': 'external_match_test_40rgb_yolo_highres_20260815', 'execution': 'batched equivalent of production adapter', 'manifest': str(args.manifest.resolve()), 'manifest_rows_processed': int(len(manifest)), 'label_counts': manifest['correct_label'].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(), 'coverage_seconds': {'start': float(manifest['start_time'].min()), 'end': float(manifest['end_time'].max()), 'duration': float(manifest['end_time'].max() - manifest['start_time'].min())}, 'model': {'name': 'rgb_yolo_quantile_highres_roi', 'deployment_artifact': str((REPO / 'data/player_geometry_highres_pose/outputs/deployment_quantile_100frames').resolve()), 'historical_weight': 0.4, 'auxiliary_weight': 0.6, 'rgb_dynamics_features': 704, 'yolo_features': 982, 'yolo_frames_per_clip': YOLO_FRAMES, 'yolo_inference_batch': args.yolo_batch, 'pose_input': 'source-equivalent ROI resized directly to 640x640', 'classifier': 'QuantileTransformer(256, normal) + LogisticRegression(C=.003, moderate weights)'}, 'metrics': _metrics(y_true, y_pred), 'errors': {'count': int((~records['correct']).sum()), 'mean_confidence_correct': float(records.loc[records['correct'], 'confidence'].mean()), 'mean_confidence_incorrect': float(records.loc[~records['correct'], 'confidence'].mean())}, 'runtime_seconds': round(time.perf_counter() - started, 3), 'artifacts': ['clip_predictions.csv', 'summary.json', 'probabilities.npz', 'rgb_feature_cache.npz', 'yolo_temporal_raw.npz']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(args.output / 'probabilities.npz', historical=historical, auxiliary=auxiliary, fused=fused)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--rgb-batch', type=int, default=512)
    parser.add_argument('--pose-model', type=Path, default=None)
    parser.add_argument('--yolo-batch', type=int, default=DEFAULT_YOLO_BATCH)
    _run(parser.parse_args())
if __name__ == '__main__':
    main()
