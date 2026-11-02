"""Extractor YOLO pose batched por clip.

Mantiene el formato de ``pose_cache.py`` pero manda los frames de un clip en
lotes a Ultralytics. Está pensado para comparar pesos YOLO alternativos sin
cambiar las características downstream.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from models.M1.config import DATA_DIR, MAX_PEOPLE, POSE_CACHE_DIR, POSE_CONFIDENCE_DEFAULT, POSE_IMAGE_SIZE_DEFAULT, POSE_KEYPOINT_CONFIDENCE_DEFAULT, POSE_MODEL_DEFAULT
from models.M1.geometry_features import build_geometry_features
from models.M1.pose_cache import crop_training_roi, manifest_for_extraction, sample_indices, safe_model_name

def decode_result(result, frame, max_people):
    boxes = np.zeros((max_people, 4), dtype=np.float32)
    person_conf = np.zeros(max_people, dtype=np.float32)
    keypoints = np.zeros((max_people, 17, 2), dtype=np.float32)
    keypoint_conf = np.zeros((max_people, 17), dtype=np.float32)
    valid = np.zeros(max_people, dtype=bool)
    if result.boxes is None or len(result.boxes) == 0:
        return (boxes, person_conf, keypoints, keypoint_conf, valid)
    raw_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    raw_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    raw_keypoints = result.keypoints.xy.detach().cpu().numpy().astype(np.float32) if result.keypoints is not None and result.keypoints.xy is not None else None
    raw_keypoint_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32) if result.keypoints is not None and result.keypoints.conf is not None else None
    order = np.argsort(raw_scores)[::-1][:max_people]
    width, height = (max(float(frame.shape[1]), 1.0), max(float(frame.shape[0]), 1.0))
    for out_i, src_i in enumerate(order):
        x1, y1, x2, y2 = raw_boxes[src_i]
        boxes[out_i] = np.clip([x1 / width, y1 / height, x2 / width, y2 / height], 0.0, 1.0)
        person_conf[out_i] = raw_scores[src_i]
        valid[out_i] = True
        if raw_keypoints is not None and src_i < len(raw_keypoints):
            keypoints[out_i] = np.clip(raw_keypoints[src_i] / np.array([width, height], dtype=np.float32), 0.0, 1.0)
        if raw_keypoint_conf is not None and src_i < len(raw_keypoint_conf):
            keypoint_conf[out_i] = raw_keypoint_conf[src_i]
    return (boxes, person_conf, keypoints, keypoint_conf, valid)

def extract_clip_batch(model, video_path, frame_count, image_size, person_threshold, keypoint_threshold, max_people, batch_size):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir el video: {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_indices(total, frame_count)
    selected = set((int(i) for i in indices))
    frames = []
    read_count = 0
    try:
        while len(frames) < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            if read_count in selected:
                roi = crop_training_roi(frame, image_size)
                if roi is None:
                    raise RuntimeError(f'No se pudo recortar {video_path}')
                frames.append(roi)
            read_count += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f'No se obtuvieron frames de {video_path}')
    while len(frames) < frame_count:
        frames.append(frames[-1].copy())
    results = model.predict(frames, imgsz=image_size, conf=person_threshold, device='cpu', verbose=False, batch=batch_size)
    boxes = np.zeros((frame_count, max_people, 4), dtype=np.float32)
    person_conf = np.zeros((frame_count, max_people), dtype=np.float32)
    keypoints = np.zeros((frame_count, max_people, 17, 2), dtype=np.float32)
    keypoint_conf = np.zeros((frame_count, max_people, 17), dtype=np.float32)
    valid = np.zeros((frame_count, max_people), dtype=bool)
    for i, result in enumerate(results[:frame_count]):
        boxes[i], person_conf[i], keypoints[i], keypoint_conf[i], valid[i] = decode_result(result, frames[i], max_people)
    positions, poses, frame_mask = build_geometry_features(boxes, person_conf, keypoints, keypoint_conf, valid, person_threshold=person_threshold, keypoint_threshold=keypoint_threshold)
    return ({'boxes': boxes, 'person_conf': person_conf, 'keypoints': keypoints, 'keypoint_conf': keypoint_conf, 'valid': valid, 'position_features': positions, 'pose_features': poses, 'frame_mask': frame_mask}, {'video': video_path.name, 'requested_frames': frame_count, 'processed_frames': min(len(results), frame_count), 'total_frames': total, 'frames_with_persons': int(np.sum(np.any(valid, axis=1))), 'mean_people': float(np.mean(np.sum(valid, axis=1))), 'mean_confidence': float(np.mean(person_conf[valid])) if np.any(valid) else 0.0})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=POSE_MODEL_DEFAULT)
    parser.add_argument('--frame-count', type=int, default=50)
    parser.add_argument('--image-size', type=int, default=POSE_IMAGE_SIZE_DEFAULT)
    parser.add_argument('--person-confidence', type=float, default=POSE_CONFIDENCE_DEFAULT)
    parser.add_argument('--keypoint-confidence', type=float, default=POSE_KEYPOINT_CONFIDENCE_DEFAULT)
    parser.add_argument('--max-people', type=int, default=MAX_PEOPLE)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--per-class', type=int, default=None)
    parser.add_argument('--max-clips', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('YOLO_CONFIG_DIR', str(experiment_directory(__file__) / '.ultralytics'))
    from ultralytics import YOLO
    model = YOLO(_resolve_model_artifact(args.model))
    target = POSE_CACHE_DIR / safe_model_name(args.model)
    target.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for_extraction(args.per_class, args.max_clips, args.seed)
    if args.offset or args.limit is not None:
        end = None if args.limit is None else args.offset + args.limit
        manifest = manifest.iloc[args.offset:end].reset_index(drop=True)
    records = []
    failures = []
    started = time.perf_counter()
    for number, row in enumerate(manifest.to_dict('records'), 1):
        name = str(row['video_name'])
        path = target / f'{Path(name).stem}.npz'
        if _resolve_model_artifact(path).exists() and (not args.overwrite):
            records.append({'video_name': name, 'tag': row['tag'], 'status': 'cached', 'path': str(path)})
            continue
        try:
            arrays, metadata = extract_clip_batch(model, DATA_DIR / name, args.frame_count, args.image_size, args.person_confidence, args.keypoint_confidence, args.max_people, args.batch_size)
            np.savez_compressed(path, **arrays)
            records.append({'video_name': name, 'tag': row['tag'], 'status': 'ok', 'path': str(path), **metadata})
            print(f'[{number}/{len(manifest)}] {name}', flush=True)
        except Exception as exc:
            failures.append({'video_name': name, 'tag': row['tag'], 'status': 'failed', 'error': repr(exc)})
            print(f'[{number}/{len(manifest)}] ERROR {name}: {exc}', flush=True)
    pd.DataFrame(records).to_csv(target / 'extraction_log.csv', index=False)
    pd.DataFrame(failures).to_csv(target / 'extraction_failures.csv', index=False)
    (target / 'metadata.json').write_text(json.dumps({'model': args.model, 'frame_count': args.frame_count, 'batch_size': args.batch_size, 'clips': len(records), 'failures': len(failures), 'seconds': time.perf_counter() - started}, indent=2), encoding='utf-8')
    print(json.dumps({'clips': len(records), 'failures': len(failures), 'seconds': time.perf_counter() - started}, indent=2), flush=True)
if __name__ == '__main__':
    main()
