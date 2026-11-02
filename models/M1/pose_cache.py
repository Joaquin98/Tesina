"""Extracción de detecciones multipersona y construcción de la caché geométrica.

Ejemplos:

    python models/M1/pose_cache.py --max-clips 2
    python models/M1/pose_cache.py --per-class 30

La ejecución completa puede ser costosa en CPU. Se recomienda empezar con
``--max-clips`` y revisar ``extraction_log.csv`` antes de procesar el corpus.
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
from models.M1.config import DATA_DIR, DEFAULT_MODELS, GEOMETRY_LENGTH, MAX_PEOPLE, NATURAL_TRAIN_CSV, POSE_CONFIDENCE_DEFAULT, POSE_FRAME_COUNT_DEFAULT, POSE_IMAGE_SIZE_DEFAULT, POSE_KEYPOINT_CONFIDENCE_DEFAULT, POSE_MODEL_DEFAULT, POSE_CACHE_DIR, TEST_CSV
from models.M1.geometry_features import build_geometry_features

def crop_training_roi(frame: np.ndarray, target_size: int) -> np.ndarray | None:
    """Aplica el mismo reescalado y recorte central del corpus de entrenamiento."""
    crop_size = 224
    height, width = frame.shape[:2]
    if height == crop_size and width == crop_size:
        return frame if target_size == crop_size else cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    if height <= 256 and width <= 256:
        square = min(height, width)
        start_x = max((width - square) // 2, 0)
        start_y = max((height - square) // 2, 0)
        crop = frame[start_y:start_y + square, start_x:start_x + square]
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        return crop if target_size == crop_size else cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    scaled_width = max(1, int(round(width * 0.5)))
    scaled_height = max(1, int(round(height * 0.5)))
    if scaled_width < crop_size or scaled_height < crop_size:
        return None
    resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
    start_x = (scaled_width - crop_size) // 2
    start_y = (scaled_height - crop_size) // 2
    crop = resized[start_y:start_y + crop_size, start_x:start_x + crop_size]
    if target_size == crop_size:
        return crop
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

def safe_model_name(model_name: str) -> str:
    return model_name.replace('/', '_').replace('\\', '_').replace('.', '_')

def output_dir(model_name: str) -> Path:
    return POSE_CACHE_DIR / safe_model_name(model_name)

def sample_indices(total_frames: int, count: int) -> np.ndarray:
    if total_frames <= 0:
        return np.arange(count, dtype=np.int64)
    if count <= 1:
        return np.array([total_frames // 2], dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, total_frames - 1, count)).astype(np.int64))

def manifest_for_extraction(per_class: int | None, max_clips: int | None, seed: int) -> pd.DataFrame:
    train = pd.read_csv(_resolve_model_artifact(NATURAL_TRAIN_CSV))[['video_name', 'tag']]
    test = pd.read_csv(_resolve_model_artifact(TEST_CSV))[['video_name', 'tag']]
    df = pd.concat([train, test], ignore_index=True)
    df['video_name'] = df['video_name'].astype(str).str.strip()
    df['tag'] = df['tag'].astype(str).str.strip().str.lower()
    df = df.drop_duplicates('video_name', keep='first')
    if per_class is not None:
        chunks = []
        for label, group in df.groupby('tag', sort=True):
            chunks.append(group.sample(n=min(per_class, len(group)), random_state=seed))
        df = pd.concat(chunks, ignore_index=True)
    if max_clips is not None:
        df = df.sample(n=min(max_clips, len(df)), random_state=seed)
    return df.reset_index(drop=True)

def detect_frame(model, frame: np.ndarray, image_size: int, person_threshold: float, max_people: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve cajas, confianza de persona, puntos, confianza de puntos y máscara."""
    result = model.predict(frame, imgsz=image_size, conf=person_threshold, device='cpu', verbose=False)[0]
    boxes = np.zeros((max_people, 4), dtype=np.float32)
    person_conf = np.zeros(max_people, dtype=np.float32)
    keypoints = np.zeros((max_people, 17, 2), dtype=np.float32)
    keypoint_conf = np.zeros((max_people, 17), dtype=np.float32)
    valid = np.zeros(max_people, dtype=bool)
    if result.boxes is None or len(result.boxes) == 0:
        return (boxes, person_conf, keypoints, keypoint_conf, valid)
    raw_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    raw_scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    raw_keypoints = None
    raw_keypoint_conf = None
    if result.keypoints is not None and result.keypoints.xy is not None:
        raw_keypoints = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)
        if result.keypoints.conf is not None:
            raw_keypoint_conf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)
    order = np.argsort(raw_scores)[::-1][:max_people]
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

def extract_clip(model, video_path: Path, frame_count: int, image_size: int, person_threshold: float, keypoint_threshold: float, max_people: int) -> tuple[dict[str, np.ndarray], dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir el video: {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_indices(total, frame_count)
    selected = set((int(index) for index in indices))
    boxes = np.zeros((frame_count, max_people, 4), dtype=np.float32)
    person_conf = np.zeros((frame_count, max_people), dtype=np.float32)
    keypoints = np.zeros((frame_count, max_people, 17, 2), dtype=np.float32)
    keypoint_conf = np.zeros((frame_count, max_people, 17), dtype=np.float32)
    valid = np.zeros((frame_count, max_people), dtype=bool)
    read_count = 0
    output_index = 0
    started = time.perf_counter()
    try:
        while output_index < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            if read_count not in selected:
                read_count += 1
                continue
            roi = crop_training_roi(frame, image_size)
            if roi is None:
                raise RuntimeError(f'El video no permite recortar una región de {image_size}x{image_size}: {video_path}')
            detected = detect_frame(model, roi, image_size, person_threshold, max_people)
            boxes[output_index], person_conf[output_index], keypoints[output_index], keypoint_conf[output_index], valid[output_index] = detected
            output_index += 1
            read_count += 1
    finally:
        cap.release()
    if output_index == 0:
        raise RuntimeError(f'No se obtuvieron fotogramas de {video_path}')
    if output_index < frame_count:
        boxes[output_index:] = boxes[output_index - 1]
        person_conf[output_index:] = person_conf[output_index - 1]
        keypoints[output_index:] = keypoints[output_index - 1]
        keypoint_conf[output_index:] = keypoint_conf[output_index - 1]
        valid[output_index:] = valid[output_index - 1]
    positions, poses, frame_mask = build_geometry_features(boxes, person_conf, keypoints, keypoint_conf, valid, person_threshold=person_threshold, keypoint_threshold=keypoint_threshold)
    metadata = {'video': video_path.name, 'requested_frames': int(frame_count), 'processed_frames': int(output_index), 'total_frames': int(total), 'frames_with_persons': int(np.sum(np.any(valid, axis=1))), 'mean_people': float(np.mean(np.sum(valid, axis=1))), 'mean_confidence': float(np.mean(person_conf[valid])) if np.any(valid) else 0.0, 'elapsed_seconds': round(time.perf_counter() - started, 4)}
    arrays = {'boxes': boxes, 'person_conf': person_conf, 'keypoints': keypoints, 'keypoint_conf': keypoint_conf, 'valid': valid, 'position_features': positions, 'pose_features': poses, 'frame_mask': frame_mask}
    return (arrays, metadata)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=POSE_MODEL_DEFAULT, help='Peso Ultralytics, por ejemplo yolo11n-pose.pt')
    parser.add_argument('--frame-count', type=int, default=POSE_FRAME_COUNT_DEFAULT)
    parser.add_argument('--image-size', type=int, default=POSE_IMAGE_SIZE_DEFAULT)
    parser.add_argument('--person-confidence', type=float, default=POSE_CONFIDENCE_DEFAULT)
    parser.add_argument('--keypoint-confidence', type=float, default=POSE_KEYPOINT_CONFIDENCE_DEFAULT)
    parser.add_argument('--max-people', type=int, default=MAX_PEOPLE)
    parser.add_argument('--per-class', type=int, default=None, help='Cantidad máxima de clips por clase')
    parser.add_argument('--max-clips', type=int, default=None)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    try:
        local_yolo_config = experiment_directory(__file__) / '.ultralytics'
        local_yolo_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(local_yolo_config))
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Falta ultralytics. Instalar con: pip install ultralytics') from exc
    print(f'Cargando estimador de pose: {args.model}')
    try:
        model = YOLO(_resolve_model_artifact(args.model))
    except Exception as exc:
        raise SystemExit(f'No se pudo cargar {args.model}. Si es la primera ejecución, se requiere descargar el peso. Detalle: {exc}') from exc
    target_dir = output_dir(args.model)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for_extraction(args.per_class, args.max_clips, args.seed)
    records: list[dict] = []
    failures: list[dict] = []
    print(f'Clips seleccionados: {len(manifest)} | salida: {target_dir}')
    for number, row in enumerate(manifest.to_dict('records'), start=1):
        video_name = str(row['video_name'])
        output_path = target_dir / f'{Path(video_name).stem}.npz'
        if _resolve_model_artifact(output_path).exists() and (not args.overwrite):
            records.append({'video_name': video_name, 'tag': row['tag'], 'status': 'cached', 'path': str(output_path)})
            continue
        video_path = DATA_DIR / video_name
        print(f'[{number}/{len(manifest)}] {video_name}')
        try:
            arrays, metadata = extract_clip(model, video_path, args.frame_count, args.image_size, args.person_confidence, args.keypoint_confidence, args.max_people)
            np.savez_compressed(output_path, **arrays)
            records.append({'video_name': video_name, 'tag': row['tag'], 'status': 'ok', 'path': str(output_path), **metadata})
        except Exception as exc:
            failures.append({'video_name': video_name, 'tag': row['tag'], 'status': 'error', 'error': repr(exc)})
            print(f'  ERROR: {exc}')
    log = pd.DataFrame(records + failures)
    log.to_csv(target_dir / 'extraction_log.csv', index=False)
    metadata = {'model': args.model, 'model_directory': str(target_dir), 'frame_count': args.frame_count, 'image_size': args.image_size, 'person_confidence': args.person_confidence, 'keypoint_confidence': args.keypoint_confidence, 'max_people': args.max_people, 'clips_requested': int(len(manifest)), 'clips_ok': int(sum((row.get('status') in {'ok', 'cached'} for row in records))), 'clips_failed': int(len(failures))}
    (target_dir / 'cache_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    if failures:
        print(f"Se registraron {len(failures)} fallos en {target_dir / 'extraction_log.csv'}")
if __name__ == '__main__':
    main()
