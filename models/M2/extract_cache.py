"""Build an isolated YOLO-pose cache from the pre-downsampled training clips.

The historical RGB input is produced by scaling each frame by 0.5 and taking
the central 224x224 crop.  This experiment maps that crop back to the source
frame, preserving the same field of view while retaining approximately
448x448 source pixels for YOLO.  Coordinates remain normalized to the shared
field of view, so downstream geometry code can consume the cache unchanged.

The command is read-only by default.  It does not import or load Ultralytics
unless ``--run`` is supplied.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parents[2]
FUSION_DIR = REPO_ROOT / 'data' / 'player_geometry_fusion'
EXPERIMENT_DIR = experiment_directory(__file__)
DEFAULT_SOURCE_DIR = REPO_ROOT / 'data' / '7seg unique'
DEFAULT_REFERENCE_DIR = REPO_ROOT / 'data' / 'videos' / 'clips_224'
DEFAULT_OUTPUT_ROOT = EXPERIMENT_DIR / 'outputs' / 'cache'
FEATURE_MANIFEST = REPO_ROOT / 'data' / 'joint_multimodal_20260813' / 'rgb_dynamics_seed42' / 'fusion_features.npz'
DEFAULT_FRAME_COUNT = 100
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(source_directory(FUSION_DIR)))
from config import MAX_PEOPLE, POSE_CONFIDENCE_DEFAULT, POSE_FRAME_COUNT_DEFAULT, POSE_IMAGE_SIZE_DEFAULT, POSE_KEYPOINT_CONFIDENCE_DEFAULT, POSE_MODEL_DEFAULT
from models.M1.geometry_features import build_geometry_features
from models.M1.pose_cache import safe_model_name, sample_indices
from models.M1.pose_cache_batch import decode_result
from models.common.spatial import extract_training_equivalent_source_roi, training_normalization_geometry

@dataclass(frozen=True)
class RoiMapping:
    source_width: int
    source_height: int
    scaled_width: int
    scaled_height: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def roi_width(self) -> int:
        return self.x1 - self.x0

    @property
    def roi_height(self) -> int:
        return self.y1 - self.y0

def equivalent_roi_mapping(width: int, height: int, *, pre_crop_scale: float=0.5, crop_size: int=224) -> RoiMapping:
    """Map the historical half-scale central crop back to source coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError('Source dimensions must be positive')
    if pre_crop_scale <= 0:
        raise ValueError('pre_crop_scale must be positive')
    if crop_size <= 0:
        raise ValueError('crop_size must be positive')
    reference_height = max(1, int(round(720 * pre_crop_scale)))
    geometry = training_normalization_geometry(width, height, target_size=crop_size, reference_height=reference_height)
    scaled_width = int(geometry['resized_width'])
    scaled_height = int(geometry['resized_height'])
    if scaled_width < crop_size or scaled_height < crop_size:
        raise ValueError(f'Scaled frame {scaled_width}x{scaled_height} is smaller than the {crop_size}x{crop_size} training crop')
    x0 = int(round(float(geometry['source_roi_x'])))
    y0 = int(round(float(geometry['source_roi_y'])))
    x1 = int(round(float(geometry['source_roi_x'] + geometry['source_roi_width'])))
    y1 = int(round(float(geometry['source_roi_y'] + geometry['source_roi_height'])))
    x0, y0 = (max(0, x0), max(0, y0))
    x1, y1 = (min(width, x1), min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Equivalent ROI is empty')
    return RoiMapping(source_width=width, source_height=height, scaled_width=scaled_width, scaled_height=scaled_height, x0=x0, y0=y0, x1=x1, y1=y1)

def equivalent_highres_roi(frame: np.ndarray, *, pre_crop_scale: float=0.5, crop_size: int=224) -> tuple[np.ndarray, RoiMapping]:
    height, width = frame.shape[:2]
    mapping = equivalent_roi_mapping(width, height, pre_crop_scale=pre_crop_scale, crop_size=crop_size)
    roi = extract_training_equivalent_source_roi(frame, target_size=crop_size, reference_height=max(1, int(round(720 * pre_crop_scale))))
    if roi.size == 0:
        raise ValueError('Equivalent high-resolution ROI is empty')
    return (roi, mapping)

def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {'open': False, 'width': 0, 'height': 0, 'fps': 0.0, 'frames': 0}
    info = {'open': True, 'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0), 'fps': float(cap.get(cv2.CAP_PROP_FPS) or 0.0), 'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)}
    cap.release()
    return info

def preflight_manifest(manifest: pd.DataFrame, source_dir: Path, reference_dir: Path, *, pre_crop_scale: float, crop_size: int) -> pd.DataFrame:
    rows: list[dict] = []
    for row in manifest.to_dict('records'):
        video_name = str(row['video_name'])
        source_path = source_dir / video_name
        reference_path = reference_dir / video_name
        record = {'video_name': video_name, 'tag': str(row['tag']), 'source_path': str(source_path), 'reference_path': str(reference_path)}
        if not _resolve_model_artifact(source_path).exists():
            rows.append({**record, 'status': 'missing_source'})
            continue
        if not _resolve_model_artifact(reference_path).exists():
            rows.append({**record, 'status': 'missing_reference'})
            continue
        source = video_info(source_path)
        reference = video_info(reference_path)
        status = 'ok'
        detail = ''
        if not source['open'] or not reference['open']:
            status, detail = ('open_failed', 'Source or reference video could not be opened')
        elif reference['width'] != crop_size or reference['height'] != crop_size:
            status, detail = ('invalid_reference_shape', f"Reference is {reference['width']}x{reference['height']}")
        elif source['frames'] != reference['frames']:
            status, detail = ('frame_count_mismatch', f"source={source['frames']} reference={reference['frames']}")
        elif abs(source['fps'] - reference['fps']) > 0.001:
            status, detail = ('fps_mismatch', f"source={source['fps']} reference={reference['fps']}")
        else:
            try:
                mapping = equivalent_roi_mapping(source['width'], source['height'], pre_crop_scale=pre_crop_scale, crop_size=crop_size)
                record.update({'roi_width': mapping.roi_width, 'roi_height': mapping.roi_height, 'roi_x0': mapping.x0, 'roi_y0': mapping.y0, 'roi_x1': mapping.x1, 'roi_y1': mapping.y1})
            except ValueError as exc:
                status, detail = ('invalid_source_geometry', str(exc))
        record.update({'source_width': source['width'], 'source_height': source['height'], 'source_fps': source['fps'], 'source_frames': source['frames'], 'reference_frames': reference['frames'], 'status': status, 'detail': detail})
        rows.append(record)
    return pd.DataFrame(rows)

def extract_highres_clip(model, video_path: Path, *, frame_count: int, image_size: int, person_threshold: float, keypoint_threshold: float, max_people: int, pre_crop_scale: float, crop_size: int, batch_size: int, device: str) -> tuple[dict[str, np.ndarray], dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = sample_indices(total, frame_count)
    selected = set((int(index) for index in indices))
    boxes = np.zeros((frame_count, max_people, 4), dtype=np.float32)
    person_conf = np.zeros((frame_count, max_people), dtype=np.float32)
    keypoints = np.zeros((frame_count, max_people, 17, 2), dtype=np.float32)
    keypoint_conf = np.zeros((frame_count, max_people, 17), dtype=np.float32)
    valid = np.zeros((frame_count, max_people), dtype=bool)
    output_index = 0
    read_count = 0
    first_mapping: RoiMapping | None = None
    selected_rois: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        while output_index < frame_count:
            ok, frame = cap.read()
            if not ok:
                break
            if read_count not in selected:
                read_count += 1
                continue
            roi, mapping = equivalent_highres_roi(frame, pre_crop_scale=pre_crop_scale, crop_size=crop_size)
            if first_mapping is None:
                first_mapping = mapping
            selected_rois.append(roi)
            output_index += 1
            read_count += 1
    finally:
        cap.release()
    if output_index == 0:
        raise RuntimeError(f'No frames were extracted from {video_path}')
    while len(selected_rois) < frame_count:
        selected_rois.append(selected_rois[-1].copy())
    decoded_index = 0
    for start in range(0, frame_count, max(1, int(batch_size))):
        chunk = selected_rois[start:start + max(1, int(batch_size))]
        actual_count = len(chunk)
        while len(chunk) < max(1, int(batch_size)):
            chunk.append(chunk[-1].copy())
        results = model.predict(chunk, imgsz=image_size, conf=person_threshold, device=device, verbose=False, batch=max(1, int(batch_size)))
        for result, roi in zip(results[:actual_count], chunk[:actual_count]):
            detected = decode_result(result, roi, max_people)
            boxes[decoded_index], person_conf[decoded_index], keypoints[decoded_index], keypoint_conf[decoded_index], valid[decoded_index] = detected
            decoded_index += 1
    if decoded_index != frame_count:
        raise RuntimeError(f'YOLO returned {decoded_index} detections for {frame_count} requested frames')
    positions, poses, frame_mask = build_geometry_features(boxes, person_conf, keypoints, keypoint_conf, valid, person_threshold=person_threshold, keypoint_threshold=keypoint_threshold)
    arrays = {'boxes': boxes, 'person_conf': person_conf, 'keypoints': keypoints, 'keypoint_conf': keypoint_conf, 'valid': valid, 'position_features': positions, 'pose_features': poses, 'frame_mask': frame_mask}
    metadata = {'video': video_path.name, 'requested_frames': int(frame_count), 'processed_frames': int(output_index), 'total_frames': int(total), 'frames_with_persons': int(np.any(valid, axis=1).sum()), 'mean_people': float(valid.sum(axis=1).mean()), 'mean_confidence': float(person_conf[valid].mean()) if np.any(valid) else 0.0, 'elapsed_seconds': round(time.perf_counter() - started, 4), 'roi_mapping': asdict(first_mapping) if first_mapping is not None else None}
    return (arrays, metadata)
_WORKER_MODEL = None
_WORKER_SETTINGS: dict | None = None

def _init_worker(model_path: str, settings: dict) -> None:
    global _WORKER_MODEL, _WORKER_SETTINGS
    os.environ.setdefault('YOLO_CONFIG_DIR', str(EXPERIMENT_DIR / '.ultralytics'))
    try:
        import torch
        torch.set_num_threads(max(1, int(os.environ.get('HIGHRES_POSE_WORKER_THREADS', '4'))))
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if 'openvino' in str(model_path).lower():
        import openvino as ov
        thread_budget = max(1, int(os.environ.get('HIGHRES_POSE_WORKER_THREADS', '2')))
        original_compile_model = ov.Core.compile_model

        def compile_model_with_budget(core, *args, **kwargs):
            config = dict(kwargs.get('config') or {})
            config[str(ov.properties.inference_num_threads)] = thread_budget
            kwargs['config'] = config
            return original_compile_model(core, *args, **kwargs)
        ov.Core.compile_model = compile_model_with_budget
    if str(settings.get('device', 'cpu')).lower() == 'dml':
        import onnxruntime as ort
        original_session = ort.InferenceSession

        def dml_session(*args, **kwargs):
            kwargs['providers'] = ['DmlExecutionProvider', 'CPUExecutionProvider']
            return original_session(*args, **kwargs)
        ort.InferenceSession = dml_session
        settings = dict(settings)
        settings['device'] = 'cpu'
    from ultralytics import YOLO
    _WORKER_MODEL = YOLO(_resolve_model_artifact(model_path))
    _WORKER_SETTINGS = settings

def _process_task(task: dict) -> dict:
    output_path = Path(task['output_path'])
    if _resolve_model_artifact(output_path).exists() and (not task['overwrite']):
        return {'video_name': task['video_name'], 'tag': task['tag'], 'status': 'cached', 'path': str(output_path)}
    started = time.perf_counter()
    try:
        arrays, metadata = extract_highres_clip(_WORKER_MODEL, Path(task['source_path']), **_WORKER_SETTINGS)
        np.savez_compressed(output_path, **arrays)
        metadata['worker_elapsed_seconds'] = round(time.perf_counter() - started, 4)
        return {'video_name': task['video_name'], 'tag': task['tag'], 'status': 'ok', 'path': str(output_path), **metadata}
    except Exception as exc:
        return {'video_name': task['video_name'], 'tag': task['tag'], 'status': 'error', 'error': repr(exc)}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument('--reference-dir', type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--model', default=POSE_MODEL_DEFAULT)
    parser.add_argument('--cache-name', default=None, help='Stable cache subdirectory; useful for an equivalent optimized backend')
    parser.add_argument('--frame-count', type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument('--image-size', type=int, default=POSE_IMAGE_SIZE_DEFAULT)
    parser.add_argument('--person-confidence', type=float, default=POSE_CONFIDENCE_DEFAULT)
    parser.add_argument('--keypoint-confidence', type=float, default=POSE_KEYPOINT_CONFIDENCE_DEFAULT)
    parser.add_argument('--max-people', type=int, default=MAX_PEOPLE)
    parser.add_argument('--pre-crop-scale', type=float, default=0.5)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--per-class', type=int)
    parser.add_argument('--max-clips', type=int)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--run', action='store_true', help='Load YOLO and write the isolated cache')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if not _resolve_model_artifact(args.source_dir).is_dir():
        raise SystemExit(f'High-resolution source directory not found: {args.source_dir}')
    if not _resolve_model_artifact(args.reference_dir).is_dir():
        raise SystemExit(f'RGB reference directory not found: {args.reference_dir}')
    if not _resolve_model_artifact(FEATURE_MANIFEST).exists():
        raise SystemExit(f'Frozen 4,082-clip feature manifest not found: {FEATURE_MANIFEST}')
    frozen = _load_model_result_array(FEATURE_MANIFEST, allow_pickle=True)
    names = frozen['names'].astype(str)
    labels = frozen['labels'].astype(int)
    label_names = np.asarray(['line', 'other', 'scrum'], dtype=object)
    manifest = pd.DataFrame({'video_name': names, 'tag': label_names[labels]})
    if args.per_class is not None:
        manifest = pd.concat([group.sample(n=min(args.per_class, len(group)), random_state=args.seed) for _, group in manifest.groupby('tag', sort=True)], ignore_index=True)
    if args.max_clips is not None:
        manifest = manifest.sample(n=min(args.max_clips, len(manifest)), random_state=args.seed)
    manifest = manifest.reset_index(drop=True)
    preflight = preflight_manifest(manifest, args.source_dir, args.reference_dir, pre_crop_scale=args.pre_crop_scale, crop_size=args.crop_size)
    status_counts = preflight['status'].value_counts().to_dict()
    print(json.dumps({'experiment': 'highres_equivalent_roi_v1', 'clips': int(len(preflight)), 'status_counts': {str(key): int(value) for key, value in status_counts.items()}, 'source_dir': str(args.source_dir.resolve()), 'reference_dir': str(args.reference_dir.resolve()), 'output_root': str(args.output_root.resolve())}, indent=2))
    invalid = preflight.loc[preflight['status'] != 'ok']
    if not invalid.empty:
        print(invalid[['video_name', 'status', 'detail']].head(20).to_string(index=False))
        if args.run:
            raise SystemExit('Preflight failed; YOLO was not loaded and no cache was written')
    if not args.run:
        print('Dry run only. YOLO was not loaded. Add --run to create the isolated cache.')
        return 0
    target_dir = args.output_root / (args.cache_name or safe_model_name(args.model))
    target_dir.mkdir(parents=True, exist_ok=True)
    preflight.to_csv(target_dir / 'source_preflight.csv', index=False)
    settings = {'frame_count': args.frame_count, 'image_size': args.image_size, 'person_threshold': args.person_confidence, 'keypoint_threshold': args.keypoint_confidence, 'max_people': args.max_people, 'pre_crop_scale': args.pre_crop_scale, 'crop_size': args.crop_size, 'batch_size': args.batch_size, 'device': args.device}
    tasks = [{'video_name': row['video_name'], 'tag': row['tag'], 'source_path': row['source_path'], 'output_path': str(target_dir / f"{Path(row['video_name']).stem}.npz"), 'overwrite': bool(args.overwrite)} for row in preflight.to_dict('records')]
    workers = max(1, int(args.workers))
    model_path = str(Path(args.model).resolve()) if _resolve_model_artifact(Path(args.model)).exists() else args.model
    records: list[dict] = []
    context = mp.get_context('spawn')
    with context.Pool(processes=workers, initializer=_init_worker, initargs=(model_path, settings)) as pool:
        for number, record in enumerate(pool.imap_unordered(_process_task, tasks), start=1):
            records.append(record)
            if number == 1 or number % 25 == 0 or number == len(tasks):
                print(f"Processed {number}/{len(tasks)}: {record['video_name']} ({record['status']})", flush=True)
    records.sort(key=lambda item: item.get('video_name', ''))
    pd.DataFrame(records).to_csv(target_dir / 'extraction_log.csv', index=False)
    metadata = {'experiment': 'highres_equivalent_roi_v1', 'model': args.model, 'cache_name': target_dir.name, 'model_directory': str(target_dir), 'source_dir': str(args.source_dir.resolve()), 'reference_dir': str(args.reference_dir.resolve()), 'frame_count': args.frame_count, 'image_size': args.image_size, 'pre_crop_scale': args.pre_crop_scale, 'crop_size': args.crop_size, 'person_confidence': args.person_confidence, 'keypoint_confidence': args.keypoint_confidence, 'max_people': args.max_people, 'workers': workers, 'batch_size': args.batch_size, 'device': args.device, 'clips_requested': len(records), 'clips_ok': sum((item.get('status') in {'ok', 'cached'} for item in records)), 'clips_failed': sum((item.get('status') == 'error' for item in records)), 'baseline_cache_modified': False}
    (target_dir / 'cache_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    return 0 if metadata['clips_failed'] == 0 else 2
if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
