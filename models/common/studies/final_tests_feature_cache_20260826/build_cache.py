"""Build resumable ResNet-50 and high-resolution YOLO caches for final tests.

Only complete, fully labelled matches are selected. A match is complete when
every canonical ``V_n_Sxx`` section has one complete fixed-block review
session. Labels are metadata only: neither extractor receives them.
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
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import cv2
import numpy as np
import pandas as pd
ROOT = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
DEFAULT_OUTPUT = HERE / 'outputs'
FINAL_VIDEO_ROOT = ROOT / 'data' / 'videos' / 'partidos_evaluacion'
REVIEW_ROOT = ROOT / 'data' / 'active_learning_review' / 'sessions'
LEGACY_V1_RGB = ROOT / 'data' / 'v1_distance_aware_comparison_20260826' / 'outputs' / 'rgb_cache'
LABELS = {'line', 'other', 'scrum'}
SECTION_PATTERN = re.compile('^V_(?P<match>\\d+)_S(?P<section>\\d+)_', re.IGNORECASE)
RGB_LENGTH = 200
RGB_FEATURES = 2048
YOLO_FRAMES = 100
MAX_PEOPLE = 30
PERSON_CONFIDENCE = 0.25
KEYPOINT_CONFIDENCE = 0.2
PROTOCOL = {'version': 'final_tests_resnet50_yolo_highres_v1', 'block_seconds': 7.0, 'labels': ['line', 'other', 'scrum'], 'no_event_included': False, 'resnet': {'extractor': 'ResNet-50 ImageNet', 'target_fps': 25.0, 'max_frames': RGB_LENGTH, 'feature_dim': RGB_FEATURES, 'spatial_preprocessing': 'reference_height_360_center224_v1', 'input_size': 224, 'stored_dtype': 'float16'}, 'yolo': {'model': 'yolo11n-pose.pt', 'frames': YOLO_FRAMES, 'sampling': 'uniform over the 25 FPS seven-second block timeline', 'spatial_preprocessing': 'equivalent_source_roi_to_640_v1', 'input_size': 640, 'person_confidence': PERSON_CONFIDENCE, 'keypoint_confidence': KEYPOINT_CONFIDENCE, 'max_people': MAX_PEOPLE, 'stored_dtype': 'float32'}}
PROTOCOL_SIGNATURE = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode('utf-8')).hexdigest()[:16]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def file_identity(path: Path) -> tuple[int, int]:
    stat = path.resolve().stat()
    return (int(stat.st_dev), int(stat.st_ino))

def completed_sessions_by_video() -> dict[tuple[int, int], dict]:
    selected: dict[tuple[int, int], tuple[tuple, dict]] = {}
    for path in sorted(_model_artifact_glob(REVIEW_ROOT, 'review_session_*.json', recursive=False)):
        try:
            session = json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8'))
            video_path = Path(str(session.get('video_path') or '')).resolve()
            identity = file_identity(video_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if session.get('review_mode') != 'clip_generator':
            continue
        candidates = session.get('candidates') or []
        labels = [item.get('final_label') for item in candidates]
        complete = bool(candidates) and all((label in LABELS for label in labels))
        rank = (int(complete), sum((label in LABELS for label in labels)), str(session.get('updated_at') or ''))
        current = selected.get(identity)
        if current is None or rank > current[0]:
            session['_session_file'] = str(path.resolve())
            selected[identity] = (rank, session)
    return {identity: session for identity, (rank, session) in selected.items() if rank[0] == 1}

def canonical_sections() -> dict[str, list[tuple[int, Path]]]:
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(_model_artifact_glob(FINAL_VIDEO_ROOT, '*.mp4', recursive=False)):
        match = SECTION_PATTERN.match(path.name)
        if match is None:
            continue
        match_id = f"V_{int(match.group('match'))}"
        grouped.setdefault(match_id, []).append((int(match.group('section')), path.resolve()))
    return grouped

def build_manifest(match_filter: set[str] | None=None) -> tuple[pd.DataFrame, dict]:
    sessions = completed_sessions_by_video()
    groups = canonical_sections()
    rows: list[dict] = []
    audit: dict[str, dict] = {}
    for match_id, sections in sorted(groups.items(), key=lambda item: int(item[0].split('_')[1])):
        if match_filter and match_id not in match_filter:
            continue
        section_ids = [index for index, _ in sections]
        duplicate_sections = sorted((index for index, count in Counter(section_ids).items() if count > 1))
        missing = []
        selected_sections = []
        for section_index, path in sections:
            session = sessions.get(file_identity(path))
            if session is None:
                missing.append(path.name)
            else:
                selected_sections.append((section_index, path, session))
        complete = not duplicate_sections and (not missing) and bool(selected_sections)
        audit[match_id] = {'complete': complete, 'section_files': len(sections), 'duplicate_section_numbers': duplicate_sections, 'missing_or_incomplete_sections': missing, 'labelled_sections': len(selected_sections)}
        if not complete:
            continue
        for section_index, path, session in sorted(selected_sections):
            candidates = sorted(session['candidates'], key=lambda item: int(item.get('candidate_index', 0)))
            for candidate in candidates:
                block_index = int(candidate.get('candidate_index', 0))
                detection_index = int(candidate.get('detection_index', block_index))
                dataset_id = f"{session['session_id']}_{detection_index:04d}"
                rows.append({'dataset_id': dataset_id, 'match': match_id, 'section': f'S{section_index:02d}', 'section_file': path.name, 'source_video': str(path), 'source_size': int(path.stat().st_size), 'source_mtime_ns': int(path.stat().st_mtime_ns), 'session_id': str(session['session_id']), 'session_file': str(session['_session_file']), 'block_index': block_index, 'start_time': float(candidate['start_time']), 'end_time': float(candidate['end_time']), 'duration': float(candidate.get('duration') or 0.0), 'true_label': str(candidate['final_label'])})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError('No fully labelled final match was found')
    frame = frame.sort_values(['match', 'section', 'block_index'], kind='mergesort').reset_index(drop=True)
    if frame.dataset_id.duplicated().any():
        raise ValueError('Duplicate dataset IDs in final-test manifest')
    if set(frame.true_label) - LABELS:
        raise ValueError(f'Unexpected labels: {sorted(set(frame.true_label) - LABELS)}')
    for (_match, _section), group in frame.groupby(['match', 'section'], sort=True):
        expected = np.arange(len(group), dtype=np.int64)
        if not np.array_equal(group.block_index.to_numpy(np.int64), expected):
            raise ValueError(f'Non-consecutive blocks in {_match}/{_section}')
    return (frame, audit)

def sampled_block_indices(fps: float, total: int, start_time: float, end_time: float) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB and YOLO source indices for one labelled time interval.

    This pure helper keeps the temporal contract independently testable without
    opening a video or loading either feature extractor.  It intentionally
    preserves the frozen v1 behaviour, including repeated YOLO indices when an
    end-of-section block contains fewer than 100 samples at 25 FPS.
    """
    fps = float(fps)
    total = int(total)
    start_time = float(start_time)
    end_time = float(end_time)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Source FPS must be positive and finite')
    if total <= 0:
        raise ValueError('Source frame count must be positive')
    if not math.isfinite(start_time) or not math.isfinite(end_time) or end_time <= start_time:
        raise ValueError('Block times must be finite and end after start')
    start = max(0, int(float(start_time) * fps))
    stop = min(total, max(start + 1, int(float(end_time) * fps)))
    if start >= total:
        raise ValueError('Block starts outside the source video')
    sample_count = max(1, int(np.floor(max(0.0, end_time - start_time) * 25.0)))
    rgb = np.rint(start + np.arange(sample_count, dtype=np.float64) * fps / 25.0).astype(np.int64)
    rgb = np.clip(rgb, start, stop - 1)
    yolo_positions = np.rint(np.linspace(0, len(rgb) - 1, YOLO_FRAMES)).astype(np.int64)
    yolo = rgb[yolo_positions]
    return (rgb, yolo)

def block_frame_indices(path: Path, start_time: float, end_time: float) -> tuple[float, int, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open source video: {path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    rgb, yolo = sampled_block_indices(fps, total, start_time, end_time)
    return (fps, total, rgb, yolo)

def read_block_frames(row, *, need_rgb: bool, need_yolo: bool) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    from models.common.spatial import normalize_training_frame, normalize_yolo_source_roi
    path = Path(str(row.source_video))
    fps, total, rgb_indices, yolo_indices = block_frame_indices(path, row.start_time, row.end_time)
    wanted = set((int(value) for value in (rgb_indices if need_rgb else yolo_indices).tolist()))
    if need_rgb and need_yolo:
        wanted.update((int(value) for value in yolo_indices.tolist()))
    first, last = (min(wanted), max(wanted))
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    decoded: dict[int, np.ndarray] = {}
    current = first
    try:
        while current <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if current in wanted:
                decoded[current] = frame
            current += 1
    finally:
        cap.release()
    missing = sorted(wanted - set(decoded))
    if missing:
        raise RuntimeError(f'Could not decode {len(missing)} selected frames from {path.name}; first={missing[:3]}')
    rgb_frames = None
    if need_rgb:
        prepared = []
        for index in rgb_indices[:RGB_LENGTH]:
            frame = normalize_training_frame(decoded[int(index)], target_size=224)
            if frame is None or frame.shape[:2] != (224, 224):
                raise RuntimeError(f'RGB normalization failed for {path.name}')
            prepared.append(frame[:, :, ::-1].copy())
        rgb_frames = np.asarray(prepared, dtype=np.uint8)
    yolo_frames = None
    if need_yolo:
        roi_cache: dict[int, np.ndarray] = {}
        prepared = []
        for index in yolo_indices:
            source_index = int(index)
            if source_index not in roi_cache:
                roi = normalize_yolo_source_roi(decoded[source_index], yolo_size=640)
                if roi is None or roi.shape[:2] != (640, 640):
                    raise RuntimeError(f'YOLO ROI normalization failed for {path.name}')
                roi_cache[source_index] = roi
            prepared.append(roi_cache[source_index])
        yolo_frames = np.asarray(prepared, dtype=np.uint8)
    metadata = {'source_fps': fps, 'source_frames': total, 'rgb_requested': int(min(len(rgb_indices), RGB_LENGTH)), 'yolo_requested': int(len(yolo_indices))}
    return (rgb_frames, yolo_frames, metadata)

def valid_resnet_cache(path: Path) -> bool:
    if not _resolve_model_artifact(path).exists():
        return False
    try:
        with _load_model_result_array(path) as data:
            return data['features'].shape == (RGB_LENGTH, RGB_FEATURES) and data['mask'].shape == (RGB_LENGTH,) and (0 < int(data['length']) <= RGB_LENGTH)
    except (OSError, KeyError, ValueError):
        return False

def valid_yolo_cache(path: Path) -> bool:
    if not _resolve_model_artifact(path).exists():
        return False
    try:
        with _load_model_result_array(path) as data:
            return data['boxes'].shape == (YOLO_FRAMES, MAX_PEOPLE, 4) and data['position_features'].shape == (YOLO_FRAMES, 31) and (data['pose_features'].shape == (YOLO_FRAMES, 30)) and (data['descriptor'].shape == (982,)) and (str(data['protocol_signature']) == PROTOCOL_SIGNATURE)
    except (OSError, KeyError, ValueError):
        return False

def extract_resnet(rgb_frames: np.ndarray, detector, output_path: Path, batch_size: int) -> None:
    values = detector.feature_extractor.predict(rgb_frames, batch_size=min(max(1, batch_size), len(rgb_frames)), verbose=0)
    length = min(len(values), RGB_LENGTH)
    features = np.zeros((RGB_LENGTH, RGB_FEATURES), dtype=np.float32)
    mask = np.zeros(RGB_LENGTH, dtype=np.uint8)
    features[:length] = np.asarray(values[:length], dtype=np.float32)
    mask[:length] = 1
    np.savez_compressed(output_path, features=features.astype(np.float16), mask=mask, length=np.asarray(length, dtype=np.int16), protocol_signature=np.asarray(PROTOCOL_SIGNATURE))

def extract_yolo(yolo_frames: np.ndarray, detector, output_path: Path, batch_size: int) -> None:
    from models.common.studies.external_match_test_40rgb_yolo_highres_20260815.run_optimized_external_test import _decode_result
    boxes = np.zeros((YOLO_FRAMES, MAX_PEOPLE, 4), dtype=np.float32)
    person_conf = np.zeros((YOLO_FRAMES, MAX_PEOPLE), dtype=np.float32)
    keypoints = np.zeros((YOLO_FRAMES, MAX_PEOPLE, 17, 2), dtype=np.float32)
    keypoint_conf = np.zeros((YOLO_FRAMES, MAX_PEOPLE, 17), dtype=np.float32)
    valid = np.zeros((YOLO_FRAMES, MAX_PEOPLE), dtype=bool)
    batch_size = max(1, int(batch_size))
    for start in range(0, YOLO_FRAMES, batch_size):
        chunk = [frame for frame in yolo_frames[start:start + batch_size]]
        actual = len(chunk)
        while len(chunk) < batch_size:
            chunk.append(chunk[-1].copy())
        results = detector.pose_detector.predict(chunk, imgsz=640, conf=PERSON_CONFIDENCE, device='cpu', verbose=False, batch=batch_size)
        for offset, (result, frame) in enumerate(zip(results[:actual], chunk[:actual])):
            index = start + offset
            boxes[index], person_conf[index], keypoints[index], keypoint_conf[index], valid[index] = _decode_result(result, frame)
    positions, poses, frame_mask = detector._build_geometry_features(boxes, person_conf, keypoints, keypoint_conf, valid, person_threshold=PERSON_CONFIDENCE, keypoint_threshold=KEYPOINT_CONFIDENCE)
    data = {'boxes': boxes, 'person_conf': person_conf, 'keypoints': keypoints, 'keypoint_conf': keypoint_conf, 'valid': valid, 'position_features': positions, 'pose_features': poses, 'frame_mask': frame_mask}
    e04 = detector._make_variant(data, 'E04_tracking_smoothing').astype(np.float32)
    e06 = detector._make_variant(data, 'E06_rugby_descriptors').astype(np.float32)
    descriptor = np.concatenate([e04, e06]).astype(np.float32)
    if descriptor.shape != (982,):
        raise ValueError(f'Unexpected YOLO descriptor shape: {descriptor.shape}')
    np.savez_compressed(output_path, **data, e04=e04, e06=e06, descriptor=descriptor, protocol_signature=np.asarray(PROTOCOL_SIGNATURE))

def write_state(frame: pd.DataFrame, output: Path) -> None:
    temporary = output / 'final_test_feature_manifest.tmp.csv'
    frame.to_csv(temporary, index=False)
    temporary.replace(output / 'final_test_feature_manifest.csv')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--modality', choices=('both', 'resnet', 'yolo', 'manifest'), default='both')
    parser.add_argument('--matches', default='', help='Optional comma-separated IDs such as V_1,V_2')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resnet-batch', type=int, default=64)
    parser.add_argument('--yolo-batch', type=int, default=16)
    parser.add_argument('--yolo-backend', choices=('openvino', 'pytorch'), default='pytorch')
    parser.add_argument('--openvino-threads', type=int, default=4, help='CPU thread budget used while compiling the audited OpenVINO backend.')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resnet50').mkdir(exist_ok=True)
    (output / 'yolo11n_pose_highres').mkdir(exist_ok=True)
    match_filter = {item.strip().upper() for item in args.matches.split(',') if item.strip()} or None
    manifest, audit = build_manifest(match_filter)
    if args.limit > 0:
        manifest = manifest.iloc[:args.limit].copy()
    manifest['protocol_signature'] = PROTOCOL_SIGNATURE
    manifest['resnet_cache_path'] = ''
    manifest['resnet_status'] = 'pending'
    manifest['yolo_cache_path'] = ''
    manifest['yolo_status'] = 'pending'
    for index, row in manifest.iterrows():
        own_rgb = output / 'resnet50' / f'{row.dataset_id}.npz'
        legacy = LEGACY_V1_RGB / f'{row.dataset_id}.npz'
        rgb_path = legacy if row.match == 'V_1' and valid_resnet_cache(legacy) else own_rgb
        yolo_path = output / 'yolo11n_pose_highres' / f'{row.dataset_id}.npz'
        manifest.at[index, 'resnet_cache_path'] = str(rgb_path.resolve())
        manifest.at[index, 'yolo_cache_path'] = str(yolo_path.resolve())
        if valid_resnet_cache(rgb_path) and (not args.overwrite):
            manifest.at[index, 'resnet_status'] = 'ready_reused_v1' if rgb_path == legacy else 'ready'
        if valid_yolo_cache(yolo_path) and (not args.overwrite):
            manifest.at[index, 'yolo_status'] = 'ready'
    protocol_payload = {'protocol': PROTOCOL, 'protocol_signature': PROTOCOL_SIGNATURE, 'selection_audit': audit, 'selected_matches': sorted(manifest.match.unique()), 'blocks': int(len(manifest)), 'label_counts': manifest.true_label.value_counts().to_dict(), 'yolo_backend_requested': args.yolo_backend, 'openvino_equivalence_evidence': str((ROOT / 'data/player_geometry_highres_pose/outputs/backend_equivalence_dynamic_b16.json').resolve())}
    (output / 'protocol.json').write_text(json.dumps(protocol_payload, indent=2, ensure_ascii=False), encoding='utf-8')
    write_state(manifest, output)
    print(json.dumps(protocol_payload, ensure_ascii=False), flush=True)
    if args.modality == 'manifest':
        return
    from models.M2.predictor import RGBYoloHighresQuantileFusionDetector
    detector = RGBYoloHighresQuantileFusionDetector(ROOT)
    if args.yolo_backend == 'openvino' and args.modality in {'both', 'yolo'}:
        import openvino as ov
        from ultralytics import YOLO
        original_compile_model = ov.Core.compile_model

        def compile_model_with_budget(core, *compile_args, **compile_kwargs):
            config = dict(compile_kwargs.get('config') or {})
            config[str(ov.properties.inference_num_threads)] = max(1, int(args.openvino_threads))
            compile_kwargs['config'] = config
            return original_compile_model(core, *compile_args, **compile_kwargs)
        ov.Core.compile_model = compile_model_with_budget
        detector.pose_detector = YOLO(_resolve_model_artifact(str(ROOT / 'yolo11n-pose_openvino_model')))
    started = time.perf_counter()
    errors: list[dict] = []
    for position, row in enumerate(manifest.itertuples(index=True), start=1):
        index = int(row.Index)
        rgb_path = Path(row.resnet_cache_path)
        yolo_path = Path(row.yolo_cache_path)
        need_rgb = args.modality in {'both', 'resnet'} and (args.overwrite or not valid_resnet_cache(rgb_path))
        need_yolo = args.modality in {'both', 'yolo'} and (args.overwrite or not valid_yolo_cache(yolo_path))
        if not need_rgb and (not need_yolo):
            continue
        block_started = time.perf_counter()
        try:
            rgb_frames, yolo_frames, metadata = read_block_frames(row, need_rgb=need_rgb, need_yolo=need_yolo)
            if need_rgb:
                extract_resnet(rgb_frames, detector, rgb_path, args.resnet_batch)
                manifest.at[index, 'resnet_status'] = 'ready'
            if need_yolo:
                extract_yolo(yolo_frames, detector, yolo_path, args.yolo_batch)
                manifest.at[index, 'yolo_status'] = 'ready'
            elapsed = time.perf_counter() - started
            per_processed = elapsed / max(1, position)
            eta = per_processed * (len(manifest) - position)
            print(f"[{position}/{len(manifest)}] {row.match}/{row.section}/b{row.block_index:04d} rgb={manifest.at[index, 'resnet_status']} yolo={manifest.at[index, 'yolo_status']} block={time.perf_counter() - block_started:.1f}s eta={eta / 60:.1f}min source_fps={metadata['source_fps']:.3f}", flush=True)
        except Exception as exc:
            if need_rgb:
                manifest.at[index, 'resnet_status'] = 'error'
            if need_yolo:
                manifest.at[index, 'yolo_status'] = 'error'
            errors.append({'dataset_id': row.dataset_id, 'error': repr(exc)})
            print(f'ERROR {row.dataset_id}: {exc!r}', flush=True)
        if position % 5 == 0 or position == len(manifest):
            write_state(manifest, output)
            (output / 'errors.json').write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding='utf-8')
    write_state(manifest, output)
    summary = {'blocks': int(len(manifest)), 'matches': sorted(manifest.match.unique()), 'resnet_status': manifest.resnet_status.value_counts().to_dict(), 'yolo_status': manifest.yolo_status.value_counts().to_dict(), 'errors': errors, 'elapsed_seconds': round(time.perf_counter() - started, 3), 'complete': bool(not errors and (args.modality == 'resnet' or manifest.yolo_status.str.startswith('ready').all()) and (args.modality == 'yolo' or manifest.resnet_status.str.startswith('ready').all()))}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
