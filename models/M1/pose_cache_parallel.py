"""Parallel pose-cache extraction for the temporal-resolution ablations.

This keeps the original ``pose_cache.py`` behavior intact while allowing a
CPU-only machine to process independent clips in parallel.  The temporal
resolution and cache root come from the same environment variables as the
main experiment configuration.
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
import multiprocessing as mp
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from models.M1.config import DATA_DIR, MAX_PEOPLE, POSE_CONFIDENCE_DEFAULT, POSE_FRAME_COUNT_DEFAULT
from models.M1.config import POSE_IMAGE_SIZE_DEFAULT, POSE_KEYPOINT_CONFIDENCE_DEFAULT, POSE_MODEL_DEFAULT
from models.M1.pose_cache import extract_clip, manifest_for_extraction, output_dir
_WORKER_MODEL = None
_WORKER_ARGS = None

def _init_worker(model_path: str, image_size: int, person_confidence: float, keypoint_confidence: float, max_people: int) -> None:
    """Load one YOLO model per worker and cap intra-worker BLAS threads."""
    global _WORKER_MODEL, _WORKER_ARGS
    os.environ.setdefault('YOLO_CONFIG_DIR', str(experiment_directory(__file__) / '.ultralytics_parallel'))
    try:
        import torch
        torch.set_num_threads(max(1, int(os.environ.get('PLAYER_GEOMETRY_WORKER_THREADS', '4'))))
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    from ultralytics import YOLO
    _WORKER_MODEL = YOLO(_resolve_model_artifact(model_path))
    _WORKER_ARGS = (image_size, person_confidence, keypoint_confidence, max_people)

def _process_row(payload: tuple[dict, str, bool]) -> dict:
    row, target_dir_text, overwrite = payload
    video_name = str(row['video_name'])
    output_path = Path(target_dir_text) / f'{Path(video_name).stem}.npz'
    if _resolve_model_artifact(output_path).exists() and (not overwrite):
        return {'video_name': video_name, 'tag': row['tag'], 'status': 'cached', 'path': str(output_path)}
    started = time.perf_counter()
    try:
        image_size, person_confidence, keypoint_confidence, max_people = _WORKER_ARGS
        arrays, metadata = extract_clip(_WORKER_MODEL, DATA_DIR / video_name, POSE_FRAME_COUNT_DEFAULT, image_size, person_confidence, keypoint_confidence, max_people)
        np.savez_compressed(output_path, **arrays)
        metadata.update({'elapsed_seconds': round(time.perf_counter() - started, 4)})
        return {'video_name': video_name, 'tag': row['tag'], 'status': 'ok', 'path': str(output_path), **metadata}
    except Exception as exc:
        return {'video_name': video_name, 'tag': row['tag'], 'status': 'error', 'error': repr(exc)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=POSE_MODEL_DEFAULT)
    parser.add_argument('--frame-count', type=int, default=POSE_FRAME_COUNT_DEFAULT)
    parser.add_argument('--image-size', type=int, default=POSE_IMAGE_SIZE_DEFAULT)
    parser.add_argument('--person-confidence', type=float, default=POSE_CONFIDENCE_DEFAULT)
    parser.add_argument('--keypoint-confidence', type=float, default=POSE_KEYPOINT_CONFIDENCE_DEFAULT)
    parser.add_argument('--max-people', type=int, default=MAX_PEOPLE)
    parser.add_argument('--per-class', type=int, default=None)
    parser.add_argument('--max-clips', type=int, default=None)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.frame_count != POSE_FRAME_COUNT_DEFAULT:
        raise SystemExit(f'--frame-count={args.frame_count} no coincide con PLAYER_GEOMETRY_LENGTH={POSE_FRAME_COUNT_DEFAULT}. Configure la variable de entorno primero.')
    target_dir = output_dir(args.model)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for_extraction(args.per_class, args.max_clips, args.seed)
    payloads = [(row, str(target_dir), args.overwrite) for row in manifest.to_dict('records')]
    workers = max(1, int(args.workers))
    worker_threads = max(1, int(os.environ.get('PLAYER_GEOMETRY_WORKER_THREADS', '4')))
    print(f'Clips: {len(payloads)} | workers: {workers} | torch threads/worker: {worker_threads}')
    print(f'Salida: {target_dir}')
    context = mp.get_context('spawn')
    records: list[dict] = []
    with context.Pool(processes=workers, initializer=_init_worker, initargs=(args.model, args.image_size, args.person_confidence, args.keypoint_confidence, args.max_people)) as pool:
        for number, record in enumerate(pool.imap_unordered(_process_row, payloads), start=1):
            records.append(record)
            if number == 1 or number % 25 == 0 or number == len(payloads):
                print(f"Procesados {number}/{len(payloads)} | últimos: {record['video_name']} ({record['status']})", flush=True)
    records.sort(key=lambda item: item.get('video_name', ''))
    pd.DataFrame(records).to_csv(target_dir / 'extraction_log.csv', index=False)
    metadata = {'model': args.model, 'model_directory': str(target_dir), 'frame_count': args.frame_count, 'image_size': args.image_size, 'person_confidence': args.person_confidence, 'keypoint_confidence': args.keypoint_confidence, 'max_people': args.max_people, 'clips_requested': len(records), 'clips_ok': sum((item.get('status') in {'ok', 'cached'} for item in records)), 'clips_failed': sum((item.get('status') == 'error' for item in records)), 'workers': workers, 'worker_threads': worker_threads}
    (target_dir / 'cache_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
if __name__ == '__main__':
    mp.freeze_support()
    main()
