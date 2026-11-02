"""Build the exact 32-frame YOLO pose cache required by frozen M1 on CV."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import hashlib
import json
import time
from pathlib import Path
import pandas as pd
from models.common.studies.final_tests_feature_cache_20260826 import build_cache as cache
from .data import M1_FINAL_POSE32
PROTOCOL = {'version': 'm1_player_geometry_pose32_final_v1', 'source_protocol': cache.PROTOCOL['version'], 'frames': 32, 'sampling': 'uniform over the 25 FPS seven-second block timeline', 'spatial_preprocessing': cache.PROTOCOL['yolo']['spatial_preprocessing'], 'input_size': 640, 'model': 'yolo11n-pose.pt', 'person_confidence': cache.PERSON_CONFIDENCE, 'keypoint_confidence': cache.KEYPOINT_CONFIDENCE}
SIGNATURE = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:16]

def valid(path: Path) -> bool:
    if not _resolve_model_artifact(path).is_file():
        return False
    try:
        import numpy as np
        with _load_model_result_array(path, allow_pickle=False) as data:
            return data['position_features'].shape == (32, 31) and data['pose_features'].shape == (32, 30) and (data['frame_mask'].shape == (32,)) and (str(data['protocol_signature']) == SIGNATURE)
    except (OSError, KeyError, ValueError):
        return False

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', default='')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--backend', choices=('openvino', 'pytorch'), default='openvino')
    parser.add_argument('--openvino-threads', type=int, default=4)
    args = parser.parse_args(argv)
    matches = {item.strip().upper() for item in args.matches.split(',') if item.strip()} or None
    manifest, _audit = cache.build_manifest(matches)
    if args.limit:
        manifest = manifest.iloc[:args.limit].copy()
    M1_FINAL_POSE32.mkdir(parents=True, exist_ok=True)
    manifest['cache_path'] = manifest.dataset_id.map(lambda value: str((M1_FINAL_POSE32 / f'{value}.npz').resolve()))
    manifest['status'] = manifest.cache_path.map(lambda value: 'ready' if valid(Path(value)) and (not args.overwrite) else 'pending')
    manifest['protocol_signature'] = SIGNATURE
    manifest_path = M1_FINAL_POSE32.parent / 'm1_pose32_manifest.csv'
    manifest.to_csv(manifest_path, index=False)
    (M1_FINAL_POSE32.parent / 'm1_pose32_protocol.json').write_text(json.dumps({'protocol': PROTOCOL, 'signature': SIGNATURE}, indent=2), encoding='utf-8')
    if args.manifest_only:
        print(json.dumps({'rows': len(manifest), 'status': manifest.status.value_counts().to_dict()}))
        return 0
    cache.YOLO_FRAMES = 32
    cache.PROTOCOL_SIGNATURE = SIGNATURE
    from models.M2.predictor import RGBYoloHighresQuantileFusionDetector
    detector = RGBYoloHighresQuantileFusionDetector(cache.ROOT)
    if args.backend == 'openvino':
        import openvino as ov
        from ultralytics import YOLO
        original_compile_model = ov.Core.compile_model

        def compile_model_with_budget(core, *compile_args, **compile_kwargs):
            config = dict(compile_kwargs.get('config') or {})
            config[str(ov.properties.inference_num_threads)] = max(1, int(args.openvino_threads))
            compile_kwargs['config'] = config
            return original_compile_model(core, *compile_args, **compile_kwargs)
        ov.Core.compile_model = compile_model_with_budget
        detector.pose_detector = YOLO(_resolve_model_artifact(str(cache.ROOT / 'yolo11n-pose_openvino_model')))
    started = time.perf_counter()
    errors = []
    for position, row in enumerate(manifest.itertuples(index=True), 1):
        index = int(row.Index)
        path = Path(row.cache_path)
        if valid(path) and (not args.overwrite):
            continue
        try:
            _rgb, frames, _metadata = cache.read_block_frames(row, need_rgb=False, need_yolo=True)
            cache.extract_yolo(frames, detector, path, args.batch_size)
            manifest.at[index, 'status'] = 'ready'
            print(f'[{position}/{len(manifest)}] {row.dataset_id} ready', flush=True)
        except Exception as error:
            manifest.at[index, 'status'] = 'error'
            errors.append({'dataset_id': row.dataset_id, 'error': repr(error)})
            print(f'ERROR {row.dataset_id}: {error!r}', flush=True)
        if position % 5 == 0 or position == len(manifest):
            manifest.to_csv(manifest_path, index=False)
    report = {'rows': len(manifest), 'status': manifest.status.value_counts().to_dict(), 'errors': errors, 'elapsed_seconds': time.perf_counter() - started}
    (M1_FINAL_POSE32.parent / 'm1_pose32_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0
if __name__ == '__main__':
    raise SystemExit(main())
