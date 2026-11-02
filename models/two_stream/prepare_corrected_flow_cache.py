"""Validate and complete the optical-flow cache for a corrected Two-Stream run."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import train as ts
WORKERS = int(os.environ.get('FLOW_PREP_WORKERS', '4'))

def _prepare(video_name: str) -> dict:
    target_path = ts._flow_cache_path(video_name)
    legacy_path = ts._verified_legacy_flow_cache_path(video_name)
    if os.path.exists(_resolve_model_artifact(target_path)):
        source = 'corrected_cache'
    elif legacy_path and os.path.exists(_resolve_model_artifact(legacy_path)):
        source = 'verified_legacy_cache'
    else:
        source = 'computed'
    flow = ts.load_or_compute_flow(video_name)
    expected_shape = (ts.FLOW_IMG_SIZE, ts.FLOW_IMG_SIZE, ts.NUM_FLOW_FRAMES * 2)
    if flow is None:
        raise RuntimeError('flow extraction returned None')
    if flow.shape != expected_shape:
        raise RuntimeError(f'invalid shape {flow.shape}, expected {expected_shape}')
    if flow.dtype != np.float32:
        raise RuntimeError(f'invalid dtype {flow.dtype}')
    if not np.isfinite(flow).all():
        raise RuntimeError('non-finite values')
    max_abs = float(np.max(np.abs(flow)))
    if max_abs > 1.00001:
        raise RuntimeError(f'out-of-range flow: max abs {max_abs}')
    return {'video_name': video_name, 'source': source, 'mean_abs': float(np.mean(np.abs(flow))), 'std': float(np.std(flow)), 'max_abs': max_abs, 'zero_fraction': float(np.mean(flow == 0.0))}

def main() -> None:
    train_df = ts.normalize_manifest(pd.read_csv(_resolve_model_artifact(ts.TRAIN_CSV)), 'train')
    test_df = ts.normalize_manifest(pd.read_csv(_resolve_model_artifact(ts.TEST_CSV)), 'test')
    manifest = pd.concat([train_df, test_df], ignore_index=True)
    manifest = manifest.drop_duplicates('video_name', keep='first')
    missing_videos = [video_name for video_name in manifest['video_name'] if ts._find_video(video_name) is None]
    if missing_videos:
        raise RuntimeError(f'Missing {len(missing_videos)} source videos')
    ineligible_videos = [video_name for video_name in manifest['video_name'] if not ts._has_minimum_flow_input(video_name)]
    eligible_manifest = manifest[~manifest['video_name'].isin(ineligible_videos)].reset_index(drop=True)
    print(f'Preparing {len(eligible_manifest)} eligible videos with {WORKERS} workers; target={ts.FLOW_CACHE_DIR}', flush=True)
    started = time.time()
    rows: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_prepare, video_name): video_name for video_name in eligible_manifest['video_name']}
        for completed, future in enumerate(as_completed(futures), start=1):
            video_name = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({'video_name': video_name, 'error': str(exc)})
            if completed % 100 == 0 or completed == len(futures):
                print(f'  {completed}/{len(futures)} checked; failures={len(failures)}', flush=True)
    result_df = pd.DataFrame(rows).sort_values('video_name')
    result_path = os.path.join(ts.FLOW_CACHE_DIR, 'cache_audit.csv')
    result_df.to_csv(result_path, index=False)
    summary = {'flow_cache_name': ts.FLOW_CACHE_NAME, 'flow_cache_dir': ts.FLOW_CACHE_DIR, 'unique_videos': len(manifest), 'eligible_videos': len(eligible_manifest), 'ineligible_videos': ineligible_videos, 'valid_flows': len(rows), 'failures': failures, 'source_counts': result_df['source'].value_counts().to_dict(), 'mean_abs_mean': float(result_df['mean_abs'].mean()), 'mean_abs_min': float(result_df['mean_abs'].min()), 'mean_abs_max': float(result_df['mean_abs'].max()), 'zero_fraction_mean': float(result_df['zero_fraction'].mean()), 'duration_s': round(time.time() - started, 1), 'workers': WORKERS}
    with open(_resolve_model_artifact(os.path.join(ts.FLOW_CACHE_DIR, 'cache_audit_summary.json')), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
