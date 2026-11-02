"""Plan or run the isolated high-resolution YOLO quantile experiment."""
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
import os
import subprocess
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
HERE = experiment_directory(__file__)
CACHE = HERE / 'outputs' / 'cache'
EVALUATION = HERE / 'outputs' / 'quantile_100frames'

def commands(workers: int, batch_size: int) -> dict[str, list[str]]:
    python = sys.executable
    return {'preflight': [python, str(_resolve_model_artifact(HERE / 'extract_cache.py'))], 'extract': [python, str(_resolve_model_artifact(HERE / 'extract_cache.py')), '--workers', str(workers), '--batch-size', str(batch_size), '--run'], 'evaluate': [python, str(_resolve_model_artifact(HERE / 'evaluate_quantile_fusion.py')), '--cache-dir', str(CACHE / 'yolo11n-pose_pt'), '--output', str(EVALUATION)], 'deploy': [python, str(_resolve_model_artifact(HERE / 'train_deployment.py')), '--dataset', str(REPO / 'data/clips_dataset'), '--batch-size', str(batch_size), '--threads', str(max(1, workers))]}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'extract', 'evaluate', 'deploy', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--worker-threads', type=int, default=4)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    available = commands(max(1, args.workers), max(1, args.batch_size))
    order = ['preflight', 'extract', 'evaluate', 'deploy']
    selected = order if args.stage == 'all' else [args.stage]
    print(f"Mode: {('EXECUTE' if args.run else 'PLAN ONLY')}")
    for stage in selected:
        print(f'[{stage}] {subprocess.list2cmdline(available[stage])}')
    if not args.run:
        return
    environment = os.environ.copy()
    environment['HIGHRES_POSE_WORKER_THREADS'] = str(max(1, args.worker_threads))
    for stage in selected:
        subprocess.run(available[stage], cwd=REPO, env=environment, check=True)
if __name__ == '__main__':
    main()
