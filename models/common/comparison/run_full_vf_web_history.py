"""Run the three selected Panel-V web models on every complete VF part.

The queue is ordered by source duration and then by the measured model speed
(M3 before M1).  Every successful run is written through the same
``_execute_analysis`` function used by the web application, so it appears in
``/api/results`` and can be restored from the web history.

The state file makes the batch resumable and records a post-save contract
audit before an item is considered complete.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
EXPERIMENT_DIR = experiment_directory(__file__)
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
WEB_SOURCE_DIR = PROJECT_ROOT / 'src' / 'web'
if str(WEB_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_SOURCE_DIR))
VF_DIR = PROJECT_ROOT / 'data' / 'videos' / 'partidos_evaluacion'
OUTPUT_DIR = EXPERIMENT_DIR / 'outputs' / 'vf_full_web_history'
STATE_PATH = OUTPUT_DIR / 'queue_state.json'
SUMMARY_PATH = OUTPUT_DIR / 'queue_summary.csv'
MODELS = ('m3_h2_d2_cross_fold_ensemble', 'm3_h1_d2_cross_fold_ensemble', 'm1_h2_d2_cross_fold_ensemble')
PREEXISTING_RESULTS = {('VF_S01_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4', 'm1_h2_d2_cross_fold_ensemble'): '20260830_064334'}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)

def _duration(path: Path) -> float:
    completed = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(path)], check=True, capture_output=True, text=True)
    return float(completed.stdout.strip())

def _videos() -> list[dict]:
    videos = []
    for path in _model_artifact_glob(VF_DIR, 'VF_*.mp4', recursive=False):
        videos.append({'name': path.name, 'path': str(path.resolve()), 'duration_seconds': _duration(path)})
    if not videos:
        raise FileNotFoundError(f'No VF_*.mp4 videos found in {VF_DIR}')
    return sorted(videos, key=lambda item: (item['duration_seconds'], item['name']))

def _queue(videos: list[dict]) -> list[dict]:
    return [{'key': f"{video['name']}::{model}", 'video_name': video['name'], 'video_path': video['path'], 'video_seconds': video['duration_seconds'], 'model_name': model, 'status': 'pending', 'result_id': None, 'result_file': None, 'started_at': None, 'completed_at': None, 'error': None} for video in videos for model in MODELS]

def _load_or_create_state() -> dict:
    videos = _videos()
    expected = _queue(videos)
    if _resolve_model_artifact(STATE_PATH).is_file():
        state = json.loads(_resolve_model_artifact(STATE_PATH).read_text(encoding='utf-8'))
        existing = {item['key']: item for item in state.get('items', [])}
        state['items'] = [existing.get(item['key'], item) for item in expected]
        state['updated_at'] = _now()
        _atomic_json(STATE_PATH, state)
        return state
    state = {'schema_version': 1, 'created_at': _now(), 'updated_at': _now(), 'ordering': 'video_duration_ascending_then_measured_model_speed', 'models': list(MODELS), 'items': expected}
    _atomic_json(STATE_PATH, state)
    return state

def _probability_vectors(payload: dict):
    for collection in ('detections', 'window_predictions'):
        for item in payload.get(collection, []):
            probabilities = item.get('probabilities')
            if probabilities:
                yield (collection, probabilities)

def _validate_saved_result(result_file: Path, video_path: Path, expected_display_name: str) -> dict:
    payload = json.loads(_resolve_model_artifact(result_file).read_text(encoding='utf-8'))
    if payload.get('status') != 'success':
        raise ValueError(f"Result status is not success: {payload.get('status')}")
    if Path(payload.get('video_path', '')).resolve() != video_path.resolve():
        raise ValueError('Saved result points to a different video')
    if payload.get('model_name') != expected_display_name:
        raise ValueError(f"Saved model mismatch: {payload.get('model_name')} != {expected_display_name}")
    vector_count = 0
    for collection, probabilities in _probability_vectors(payload):
        values = [float(value) for value in probabilities.values()]
        if not values or not all((math.isfinite(value) for value in values)):
            raise ValueError(f'Invalid finite probabilities in {collection}')
        if any((value < -1e-06 or value > 1.0 + 1e-06 for value in values)):
            raise ValueError(f'Probability outside [0,1] in {collection}')
        if abs(sum(values) - 1.0) > 0.0001:
            raise ValueError(f'Probabilities do not sum to one in {collection}')
        vector_count += 1
    if vector_count == 0:
        raise ValueError('Saved result contains no probability vectors')
    timing = payload.get('timing') or payload.get('metadata', {}).get('timing') or {}
    inference_seconds = float(timing.get('inference_seconds') or 0.0)
    if not math.isfinite(inference_seconds) or inference_seconds <= 0:
        raise ValueError('Saved result has no valid inference time')
    return {'detections': len(payload.get('detections', [])), 'line': sum((item.get('play_type') == 'line' for item in payload.get('detections', []))), 'scrum': sum((item.get('play_type') == 'scrum' for item in payload.get('detections', []))), 'window_predictions': len(payload.get('window_predictions', [])), 'validated_probability_vectors': vector_count, 'timing': timing}

def _write_summary(state: dict) -> None:
    fields = ['status', 'video_name', 'video_seconds', 'model_name', 'result_id', 'inference_seconds', 'realtime_factor', 'detections', 'line', 'scrum', 'completed_at', 'error']
    with _resolve_model_artifact(SUMMARY_PATH).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in state['items']:
            audit = item.get('audit') or {}
            timing = audit.get('timing') or {}
            writer.writerow({'status': item['status'], 'video_name': item['video_name'], 'video_seconds': item['video_seconds'], 'model_name': item['model_name'], 'result_id': item.get('result_id'), 'inference_seconds': timing.get('inference_seconds'), 'realtime_factor': timing.get('inference_realtime_factor'), 'detections': audit.get('detections'), 'line': audit.get('line'), 'scrum': audit.get('scrum'), 'completed_at': item.get('completed_at'), 'error': item.get('error')})

def _save_state(state: dict) -> None:
    state['updated_at'] = _now()
    _atomic_json(STATE_PATH, state)
    _write_summary(state)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--do-not-reuse-existing', action='store_true', help='Repeat even the already completed VF_S01/M1 web result.')
    parser.add_argument('--video', action='append', default=[], help='Process only this exact VF filename (repeatable).')
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = _load_or_create_state()
    for item in state['items']:
        if item['status'] == 'running':
            item['status'] = 'interrupted'
            item['error'] = 'Previous process ended before saving a validated result.'
        if args.do_not_reuse_existing and item['status'] == 'completed_preexisting':
            item.update({'status': 'pending', 'result_id': None, 'result_file': None, 'started_at': None, 'completed_at': None, 'audit': None, 'error': None})
    _save_state(state)
    from src.web import app as web_app
    available = {item['name']: item for item in web_app.inference_engine.list_available_models()}
    missing = [model for model in MODELS if model not in available]
    if missing:
        raise RuntimeError(f'Models unavailable in current web source: {missing}')
    if not args.do_not_reuse_existing:
        for item in state['items']:
            result_id = PREEXISTING_RESULTS.get((item['video_name'], item['model_name']))
            if not result_id or item['status'] == 'completed':
                continue
            result_file = web_app.app.config['RESULTS_FOLDER'] / f'result_{result_id}.json'
            if not _resolve_model_artifact(result_file).is_file():
                continue
            payload = json.loads(_resolve_model_artifact(result_file).read_text(encoding='utf-8'))
            if Path(payload.get('video_path', '')).resolve() == Path(item['video_path']).resolve() and payload.get('model_name') == available[item['model_name']]['display_name'] and (payload.get('status') == 'success'):
                item.update({'status': 'completed_preexisting', 'result_id': result_id, 'result_file': str(result_file), 'completed_at': result_file.stat().st_mtime_ns, 'audit': {'detections': len(payload.get('detections', [])), 'line': sum((d.get('play_type') == 'line' for d in payload.get('detections', []))), 'scrum': sum((d.get('play_type') == 'scrum' for d in payload.get('detections', []))), 'timing': {}, 'note': 'Pre-existing full-source web result; no persisted timing.'}})
        _save_state(state)
    selected_videos = set(args.video)
    unknown_videos = selected_videos - {item['video_name'] for item in state['items']}
    if unknown_videos:
        raise ValueError(f'Unknown --video values: {sorted(unknown_videos)}')
    if selected_videos:
        for item in state['items']:
            if item['video_name'] not in selected_videos and item['status'] not in {'completed', 'completed_preexisting'}:
                item.update({'status': 'not_selected', 'started_at': None, 'completed_at': None, 'error': None})
        _save_state(state)
    pending = [item for item in state['items'] if item['status'] not in {'completed', 'completed_preexisting'} and (not selected_videos or item['video_name'] in selected_videos)]
    print(f"Queue: {len(pending)} pending of {len(state['items'])}", flush=True)
    for position, item in enumerate(pending, start=1):
        model_metadata = available[item['model_name']]
        defaults = model_metadata.get('analysis_defaults') or {}
        item.update({'status': 'running', 'started_at': _now(), 'error': None})
        _save_state(state)
        print(f"[{position}/{len(pending)}] {item['video_name']} | {item['model_name']}", flush=True)
        last_reported = -10.0

        def progress(percent, stage, message):
            nonlocal last_reported
            if float(percent) >= last_reported + 5.0 or float(percent) >= 99.0:
                last_reported = float(percent)
                print(f'  {percent:5.1f}% | {stage} | {message}', flush=True)
        try:
            response = web_app._execute_analysis({'model_name': item['model_name'], 'video_path': item['video_path'], 'parameters': {'confidence_threshold': float(defaults.get('confidence_threshold', 0.7)), 'line_confidence_threshold': float(defaults.get('line_confidence_threshold', 0.95)), 'min_play_duration': 3.0, 'max_play_duration': 15.0, 'overlap_threshold': 0.5, 'batch_size': 8, 'segment_duration': 7, 'use_segments': True, 'temporal_fps_mode': 'training_match_25', 'feature_batch_size': 256, 'feature_predict_batch_size': 8}}, progress_callback=progress)
            result_id = response['result_id']
            result_file = web_app.app.config['RESULTS_FOLDER'] / f'result_{result_id}.json'
            audit = _validate_saved_result(result_file, Path(item['video_path']), model_metadata['display_name'])
            item.update({'status': 'completed', 'result_id': result_id, 'result_file': str(result_file), 'completed_at': _now(), 'audit': audit})
            print(f"  completed result={result_id} inference={audit['timing']['inference_seconds']:.2f}s detections={audit['detections']}", flush=True)
        except Exception as exc:
            item.update({'status': 'error', 'error': repr(exc), 'completed_at': _now()})
            _save_state(state)
            raise
        _save_state(state)
        time.sleep(1.1)
    print(f'All queue items complete. State: {STATE_PATH}', flush=True)
if __name__ == '__main__':
    main()
