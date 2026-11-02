"""Run the three selected web models on fixed, source-resolution VF segments.

The benchmark records model-load and inference wall times separately.  Clips
are exact re-encodes at the original 1280x720 geometry and 30000/1001 FPS so
the web pipeline, rather than a prepared 224x224 cache, performs preprocessing.
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
import contextlib
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
EXPERIMENT_DIR = experiment_directory(__file__)
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = EXPERIMENT_DIR / 'outputs' / 'vf_web_benchmark'
CLIP_DIR = OUTPUT_DIR / 'clips'
RESULT_DIR = OUTPUT_DIR / 'results'
LOG_DIR = OUTPUT_DIR / 'logs'
MODELS = ('m1_h2_d2_cross_fold_ensemble', 'm3_h2_d2_cross_fold_ensemble', 'm3_h1_d2_cross_fold_ensemble')
SEGMENTS = ({'id': 'VF_S01_scrum_492_513', 'source': 'data/videos/partidos_evaluacion/VF_S01_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4', 'start_seconds': 492.0, 'duration_seconds': 21.0, 'visual_note': 'formacion y desarrollo de scrum'}, {'id': 'VF_S03_line_894_915', 'source': 'data/videos/partidos_evaluacion/VF_S03_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4', 'start_seconds': 894.0, 'duration_seconds': 21.0, 'visual_note': 'formacion, lanzamiento y salida de line'}, {'id': 'VF_S05_other_38_59', 'source': 'data/videos/partidos_evaluacion/VF_S05_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4', 'start_seconds': 38.0, 'duration_seconds': 21.0, 'visual_note': 'reinicio y juego abierto sin formacion completa visible'})

def _create_clip(segment: dict) -> Path:
    source = PROJECT_ROOT / segment['source']
    if not _resolve_model_artifact(source).is_file():
        raise FileNotFoundError(source)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    destination = CLIP_DIR / f"{segment['id']}.mp4"
    if _resolve_model_artifact(destination).is_file():
        return destination
    subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-ss', str(segment['start_seconds']), '-i', str(source), '-t', str(segment['duration_seconds']), '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', str(destination)], check=True)
    return destination

def _write_csv(rows: list[dict], destination: Path) -> None:
    fields = ['model_name', 'segment_id', 'video_seconds', 'model_load_seconds', 'inference_seconds', 'total_seconds_including_shared_load', 'inference_realtime_factor', 'processed_video_seconds_per_wall_second', 'detection_count', 'line_count', 'scrum_count', 'window_count', 'status']
    with _resolve_model_artifact(destination).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fields} for row in rows])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='*', choices=MODELS, default=list(MODELS))
    parser.add_argument('--segments', nargs='*', choices=[item['id'] for item in SEGMENTS], default=[item['id'] for item in SEGMENTS])
    parser.add_argument('--merge-existing', action='store_true', help='Keep unselected model runs from the existing benchmark summary.')
    args = parser.parse_args()
    selected_segments = [item for item in SEGMENTS if item['id'] in args.segments]
    clips = {item['id']: _create_clip(item) for item in selected_segments}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    from src.web.keras_inference_engine import KerasInferenceEngine
    engine = KerasInferenceEngine()
    available = {item['name']: item for item in engine.list_available_models()}
    missing = [model for model in args.models if model not in available]
    if missing:
        raise RuntimeError(f'Selected models are not listed by the web engine: {missing}')
    rows = []
    summary_json = OUTPUT_DIR / 'benchmark_summary.json'
    detailed = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'clock': 'time.perf_counter wall time', 'timing_policy': 'one model load followed by identical segment inference; first segment includes framework graph warm-up in its inference time', 'inference_parameters': {'use_segments': True, 'window_seconds': 7, 'stride_seconds': 1, 'temporal_fps_mode': 'training_match_25', 'feature_batch_size': 256, 'feature_predict_batch_size': 8}, 'environment': {'python': sys.version, 'platform': platform.platform()}, 'segments': selected_segments, 'models': {}}
    if args.merge_existing and _resolve_model_artifact(summary_json).is_file():
        previous = json.loads(_resolve_model_artifact(summary_json).read_text(encoding='utf-8'))
        detailed['models'].update(previous.get('models') or {})
    for model_name in args.models:
        engine.clear_model()
        started = time.perf_counter()
        engine.load_model(_resolve_model_artifact(model_name))
        load_seconds = time.perf_counter() - started
        defaults = available[model_name].get('analysis_defaults') or {}
        model_results = []
        for segment in selected_segments:
            clip = clips[segment['id']]
            log_path = LOG_DIR / f"{model_name}__{segment['id']}.log"
            started = time.perf_counter()
            with _resolve_model_artifact(log_path).open('w', encoding='utf-8') as log_handle:
                with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                    result = engine.analyze_video(clip, confidence_threshold=float(defaults.get('confidence_threshold', 0.7)), line_confidence_threshold=float(defaults.get('line_confidence_threshold', 0.95)), segment_duration=7, use_segments=True, temporal_fps_mode='training_match_25', feature_batch_size=256, feature_predict_batch_size=8)
            inference_seconds = time.perf_counter() - started
            if result.get('status') != 'success':
                raise RuntimeError(f"{model_name} failed on {segment['id']}: {result.get('error')}")
            video_seconds = float(result.get('metadata', {}).get('duration') or 21.0)
            detections = result.get('detections', [])
            row = {'model_name': model_name, 'segment_id': segment['id'], 'video_seconds': video_seconds, 'model_load_seconds': load_seconds, 'inference_seconds': inference_seconds, 'total_seconds_including_shared_load': load_seconds + inference_seconds, 'inference_realtime_factor': inference_seconds / max(video_seconds, 1e-09), 'processed_video_seconds_per_wall_second': video_seconds / max(inference_seconds, 1e-09), 'detection_count': len(detections), 'line_count': sum((item.get('play_type') == 'line' for item in detections)), 'scrum_count': sum((item.get('play_type') == 'scrum' for item in detections)), 'window_count': len(result.get('window_predictions', [])), 'status': result['status']}
            rows.append(row)
            result.setdefault('benchmark_timing', {}).update(row)
            result_path = RESULT_DIR / f"{model_name}__{segment['id']}.json"
            result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
            model_results.append({'summary': row, 'result_file': str(result_path)})
            print(f"{model_name} | {segment['id']} | inference={inference_seconds:.2f}s | detections={len(detections)}", flush=True)
        detailed['models'][model_name] = {'load_seconds': load_seconds, 'web_metadata': available[model_name], 'runs': model_results}
    summary_csv = OUTPUT_DIR / 'benchmark_summary.csv'
    rows = [run['summary'] for model in MODELS for run in detailed['models'].get(model, {}).get('runs', [])]
    summary_json.write_text(json.dumps(detailed, indent=2), encoding='utf-8')
    _write_csv(rows, summary_csv)
    print(summary_json)
    print(summary_csv)
if __name__ == '__main__':
    main()
