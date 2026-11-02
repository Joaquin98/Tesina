"""
Standalone professor demo app for the BEST model only.

This file is intentionally separate from app.py to avoid breaking existing flows.
Run it directly for demo mode.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Dict, List
import uuid
import zipfile
import logging
import os
from flask import Flask, jsonify, render_template, request, send_file
if __package__:
    from .keras_inference_engine import KerasInferenceEngine
else:
    from keras_inference_engine import KerasInferenceEngine
try:
    from .video_metadata import read_video_metadata_many
except ImportError:
    from video_metadata import read_video_metadata_many
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('best_model_demo')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'rugby-best-model-demo-2026'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
PROJECT_ROOT = Path(__file__).parent.parent.parent
WEB_RUNTIME = Path(os.environ.get('TESINA_WEB_RUNTIME', PROJECT_ROOT / '.runtime' / 'web')).resolve()
UPLOAD_FOLDER = WEB_RUNTIME / 'uploads'
VIDEO_LIBRARY = PROJECT_ROOT / 'data' / 'videos' / 'partidos_evaluacion'
RESULTS_FOLDER = WEB_RUNTIME / 'results'
DEMO_TEMP_FOLDER = RESULTS_FOLDER / 'temp' / 'demo_best_model'
DEMO_EXPORTS_FOLDER = RESULTS_FOLDER / 'exports' / 'demo_best_model'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
DEMO_TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
DEMO_EXPORTS_FOLDER.mkdir(parents=True, exist_ok=True)
BEST_MODEL_NAME = 'best_model/best_model.keras'
ANALYSIS_CACHE: Dict[str, Dict] = {}
TARGET_DEMO_PLAYS = {'line', 'scrum'}

def _is_target_demo_play(play_type: str) -> bool:
    return (play_type or '').strip().lower() in TARGET_DEMO_PLAYS

def create_engine() -> KerasInferenceEngine:
    engine = KerasInferenceEngine()
    engine.load_model(BEST_MODEL_NAME)
    logger.info('Demo engine ready with fixed best model: %s', BEST_MODEL_NAME)
    return engine
inference_engine = create_engine()

@app.route('/')
def demo_home():
    return render_template('demo_best_model.html')

@app.route('/api/demo/health')
def demo_health():
    info = inference_engine.get_model_info()
    return jsonify({'success': True, 'mode': 'best_model_demo', 'fixed_model': BEST_MODEL_NAME, 'model_info': info})

def _list_videos() -> List[Dict]:
    videos: List[Dict] = []
    seen = set()
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    library_files = [p for p in sorted(_model_artifact_glob(VIDEO_LIBRARY, '*', recursive=True)) if _resolve_model_artifact(p).is_file() and p.suffix.lower() in video_exts] if _resolve_model_artifact(VIDEO_LIBRARY).exists() else []
    upload_files = [p for p in sorted(_model_artifact_glob(UPLOAD_FOLDER, '*', recursive=False)) if _resolve_model_artifact(p).is_file() and p.suffix.lower() in video_exts] if _resolve_model_artifact(UPLOAD_FOLDER).exists() else []
    metadata_by_path = read_video_metadata_many(library_files + upload_files)
    if library_files:
        for p in library_files:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            rel = p.relative_to(VIDEO_LIBRARY)
            metadata = metadata_by_path[p]
            videos.append({'name': str(rel), 'path': str(p.resolve()), 'size': p.stat().st_size, 'duration': metadata['duration']})
    if upload_files:
        for p in upload_files:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            metadata = metadata_by_path[p]
            videos.append({'name': p.name, 'path': str(p.resolve()), 'size': p.stat().st_size, 'duration': metadata['duration']})
    return videos

@app.route('/api/demo/videos')
def demo_videos():
    try:
        return jsonify({'success': True, 'videos': _list_videos()})
    except Exception as exc:
        logger.exception('Error listing videos')
        return (jsonify({'success': False, 'error': str(exc)}), 500)

def _resolve_video_path(video_path_str: str) -> Path:
    p = Path(video_path_str)
    if p.is_absolute() and _resolve_model_artifact(p).exists():
        return p
    candidates = [UPLOAD_FOLDER / p.name, VIDEO_LIBRARY / p.name, PROJECT_ROOT / p]
    for c in candidates:
        if _resolve_model_artifact(c).exists():
            return c
    for found in _model_artifact_glob(VIDEO_LIBRARY, p.name, recursive=True):
        if _resolve_model_artifact(found).is_file():
            return found
    raise FileNotFoundError(f'Video not found: {video_path_str}')

def _merge_contiguous_detections(detections: List[Dict], gap_seconds: float=0.2) -> List[Dict]:
    """Merge contiguous detections of the same class into larger blocks."""
    if not detections:
        return []
    indexed = [{'source_index': i, 'play_type': d.get('play_type', 'other'), 'start_time': float(d.get('start_time', 0.0)), 'end_time': float(d.get('end_time', 0.0)), 'confidence': float(d.get('confidence', 0.0))} for i, d in enumerate(detections)]
    indexed.sort(key=lambda x: x['start_time'])
    merged: List[Dict] = []
    current = None
    for item in indexed:
        if current is None:
            current = {'play_type': item['play_type'], 'start_time': item['start_time'], 'end_time': item['end_time'], 'confidence': item['confidence'], 'source_indices': [item['source_index']]}
            continue
        same_type = item['play_type'] == current['play_type']
        contiguous = item['start_time'] <= current['end_time'] + gap_seconds
        if same_type and contiguous:
            current['end_time'] = max(current['end_time'], item['end_time'])
            current['confidence'] = max(current['confidence'], item['confidence'])
            current['source_indices'].append(item['source_index'])
        else:
            current['duration'] = current['end_time'] - current['start_time']
            merged.append(current)
            current = {'play_type': item['play_type'], 'start_time': item['start_time'], 'end_time': item['end_time'], 'confidence': item['confidence'], 'source_indices': [item['source_index']]}
    if current is not None:
        current['duration'] = current['end_time'] - current['start_time']
        merged.append(current)
    return merged

def _safe_segment_name(play_type: str, start_time: float, end_time: float, idx: int) -> str:
    start_ms = int(round(start_time * 1000))
    end_ms = int(round(end_time * 1000))
    play = (play_type or 'other').strip().lower().replace(' ', '_')
    return f'{idx:03d}_{play}_{start_ms}-{end_ms}.mp4'

def _create_export_zip(video_path: Path, merged_segments: List[Dict], analysis_id: str) -> Path:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_root = DEMO_EXPORTS_FOLDER / f'{analysis_id}_{timestamp}'
    export_root.mkdir(parents=True, exist_ok=True)
    metadata_records = []
    for i, seg in enumerate(merged_segments, start=1):
        class_name = seg.get('play_type', 'other')
        class_dir = export_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        clip_name = _safe_segment_name(class_name, float(seg.get('start_time', 0.0)), float(seg.get('end_time', 0.0)), i)
        clip_path = class_dir / clip_name
        ffmpeg_cmd = ['ffmpeg', '-y', '-ss', f"{float(seg.get('start_time', 0.0)):.3f}", '-to', f"{float(seg.get('end_time', 0.0)):.3f}", '-i', str(video_path), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-c:a', 'aac', '-movflags', '+faststart', str(clip_path)]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr_tail = (result.stderr or '')[-600:]
            raise RuntimeError(f'ffmpeg failed while creating {clip_name}: {stderr_tail}')
        metadata_records.append({'clip_file': str(clip_path.relative_to(export_root)).replace('\\', '/'), 'play_type': class_name, 'start_time': seg.get('start_time', 0.0), 'end_time': seg.get('end_time', 0.0), 'duration': seg.get('duration', 0.0), 'confidence': seg.get('confidence', 0.0), 'source_indices': seg.get('source_indices', [])})
    metadata = {'analysis_id': analysis_id, 'video_path': str(video_path), 'created_at': datetime.now().isoformat(timespec='seconds'), 'total_merged_clips': len(metadata_records), 'clips': metadata_records}
    with _resolve_model_artifact(export_root / 'merged_segments.json').open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    zip_path = export_root.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in _model_artifact_glob(export_root, '*', recursive=True):
            if _resolve_model_artifact(file_path).is_file():
                rel = file_path.relative_to(export_root)
                zf.write(file_path, arcname=str(rel).replace('\\', '/'))
    return zip_path

def _cache_analysis(results: Dict, resolved_video_path: Path) -> str:
    analysis_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    ANALYSIS_CACHE[analysis_id] = {'results': results, 'video_path': str(resolved_video_path), 'created_at': datetime.now().isoformat(timespec='seconds')}
    if len(ANALYSIS_CACHE) > 30:
        oldest = sorted(ANALYSIS_CACHE.items(), key=lambda kv: kv[1].get('created_at', ''))[:10]
        for key, _ in oldest:
            ANALYSIS_CACHE.pop(key, None)
    return analysis_id

@app.route('/api/demo/analyze', methods=['POST'])
def demo_analyze():
    try:
        payload = request.get_json(force=True) or {}
        video_path_str = payload.get('video_path')
        if not video_path_str:
            return (jsonify({'success': False, 'error': 'video_path is required'}), 400)
        confidence = float(payload.get('confidence_threshold', 0.5))
        segment_duration = int(payload.get('segment_duration', 7))
        video_path = _resolve_video_path(video_path_str)
        logger.info('Demo analyze request: %s', video_path)
        results = inference_engine.analyze_video(video_path=video_path, confidence_threshold=confidence, segment_duration=segment_duration, use_segments=True)
        if results.get('status') == 'error':
            return (jsonify({'success': False, 'error': results.get('error', 'Unknown error')}), 500)
        analysis_id = _cache_analysis(results=results, resolved_video_path=video_path)
        return jsonify({'success': True, 'fixed_model': BEST_MODEL_NAME, 'analysis_id': analysis_id, 'video_url': f'/api/demo/video-for-analysis/{analysis_id}', 'results': results})
    except Exception as exc:
        logger.exception('Demo analyze failed')
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/demo/video')
def demo_video_stream():
    """Serve a selected video safely using the resolver (supports absolute paths)."""
    try:
        video_path_str = request.args.get('video_path', '')
        if not video_path_str:
            return (jsonify({'success': False, 'error': 'video_path is required'}), 400)
        video_path = _resolve_video_path(video_path_str)
        return send_file(video_path, mimetype='video/mp4', conditional=True)
    except Exception as exc:
        logger.exception('Demo video stream failed')
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/demo/video-for-analysis/<analysis_id>')
def demo_video_for_analysis(analysis_id: str):
    """Serve source video from cached analysis id for stable playback links."""
    try:
        entry = ANALYSIS_CACHE.get(analysis_id)
        if not entry:
            return (jsonify({'success': False, 'error': 'analysis_id not found'}), 404)
        video_path = _resolve_video_path(entry.get('video_path', ''))
        return send_file(video_path, mimetype='video/mp4', conditional=True)
    except Exception as exc:
        logger.exception('Demo video-for-analysis failed')
        return (jsonify({'success': False, 'error': str(exc)}), 500)

@app.route('/api/demo/export-clips', methods=['POST'])
def demo_export_clips():
    """Export merged contiguous detections as class-grouped clips in a ZIP."""
    try:
        payload = request.get_json(force=True) or {}
        analysis_id = payload.get('analysis_id')
        if not analysis_id:
            return (jsonify({'success': False, 'error': 'analysis_id is required'}), 400)
        entry = ANALYSIS_CACHE.get(analysis_id)
        if not entry:
            return (jsonify({'success': False, 'error': 'analysis_id not found'}), 404)
        merge_gap = float(payload.get('merge_gap_seconds', 0.2))
        min_confidence = float(payload.get('min_confidence', 0.0))
        results = entry.get('results', {})
        detections = results.get('detections', [])
        filtered = [d for d in detections if _is_target_demo_play(d.get('play_type', '')) and float(d.get('confidence', 0.0)) >= min_confidence]
        if not filtered:
            return (jsonify({'success': False, 'error': 'No line/scrum detections to export with current filters'}), 400)
        merged = _merge_contiguous_detections(filtered, gap_seconds=merge_gap)
        video_path = _resolve_video_path(entry.get('video_path', ''))
        zip_path = _create_export_zip(video_path, merged, analysis_id)
        return send_file(zip_path, mimetype='application/zip', as_attachment=True, download_name=f'demo_merged_clips_{analysis_id}.zip')
    except Exception as exc:
        logger.exception('Demo export clips failed')
        return (jsonify({'success': False, 'error': str(exc)}), 500)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False, use_reloader=False)
