"""Isolated graph inference on the full final-test VF_S01 video, timed end to end.

The original window method is used unchanged. Only the dual model callable is
wrapped. No feature cache is used in end-to-end runs. Python write audit is a
second safeguard, not an OS-level sandbox for native libraries. The old saved
full-video result is a read-only prediction reference, with no timing baseline.
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
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import types
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for key, value in {'CUDA_VISIBLE_DEVICES': '-1', 'TF_CPP_MIN_LOG_LEVEL': '2', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'NUMEXPR_NUM_THREADS': '8', 'PYTHONDONTWRITEBYTECODE': '1', 'YOLO_CONFIG_DIR': str(HERE / 'runtime/ultralytics'), 'KERAS_HOME': str(HERE / 'runtime/keras'), 'TORCH_HOME': str(HERE / 'runtime/torch'), 'MPLCONFIGDIR': str(HERE / 'runtime/matplotlib'), 'XDG_CACHE_HOME': str(HERE / 'runtime/cache'), 'PLAYER_GEOMETRY_POSE_CACHE_ROOT': str(HERE / 'runtime/pose_cache'), 'TEMP': str(HERE / 'runtime/tmp'), 'TMP': str(HERE / 'runtime/tmp')}.items():
    os.environ[key] = value
for name in ['runtime/tmp', 'runtime/keras', 'runtime/ultralytics', 'logs', 'results', 'inputs']:
    (HERE / name).mkdir(parents=True, exist_ok=True)
os.chdir(HERE)
WRITE_EVENTS = []

def audit(event, args):
    paths = []
    if event == 'open':
        path, mode, flags = args
        if isinstance(path, (str, bytes, os.PathLike)) and (mode and any((c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            paths = [path]
    elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'):
        paths = [args[0]]
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        paths = [args[0], args[1]]
    for path in paths:
        absolute = Path(os.fsdecode(path)).resolve()
        if not absolute.is_relative_to(HERE):
            raise PermissionError(f'WRITE BLOCKED by experiment audit: {event} {absolute}')
        WRITE_EVENTS.append({'event': event, 'path': str(absolute)})
sys.addaudithook(audit)

def save(path, value):
    with _resolve_model_artifact(HERE / path).open('x', encoding='utf-8') as file:
        json.dump(value, file, indent=2, ensure_ascii=False)

def snapshot():
    result = {}
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d != '.git' and Path(base, d).resolve() != HERE]
        for name in files:
            p = Path(base, name)
            stat = p.stat()
            result[str(p.relative_to(ROOT))] = [stat.st_size, stat.st_mtime_ns]
    return result

def sha(path):
    h = hashlib.sha256()
    with _resolve_model_artifact(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    job_start = time.perf_counter()
    video = ROOT / 'data/videos/partidos_evaluacion/VF_S01_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4'
    historical_path = ROOT / 'data/results/result_20260830_064334.json'
    if _resolve_model_artifact(HERE / 'results/result.json').exists():
        raise FileExistsError('Existing result: refusing overwrite')
    historical = json.loads(_resolve_model_artifact(historical_path).read_text(encoding='utf-8'))
    sys.path.insert(0, str(ROOT))
    t = time.perf_counter()
    import cv2
    import numpy as np
    import tensorflow as tf
    import torch
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    from src.web import keras_inference_engine as engine_module
    imports_seconds = time.perf_counter() - t
    source_path = _resolve_model_artifact(ROOT / 'src/web/keras_inference_engine.py')
    tree = ast.parse(_resolve_model_artifact(source_path).read_text(encoding='utf-8'))
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'M1H2D2CrossFoldDetector'))
    original_predict_ast = ast.dump(next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '_predict_sequence_window')))
    changes = 0
    for subnode in ast.walk(node):
        if isinstance(subnode, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'yolo_config_dir' for t in subnode.targets)):
            subnode.value = ast.parse("Path(os.environ['YOLO_CONFIG_DIR'])", mode='eval').body
            changes += 1
    assert changes == 1
    assert ast.dump(next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '_predict_sequence_window'))) == original_predict_ast
    class_tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    copy_path = _resolve_model_artifact(HERE / 'inputs/isolated_class.py')
    copy_path.write_text(ast.unparse(class_tree), encoding='utf-8')
    namespace = dict(vars(engine_module))
    exec(compile(class_tree, str(copy_path), 'exec'), namespace)
    t = time.perf_counter()
    with tf.device('/CPU:0'):
        detector = namespace['M1H2D2CrossFoldDetector'](ROOT)
    load_seconds = time.perf_counter() - t
    engine = engine_module.KerasInferenceEngine()
    engine.current_model = detector
    engine.current_model_type = 'm1_h2_d2_cross_fold'
    engine.current_model_name = detector.display_name
    stats, counts = (defaultdict(float), defaultdict(int))
    traces = []

    class GraphProxy:

        def __init__(self, model):
            self.graph = tf.function(lambda values: model(values, training=False), input_signature=[{'rgb': tf.TensorSpec([None, 200, 2048], tf.float32), 'pose': tf.TensorSpec([None, 200, 62], tf.float32), 'sequence_mask': tf.TensorSpec([None, 200], tf.bool)}], autograph=False)

        def __call__(self, inputs, training=False):
            assert training is False
            t = time.perf_counter()
            result = self.graph({k: tf.convert_to_tensor(v) for k, v in inputs.items()})
            result.numpy()
            stats['recurrent_seconds'] += time.perf_counter() - t
            counts['recurrent_calls'] += 1
            return result
    proxies = []
    for item in detector.fold_models:
        item['dual'] = GraphProxy(item['dual'])
        proxies.append(item['dual'])
    forest_fn = detector._ordered_forest_probability

    def timed_forest(forest, vector):
        t = time.perf_counter()
        result = forest_fn(forest, vector)
        stats['forest_seconds'] += time.perf_counter() - t
        counts['forest_calls'] += 1
        return result
    detector._ordered_forest_probability = timed_forest
    for name, metric in [('_predict_feature_batch', 'rgb'), ('_on_sampled_frame', 'yolo'), ('_finish_auxiliary_sequence', 'geometry_sequence'), ('_predict_sequence_window', 'window'), ('_post_process_event_detections', 'postprocess')]:

        def make_wrapper(fn, metric):

            def wrapper(*a, **kw):
                t = time.perf_counter()
                result = fn(*a, **kw)
                stats[metric + '_seconds'] += time.perf_counter() - t
                counts[metric + '_calls'] += 1
                return result
            return wrapper
        setattr(detector, name, make_wrapper(getattr(detector, name), metric))
    cap = cv2.VideoCapture(str(video))
    ok, raw = cap.read()
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert ok and source_fps > 0
    from models.M0.web_adapter import preprocess_raw_frame
    frame = preprocess_raw_frame(raw, target_size=224)
    warm = {}
    t = time.perf_counter()
    detector._predict_feature_batch([frame[:, :, [2, 1, 0]]] * 8, 8)
    warm['rgb_first_call_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    detector._on_sampled_frame(frame)
    warm['yolo_first_call_seconds'] = time.perf_counter() - t
    dummy = {'rgb': tf.zeros((1, 200, 2048), tf.float32), 'pose': tf.zeros((1, 200, 62), tf.float32), 'sequence_mask': tf.convert_to_tensor(np.arange(200)[None] < 175)}
    t = time.perf_counter()
    for proxy in proxies:
        proxy.graph(dummy).numpy()
    warm['graph_trace_and_first_call_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    for proxy in proxies:
        proxy.graph(dummy).numpy()
    warm['graph_second_call_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    for item in detector.fold_models:
        forest_fn(item['geometry'], np.zeros((1, 549), np.float32))
    warm['forest_first_call_seconds'] = time.perf_counter() - t
    references = [source_path, _resolve_model_artifact(ROOT / 'models/M0/web_adapter.py'), detector.manifest_path, ROOT / 'models/M0/checkpoints/feature_extractor_resnet50.keras', detector.pose_model_path, ROOT / 'models/M0/checkpoints/best_model/best_model.keras', historical_path, video]
    references += [ROOT / p for p in ['models/M0/data_utils.py', 'models/M0/config.py', 'models/common/spatial.py', 'models/M1/geometry_features.py', 'models/M1/pose_cache.py', 'models/M1/config.py']]
    references += [ROOT / row[key] for row in detector.manifest['folds'] for key in ['dual_model', 'geometry_rf']]
    hashes_before = {str(p): sha(p) for p in references}
    startup = {'video': str(video), 'duration_seconds': total_frames / source_fps, 'source_frames': total_frames, 'source_fps': source_fps, 'sample_fps': 25, 'window_seconds': 7, 'stride_seconds': 1, 'confidence_threshold': historical['metadata']['confidence_threshold'], 'line_confidence_threshold': historical['metadata']['line_confidence_threshold'], 'threshold_source': 'same_as_historical_20260830_064334', 'imports_seconds': imports_seconds, 'model_load_seconds': load_seconds, 'warmup': warm, 'hashes_before': hashes_before, 'threads': {'tf_intra': 8, 'tf_inter': 1, 'torch_intra': 8, 'torch_inter': 1, 'opencv': 1}, 'rf_n_jobs': [i['geometry'].n_jobs for i in detector.fold_models], 'versions': {n: importlib.metadata.version(n) for n in ['tensorflow', 'keras', 'torch', 'ultralytics']}, 'external_load': 'Other CPU training remained active; measured times include external load.'}
    save('results/startup.json', startup)
    stats.clear()
    counts.clear()
    last_progress = [0.0]
    inference_start_utc = datetime.now(timezone.utc).isoformat()
    inference_start = time.perf_counter()
    print('INFERENCE_BEGIN ' + inference_start_utc, flush=True)

    def progress(percent, stage, message):
        now = time.perf_counter()
        if now - last_progress[0] >= 20 or percent in (18, 72, 96):
            row = {'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': now - inference_start, 'percent': percent, 'stage': stage, 'message': message, 'counts': dict(counts)}
            traces.append(row)
            with _resolve_model_artifact(HERE / 'logs/progress.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
            print('PROGRESS ' + json.dumps(row, ensure_ascii=False), flush=True)
            last_progress[0] = now
    result = engine.analyze_video(video, confidence_threshold=startup['confidence_threshold'], line_confidence_threshold=startup['line_confidence_threshold'], use_segments=True, segment_duration=7, temporal_fps_mode='training_match_25', feature_batch_size=256, feature_predict_batch_size=8, progress_callback=progress)
    inference_seconds = time.perf_counter() - inference_start
    inference_end_utc = datetime.now(timezone.utc).isoformat()
    if result['status'] != 'success':
        save('results/failure.json', result)
        raise RuntimeError(result)
    timing = dict(stats)
    timing.update(inference_seconds=inference_seconds, inference_start_utc=inference_start_utc, inference_end_utc=inference_end_utc, imports_seconds=imports_seconds, model_load_seconds=load_seconds, warmup=warm, duration_seconds=result['metadata']['duration'], seconds_per_video_second=inference_seconds / result['metadata']['duration'], counts=dict(counts), external_load=startup['external_load'], includes='decode, preprocessing, ResNet, YOLO, geometry, six graph duals, six forests, fusion, postprocessing and result formatting; no feature cache')
    timing['window_preparation_fusion_seconds'] = stats['window_seconds'] - stats['recurrent_seconds'] - stats['forest_seconds']
    timing['rest_seconds'] = inference_seconds - sum((stats[k] for k in ['rgb_seconds', 'yolo_seconds', 'window_seconds', 'geometry_sequence_seconds', 'postprocess_seconds']))
    result['metadata']['timing'] = timing
    result['metadata']['acceleration'] = 'tf.function cached per dual fold, fixed signature, batch 1; unchanged sampling and original window method'
    result['result_id'] = 'm1_graph_VF_S01_20261011'
    save('results/result.json', result)
    save('results/timing.json', timing)
    print('INFERENCE_COMPLETE ' + json.dumps(timing), flush=True)
    old, new = (historical['window_predictions'], result['window_predictions'])
    comparison = {'reference': str(historical_path), 'reference_has_measured_inference_time': False, 'window_counts': [len(old), len(new)], 'tolerance_absolute': 1e-06, 'tolerance_relative': 0}
    if len(old) == len(new):
        oldp = np.array([[w['probabilities'][k] for k in ['line', 'other', 'scrum']] for w in old])
        newp = np.array([[w['probabilities'][k] for k in ['line', 'other', 'scrum']] for w in new])
        comparison.update(max_absolute_probability_difference=float(np.max(np.abs(oldp - newp))), probabilities_equivalent=bool(np.allclose(oldp, newp, atol=1e-06, rtol=0)), class_differences=sum((a['play_type'] != b['play_type'] for a, b in zip(old, new))), window_time_differences=sum(((a['start_time'], a['end_time']) != (b['start_time'], b['end_time']) for a, b in zip(old, new))), historical_events=len(historical['detections']), new_events=len(result['detections']), event_classes_and_times_equal=[(e['play_type'], e['start_time'], e['end_time']) for e in historical['detections']] == [(e['play_type'], e['start_time'], e['end_time']) for e in result['detections']])
    save('results/historical_comparison.json', comparison)
    thresholds = detector.manifest['recommended_web_thresholds']
    alternate_events = detector._post_process_event_detections(new, confidence_threshold=float(thresholds['scrum']), video_duration=result['metadata']['duration'], source_fps=source_fps, class_confidence_thresholds={'line': float(thresholds['line'])})
    save('results/recommended_threshold_events.json', {'thresholds': thresholds, 'detections': alternate_events})
    import csv
    for name, rows in [('windows', new), ('events', result['detections'])]:
        with _resolve_model_artifact(HERE / f'results/{name}.csv').open('x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['play_type', 'start_seconds', 'end_seconds', 'duration_seconds', 'confidence', 'p_line', 'p_other', 'p_scrum'])
            for row in rows:
                writer.writerow([row['play_type'], row['start_time'], row['end_time'], row['duration'], row['confidence'], *[row['probabilities'][k] for k in ['line', 'other', 'scrum']]])
    hashes_after = {str(p): sha(p) for p in references}
    assert hashes_after == hashes_before, 'Existing inference references changed during run'
    save('results/preservation.json', {'hashes_before': hashes_before, 'hashes_after': hashes_after, 'all_equal': True, 'python_write_events': WRITE_EVENTS, 'scope': 'Python write audit and hashes of source, video, models, historical result; native library writes not intercepted by audit'})
    save('results/completion.json', {'utc': datetime.now(timezone.utc).isoformat(), 'job_seconds': time.perf_counter() - job_start, 'comparison': comparison, 'graph_trace_counts': [p.graph.experimental_get_tracing_count() for p in proxies]})
    print('DONE ' + json.dumps(comparison), flush=True)
if __name__ == '__main__':
    main()
