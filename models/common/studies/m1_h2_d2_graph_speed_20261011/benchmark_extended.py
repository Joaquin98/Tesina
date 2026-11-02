"""Isolated, sequential end-to-end comparison. Existing sources are read only.

The original window method is used unchanged. Only the dual model callable is
wrapped. No feature cache is used in end-to-end runs. Python write audit is a
second safeguard, not an OS-level sandbox for native libraries.
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True, help='New result prefix, refuses overwrite')
    parser.add_argument('--videos', nargs='+', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    if list(_model_artifact_glob(HERE / 'results', args.campaign + '*', recursive=False)):
        raise FileExistsError('Campaign prefix already used')
    print('Snapshot existing file metadata...', flush=True)
    before = snapshot()
    save(f'results/{args.campaign}_before_metadata.json', before)
    sys.path.insert(0, str(ROOT))
    imports_start = time.perf_counter()
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
    imports_seconds = time.perf_counter() - imports_start
    source_path = _resolve_model_artifact(ROOT / 'src/web/keras_inference_engine.py')
    tree = ast.parse(_resolve_model_artifact(source_path).read_text(encoding='utf-8'))
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'M1H2D2CrossFoldDetector'))
    changes = 0
    for subnode in ast.walk(node):
        if isinstance(subnode, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'yolo_config_dir' for t in subnode.targets)):
            subnode.value = ast.parse("Path(os.environ['YOLO_CONFIG_DIR'])", mode='eval').body
            changes += 1
    assert changes == 1
    class_tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    copy_path = HERE / f'inputs/{args.campaign}_isolated_class.py'
    copy_path.write_text(ast.unparse(class_tree), encoding='utf-8')
    namespace = dict(vars(engine_module))
    exec(compile(class_tree, str(copy_path), 'exec'), namespace)
    isolated_class = namespace['M1H2D2CrossFoldDetector']
    load_start = time.perf_counter()
    with tf.device('/CPU:0'):
        detector = isolated_class(ROOT)
    load_seconds = time.perf_counter() - load_start
    original = engine_module.M1H2D2CrossFoldDetector
    assert ast.dump(next((n for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n.name == '_predict_sequence_window'))) == ast.dump(next((n for n in ast.walk(ast.parse(_resolve_model_artifact(source_path).read_text(encoding='utf-8'))) if isinstance(n, ast.ClassDef) and n.name == original.__name__)).body[-1])
    engine = engine_module.KerasInferenceEngine()
    engine.current_model = detector
    engine.current_model_type = 'm1_h2_d2_cross_fold_ensemble'
    engine.current_model_name = 'm1_h2_d2_cross_fold_ensemble'
    stats = defaultdict(float)
    counts = defaultdict(int)
    traces = []
    current = {'mode': 'direct', 'fold_outputs': [], 'inputs': [], 'window': None}
    captured_inputs = []
    captured_geometry = []

    def array_hash(value):
        a = np.ascontiguousarray(value)
        return hashlib.sha256(a.tobytes()).hexdigest()

    class DualProxy:

        def __init__(self, model, fold):
            self.model, self.fold = (model, fold)
            self.graph = tf.function(lambda values: model(values, training=False), input_signature=[{'rgb': tf.TensorSpec([None, 200, 2048], tf.float32), 'pose': tf.TensorSpec([None, 200, 62], tf.float32), 'sequence_mask': tf.TensorSpec([None, 200], tf.bool)}], autograph=False)

        def __call__(self, inputs, training=False):
            if self.fold == 0:
                t = time.perf_counter()
                current['inputs'].append({k: array_hash(v) for k, v in inputs.items()})
                captured_inputs.append(inputs)
                stats['instrumentation_seconds'] += time.perf_counter() - t
            t = time.perf_counter()
            if current['mode'] == 'direct':
                value = self.model(inputs, training=False)
            else:
                tensors = {k: tf.convert_to_tensor(v) for k, v in inputs.items()}
                value = self.graph(tensors)
            materialized = value.numpy()
            stats['recurrent_seconds'] += time.perf_counter() - t
            counts['recurrent_calls'] += 1
            current['fold_outputs'].append(materialized[0].tolist())
            return value
    proxies = []
    for i, item in enumerate(detector.fold_models):
        proxy = DualProxy(item['dual'], i)
        item['dual'] = proxy
        proxies.append(proxy)
    ordered_forest = detector._ordered_forest_probability

    def timed_forest(forest, vector):
        if forest is detector.fold_models[0]['geometry']:
            captured_geometry.append(vector)
        t = time.perf_counter()
        value = ordered_forest(forest, vector)
        stats['forest_seconds'] += time.perf_counter() - t
        counts['forest_calls'] += 1
        return value
    detector._ordered_forest_probability = timed_forest

    def wrap(name, metric, after=None):
        fn = getattr(detector, name)

        def call(*a, **kw):
            t = time.perf_counter()
            value = fn(*a, **kw)
            stats[metric] += time.perf_counter() - t
            counts[metric.replace('_seconds', '_calls')] += 1
            if after:
                t = time.perf_counter()
                after(value, *a, **kw)
                stats['instrumentation_seconds'] += time.perf_counter() - t
            return value
        setattr(detector, name, call)
    feature_hashes = []
    captured_windows = []

    def capture_features(value, *a, **kw):
        feature_hashes.append(array_hash(value))
    wrap('_predict_feature_batch', 'rgb_seconds', capture_features)
    wrap('_on_sampled_frame', 'yolo_seconds')
    wrap('_finish_auxiliary_sequence', 'geometry_sequence_seconds')
    wrap('_post_process_event_detections', 'postprocess_seconds')
    predict_window = detector._predict_sequence_window

    def timed_window(features, mask, start, end):
        t = time.perf_counter()
        current['fold_outputs'] = []
        output = predict_window(features, mask, start, end)
        stats['window_seconds'] += time.perf_counter() - t
        counts['window_calls'] += 1
        traces.append({'start_sample': start, 'end_sample': end, 'rgb_mask_sha256': array_hash(mask), 'dual_probabilities': current['fold_outputs'], 'probabilities': output[0].tolist()})
        return output
    detector._predict_sequence_window = timed_window
    cap = cv2.VideoCapture(str(args.videos[0].resolve()))
    ok, frame = cap.read()
    cap.release()
    assert ok
    from models.M0.web_adapter import preprocess_raw_frame
    frame = preprocess_raw_frame(frame, target_size=224)
    warm = {}
    t = time.perf_counter()
    detector._predict_feature_batch([frame[:, :, [2, 1, 0]]] * 8, 8)
    warm['rgb_first_call_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    detector._on_sampled_frame(frame)
    warm['yolo_first_call_seconds'] = time.perf_counter() - t
    dummy = {'rgb': np.zeros((1, 200, 2048), np.float32), 'pose': np.zeros((1, 200, 62), np.float32), 'sequence_mask': np.arange(200)[None] < 175}
    t = time.perf_counter()
    for p in proxies:
        p(dummy).numpy()
    warm['direct_heads_first_call_seconds'] = time.perf_counter() - t
    tensor_dummy = {k: tf.convert_to_tensor(v) for k, v in dummy.items()}
    t = time.perf_counter()
    for p in proxies:
        p.graph(tensor_dummy).numpy()
    warm['graph_trace_and_first_call_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    for p in proxies:
        p.graph(tensor_dummy).numpy()
    warm['graph_second_call_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    for item in detector.fold_models:
        ordered_forest(item['geometry'], np.zeros((1, 549), np.float32))
    warm['forest_first_call_seconds'] = time.perf_counter() - t
    print('LOAD/WARMUP ' + json.dumps({'imports_seconds': imports_seconds, 'load_seconds': load_seconds, **warm}), flush=True)
    manifest = detector.manifest
    references = [source_path, _resolve_model_artifact(ROOT / 'models/M0/web_adapter.py'), detector.manifest_path, ROOT / 'models/M0/checkpoints/feature_extractor_resnet50.keras', detector.pose_model_path]
    references += [ROOT / p for p in ['models/M0/data_utils.py', 'models/M0/config.py', 'models/common/spatial.py', 'models/M1/geometry_features.py', 'models/M1/pose_cache.py', 'models/M1/config.py']]
    references += [ROOT / row[key] for row in manifest['folds'] for key in ['dual_model', 'geometry_rf']]
    references += [p.resolve() for p in args.videos]
    hashes_before = {str(p): sha(p) for p in references}
    summary = {'campaign': args.campaign, 'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable, 'versions': {n: importlib.metadata.version(n) for n in ['tensorflow', 'keras', 'torch', 'ultralytics']}, 'threads': {'tf_intra': 8, 'tf_inter': 1, 'torch_intra': 8, 'torch_inter': 1, 'opencv': 1}, 'rf_n_jobs': [item['geometry'].n_jobs for item in detector.fold_models], 'imports_seconds': imports_seconds, 'load_seconds': load_seconds, 'warmup': warm, 'hashes_before': hashes_before, 'runs': [], 'equivalence': []}
    save(f'results/{args.campaign}_startup.json', summary)
    for rep in range(args.repeats):
        for video_index, path in enumerate(args.videos):
            path = path.resolve()
            pair = []
            modes = ['direct', 'graph'] if (rep + video_index) % 2 == 0 else ['graph', 'direct']
            for mode in modes:
                key = f'{args.campaign}_{path.stem}_r{rep + 1}_{mode}'
                stats.clear()
                counts.clear()
                traces.clear()
                feature_hashes.clear()
                current['inputs'].clear()
                captured_inputs.clear()
                captured_geometry.clear()
                current['mode'] = mode
                print(f'BEGIN {key}', flush=True)
                t = time.perf_counter()
                result = engine.analyze_video(path, confidence_threshold=float(manifest['recommended_web_thresholds']['scrum']), line_confidence_threshold=float(manifest['recommended_web_thresholds']['line']), use_segments=True, segment_duration=7, temporal_fps_mode='training_match_25', feature_batch_size=256, feature_predict_batch_size=8, progress_callback=lambda percent, stage, message: print(f'PROGRESS {mode} {percent:.0f}% {message}', flush=True))
                total = time.perf_counter() - t
                if result['status'] != 'success':
                    raise RuntimeError(result)
                timings = dict(stats)
                timings['total_seconds'] = total
                timings['window_preparation_fusion_seconds'] = stats['window_seconds'] - stats['recurrent_seconds'] - stats['forest_seconds']
                timings['rest_seconds'] = total - stats['rgb_seconds'] - stats['yolo_seconds'] - stats['window_seconds'] - stats['geometry_sequence_seconds'] - stats['postprocess_seconds']
                row = {'id': key, 'mode': mode, 'repeat': rep + 1, 'video': str(path), 'timings': timings, 'counts': dict(counts), 'feature_sha256': list(feature_hashes), 'window_inputs_sha256': list(current['inputs']), 'geometry_mask_sha256': array_hash(detector._geometry_mask), 'geometry_positions_sha256': array_hash(detector._geometry_positions), 'geometry_poses_sha256': array_hash(detector._geometry_poses), 'geometry_vectors_sha256': [array_hash(v) for v in captured_geometry], 'window_trace': list(traces), 'result': result}
                save(f'results/{key}.json', row)
                if mode == 'graph':
                    np.savez_compressed(HERE / f'inputs/{key}_window_arrays.npz', **{k: np.concatenate([v[k] for v in captured_inputs], axis=0) for k in captured_inputs[0]}, geometry=np.concatenate(captured_geometry, axis=0))
                summary['runs'].append({k: row[k] for k in ['id', 'mode', 'repeat', 'video', 'timings', 'counts']})
                pair.append(row)
                print('END ' + json.dumps(summary['runs'][-1]), flush=True)
            direct = next((r for r in pair if r['mode'] == 'direct'))
            graph = next((r for r in pair if r['mode'] == 'graph'))
            checks = {}
            for field in ['feature_sha256', 'window_inputs_sha256', 'geometry_mask_sha256', 'geometry_positions_sha256', 'geometry_poses_sha256', 'geometry_vectors_sha256']:
                checks[field] = direct[field] == graph[field]
            dp = np.array([w['probabilities'] for w in direct['window_trace']])
            gp = np.array([w['probabilities'] for w in graph['window_trace']])
            df = np.array([w['dual_probabilities'] for w in direct['window_trace']])
            gf = np.array([w['dual_probabilities'] for w in graph['window_trace']])
            checks['probabilities_allclose'] = bool(np.allclose(dp, gp, atol=1e-06, rtol=0))
            checks['dual_probabilities_allclose'] = bool(np.allclose(df, gf, atol=1e-06, rtol=0))
            wp_d, wp_g = [r['result']['window_predictions'] for r in [direct, graph]]
            checks['window_classes_and_times_exact'] = all((all((a[k] == b[k] for k in ['play_type', 'start_time', 'end_time', 'frame_start', 'frame_end'])) for a, b in zip(wp_d, wp_g))) and len(wp_d) == len(wp_g)

            def close(a, b):
                if isinstance(a, dict):
                    return a.keys() == b.keys() and all((close(a[k], b[k]) for k in a))
                if isinstance(a, list):
                    return len(a) == len(b) and all((close(x, y) for x, y in zip(a, b)))
                if isinstance(a, float):
                    return abs(a - b) <= 1e-06
                return a == b
            checks['events_all_fields_close'] = close(direct['result']['detections'], graph['result']['detections'])
            checks['review_events_all_fields_close'] = close(direct['result']['review_detections_7s'], graph['result']['review_detections_7s'])
            checks['event_classes_and_times_exact'] = len(direct['result']['detections']) == len(graph['result']['detections']) and all((all((a.get(k) == b.get(k) for k in ['play_type', 'start_time', 'end_time', 'frame_start', 'frame_end'])) for a, b in zip(direct['result']['detections'], graph['result']['detections'])))
            eq = {'video': str(path), 'repeat': rep + 1, 'absolute_tolerance': 1e-06, 'relative_tolerance': 0, 'max_fused_probability_difference': float(np.max(np.abs(dp - gp))), 'max_dual_probability_difference': float(np.max(np.abs(df - gf))), 'checks': checks, 'events_byte_equal': direct['result']['detections'] == graph['result']['detections'], 'all_pass': all(checks.values()), 'minimum_probability_distance_to_threshold': float(min((abs(w['probabilities'][label] - manifest['recommended_web_thresholds'][label]) for w in wp_d for label in ['line', 'scrum'])))}
            summary['equivalence'].append(eq)
            save(f'results/{args.campaign}_{path.stem}_r{rep + 1}_equivalence.json', eq)
            print('EQUIVALENCE ' + json.dumps(eq), flush=True)
            if not eq['all_pass']:
                raise AssertionError(eq)
    summary['hashes_after'] = {str(p): sha(p) for p in references}
    summary['reference_content_unchanged'] = summary['hashes_after'] == hashes_before
    after = snapshot()
    summary['existing_files_changed'] = [p for p in before if after.get(p) != before[p]]
    summary['unexpected_new_files_outside_experiment'] = [p for p in after if p not in before]
    summary['existing_files_deleted'] = [p for p in before if p not in after]
    summary['graph_tracing_counts'] = [p.graph.experimental_get_tracing_count() for p in proxies]
    summary['write_audit_events'] = WRITE_EVENTS
    save(f'results/{args.campaign}_summary.json', summary)
    assert summary['reference_content_unchanged']
    print('CAMPAIGN COMPLETE', flush=True)
if __name__ == '__main__':
    main()
