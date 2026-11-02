"""Separate prepared-input experiment: conversions and batching, NOT end-to-end.
Uses only arrays recorded by benchmark_extended.py inside this investigation.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time
import benchmark as isolation
HERE, ROOT = (isolation.HERE, isolation.ROOT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--class-copy', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if _resolve_model_artifact(HERE / f'results/{args.name}_batches.json').exists():
        raise FileExistsError(args.name)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import cv2
    import tensorflow as tf
    import torch
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    from src.web import keras_inference_engine as module
    namespace = dict(vars(module))
    exec(compile(_resolve_model_artifact(args.class_copy).read_text(encoding='utf-8'), str(args.class_copy), 'exec'), namespace)
    started = time.perf_counter()
    detector = namespace['M1H2D2CrossFoldDetector'](ROOT)
    load_seconds = time.perf_counter() - started
    data = dict(_load_model_result_array(args.input))
    geometry = data.pop('geometry')
    n = len(geometry)
    reference = json.loads(_resolve_model_artifact(args.reference).read_text(encoding='utf-8'))
    reference_prob = np.array([w['probabilities'] for w in reference['window_trace']])
    reference_heads = np.array([w['dual_probabilities'] for w in reference['window_trace']])
    graphs = []
    for item in detector.fold_models:

        def factory(model):
            return tf.function(lambda x: model(x, training=False), autograph=False, input_signature=[{'rgb': tf.TensorSpec([None, 200, 2048], tf.float32), 'pose': tf.TensorSpec([None, 200, 62], tf.float32), 'sequence_mask': tf.TensorSpec([None, 200], tf.bool)}])
        graphs.append(factory(item['dual']))
    t = time.perf_counter()
    first = {k: tf.convert_to_tensor(v[:1]) for k, v in data.items()}
    for graph in graphs:
        graph(first).numpy()
        graph(first).numpy()
    startup = time.perf_counter() - t

    def heads(batch, shared):
        outputs = []
        for start in range(0, n, batch):
            arrays = {k: v[start:start + batch] for k, v in data.items()}
            tensors = {k: tf.convert_to_tensor(v) for k, v in arrays.items()} if shared else None
            outputs.append(np.stack([graph(tensors if shared else {k: tf.convert_to_tensor(v) for k, v in arrays.items()}).numpy() for graph in graphs], axis=1))
        return np.concatenate(outputs, axis=0)

    def forests(batch):
        outputs = []
        for start in range(0, n, batch):
            part = []
            for item in detector.fold_models:
                forest = item['geometry']
                raw = forest.predict_proba(geometry[start:start + batch])
                ordered = np.zeros((len(raw), 3), np.float64)
                ordered[:, np.asarray(forest.classes_).astype(int)] = raw
                part.append(ordered)
            outputs.append(np.stack(part, axis=1))
        return np.concatenate(outputs, axis=0)
    head_cases = [('graph_per_fold_conversion_b1', 1, False), ('graph_shared_conversion_b1', 1, True), ('graph_shared_conversion_b4', 4, True), ('graph_shared_conversion_b8', 8, True)]
    forest_cases = [('forest_b1', 1), ('forest_b4', 4), ('forest_b8', 8), ('forest_all', n)]
    for name, batch, shared in head_cases:
        arrays = {k: tf.convert_to_tensor(v[:min(batch, n)]) for k, v in data.items()}
        for graph in graphs:
            graph(arrays).numpy()
    for _, batch in forest_cases:
        for item in detector.fold_models:
            item['geometry'].predict_proba(geometry[:batch])
    runs = []
    values = {}
    for repeat in range(args.repeats):
        cases = [('heads', *case) for case in head_cases] + [('forests', name, batch, True) for name, batch in forest_cases]
        if repeat % 2:
            cases.reverse()
        for group, name, batch, shared in cases:
            t = time.perf_counter()
            value = heads(batch, shared) if group == 'heads' else forests(batch)
            seconds = time.perf_counter() - t
            row = {'scenario': name, 'group': group, 'repeat': repeat + 1, 'batch': batch, 'seconds': seconds}
            runs.append(row)
            values[name] = value
            print(json.dumps(row), flush=True)
    cap = cv2.VideoCapture(reference['video'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    manifest = detector.manifest
    head_checks = [{'scenario': name, 'max_dual_probability_difference': float(np.max(np.abs(values[name] - reference_heads))), 'allclose': bool(np.allclose(values[name], reference_heads, atol=1e-06, rtol=0))} for name, _, _ in head_cases]
    forest_checks = [{'scenario': name, 'max_probability_difference_vs_b1': float(np.max(np.abs(values[name] - values['forest_b1']))), 'allclose': bool(np.allclose(values[name], values['forest_b1'], atol=1e-12, rtol=0))} for name, _ in forest_cases]
    checks = []
    for head_name, _, _ in head_cases:
        for forest_name, _ in forest_cases:
            hp, fp = (values[head_name], values[forest_name])
            fused = []
            for fold, item in enumerate(detector.fold_models):
                alpha = item['geometry_alpha']
                fused.append(detector._normalize(np.exp((1 - alpha) * np.log(np.clip(hp[:, fold], 1e-07, 1.0)) + alpha * np.log(np.clip(fp[:, fold], 1e-07, 1.0)))))
            probs = detector._normalize(np.mean(fused, axis=0)).astype(np.float32)
            windows = copy.deepcopy(reference['result']['window_predictions'])
            for w, p in zip(windows, probs):
                w['probabilities'] = {str(label): float(prob) for label, prob in zip(detector.label_vocab, p)}
                w['play_type'] = str(detector.label_vocab[np.argmax(p)])
                w['confidence'] = float(p.max())
            events = detector._post_process_event_detections(windows, confidence_threshold=manifest['recommended_web_thresholds']['scrum'], video_duration=reference['result']['metadata']['duration'], source_fps=fps, class_confidence_thresholds={'line': manifest['recommended_web_thresholds']['line']})

            def close(a, b):
                if isinstance(a, dict):
                    return a.keys() == b.keys() and all((close(a[k], b[k]) for k in a))
                if isinstance(a, list):
                    return len(a) == len(b) and all((close(x, y) for x, y in zip(a, b)))
                if isinstance(a, float):
                    return abs(a - b) <= 1e-06
                return a == b
            row = {'heads': head_name, 'forests': forest_name, 'max_fused_probability_difference': float(np.max(np.abs(probs - reference_prob))), 'probabilities_allclose': bool(np.allclose(probs, reference_prob, atol=1e-06, rtol=0)), 'classes_exact': all((a['play_type'] == b['play_type'] for a, b in zip(windows, reference['result']['window_predictions']))), 'events_close': close(events, reference['result']['detections']), 'event_classes_times_exact': len(events) == len(reference['result']['detections']) and all((all((a.get(k) == b.get(k) for k in ['play_type', 'start_time', 'end_time', 'frame_start', 'frame_end', 'duration'])) for a, b in zip(events, reference['result']['detections']))), 'review_events_close': close(detector._last_review_window_detections, reference['result']['review_detections_7s'])}
            checks.append(row)
    result = {'scope': 'prepared window arrays only, NOT full video processing', 'input': str(args.input.resolve()), 'reference': str(args.reference.resolve()), 'windows': n, 'load_seconds': load_seconds, 'trace_and_warm_seconds': startup, 'repeats': args.repeats, 'absolute_tolerance': 1e-06, 'relative_tolerance': 0, 'runs': runs, 'median_seconds': {name: statistics.median((r['seconds'] for r in runs if r['scenario'] == name)) for name in values}, 'checks': checks, 'head_checks': head_checks, 'forest_checks': forest_checks, 'forest_absolute_tolerance': 1e-12, 'graph_tracing_counts': [g.experimental_get_tracing_count() for g in graphs], 'rf_n_jobs': [item['geometry'].n_jobs for item in detector.fold_models]}
    isolation.save(f'results/{args.name}_batches.json', result)
    assert all((c['probabilities_allclose'] and c['classes_exact'] and c['events_close'] and c['event_classes_times_exact'] and c['review_events_close'] for c in checks))
    assert all((c['allclose'] for c in head_checks + forest_checks))
    print('BATCH EXPERIMENT COMPLETE', flush=True)
if __name__ == '__main__':
    main()
