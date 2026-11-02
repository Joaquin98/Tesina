"""One model/video per process; all measurements and writes stay in this campaign."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import runtime as rt
from runtime import HERE, ROOT, save, sha
import argparse, ast, csv, json, time, types, sys
from collections import defaultdict
from datetime import datetime, timezone

def isolated_class(module, name):
    tree = ast.parse(_resolve_model_artifact(Path(module.__file__)).read_text(encoding='utf-8'))
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name))
    for n in ast.walk(node):
        if isinstance(n, ast.Assign) and any((isinstance(t, ast.Attribute) and t.attr == 'manifest_path' for t in n.targets)):
            n.value = ast.parse('SELECTED_MANIFEST_PATH', mode='eval').body
        if isinstance(n, ast.Compare) and any((isinstance(v, ast.Constant) and v.value == 6 for v in n.comparators)):
            n.comparators = [ast.parse('EXPECTED_MEMBERS', mode='eval').body if isinstance(v, ast.Constant) and v.value == 6 else v for v in n.comparators]
    changed = 0
    for n in ast.walk(node):
        if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'yolo_config_dir' for t in n.targets)):
            n.value = ast.parse("Path(os.environ['YOLO_CONFIG_DIR'])", mode='eval').body
            changed += 1
    assert changed == 1
    code = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(vars(module))
    namespace['SELECTED_MANIFEST_PATH'] = HERE / 'inputs' / f'{CURRENT_KEY}_manifest.json'
    namespace['EXPECTED_MEMBERS'] = EXPECTED_MEMBERS
    exec(compile(code, str(HERE / 'inputs' / f'{name}_isolated.py'), 'exec'), namespace)
    (HERE / 'inputs' / f'{name}_isolated.py').write_text(ast.unparse(code), encoding='utf-8')
    return namespace[name]
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', required=True)
    args = parser.parse_args()
    global CURRENT_KEY, EXPECTED_MEMBERS
    CURRENT_KEY = args.key
    EXPECTED_MEMBERS = 6 if args.key == 'M1-ensemble' else 1
    key = args.key
    prefix = f'results/{key}'
    job_start = time.perf_counter()
    if _resolve_model_artifact(HERE / f'{prefix}_result.json').exists():
        raise FileExistsError('Refusing overwrite')
    manifest = json.loads(_resolve_model_artifact(HERE / 'inputs' / f'{key}_manifest.json').read_text(encoding='utf-8'))
    video = ROOT / 'data/videos/partidos_evaluacion/VF_S04_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4'
    sys.path.insert(0, str(ROOT))
    t = time.perf_counter()
    import cv2, numpy as np, tensorflow as tf, torch
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    from src.web import keras_inference_engine as engine_module
    imports = time.perf_counter() - t
    t = time.perf_counter()
    if key.startswith('M0'):
        from m0_adapter import M0CrossFoldDetector
        detector = M0CrossFoldDetector(manifest)
    elif key.startswith('M1'):
        detector = isolated_class(engine_module, 'M1H2D2CrossFoldDetector')(ROOT)
    else:
        from adapters import build_detector
        detector = build_detector(key, manifest)
    load = time.perf_counter() - t
    assert len(detector.fold_models) == EXPECTED_MEMBERS
    detector.display_label = 'M1-H2-D2 · ' + ('ensemble de seis folds' if EXPECTED_MEMBERS == 6 else 'nuevo entrenamiento con todos los datos')
    stats, counts = (defaultdict(float), defaultdict(int))
    captured_features = []
    proxies = []
    if key.startswith('M1'):

        class GraphProxy:

            def __init__(self, model):
                self.graph = tf.function(lambda values: model(values, training=False), input_signature=[{'rgb': tf.TensorSpec([None, 200, 2048], tf.float32), 'pose': tf.TensorSpec([None, 200, 62], tf.float32), 'sequence_mask': tf.TensorSpec([None, 200], tf.bool)}], autograph=False)

            def __call__(self, inputs, training=False):
                t = time.perf_counter()
                value = self.graph({k: tf.convert_to_tensor(v) for k, v in inputs.items()})
                value.numpy()
                stats['recurrent_seconds'] += time.perf_counter() - t
                counts['recurrent_calls'] += 1
                return value
        for member in detector.fold_models:
            proxy = GraphProxy(member['dual'])
            member['dual'] = proxy
            proxies.append(proxy)
    else:

        class HistoricalProxy:

            def __init__(self, model):
                self.model = model

            def predict(self, *a, **kw):
                t = time.perf_counter()
                value = self.model.predict(*a, **kw)
                stats['historical_heads_seconds'] += time.perf_counter() - t
                counts['historical_head_calls'] += 1
                return value
        for member in detector.fold_models:
            member['historical'] = HistoricalProxy(member['historical'])
    for name in ['_predict_feature_batch', '_on_sampled_frame', '_on_raw_sampled_frame', '_finish_auxiliary_sequence', '_predict_sequence_window', '_post_process_event_detections']:

        def wrapped(fn, name):

            def call(*a, **kw):
                t = time.perf_counter()
                value = fn(*a, **kw)
                if name == '_predict_feature_batch':
                    captured_features.append(value)
                stats[name + '_seconds'] += time.perf_counter() - t
                counts[name + '_calls'] += 1
                return value
            return call
        setattr(detector, name, wrapped(getattr(detector, name), name))
    if key.startswith(('M2', 'M3')):
        detect = detector._detect_frame

        def detect_frame(*a, **kw):
            t = time.perf_counter()
            value = detect(*a, **kw)
            stats['yolo_detector_seconds'] += time.perf_counter() - t
            counts['yolo_detector_calls'] += 1
            return value
        detector._detect_frame = detect_frame
    warm = {}
    cap = cv2.VideoCapture(str(video))
    ok, raw = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert ok
    from models.M0.web_adapter import preprocess_raw_frame
    crop = preprocess_raw_frame(raw, target_size=224)
    t = time.perf_counter()
    detector._predict_feature_batch([crop[:, :, [2, 1, 0]]] * 8, 8)
    warm['rgb_seconds'] = time.perf_counter() - t
    detector._web_effective_fps = 25
    detector._begin_auxiliary_sequence(1)
    t = time.perf_counter()
    detector._on_raw_sampled_frame(raw)
    detector._on_sampled_frame(crop)
    warm['pose_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    if proxies:
        dummy = {'rgb': tf.zeros([1, 200, 2048]), 'pose': tf.zeros([1, 200, 62]), 'sequence_mask': tf.convert_to_tensor(np.arange(200)[None] < 175)}
        for p in proxies:
            p.graph(dummy).numpy()
        for member in detector.fold_models:
            detector._ordered_forest_probability(member['geometry'], np.zeros((1, 549), np.float32))
    else:
        for member in detector.fold_models:
            member['historical'].predict([np.zeros((1, 200, 2048), np.float32), np.ones((1, 200), np.float32)], verbose=0)
    warm['heads_first_call_seconds'] = time.perf_counter() - t
    if key.startswith('M2'):
        from adapters import validate_m2
        save(f'{prefix}_adapter_validation.json', validate_m2(detector, manifest))
    if key.startswith('M0'):
        from m0_adapter import validate_m0
        save(f'{prefix}_adapter_validation.json', validate_m0(detector, manifest))
    source_refs = [_resolve_model_artifact(ROOT / 'src/web/keras_inference_engine.py'), _resolve_model_artifact(ROOT / 'models/M0/web_adapter.py'), _resolve_model_artifact(ROOT / 'models/M2/predictor.py'), _resolve_model_artifact(ROOT / 'models/M3/predictor.py'), _resolve_model_artifact(ROOT / 'models/common/spatial.py'), _resolve_model_artifact(ROOT / 'models/M1/geometry_features.py'), _resolve_model_artifact(ROOT / 'models/M1/pose_cache.py'), _resolve_model_artifact(ROOT / 'models/common/event_verifier.py'), _resolve_model_artifact(ROOT / 'models/common/comparison/models.py'), video, ROOT / 'models/M0/checkpoints/feature_extractor_resnet50.keras', ROOT / 'yolo11n-pose.pt']
    for fold in manifest['folds']:
        for field in ['dual_model', 'geometry_rf', 'dynamic_pca', 'quantile_scaler', 'logistic', 'historical_model', 'temporal_model', 'line_verifier', 'run_metadata', 'predictions']:
            if field in fold:
                source_refs.append(ROOT / fold[field])
    hashes = {str(p): sha(p) for p in set(source_refs)}
    thresholds = manifest['recommended_web_thresholds']
    if key.startswith('M1'):
        thresholds = manifest['recommended_web_thresholds']
    save(f'{prefix}_startup.json', {'model': key, 'imports_seconds': imports, 'model_load_seconds': load, 'warmup': warm, 'thresholds': thresholds, 'duration_seconds': frames / fps, 'hashes_before': hashes, 'implementation': 'cached_tf_function_batch1' if proxies else 'existing_keras_predict_batch1', 'threads': {'tf_intra': 8, 'tf_inter': 1, 'torch_intra': 8, 'torch_inter': 1, 'opencv': 1}})
    engine = engine_module.KerasInferenceEngine()
    engine.current_model = detector
    engine.current_model_name = detector.display_label
    engine.current_model_type = 'm1_h2_d2_cross_fold' if proxies else manifest['model_name']
    stats.clear()
    counts.clear()
    captured_features.clear()
    last = [0.0]
    start = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()

    def progress(percent, stage, message):
        now = time.perf_counter()
        if now - last[0] >= 20 or percent in [18, 72, 96]:
            row = {'model': key, 'utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': now - start, 'stage': stage, 'message': message, 'percent': percent, 'counts': dict(counts)}
            with _resolve_model_artifact(HERE / 'logs' / f'{key}_progress.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
            (HERE / 'progress.json').write_text(json.dumps(row, ensure_ascii=False), encoding='utf-8')
            print('PROGRESS ' + json.dumps(row, ensure_ascii=False), flush=True)
            last[0] = now
    result = engine.analyze_video(video, confidence_threshold=float(thresholds['scrum']), line_confidence_threshold=float(thresholds['line']), use_segments=True, segment_duration=7, temporal_fps_mode='training_match_25', feature_batch_size=256, feature_predict_batch_size=8, progress_callback=progress)
    elapsed = time.perf_counter() - start
    end_utc = datetime.now(timezone.utc).isoformat()
    if result['status'] != 'success':
        save(f'{prefix}_failure.json', result)
        raise RuntimeError(result)
    timing = {'inference_seconds': elapsed, 'inference_start_utc': start_utc, 'inference_end_utc': end_utc, 'duration_seconds': result['metadata']['duration'], 'imports_seconds': imports, 'model_load_seconds': load, 'warmup': warm, 'stage_seconds': dict(stats), 'counts': dict(counts), 'projection_80min_seconds': elapsed * 4800 / result['metadata']['duration'], 'includes': 'decode, preprocess, RGB, pose, geometry, selected members (six or one), event policy and result formatting; no shared feature cache'}
    if key.startswith('M0'):
        timing['includes'] = 'decode, preprocess, fresh ResNet-50 RGB, six BiGRU folds, fusion, event policy and result formatting; no pose branch or persisted feature reuse'
        timing['counts']['yolo_detector_calls'] = 0
    result['model_name'] = detector.display_label
    result['metadata']['timing'] = timing
    result['metadata']['campaign'] = 'm1_ensemble_vs_single_vf_s04_20261013'
    result['result_id'] = key + '_VF_S04_20261013'
    save(f'{prefix}_result.json', result)
    save(f'{prefix}_timing.json', timing)
    wins = result['window_predictions']
    assert len(wins) == 1195 and counts['_on_sampled_frame_calls'] == 30030 and (counts['_predict_sequence_window_calls'] == 1195)
    assert counts['recurrent_calls' if proxies else 'historical_head_calls'] == 1195 * EXPECTED_MEMBERS
    assert all((np.all(np.isfinite(list(w['probabilities'].values()))) and abs(sum(w['probabilities'].values()) - 1) < 1e-05 for w in wins))
    comparison = {'scope': 'new comparable fresh-video measurement; prior VF timing not treated as the paired baseline'}
    assert [p.graph.experimental_get_tracing_count() for p in proxies] == [1] * EXPECTED_MEMBERS
    from quality_from_fresh_features import evaluate
    evaluate(detector, captured_features, key, thresholds, video)
    after = {str(p): sha(p) for p in set(source_refs)}
    assert after == hashes
    save(f'{prefix}_completion.json', {'model': key, 'all_checks_pass': True, 'hashes_unchanged': True, 'hashes_after': after, 'comparison_previous_run': comparison, 'graph_trace_counts': [p.graph.experimental_get_tracing_count() for p in proxies], 'job_seconds': time.perf_counter() - job_start, 'python_write_events': rt.WRITE_EVENTS})
    for name, rows in [('events', result['detections']), ('windows', wins)]:
        with _resolve_model_artifact(HERE / f'{prefix}_{name}.csv').open('x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['class', 'start_seconds', 'end_seconds', 'confidence', 'p_line', 'p_other', 'p_scrum'])
            for w in rows:
                writer.writerow([w['play_type'], w['start_time'], w['end_time'], w['confidence'], *[w['probabilities'][label] for label in ['line', 'other', 'scrum']]])
    print('COMPLETE ' + json.dumps({'model': key, 'inference_seconds': elapsed, 'windows': len(wins), 'events': len(result['detections'])}), flush=True)
if __name__ == '__main__':
    main()
