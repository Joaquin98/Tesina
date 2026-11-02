"""Resumable frozen single-model inference, with source timestamps intact.

Extraction is checkpointed every 120 seconds (first block is a 30s pilot).
All overlapping 7s/1s windows are inferred globally, including block boundaries.
No thresholds, weights, crops, or source frames are adjusted using predictions.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import os, sys, json, time, hashlib, csv, platform, traceback
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
from datetime import datetime, timezone
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for name in ['runtime/tmp', 'runtime/keras', 'runtime/ultralytics', 'runtime/cache', 'runtime/matplotlib', 'results/features']:
    (HERE / name).mkdir(parents=True, exist_ok=True)
for k, v in {'CUDA_VISIBLE_DEVICES': '-1', 'TF_CPP_MIN_LOG_LEVEL': '2', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'NUMEXPR_NUM_THREADS': '4', 'PYTHONDONTWRITEBYTECODE': '1', 'YOLO_CONFIG_DIR': str(HERE / 'runtime/ultralytics'), 'KERAS_HOME': str(HERE / 'runtime/keras'), 'TORCH_HOME': str(HERE / 'runtime/torch'), 'MPLCONFIGDIR': str(HERE / 'runtime/matplotlib'), 'XDG_CACHE_HOME': str(HERE / 'runtime/cache'), 'PLAYER_GEOMETRY_POSE_CACHE_ROOT': str(HERE / 'runtime/pose_cache'), 'TEMP': str(HERE / 'runtime/tmp'), 'TMP': str(HERE / 'runtime/tmp')}.items():
    os.environ[k] = v
sys.path.insert(0, str(ROOT))

def save(p, obj):
    dest = HERE / p
    tmp = dest.with_suffix(dest.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(dest)

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(p)).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()
started = time.perf_counter()

def progress(stage, **kw):
    row = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), process_elapsed_s=time.perf_counter() - started, **kw)
    save('progress.json', row)
    with _resolve_model_artifact(HERE / 'logs/progress.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)

def main():
    import cv2, numpy as np, pandas as pd, tensorflow as tf, torch, psutil
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    source = ROOT / 'data/m1_ensemble_vs_all_development_20261012'
    manifestpath = source / 'inputs/M1-single_manifest.json'
    manifest = json.loads(_resolve_model_artifact(manifestpath).read_text(encoding='utf-8'))
    video = HERE / 'video/jQqN_K0o5_I.mp4'
    refs = [manifestpath, source / 'inputs/all_training_examples.csv', source / 'results/training.json', ROOT / 'models/M0/checkpoints/feature_extractor_resnet50.keras', ROOT / 'yolo11n-pose.pt', _resolve_model_artifact(ROOT / 'src/web/keras_inference_engine.py'), _resolve_model_artifact(ROOT / 'models/M0/web_adapter.py'), _resolve_model_artifact(ROOT / 'models/M0/data_utils.py'), _resolve_model_artifact(ROOT / 'models/common/spatial.py'), _resolve_model_artifact(ROOT / 'models/M1/pose_cache.py'), _resolve_model_artifact(ROOT / 'models/M1/geometry_features.py')] + [ROOT / manifest['folds'][0][k] for k in ['dual_model', 'geometry_rf']]
    hashes = {str(p): sha(p) for p in refs}
    trained = json.loads(_resolve_model_artifact(source / 'results/training.json').read_text())
    assert all((hashes[p] == v for p, v in trained['new_weight_hashes'].items()))
    assert manifest['deployment_policy'] == 'one_new_dual_and_one_new_forest_trained_on_all_8796_examples'
    save('inputs/model_hashes.json', hashes)
    save('inputs/model_manifest.json', manifest)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(3))
    h = int(cap.get(4))
    cap.release()
    assert fps == 25 and total > 100000
    duration = total / fps
    save('inputs/video_properties.json', dict(path=str(video), sha256=sha(video), width=w, height=h, fps=fps, frames=total, duration_s=duration, audio='not acquired; video only; timestamps unchanged'))
    from src.web import keras_inference_engine as em
    classcode = _resolve_model_artifact(_resolve_model_artifact(source / 'inputs/M1H2D2CrossFoldDetector_isolated.py')).read_text(encoding='utf-8')
    old_selection = 'np.linspace(0, len(positions) - 1, self.POSE_FRAMES_PER_WINDOW, dtype=np.int32)'
    assert classcode.count(old_selection) == 1
    classcode = classcode.replace(old_selection, 'np.rint(np.linspace(0, len(positions) - 1, self.POSE_FRAMES_PER_WINDOW)).astype(np.int32)')
    (_resolve_model_artifact(HERE / 'inputs/isolated_model_class.py')).write_text(classcode, encoding='utf-8')
    ns = dict(vars(em))
    ns.update(SELECTED_MANIFEST_PATH=manifestpath, EXPECTED_MEMBERS=1)
    exec(compile(classcode, str(_resolve_model_artifact(HERE / 'inputs/isolated_model_class.py')), 'exec'), ns)
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    t = time.perf_counter()
    detector = ns['M1H2D2CrossFoldDetector'](ROOT)
    load_s = time.perf_counter() - t
    faulthandler.cancel_dump_traceback_later()
    assert len(detector.fold_models) == 1

    class GraphProxy:

        def __init__(self, model):
            self.graph = tf.function(lambda x: model(x, training=False), input_signature=[{'rgb': tf.TensorSpec([None, 200, 2048], tf.float32), 'pose': tf.TensorSpec([None, 200, 62], tf.float32), 'sequence_mask': tf.TensorSpec([None, 200], tf.bool)}], autograph=False)

        def __call__(self, inputs, training=False):
            return self.graph({k: tf.convert_to_tensor(v) for k, v in inputs.items()})
    for item in detector.fold_models:
        item['geometry'].n_jobs = 1
        item['dual'] = GraphProxy(item['dual'])
    from models.M0.web_adapter import preprocess_raw_frame
    from models.common.spatial import training_normalization_geometry, normalize_yolo_source_roi
    policy = {k: getattr(detector, k) for k in dir(detector) if k.startswith('EVENT_')}
    save('inputs/config.json', dict(model=manifest['source_configuration'], protocol='training_faithful_highres_rint100', threads=4, forest_jobs=1, feature_predict_batch_size=8, effective_fps=25, window_s=7, stride_s=1, padded_length=200, valid_rgb_frames=175, pose_frames_per_window=100, pose_sampling='rint(linspace(0,174,100)), as build_cache.sampled_block_indices', rgb_color='BGR to RGB; uint8 range 0..255; ResNet preprocessing inside frozen extractor; no /255', pose='equivalent_source_roi_to_640_v1; crop at native source resolution before resize to 640, matching actual all-data training cache', geometry=training_normalization_geometry(w, h), thresholds=manifest['recommended_web_thresholds'], event_policy=policy, source_selection='entire downloaded broadcast, no replay/closeup/cut removal', hardware=dict(platform=platform.platform(), processor=platform.processor(), logical_cpus=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total), load_s=load_s))
    detector._web_effective_fps = 25
    detector._web_window_duration = 7
    starts = sorted(set(range(0, total, 3000)) | {7500, 8250})
    pilot_block = starts.index(7500)
    order = [pilot_block] + [b for b in range(len(starts)) if b != pilot_block]
    cap = cv2.VideoCapture(str(video))
    for block in order:
        a = starts[block]
        z = starts[block + 1] if block + 1 < len(starts) else total
        path = HERE / f'results/features/{block:03d}_{a:08d}_{z:08d}.npz'
        if _resolve_model_artifact(path).exists():
            with _load_model_result_array(path) as d:
                assert d['rgb'].shape == (z - a, 2048) and int(d['a']) == a and (int(d['z']) == z)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - a) < 0.5
        detector._begin_auxiliary_sequence(z - a)
        old_rgb = HERE / 'legacy_adapter_partial/results/features' / path.name
        reusable_rgb = None
        if _resolve_model_artifact(old_rgb).exists():
            with _load_model_result_array(old_rgb) as d:
                assert int(d['a']) == a and int(d['z']) == z
                reusable_rgb = np.asarray(d['rgb'], np.float32)
        rgb = []
        batch = []
        stage_times = dict(decode_preprocess_s=0.0, pose_s=0.0, rgb_s=0.0, geometry_s=0.0)
        tblock = time.perf_counter()
        last = tblock
        for i in range(a, z):
            t = time.perf_counter()
            ok, raw = cap.read()
            assert ok, (i, total)
            crop = preprocess_raw_frame(raw, target_size=224)
            assert crop.shape == (224, 224, 3)
            pose_frame = normalize_yolo_source_roi(raw, yolo_size=640)
            assert pose_frame.shape == (640, 640, 3)
            stage_times['decode_preprocess_s'] += time.perf_counter() - t
            t = time.perf_counter()
            detector._geometry_frames.append(detector._detect_frame(detector.pose_detector, pose_frame, image_size=640, person_threshold=detector.POSE_PERSON_CONFIDENCE, max_people=detector.MAX_PEOPLE))
            stage_times['pose_s'] += time.perf_counter() - t
            if reusable_rgb is None:
                batch.append(detector._apply_frame_filter(crop)[:, :, [2, 1, 0]])
                if len(batch) >= 128 or i == z - 1:
                    t = time.perf_counter()
                    rgb.append(detector._predict_feature_batch(batch, 8))
                    batch.clear()
                    stage_times['rgb_s'] += time.perf_counter() - t
            if time.perf_counter() - last >= 25:
                progress('extracting', block=block, frame=i + 1, total_frames=total, source_s=(i + 1) / 25, block_elapsed_s=time.perf_counter() - tblock, available_ram_gb=psutil.virtual_memory().available / 1000000000.0)
                last = time.perf_counter()
        t = time.perf_counter()
        detector._finish_auxiliary_sequence()
        stage_times['geometry_s'] = time.perf_counter() - t
        values = reusable_rgb if reusable_rgb is not None else np.concatenate(rgb)
        assert values.shape == (z - a, 2048) and np.isfinite(values).all()
        with _resolve_model_artifact(path.with_suffix('.tmp')).open('wb') as f:
            np.savez(f, a=a, z=z, rgb=values, positions=detector._geometry_positions, poses=detector._geometry_poses, pose_mask=detector._geometry_mask)
        path.with_suffix('.tmp').replace(path)
        save(f'results/features/{block:03d}_timing.json', dict(a=a, z=z, elapsed_s=time.perf_counter() - tblock, rgb_reused_from_compatible_partial=reusable_rgb is not None, **stage_times))
        progress('checkpoint', block=block, frame=z, total_frames=total, source_s=z / 25, block_elapsed_s=time.perf_counter() - tblock)
        if block == pilot_block:
            pad = np.zeros((200, 2048), np.float32)
            pad[:175] = values[:175]
            mask = np.arange(200) < 175
            t = time.perf_counter()
            p = detector._predict_sequence_window(pad, mask, 0, 175)[0]
            assert np.isfinite(p).all() and np.isclose(p.sum(), 1, atol=1e-06)
            save('results/pilot.json', dict(source_start_s=300, seconds=30, frames=750, stage_times=stage_times, first_window_probabilities=p.tolist(), head_seconds=time.perf_counter() - t, projected_extraction_s=(time.perf_counter() - tblock) * total / 750, quality='functional pilot; no accuracy estimate'))
        del values, rgb
    cap.release()
    progress('loading_feature_checkpoints')
    paths = sorted(_model_artifact_glob(HERE / 'results/features', '*.npz', recursive=False))
    rgb = np.lib.format.open_memmap(HERE / 'results/rgb_full.npy', mode='w+', dtype=np.float32, shape=(total, 2048))
    positions = np.zeros((total, 31), np.float32)
    poses = np.zeros((total, 30), np.float32)
    pm = np.zeros(total, bool)
    cursor = 0
    for path in paths:
        with _load_model_result_array(path) as d:
            a, z = (int(d['a']), int(d['z']))
            assert a == cursor
            rgb[a:z] = d['rgb']
            positions[a:z] = d['positions']
            poses[a:z] = d['poses']
            pm[a:z] = d['pose_mask']
            cursor = z
    assert cursor == total
    rgb.flush()
    detector._geometry_positions = positions
    detector._geometry_poses = poses
    detector._geometry_mask = pm
    windows_path = HERE / 'results/windows.jsonl'
    windows = [json.loads(s) for s in _resolve_model_artifact(windows_path).read_text().splitlines()] if _resolve_model_artifact(windows_path).exists() else []
    t = time.perf_counter()
    last = t
    with _resolve_model_artifact(windows_path).open('a', encoding='utf-8') as f:
        for wi, a in enumerate(range(0, total - 175 + 1, 25)):
            if wi < len(windows):
                assert windows[wi]['start_time'] == a / 25
                continue
            z = a + 175
            pad = np.zeros((200, 2048), np.float32)
            pad[:175] = rgb[a:z]
            mask = np.arange(200) < 175
            p = detector._predict_sequence_window(pad, mask, a, z)[0]
            assert np.isfinite(p).all() and np.isclose(p.sum(), 1, atol=1e-06)
            row = dict(play_type=['line', 'other', 'scrum'][int(p.argmax())], start_time=a / 25, end_time=z / 25, duration=7.0, confidence=float(p.max()), frame_start=a, frame_end=z, probabilities={k: float(v) for k, v in zip(['line', 'other', 'scrum'], p)})
            windows.append(row)
            f.write(json.dumps(row) + '\n')
            f.flush()
            if time.perf_counter() - last >= 25:
                progress('heads', window=wi + 1, total_windows=(total - 175) // 25 + 1)
                last = time.perf_counter()
    heads_s = time.perf_counter() - t
    thresholds = manifest['recommended_web_thresholds']
    events = detector._post_process_event_detections(windows, float(thresholds['scrum']), duration, fps, {'line': float(thresholds['line'])})
    save('results/events.json', events)
    for name, rows in [('probabilities', windows), ('candidates', events)]:
        with _resolve_model_artifact(HERE / f'results/{name}.csv').open('w', newline='', encoding='utf-8') as f:
            out = csv.writer(f)
            out.writerow(['class', 'start_s', 'end_s', 'confidence', 'p_line', 'p_other', 'p_scrum'])
            for row in rows:
                out.writerow([row['play_type'], row['start_time'], row['end_time'], row['confidence'], *[row['probabilities'][k] for k in ['line', 'other', 'scrum']]])
    pd.DataFrame([dict(start_s=e['start_time'], end_s=e['end_time'], predicted=e['play_type'], manual_label='', review_notes='') for e in events]).to_csv(HERE / 'results/manual_candidate_review.csv', index=False)
    after = {str(p): sha(p) for p in refs}
    assert hashes == after
    timings = [json.loads(_resolve_model_artifact(p).read_text()) for p in _model_artifact_glob(HERE / 'results/features', '*_timing.json', recursive=False)]
    save('results/completion.json', dict(success=True, frames=total, duration_s=duration, windows=len(windows), candidates=len(events), counts={k: sum((e['play_type'] == k for e in events)) for k in ['line', 'scrum']}, extraction_s=sum((d['elapsed_s'] for d in timings)), stage_times={k: sum((d[k] for d in timings)) for k in ['decode_preprocess_s', 'pose_s', 'rgb_s', 'geometry_s']}, heads_s_this_process=heads_s, process_elapsed_s=time.perf_counter() - started, model_hashes_unchanged=hashes == after, errors=0, labels='none; qualitative review only', tail_not_covered_by_full_window_s=duration - windows[-1]['end_time']))
    progress('inference_complete', windows=len(windows), candidates=len(events))
    from make_report import main as report_main
    report_main()
    progress('complete', windows=len(windows), candidates=len(events), report=str(HERE / 'report.html'))
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        progress('failed', error=repr(e))
        traceback.print_exc()
        raise
