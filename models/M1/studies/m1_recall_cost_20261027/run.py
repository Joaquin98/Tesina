"""Train fixed M1 cost variants, then evaluate only immutable professional cache."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import os, sys, json, time, hashlib, argparse, gc, functools, traceback
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for k, v in {'CUDA_VISIBLE_DEVICES': '-1', 'TF_CPP_MIN_LOG_LEVEL': '2', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'TF_NUM_INTRAOP_THREADS': '8', 'TF_NUM_INTEROP_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'KERAS_HOME': str(HERE / 'runtime/keras'), 'YOLO_CONFIG_DIR': str(HERE / 'runtime/ultralytics'), 'TEMP': str(HERE / 'runtime/tmp'), 'TMP': str(HERE / 'runtime/tmp')}.items():
    os.environ[k] = v
for folder in ('runtime/keras', 'runtime/ultralytics', 'runtime/tmp', 'models', 'evaluation'):
    (HERE / folder).mkdir(parents=True, exist_ok=True)
START = time.perf_counter()
COSTS = (2, 4, 8)
BASE = ROOT / 'data/m1_ensemble_vs_all_development_20261012'
GAME = ROOT / 'data/professional_rugby_m1_game1_20261024'

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def save(p, v):
    p = HERE / p
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')
    t.replace(p)

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def progress(stage, **kw):
    value = dict(stage=stage, pid=os.getpid(), elapsed_s=time.perf_counter() - START, updated_unix=time.time(), **kw)
    save('progress.json', value)
    print(json.dumps(value), flush=True)

def initialize():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import psutil
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except psutil.Error:
        pass
    return tf

def train():
    tf = initialize()
    import numpy as np, pandas as pd, keras, joblib
    from sklearn.ensemble import RandomForestClassifier
    from models.common.comparison import models, data
    from models.M1.studies.m1_recall_cost_20261027.core import training_weights
    from models.M1.studies.m1_recall_cost_20261027.prepared_sequence import prepare, make_sequence
    assert read(HERE / 'input_equivalence.json')['inputs_bitwise_identical']
    data._load_resnet_path = functools.lru_cache(maxsize=256)(data._load_resnet_path.__wrapped__)
    frame = pd.read_csv(_resolve_model_artifact(BASE / 'inputs/all_training_examples.csv'))
    assert len(frame) == 8796 and (not frame.block_id.duplicated().any())
    assert (frame.domain == 'CH').sum() == 4082 and (frame.domain == 'CV').sum() == 4714
    assert set(frame[frame.domain == 'CV'].group_id) == {'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6'}
    assert not frame.block_id.astype(str).str.contains('20260830_112524_0b5e58|Argentina|VF_', case=False, regex=True).any()
    paths = set((Path(p) for col in ('resnet_path', 'yolo_path') for p in frame[col]))
    paths.update((data.PREPARED_FINAL / f'{block}.npz' for block in frame[frame.domain == 'CV'].block_id))
    stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    references = [BASE / 'models/dual.keras', BASE / 'models/geometry_rf.joblib', models.M0_PATH, BASE / 'inputs/all_training_examples.csv', _resolve_model_artifact(ROOT / 'models/common/comparison/models.py'), _resolve_model_artifact(ROOT / 'models/common/comparison/data.py'), _resolve_model_artifact(HERE / 'core.py'), _resolve_model_artifact(HERE / 'run.py'), _resolve_model_artifact(HERE / 'prepared_sequence.py')]
    hashes = {str(p): sha(p) for p in references}
    protocol = dict(model='M1-H2-D2', costs=list(COSTS), labels=['line', 'other', 'scrum'], training_examples=8796, seed=42, epochs=5, batch_size=16, optimizer='Adam1e-4', initialization='same original M0/H2 recipe for every cost; no fine-tuning baseline M1', cost_weighting='balanced training class weights times[cost,1,cost], normalized to mean1; both dual and forest', geometry_alpha=0.475, forest=dict(n_estimators=500, max_features='sqrt', random_state=42, n_jobs=8), extractors_executed=0, evaluation='Argentina-NZ20250816, full and every5; same class overlap>=1s one-to-one; no thresholds', user_priority='maximum sensitivity with all precision and review costs shown, no precision floor', selection='all three final epoch5 checkpoints published; no evaluation-based model/hyperparameter selection', vf='pending missing native cache; no new extraction authorized', independent_test_claim_allowed=False, source_hashes=hashes, cache_files=len(paths), rgb_lru_limit=256, training_inputs='file-backed read-only exact float32 original-reader outputs', input_equivalence='both domains and all labels verified; inherited shuffle/order and batch16')
    save('protocol.json', protocol)
    save('cache_stats_before.json', stats)
    progress('training_geometry_preparation', examples=len(frame))
    labels = data.labels_for(frame)
    tick = time.perf_counter()
    geom = data.compact_matrix(frame, ('geometry',))
    assert geom.shape == (8796, 549) and np.isfinite(geom).all()
    preparation_s = time.perf_counter() - tick
    selected = []
    initial_hash = None
    arrays = prepare(frame, data, HERE / 'prepared_inputs', progress)
    CachedSequence = make_sequence(models, arrays)
    for cost in COSTS:
        out = HERE / 'models' / f'cost_{cost}'
        out.mkdir(exist_ok=True)
        if _resolve_model_artifact(out / 'complete.json').exists():
            existing = read(out / 'complete.json')
            assert all((sha(Path(p)) == h for p, h in existing['model_hashes'].items()))
            selected.append(existing)
            continue
        keras.backend.clear_session()
        gc.collect()
        models.seed_everything(42)
        model = models.build_dual_model(models.M0_PATH, level='H2')
        fingerprint = hashlib.sha256(b''.join((w.tobytes() for w in model.get_weights()))).hexdigest()
        if initial_hash is None:
            initial_hash = fingerprint
        assert fingerprint == initial_hash
        weights = training_weights(labels, cost)
        cw = {i: float(w) for i, w in enumerate(weights)}
        model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        progress('training_forest', cost=cost, class_weights=cw)
        tick = time.perf_counter()
        forest = RandomForestClassifier(**protocol['forest'], class_weight=cw).fit(geom, labels)
        forest_s = time.perf_counter() - tick
        joblib.dump(forest, out / 'geometry_rf.joblib')
        del forest
        training_start = time.perf_counter()

        class Progress(keras.callbacks.Callback):

            def __init__(self):
                super().__init__()
                self.epoch = 0
                self.last = 0

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch + 1
                self.epoch_start = time.perf_counter()

            def on_train_batch_end(self, batch, logs=None):
                if time.perf_counter() - self.last >= 35:
                    progress('training_dual', cost=cost, epoch=self.epoch, epochs=5, batch=batch + 1, batches=550, training_elapsed_s=time.perf_counter() - training_start, logs={k: float(v) for k, v in (logs or {}).items()})
                    self.last = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                progress('epoch_complete', cost=cost, epoch=epoch + 1, epochs=5, epoch_s=time.perf_counter() - self.epoch_start, training_elapsed_s=time.perf_counter() - training_start, logs={k: float(v) for k, v in (logs or {}).items()})
        history = model.fit(CachedSequence(frame, batch_size=16, shuffle=True, seed=42, dual=True), epochs=5, class_weight=cw, callbacks=[Progress()], verbose=2)
        dual_s = time.perf_counter() - training_start
        model.save(out / 'dual.keras')
        save(f'models/cost_{cost}/history.json', history.history)
        record = dict(cost=cost, epochs=5, examples=8796, initial_weights_sha256=fingerprint, class_weights=cw, forest_s=forest_s, dual_s=dual_s, saved_unix=time.time(), model_hashes={str(p): sha(p) for p in (out / 'dual.keras', out / 'geometry_rf.joblib')})
        save(f'models/cost_{cost}/complete.json', record)
        selected.append(record)
        del model
        data._load_resnet_path.cache_clear()
        gc.collect()
    assert hashes == {str(p): sha(Path(p)) for p in hashes}
    assert stats == {str(p): [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in stats}
    assert len({s['initial_weights_sha256'] for s in selected}) == 1
    save('training_complete.json', dict(status='complete', models=selected, cache_preparation_s=preparation_s, elapsed_s=time.perf_counter() - START, source_models_and_caches_unchanged=True, all_models_saved_before_evaluation_unix=time.time()))
    progress('all_training_complete', models=3)

def evaluate():
    tf = initialize()
    import numpy as np, joblib
    from models.M1.studies.m1_recall_cost_20261027.core import LABELS, SelectedRGB, reconstruct_geometry, evaluate_events
    from src.web import keras_inference_engine as engine
    assert read(HERE / 'training_complete.json')['status'] == 'complete'
    report = read(ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json')
    for source in report['provenance']:
        assert sha(Path(source['path'])) == source['sha256']
    manifest_path = GAME / 'inputs/model_manifest.json'
    manifest = read(manifest_path)
    fold = manifest['folds'][0]
    namespace = dict(vars(engine), SELECTED_MANIFEST_PATH=manifest_path, EXPECTED_MEMBERS=1)
    adapter = _resolve_model_artifact(GAME / 'inputs/isolated_model_class.py')
    exec(compile(_resolve_model_artifact(adapter).read_text(encoding='utf-8'), str(adapter), 'exec'), namespace)
    detector = namespace['M1H2D2CrossFoldDetector'].__new__(namespace['M1H2D2CrossFoldDetector'])
    original = read(ROOT / 'data/results/result_20261025_argentina_allblacks_20250816_argmax.json')
    for k, v in original['event_post_processing'].items():
        setattr(detector, k, v)
    cache_paths = [GAME / 'results/rgb_full.npy'] + sorted(_model_artifact_glob(GAME / 'results/features', '*.npz', recursive=False))
    cache_stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in cache_paths}
    rgb = _load_model_result_array(cache_paths[0], mmap_mode='r')
    n = len(rgb)
    assert n == 154161
    pp = []
    qq = []
    mm = []
    cursor = 0
    for path in cache_paths[1:]:
        with _load_model_result_array(path) as d:
            assert int(d['a']) == cursor
            cursor = int(d['z'])
            pp.append(d['positions'])
            qq.append(d['poses'])
            mm.append(d['pose_mask'])
    assert cursor == n
    pos, pose, mask = (np.concatenate(pp), np.concatenate(qq), np.concatenate(mm))
    del pp, qq, mm
    starts = np.arange(0, n - 174, 25)
    gi = np.rint(np.linspace(0, 174, 100)).astype(np.int32)
    assert len(starts) == 6160

    def inputs(a, reader, li, ri, w):
        z = a + 175
        weights = w[a:z, None]
        pad = np.zeros((200, 2048), np.float32)
        pad[:175] = reader[li[a:z]] * (1 - weights) + reader[ri[a:z]] * weights
        ii = a + gi
        values = np.concatenate([detector._geometry_positions[ii], detector._geometry_poses[ii]], axis=1)
        valid_mask = detector._geometry_mask[ii].astype(bool)
        pi = np.concatenate([detector._align(values, 200), detector._align(valid_mask.astype(np.float32)[:, None], 200)], axis=1)
        valid = values[valid_mask] if valid_mask.any() else values
        delta = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, 61), np.float32)
        gv = np.concatenate([valid.mean(0), valid.std(0), valid.min(0), valid.max(0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), delta.mean(0), delta.std(0), np.abs(delta).mean(0)]).astype(np.float32)
        return (pad, pi, gv)

    def predictor(dual, forest):
        graph = tf.function(lambda x: dual(x, training=False), input_signature=[{'rgb': tf.TensorSpec([None, 200, 2048], tf.float32), 'pose': tf.TensorSpec([None, 200, 62], tf.float32), 'sequence_mask': tf.TensorSpec([None, 200], tf.bool)}], autograph=False)

        def predict(batch):
            dp = np.asarray(graph({'rgb': np.stack([b[0] for b in batch]), 'pose': np.stack([b[1] for b in batch]), 'sequence_mask': np.tile(np.arange(200) < 175, (len(batch), 1))}).numpy(), np.float64)
            gp = np.zeros_like(dp)
            gp[:, np.asarray(forest.classes_).astype(int)] = forest.predict_proba(np.stack([b[2] for b in batch]))
            f = np.exp((1 - 0.475) * np.log(np.clip(dp, 1e-07, 1)) + 0.475 * np.log(np.clip(gp, 1e-07, 1)))
            return (f / f.sum(1, keepdims=True)).astype(np.float32)
        return predict
    controls = [0, 20, 266, 486, 1260, 2695, 3948, 4942, 5536, 6000]
    olddual = tf.keras.models.load_model(_resolve_model_artifact(BASE / 'models/dual.keras'), compile=False)
    oldforest = joblib.load(_resolve_model_artifact(BASE / 'models/geometry_rf.joblib'))
    oldforest.n_jobs = 1
    predict = predictor(olddual, oldforest)
    detector._geometry_positions, detector._geometry_poses, detector._geometry_mask = (pos, pose, mask)
    dense = np.arange(n)
    zero = np.zeros(n, np.float32)
    reader = SelectedRGB(rgb, np.ones(n, bool))
    actual = predict([inputs(i * 25, reader, dense, dense, zero) for i in controls])
    expected = np.array([[original['window_predictions'][i]['probabilities'][k] for k in LABELS] for i in controls])
    error = float(np.abs(actual - expected).max())
    assert error < 1e-05
    save('baseline_validation.json', dict(passed=True, windows=controls, max_probability_error=error))
    del olddual, oldforest, predict
    gc.collect()
    records = []
    for cost in COSTS:
        model_dir = HERE / 'models' / f'cost_{cost}'
        trained = read(model_dir / 'complete.json')
        assert all((sha(Path(p)) == h for p, h in trained['model_hashes'].items()))
        dual = tf.keras.models.load_model(_resolve_model_artifact(model_dir / 'dual.keras'), compile=False)
        forest = joblib.load(_resolve_model_artifact(model_dir / 'geometry_rf.joblib'))
        forest.n_jobs = 1
        predict = predictor(dual, forest)
        for stride in (1, 5):
            out = HERE / 'evaluation' / f'cost_{cost}_stride_{stride}'
            out.mkdir(exist_ok=True)
            if _resolve_model_artifact(out / 'metrics.json').exists():
                records.append(read(out / 'metrics.json'))
                continue
            tick = time.perf_counter()
            p, q, m, idx, li, ri, w = reconstruct_geometry(pos, pose, mask, stride)
            detector._geometry_positions, detector._geometry_poses, detector._geometry_mask = (p, q, m)
            keep = np.zeros(n, bool)
            keep[idx] = True
            reader = SelectedRGB(rgb, keep)
            wp = out / 'windows.jsonl'
            rows = [json.loads(line) for line in _resolve_model_artifact(wp).read_text().splitlines()] if _resolve_model_artifact(wp).exists() else []
            resumed = len(rows)
            assert resumed <= 6160 and all((r['frame_start'] == int(starts[i]) for i, r in enumerate(rows)))
            last = 0
            with _resolve_model_artifact(wp).open('a', encoding='utf-8') as f:
                for wi in range(resumed, len(starts), 8):
                    batch = [inputs(a, reader, li, ri, w) for a in starts[wi:wi + 8]]
                    probs = predict(batch)
                    assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1, atol=1e-06)
                    for j, prob in enumerate(probs):
                        a = int(starts[wi + j])
                        row = dict(play_type=LABELS[int(prob.argmax())], start_time=a / 25, end_time=(a + 175) / 25, duration=7.0, confidence=float(prob.max()), frame_start=a, frame_end=a + 175, probabilities=dict(zip(LABELS, map(float, prob))))
                        rows.append(row)
                        f.write(json.dumps(row) + '\n')
                    f.flush()
                    if time.perf_counter() - last > 30:
                        progress('cached_evaluation', cost=cost, stride=stride, windows=len(rows), total_windows=6160)
                        last = time.perf_counter()
            events = detector._post_process_event_detections(rows, 0.0, 6166.44, 25, {'line': 0.0})
            events = [e for e in events if e['end_time'] > 188.2]
            value = evaluate_events(report['reference_segments'], events)
            value.update(cost=cost, stride=stride, reference_id=report['revision_id'], reference_sha256=report['labels_sha256'], windows=6160, selected_frames=len(idx), unique_rgb_selected_reads=len(reader.seen), extractors_executed=0, resumed_windows=resumed, cached_evaluation_seconds=time.perf_counter() - tick, model_hashes=trained['model_hashes'])
            save(f'evaluation/cost_{cost}_stride_{stride}/events.json', events)
            save(f'evaluation/cost_{cost}_stride_{stride}/metrics.json', value)
            records.append(value)
            progress('configuration_complete', cost=cost, stride=stride, metrics=value['metrics']['total'])
        del dual, forest, predict
        tf.keras.backend.clear_session()
        gc.collect()
    assert cache_stats == {str(p): [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in cache_stats}
    for source in report['provenance']:
        assert sha(Path(source['path'])) == source['sha256']
    save('evaluation_complete.json', dict(status='complete', records=records, source_models_and_caches_unchanged=True, extractors_executed=0, elapsed_s=time.perf_counter() - START, reference_id=report['revision_id']))
    progress('evaluation_complete', configurations=len(records))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('train', 'evaluate'))
    args = parser.parse_args()
    try:
        if args.mode == 'train':
            train()
        else:
            evaluate()
    except Exception:
        progress('error', traceback=traceback.format_exc())
        raise
