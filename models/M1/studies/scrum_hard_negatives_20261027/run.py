"""Cached exploratory hard-negative fine tuning, fixed before target evaluation."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import os, sys, json, time, hashlib, gc, argparse, traceback, functools
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(HERE)), str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
sys.dont_write_bytecode = True
THREADS = 4
for k, v in dict(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS), TF_NUM_INTRAOP_THREADS=str(THREADS), TF_NUM_INTEROP_THREADS='1', PYTHONDONTWRITEBYTECODE='1').items():
    os.environ[k] = v
START = time.perf_counter()
FACTORS = (1, 1.5, 2)

def read(p):
    return json.loads(_resolve_model_artifact(Path(p)).read_text(encoding='utf-8-sig'))

def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')
    t.replace(p)

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(p)).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def progress(folder, stage, **kw):
    d = dict(stage=stage, pid=os.getpid(), elapsed_s=time.perf_counter() - START, updated=time.time(), **kw)
    save(folder / 'progress.json', d)
    print(json.dumps(d), flush=True)

def initialize():
    import tensorflow as tf, psutil
    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except psutil.Error:
        pass
    return tf

def graph_for(net, family, tf):
    specs = {'features': tf.TensorSpec([None, 200, 2048], tf.float32), 'mask': tf.TensorSpec([None, 200], tf.bool)} if family == 'm0' else {'rgb': tf.TensorSpec([None, 200, 2048], tf.float32), 'pose': tf.TensorSpec([None, 200, 62], tf.float32), 'sequence_mask': tf.TensorSpec([None, 200], tf.bool)}
    return tf.function(lambda x: net(x, training=False), input_signature=[specs], autograph=False)

def fuse(dp, gp):
    import numpy as np
    f = np.exp(0.525 * np.log(np.clip(dp, 1e-07, 1)) + 0.475 * np.log(np.clip(gp, 1e-07, 1)))
    return f / f.sum(1, keepdims=True)

def train(family):
    tf = initialize()
    import numpy as np, pandas as pd, keras, joblib
    from sklearn.ensemble import RandomForestClassifier
    from models.common.comparison import models, data
    from core import training_weights
    from models.M1.studies.scrum_hard_negatives_20261027.weights import hard_negative_weights
    folder = HERE / family
    folder.mkdir(exist_ok=True)
    prior = ROOT / 'data/m1_recall_cost_20261027'
    prepared = prior / 'prepared_inputs'
    csv = ROOT / 'data/m1_ensemble_vs_all_development_20261012/inputs/all_training_examples.csv'
    frame = pd.read_csv(_resolve_model_artifact(csv))
    assert len(frame) == 8796 and (not frame.block_id.duplicated().any())
    assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == read(prepared / 'manifest.json')['frame_sha256']
    arrays = {k: _load_model_result_array(prepared / (k + '.npy'), mmap_mode='r') for k in ('rgb', 'pose', 'mask', 'labels')}
    y = data.labels_for(frame)
    np.testing.assert_array_equal(y, arrays['labels'])
    cw = training_weights(y, 2)
    initial = ROOT / 'data/m0_cost2_professional_human_20261027/models/m0_cost2.keras' if family == 'm0' else prior / 'models/cost_2/dual.keras'
    original_forest = prior / 'models/cost_2/geometry_rf.joblib'
    refs = [initial, csv, prepared / 'manifest.json', Path(__file__), _resolve_model_artifact(HERE / 'weights.py'), _resolve_model_artifact(ROOT / 'models/common/comparison/models.py'), _resolve_model_artifact(prior / 'core.py')]
    if family == 'm1':
        refs.append(original_forest)
    hashes = {str(p): sha(p) for p in refs}
    stats = {str(prepared / (k + '.npy')): [(prepared / (k + '.npy')).stat().st_size, (prepared / (k + '.npy')).stat().st_mtime_ns] for k in arrays}

    def inputs(idx):
        if family == 'm0':
            return {'features': arrays['rgb'][idx], 'mask': arrays['mask'][idx]}
        return {'rgb': arrays['rgb'][idx], 'pose': arrays['pose'][idx], 'sequence_mask': arrays['mask'][idx]}
    save(folder / 'protocol.json', dict(family=family, source_checkpoint=str(initial), source_hashes=hashes, cache_stats=stats, examples=8796, seed=42, epochs=1, batch_size=16, optimizer='Adam1e-4 reset for all arms', factors=FACTORS, class_weights=cw.tolist(), hard_negatives='true other and own frozen x2 argmax scrum, mined only on D2 training; fixed before target evaluation', sample_weights='class_weight[y]*factor if hard, normalized to empirical mean1', forest_weights='same effective sample weights as neural net', evaluation='fixed human fd96032b00fa, full/every5; common M1 raw argmax event policy; sameclass overlap>=1s one-to-one', new_extraction=False, selection='all one-epoch arms and continuation controls, no target-based hyperparameter selection'))
    geom = None
    if family == 'm1':
        data._load_resnet_path = functools.lru_cache(maxsize=64)(data._load_resnet_path.__wrapped__)
        progress(folder, 'reading_cached_training_geometry')
        geom = data.compact_matrix(frame, ('geometry',))
        assert geom.shape == (8796, 549) and np.isfinite(geom).all()
        np.save(folder / 'training_geometry.npy', geom)
    if not _resolve_model_artifact(folder / 'mining.json').exists():
        progress(folder, 'mining_training_other', examples=int((y == 1).sum()))
        net = keras.models.load_model(_resolve_model_artifact(initial), compile=False)
        g = graph_for(net, family, tf)
        forest = joblib.load(_resolve_model_artifact(original_forest)) if family == 'm1' else None
        if forest is not None:
            forest.n_jobs = 1
        indices = np.flatnonzero(y == 1)
        prob = np.zeros((len(y), 3), np.float32)
        pred = y.copy()
        last = 0
        tick = time.perf_counter()
        for offset in range(0, len(indices), 16):
            ids = indices[offset:offset + 16]
            dp = np.asarray(g(inputs(ids)).numpy(), np.float64)
            if forest is not None:
                gp = np.zeros_like(dp)
                gp[:, np.asarray(forest.classes_, int)] = forest.predict_proba(geom[ids])
                dp = fuse(dp, gp)
            prob[ids] = dp
            pred[ids] = dp.argmax(1)
            if time.perf_counter() - last > 30:
                progress(folder, 'mining_training_other', completed=min(offset + 16, len(indices)), total=len(indices))
                last = time.perf_counter()
        _, hard = hard_negative_weights(y, pred, cw, 1)
        np.save(folder / 'mining_other_probabilities.npy', prob[indices])
        np.save(folder / 'mining_other_indices.npy', indices)
        np.save(folder / 'hard_indices.npy', np.flatnonzero(hard))
        np.save(folder / 'training_selection_predictions.npy', pred)
        save(folder / 'mining.json', dict(hard_negatives=int(hard.sum()), training_other=len(indices), seconds=time.perf_counter() - tick, selected_blocks=frame.loc[hard, ['block_id', 'domain', 'group_id', 'label']].to_dict('records'), prediction_note='Only other labels were inferred; non-other entries retain labels and are never selected', initial_model_sha256=sha(initial), selection_sha256=sha(folder / 'hard_indices.npy')))
        del net, g, forest
        keras.backend.clear_session()
        gc.collect()
    pred = _load_model_result_array(folder / 'training_selection_predictions.npy')
    mining = read(folder / 'mining.json')
    assert mining['hard_negatives'] > 0
    np.testing.assert_array_equal(np.flatnonzero((y == 1) & (pred == 2)), _load_model_result_array(folder / 'hard_indices.npy'))
    records = []
    for factor in FACTORS:
        out = folder / 'models' / f'cost_{factor}'
        out.mkdir(parents=True, exist_ok=True)
        if _resolve_model_artifact(out / 'complete.json').exists():
            r = read(out / 'complete.json')
            assert all((sha(p) == h for p, h in r['model_hashes'].items()))
            records.append(r)
            continue
        keras.backend.clear_session()
        gc.collect()
        models.seed_everything(42)
        net = keras.models.load_model(_resolve_model_artifact(initial), compile=False)
        if family == 'm0':
            models._configure_m0_trainable(net, 'H2')
        initial_fingerprint = hashlib.sha256(b''.join((w.tobytes() for w in net.get_weights()))).hexdigest()
        weights, hard = hard_negative_weights(y, pred, cw, factor)
        np.save(out / 'sample_weights.npy', weights)

        class Sequence(models.ResNetCacheSequence):

            def __getitem__(self, b):
                idx = self.indices[b * self.batch_size:(b + 1) * self.batch_size]
                return (inputs(idx), y[idx], weights[idx].astype(np.float32))
        seq = Sequence(frame, batch_size=16, shuffle=True, seed=42, dual=family == 'm1')
        inp, labels, sw = seq[0]
        np.testing.assert_array_equal(sw, weights[seq.indices[:16]].astype(np.float32))
        forest_s = 0.0
        if family == 'm1':
            progress(folder, 'training_forest', factor=factor, hard_negatives=int(hard.sum()))
            tick = time.perf_counter()
            forest = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=42, n_jobs=4).fit(geom, y, sample_weight=weights)
            forest_s = time.perf_counter() - tick
            joblib.dump(forest, out / 'geometry_rf.joblib')
            del forest
        net.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        tick = time.perf_counter()

        class Progress(keras.callbacks.Callback):

            def __init__(self):
                super().__init__()
                self.last = 0

            def on_train_batch_end(self, batch, logs=None):
                if time.perf_counter() - self.last > 30:
                    progress(folder, 'training', factor=factor, epoch=1, batch=batch + 1, batches=len(seq), training_s=time.perf_counter() - tick)
                    self.last = time.perf_counter()
        h = net.fit(seq, epochs=1, callbacks=[Progress()], verbose=2)
        elapsed = time.perf_counter() - tick
        path = out / ('dual.keras' if family == 'm1' else 'm0.keras')
        net.save(path)
        paths = [path] + ([out / 'geometry_rf.joblib'] if family == 'm1' else [])
        r = dict(family=family, factor=factor, epochs=1, seed=42, initial_weights_sha256=initial_fingerprint, hard_negatives=int(hard.sum()), class_weights=cw.tolist(), sample_weight_mean=float(weights.mean()), training_s=elapsed, forest_s=forest_s, history=h.history, model_hashes={str(p): sha(p) for p in paths}, saved_unix=time.time())
        save(out / 'complete.json', r)
        records.append(r)
        progress(folder, 'arm_training_complete', factor=factor, training_s=elapsed)
        del net, seq
        keras.backend.clear_session()
        gc.collect()
    assert len({r['initial_weights_sha256'] for r in records}) == 1
    assert hashes == {p: sha(p) for p in hashes}
    assert stats == {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in stats}
    save(folder / 'training_complete.json', dict(status='complete', models=records, mining=mining, source_hashes=hashes, source_cache_stats_unchanged=True, finished_unix=time.time()))
    progress(folder, 'all_training_complete')

def evaluate_m1():
    source = _resolve_model_artifact(ROOT / 'models/M1/studies/m1_recall_cost_20261027/run.py')
    text = _resolve_model_artifact(source).read_text(encoding='utf-8-sig')
    text = text.replace('HERE=experiment_directory(__file__);ROOT=HERE.parents[1]', f"ROOT=Path({str(ROOT)!r});HERE=ROOT/'data/scrum_hard_negatives_20261027/m1'")
    text = text.replace('COSTS=(2,4,8)', 'COSTS=(1,1.5,2)')
    text = text.replace("'8'", "'4'").replace('set_intra_op_parallelism_threads(8)', 'set_intra_op_parallelism_threads(4)')
    ns = dict(__file__=str(source), __name__='isolated_cached_m1_evaluation')
    exec(compile(text, str(source), 'exec'), ns)
    ns['evaluate']()

def evaluate_m0():
    tf = initialize()
    import numpy as np, keras
    from core import SelectedRGB, evaluate_events
    from review_coverage import review_coverage
    from models.M2.predictor import RGBYoloHighresQuantileFusionDetector as Policy
    folder = HERE / 'm0'
    trained = read(folder / 'training_complete.json')
    assert trained['status'] == 'complete'
    rgbpath = ROOT / 'data/professional_rugby_m1_game1_20261024/results/rgb_full.npy'
    rgb = _load_model_result_array(rgbpath, mmap_mode='r')
    n = len(rgb)
    assert n == 154161
    stats = [rgbpath.stat().st_size, rgbpath.stat().st_mtime_ns]
    ref = read(ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json')
    for s in ref['provenance']:
        assert sha(s['path']) == s['sha256']
    baseline = read(ROOT / 'data/results/result_20261025_argentina_allblacks_20250816_argmax.json')
    policy = Policy.__new__(Policy)
    for k, v in baseline['event_post_processing'].items():
        setattr(policy, k, v)
    starts = np.arange(0, n - 174, 25)
    assert len(starts) == 6160
    records = []
    for factor in FACTORS:
        record = read(folder / 'models' / f'cost_{factor}' / 'complete.json')
        assert all((sha(p) == h for p, h in record['model_hashes'].items()))
        net = keras.models.load_model(_resolve_model_artifact(folder / 'models' / f'cost_{factor}' / 'm0.keras'), compile=False)
        graph = graph_for(net, 'm0', tf)
        for stride in (1, 5):
            out = folder / 'evaluation' / f'cost_{factor}_stride_{stride}'
            out.mkdir(parents=True, exist_ok=True)
            if _resolve_model_artifact(out / 'metrics.json').exists():
                records.append(read(out / 'metrics.json'))
                continue
            tick = time.perf_counter()
            idx = np.unique(np.r_[np.arange(0, n, stride), n - 1])
            keep = np.zeros(n, bool)
            keep[idx] = True
            reader = SelectedRGB(rgb, keep)
            grid = np.arange(n)
            rr = np.searchsorted(idx, grid).clip(0, len(idx) - 1)
            ll = (rr - 1).clip(0)
            li, ri = (idx[ll], idx[rr])
            w = ((grid - li) / np.maximum(ri - li, 1)).astype(np.float32)

            def inputs(a):
                x = np.zeros((200, 2048), np.float32)
                x[:175] = reader[li[a:a + 175]] * (1 - w[a:a + 175, None]) + reader[ri[a:a + 175]] * w[a:a + 175, None]
                return x
            if stride == 1:
                np.testing.assert_array_equal(inputs(0)[:175], rgb[:175])
            checks = np.stack([inputs(int(starts[i])) for i in (0, 1260, 4942, 6000)])
            masks = np.tile(np.arange(200) < 175, (4, 1))
            control = {'features': checks, 'mask': masks}
            err = float(np.abs(net(control, training=False).numpy() - graph(control).numpy()).max())
            assert err < 1e-06
            wp = out / 'windows.jsonl'
            rows = [json.loads(x) for x in _resolve_model_artifact(wp).read_text().splitlines()] if _resolve_model_artifact(wp).exists() else []
            assert all((r['frame_start'] == int(starts[i]) for i, r in enumerate(rows)))
            last = 0
            with _resolve_model_artifact(wp).open('a', encoding='utf-8') as f:
                for wi in range(len(rows), len(starts), 8):
                    batch = np.stack([inputs(int(a)) for a in starts[wi:wi + 8]])
                    prob = graph({'features': batch, 'mask': np.tile(np.arange(200) < 175, (len(batch), 1))}).numpy()
                    assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-06)
                    for j, p in enumerate(prob):
                        a = int(starts[wi + j])
                        r = dict(play_type=['line', 'other', 'scrum'][int(p.argmax())], start_time=a / 25, end_time=(a + 175) / 25, duration=7.0, confidence=float(p.max()), frame_start=a, frame_end=a + 175, probabilities=dict(zip(['line', 'other', 'scrum'], map(float, p))))
                        rows.append(r)
                        f.write(json.dumps(r) + '\n')
                    f.flush()
                    if time.perf_counter() - last > 30:
                        progress(folder, 'cached_evaluation', factor=factor, stride=stride, windows=len(rows), total_windows=6160)
                        last = time.perf_counter()
            events = [e for e in policy._post_process_event_detections(rows, 0.0, 6166.44, 25, {'line': 0.0}) if e['end_time'] > 188.2]
            v = evaluate_events(ref['reference_segments'], events)
            v.update(factor=factor, stride=stride, windows=6160, selected_frames=len(idx), cached_evaluation_seconds=time.perf_counter() - tick, control_max_error=err, reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], review_coverage=review_coverage(ref['reference_segments'], events), model_hashes=record['model_hashes'])
            save(out / 'events.json', events)
            save(out / 'metrics.json', v)
            records.append(v)
            progress(folder, 'configuration_complete', factor=factor, stride=stride, metrics=v['metrics']['total'])
        del net, graph
        keras.backend.clear_session()
        gc.collect()
    assert stats == [rgbpath.stat().st_size, rgbpath.stat().st_mtime_ns]
    for s in ref['provenance']:
        assert sha(s['path']) == s['sha256']
    save(folder / 'evaluation_complete.json', dict(status='complete', records=records, extractors_executed=0, source_models_and_caches_unchanged=True))
    progress(folder, 'evaluation_complete', configurations=len(records))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'evaluate'])
    p.add_argument('family', choices=['m0', 'm1'])
    a = p.parse_args()
    folder = HERE / a.family
    folder.mkdir(exist_ok=True)
    try:
        if a.mode == 'train':
            train(a.family)
        elif a.family == 'm0':
            evaluate_m0()
        else:
            evaluate_m1()
    except Exception:
        progress(folder, 'error', traceback=traceback.format_exc())
        raise
