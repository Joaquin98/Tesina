"""Frozen cost2 heads on existing professional caches; no extraction or training."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import os, sys, json, time, hashlib, traceback, gc
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
sys.dont_write_bytecode = True
for k, v in dict(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', TF_NUM_INTRAOP_THREADS='8', TF_NUM_INTEROP_THREADS='1', PYTHONDONTWRITEBYTECODE='1').items():
    os.environ[k] = v
START = time.perf_counter()
E = ROOT / 'data'
M = E / 'm1_recall_cost_20261027/models/cost_2'
OLD = E / 'm1_ensemble_vs_all_development_20261012/models'
G = E / 'professional_rugby_m1_single_20261022'
R = E / 'professional_rugby_rwc2023_every25_20261025'
JOBS = [dict(date='20250823', stride=s, n=142011, duration=5680.44, kickoff=184.2, cache=G, rid='20261025_argentina_allblacks_20250823_argmax_' + ('full' if s == 1 else 'every25')) for s in (1, 25)]
JOBS.append(dict(date='20231020', stride=25, n=144422, duration=5776.88, kickoff=6.0, cache=R, rid='20261025_argentina_allblacks_20231020_argmax_every25'))

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')
    t.replace(p)

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def progress(stage, **v):
    d = dict(stage=stage, elapsed_s=time.perf_counter() - START, pid=os.getpid(), updated_unix=time.time(), **v)
    save(HERE / 'progress.json', d)
    print(json.dumps(d), flush=True)

def load_cache(job):
    import numpy as np
    paths = sorted(_model_artifact_glob(job['cache'] / 'results/features', '*.npz', recursive=False))
    stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    pp = []
    qq = []
    mm = []
    ii = []
    rr = []
    cursor = 0
    for path in paths:
        with _load_model_result_array(path) as d:
            pp.append(d['positions'])
            qq.append(d['poses'])
            if job['date'] == '20250823':
                assert int(d['a']) == cursor
                cursor = int(d['z'])
                mm.append(d['pose_mask'])
            else:
                ii.append(d['indices'])
                rr.append(d['rgb'])
                mm.append(d['mask'])
    p, q, m = (np.concatenate(pp), np.concatenate(qq), np.concatenate(mm))
    if job['date'] == '20250823':
        path = G / 'results/rgb_full.npy'
        stats[str(path)] = [path.stat().st_size, path.stat().st_mtime_ns]
        rgb = _load_model_result_array(path, mmap_mode='r')
        assert cursor == len(rgb) == job['n']
        indices = np.unique(np.r_[np.arange(0, job['n'], job['stride']), job['n'] - 1]).astype(np.int32)
        rgb, p, q, m = (rgb[indices], p[indices], q[indices], m[indices])
    else:
        rgb = np.concatenate(rr)
        indices = np.concatenate(ii)
        assert np.array_equal(indices, np.unique(np.r_[np.arange(0, job['n'], 25), job['n'] - 1]))
    assert rgb.shape == (len(indices), 2048) and p.shape == (len(indices), 31) and (q.shape == (len(indices), 30))
    assert all((np.isfinite(v).all() for v in (rgb, p, q)))
    return (dict(indices=indices, rgb=rgb, positions=p, poses=q, mask=m), stats)

def input_window(s, a, align):
    import numpy as np
    grid = np.arange(a, a + 175)
    idx = s['indices']
    right = np.searchsorted(idx, grid).clip(0, len(idx) - 1)
    left = (right - 1).clip(0)
    w = ((grid - idx[left]) / np.maximum(idx[right] - idx[left], 1)).astype(np.float32)[:, None]
    pad = np.zeros((200, 2048), np.float32)
    pad[:175] = s['rgb'][left] * (1 - w) + s['rgb'][right] * w
    pos = s['positions'][left] * (1 - w) + s['positions'][right] * w
    pose = s['poses'][left] * (1 - w) + s['poses'][right] * w
    mask = s['mask'][np.where(grid - idx[left] <= idx[right] - grid, left, right)]
    chosen = np.rint(np.linspace(0, 174, 100)).astype(np.int32)
    v = np.concatenate([pos[chosen], pose[chosen]], axis=1)
    m = mask[chosen]
    pi = np.concatenate([align(v, 200), align(m.astype(np.float32)[:, None], 200)], axis=1)
    valid = v[m] if m.any() else v
    delta = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, 61), np.float32)
    gv = np.concatenate([valid.mean(0), valid.std(0), valid.min(0), valid.max(0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), delta.mean(0), delta.std(0), np.abs(delta).mean(0)]).astype(np.float32)
    assert pi.shape == (200, 62) and gv.shape == (549,) and all((np.isfinite(x).all() for x in (pad, pi, gv)))
    return (pad, pi, gv)

def main():
    import tensorflow as tf, numpy as np, joblib, psutil
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except psutil.Error:
        pass
    from core import LABELS, evaluate_events
    from review_coverage import review_coverage
    from src.web import keras_inference_engine as engine
    ns = dict(vars(engine))
    adapter = _resolve_model_artifact(G / 'inputs/isolated_model_class.py')
    exec(compile(_resolve_model_artifact(adapter).read_text(encoding='utf-8'), str(adapter), 'exec'), ns)
    detector = ns['M1H2D2CrossFoldDetector'].__new__(ns['M1H2D2CrossFoldDetector'])
    trained = read(M / 'complete.json')
    assert all((sha(Path(p)) == h for p, h in trained['model_hashes'].items()))
    sources = [_resolve_model_artifact(HERE / 'run.py'), adapter, _resolve_model_artifact(ROOT / 'models/M0/web_adapter.py'), _resolve_model_artifact(E / 'm1_recall_cost_20261027/core.py'), _resolve_model_artifact(E / 'm1_recall_cost_20261027/review_coverage.py'), M / 'dual.keras', M / 'geometry_rf.joblib', OLD / 'dual.keras', OLD / 'geometry_rf.joblib']
    for j in JOBS:
        sources.extend([ROOT / f"data/results/result_{j['rid']}.json", E / f"sampling_audit_20261026/reconciliation/{j['date'][:4]}-{j['date'][4:6]}-{j['date'][6:]}_reference_v3.json"])
    hashes = {str(p): sha(p) for p in sources}
    save(HERE / 'protocol.json', dict(jobs=[{k: str(v) if isinstance(v, Path) else v for k, v in j.items()} for j in JOBS], cost=2, model='saved M1-H2-D2 cost2 from previous trial; no fitting or threshold changes', source_hashes=hashes, event_matching='same class, overlap>=1s, maximum one-to-one cardinality then overlap', reference_scope='automated diagnostic v3 partial inventories; not human labels or certified full-match recall', reference_gaps='2 unlocalized lineouts in each game versus competition PDF; counts/definitions also differ between suppliers', precision_scope='temporal matching precision against fixed inventory, not historical visually judged precision; unmatched alerts are not all confirmed false positives', extraction=False, video_opened=False, new_training=False))

    def predictor(folder):
        dual = tf.keras.models.load_model(_resolve_model_artifact(folder / 'dual.keras'), compile=False)
        forest = joblib.load(_resolve_model_artifact(folder / 'geometry_rf.joblib'))
        forest.n_jobs = 1
        graph = tf.function(lambda x: dual(x, training=False), input_signature=[dict(rgb=tf.TensorSpec([None, 200, 2048], tf.float32), pose=tf.TensorSpec([None, 200, 62], tf.float32), sequence_mask=tf.TensorSpec([None, 200], tf.bool))], autograph=False)

        def predict(batch):
            dp = np.asarray(graph(dict(rgb=np.stack([x[0] for x in batch]), pose=np.stack([x[1] for x in batch]), sequence_mask=np.tile(np.arange(200) < 175, (len(batch), 1)))).numpy(), np.float64)
            gp = np.zeros_like(dp)
            gp[:, forest.classes_.astype(int)] = forest.predict_proba(np.stack([x[2] for x in batch]))
            f = np.exp(0.525 * np.log(np.clip(dp, 1e-07, 1)) + 0.475 * np.log(np.clip(gp, 1e-07, 1)))
            return (f / f.sum(1, keepdims=True)).astype(np.float32)
        return predict
    old_predict = predictor(OLD)
    cost_predict = predictor(M)
    records = []
    all_stats = {}
    for j in JOBS:
        key = f"{j['date']}_stride{j['stride']}"
        out = HERE / key
        out.mkdir(exist_ok=True)
        if _resolve_model_artifact(out / 'metrics.json').exists():
            records.append(read(out / 'metrics.json'))
            continue
        progress('loading_cache', job=key)
        tick = time.perf_counter()
        s, stats = load_cache(j)
        all_stats.update(stats)
        original = read(ROOT / f"data/results/result_{j['rid']}.json")
        windows = original['window_predictions']
        starts = np.arange(0, j['n'] - 174, 25)
        assert len(windows) == len(starts)
        assert all((r['frame_start'] == int(a) for r, a in zip(windows, starts)))
        for k, v in original['event_post_processing'].items():
            setattr(detector, k, v)
        control = sorted(set([0, len(windows) - 1] + np.linspace(0, len(windows) - 1, 30, dtype=int).tolist()))
        actual = []
        for off in range(0, len(control), 8):
            actual.extend(old_predict([input_window(s, int(starts[i]), detector._align) for i in control[off:off + 8]]))
        expected = np.array([[windows[i]['probabilities'][k] for k in LABELS] for i in control])
        actual = np.asarray(actual)
        err = float(np.abs(expected - actual).max())
        assert err < 2e-05 and np.array_equal(actual.argmax(1), expected.argmax(1)), (key, err)

        def post(rows):
            return [e for e in detector._post_process_event_detections(rows, 0.0, j['duration'], 25, {'line': 0.0}) if e['end_time'] > j['kickoff']]
        baseline = post(windows)
        signature = lambda es: [(e['play_type'], e['start_time'], e['end_time']) for e in es]
        assert signature(baseline) == signature(original['detections']), (key, 'Postprocessing drift')
        save(out / 'baseline_validation.json', dict(status='PASS', windows=control, max_probability_error=err, argmax_identical=True, all_original_events_reproduced=True))
        load_s = time.perf_counter() - tick
        tick = time.perf_counter()
        wp = out / 'windows.jsonl'
        rows = [json.loads(x) for x in _resolve_model_artifact(wp).read_text(encoding='utf-8').splitlines()] if _resolve_model_artifact(wp).exists() else []
        resumed = len(rows)
        assert resumed <= len(starts) and all((r['frame_start'] == int(starts[i]) for i, r in enumerate(rows)))
        last = 0
        with _resolve_model_artifact(wp).open('a', encoding='utf-8') as f:
            for wi in range(resumed, len(starts), 8):
                prob = cost_predict([input_window(s, int(a), detector._align) for a in starts[wi:wi + 8]])
                assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-06)
                for off, p in enumerate(prob):
                    a = int(starts[wi + off])
                    row = dict(play_type=LABELS[int(p.argmax())], start_time=a / 25, end_time=(a + 175) / 25, duration=7.0, confidence=float(p.max()), frame_start=a, frame_end=a + 175, probabilities=dict(zip(LABELS, map(float, p))))
                    rows.append(row)
                    f.write(json.dumps(row) + '\n')
                f.flush()
                if time.perf_counter() - last > 30:
                    progress('cached_inference', job=key, windows=len(rows), total=len(starts))
                    last = time.perf_counter()
        inference_s = time.perf_counter() - tick
        events = post(rows)
        assert len(rows) == len(starts)
        truth = read(E / f"sampling_audit_20261026/reconciliation/{j['date'][:4]}-{j['date'][4:6]}-{j['date'][6:]}_reference_v3.json")['events']
        value = dict(job=key, date=j['date'], stride=j['stride'], duration_s=j['duration'], baseline_result_id=j['rid'], windows=len(rows), selected_frames=len(s['indices']), reference_segments=truth, original=evaluate_events(truth, baseline), cost2=evaluate_events(truth, events), baseline_validation=read(out / 'baseline_validation.json'), model_hashes=trained['model_hashes'], timing=dict(cache_load_and_baseline_control_s=load_s, cached_cost2_inference_s=inference_s, resumed_windows=resumed), extractors_executed=0, video_opened=False, new_training=False)
        for mode, es in [('original', baseline), ('cost2', events)]:
            value[mode]['review_coverage'] = review_coverage(truth, es)
        save(out / 'events.json', events)
        save(out / 'metrics.json', value)
        records.append(value)
        assert stats == {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in stats}
        progress('configuration_complete', job=key, original=value['original']['metrics']['total'], cost2=value['cost2']['metrics']['total'])
        del s, windows, original, rows
        gc.collect()
    assert hashes == {p: sha(Path(p)) for p in hashes}
    assert all_stats == {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in all_stats}
    save(HERE / 'completion.json', dict(status='complete', records=records, elapsed_s=time.perf_counter() - START, source_hashes_unchanged=True, source_cache_stats_unchanged=True, cache_files=len(all_stats), extractors_executed=0, video_opened=False, new_training=False))
    progress('complete', configurations=len(records))
if __name__ == '__main__':
    try:
        main()
    except Exception:
        progress('error', traceback=traceback.format_exc())
        raise
