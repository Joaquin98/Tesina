"""Evaluate both frozen heads on the same completed native cache."""
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
import os, sys, json, time, importlib.util, gc
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(HERE)), str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
from extract import read, save, sha, anchors, N, BLOCK, OLD

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
prior = module('other_games_run', _resolve_model_artifact(ROOT / 'models/common/studies/m1_cost2_other_professionals_20261027/run.py'))

def main():
    import numpy as np, tensorflow as tf, joblib
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from core import evaluate_events, LABELS
    from review_coverage import review_coverage
    from src.web import keras_inference_engine as engine
    tick = time.perf_counter()
    sparse = anchors()
    paths = sorted(_model_artifact_glob(HERE / 'features', '*.npz', recursive=False))
    assert len(paths) == len(range(0, N, BLOCK))
    rgb = np.lib.format.open_memmap(HERE / 'rgb_full.npy', mode='w+', dtype=np.float32, shape=(N, 2048))
    p = np.empty((N, 31), np.float32)
    q = np.empty((N, 30), np.float32)
    mask = np.empty(N, bool)
    cursor = 0
    reused = 0
    fresh = 0
    stage = {}
    for path in paths:
        with _load_model_result_array(path) as d:
            a, z = (int(d['a']), int(d['z']))
            assert a == cursor
            rgb[a:z] = d['rgb']
            p[a:z] = d['positions']
            q[a:z] = d['poses']
            mask[a:z] = d['mask']
            cursor = z
        t = read(path.with_suffix('.json'))
        reused += t['reused_anchors']
        fresh += t['new_observations']
        for k in ('rgb_s', 'yolo_s', 'geometry_s', 'decode_s'):
            stage[k] = stage.get(k, 0.0) + t[k]
    assert cursor == N and reused == 5778 and (fresh == 138644)
    rgb.flush()
    full = dict(indices=np.arange(N, dtype=np.int32), rgb=rgb, positions=p, poses=q, mask=mask)
    for k in ('rgb', 'positions', 'poses', 'mask'):
        assert np.array_equal(full[k][sparse['indices']], sparse[k]), ('Anchor changed', k)
    protocol = read(HERE / 'protocol.json')
    for path, h in protocol['immutable_hashes'].items():
        assert sha(path) == h
    save(HERE / 'cache_verification.json', dict(status='PASS', native_observations=N, reused_anchors=reused, new_observations=fresh, all_anchors_bitwise_identical=True, extractor_stage_cpu_wall_s_sum=stage, cache_assembly_s=time.perf_counter() - tick))
    ns = dict(vars(engine))
    adapter = _resolve_model_artifact(ROOT / 'data/professional_rugby_m1_single_20261022/inputs/isolated_model_class.py')
    exec(compile(_resolve_model_artifact(adapter).read_text(encoding='utf-8'), str(adapter), 'exec'), ns)
    policy = ns['M1H2D2CrossFoldDetector'].__new__(ns['M1H2D2CrossFoldDetector'])
    config = read(OLD / 'inputs/config.json')
    for k, v in config['event_policy'].items():
        setattr(policy, k, v)
    starts = np.arange(0, N - 174, 25)
    duration = N / 25
    truth = read(ROOT / 'data/sampling_audit_20261026/reconciliation/2023-10-20_reference_v3.json')['events']
    controls = {mode: [json.loads(x) for x in _resolve_model_artifact(path).read_text(encoding='utf-8').splitlines()] for mode, path in (('original', OLD / 'results/windows.jsonl'), ('cost2', ROOT / 'data/m1_cost2_other_professionals_20261027/20231020_stride25/windows.jsonl'))}
    out = []
    for mode, folder in [('original', prior.OLD), ('cost2', prior.M)]:
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
        ids = np.unique(np.linspace(0, len(starts) - 1, 30, dtype=int))
        actual = []
        for off in range(0, len(ids), 8):
            actual.extend(predict([prior.input_window(sparse, int(starts[i]), policy._align) for i in ids[off:off + 8]]))
        expected = np.array([[controls[mode][i]['probabilities'][k] for k in LABELS] for i in ids])
        err = float(np.abs(np.array(actual) - expected).max())
        assert err < 2e-05 and np.array_equal(np.array(actual).argmax(1), expected.argmax(1))
        save(HERE / f'{mode}_sparse_control.json', dict(status='PASS', windows=len(ids), max_probability_error=err))
        dest = HERE / mode
        dest.mkdir(exist_ok=True)
        wp = dest / 'windows.jsonl'
        rows = [json.loads(x) for x in _resolve_model_artifact(wp).read_text(encoding='utf-8').splitlines()] if _resolve_model_artifact(wp).exists() else []
        assert len(rows) <= len(starts) and all((r['frame_start'] == int(starts[i]) for i, r in enumerate(rows)))
        resumed = len(rows)
        begin = time.perf_counter()
        last = 0
        with _resolve_model_artifact(wp).open('a', encoding='utf-8') as f:
            for wi in range(resumed, len(starts), 8):
                prob = predict([prior.input_window(full, int(a), policy._align) for a in starts[wi:wi + 8]])
                assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-06)
                for off, v in enumerate(prob):
                    a = int(starts[wi + off])
                    r = dict(play_type=LABELS[int(v.argmax())], start_time=a / 25, end_time=(a + 175) / 25, duration=7.0, confidence=float(v.max()), frame_start=a, frame_end=a + 175, probabilities=dict(zip(LABELS, map(float, v))))
                    rows.append(r)
                    f.write(json.dumps(r) + '\n')
                f.flush()
                if time.perf_counter() - last > 25:
                    save(HERE / 'progress_evaluate.json', dict(stage=mode, windows=len(rows), total=len(starts), elapsed_s=time.perf_counter() - begin))
                    last = time.perf_counter()
        events = [e for e in policy._post_process_event_detections(rows, 0.0, duration, 25, {'line': 0.0}) if e['end_time'] > 6.0]
        metrics = evaluate_events(truth, events)
        metrics['review_coverage'] = review_coverage(truth, events)
        value = dict(mode=mode, metrics=metrics, windows=len(rows), cached_inference_s=time.perf_counter() - begin, resumed_windows=resumed, model_hashes={str(folder / k): sha(folder / k) for k in ('dual.keras', 'geometry_rf.joblib')})
        save(dest / 'events.json', events)
        save(dest / 'metrics.json', value)
        out.append(value)
        del dual, forest
        tf.keras.backend.clear_session()
        gc.collect()
    for path, h in protocol['immutable_hashes'].items():
        assert sha(path) == h
    save(HERE / 'evaluation_complete.json', dict(status='complete', records=out, reference_segments=truth, duration_s=duration, stride=1, new_training=False, immutable_hashes_unchanged=True))
    print(json.dumps(dict(status='complete', results=[dict(mode=x['mode'], metrics=x['metrics']['metrics']['total'], review=x['metrics']['review']) for x in out])), flush=True)
if __name__ == '__main__':
    main()
