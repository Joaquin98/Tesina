"""Three full professional caches; fixed truth and event policy per game."""
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
import sys, time, gc, importlib.util, traceback
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(HERE)), str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
from models.M1.studies.m1_line_costs_20261028.run import read, save, sha, initialize, progress, COSTS

def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m
old = module('previous_native_run', _resolve_model_artifact(ROOT / 'models/common/studies/m1_cost2_other_professionals_20261027/run.py'))
JOBS = [dict(date='20250816', n=154161, duration=6166.44, kickoff=188.2, cache=ROOT / 'data/professional_rugby_m1_game1_20261024', mask='pose_mask', baseline_id='20261027_argentina_allblacks_20250816_m1_cost2_stride1', original_id='20261025_argentina_allblacks_20250816_argmax', reference=ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json', reference_kind='human'), dict(date='20250823', n=142011, duration=5680.44, kickoff=184.2, cache=ROOT / 'data/professional_rugby_m1_single_20261022', mask='pose_mask', baseline_id='20261027_argentina_allblacks_20250823_m1_cost2_stride1', original_id='20261025_argentina_allblacks_20250823_argmax_full', reference=ROOT / 'data/sampling_audit_20261026/reconciliation/2025-08-23_reference_v3.json', reference_kind='partial automated diagnostic reference'), dict(date='20231020', n=144422, duration=5776.88, kickoff=6.0, cache=ROOT / 'data/m1_cost2_rwc2023_full_20261028', mask='mask', baseline_id='20261028_argentina_allblacks_20231020_m1_cost2_stride1', original_id='20261028_argentina_allblacks_20231020_m1_original_stride1', reference=ROOT / 'data/sampling_audit_20261026/reconciliation/2023-10-20_reference_v3.json', reference_kind='partial automated diagnostic reference')]

def load(job):
    import numpy as np
    folder = job['cache']
    features = folder / 'features' if job['date'] == '20231020' else folder / 'results/features'
    rgb_path = folder / 'rgb_full.npy' if job['date'] == '20231020' else folder / 'results/rgb_full.npy'
    rgb = _load_model_result_array(rgb_path, mmap_mode='r')
    assert rgb.shape == (job['n'], 2048)
    p = []
    q = []
    m = []
    cursor = 0
    paths = [rgb_path] + sorted(_model_artifact_glob(features, '*.npz', recursive=False))
    stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
    for path in paths[1:]:
        with _load_model_result_array(path) as d:
            assert int(d['a']) == cursor
            cursor = int(d['z'])
            p.append(d['positions'])
            q.append(d['poses'])
            m.append(d[job['mask']])
    assert cursor == job['n']
    return (dict(indices=np.arange(job['n'], dtype=np.int32), rgb=rgb, positions=np.concatenate(p), poses=np.concatenate(q), mask=np.concatenate(m)), stats)

def main():
    tf = initialize()
    import numpy as np, joblib
    from core import LABELS, evaluate_events
    from review_coverage import review_coverage
    from src.web import keras_inference_engine as engine
    trained = read(HERE / 'training_complete.json')
    assert trained['status'] == 'complete'
    for r in trained['models']:
        assert all((sha(p) == h for p, h in r['model_hashes'].items()))
    assert max((x['saved_unix'] for x in trained['models'])) <= trained['all_models_saved_before_evaluation_unix']

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
    ns = dict(vars(engine))
    adapter = _resolve_model_artifact(ROOT / 'data/professional_rugby_m1_single_20261022/inputs/isolated_model_class.py')
    exec(compile(_resolve_model_artifact(adapter).read_text(encoding='utf-8'), str(adapter), 'exec'), ns)
    records = []
    proofs = []
    for job in JOBS:
        date = job['date']
        progress('loading_cache', date=date)
        s, stats = load(job)
        baseline = read(ROOT / f"data/results/result_{job['baseline_id']}.json")
        ref = read(job['reference'])
        ref_sha = sha(job['reference'])
        truth = ref['reference_segments'] if job['reference_kind'] == 'human' else ref['events']
        if job['reference_kind'] == 'human':
            for source in ref['provenance']:
                assert sha(source['path']) == source['sha256']
        policy = ns['M1H2D2CrossFoldDetector'].__new__(ns['M1H2D2CrossFoldDetector'])
        for k, v in baseline['event_post_processing'].items():
            setattr(policy, k, v)
        starts = np.arange(0, job['n'] - 174, 25)
        assert len(starts) == len(baseline['window_predictions'])
        pred = predictor(ROOT / 'data/m1_recall_cost_20261027/models/cost_2')
        ids = np.unique(np.linspace(0, len(starts) - 1, 30, dtype=int))
        actual = []
        for off in range(0, len(ids), 8):
            actual.extend(pred([old.input_window(s, int(starts[i]), policy._align) for i in ids[off:off + 8]]))
        expected = np.array([[baseline['window_predictions'][i]['probabilities'][k] for k in LABELS] for i in ids])
        err = float(np.abs(np.array(actual) - expected).max())
        assert err < 2e-05 and np.array_equal(np.array(actual).argmax(1), expected.argmax(1))

        def post(rows):
            return [e for e in policy._post_process_event_detections(rows, 0.0, job['duration'], 25, {'line': 0.0}) if e['end_time'] > job['kickoff']]
        assert [(x['play_type'], x['start_time'], x['end_time']) for x in post(baseline['window_predictions'])] == [(x['play_type'], x['start_time'], x['end_time']) for x in baseline['detections']]
        proof = dict(date=date, status='PASS', control_windows=len(ids), max_probability_error=err, all_baseline_events_reproduced=True)
        proofs.append(proof)
        save(HERE / f'evaluation/{date}/baseline_validation.json', proof)
        del pred
        tf.keras.backend.clear_session()
        gc.collect()
        for cost in COSTS:
            folder = HERE / f'models/line{cost}_scrum2'
            trained_model = read(folder / 'complete.json')
            out = HERE / f'evaluation/{date}/line{cost}_scrum2'
            out.mkdir(parents=True, exist_ok=True)
            if _resolve_model_artifact(out / 'metrics.json').exists():
                records.append(read(out / 'metrics.json'))
                continue
            pred = predictor(folder)
            wp = out / 'windows.jsonl'
            rows = [__import__('json').loads(x) for x in _resolve_model_artifact(wp).read_text(encoding='utf-8').splitlines()] if _resolve_model_artifact(wp).exists() else []
            assert len(rows) <= len(starts) and all((r['frame_start'] == int(starts[i]) for i, r in enumerate(rows)))
            resumed = len(rows)
            tick = time.perf_counter()
            last = 0
            with _resolve_model_artifact(wp).open('a', encoding='utf-8') as f:
                for wi in range(resumed, len(starts), 8):
                    probabilities = pred([old.input_window(s, int(a), policy._align) for a in starts[wi:wi + 8]])
                    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1, atol=1e-06)
                    for off, v in enumerate(probabilities):
                        a = int(starts[wi + off])
                        r = dict(play_type=LABELS[int(v.argmax())], start_time=a / 25, end_time=(a + 175) / 25, duration=7.0, confidence=float(v.max()), frame_start=a, frame_end=a + 175, probabilities=dict(zip(LABELS, map(float, v))))
                        rows.append(r)
                        f.write(__import__('json').dumps(r) + '\n')
                    f.flush()
                    if time.perf_counter() - last > 25:
                        progress('evaluation', date=date, line_cost=cost, windows=len(rows), total=len(starts))
                        last = time.perf_counter()
            events = post(rows)
            value = evaluate_events(truth, events)
            value.update(date=date, line_cost=cost, scrum_cost=2, reference_kind=job['reference_kind'], reference_sha256=ref_sha, reference_id=ref.get('revision_id'), labels_sha256=ref.get('labels_sha256'), reference_segments=truth, windows=len(rows), stride=1, cached_inference_s=time.perf_counter() - tick, resumed_windows=resumed, extractors_executed=0, model_hashes=trained_model['model_hashes'])
            value['review_coverage'] = review_coverage(truth, events)
            save(out / 'events.json', events)
            save(out / 'metrics.json', value)
            records.append(value)
            progress('configuration_complete', date=date, line_cost=cost, metrics=value['metrics']['total'])
            del pred
            tf.keras.backend.clear_session()
            gc.collect()
        assert stats == {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in stats}
        assert sha(job['reference']) == ref_sha
        del s
        gc.collect()
    save(HERE / 'evaluation_complete.json', dict(status='complete', records=records, baseline_controls=proofs, extractors_executed=0, source_cache_stats_unchanged=True))
if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(HERE / 'evaluation_error.json', dict(error=traceback.format_exc()))
        raise
