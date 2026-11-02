"""Cached professional windows and VF central25, with fixed references/policies."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.M1.studies.m0_m1_cost_sampling_20261029.common import *
import argparse, importlib.util, gc, copy, traceback
import numpy as np
from models.M1.studies.m0_m1_cost_sampling_20261029.inputs import RGBSampler
from core import LABELS, evaluate_events
from review_coverage import review_coverage

def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m
old = module('m0_trial_old_input', _resolve_model_artifact(ROOT / 'models/common/studies/m1_cost2_other_professionals_20261027/run.py'))
check = module('m0_trial_independent', _resolve_model_artifact(ROOT / 'models/M1/studies/m1_recall_cost_20261027/publish.py')).independent_check

def verified(truth, events, duration):
    v = evaluate_events(truth, events)
    pr = check(truth, events, v['metrics'], duration)
    for k in ['unique_flagged_s', 'flagged_outside_true_segments_s']:
        assert abs(pr[k] - v['review'][k]) < 1e-06
    v['independent_check'] = pr
    v['review_coverage'] = review_coverage(truth, events)
    return v

def freeze_reference(job):
    ref = read(job['reference'])
    truth = ref['reference_segments'] if job['kind'] == 'human' else ref['events']
    if job['kind'] == 'human':
        for p in ref['provenance']:
            assert sha(p['path']) == p['sha256']
    return (ref, truth)

def policy_for(job):
    from models.M2.predictor import RGBYoloHighresQuantileFusionDetector as Policy
    d = read(ROOT / f"data/results/result_{job['baseline']}.json")
    policy = Policy.__new__(Policy)
    for k, v in d['event_post_processing'].items():
        setattr(policy, k, v)
    return (policy, d)

def post(rows, job, policy):
    return [e for e in policy._post_process_event_detections(rows, 0.0, job['duration'], 25, {'line': 0.0}) if e['end_time'] > job['kickoff']]

def predictor(family, key, tf):
    if family == 'm0':
        path = model_path(key)
        info = read(HERE / f'models/{key}/complete.json') if _resolve_model_artifact(HERE / f'models/{key}/complete.json').exists() else read(ROOT / 'data/m0_cost2_professional_human_20261027/training_complete.json')
        assert sha(path) == info['model_sha256']
        net = tf.keras.models.load_model(_resolve_model_artifact(path), compile=False)
        graph = tf.function(lambda a, b: net([a, b], training=False), input_signature=[tf.TensorSpec([None, 200, 2048], tf.float32), tf.TensorSpec([None, 200], tf.bool)], autograph=False)
        return (lambda batch: graph(np.stack([b[0] for b in batch]), np.stack([b[2] for b in batch])).numpy(), {str(path): sha(path)})
    import joblib
    dp, rp = m1_paths(key)
    net = tf.keras.models.load_model(_resolve_model_artifact(dp), compile=False)
    rf = joblib.load(_resolve_model_artifact(rp))
    rf.n_jobs = 1
    graph = tf.function(lambda x: net(x, training=False), input_signature=[dict(rgb=tf.TensorSpec([None, 200, 2048], tf.float32), pose=tf.TensorSpec([None, 200, 62], tf.float32), sequence_mask=tf.TensorSpec([None, 200], tf.bool))], autograph=False)

    def pred(batch):
        a = graph(dict(rgb=np.stack([b[0] for b in batch]), pose=np.stack([b[1] for b in batch]), sequence_mask=np.stack([b[2] for b in batch]))).numpy().astype(np.float64)
        b = np.zeros_like(a)
        b[:, rf.classes_.astype(int)] = rf.predict_proba(np.stack([b[3] for b in batch]))
        f = np.exp(0.525 * np.log(np.clip(a, 1e-07, 1)) + 0.475 * np.log(np.clip(b, 1e-07, 1)))
        return (f / f.sum(1, keepdims=True)).astype(np.float32)
    return (pred, {str(dp): sha(dp), str(rp): sha(rp)})

def existing_m1(job, key, stride):
    date = job['date']
    if stride == 1:
        v = next((v for v in read(ROOT / 'data/m1_line_only_costs_20261028/results.json')['variants'] if v['date'] == date and v['key'] == key))
        return ROOT / f"data/results/result_{v['viewer_url'].split('result_id=')[1]}.json"
    if key in ('original', 'x2'):
        ref = read(ROOT / 'data/m1_recall_cost_20261027/results.json')
        k = 'every5' if key == 'original' else 'cost_2_stride_5'
        v = next((v for v in ref['variants'] if v['key'] == k))
        return ROOT / f"data/results/result_{v['viewer_url'].split('result_id=')[1]}.json"
    return None

def load_geometry(job, stride):
    folder = job['rgb'].parent / 'features'
    pp = []
    qq = []
    mm = []
    cursor = 0
    stats = {}
    for path in sorted(_model_artifact_glob(folder, '*.npz', recursive=False)):
        stats[str(path)] = stat(path)
        with _load_model_result_array(path) as z:
            assert int(z['a']) == cursor
            cursor = int(z['z'])
            pp.append(z['positions'])
            qq.append(z['poses'])
            mm.append(z['mask' if job['date'] == '20231020' else 'pose_mask'])
    assert cursor == job['n']
    idx = np.unique(np.r_[np.arange(0, job['n'], stride), job['n'] - 1]).astype(np.int32)
    return (dict(indices=idx, rgb=_load_model_result_array(job['rgb'], mmap_mode='r')[idx], positions=np.concatenate(pp)[idx], poses=np.concatenate(qq)[idx], mask=np.concatenate(mm)[idx]), stats)

def align(v, n):
    x = np.linspace(0, 1, len(v), dtype=np.float32)
    t = np.linspace(0, 1, n, dtype=np.float32)
    return np.stack([np.interp(t, x, v[:, j]) for j in range(v.shape[1])], axis=1).astype(np.float32)

def professional(tf, key, family, only_date=None):
    for job in JOBS:
        if only_date and job['date'] != only_date:
            continue
        ref, truth = freeze_reference(job)
        ref_hash = sha(job['reference'])
        policy, baseline = policy_for(job)
        for stride in [1, 5] if job['kind'] == 'human' else [1]:
            out = HERE / f"evaluation/{job['date']}/{family}_{key}_stride{stride}"
            out.mkdir(parents=True, exist_ok=True)
            if _resolve_model_artifact(out / 'metrics.json').exists():
                continue
            reuse = existing_m1(job, key, stride) if family == 'm1' else None
            if reuse:
                original = read(reuse)
                events = original['detections']
                rows = original['window_predictions']
                value = verified(truth, events, job['duration'])
                save(out / 'events.json', events)
                with _resolve_model_artifact(out / 'windows.jsonl').open('w', encoding='utf-8') as f:
                    for row in rows:
                        f.write(json.dumps(row) + '\n')
                value.update(family=family, key=key, stride=stride, date=job['date'], reference_kind=job['kind'], reference_file_sha256=ref_hash, reused=True, source_result=str(reuse), source_result_sha256=sha(reuse), cached_inference_s=None)
                save(out / 'metrics.json', value)
                continue
            stats = {str(job['rgb']): stat(job['rgb'])}
            rgb = _load_model_result_array(job['rgb'], mmap_mode='r')
            assert rgb.shape == (job['n'], 2048)
            sampler = RGBSampler(rgb, stride)
            s, gs = load_geometry(job, stride) if family == 'm1' else (None, {})
            stats.update(gs)

            def item(a):
                if family == 'm0':
                    x, m = sampler.window(a)
                    return (x, None, m, None)
                x, pose, geom = old.input_window(s, a, align)
                return (x, pose, np.arange(200) < 175, geom)
            pred, hashes = predictor(family, key, tf)
            starts = np.arange(0, job['n'] - 174, 25)
            assert len(starts) == len(baseline['window_predictions'])
            if family == 'm0' and key == 'x2' and (job['kind'] == 'human'):
                prior = ROOT / f'data/m0_cost2_professional_human_20261027/stride_{stride}'
                prior_rows = [json.loads(x) for x in _resolve_model_artifact(prior / 'windows.jsonl').read_text(encoding='utf-8').splitlines()]
                ids = np.unique(np.linspace(0, len(starts) - 1, 30, dtype=int))
                actual = []
                for off in range(0, len(ids), 8):
                    actual.extend(pred([item(int(starts[i])) for i in ids[off:off + 8]]))
                expected = np.array([[prior_rows[i]['probabilities'][k] for k in LABELS] for i in ids])
                err = float(np.abs(actual - expected).max())
                assert err < 2e-05 and np.array_equal(np.asarray(actual).argmax(1), expected.argmax(1))
                assert post(prior_rows, job, policy) == read(prior / 'events.json')
                events = read(prior / 'events.json')
                value = verified(truth, events, job['duration'])
                save(out / 'events.json', events)
                with _resolve_model_artifact(out / 'windows.jsonl').open('w', encoding='utf-8') as f:
                    for row in prior_rows:
                        f.write(json.dumps(row) + '\n')
                value.update(family=family, key=key, stride=stride, date=job['date'], reference_kind=job['kind'], reference_file_sha256=ref_hash, reused=True, baseline_control=dict(windows=len(ids), max_probability_error=err, all_events_reproduced=True), model_hashes=hashes, cached_inference_s=None)
                save(out / 'metrics.json', value)
                del pred, rgb, sampler, s
                tf.keras.backend.clear_session()
                gc.collect()
                continue
            wp = out / 'windows.jsonl'
            rows = [json.loads(x) for x in _resolve_model_artifact(wp).read_text(encoding='utf-8').splitlines()] if _resolve_model_artifact(wp).exists() else []
            assert all((r['frame_start'] == int(starts[i]) for i, r in enumerate(rows)))
            tick = time.perf_counter()
            last = 0
            resumed = len(rows)
            with _resolve_model_artifact(wp).open('a', encoding='utf-8') as f:
                for wi in range(len(rows), len(starts), 8):
                    prob = pred([item(int(a)) for a in starts[wi:wi + 8]])
                    assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-06)
                    for off, v in enumerate(prob):
                        a = int(starts[wi + off])
                        row = dict(play_type=LABELS[int(v.argmax())], start_time=a / 25, end_time=(a + 175) / 25, duration=7.0, confidence=float(v.max()), frame_start=a, frame_end=a + 175, probabilities=dict(zip(LABELS, map(float, v))))
                        rows.append(row)
                        f.write(json.dumps(row) + '\n')
                    f.flush()
                    if time.perf_counter() - last > 30:
                        save(HERE / 'evaluation_progress.json', dict(stage='professional', date=job['date'], key=key, family=family, stride=stride, windows=len(rows), total=len(starts), updated_unix=time.time(), pid=os.getpid()))
                        last = time.perf_counter()
            events = post(rows, job, policy)
            value = verified(truth, events, job['duration'])
            value.update(family=family, key=key, stride=stride, date=job['date'], reference_kind=job['kind'], reference_file_sha256=ref_hash, reused=False, cached_inference_s=time.perf_counter() - tick, resumed_windows=resumed, model_hashes=hashes, selected_frames=len(sampler.indices))
            assert hashes == {p: sha(p) for p in hashes} and stats == {p: stat(p) for p in stats} and (sha(job['reference']) == ref_hash)
            save(out / 'events.json', events)
            save(out / 'metrics.json', value)
            print(json.dumps(dict(date=job['date'], family=family, key=key, stride=stride, metrics=value['metrics']['total'])), flush=True)
            del pred, rgb, sampler, s
            tf.keras.backend.clear_session()
            gc.collect()

def vf_batches():
    source = ROOT / 'data/vf_full_sampling_20261025'
    rows = read(source / 'timelines.json')['rows']
    assert len(rows) == 688
    batches = []
    stats = {}
    for section in ['S01', 'S02', 'S03', 'S04']:
        chunks = []
        for path in sorted(_model_artifact_glob(source / 'features', section + '_*.npz', recursive=False)):
            stats[str(path)] = stat(path)
            with _load_model_result_array(path) as z:
                assert str(z['protocol']) == 'VF_center224_upscaled640_every25_grid25fps_v1'
                chunks.append({k: z[k].copy() for k in ['indices', 'rgb', 'positions', 'poses', 'mask']})
        sparse = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        idx = sparse['indices']
        np.testing.assert_array_equal(idx, np.unique(np.r_[np.arange(0, 30030, 25), 30029]))
        for row in (r for r in rows if r['section'] == section):
            a = int(round(row['start_s'] * 25))
            z = min(30030, int(round(row['end_s'] * 25)))
            grid = np.arange(a, z)
            right = np.searchsorted(idx, grid).clip(0, len(idx) - 1)
            left = (right - 1).clip(0)
            w = ((grid - idx[left]) / np.maximum(idx[right] - idx[left], 1)).astype(np.float32)
            rgb = np.zeros((200, 2048), np.float32)
            rgb[:len(grid)] = sparse['rgb'][left] * (1 - w[:, None]) + sparse['rgb'][right] * w[:, None]
            pos = sparse['positions'][left] * (1 - w[:, None]) + sparse['positions'][right] * w[:, None]
            pose = sparse['poses'][left] * (1 - w[:, None]) + sparse['poses'][right] * w[:, None]
            mask = sparse['mask'][np.where(grid - idx[left] <= idx[right] - grid, left, right)]
            chosen = np.linspace(0, len(grid) - 1, 100, dtype=np.int32) if len(grid) > 100 else np.arange(len(grid))
            v = np.concatenate([pos[chosen], pose[chosen]], 1)
            m = mask[chosen]
            pose200 = np.concatenate([align(v, 200), align(m.astype(np.float32)[:, None], 200)], 1)
            valid = v[m] if m.any() else v
            delta = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, 61), np.float32)
            geom = np.concatenate([valid.mean(0), valid.std(0), valid.min(0), valid.max(0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), delta.mean(0), delta.std(0), np.abs(delta).mean(0)]).astype(np.float32)
            batches.append((rgb, pose200, np.arange(200) < len(grid), geom))
    assert len(batches) == 688
    return (rows, batches, stats)

def group_vf(rows):
    events = []
    for r in rows:
        if r['predicted'] not in ('line', 'scrum'):
            continue
        if events and events[-1]['play_type'] == r['predicted'] and (events[-1]['section'] == r['section']) and (abs(events[-1]['end_time'] - r['global_start_s']) < 1e-06):
            events[-1]['end_time'] = r['global_end_s']
            events[-1]['duration'] = events[-1]['end_time'] - events[-1]['start_time']
        else:
            events.append(dict(play_type=r['predicted'], section=r['section'], start_time=r['global_start_s'], end_time=r['global_end_s'], duration=r['global_end_s'] - r['global_start_s'], confidence=max(r['probabilities'].values())))
    return events

def vf_evaluate(tf, key, family):
    out = HERE / f'evaluation/VF/{family}_{key}_stride25'
    if _resolve_model_artifact(out / 'metrics.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    rows, batches, stats = vf_batches()
    truth = read(ROOT / 'data/m1_recall_cost_20261027/vf25_results.json')['reference_segments']
    pred, hashes = predictor(family, key, tf)
    tick = time.perf_counter()
    prob = []
    for off in range(0, 688, 8):
        prob.extend(pred(batches[off:off + 8]))
    prob = np.asarray(prob)
    assert prob.shape == (688, 3) and np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-06)
    control = None
    if family == 'm1' and key in ['original', 'x2']:
        oldvf = read(ROOT / 'data/m1_recall_cost_20261027/vf25_results.json')
        v = oldvf['variants'][1 if key == 'original' else 2]
        expected = np.array([[r['probabilities'][l] for l in LABELS] for r in v['rows']]) if 'rows' in v else np.array([[r['p_' + l] for l in LABELS] for r in read(ROOT / 'data/vf_full_sampling_20261025/results.json')['rows']])
        control = float(np.abs(prob - expected).max())
        assert control < 2e-05 and np.array_equal(prob.argmax(1), expected.argmax(1))
    predicted = [dict(r, predicted=LABELS[int(v.argmax())], probabilities=dict(zip(LABELS, map(float, v)))) for r, v in zip(rows, prob)]
    events = group_vf(predicted)
    value = verified(truth, events, 4804.8)
    value.update(family=family, key=key, stride=25, date='VF', reference_kind='human VF blocks', reused=False, cached_inference_s=time.perf_counter() - tick, model_hashes=hashes, baseline_max_probability_error=control, blocks=688, reference_segments=truth)
    save(out / 'rows.json', predicted)
    save(out / 'events.json', events)
    save(out / 'metrics.json', value)
    assert hashes == {p: sha(p) for p in hashes} and stats == {p: stat(p) for p in stats}
    print(json.dumps(dict(date='VF', family=family, key=key, metrics=value['metrics']['total'])), flush=True)
    del pred, batches
    tf.keras.backend.clear_session()
    gc.collect()

def main(key, family, date=None):
    import tensorflow as tf, psutil
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    try:
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except psutil.Error:
        pass
    if date != 'VF':
        professional(tf, key, family, date)
    if date in [None, 'VF']:
        vf_evaluate(tf, key, family)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--key', required=True, choices=[c[0] for c in CONFIGS])
    p.add_argument('--family', required=True, choices=['m0', 'm1'])
    p.add_argument('--date', choices=['20250816', '20250823', '20231020', 'VF'])
    a = p.parse_args()
    try:
        main(a.key, a.family, a.date)
    except Exception:
        save(HERE / 'evaluation_error.json', dict(key=a.key, family=a.family, date=a.date, error=traceback.format_exc()))
        raise
