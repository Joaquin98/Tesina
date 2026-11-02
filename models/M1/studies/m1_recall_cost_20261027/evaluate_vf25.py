"""Evaluate the same trained costs on the complete existing native stride25 VF cache."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import os, sys, json, time, hashlib, gc
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
for k, v in {'CUDA_VISIBLE_DEVICES': '-1', 'TF_CPP_MIN_LOG_LEVEL': '2', 'OMP_NUM_THREADS': '4', 'TF_NUM_INTRAOP_THREADS': '4', 'TF_NUM_INTEROP_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}.items():
    os.environ[k] = v
import numpy as np
from models.M1.studies.m1_recall_cost_20261027.core import LABELS, evaluate_events
from models.M1.studies.m1_recall_cost_20261027.review_coverage import review_coverage

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')

def progress(stage, **values):
    value = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **values)
    temp = HERE / 'progress.vf.tmp'
    save(temp, value)
    temp.replace(HERE / 'progress.json')

def events(rows, column):
    result = []
    for row in rows:
        label = row[column]
        if label not in ('line', 'scrum'):
            continue
        a, b = (row['global_start_s'], row['global_end_s'])
        if result and result[-1]['play_type'] == label and (result[-1]['section'] == row['section']) and (abs(result[-1]['end_s'] - a) < 1e-06):
            result[-1]['end_s'] = b
        else:
            result.append(dict(id=f'VF_{label}_{len(result) + 1}', play_type=label, section=row['section'], start_s=a, end_s=b, blocks=[]))
    return result

def main():
    began = time.perf_counter()
    assert read(HERE / 'training_complete.json')['status'] == 'complete'
    progress('preparing_vf25_cached_inputs', blocks=0, total=688)
    source = ROOT / 'data/vf_full_sampling_20261025'
    historical = read(source / 'results.json')
    timeline = read(source / 'timelines.json')
    assert len(timeline['rows']) == 688 and historical['evaluated_blocks'] == 688
    rows = timeline['rows']
    truth = events(rows, 'manual')
    assert len(truth) == 40
    cache_paths = sorted(_model_artifact_glob(source / 'features', '*.npz', recursive=False))
    stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in cache_paths}
    batches = []

    def align(values):
        x = np.linspace(0, 1, len(values), dtype=np.float32)
        target = np.linspace(0, 1, 200, dtype=np.float32)
        return np.stack([np.interp(target, x, values[:, j]) for j in range(values.shape[1])], axis=1).astype(np.float32)
    for section in ('S01', 'S02', 'S03', 'S04'):
        chunks = []
        for path in cache_paths:
            if not path.name.startswith(section + '_'):
                continue
            with _load_model_result_array(path) as z:
                assert str(z['protocol']) == 'VF_center224_upscaled640_every25_grid25fps_v1'
                chunks.append({k: z[k].copy() for k in ('indices', 'rgb', 'positions', 'poses', 'mask')})
        sparse = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        idx = sparse['indices']
        np.testing.assert_array_equal(idx, np.unique(np.r_[np.arange(0, 30030, 25), 30029]))
        assert len(idx) == 1203
        for row in (r for r in rows if r['section'] == section):
            a = int(round(row['start_s'] * 25))
            z = min(30030, int(round(row['end_s'] * 25)))
            grid = np.arange(a, z)
            right = np.searchsorted(idx, grid).clip(0, len(idx) - 1)
            left = (right - 1).clip(0)
            w = ((grid - idx[left]) / np.maximum(idx[right] - idx[left], 1)).astype(np.float32)
            rgb = np.zeros((200, 2048), np.float32)
            rgb[:len(grid)] = sparse['rgb'][left] * (1 - w[:, None]) + sparse['rgb'][right] * w[:, None]
            p = sparse['positions'][left] * (1 - w[:, None]) + sparse['positions'][right] * w[:, None]
            q = sparse['poses'][left] * (1 - w[:, None]) + sparse['poses'][right] * w[:, None]
            nearest = np.where(grid - idx[left] <= idx[right] - grid, left, right)
            mask = sparse['mask'][nearest]
            chosen = np.linspace(0, len(grid) - 1, 100, dtype=np.int32) if len(grid) > 100 else np.arange(len(grid))
            values = np.concatenate([p[chosen], q[chosen]], axis=1)
            valid_mask = mask[chosen]
            pose = np.concatenate([align(values), align(valid_mask.astype(np.float32)[:, None])], axis=1)
            valid = values[valid_mask] if valid_mask.any() else values
            delta = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, 61), np.float32)
            geo = np.concatenate([valid.mean(0), valid.std(0), valid.min(0), valid.max(0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), delta.mean(0), delta.std(0), np.abs(delta).mean(0)]).astype(np.float32)
            batches.append((rgb, pose, np.arange(200) < len(grid), geo))
    assert len(batches) == 688
    assert [(r['section'], r['block_index']) for r in rows] == sorted(((r['section'], r['block_index']) for r in rows))
    import tensorflow as tf, joblib
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    old = ROOT / 'data/m1_ensemble_vs_all_development_20261012/models'
    paths = [('Original Â· cada 25', old, 1)] + [(f'Costo Ã—{c} Â· cada 25', HERE / 'models' / f'cost_{c}', c) for c in (2, 4, 8)]
    outputs = []
    baseline_error = None
    for title, folder, cost in paths:
        progress('vf_cached25_evaluation', cost=cost, windows=0, total_windows=688)
        weights = {str(p): sha(p) for p in (folder / 'dual.keras', folder / 'geometry_rf.joblib')}
        if cost == 1:
            assert all((historical['model_hashes'][p] == h for p, h in weights.items()))
        else:
            assert weights == read(folder / 'complete.json')['model_hashes']
        dual = tf.keras.models.load_model(_resolve_model_artifact(folder / 'dual.keras'), compile=False)
        forest = joblib.load(_resolve_model_artifact(folder / 'geometry_rf.joblib'))
        forest.n_jobs = 1
        graph = tf.function(lambda x: dual(x, training=False), input_signature=[{'rgb': tf.TensorSpec([None, 200, 2048], tf.float32), 'pose': tf.TensorSpec([None, 200, 62], tf.float32), 'sequence_mask': tf.TensorSpec([None, 200], tf.bool)}], autograph=False)
        probabilities = []
        tick = time.perf_counter()
        last = 0
        for off in range(0, len(batches), 8):
            batch = batches[off:off + 8]
            dp = np.asarray(graph({'rgb': np.stack([b[0] for b in batch]), 'pose': np.stack([b[1] for b in batch]), 'sequence_mask': np.stack([b[2] for b in batch])}).numpy(), np.float64)
            gp = np.zeros_like(dp)
            gp[:, np.asarray(forest.classes_).astype(int)] = forest.predict_proba(np.stack([b[3] for b in batch]))
            f = np.exp((1 - 0.475) * np.log(np.clip(dp, 1e-07, 1)) + 0.475 * np.log(np.clip(gp, 1e-07, 1)))
            probabilities.extend(f / f.sum(1, keepdims=True))
            if time.perf_counter() - last > 30:
                progress('vf_cached25_evaluation', cost=cost, windows=len(probabilities), total_windows=688)
                last = time.perf_counter()
        probabilities = np.asarray(probabilities)
        assert probabilities.shape == (688, 3)
        if cost == 1:
            expected = np.array([[r['p_' + label] for label in LABELS] for r in historical['rows']])
            baseline_error = float(np.abs(probabilities - expected).max())
            assert baseline_error < 2e-05
            assert np.array_equal(probabilities.argmax(1), expected.argmax(1))
        predicted = [dict(r, predicted=LABELS[int(p.argmax())], probabilities=dict(zip(LABELS, map(float, p)))) for r, p in zip(rows, probabilities)]
        grouped = events(predicted, 'predicted')
        alerts = [dict(play_type=e['play_type'], start_time=e['start_s'], end_time=e['end_s']) for e in grouped]
        value = evaluate_events(truth, alerts)
        value.update(title=title, cost=cost, stride=25, review_coverage=review_coverage(truth, alerts), cached_evaluation_seconds=time.perf_counter() - tick, model_hashes=weights, rows=predicted)
        outputs.append(value)
        del dual, forest, graph
        tf.keras.backend.clear_session()
        gc.collect()
        print(json.dumps(dict(stage='vf_cached25_complete', cost=cost, metrics=value['metrics']['total'])), flush=True)
        assert weights == {p: sha(Path(p)) for p in weights}
    dense = events(rows, 'original')
    dense_alerts = [dict(play_type=e['play_type'], start_time=e['start_s'], end_time=e['end_s']) for e in dense]
    dense_value = evaluate_events(truth, dense_alerts)
    dense_value.update(title='Original Â· completo', cost=1, stride=1, review_coverage=review_coverage(truth, dense_alerts))
    outputs.insert(0, dense_value)
    assert stats == {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in stats}
    result = dict(status='complete', scope='VF S01-S04; S05 excluded; every5 remains pending', duration_s=4804.8, reference_counts=dict(line=30, scrum=10, total=40), blocks=688, variants=outputs, reference_segments=truth, baseline_max_probability_error=baseline_error, baseline_all688_argmax_reproduced=True, extractors_executed=0, video_opened=False, source_cache_unchanged=True, evaluation='same-class one-to-one overlap>=1s; no35s anchor tolerance', elapsed_s=time.perf_counter() - began)
    save(HERE / 'vf25_results.json', result)
    progress('vf25_evaluation_complete', configurations=5, elapsed_s=time.perf_counter() - began)
if __name__ == '__main__':
    main()
