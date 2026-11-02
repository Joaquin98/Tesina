"""M0-H2-D2 cost2, true cached-feature training and frozen evaluation."""
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
import os, sys, time, json, hashlib, functools, gc
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
sys.dont_write_bytecode = True
for k, v in dict(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', TF_NUM_INTRAOP_THREADS='8', TF_NUM_INTEROP_THREADS='1').items():
    os.environ[k] = v
START = time.perf_counter()

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def save(p, v):
    p = HERE / p
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')
    t.replace(p)

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(p).open('rb') as f:
        for x in iter(lambda: f.read(1048576), b''):
            h.update(x)
    return h.hexdigest()

def progress(stage, **v):
    save('progress.json', dict(stage=stage, elapsed_s=time.perf_counter() - START, updated=time.time(), pid=os.getpid(), **v))

def main():
    import tensorflow as tf, keras, numpy as np, pandas as pd, psutil
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except psutil.Error:
        pass
    from models.common.comparison import models, data
    from core import training_weights, SelectedRGB, evaluate_events
    from review_coverage import review_coverage
    from models.M2.predictor import RGBYoloHighresQuantileFusionDetector as Policy
    prior = ROOT / 'data/m1_recall_cost_20261027'
    folder = prior / 'prepared_inputs'
    csv = ROOT / 'data/m1_ensemble_vs_all_development_20261012/inputs/all_training_examples.csv'
    frame = pd.read_csv(_resolve_model_artifact(csv))
    assert len(frame) == 8796 and len(frame.block_id.unique()) == 8796
    assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == read(folder / 'manifest.json')['frame_sha256']
    arrays = {k: _load_model_result_array(folder / (k + '.npy'), mmap_mode='r') for k in ('rgb', 'mask', 'labels')}
    y = data.labels_for(frame)
    np.testing.assert_array_equal(arrays['labels'], y)
    for i in [0, 10, 4081, 4082, 5000, 8795]:
        row = frame.iloc[i]
        a, b, _ = data._load_resnet_path.__wrapped__(str(row.resnet_path), str(row.block_id))
        np.testing.assert_array_equal(a, arrays['rgb'][i])
        np.testing.assert_array_equal(b, arrays['mask'][i])
    weights = {i: float(w) for i, w in enumerate(training_weights(y, 2))}
    assert weights == {int(k): v for k, v in read(prior / 'models/cost_2/complete.json')['class_weights'].items()}
    refs = [models.M0_PATH, csv, folder / 'manifest.json', Path(__file__), _resolve_model_artifact(ROOT / 'models/common/comparison/models.py'), _resolve_model_artifact(ROOT / 'models/common/comparison/data.py'), _resolve_model_artifact(prior / 'core.py')]
    hashes = {str(p): sha(p) for p in refs}
    cache_stats = {str(folder / (k + '.npy')): [(folder / (k + '.npy')).stat().st_size, (folder / (k + '.npy')).stat().st_mtime_ns] for k in arrays}
    save('protocol.json', dict(model='M0-H2-D2-cost2', seed=42, epochs=5, batch=16, examples=8796, optimizer='Adam1e-4', loss='sparse CE', class_weights=weights, initialization='original historical M0 checkpoint, full H2 temporal/head fine tuning', inputs='existing exact RGB/mask cached tensors; no pose', source_hashes=hashes, cache_stats=cache_stats, evaluation='original human labels fd96032b00fa, full/every5; common M1 argmax event policy; class overlap>=1s one-to-one', selection='final epoch5 fixed; no target-based tuning'))
    model_path = HERE / 'models/m0_cost2.keras'
    if not _resolve_model_artifact(HERE / 'training_complete.json').exists():
        models.seed_everything(42)
        net = keras.models.load_model(_resolve_model_artifact(models.M0_PATH), compile=False)
        models._configure_m0_trainable(net, 'H2')
        net.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        class Sequence(models.ResNetCacheSequence):

            def __getitem__(self, b):
                idx = self.indices[b * self.batch_size:(b + 1) * self.batch_size]
                return ({'features': arrays['rgb'][idx], 'mask': arrays['mask'][idx]}, arrays['labels'][idx])
        train_start = time.perf_counter()

        class Progress(keras.callbacks.Callback):

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch + 1
                self.last = 0

            def on_train_batch_end(self, batch, logs=None):
                if time.perf_counter() - self.last >= 30:
                    progress('training_m0', epoch=self.epoch, epochs=5, batch=batch + 1, batches=550)
                    self.last = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                progress('epoch_complete', epoch=epoch + 1, epochs=5, logs={k: float(v) for k, v in (logs or {}).items()})
        model_path.parent.mkdir(exist_ok=True)
        h = net.fit(Sequence(frame, batch_size=16, shuffle=True, seed=42), epochs=5, class_weight=weights, callbacks=[Progress()], verbose=2)
        net.save(model_path)
        save('history.json', h.history)
        save('training_complete.json', dict(status='complete', training_s=time.perf_counter() - train_start, model_sha256=sha(model_path), saved_unix=time.time(), parameters=net.count_params()))
        del net
        keras.backend.clear_session()
        gc.collect()
    trained = read(HERE / 'training_complete.json')
    assert sha(model_path) == trained['model_sha256']
    net = keras.models.load_model(_resolve_model_artifact(model_path), compile=False)
    graph = tf.function(lambda a, b: net([a, b], training=False), input_signature=[tf.TensorSpec([None, 200, 2048], tf.float32), tf.TensorSpec([None, 200], tf.bool)], autograph=False)
    rgbpath = ROOT / 'data/professional_rugby_m1_game1_20261024/results/rgb_full.npy'
    rgb = _load_model_result_array(rgbpath, mmap_mode='r')
    n = len(rgb)
    assert n == 154161
    cache_stats[str(rgbpath)] = [rgbpath.stat().st_size, rgbpath.stat().st_mtime_ns]
    refpath = ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json'
    ref = read(refpath)
    for s in ref['provenance']:
        assert sha(Path(s['path'])) == s['sha256']
    baseline = read(ROOT / 'data/results/result_20261025_argentina_allblacks_20250816_argmax.json')
    policy = Policy.__new__(Policy)
    for k, v in baseline['event_post_processing'].items():
        setattr(policy, k, v)
    records = []
    starts = np.arange(0, n - 174, 25)
    assert len(starts) == 6160
    for stride in (1, 5):
        out = HERE / f'stride_{stride}'
        out.mkdir(exist_ok=True)
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
        checks = [inputs(int(starts[i])) for i in (0, 1260, 4942, 6000)]
        masks = np.tile(np.arange(200) < 175, (4, 1))
        controls = np.asarray(net([np.stack(checks), masks], training=False).numpy())
        actual = np.asarray(graph(np.stack(checks), masks).numpy())
        err = float(np.abs(controls - actual).max())
        assert err < 1e-06
        rows = [json.loads(x) for x in _resolve_model_artifact(out / 'windows.jsonl').read_text(encoding='utf-8').splitlines()] if _resolve_model_artifact(out / 'windows.jsonl').exists() else []
        assert all((r['frame_start'] == int(starts[i]) for i, r in enumerate(rows)))
        last = 0
        with _resolve_model_artifact(out / 'windows.jsonl').open('a', encoding='utf-8') as f:
            for wi in range(len(rows), len(starts), 8):
                batch = np.stack([inputs(int(a)) for a in starts[wi:wi + 8]])
                prob = graph(batch, np.tile(np.arange(200) < 175, (len(batch), 1))).numpy()
                assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-06)
                for j, p in enumerate(prob):
                    a = int(starts[wi + j])
                    r = dict(play_type=['line', 'other', 'scrum'][int(p.argmax())], start_time=a / 25, end_time=(a + 175) / 25, duration=7.0, confidence=float(p.max()), frame_start=a, frame_end=a + 175, probabilities=dict(zip(['line', 'other', 'scrum'], map(float, p))))
                    rows.append(r)
                    f.write(json.dumps(r) + '\n')
                f.flush()
                if time.perf_counter() - last > 30:
                    progress('evaluating_m0', stride=stride, windows=len(rows), total=6160)
                    last = time.perf_counter()
        events = [e for e in policy._post_process_event_detections(rows, 0.0, 6166.44, 25, {'line': 0.0}) if e['end_time'] > 188.2]
        v = evaluate_events(ref['reference_segments'], events)
        v.update(stride=stride, windows=len(rows), selected_frames=len(idx), cached_evaluation_s=time.perf_counter() - tick, control_max_error=err, reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], review_coverage=review_coverage(ref['reference_segments'], events))
        save(f'stride_{stride}/events.json', events)
        save(f'stride_{stride}/metrics.json', v)
        records.append(v)
        progress('configuration_complete', stride=stride, metrics=v['metrics']['total'])
    assert hashes == {p: sha(Path(p)) for p in hashes}
    assert cache_stats == {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in cache_stats}
    assert sha(model_path) == trained['model_sha256']
    for s in ref['provenance']:
        assert sha(Path(s['path'])) == s['sha256']
    save('completion.json', dict(status='complete', records=records, training=trained, source_hashes=hashes, source_cache_stats_unchanged=True, extractors_executed=0, new_yolo_observations=0, total_process_s=time.perf_counter() - START))
    progress('evaluation_complete')
if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        progress('error', traceback=traceback.format_exc())
        raise
