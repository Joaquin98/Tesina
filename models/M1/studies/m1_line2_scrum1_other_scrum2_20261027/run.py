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
import os, sys, time, json, hashlib, gc, traceback, argparse
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(HERE)), str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
sys.dont_write_bytecode = True
for k, v in dict(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', TF_NUM_INTRAOP_THREADS='8', TF_NUM_INTEROP_THREADS='1', PYTHONDONTWRITEBYTECODE='1').items():
    os.environ[k] = v
START = time.perf_counter()

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

def progress(stage, **kw):
    d = dict(stage=stage, pid=os.getpid(), elapsed_s=time.perf_counter() - START, updated=time.time(), **kw)
    save(HERE / 'progress.json', d)
    print(json.dumps(d), flush=True)

def train():
    import tensorflow as tf, keras, numpy as np, pandas as pd, psutil, joblib
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except psutil.Error:
        pass
    from sklearn.ensemble import RandomForestClassifier
    from models.common.comparison import models, data
    from models.M1.studies.m1_line2_scrum1_other_scrum2_20261027.weights import effective_weights
    prior = ROOT / 'data/m1_recall_cost_20261027'
    prepared = prior / 'prepared_inputs'
    hn = ROOT / 'data/scrum_hard_negatives_20261027/m1'
    csv = ROOT / 'data/m1_ensemble_vs_all_development_20261012/inputs/all_training_examples.csv'
    frame = pd.read_csv(_resolve_model_artifact(csv))
    assert len(frame) == 8796 and (not frame.block_id.duplicated().any())
    assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == read(prepared / 'manifest.json')['frame_sha256']
    arrays = {k: _load_model_result_array(prepared / (k + '.npy'), mmap_mode='r') for k in ('rgb', 'pose', 'mask', 'labels')}
    y = data.labels_for(frame)
    np.testing.assert_array_equal(y, arrays['labels'])
    idx = _load_model_result_array(hn / 'hard_indices.npy')
    pred = _load_model_result_array(hn / 'training_selection_predictions.npy')
    assert len(idx) == 106 and sha(hn / 'hard_indices.npy') == read(hn / 'mining.json')['selection_sha256']
    np.testing.assert_array_equal(idx, np.flatnonzero((y == 1) & (pred == 2)))
    weights = effective_weights(y, idx)
    assert np.isclose(weights.mean(), 1)
    geom = _load_model_result_array(hn / 'training_geometry.npy', mmap_mode='r')
    assert geom.shape == (8796, 549) and np.isfinite(geom).all()
    refs = [Path(__file__), _resolve_model_artifact(HERE / 'weights.py'), csv, prepared / 'manifest.json', models.M0_PATH, hn / 'hard_indices.npy', hn / 'training_geometry.npy', _resolve_model_artifact(ROOT / 'models/common/comparison/models.py'), _resolve_model_artifact(prior / 'run.py'), _resolve_model_artifact(prior / 'core.py')]
    hashes = {str(p): sha(p) for p in refs}
    stats = {str(prepared / (k + '.npy')): [(prepared / (k + '.npy')).stat().st_size, (prepared / (k + '.npy')).stat().st_mtime_ns] for k in arrays}
    save(HERE / 'protocol.json', dict(model='M1-L2-S1-OTHER_SCRUM2', requested_priorities=dict(line_any_error=2, scrum_any_error=1, other_to_scrum=2, other_to_line=1), implementation='class-balanced sample weights times[2,1,1], other weight doubled on106 training-only frozen hard negatives; normalized mean1; identical for neural and forest', limitation='Frozen hard-negative sample weighting, not an instantaneous per-confusion loss matrix. Corrected examples retain their selected training weight.', examples=8796, epochs=5, batch=16, seed=42, optimizer='Adam1e-4', initialization=str(models.M0_PATH), level='H2', fusion_geometry_alpha=0.475, selection='final epoch5 fixed, no target selection', extractors_executed=0, source_hashes=hashes, cache_stats=stats, evaluation='same human fd96032b00fa, full/every5, common argmax event policy, sameclass overlap>=1s one-to-one'))
    out = HERE / 'models/cost_2'
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'sample_weights.npy', weights)
    models.seed_everything(42)
    net = models.build_dual_model(models.M0_PATH, level='H2')
    fp = hashlib.sha256(b''.join((w.tobytes() for w in net.get_weights()))).hexdigest()
    assert fp == read(prior / 'models/cost_2/complete.json')['initial_weights_sha256']
    progress('training_forest', hard_negatives=106)
    tick = time.perf_counter()
    forest = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=42, n_jobs=8).fit(geom, y, sample_weight=weights)
    forest_s = time.perf_counter() - tick
    joblib.dump(forest, out / 'geometry_rf.joblib')
    del forest

    class Sequence(models.ResNetCacheSequence):

        def __getitem__(self, b):
            ii = self.indices[b * self.batch_size:(b + 1) * self.batch_size]
            return ({'rgb': arrays['rgb'][ii], 'pose': arrays['pose'][ii], 'sequence_mask': arrays['mask'][ii]}, y[ii], weights[ii].astype(np.float32))
    net.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    seq = Sequence(frame, batch_size=16, shuffle=True, seed=42, dual=True)
    tick = time.perf_counter()
    history = []

    class Progress(keras.callbacks.Callback):

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch + 1
            self.last = 0
            self.tick = time.perf_counter()

        def on_train_batch_end(self, batch, logs=None):
            if time.perf_counter() - self.last > 30:
                progress('training', epoch=self.epoch, epochs=5, batch=batch + 1, batches=len(seq), training_s=time.perf_counter() - tick)
                self.last = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            logs = {k: float(v) for k, v in (logs or {}).items()}
            history.append(dict(epoch=epoch + 1, logs=logs, epoch_s=time.perf_counter() - self.tick))
            net.save(out / 'last.keras')
            save(out / 'history.json', history)
            progress('epoch_complete', epoch=epoch + 1, epochs=5, logs=logs, training_s=time.perf_counter() - tick)
    net.fit(seq, epochs=5, callbacks=[Progress()], verbose=2)
    elapsed = time.perf_counter() - tick
    net.save(out / 'dual.keras')
    r = dict(epochs=5, examples=8796, initial_weights_sha256=fp, forest_s=forest_s, dual_s=elapsed, hard_negatives=106, effective_weights_by_group=dict(line=float(weights[y == 0][0]), scrum=float(weights[y == 2][0]), other_regular=float(weights[(y == 1) & (pred != 2)][0]), other_hard=float(weights[idx][0])), model_hashes={str(p): sha(p) for p in (out / 'dual.keras', out / 'geometry_rf.joblib')}, saved_unix=time.time())
    save(out / 'complete.json', r)
    assert hashes == {p: sha(p) for p in hashes}
    assert stats == {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in stats}
    save(HERE / 'training_complete.json', dict(status='complete', models=[r], source_hashes=hashes, source_cache_stats_unchanged=True, all_models_saved_before_evaluation_unix=time.time()))
    progress('training_complete', dual_s=elapsed, forest_s=forest_s)

def evaluate():
    source = _resolve_model_artifact(ROOT / 'models/M1/studies/m1_recall_cost_20261027/run.py')
    text = _resolve_model_artifact(source).read_text(encoding='utf-8-sig')
    text = text.replace('HERE=experiment_directory(__file__);ROOT=HERE.parents[1]', f'ROOT=Path({str(ROOT)!r});HERE=ROOT/"data/m1_line2_scrum1_other_scrum2_20261027"')
    text = text.replace('COSTS=(2,4,8)', 'COSTS=(2,)')
    ns = dict(__file__=str(source), __name__='isolated_cached_evaluation')
    exec(compile(text, str(source), 'exec'), ns)
    ns['evaluate']()
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'evaluate'])
    a = p.parse_args()
    try:
        train() if a.mode == 'train' else evaluate()
    except Exception:
        progress('error', traceback=traceback.format_exc())
        raise
