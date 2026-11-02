"""Line-only cost sweep: unchanged amateur training and frozen feature caches."""
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
import os, sys, time, json, hashlib, gc, argparse, traceback
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(HERE)), str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
sys.dont_write_bytecode = True
for k, v in dict(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', TF_NUM_INTRAOP_THREADS='8', TF_NUM_INTEROP_THREADS='1', PYTHONDONTWRITEBYTECODE='1').items():
    os.environ[k] = v
START = time.perf_counter()
COSTS = (2, 4, 8)

def read(p):
    return json.loads(_resolve_model_artifact(Path(p)).read_text(encoding='utf-8-sig'))

def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')
    t.replace(p)

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(p)).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def progress(stage, **kw):
    d = dict(stage=stage, pid=os.getpid(), elapsed_s=time.perf_counter() - START, updated_unix=time.time(), **kw)
    save(HERE / 'progress.json', d)
    print(json.dumps(d), flush=True)

def initialize():
    import tensorflow as tf, psutil
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    return tf

def train():
    tf = initialize()
    import keras, numpy as np, pandas as pd, joblib
    from sklearn.ensemble import RandomForestClassifier
    from models.common.comparison import models, data
    from prepared_sequence import make_sequence
    from models.M1.studies.m1_line_only_costs_20261028.weights import class_weights
    prior = ROOT / 'data/m1_recall_cost_20261027'
    prepared = prior / 'prepared_inputs'
    csv = ROOT / 'data/m1_ensemble_vs_all_development_20261012/inputs/all_training_examples.csv'
    frame = pd.read_csv(_resolve_model_artifact(csv))
    assert len(frame) == 8796 and (not frame.block_id.duplicated().any()) and ((frame.domain == 'CH').sum() == 4082) and ((frame.domain == 'CV').sum() == 4714)
    assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == read(prepared / 'manifest.json')['frame_sha256']
    arrays = {k: _load_model_result_array(prepared / (k + '.npy'), mmap_mode='r') for k in ('rgb', 'pose', 'mask', 'labels')}
    y = data.labels_for(frame)
    np.testing.assert_array_equal(y, arrays['labels'])
    geometry = ROOT / 'data/scrum_hard_negatives_20261027/m1/training_geometry.npy'
    geom = _load_model_result_array(geometry, mmap_mode='r')
    assert geom.shape == (8796, 549) and np.isfinite(geom).all()
    refs = [Path(__file__), _resolve_model_artifact(HERE / 'weights.py'), csv, prepared / 'manifest.json', models.M0_PATH, geometry, _resolve_model_artifact(ROOT / 'models/common/comparison/models.py'), _resolve_model_artifact(prior / 'prepared_sequence.py'), _resolve_model_artifact(prior / 'core.py'), prior / 'models/cost_2/dual.keras', prior / 'models/cost_2/geometry_rf.joblib']
    hashes = {str(p): sha(p) for p in refs}
    stats = {str(prepared / (k + '.npy')): [(prepared / (k + '.npy')).stat().st_size, (prepared / (k + '.npy')).stat().st_mtime_ns] for k in arrays}
    protocol = dict(model='M1-H2-D2 extra cost only for line', line_costs=list(COSTS), baseline_costs=[2, 1, 2], new_costs=[[2, 1, 1], [4, 1, 1], [8, 1, 1]], label_order=['line', 'other', 'scrum'], implementation='class-balanced weights times[line_cost,1,1], mean1; weighted sparse CE and same RF class weights', limitation='All errors on true line examples receive higher weight, not only line-to-other; correct examples also carry their class weight', examples=8796, epochs=5, batch_size=16, seed=42, optimizer='Adam1e-4', initialization=str(models.M0_PATH), level='H2', fusion_geometry_alpha=0.475, forest=dict(n_estimators=500, max_features='sqrt', random_state=42, n_jobs=8), selection='all three final epoch5 checkpoints fixed before evaluation; no professional data in training', extractors_executed=0, source_hashes=hashes, cache_stats=stats, evaluation='three professional full native caches;16/08 human fd96032b00fa;23/08 and RWC2023 partial automated diagnostic inventories; exploratory already-inspected matches')
    if _resolve_model_artifact(HERE / 'protocol.json').exists():
        assert read(HERE / 'protocol.json') == protocol
    else:
        save(HERE / 'protocol.json', protocol)
    Sequence = make_sequence(models, arrays)
    records = []
    for cost in COSTS:
        out = HERE / f'models/line{cost}_scrum1'
        out.mkdir(parents=True, exist_ok=True)
        if _resolve_model_artifact(out / 'complete.json').exists():
            r = read(out / 'complete.json')
            assert all((sha(p) == h for p, h in r['model_hashes'].items()))
            records.append(r)
            continue
        keras.backend.clear_session()
        gc.collect()
        models.seed_everything(42)
        net = models.build_dual_model(models.M0_PATH, level='H2')
        fp = hashlib.sha256(b''.join((w.tobytes() for w in net.get_weights()))).hexdigest()
        assert fp == read(prior / 'models/cost_2/complete.json')['initial_weights_sha256']
        w = class_weights(y, cost)
        cw = {i: float(x) for i, x in enumerate(w)}
        progress('training_forest', line_cost=cost, scrum_cost=1, class_weights=cw)
        t = time.perf_counter()
        forest = RandomForestClassifier(**protocol['forest'], class_weight=cw).fit(geom, y)
        forest_s = time.perf_counter() - t
        joblib.dump(forest, out / 'geometry_rf.joblib')
        del forest
        seq = Sequence(frame, batch_size=16, shuffle=True, seed=42, dual=True)
        net.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        t = time.perf_counter()
        history = []

        class Progress(keras.callbacks.Callback):

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch + 1
                self.last = 0
                self.tick = time.perf_counter()

            def on_train_batch_end(self, batch, logs=None):
                if time.perf_counter() - self.last > 30:
                    progress('training', line_cost=cost, scrum_cost=1, epoch=self.epoch, epochs=5, batch=batch + 1, batches=len(seq), training_s=time.perf_counter() - t)
                    self.last = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                history.append(dict(epoch=epoch + 1, epoch_s=time.perf_counter() - self.tick, logs={k: float(v) for k, v in (logs or {}).items()}))
                net.save(out / 'last.keras')
                save(out / 'history.json', history)
                progress('epoch_complete', line_cost=cost, epoch=epoch + 1, epochs=5, training_s=time.perf_counter() - t)
        net.fit(seq, epochs=5, class_weight=cw, callbacks=[Progress()], verbose=2)
        dual_s = time.perf_counter() - t
        net.save(out / 'dual.keras')
        r = dict(line_cost=cost, scrum_cost=1, other_cost=1, epochs=5, examples=8796, initial_weights_sha256=fp, class_weights=cw, forest_s=forest_s, dual_s=dual_s, model_hashes={str(p): sha(p) for p in (out / 'dual.keras', out / 'geometry_rf.joblib')}, saved_unix=time.time())
        save(out / 'complete.json', r)
        records.append(r)
        del net
        keras.backend.clear_session()
        gc.collect()
    assert hashes == {p: sha(p) for p in hashes}
    assert stats == {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in stats}
    save(HERE / 'training_complete.json', dict(status='complete', models=records, all_models_saved_before_evaluation_unix=time.time(), elapsed_s=time.perf_counter() - START, source_models_and_caches_unchanged=True))
    progress('training_complete')
if __name__ == '__main__':
    try:
        train()
    except Exception:
        save(HERE / 'training_error.json', dict(error=traceback.format_exc()))
        raise
