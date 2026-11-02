"""One family worker: fixed five-epoch controls/costs on held-out CV folds."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.M1.studies.m0_m1_cost_folds_20261030.settings import *
import argparse, gc, traceback, importlib.util
import numpy as np, pandas as pd
from models.M1.studies.m0_m1_cost_folds_20261030.fold_core import class_weights, selected_indices, evaluate

def main(family):
    import tensorflow as tf, keras, joblib, psutil
    from sklearn.ensemble import RandomForestClassifier
    from models.common.comparison import models, data
    from models.common.comparison.metrics import block_metrics, temporal_metrics
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except psutil.Error:
        pass
    protocol = read(HERE / 'protocol.json')
    splits = read(HERE / 'splits.json')
    frame = pd.read_csv(_resolve_model_artifact(CSV))
    assert protocol['source_hashes'] == {p: sha(p) for p in protocol['source_hashes']}
    arrays = {k: _load_model_result_array(ARRAYS / f'{k}.npy', mmap_mode='r') for k in ['rgb', 'pose', 'mask', 'labels']}
    geom = _load_model_result_array(HERE / 'prepared/geometry.npy', mmap_mode='r')
    source_stats = read(HERE / 'source_cache_stats.json')
    spec = importlib.util.spec_from_file_location('fold_independent_check', _resolve_model_artifact(ROOT / 'models/M1/studies/m1_recall_cost_20261027/publish.py'))
    ind = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ind)
    expected_initial = read(PRO / 'models/original/initial.json')['sha256'] if family == 'm0' else read(ROOT / 'data/m1_recall_cost_20261027/models/cost_2/complete.json')['initial_weights_sha256']

    class Sequence(models.ResNetCacheSequence):

        def __init__(self, ids, **kw):
            self.global_ids = np.asarray(ids, dtype=int)
            super().__init__(frame.iloc[ids], dual=family == 'm1', **kw)

        def __getitem__(self, batch):
            idx = selected_indices(self.global_ids, self.indices, batch, self.batch_size)
            if family == 'm0':
                inputs = dict(features=arrays['rgb'][idx], mask=arrays['mask'][idx])
            else:
                inputs = dict(rgb=arrays['rgb'][idx], pose=arrays['pose'][idx], sequence_mask=arrays['mask'][idx])
            return (inputs, arrays['labels'][idx])
    for fold in FOLDS:
        train_ids = np.array(splits[fold]['train']['indices'])
        test_ids = np.array(splits[fold]['test']['indices'])
        train_y = np.asarray(arrays['labels'][train_ids])
        test = frame.iloc[test_ids].reset_index(drop=True)
        for key, fam, line, scrum in CONFIGS:
            if fam != family:
                continue
            out = HERE / 'runs' / fold / key
            out.mkdir(parents=True, exist_ok=True)
            if _resolve_model_artifact(out / 'metrics.json').exists():
                old = read(out / 'metrics.json')
                assert old['model_hashes'] == {p: sha(p) for p in old['model_hashes']}
                continue
            keras.backend.clear_session()
            gc.collect()
            models.seed_everything(42)
            cw = {i: float(v) for i, v in enumerate(class_weights(train_y, line, scrum))}
            model_file = out / ('model.keras' if family == 'm0' else 'dual.keras')
            if not _resolve_model_artifact(out / 'trained.json').exists():
                if family == 'm0':
                    net = keras.models.load_model(_resolve_model_artifact(BASE), compile=False)
                    models._configure_m0_trainable(net, 'H2')
                else:
                    net = models.build_dual_model(BASE, level='H2')
                fingerprint = hashlib.sha256(b''.join((w.tobytes() for w in net.get_weights()))).hexdigest()
                assert fingerprint == expected_initial, (family, fingerprint, expected_initial)
                forest_s = 0.0
                if family == 'm1':
                    progress(family, stage='training_forest', fold=fold, key=key, train_blocks=len(train_ids))
                    tick = time.perf_counter()
                    forest = RandomForestClassifier(**protocol['m1_forest'], class_weight=cw).fit(geom[train_ids], train_y)
                    forest_s = time.perf_counter() - tick
                    joblib.dump(forest, out / 'geometry_rf.joblib')
                    del forest
                net.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                seq = Sequence(train_ids, batch_size=16, shuffle=True, seed=42)
                tick = time.perf_counter()
                history = []

                class Progress(keras.callbacks.Callback):

                    def on_epoch_begin(self, epoch, logs=None):
                        self.epoch = epoch + 1
                        self.last = 0

                    def on_train_batch_end(self, batch, logs=None):
                        if time.perf_counter() - self.last > 30:
                            progress(family, stage='training', fold=fold, key=key, epoch=self.epoch, epochs=5, batch=batch + 1, batches=len(seq), training_s=time.perf_counter() - tick)
                            self.last = time.perf_counter()

                    def on_epoch_end(self, epoch, logs=None):
                        history.append(dict(epoch=epoch + 1, logs={k: float(v) for k, v in (logs or {}).items()}))
                        net.save(out / 'last.keras')
                        save(out / 'history.json', history)
                net.fit(seq, epochs=5, class_weight=cw, callbacks=[Progress()], verbose=2)
                net.save(model_file)
                hashes = {str(p): sha(p) for p in ([model_file] if family == 'm0' else [model_file, out / 'geometry_rf.joblib'])}
                trained = dict(fold=fold, key=key, family=family, epochs=5, seed=42, initial_weights_sha256=fingerprint, class_weights=cw, model_hashes=hashes, train_blocks=len(train_ids), test_blocks=len(test_ids), training_s=time.perf_counter() - tick, forest_s=forest_s, split_sha256=sha(HERE / 'splits.json'))
                save(out / 'trained.json', trained)
                del net
                keras.backend.clear_session()
                gc.collect()
            trained = read(out / 'trained.json')
            assert trained['model_hashes'] == {p: sha(p) for p in trained['model_hashes']}
            progress(family, stage='evaluating_reserved_test', fold=fold, key=key, blocks=len(test))
            net = keras.models.load_model(_resolve_model_artifact(model_file), compile=False)
            tick = time.perf_counter()
            predictions = []
            if family == 'm0':
                graph = tf.function(lambda rgb, mask: net([rgb, mask], training=False), input_signature=[tf.TensorSpec([None, 200, 2048], tf.float32), tf.TensorSpec([None, 200], tf.bool)], autograph=False)
            else:
                graph = tf.function(lambda rgb, pose, mask: net(dict(rgb=rgb, pose=pose, sequence_mask=mask), training=False), input_signature=[tf.TensorSpec([None, 200, 2048], tf.float32), tf.TensorSpec([None, 200, 62], tf.float32), tf.TensorSpec([None, 200], tf.bool)], autograph=False)
                forest = joblib.load(_resolve_model_artifact(out / 'geometry_rf.joblib'))
                forest.n_jobs = 1
            for a in range(0, len(test_ids), 8):
                idx = test_ids[a:a + 8]
                if family == 'm0':
                    p = graph(arrays['rgb'][idx], arrays['mask'][idx]).numpy()
                else:
                    p = graph(arrays['rgb'][idx], arrays['pose'][idx], arrays['mask'][idx]).numpy().astype(np.float64)
                    q = np.zeros_like(p)
                    q[:, np.asarray(forest.classes_).astype(int)] = forest.predict_proba(geom[idx])
                    p = np.exp(0.525 * np.log(np.clip(p, 1e-07, 1)) + 0.475 * np.log(np.clip(q, 1e-07, 1)))
                    p = (p / p.sum(1, keepdims=True)).astype(np.float32)
                predictions.extend(p)
            prob = np.asarray(predictions)
            assert prob.shape == (len(test), 3) and np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-06)
            names = np.array(['line', 'other', 'scrum'])
            decision = names[prob.argmax(1)]
            result = evaluate(test, test.label.tolist(), decision, prob)
            independent = ind.independent_check(result['reference_segments'], result['predictions'], result['metrics'], result['evaluated_video_s'])
            assert all((abs(independent[k] - result['review'][k]) < 1e-06 for k in ['unique_flagged_s', 'flagged_outside_true_segments_s']))
            scored = test.copy()
            scored['y_true'] = test.label
            scored['y_final'] = decision
            for i, label in enumerate(names):
                scored['p_' + label] = prob[:, i]
            scored['fold'] = fold
            scored['configuration'] = key
            scored.to_csv(out / 'predictions.csv', index=False)
            np.save(out / 'probabilities.npy', prob)
            result.update(fold=fold, key=key, family=family, test_group=splits[fold]['test']['groups'][0], test_blocks=len(test), training=trained, model_hashes=trained['model_hashes'], block_metrics=block_metrics(np.asarray(arrays['labels'][test_ids]), prob), temporal_secondary=temporal_metrics(scored), independent_check=independent, cached_test_inference_s=time.perf_counter() - tick, extractors_executed=0)
            assert protocol['source_hashes'] == {p: sha(p) for p in protocol['source_hashes']} and source_stats == {p: stat(p) for p in source_stats}
            save(out / 'metrics.json', result)
            progress(family, stage='fold_configuration_complete', fold=fold, key=key, total_metrics=result['metrics']['total'])
            del net, graph
            if family == 'm1':
                del forest
            keras.backend.clear_session()
            gc.collect()
    save(HERE / f'complete_{family}.json', dict(status='complete', family=family, runs=12, extractors_executed=0))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', required=True, choices=['m0', 'm1'])
    args = parser.parse_args()
    try:
        main(args.family)
    except Exception:
        save(HERE / f'error_{args.family}.json', dict(error=traceback.format_exc(), family=args.family))
        raise
