"""Exact historical M0 H2/D2 RGB training; no extractors imported or called."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.M1.studies.m0_m1_cost_sampling_20261029.common import *
import argparse, gc, traceback
from models.M1.studies.m0_m1_cost_sampling_20261029.inputs import cost_weights

def main(only=None):
    tf = initialize()
    import keras, numpy as np, pandas as pd
    from models.common.comparison import models, data
    prepared = ROOT / 'data/m1_recall_cost_20261027/prepared_inputs'
    csv = ROOT / 'data/m1_ensemble_vs_all_development_20261012/inputs/all_training_examples.csv'
    frame = pd.read_csv(_resolve_model_artifact(csv))
    assert len(frame) == 8796 and (not frame.block_id.duplicated().any())
    assert (frame.domain == 'CH').sum() == 4082 and (frame.domain == 'CV').sum() == 4714
    assert hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == read(prepared / 'manifest.json')['frame_sha256']
    arrays = {k: _load_model_result_array(prepared / f'{k}.npy', mmap_mode='r') for k in ['rgb', 'mask', 'labels']}
    y = data.labels_for(frame)
    np.testing.assert_array_equal(y, arrays['labels'])
    assert arrays['rgb'].shape == (8796, 200, 2048)
    for i in [0, 4081, 4082, 8795]:
        row = frame.iloc[i]
        x, m, _ = data._load_resnet_path.__wrapped__(str(row.resnet_path), str(row.block_id))
        np.testing.assert_array_equal(x, arrays['rgb'][i])
        np.testing.assert_array_equal(m, arrays['mask'][i])
    stats = {str(prepared / f'{k}.npy'): stat(prepared / f'{k}.npy') for k in arrays}
    sources = [models.M0_PATH, csv, prepared / 'manifest.json', Path(__file__), _resolve_model_artifact(HERE / 'common.py'), _resolve_model_artifact(HERE / 'inputs.py'), _resolve_model_artifact(ROOT / 'models/common/comparison/models.py'), _resolve_model_artifact(ROOT / 'models/common/comparison/data.py')]
    hashes = {str(p): sha(p) for p in sources}
    old = read(ROOT / 'data/m0_cost2_professional_human_20261027/protocol.json')
    assert hashes[str(models.M0_PATH)] == old['source_hashes'][str(models.M0_PATH)]
    protocol = dict(configurations=CONFIGS, model='M0-H2-D2 RGB200; no YOLO/RF', examples=8796, epochs=5, batch_size=16, seed=42, optimizer='Adam1e-4', inputs='exact read-only prepared RGB and temporal masks', initialization=str(models.M0_PATH), class_cost='class-balanced weights times[line,1,scrum], normalized empirical mean1', extractors_executed=0, source_hashes=hashes, cache_stats=stats, selection='Fixed final epoch5, fixed costs before professional/VF evaluation, exploratory observed videos', sampling='Inference only, no sparse retraining', scope='3 professional full, human every5, VF central every25 separately')
    if _resolve_model_artifact(HERE / 'protocol.json').exists():
        assert read(HERE / 'protocol.json') == json.loads(json.dumps(protocol))
    else:
        save(HERE / 'protocol.json', protocol)

    class Sequence(models.ResNetCacheSequence):

        def __getitem__(self, b):
            ids = self.indices[b * self.batch_size:(b + 1) * self.batch_size]
            return ({'features': arrays['rgb'][ids], 'mask': arrays['mask'][ids]}, arrays['labels'][ids])
    for key, line, scrum in CONFIGS:
        if only and key != only:
            continue
        out = HERE / f'models/{key}'
        out.mkdir(exist_ok=True, parents=True)
        path = model_path(key)
        cw = {i: float(v) for i, v in enumerate(cost_weights(y, line, scrum))}
        if key == 'x2':
            complete = read(ROOT / 'data/m0_cost2_professional_human_20261027/training_complete.json')
            assert sha(path) == complete['model_sha256']
            assert cw == {int(k): v for k, v in old['class_weights'].items()}
            save(out / 'complete.json', dict(key=key, line_cost=line, scrum_cost=scrum, model_path=str(path), model_sha256=sha(path), class_weights=cw, training_s=complete['training_s'], reused=True, epochs=5))
            continue
        if _resolve_model_artifact(out / 'complete.json').exists():
            assert sha(path) == read(out / 'complete.json')['model_sha256']
            continue
        keras.backend.clear_session()
        gc.collect()
        models.seed_everything(42)
        net = keras.models.load_model(_resolve_model_artifact(models.M0_PATH), compile=False)
        models._configure_m0_trainable(net, 'H2')
        initial = hashlib.sha256(b''.join((v.tobytes() for v in net.get_weights()))).hexdigest()
        save(out / 'initial.json', dict(sha256=initial, source_sha256=hashes[str(models.M0_PATH)]))
        net.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        seq = Sequence(frame, batch_size=16, shuffle=True, seed=42)
        tick = time.perf_counter()
        history = []

        class Progress(keras.callbacks.Callback):

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch + 1
                self.last = 0
                self.tick = time.perf_counter()

            def on_train_batch_end(self, batch, logs=None):
                if time.perf_counter() - self.last > 30:
                    progress('training', key=key, line_cost=line, scrum_cost=scrum, epoch=self.epoch, epochs=5, batch=batch + 1, batches=len(seq), training_s=time.perf_counter() - tick)
                    self.last = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                history.append(dict(epoch=epoch + 1, epoch_s=time.perf_counter() - self.tick, logs={k: float(v) for k, v in (logs or {}).items()}))
                net.save(out / 'last.keras')
                save(out / 'history.json', history)
        net.fit(seq, epochs=5, class_weight=cw, callbacks=[Progress()], verbose=2)
        net.save(path)
        save(out / 'complete.json', dict(key=key, line_cost=line, scrum_cost=scrum, model_path=str(path), model_sha256=sha(path), initial_weights_sha256=initial, class_weights=cw, training_s=time.perf_counter() - tick, parameters=net.count_params(), reused=False, epochs=5, saved_unix=time.time()))
        del net
        keras.backend.clear_session()
        gc.collect()
        progress('model_complete', key=key)
        assert stats == {p: stat(p) for p in stats} and hashes == {p: sha(p) for p in hashes}
    records = [read(HERE / f'models/{k}/complete.json') for k, _, _ in CONFIGS if _resolve_model_artifact(HERE / f'models/{k}/complete.json').exists()]
    save(HERE / 'training_status.json', dict(status='complete' if len(records) == 7 else 'partial', models=records))
    if len(records) == 7:
        save(HERE / 'training_complete.json', dict(status='complete', models=records, sources_unchanged=True, source_hashes=hashes, cache_stats=stats))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', choices=[c[0] for c in CONFIGS])
    a = parser.parse_args()
    try:
        main(a.key)
    except Exception:
        save(HERE / 'training_error.json', dict(error=traceback.format_exc(), key=a.key))
        raise
