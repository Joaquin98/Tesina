"""Validate read-only parent, train honest pilot central, cache frozen outputs."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import time, gc, shutil
import pandas as pd
from common import *
from analysis_math import split_indices
from baseline_model import build, core_hash, keras, tf

def prepare_references():
    config = check_frozen()
    assert load(CACHE / 'analysis/postrun_verification.json')['passed']
    for name, h in config['cache_files_sha256'].items():
        assert digest(CACHE / 'features' / name) == h
    assert _load_model_result_array(CACHE / 'features/done.npy').all()
    for seed in SEEDS:
        out = HERE / f'seed_{seed}/central'
        out.mkdir(parents=True, exist_ok=True)
        for name in ['best.weights.h5', 'selected.json']:
            source = CACHE / f'seed_{seed}/fixed' / name
            target = out / name
            if _resolve_model_artifact(target).exists():
                assert digest(target) == digest(source)
            else:
                shutil.copy2(source, target)
    save(HERE / 'reference_verified.json', dict(passed=True, central_models_reused=3, cache_verified=True))

def pilot_central(rows, train):
    out = HERE / 'pilot/central'
    out.mkdir(parents=True, exist_ok=True)
    if _resolve_model_artifact(out / 'selected.json').exists():
        assert digest(out / 'best.weights.h5') == load(out / 'selected.json')['checkpoint_sha256']
        return
    assert all((rows[i]['domain'] != 'VF' and rows[i]['group_id'] != 'V_6' for i in train))
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    net, core, _ = build('fixed')
    initial = core_hash(core)
    f = _load_model_result_array(CACHE / 'features/multi.npy', mmap_mode='r')
    m = _load_model_result_array(CACHE / 'features/masks.npy', mmap_mode='r')
    y = np.array([LABELS.index(rows[i]['tag']) for i in train])
    counts = np.bincount(y, minlength=3)
    cw = {k: len(y) / (3 * int(n)) for k, n in enumerate(counts)}
    rng = np.random.default_rng(42)
    history = []
    start = time.perf_counter()
    for epoch in range(1, 16):
        t = time.perf_counter()
        order = rng.permutation(len(train))
        net.reset_metrics()
        for off in range(0, len(order), 32):
            pos = order[off:off + 32]
            idx = train[pos]
            result = net.train_on_batch([np.asarray(f[idx, :, 0]), np.asarray(m[idx])], y[pos], class_weight=cw, return_dict=True)
        row = dict(epoch=epoch, seconds=time.perf_counter() - t, **{k: float(v) for k, v in result.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        save(HERE / 'status.json', dict(phase='pilot_central_training', seed=42, epochs_total=15, **row))
        print('pilot central', row, flush=True)
    net.save_weights(out / 'best.weights.h5')
    save(out / 'selected.json', dict(seed=42, epochs=15, train_examples=len(train), excluded_group='V_6', initial_core_sha256=initial, checkpoint_sha256=digest(out / 'best.weights.h5'), seconds=time.perf_counter() - start))
    del net, core
    keras.backend.clear_session()
    gc.collect()

def extract(name, checkpoint, rows, indices, seed):
    out = HERE / 'embeddings' / name
    out.mkdir(parents=True, exist_ok=True)
    h = digest(checkpoint)
    identity = dict(checkpoint_sha256=h, row_indices=indices.tolist(), feature_cache_sha256=load(HERE / 'protocol.json')['cache_files_sha256']['multi.npy'], embedding='bigru2 output64; training=False', logits='dense64relu then classes.kernel+bias before softmax')
    if _resolve_model_artifact(out / 'identity.json').exists():
        assert load(out / 'identity.json') == identity
    else:
        save(out / 'identity.json', identity)
    if _resolve_model_artifact(out / 'complete.json').exists():
        for file, expected in load(out / 'complete.json')['files'].items():
            assert digest(out / file) == expected
        return
    n = len(rows)

    def array(file, shape, dtype):
        path = out / file
        if _resolve_model_artifact(path).exists():
            return _load_model_result_array(path, mmap_mode='r+')
        a = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
        a[:] = 0
        a.flush()
        return a
    e = array('embeddings.npy', (n, 9, 64), 'float32')
    z = array('logits.npy', (n, 9, 3), 'float32')
    done = array('done.npy', (n,), bool)
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    net, core, _ = build('fixed')
    net.load_weights(_resolve_model_artifact(checkpoint))
    net.trainable = False
    encoder = keras.Model(core.inputs, core.get_layer('bigru2').output)
    dense = core.get_layer('dense')
    clf = core.get_layer('classes')

    @tf.function(input_signature=[tf.TensorSpec([None, 8, 9, 2048], tf.float32), tf.TensorSpec([None, 8], tf.bool)])
    def encode(x, mask):
        flat = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [-1, 8, 2048])
        valid = tf.repeat(mask, 9, axis=0)
        emb = encoder([flat, valid], training=False)
        hidden = dense(emb)
        logits = tf.matmul(hidden, clf.kernel) + clf.bias
        return (tf.reshape(emb, [-1, 9, 64]), tf.reshape(logits, [-1, 9, 3]))
    f = _load_model_result_array(CACHE / 'features/multi.npy', mmap_mode='r')
    m = _load_model_result_array(CACHE / 'features/masks.npy', mmap_mode='r')
    t = time.perf_counter()
    processed = 0
    checked = False
    for off in range(0, len(indices), 32):
        idx = indices[off:off + 32]
        idx = idx[~done[idx]]
        if not len(idx):
            continue
        x = np.asarray(f[idx])
        mask = np.asarray(m[idx])
        ee, zz = encode(x, mask)
        if not checked:
            np.testing.assert_allclose(tf.nn.softmax(zz[:, 0]), net([x[:, :, 0], mask], training=False), atol=1e-06, rtol=1e-05)
            checked = True
        assert np.isfinite(ee).all() and np.isfinite(zz).all()
        e[idx] = ee.numpy()
        z[idx] = zz.numpy()
        e.flush()
        z.flush()
        done[idx] = True
        done.flush()
        processed += len(idx)
        save(HERE / 'status.json', dict(phase='frozen_embeddings', source=name, completed=int(done[indices].sum()), total=len(indices)))
    if name.startswith('seed_'):
        test = np.array([i for i, r in enumerate(rows) if r['domain'] == 'VF'])
        np.testing.assert_allclose(tf.nn.softmax(np.asarray(z[test, 0])), _load_model_result_array(CACHE / f'seed_{seed}/fixed/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
    assert done[indices].all() and digest(checkpoint) == h
    previous = load(out / 'timing.json') if _resolve_model_artifact(out / 'timing.json').exists() else {'runs': []}
    previous['runs'].append(dict(seconds=time.perf_counter() - t, processed_rows=processed))
    save(out / 'timing.json', previous)
    save(out / 'complete.json', dict(passed=True, rows=len(indices), identity_sha256=digest(out / 'identity.json'), files={name: digest(out / name) for name in ['embeddings.npy', 'logits.npy', 'done.npy']}, seconds_this_session=time.perf_counter() - t))
    del net, core, encoder
    keras.backend.clear_session()
    gc.collect()

def main():
    with writer_lock('extraction.lock'):
        prepare_references()
        rows = load(HERE / 'manifest.json')
        train, val, test = split_indices(rows)
        pilot_central(rows, train)
        extract('pilot', HERE / 'pilot/central/best.weights.h5', rows, np.sort(np.concatenate([train, val])), 42)
        for seed in SEEDS:
            extract(f'seed_{seed}', HERE / f'seed_{seed}/central/best.weights.h5', rows, np.arange(len(rows)), seed)
        save(HERE / 'prepared.json', dict(passed=True, pilot_train=len(train), pilot_validation=len(val), final_train=len(train) + len(val), vf=len(test)))
if __name__ == '__main__':
    main()
