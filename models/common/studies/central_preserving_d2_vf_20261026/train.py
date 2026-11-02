"""Train all final heads before any new head is evaluated on VF."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import gc, time
import pandas as pd
from common import *
from analysis_math import split_indices, metrics, changes
from model import build, protection_loss, keras, tf
STRATEGIES = ['residual', 'late_mix', 'protected_residual']

def train_head(strategy, seed, rows, indices, cache, out):
    out.mkdir(parents=True, exist_ok=True)
    if _resolve_model_artifact(out / 'selected.json').exists():
        assert digest(out / 'best.weights.h5') == load(out / 'selected.json')['checkpoint_sha256']
        return
    assert all((rows[i]['domain'] != 'VF' for i in indices))
    e = _load_model_result_array(cache / 'embeddings.npy', mmap_mode='r')
    z = _load_model_result_array(cache / 'logits.npy', mmap_mode='r')
    assert _load_model_result_array(cache / 'done.npy')[indices].all()
    keras.backend.clear_session()
    gc.collect()
    keras.utils.set_random_seed(seed)
    net, _ = build(strategy)
    optimizer = keras.optimizers.Adam(0.001)
    y = np.array([LABELS.index(rows[i]['tag']) for i in indices], dtype='int32')
    counts = np.bincount(y, minlength=3)
    cw = tf.constant(len(y) / (3 * counts), dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec([None, 9, 64], tf.float32), tf.TensorSpec([None, 9, 3], tf.float32), tf.TensorSpec([None], tf.int32)])
    def step(ee, zz, yy):
        with tf.GradientTape() as tape:
            p = net([ee, zz], training=True)
            ce = keras.losses.sparse_categorical_crossentropy(yy, p)
            classification = tf.reduce_mean(ce * tf.gather(cw, yy))
            protect = protection_loss(yy, tf.nn.softmax(zz[:, 0]), p) if strategy == 'protected_residual' else tf.constant(0.0)
            loss = classification + protect
        grads = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return (loss, classification, protect, tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p, 1, output_type=tf.int32), yy), tf.float32)))
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    history = []
    for epoch in range(1, 16):
        t = time.perf_counter()
        order = rng.permutation(len(indices))
        tot = np.zeros(4)
        for off in range(0, len(order), 64):
            pos = order[off:off + 64]
            idx = indices[pos]
            v = step(np.asarray(e[idx]), np.asarray(z[idx]), y[pos])
            tot += len(pos) * np.array([float(k) for k in v])
        tot /= len(indices)
        row = dict(epoch=epoch, loss=tot[0], classification=tot[1], protection=tot[2], accuracy=tot[3], seconds=time.perf_counter() - t)
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        save(HERE / 'status.json', dict(phase='head_training', scope='pilot' if cache.name == 'pilot' else 'D2', seed=seed, strategy=strategy, epochs_total=15, **row))
        print(out.relative_to(HERE), row, flush=True)
    net.save_weights(out / 'best.weights.h5')
    save(out / 'selected.json', dict(seed=seed, strategy=strategy, epochs=15, training_examples=len(indices), parameters=net.count_params(), checkpoint_sha256=digest(out / 'best.weights.h5'), embedding_identity_sha256=digest(cache / 'identity.json'), seconds=time.perf_counter() - start, selection='fixed epoch15; no validation selection'))

def predict(strategy, out, cache, indices):
    keras.backend.clear_session()
    net, probe = build(strategy)
    net.load_weights(_resolve_model_artifact(out / 'best.weights.h5'))
    e = _load_model_result_array(cache / 'embeddings.npy', mmap_mode='r')
    z = _load_model_result_array(cache / 'logits.npy', mmap_mode='r')
    pp = []
    aa = []
    for off in range(0, len(indices), 128):
        ix = indices[off:off + 128]
        inputs = [np.asarray(e[ix]), np.asarray(z[ix])]
        pp.append(net(inputs, training=False).numpy())
        aa.append(probe(inputs, training=False).numpy())
    return (np.concatenate(pp), np.concatenate(aa))

def main():
    with writer_lock('training.lock'):
        check_frozen()
        assert load(HERE / 'prepared.json')['passed']
        rows = load(HERE / 'manifest.json')
        train, val, test = split_indices(rows)
        for strategy in STRATEGIES:
            train_head(strategy, 42, rows, train, HERE / 'embeddings/pilot', HERE / 'pilot' / strategy)
        y = np.array([LABELS.index(rows[i]['tag']) for i in val])
        base = tf.nn.softmax(_load_model_result_array(HERE / 'embeddings/pilot/logits.npy', mmap_mode='r')[val, 0]).numpy()
        records = [dict(strategy='central', **metrics(y, base.argmax(1)))]
        np.save(HERE / 'pilot/central/validation_probabilities.npy', base)
        for strategy in STRATEGIES:
            out = HERE / 'pilot' / strategy
            p, a = predict(strategy, out, HERE / 'embeddings/pilot', val)
            np.save(out / 'validation_probabilities.npy', p)
            np.save(out / 'validation_attention.npy', a)
            records.append(dict(strategy=strategy, **metrics(y, p.argmax(1))))
            save(out / 'validation_changes.json', changes(y, base.argmax(1), p.argmax(1)))
        pd.DataFrame(records).to_csv(HERE / 'pilot/results.csv', index=False)
        save(HERE / 'pilot/validation.json', dict(complete=True, group='V_6', blocks=len(val), no_selection_or_tuning=True, unknown_CH_provenance=True))
        alltrain = np.sort(np.concatenate([train, val]))
        for seed in SEEDS:
            for strategy in STRATEGIES:
                train_head(strategy, seed, rows, alltrain, HERE / f'embeddings/seed_{seed}', HERE / f'seed_{seed}' / strategy)
        selections = [load(HERE / f'seed_{s}/{m}/selected.json') for s in SEEDS for m in STRATEGIES]
        save(HERE / 'selected_models.json', selections)
        save(HERE / 'training_complete.json', dict(complete=True, new_final_heads=9, pilot_heads=3, pilot_central=1, reused_final_centrals=3, all_saved_before_vf=True))
if __name__ == '__main__':
    main()
