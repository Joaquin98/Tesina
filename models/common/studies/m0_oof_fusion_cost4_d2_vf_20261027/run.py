import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import os, sys, time, json, hashlib, gc, traceback
from pathlib import Path
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', TF_NUM_INTRAOP_THREADS='4', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='4')
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__)
PARENT = ROOT / 'data/multicrop_d2_stride25_vf_20261026'
BASE = ROOT / 'data/central_recall_cost_d2_vf_20261027'
sys.path.insert(0, str(source_directory(BASE)))
from model import build, keras
from common import lock, digest
from fusion import fold_ids, weights
LABELS = ['line', 'other', 'scrum']
SEEDS = [17, 42, 2026]

def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)

def status(**data):
    save(HERE / 'status.json', dict(time=time.time(), **data))
    print(json.dumps(data), flush=True)

def view_predict(net, features, mask, idx):

    @tf.function(input_signature=[tf.TensorSpec([None, 8, 2048], tf.float32), tf.TensorSpec([None, 8], tf.bool)])
    def infer(x, m):
        return net([x, m], training=False)
    result = np.empty((len(idx), 9, 3), np.float32)
    for v in range(9):
        for i in range(0, len(idx), 128):
            pick = idx[i:i + 128]
            result[i:i + len(pick), v] = infer(np.asarray(features[pick, :, v, :]), np.asarray(mask[pick])).numpy()
    np.testing.assert_allclose(result.sum(axis=2), 1, atol=1e-05)
    return result

def fit_m0(seed, fold, tr, va, features, mask, y):
    out = HERE / f'seed_{seed}/fold_{fold}'
    out.mkdir(parents=True, exist_ok=True)
    if _resolve_model_artifact(out / 'complete.json').exists():
        info = json.loads(_resolve_model_artifact(out / 'complete.json').read_text())
        assert digest(out / 'pred.npy') == info['prediction_sha256'] and digest(out / 'model.weights.h5') == info['checkpoint_sha256']
        return _load_model_result_array(out / 'pred.npy')
    keras.backend.clear_session()
    gc.collect()
    keras.utils.set_random_seed(seed)
    net, _, _ = build('fixed')
    x = np.asarray(features[tr, :, 0, :])
    m = np.asarray(mask[tr])
    yy = y[tr]
    w = weights(yy)
    rng = np.random.default_rng(seed)
    for epoch in range(1, 16):
        order = rng.permutation(len(tr))
        net.reset_metrics()
        for off in range(0, len(tr), 32):
            b = order[off:off + 32]
            net.train_on_batch([x[b], m[b]], yy[b], class_weight={i: float(w[i]) for i in range(3)})
        status(phase='oof_training', seed=seed, fold=fold + 1, epoch=epoch, epochs=15)
    net.save_weights(out / 'model.weights.h5')
    status(phase='oof_inference', seed=seed, fold=fold + 1, views=9)
    pred = view_predict(net, features, mask, va)
    np.save(out / 'pred.npy', pred)
    np.save(out / 'indices.npy', va)
    save(out / 'complete.json', dict(checkpoint_sha256=digest(out / 'model.weights.h5'), prediction_sha256=digest(out / 'pred.npy'), indices_sha256=digest(out / 'indices.npy'), train_count=len(tr), held_out_count=len(va)))
    return pred

def head(seed, oof, y):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    inp = keras.Input((9, 3))
    flat = keras.layers.Flatten()(inp)
    delta = keras.layers.Dense(3, kernel_initializer='zeros', bias_initializer='zeros')(flat)

    class Residual(keras.layers.Layer):

        def call(self, inputs):
            p, d = inputs
            return tf.nn.softmax(tf.math.log(tf.maximum(p[:, 0, :], 1e-07)) + d, axis=-1)
    pred = Residual()([inp, delta])
    net = keras.Model(inp, pred)
    net.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy')
    np.testing.assert_allclose(net(oof[:10], training=False), oof[:10, 0], atol=1e-06)
    rng = np.random.default_rng(seed)
    w = weights(y)
    for epoch in range(1, 16):
        order = rng.permutation(len(y))
        for off in range(0, len(y), 32):
            b = order[off:off + 32]
            net.train_on_batch(oof[b], y[b], class_weight={i: float(w[i]) for i in range(3)})
        status(phase='fusion_training', seed=seed, epoch=epoch, epochs=15)
    out = HERE / f'seed_{seed}'
    net.save_weights(out / 'fusion.weights.h5')
    return net

def main():
    began = time.time()
    rows = json.loads(_resolve_model_artifact(PARENT / 'manifest.json').read_text())
    train = np.array([i for i, r in enumerate(rows) if r['domain'] != 'VF'])
    test = np.array([i for i, r in enumerate(rows) if r['domain'] == 'VF'])
    assert len(train) == 8796 and len(test) == 688
    labels = np.array([LABELS.index(r['tag']) for r in rows])
    y = labels[train]
    groups = np.array([r['domain'] + ':' + (r['group_id'] if r['domain'] == 'CV' else r['source_sha256']) for r in np.array(rows, dtype=object)[train]])
    fold = fold_ids(y, groups)
    np.save(HERE / 'folds.npy', fold)
    np.save(HERE / 'train_indices.npy', train)
    features = _load_model_result_array(PARENT / 'features/multi.npy', mmap_mode='r')
    mask = _load_model_result_array(PARENT / 'features/masks.npy', mmap_mode='r')
    assert _load_model_result_array(PARENT / 'features/done.npy').all()
    hashes = {str(p): digest(p) for p in [PARENT / 'manifest.json', PARENT / 'features/multi.npy', PARENT / 'features/masks.npy', _resolve_model_artifact(BASE / 'model.py'), BASE / 'results.csv'] + [BASE / f'seed_{s}/cost_4/best.weights.h5' for s in SEEDS]}
    save(HERE / 'protocol.json', dict(cost=4, seeds=SEEDS, epochs=15, folds=3, split='CV by match, CH by source hash', rows=8796, vf=688, decision='argmax without thresholds', input_hashes=hashes, adaptive_exploratory=True, started=began))
    nets = {}
    for seed in SEEDS:
        oof = np.full((len(train), 9, 3), np.nan, np.float32)
        for k in range(3):
            a = np.flatnonzero(fold != k)
            b = np.flatnonzero(fold == k)
            oof[b] = fit_m0(seed, k, train[a], train[b], features, mask, labels)
        assert np.isfinite(oof).all()
        np.save(HERE / f'seed_{seed}/oof.npy', oof)
        nets[seed] = head(seed, oof, y)
    save(HERE / 'training_complete.json', dict(complete=True, saved_before_vf=True, time=time.time(), fusion_hashes={str(s): digest(HERE / f'seed_{s}/fusion.weights.h5') for s in SEEDS}))
    result = []
    predsets = {v: [] for v in ['central', 'fusion']}
    frame = pd.DataFrame([rows[i] for i in test])
    truth = labels[test]
    for seed in SEEDS:
        status(phase='vf_inference', seed=seed)
        keras.utils.set_random_seed(seed)
        net, _, _ = build('fixed')
        net.load_weights(_resolve_model_artifact(BASE / f'seed_{seed}/cost_4/best.weights.h5'))
        p = view_predict(net, features, mask, test)
        np.testing.assert_allclose(p[:, 0], _load_model_result_array(BASE / f'seed_{seed}/cost_4/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
        fused = nets[seed](p, training=False).numpy()
        np.save(HERE / f'seed_{seed}/vf_views.npy', p)
        np.save(HERE / f'seed_{seed}/vf_fusion.npy', fused)
        for name, prob in [('central', p[:, 0]), ('fusion', fused)]:
            pred = prob.argmax(1)
            pr, re, f, sup = precision_recall_fscore_support(truth, pred, labels=[0, 1, 2], zero_division=0)
            cm = confusion_matrix(truth, pred, labels=[0, 1, 2])
            d = dict(seed=seed, variant=name, accuracy=float(accuracy_score(truth, pred)), f1_macro=float(f.mean()))
            for j, k in enumerate(LABELS):
                d.update({k + '_precision': float(pr[j]), k + '_recall': float(re[j]), k + '_f1': float(f[j]), k + '_tp': int(cm[j, j]), k + '_fn': int(cm[j].sum() - cm[j, j]), k + '_fp': int(cm[:, j].sum() - cm[j, j])})
            result.append(d)
            predsets[name].append(pred)
        frame[f'central_{seed}'] = [LABELS[i] for i in p[:, 0].argmax(1)]
        frame[f'fusion_{seed}'] = [LABELS[i] for i in fused.argmax(1)]
    pd.DataFrame(result).to_csv(HERE / 'results.csv', index=False)
    frame.to_csv(HERE / 'vf_predictions.csv', index=False)
    groups60 = []
    sections = frame.section.to_numpy()
    bins = (frame.start_s.to_numpy() // 60).astype(int)
    for s in np.unique(sections):
        groups60.append([np.flatnonzero((sections == s) & (bins == b)) for b in np.unique(bins[sections == s])])

    def metric(t, p):
        pr, re, f, _ = precision_recall_fscore_support(t, p, labels=[0, 1, 2], zero_division=0)
        return np.r_[pr[[0, 2]], re[[0, 2]], f[[0, 2]], f.mean(), accuracy_score(t, p)]
    rng = np.random.default_rng(20261027)
    deltas = []
    for rep in range(2000):
        ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g), size=len(g))]) for g in groups60])
        if any((np.sum(truth[ix] == j) == 0 for j in range(3))):
            continue
        deltas.append(np.mean([metric(truth[ix], n[ix]) - metric(truth[ix], b[ix]) for b, n in zip(predsets['central'], predsets['fusion'])], axis=0))
    keys = ['line_precision', 'scrum_precision', 'line_recall', 'scrum_recall', 'line_f1', 'scrum_f1', 'f1_macro', 'accuracy']
    point = np.mean([metric(truth, n) - metric(truth, b) for b, n in zip(predsets['central'], predsets['fusion'])], axis=0)
    ci = {k: dict(difference=float(point[j]), ci95=np.quantile(np.array(deltas)[:, j], [0.025, 0.975]).tolist()) for j, k in enumerate(keys)}
    save(HERE / 'paired_bootstrap.json', dict(method='paired 60s clusters by section; exploratory conditional on fitted models', valid=len(deltas), metrics=ci))
    for path, h in hashes.items():
        assert digest(path) == h, path
    save(HERE / 'verification.json', dict(passed=True, central_probabilities_reproduced=True, oof_group_partition=True, all_fusion_saved_before_vf=True, metrics_sklearn=True, hashes_unchanged=True))
    save(HERE / 'completion.json', dict(complete=True, seconds=time.time() - began, results=pd.DataFrame(result).groupby('variant').mean(numeric_only=True).to_dict(orient='index'), note='VF reused; exploratory. Means across seeds not an ensemble. OOF models see 2/3 of D2; full model sees all D2.'))
    status(phase='complete')
if __name__ == '__main__':
    try:
        import msvcrt
        with _resolve_model_artifact(HERE / 'run.lock').open('a+b') as f:
            f.seek(0, 2)
            if f.tell() == 0:
                f.write(b'0')
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            main()
    except Exception as e:
        save(HERE / 'error.json', dict(error=repr(e), traceback=traceback.format_exc()))
        raise
