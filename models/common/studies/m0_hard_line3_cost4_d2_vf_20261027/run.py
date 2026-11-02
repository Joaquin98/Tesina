import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import os, sys, time, json, gc, traceback
from pathlib import Path
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', TF_NUM_INTRAOP_THREADS='4', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='4')
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__)
DATA = ROOT / 'data/multicrop_d2_stride25_vf_20261026'
BASE = ROOT / 'data/central_recall_cost_d2_vf_20261027'
OOF = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
sys.path.insert(0, str(source_directory(BASE)))
from model import build, core_hash, keras
from common import digest
from evaluate import bootstrap
from hard import sample_weights
LABELS = ['line', 'other', 'scrum']
SEEDS = [17, 42, 2026]

def save(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(temp, path)

def status(**kw):
    save(HERE / 'status.json', dict(time=time.time(), **kw))
    print(json.dumps(kw), flush=True)

def metrics(y, p):
    pr, re, f, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    d = dict(accuracy=float(accuracy_score(y, p)), f1_macro=float(f.mean()))
    for i, k in enumerate(LABELS):
        d.update({k + '_precision': float(pr[i]), k + '_recall': float(re[i]), k + '_f1': float(f[i]), k + '_tp': int(cm[i, i]), k + '_fn': int(cm[i].sum() - cm[i, i]), k + '_fp': int(cm[:, i].sum() - cm[i, i])})
    return d

def main():
    start = time.time()
    assert json.loads(_resolve_model_artifact(OOF / 'completion.json').read_text())['complete']
    rows = json.loads(_resolve_model_artifact(DATA / 'manifest.json').read_text())
    tr = np.array([i for i, r in enumerate(rows) if r['domain'] != 'VF'])
    test = np.array([i for i, r in enumerate(rows) if r['domain'] == 'VF'])
    assert len(tr) == 8796 and len(test) == 688
    np.testing.assert_array_equal(tr, _load_model_result_array(OOF / 'train_indices.npy'))
    fold = _load_model_result_array(OOF / 'folds.npy')
    assert len(fold) == len(tr)
    groups = [r['domain'] + ':' + (r['group_id'] if r['domain'] == 'CV' else r['source_sha256']) for r in [rows[i] for i in tr]]
    group_to_fold = {}
    for g, f in zip(groups, fold):
        if g in group_to_fold:
            assert group_to_fold[g] == f
        else:
            group_to_fold[g] = f
    labels = np.array([LABELS.index(r['tag']) for r in rows])
    y = labels[tr]
    inputs = [_resolve_model_artifact(HERE / 'run.py'), _resolve_model_artifact(HERE / 'hard.py'), _resolve_model_artifact(BASE / 'model.py'), _resolve_model_artifact(BASE / 'evaluate.py'), DATA / 'manifest.json', DATA / 'features/multi.npy', DATA / 'features/masks.npy', OOF / 'folds.npy', OOF / 'train_indices.npy']
    for s in SEEDS:
        inputs.extend([OOF / f'seed_{s}/oof.npy', BASE / f'seed_{s}/cost_4/best.weights.h5', BASE / f'seed_{s}/cost_4/vf_probabilities.npy', ROOT / f'data/m0_hard_line_cost4_d2_vf_20261027/seed_{s}/vf_probabilities.npy'])
        for k in range(3):
            part = OOF / f'seed_{s}/fold_{k}'
            info = json.loads(_resolve_model_artifact(part / 'complete.json').read_text())
            np.testing.assert_array_equal(_load_model_result_array(part / 'indices.npy'), tr[fold == k])
            assert digest(part / 'model.weights.h5') == info['checkpoint_sha256']
            assert digest(part / 'pred.npy') == info['prediction_sha256']
            assert digest(part / 'indices.npy') == info['indices_sha256']
            inputs.extend([part / 'indices.npy', part / 'pred.npy', part / 'complete.json'])
    hashes = {str(p): digest(p) for p in inputs}
    save(HERE / 'protocol.json', dict(seed=SEEDS, cost=4, hard_multiplier=3, epochs=15, selection='OOF line argmax != line, no VF', labels=LABELS, hashes=hashes, adaptive_exploratory=True))
    features = _load_model_result_array(DATA / 'features/multi.npy', mmap_mode='r')
    mask = _load_model_result_array(DATA / 'features/masks.npy', mmap_mode='r')
    x = np.asarray(features[tr, :, 0])
    m = np.asarray(mask[tr])
    infos = []
    for seed in SEEDS:
        out = HERE / f'seed_{seed}'
        out.mkdir(exist_ok=True)
        oof = _load_model_result_array(OOF / f'seed_{seed}/oof.npy')
        assert oof.shape == (len(tr), 9, 3)
        for k in range(3):
            np.testing.assert_array_equal(oof[fold == k], _load_model_result_array(OOF / f'seed_{seed}/fold_{k}/pred.npy'))
        pred = oof[:, 0].argmax(1)
        w, hard = sample_weights(y, pred)
        np.save(out / 'sample_weights.npy', w)
        np.save(out / 'hard_mask.npy', hard)
        keras.backend.clear_session()
        gc.collect()
        keras.utils.set_random_seed(seed)
        net, core, _ = build('fixed')
        initial = core_hash(core)
        assert initial == json.loads(_resolve_model_artifact(DATA / f'seed_{seed}/fixed/selected.json').read_text())['initial_core_sha256']
        rng = np.random.default_rng(seed)
        history = []
        t = time.perf_counter()
        for epoch in range(1, 16):
            order = rng.permutation(len(tr))
            net.reset_metrics()
            epoch_start = time.perf_counter()
            for off in range(0, len(tr), 32):
                b = order[off:off + 32]
                net.train_on_batch([x[b], m[b]], y[b], sample_weight=w[b])
            history.append(dict(epoch=epoch, seconds=time.perf_counter() - epoch_start))
            pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
            status(phase='training', seed=seed, epoch=epoch, hard_line=int(hard.sum()))
        net.save_weights(out / 'model.weights.h5')
        info = dict(seed=seed, hard_line=int(hard.sum()), hard_as_other=int(((y == 0) & (pred == 1)).sum()), hard_as_scrum=int(((y == 0) & (pred == 2)).sum()), line_count=int((y == 0).sum()), checkpoint_sha256=digest(out / 'model.weights.h5'), initial_core_sha256=initial, seconds=time.perf_counter() - t, saved_unix=time.time())
        save(out / 'selected.json', info)
        infos.append(info)
    save(HERE / 'training_complete.json', dict(complete=True, saved_before_vf=True, completed_unix=time.time(), models=infos))
    f = np.asarray(features[test, :, 0])
    mm = np.asarray(mask[test])
    truth = labels[test]
    frame = pd.DataFrame([rows[i] for i in test])
    preds = {'central': [], 'hard_line': []}
    results = []
    for seed in SEEDS:
        out = HERE / f'seed_{seed}'
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        net, _, _ = build('fixed')
        net.load_weights(_resolve_model_artifact(out / 'model.weights.h5'))

        @tf.function(input_signature=[tf.TensorSpec([None, 8, 2048], tf.float32), tf.TensorSpec([None, 8], tf.bool)])
        def infer(xx, mask):
            return net([xx, mask], training=False)
        prob = np.concatenate([infer(f[i:i + 128], mm[i:i + 128]).numpy() for i in range(0, len(f), 128)])
        np.save(out / 'vf_probabilities.npy', prob)
        base = _load_model_result_array(BASE / f'seed_{seed}/cost_4/vf_probabilities.npy')
        np.testing.assert_allclose(base.sum(1), 1, atol=1e-06)
        np.testing.assert_allclose(prob.sum(1), 1, atol=1e-06)
        for mode, p in [('central', base), ('hard_line', prob)]:
            predicted = p.argmax(1)
            preds[mode].append(predicted)
            results.append(dict(seed=seed, variant=mode, **metrics(truth, predicted)))
            frame[f'{mode}_{seed}'] = [LABELS[j] for j in predicted]
    pd.DataFrame(results).to_csv(HERE / 'results.csv', index=False)
    frame.to_csv(HERE / 'predictions.csv', index=False)
    ci = bootstrap(truth, preds['central'], preds['hard_line'], frame.section, frame.start_s)
    save(HERE / 'paired_bootstrap.json', ci)
    for path, h in hashes.items():
        assert digest(path) == h, path
    save(HERE / 'completion.json', dict(complete=True, all_weights_saved_before_vf=True, seconds=time.time() - start, models=infos, means=pd.DataFrame(results).groupby('variant').mean(numeric_only=True).to_dict(orient='index'), limitations='VF previously observed; exploratory, CH provenance incomplete; mean of seeds not ensemble.'))
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
