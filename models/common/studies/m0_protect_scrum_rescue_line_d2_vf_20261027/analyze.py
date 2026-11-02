import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from rule import decide
H = experiment_directory(__file__)
P = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/m0_oof_fusion_cost4_d2_vf_20261027')
C = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/central_recall_cost_d2_vf_20261027')
M = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/multicrop_d2_stride25_vf_20261026/manifest.json')
SEEDS = [17, 42, 2026]
LEVELS = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
LABELS = ['line', 'other', 'scrum']

def sha(path):
    h = hashlib.sha256()
    with open(_resolve_model_artifact(path), 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf8')

def stats(y, p, base):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    d = {'accuracy': accuracy_score(y, p), 'macro_f1': f1_score(y, p, labels=[0, 1, 2], average='macro', zero_division=0), 'changes': int((p != base).sum()), 'corrected': int(((base != y) & (p == y)).sum()), 'broken': int(((base == y) & (p != y)).sum())}
    for i, name in enumerate(LABELS):
        d.update({name + '_precision': pr[i], name + '_recall': rc[i], name + '_f1': f1[i], name + '_tp': int(((y == i) & (p == i)).sum()), name + '_fp': int(((y != i) & (p == i)).sum()), name + '_fn': int(((y == i) & (p != i)).sum())})
    return d

def assert_rule(p, base):
    assert np.array_equal(p == 2, base == 2)
    assert np.array_equal(p[base != 1], base[base != 1])
    assert np.all((p == base) | (base == 1) & (p == 0))

def fast(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    pr = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) != 0)
    rc = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) != 0)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(3), where=pr + rc != 0)
    return np.array([pr[0], rc[0], f[0], pr[2], rc[2], f[2], f.mean(), tp.sum() / len(y)])

def main():
    assert json.loads(_resolve_model_artifact(P / 'completion.json').read_text())['complete']
    assert not _resolve_model_artifact(H / 'protocol.json').exists(), 'already frozen; avoid overwriting'
    manifest = json.loads(_resolve_model_artifact(M).read_text(encoding='utf8'))
    idx = _load_model_result_array(P / 'train_indices.npy')
    fold = _load_model_result_array(P / 'folds.npy')
    assert len(idx) == 8796 and len(fold) == 8796
    rows = [manifest[i] for i in idx]
    assert all((r['domain'] != 'VF' for r in rows))
    keys = [('CH', r['source_sha256']) if r['domain'] == 'CH' else ('CV', r['group_id']) for r in rows]
    mapping = {}
    for k, f in zip(keys, fold):
        if k in mapping:
            assert mapping[k] == f
        mapping[k] = f
    y = np.array([LABELS.index(r['tag']) for r in rows])
    d2 = []
    hashes = {str(M): sha(M), str(P / 'folds.npy'): sha(P / 'folds.npy'), str(P / 'train_indices.npy'): sha(P / 'train_indices.npy'), str(_resolve_model_artifact(H / 'rule.py')): sha(_resolve_model_artifact(H / 'rule.py')), str(_resolve_model_artifact(H / 'analyze.py')): sha(_resolve_model_artifact(H / 'analyze.py'))}
    for seed in SEEDS:
        path = P / f'seed_{seed}/oof.npy'
        oof = _load_model_result_array(path)
        hashes[str(path)] = sha(path)
        assert oof.shape == (8796, 9, 3)
        for k in range(3):
            root = P / f'seed_{seed}/fold_{k}'
            meta = json.loads(_resolve_model_artifact(root / 'complete.json').read_text())
            for fname, key in [('pred.npy', 'prediction_sha256'), ('indices.npy', 'indices_sha256'), ('model.weights.h5', 'checkpoint_sha256')]:
                assert sha(root / fname) == meta[key]
            assert np.array_equal(_load_model_result_array(root / 'indices.npy'), idx[fold == k])
            assert np.array_equal(_load_model_result_array(root / 'pred.npy'), oof[fold == k])
        base = oof[:, 0].argmax(1)
        for t in [None] + LEVELS:
            pred = base if t is None else decide(oof, t)
            assert_rule(pred, base)
            d2.append({'seed': seed, 'threshold': 'central' if t is None else str(t), **stats(y, pred, base)})
    pd.DataFrame(d2).to_csv(H / 'd2_results.csv', index=False)
    save(H / 'protocol.json', {'rule': 'central scrum and line unchanged; central other changes to line only if at least two context crops individually vote line by argmax and each line probability >= threshold', 'thresholds': LEVELS, 'zero_threshold': 'no probability cutoff', 'seeds': SEEDS, 'training_evaluation': 'grouped D2 out-of-fold; context models trained on two folds', 'vf_evaluation': 'existing full D2 models; repeated VF use; descriptive only', 'selection': 'no threshold selected using VF', 'inputs_sha256_before_vf': hashes})
    frame = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
    assert len(frame) == 688
    y = np.array([LABELS.index(x) for x in frame.tag])
    records = []
    outputs = []
    allpred = {}
    bases = []
    for seed in SEEDS:
        path = P / f'seed_{seed}/vf_views.npy'
        prob = _load_model_result_array(path)
        assert prob.shape == (688, 9, 3)
        np.testing.assert_allclose(prob[:, 0], _load_model_result_array(C / f'seed_{seed}/cost_4/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
        hashes[str(path)] = sha(path)
        base = prob[:, 0].argmax(1)
        bases.append(base)
        out = frame[['clip_id', 'section', 'start_s', 'tag']].copy()
        out['seed'] = seed
        out['central'] = base
        for t in [None] + LEVELS:
            pred = base if t is None else decide(prob, t)
            assert_rule(pred, base)
            name = 'central' if t is None else str(t)
            records.append({'seed': seed, 'threshold': name, **stats(y, pred, base)})
            if t is not None:
                out['t_' + str(t)] = pred
                allpred.setdefault(str(t), []).append(pred)
        outputs.append(out)
    pd.DataFrame(records).to_csv(H / 'vf_results.csv', index=False)
    pd.concat(outputs).to_csv(H / 'vf_predictions.csv', index=False)
    groups = []
    sections = frame.section.to_numpy()
    bins = np.floor(frame.start_s.to_numpy() / 60).astype(int)
    for s in np.unique(sections):
        groups.append([np.flatnonzero((sections == s) & (bins == b)) for b in np.unique(bins[sections == s])])
    rng = np.random.default_rng(20261027)
    deltas = {str(t): [] for t in LEVELS}
    names = ['line_precision', 'line_recall', 'line_f1', 'scrum_precision', 'scrum_recall', 'scrum_f1', 'macro_f1', 'accuracy']
    for _ in range(2000):
        ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g), size=len(g))]) for g in groups])
        if len(np.unique(y[ix])) != 3:
            continue
        aa = np.stack([fast(y[ix], a[ix]) for a in bases])
        for t in LEVELS:
            pp = allpred[str(t)]
            bb = np.stack([fast(y[ix], p[ix]) for p in pp])
            deltas[str(t)].append((bb - aa).mean(axis=0))
    boot = {}
    for t in LEVELS:
        arr = np.asarray(deltas[str(t)])
        aa = [stats(y, b, b) for b in bases]
        bb = [stats(y, p, p) for p in allpred[str(t)]]
        boot[str(t)] = {n: {'difference': float(np.mean([b[n] - a[n] for a, b in zip(aa, bb)])), 'ci95': np.quantile(arr[:, i], [0.025, 0.975]).tolist()} for i, n in enumerate(names)}
    save(H / 'paired_bootstrap.json', {'method': '60-second blocks within section, 2000 paired resamples, mean metrics across seeds; descriptive', 'valid': len(arr), 'thresholds': boot})
    save(H / 'input_hashes.json', hashes)
if __name__ == '__main__':
    main()
