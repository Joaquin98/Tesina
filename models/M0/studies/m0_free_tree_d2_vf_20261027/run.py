import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json, hashlib, sys, joblib
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
H = experiment_directory(__file__)
P = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/m0_oof_fusion_cost4_d2_vf_20261027')
R = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/m0_protect_scrum_rescue_line_d2_vf_20261027')
M = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/multicrop_d2_stride25_vf_20261026/manifest.json')
SEEDS = [17, 42, 2026]
LABELS = ['line', 'other', 'scrum']

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()

def save(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False), encoding='utf8')

def fit():
    assert json.loads(_resolve_model_artifact(P / 'completion.json').read_text())['complete']
    assert not _resolve_model_artifact(H / 'trained.json').exists()
    data = json.loads(_resolve_model_artifact(M).read_text(encoding='utf8'))
    ix = _load_model_result_array(P / 'train_indices.npy')
    fold = _load_model_result_array(P / 'folds.npy')
    rows = [data[i] for i in ix]
    assert len(rows) == 8796 and all((x['domain'] != 'VF' for x in rows))
    labels = np.array([LABELS.index(x['tag']) for x in rows])
    counts = np.bincount(labels, minlength=3)
    weights = len(labels) / (3 * counts) * np.array([4, 1, 4])
    weights /= np.average(weights, weights=counts)
    hashes = {str(M): sha(M), str(P / 'train_indices.npy'): sha(P / 'train_indices.npy'), str(P / 'folds.npy'): sha(P / 'folds.npy'), str(_resolve_model_artifact(H / 'run.py')): sha(_resolve_model_artifact(H / 'run.py'))}
    models = {}
    for seed in SEEDS:
        f = P / f'seed_{seed}/oof.npy'
        a = _load_model_result_array(f)
        assert a.shape == (8796, 9, 3)
        hashes[str(f)] = sha(f)
        for k in range(3):
            d = P / f'seed_{seed}/fold_{k}'
            m = json.loads(_resolve_model_artifact(d / 'complete.json').read_text())
            assert sha(d / 'pred.npy') == m['prediction_sha256'] and sha(d / 'indices.npy') == m['indices_sha256']
            assert np.array_equal(_load_model_result_array(d / 'indices.npy'), ix[fold == k]) and np.array_equal(a[fold == k], _load_model_result_array(d / 'pred.npy'))
        model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, class_weight={i: float(w) for i, w in enumerate(weights)}, random_state=seed)
        model.fit(a.reshape(-1, 27), labels)
        f = H / f'tree_{seed}.joblib'
        joblib.dump(model, f)
        models[str(seed)] = {'sha256': sha(f), 'depth': model.get_depth(), 'leaves': model.get_n_leaves()}
        (H / f'tree_{seed}.txt').write_text(export_text(model, feature_names=[f'crop{v}_{c}' for v in range(9) for c in LABELS]), encoding='utf8')
    save(H / 'protocol.json', {'training': '8796 D2 predictions out-of-fold; features 27 per-crop class probabilities; 3 seeds', 'tree': 'free multiclass decision tree; maximum depth 4, minimum 50 samples per leaf; inverse-frequency class weights with line and scrum multiplied by 4, fixed before VF', 'decision': 'argmax leaf weighted class distribution; no external thresholds; may override any central class', 'evaluation': 'existing 688 VF blocks, exploration only; VF previously observed repeatedly', 'model_2_3_vs_full_D2': 'OOF source model trained on two thirds of D2; VF features from full D2 model', 'input_hashes': hashes, 'weights': weights.tolist()})
    save(H / 'trained.json', {'trained_before_vf': True, 'models': models})

def stat(y, p):
    pr, rc, f, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    d = {'accuracy': accuracy_score(y, p), 'macro_f1': f1_score(y, p, labels=[0, 1, 2], average='macro', zero_division=0)}
    for i, n in enumerate(LABELS):
        d.update({n + '_precision': pr[i], n + '_recall': rc[i], n + '_f1': f[i], n + '_tp': int(((y == i) & (p == i)).sum()), n + '_fp': int(((y != i) & (p == i)).sum()), n + '_fn': int(((y == i) & (p != i)).sum())})
    return d

def evaluate():
    tr = json.loads(_resolve_model_artifact(H / 'trained.json').read_text())
    assert tr['trained_before_vf']
    for s in SEEDS:
        assert sha(H / f'tree_{s}.joblib') == tr['models'][str(s)]['sha256']
    prev = pd.read_csv(_resolve_model_artifact(R / 'vf_predictions.csv'))
    records = []
    predrows = []
    arrays = {}
    frame = None
    for seed in SEEDS:
        q = prev[prev.seed == seed].reset_index(drop=True)
        assert len(q) == 688
        v = _load_model_result_array(P / f'seed_{seed}/vf_views.npy')
        assert v.shape == (688, 9, 3)
        central = v[:, 0].argmax(1)
        assert np.array_equal(central, q.central)
        model = joblib.load(_resolve_model_artifact(H / f'tree_{seed}.joblib'))
        tree = model.predict(v.reshape(-1, 27))
        y = np.array([LABELS.index(x) for x in q.tag])
        rule = q['t_0.9'].to_numpy()
        assert np.array_equal(tree == 2, np.argmax(model.predict_proba(v.reshape(-1, 27)), axis=1) == 2)
        for name, pred in [('central', central), ('regla_90', rule), ('arbol', tree)]:
            records.append({'seed': seed, 'model': name, **stat(y, pred)})
        q = q[['clip_id', 'section', 'start_s', 'tag']].copy()
        q['seed'] = seed
        q['central'] = central
        q['regla_90'] = rule
        q['arbol'] = tree
        predrows.append(q)
        arrays[seed] = (y, central, rule, tree)
        frame = q
    pd.DataFrame(records).to_csv(H / 'results.csv', index=False)
    pd.concat(predrows).to_csv(H / 'predictions.csv', index=False)
    groups = []
    sec = frame.section.to_numpy()
    bins = np.floor(frame.start_s.to_numpy() / 60).astype(int)
    for s in np.unique(sec):
        groups.append([np.flatnonzero((sec == s) & (bins == b)) for b in np.unique(bins[sec == s])])

    def quick(y, p):
        cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
        tp = np.diag(cm)
        pre = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) != 0)
        rec = np.divide(tp, cm.sum(1), out=np.zeros(3), where=cm.sum(1) != 0)
        f = np.divide(2 * pre * rec, pre + rec, out=np.zeros(3), where=pre + rec != 0)
        return np.array([pre[0], rec[0], f[0], pre[2], rec[2], f[2], f.mean(), tp.sum() / len(y)])
    names = ['line_precision', 'line_recall', 'line_f1', 'scrum_precision', 'scrum_recall', 'scrum_f1', 'macro_f1', 'accuracy']
    rng = np.random.default_rng(20261027)
    allv = {1: [], 2: []}
    for _ in range(2000):
        idx = np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g), size=len(g))]) for g in groups])
        y = arrays[17][0][idx]
        if len(np.unique(y)) < 3:
            continue
        for b in [1, 2]:
            allv[b].append(np.mean([quick(a[0][idx], a[3][idx]) - quick(a[0][idx], a[b][idx]) for a in arrays.values()], axis=0))
    boot = {}
    for b, label in [(1, 'central'), (2, 'regla_90')]:
        a = np.array(allv[b])
        point = np.mean([quick(x[0], x[3]) - quick(x[0], x[b]) for x in arrays.values()], axis=0)
        boot[label] = {n: {'difference': float(point[i]), 'ci95': np.quantile(a[:, i], [0.025, 0.975]).tolist()} for i, n in enumerate(names)}
    save(H / 'bootstrap.json', {'method': '2000 paired resamples of 60s blocks stratified by section, mean of 3 seeds, descriptive', 'comparisons': boot})
    save(H / 'completion.json', {'complete': True, 'models': 3, 'vf_blocks': 688, 'trained_before_reading_vf': True})
if __name__ == '__main__':
    {'fit': fit, 'evaluate': evaluate}[sys.argv[1]]()
