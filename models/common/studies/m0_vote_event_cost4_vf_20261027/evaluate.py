import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from priority import decide
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
HERE = experiment_directory(__file__)
PARENT = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
BASE = ROOT / 'data/central_recall_cost_d2_vf_20261027'
DATA = ROOT / 'data/multicrop_d2_stride25_vf_20261026'
LABELS = ['line', 'other', 'scrum']
SEEDS = [17, 42, 2026]

def metric(y, p):
    pr, re, f, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    d = dict(accuracy=float(accuracy_score(y, p)), f1_macro=float(f.mean()))
    for j, k in enumerate(LABELS):
        d.update({k + '_precision': float(pr[j]), k + '_recall': float(re[j]), k + '_f1': float(f[j]), k + '_tp': int(cm[j, j]), k + '_fn': int(cm[j].sum() - cm[j, j]), k + '_fp': int(cm[:, j].sum() - cm[j, j])})
    return d

def sha(path):
    return hashlib.sha256(_resolve_model_artifact(path).read_bytes()).hexdigest()

def main():
    assert json.loads(_resolve_model_artifact(PARENT / 'completion.json').read_text())['complete']
    rows = json.loads(_resolve_model_artifact(DATA / 'manifest.json').read_text())
    frame = pd.DataFrame([r for r in rows if r['domain'] == 'VF'])
    assert len(frame) == 688
    y = np.array([LABELS.index(t) for t in frame.tag])
    assert np.bincount(y).tolist() == [68, 581, 39]
    results = []
    records = []
    preds = {'central': [], 'priority': []}
    hashes = {}
    baseline = pd.read_csv(_resolve_model_artifact(BASE / 'results.csv'))
    for seed in SEEDS:
        path = PARENT / f'seed_{seed}/vf_views.npy'
        hashes[str(path)] = sha(path)
        p = _load_model_result_array(path)
        assert p.shape == (688, 9, 3)
        np.testing.assert_allclose(p[:, 0], _load_model_result_array(BASE / f'seed_{seed}/cost_4/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
        central = p[:, 0].argmax(1)
        priority, view = decide(p)
        per = p.argmax(2)
        for variant, pred in [('central', central), ('priority', priority)]:
            m = metric(y, pred)
            results.append(dict(seed=seed, variant=variant, **m))
            preds[variant].append(pred)
            if variant == 'central':
                saved = baseline[(baseline.seed == seed) & (baseline.variant == 'cost_4')].iloc[0]
                for k in m:
                    np.testing.assert_allclose(m[k], saved[k], atol=1e-12)
        for i in range(len(y)):
            records.append(dict(seed=seed, clip_id=frame.clip_id.iloc[i], section=frame.section.iloc[i], start_s=frame.start_s.iloc[i], y_true=LABELS[y[i]], central=LABELS[central[i]], priority=LABELS[priority[i]], line_votes=int((per[i] == 0).sum()), scrum_votes=int((per[i] == 2).sum()), other_votes=int((per[i] == 1).sum()), **{f'v{v}_{k}': float(p[i, v, j]) for v in range(9) for j, k in enumerate(LABELS)}))
    pd.DataFrame(results).to_csv(HERE / 'results.csv', index=False)
    pd.DataFrame(records).to_csv(HERE / 'predictions.csv', index=False)
    sections = frame.section.to_numpy()
    bins = (frame.start_s.to_numpy() // 60).astype(int)
    groups = [[np.flatnonzero((sections == s) & (bins == b)) for b in np.unique(bins[sections == s])] for s in np.unique(sections)]
    keys = ['line_precision', 'line_recall', 'line_f1', 'scrum_precision', 'scrum_recall', 'scrum_f1', 'f1_macro', 'accuracy']
    rng = np.random.default_rng(20261027)
    d = []
    for _ in range(2000):
        ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g), size=len(g))]) for g in groups])
        if any(((y[ix] == c).sum() == 0 for c in range(3))):
            continue
        a = [metric(y[ix], x[ix]) for x in preds['central']]
        b = [metric(y[ix], x[ix]) for x in preds['priority']]
        d.append([np.mean([m[k] - n[k] for m, n in zip(b, a)]) for k in keys])
    intervals = {k: dict(difference=float(np.mean([metric(y, n)[k] - metric(y, b)[k] for b, n in zip(preds['central'], preds['priority'])])), ci95=np.quantile(np.array(d)[:, j], [0.025, 0.975]).tolist()) for j, k in enumerate(keys)}
    result = dict(complete=True, rule='each view argmax; compare line and scrum votes; ties favor scrum; other only when all votes other', thresholds=False, seeds=SEEDS, blocks=688, means=pd.DataFrame(results).groupby('variant').mean(numeric_only=True).to_dict(orient='index'), bootstrap=dict(method='paired 60s clusters within sections; exploratory and conditional on fitted models', valid=len(d), metrics=intervals), input_hashes=hashes, limitations=['VF repeatedly observed; rule designed after VF findings; not independent', 'mean seeds not an ensemble', 'events priority is a decision rule favoring line/scrum, not a probability threshold'])
    (HERE / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
if __name__ == '__main__':
    main()
