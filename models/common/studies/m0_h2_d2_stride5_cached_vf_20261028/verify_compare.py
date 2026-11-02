import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import numpy as np, pandas as pd, json
from sklearn.metrics import precision_recall_fscore_support
H = experiment_directory(__file__)
A = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/m0_h2_d2_allframes_nine_vf_20261027')
R = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/m0_allframes_nine_rule_sweep_vf_20261028')
manifest = json.loads(_resolve_model_artifact(Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/multicrop_d2_stride25_vf_20261026/manifest.json')).read_text(encoding='utf8'))
f = pd.DataFrame([r for r in manifest if r['domain'] == 'VF' and r['split'] == 'test'])
assert len(f) == 688
y = f.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy()
saved = pd.read_csv(_resolve_model_artifact(H / 'predictions.csv'))
reported = pd.read_csv(_resolve_model_artifact(H / 'results.csv'))
arr = np.stack([_load_model_result_array(H / 'blocks' / f'{i:04d}.npy') for i in range(688)])
old = np.stack([_load_model_result_array(A / 'blocks' / f'{i:04d}.npy') for i in range(688)])
assert arr.shape == old.shape == (688, 6, 9, 3)

def decide(v):
    result = []
    for row in v:
        b = int(row[0].argmax())
        votes = [x for x in row[1:] if x.argmax() == 0]
        result.append(0 if b == 1 and len(votes) >= 2 and any((x[0] >= 0.5 for x in votes)) else b)
    return np.array(result)
newpred = []
oldpred = []
for j in range(6):
    p = arr[:, j]
    central = p[:, 0].argmax(1)
    a = decide(p)
    b = decide(old[:, j])
    newpred.append(a)
    oldpred.append(b)
    q = saved[saved.fold == j + 1]
    assert list(q.clip_id) == list(f.clip_id) and np.array_equal(a, q.nine) and np.array_equal(central, q.central) and np.array_equal(central == 2, a == 2)
    for name, pred in [('central', central), ('nine', a)]:
        pr, re, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        r = reported[(reported.fold == j + 1) & (reported.model == name)].iloc[0]
        assert np.isclose(r.line_precision, pr[0]) and np.isclose(r.line_recall, re[0]) and np.isclose(r.scrum_recall, re[2]) and np.isclose(r.scrum_precision, pr[2])
sec = f.section.to_numpy()
bins = np.floor(f.start_s.to_numpy() / 60).astype(int)
groups = [[np.flatnonzero((sec == s) & (bins == b)) for b in np.unique(bins[sec == s])] for s in np.unique(sec)]
rng = np.random.default_rng(20261028)
diff = []
for _ in range(2000):
    ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g), size=len(g))]) for g in groups])
    yy = y[ix]
    if sum(yy == 0) == 0:
        continue
    diff.append(np.mean([(sum((yy == 0) & (a[ix] == 0)) - sum((yy == 0) & (b[ix] == 0))) / sum(yy == 0) for a, b in zip(newpred, oldpred)]))
ci = np.quantile(diff, [0.025, 0.975]).tolist()
(H / 'verification.json').write_text(json.dumps({'passed': True, 'all_probabilities': list(arr.shape), 'decision_independently_checked': True, 'block_metrics_recomputed': True, 'scrum_equals_central': True, 'comparison': 'stride5 vs allframes same fold checkpoints and same rule 50', 'paired_bootstrap_line_recall_diff_ci95': ci, 'bootstrap': '2000 stratified 60s clusters, descriptive, VF used previously'}, indent=2), encoding='utf8')
oldm = pd.read_csv(_resolve_model_artifact(R / 'metrics_by_fold.csv'))
oldm = oldm[(oldm.high == 0.5) & (oldm.low == 0) & (oldm.min_votes == 2) & (oldm.central_min == 0)]
newm = reported[reported.model == 'nine']
print('NEW BLOCK MEAN', newm.mean(numeric_only=True).to_dict())
print('OLD BLOCK MEAN', oldm.mean(numeric_only=True).to_dict())
print('NEW EVENTS', pd.read_csv(_resolve_model_artifact(H / 'events.csv')).groupby(['model', 'label']).mean(numeric_only=True).to_string())
print('CI stride5 minus allframes line recall', ci)
