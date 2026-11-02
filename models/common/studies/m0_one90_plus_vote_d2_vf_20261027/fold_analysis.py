import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
H = experiment_directory(__file__)
P = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/m0_oof_fusion_cost4_d2_vf_20261027')
M = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/multicrop_d2_stride25_vf_20261026/manifest.json')
L = ['line', 'other', 'scrum']
seeds = [17, 42, 2026]

def digest(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()
data = json.loads(_resolve_model_artifact(M).read_text(encoding='utf8'))
idx = _load_model_result_array(P / 'train_indices.npy')
fold = _load_model_result_array(P / 'folds.npy')
assert len(idx) == len(fold) == 8796
rows = [data[i] for i in idx]
y = np.array([L.index(r['tag']) for r in rows])
seen = {}
for r, k in zip(rows, fold):
    assert r['domain'] != 'VF'
    key = (r['domain'], r['source_sha256'] if r['domain'] == 'CH' else r['group_id'])
    if key in seen:
        assert seen[key] == k
    seen[key] = k
out = []
predictions = []
for seed in seeds:
    a = _load_model_result_array(P / f'seed_{seed}/oof.npy')
    assert a.shape == (8796, 9, 3)
    for k in range(3):
        root = P / f'seed_{seed}/fold_{k}'
        meta = json.loads(_resolve_model_artifact(root / 'complete.json').read_text())
        assert digest(root / 'pred.npy') == meta['prediction_sha256'] and digest(root / 'indices.npy') == meta['indices_sha256']
        assert np.array_equal(_load_model_result_array(root / 'indices.npy'), idx[fold == k]) and np.array_equal(_load_model_result_array(root / 'pred.npy'), a[fold == k])
        v = a[fold == k]
        truth = y[fold == k]
        base = v[:, 0].argmax(1)
        old = []
        new = []
        for row in v:
            c = int(np.argmax(row[0]))
            line = [r for r in row[1:] if int(np.argmax(r)) == 0]
            high = sum((r[0] >= 0.9 for r in line))
            old.append(0 if c == 1 and high >= 2 else c)
            new.append(0 if c == 1 and high >= 1 and (len(line) >= 2) else c)
        old = np.array(old)
        new = np.array(new)
        assert np.array_equal(base == 2, old == 2) and np.array_equal(base == 2, new == 2)
        for name, pred in [('central', base), ('dos90', old), ('uno90_mas_uno', new)]:
            pr, re, f, support = precision_recall_fscore_support(truth, pred, labels=[0, 1, 2], zero_division=0)
            d = dict(seed=seed, fold=k, model=name, n=len(truth), accuracy=accuracy_score(truth, pred), macro_f1=f1_score(truth, pred, labels=[0, 1, 2], average='macro', zero_division=0), changes=int(sum(base != pred)), corrected=int(sum((base != truth) & (pred == truth))), broken=int(sum((base == truth) & (pred != truth))))
            for i, l in enumerate(L):
                d.update({l + '_precision': pr[i], l + '_recall': re[i], l + '_f1': f[i], l + '_support': int(support[i]), l + '_tp': int(sum((truth == i) & (pred == i))), l + '_fp': int(sum((truth != i) & (pred == i))), l + '_fn': int(sum((truth == i) & (pred != i)))})
            out.append(d)
        predictions.extend([dict(seed=seed, fold=int(k), index=int(ix), tag=L[tr], central=int(b), dos90=int(o), uno90_mas_uno=int(n)) for ix, tr, b, o, n in zip(idx[fold == k], truth, base, old, new)])
pd.DataFrame(out).to_csv(H / 'fold_results.csv', index=False)
pd.DataFrame(predictions).to_csv(H / 'fold_predictions.csv', index=False)
print(pd.DataFrame(out).groupby(['fold', 'model'])[['n', 'line_support', 'line_precision', 'line_recall', 'line_tp', 'line_fp', 'scrum_support', 'scrum_precision', 'scrum_recall', 'scrum_fp', 'corrected', 'broken']].mean().round(4).to_string())
print('total seed mean')
print(pd.DataFrame(out).groupby(['model', 'seed'])[['line_tp', 'line_fp', 'line_fn', 'scrum_tp', 'scrum_fp', 'scrum_fn', 'corrected', 'broken']].sum().groupby('model').mean().to_string())
(H / 'fold_verification.json').write_text(json.dumps({'rows': 8796, 'fold_sizes': [int(sum(fold == i)) for i in range(3)], 'group_isolation': 'CV group_id; CH source sha256, original provenance incomplete', 'seed_models': 3, 'fold_models_per_seed': 3, 'hashes_and_oof_predictions_verified': True, 'scrum_predictions_bit_identical': True, 'fold_indices_same_across_seeds': True, 'metric_scope': 'per clip, not events'}, indent=2), encoding='utf8')
