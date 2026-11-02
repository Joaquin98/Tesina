import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json, itertools
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
H = experiment_directory(__file__)
P = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
M = ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json'
S = [17, 42, 2026]
L = ['line', 'other', 'scrum']

def predict(v, subset):
    base = v[:, 0].argmax(1)
    if len(subset) < 2:
        return base.copy()
    w = v[:, list(subset)]
    votes = w.argmax(2) == 0
    high = votes & (w[:, :, 0] >= 0.9)
    return np.where((base == 1) & (votes.sum(1) >= 2) & (high.sum(1) >= 1), 0, base)

def stats(y, p):
    a, b, f, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    return dict(line_precision=a[0], line_recall=b[0], line_f1=f[0], line_fp=int(sum((y != 0) & (p == 0))), scrum_precision=a[2], scrum_recall=b[2], scrum_f1=f[2])

def select():
    assert not _resolve_model_artifact(H / 'selection.json').exists()
    data = json.loads(_resolve_model_artifact(M).read_text(encoding='utf8'))
    idx = _load_model_result_array(P / 'train_indices.npy')
    y = np.array([L.index(data[i]['tag']) for i in idx])
    views = {s: _load_model_result_array(P / f'seed_{s}/oof.npy') for s in S}
    allrows = []
    chosen = {}
    for k in range(0, 9):
        candidates = []
        for subset in itertools.combinations(range(1, 9), k):
            metrics = [stats(y, predict(views[s], subset)) for s in S]
            mean = {key: float(np.mean([d[key] for d in metrics])) for key in metrics[0]}
            candidates.append((subset, mean))
            allrows.append(dict(total_views=k + 1, subset=' '.join(map(str, subset)), **mean))
        best = max(candidates, key=lambda x: (x[1]['line_f1'], tuple((-i for i in x[0]))))
        chosen[str(k + 1)] = {'contexts': list(best[0]), 'd2_mean': best[1]}
    pd.DataFrame(allrows).to_csv(H / 'd2_subsets.csv', index=False)
    (H / 'selection.json').write_text(json.dumps({'selection': 'highest mean line F1 on the same D2 OOF predictions, lexicographic tie; exploratory, multiple comparisons, not nested', 'rule': 'central other to line if at least two selected context crops vote line and at least one has p>=0.9', 'central_fixed': True, 'subsets': chosen}, indent=2), encoding='utf8')

def evaluate():
    chosen = json.loads(_resolve_model_artifact(H / 'selection.json').read_text())['subsets']
    src = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
    y = np.array([L.index(x) for x in src.tag])
    rows = []
    ev = []
    predrows = []
    for s in S:
        v = _load_model_result_array(P / f'seed_{s}/vf_views.npy')
        base = v[:, 0].argmax(1)
        o = src[['clip_id', 'tag', 'section', 'start_s']].copy()
        o['seed'] = s
        for n in range(1, 10):
            subset = chosen[str(n)]['contexts']
            pred = predict(v, subset)
            assert np.array_equal(pred == 2, base == 2)
            o[f'views_{n}'] = pred
            rows.append(dict(seed=s, views=n, subset=' '.join(map(str, subset)), **stats(y, pred)))
            f = src[['clip_id', 'group_id', 'section', 'block_index', 'start_s', 'end_s', 'tag']].copy()
            f['y_true'] = f.tag
            f['y_final'] = [L[j] for j in pred]
            f['p_line'] = v[:, 0, 0]
            f['p_scrum'] = v[:, 0, 2]
            if len(subset) >= 2:
                z = v[:, subset]
                votes = z.argmax(2) == 0
                score = np.sort(np.where(votes, z[:, :, 0], 0), axis=1)[:, -2]
                changed = pred != base
                f.loc[changed, 'p_line'] = score[changed]
            for lab in ['line', 'scrum']:
                t = events_from_blocks(f, lab, prediction_column='y_true', probability_column='p_' + lab)
                d = events_from_blocks(f, lab, prediction_column='y_final', probability_column='p_' + lab)
                nmatch = len(match_events(t, d))
                ev.append(dict(seed=s, views=n, label=lab, true=len(t), predicted=len(d), matched=nmatch, unmatched=len(d) - nmatch, precision=nmatch / len(d) if d else 0, recall=nmatch / len(t) if t else 0))
        predrows.append(o)
    pd.DataFrame(rows).to_csv(H / 'vf_results.csv', index=False)
    pd.DataFrame(ev).to_csv(H / 'vf_events.csv', index=False)
    pd.concat(predrows).to_csv(H / 'vf_predictions.csv', index=False)
    print('CHOICES')
    print({k: v['contexts'] for k, v in chosen.items()})
    print('BLOCKS')
    print(pd.DataFrame(rows).groupby('views').mean(numeric_only=True).round(4).to_string())
    print('EVENTS')
    print(pd.DataFrame(ev).groupby(['views', 'label']).mean(numeric_only=True).round(4).to_string())
select()
evaluate()
