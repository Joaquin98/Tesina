import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json, joblib
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
H = ROOT / 'data/m0_free_tree_d2_vf_20261027'
P = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
L = ['line', 'other', 'scrum']
records = []
events = []
output = []
prior = pd.read_csv(_resolve_model_artifact(H / 'predictions.csv'))
src = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
for seed in [17, 42, 2026]:
    q = prior[prior.seed == seed].reset_index(drop=True)
    v = _load_model_result_array(P / f'seed_{seed}/vf_views.npy')
    tree = joblib.load(_resolve_model_artifact(H / f'tree_{seed}.joblib'))
    prob = tree.predict_proba(v.reshape(-1, 27))
    raw = tree.predict(v.reshape(-1, 27))
    base = v[:, 0].argmax(1)
    y = np.array([L.index(x) for x in q.tag])
    assert np.array_equal(raw, q.arbol) and np.array_equal(base, q.central) and q.clip_id.equals(src.clip_id)
    frame = src[['clip_id', 'group_id', 'section', 'block_index', 'start_s', 'end_s', 'tag']].copy()
    frame['y_true'] = frame.tag
    row = q[['clip_id', 'seed', 'central', 'arbol']].copy()
    for name, t in [('proteccion_90', 0.9), ('proteccion_95', 0.95), ('proteccion_99', 0.99), ('proteccion_total', 1.01)]:
        pred = raw.copy()
        keep = (base == 2) & (raw != 2) & (prob.max(1) < t)
        pred[keep] = 2
        row[name] = pred
        assert np.array_equal(pred[base != 2], raw[base != 2])
        assert np.all(pred[base == 2][prob.max(1)[base == 2] < t] == 2)
        pr, re, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        d = {'seed': seed, 'model': name, 'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, average='macro'), 'central_scrum_changed': int(((base == 2) & (pred != 2)).sum())}
        for i, l in [(0, 'line'), (2, 'scrum')]:
            d.update({l + '_precision': pr[i], l + '_recall': re[i], l + '_f1': f1[i], l + '_fp': int(((pred == i) & (y != i)).sum())})
        records.append(d)
        z = frame.copy()
        z['y_final'] = [L[i] for i in pred]
        for i, l in enumerate(L):
            z['p_' + l] = prob[:, i]
        z.loc[keep, 'p_scrum'] = v[keep, 0, 2]
        for l in ['line', 'scrum']:
            real = events_from_blocks(z, l, prediction_column='y_true', probability_column='p_' + l)
            det = events_from_blocks(z, l, prediction_column='y_final', probability_column='p_' + l)
            matched = len(match_events(real, det))
            events.append(dict(seed=seed, model=name, label=l, real=len(real), detected=len(det), matched=matched, unmatched=len(det) - matched, precision=matched / len(det) if det else 0, recall=matched / len(real) if real else 0))
    output.append(row)
pd.DataFrame(records).to_csv(H / 'protected_scrum_blocks.csv', index=False)
pd.DataFrame(events).to_csv(H / 'protected_scrum_events.csv', index=False)
pd.concat(output).to_csv(H / 'protected_scrum_predictions.csv', index=False)
print('BLOCKS')
print(pd.DataFrame(records).groupby('model').mean(numeric_only=True).to_string())
print('EVENTS')
print(pd.DataFrame(events).groupby(['model', 'label']).mean(numeric_only=True).to_string())
