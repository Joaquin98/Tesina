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
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
H = ROOT / 'data/m0_free_tree_d2_vf_20261027'
P = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
LABELS = ['line', 'other', 'scrum']
src = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
pred = pd.read_csv(_resolve_model_artifact(H / 'predictions.csv'))
out = []
details = []
for seed in [17, 42, 2026]:
    q = pred[pred.seed == seed].reset_index(drop=True)
    f = src[['clip_id', 'group_id', 'section', 'block_index', 'start_s', 'end_s', 'tag']].copy()
    assert q.clip_id.equals(f.clip_id)
    v = _load_model_result_array(P / f'seed_{seed}/vf_views.npy')
    model = joblib.load(_resolve_model_artifact(H / f'tree_{seed}.joblib'))
    treeprob = model.predict_proba(v.reshape(-1, 27))
    assert np.array_equal(model.predict(v.reshape(-1, 27)), q.arbol)
    for name in ['central', 'regla_90', 'arbol']:
        z = f.copy()
        z['y_true'] = z.tag
        z['y_final'] = [LABELS[i] for i in q[name]]
        for j, lab in enumerate(LABELS):
            z['p_' + lab] = treeprob[:, j] if name == 'arbol' else v[:, 0, j]
        if name == 'regla_90':
            changed = q.central.to_numpy() != q.regla_90.to_numpy()
            votes = v[:, 1:].argmax(2) == 0
            score = np.sort(np.where(votes, v[:, 1:, 0], 0), axis=1)[:, -2]
            z.loc[changed, 'p_line'] = score[changed]
        for lab in ['line', 'scrum']:
            true = events_from_blocks(z, lab, prediction_column='y_true', probability_column='p_' + lab)
            found = events_from_blocks(z, lab, prediction_column='y_final', probability_column='p_' + lab)
            matches = match_events(true, found)
            n = len(matches)
            prec = n / len(found) if found else 0
            rec = n / len(true) if true else 0
            f1 = 2 * n / (len(found) + len(true)) if len(found) + len(true) else 0
            out.append(dict(seed=seed, model=name, label=lab, true_events=len(true), predicted_events=len(found), matched=n, unmatched=len(found) - n, missed=len(true) - n, precision=prec, recall=rec, f1=f1))
            details.append(dict(seed=seed, model=name, label=lab, matches=[dict(truth=i, prediction=j, distance=d) for i, j, d, w in matches]))
pd.DataFrame(out).to_csv(H / 'event_results.csv', index=False)
(H / 'event_assignments.json').write_text(json.dumps(details, indent=2), encoding='utf8')
print(pd.DataFrame(out).groupby(['model', 'label']).mean(numeric_only=True).to_string())
