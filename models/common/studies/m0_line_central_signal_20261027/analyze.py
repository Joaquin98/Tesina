import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json
import numpy as np, pandas as pd, cv2
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
H = ROOT / 'data/m0_line_central_signal_20261027'
P = ROOT / 'data/m0_oof_fusion_cost4_d2_vf_20261027'
R = ROOT / 'data/m0_protect_scrum_rescue_line_d2_vf_20261027'
M = ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json'
L = ['line', 'other', 'scrum']
seeds = [17, 42, 2026]
assert not _resolve_model_artifact(H / 'protocol.json').exists()
data = json.loads(_resolve_model_artifact(M).read_text())
idx = _load_model_result_array(P / 'train_indices.npy')
y = np.array([L.index(data[i]['tag']) for i in idx])
d2 = []

def predict(v, signal):
    b = v[:, 0].argmax(1)
    votes = ((v[:, 1:].argmax(2) == 0) & (v[:, 1:, 0] >= 0.9)).sum(1)
    return np.where((b == 1) & (votes >= 2) & (v[:, 0, 0] >= signal), 0, b)

def counts(y, p):
    pr, re, f, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    return {'line_precision': pr[0], 'line_recall': re[0], 'line_f1': f[0], 'line_fp': int(sum((y != 0) & (p == 0))), 'scrum_precision': pr[2], 'scrum_recall': re[2], 'scrum_fp': int(sum((y != 2) & (p == 2))), 'macro_f1': f1_score(y, p, average='macro'), 'accuracy': accuracy_score(y, p)}
for s in seeds:
    v = _load_model_result_array(P / f'seed_{s}/oof.npy')
    b = v[:, 0].argmax(1)
    for name, sig in [('central', None), ('regla90', 0), ('central10', 0.1)]:
        a = b if sig is None else predict(v, sig)
        d2.append(dict(seed=s, model=name, **counts(y, a), corrected=int(sum((b != y) & (a == y))), broken=int(sum((b == y) & (a != y)))))
pd.DataFrame(d2).to_csv(H / 'd2_results.csv', index=False)
(H / 'protocol.json').write_text(json.dumps({'rule': 'central unless central predicts other AND two contexts vote line with probability >= .90 AND central line probability >= .10', 'selection': '0.10 exploratory based on aggregated D2 OOF rescues; D2 itself not independent evaluation', 'models': 'M0 cost4, previously trained; seeds 17,42,2026', 'limitations': 'VF repeatedly observed; threshold selected after trying alternatives in D2; scores uncalibrated; no independent test'}, indent=2), encoding='utf8')
src = pd.read_csv(_resolve_model_artifact(P / 'vf_predictions.csv'))
old = pd.read_csv(_resolve_model_artifact(R / 'vf_predictions.csv'))
rows = []
event = []
preds = []
for s in seeds:
    v = _load_model_result_array(P / f'seed_{s}/vf_views.npy')
    base = v[:, 0].argmax(1)
    y = np.array([L.index(t) for t in src.tag])
    q = old[old.seed == s].reset_index(drop=True)
    assert q.clip_id.equals(src.clip_id)
    new = predict(v, 0.1)
    prior = predict(v, 0)
    assert np.array_equal(prior, q['t_0.9'])
    assert np.array_equal(new == 2, base == 2)
    out = src[['clip_id', 'section', 'start_s', 'end_s', 'source', 'tag']].copy()
    out['seed'] = s
    out['central'] = base
    out['regla90'] = prior
    out['central10'] = new
    out['p_central_line'] = v[:, 0, 0]
    preds.append(out)
    for name, a in [('central', base), ('regla90', prior), ('central10', new)]:
        rows.append(dict(seed=s, model=name, **counts(y, a), corrected=int(sum((base != y) & (a == y))), broken=int(sum((base == y) & (a != y)))))
        f = src[['clip_id', 'group_id', 'section', 'block_index', 'start_s', 'end_s', 'tag']].copy()
        f['y_true'] = f.tag
        f['y_final'] = [L[i] for i in a]
        for j, lab in enumerate(L):
            f['p_' + lab] = v[:, 0, j]
        if name != 'central':
            changed = a != base
            votes = v[:, 1:].argmax(2) == 0
            score = np.sort(np.where(votes, v[:, 1:, 0], 0), axis=1)[:, -2]
            f.loc[changed, 'p_line'] = score[changed]
        for lab in ['line', 'scrum']:
            real = events_from_blocks(f, lab, prediction_column='y_true', probability_column='p_' + lab)
            det = events_from_blocks(f, lab, prediction_column='y_final', probability_column='p_' + lab)
            mt = match_events(real, det)
            n = len(mt)
            event.append(dict(seed=s, model=name, label=lab, real=len(real), detected=len(det), matched=n, unmatched=len(det) - n, precision=n / len(det) if det else 0, recall=n / len(real) if real else 0))
pd.DataFrame(rows).to_csv(H / 'vf_results.csv', index=False)
pd.DataFrame(event).to_csv(H / 'events.csv', index=False)
allp = pd.concat(preds)
allp.to_csv(H / 'predictions.csv', index=False)
print('D2')
print(pd.DataFrame(d2).groupby('model').mean(numeric_only=True).to_string())
print('VF BLOCK')
print(pd.DataFrame(rows).groupby('model').mean(numeric_only=True).to_string())
print('VF EVENTS')
print(pd.DataFrame(event).groupby(['model', 'label']).mean(numeric_only=True).to_string())
v = allp[allp.seed == 42]
changed = v[(v.central == 1) & (v.regla90 == 0)]
examples = pd.concat([changed[changed.tag == 'line'].head(4), changed[changed.tag != 'line'].head(5)])
cv2.setNumThreads(1)
for j, (_, r) in enumerate(examples.iterrows()):
    cap = cv2.VideoCapture(r.source)
    tiles = []
    for dt in [1, 3.5, 6]:
        cap.set(cv2.CAP_PROP_POS_MSEC, (r.start_s + dt) * 1000)
        ok, im = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode {r.source} {r.start_s + dt}')
        im = cv2.resize(im, (480, 270))
        cv2.putText(im, f'{r.section} {r.start_s + dt:.1f}s', (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        tiles.append(im)
    cap.release()
    image = np.vstack([np.full((55, 1440, 3), 35, np.uint8), np.hstack(tiles)])
    cv2.putText(image, f"tag={r.tag} p_line_central={r.p_central_line:.3f} new={('line' if r.central10 == 0 else 'other')}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    assert cv2.imwrite(str(H / f'example_{j + 1:02d}.jpg'), image)
examples.to_csv(H / 'examples.csv', index=False)
