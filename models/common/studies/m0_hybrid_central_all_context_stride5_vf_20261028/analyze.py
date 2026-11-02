import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
H = experiment_directory(__file__)
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
A = ROOT / 'data/m0_h2_d2_allframes_nine_vf_20261027'
B = ROOT / 'data/m0_h2_d2_stride5_cached_vf_20261028'
M = ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json'
L = ['line', 'other', 'scrum']
assert json.loads(_resolve_model_artifact(A / 'verification.json').read_text())['passed'] and json.loads(_resolve_model_artifact(B / 'verification.json').read_text())['passed'] and json.loads(_resolve_model_artifact(B / 'cache_verification.json').read_text())['passed']
rows = [r for r in json.loads(_resolve_model_artifact(M).read_text(encoding='utf8')) if r['domain'] == 'VF' and r['split'] == 'test']
frame = pd.DataFrame(rows)
y = np.array([L.index(x) for x in frame.tag])
assert len(y) == 688
allp = np.stack([_load_model_result_array(A / 'blocks' / f'{i:04d}.npy') for i in range(688)])
sparse = np.stack([_load_model_result_array(B / 'blocks' / f'{i:04d}.npy') for i in range(688)])
assert allp.shape == sparse.shape == (688, 6, 9, 3)

def classify(v):
    c = v[:, 0].argmax(1)
    votes = v[:, 1:].argmax(2) == 0
    high = votes & (v[:, 1:, 0] >= 0.5)
    return np.where((c == 1) & (votes.sum(1) >= 2) & (high.sum(1) >= 1), 0, c)

def metrics(y, p):
    pr, rc, f, s = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    d = {'accuracy': accuracy_score(y, p), 'macro_f1': f1_score(y, p, average='macro')}
    for j, l in enumerate(L):
        d.update({l + '_precision': pr[j], l + '_recall': rc[j], l + '_f1': f[j], l + '_tp': int(sum((y == j) & (p == j))), l + '_fp': int(sum((y != j) & (p == j))), '_' + l + '_support': int(s[j])})
    return d
results = []
events = []
frames = []
preds = {k: [] for k in ['central_all', 'all_frames', 'hybrid', 'stride5']}
for j in range(6):
    full = allp[:, j]
    thin = sparse[:, j]
    hybrid = np.concatenate([full[:, :1], thin[:, 1:]], axis=1)
    base = full[:, 0].argmax(1)
    selections = {'central_all': (full, base), 'all_frames': (full, classify(full)), 'hybrid': (hybrid, classify(hybrid)), 'stride5': (thin, classify(thin))}
    fout = frame[['clip_id', 'section', 'start_s', 'end_s', 'tag']].copy()
    fout['fold'] = j + 1
    for name, (p, pred) in selections.items():
        if name in ['central_all', 'all_frames', 'hybrid']:
            assert np.array_equal(pred == 2, base == 2)
        preds[name].append(pred)
        fout[name] = pred
        results.append({'fold': j + 1, 'variant': name, **metrics(y, pred)})
        f = frame.copy()
        f['y_true'] = f.tag
        f['y_final'] = [L[x] for x in pred]
        f['p_line'] = p[:, 0, 0]
        f['p_scrum'] = p[:, 0, 2]
        if name != 'central_all':
            vote = p[:, 1:].argmax(2) == 0
            sc = np.sort(np.where(vote, p[:, 1:, 0], 0), axis=1)[:, -2]
            c = p[:, 0].argmax(1)
            f.loc[pred != c, 'p_line'] = sc[pred != c]
        for lab in ['line', 'scrum']:
            t = events_from_blocks(f, lab, prediction_column='y_true', probability_column='p_' + lab)
            z = events_from_blocks(f, lab, prediction_column='y_final', probability_column='p_' + lab)
            matches = match_events(t, z)
            n = len(matches)
            events.append({'fold': j + 1, 'variant': name, 'label': lab, 'real': len(t), 'predicted': len(z), 'matched': n, 'unmatched': len(z) - n, 'precision': n / len(z) if z else 0, 'recall': n / len(t) if t else 0, 'f1': 2 * n / (len(t) + len(z)) if len(t) + len(z) else 0})
    frames.append(fout)
pd.DataFrame(results).to_csv(H / 'results.csv', index=False)
pd.DataFrame(events).to_csv(H / 'events.csv', index=False)
pd.concat(frames).to_csv(H / 'predictions.csv', index=False)
bins = np.floor(frame.start_s.to_numpy() / 60).astype(int)
sec = frame.section.to_numpy()
groups = [[np.flatnonzero((sec == s) & (bins == b)) for b in np.unique(bins[sec == s])] for s in np.unique(sec)]
rng = np.random.default_rng(20261028)
diff = {k: [] for k in ['all_frames', 'stride5', 'central_all']}
for _ in range(2000):
    ix = np.concatenate([np.concatenate([g[x] for x in rng.integers(len(g), size=len(g))]) for g in groups])
    yy = y[ix]
    if not (yy == 0).any():
        continue
    for k in diff:
        diff[k].append(np.mean([(sum((yy == 0) & (a[ix] == 0)) - sum((yy == 0) & (b[ix] == 0))) / sum(yy == 0) for a, b in zip(preds['hybrid'], preds[k])]))
boot = {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in diff.items()}
(H / 'bootstrap.json').write_text(json.dumps({'method': 'paired 2000 resamples of 60s clusters within section, mean of six fold checkpoints; descriptive post-selection VF', 'line_recall_difference_hybrid_minus': boot}, indent=2), encoding='utf8')
(H / 'protocol.json').write_text(json.dumps({'central': 'all frames, up to 175', 'context': 'one of every five 25fps grid frames, up to 35', 'decision': 'central other changes to line if 2 contexts vote line and at least one p>=0.50; central scrum protected', 'six_fold_models': 'same pretrained checkpoints for both rates; average of six, not ensemble', 'VF': '688 labelled blocks, previously used; exploratory', 'source_probabilities': 'reused existing saved predictions; no frames re-extracted'}, indent=2), encoding='utf8')
print('BLOCK')
print(pd.DataFrame(results).groupby('variant').mean(numeric_only=True).to_string())
print('EVENTS')
print(pd.DataFrame(events).groupby(['variant', 'label']).mean(numeric_only=True).to_string())
print('BOOT', boot)
