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
from sklearn.metrics import precision_recall_fscore_support
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
H = experiment_directory(__file__)
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
cfg = json.loads(_resolve_model_artifact(H / 'protocol.json').read_text())
labels = ['line', 'other', 'scrum']
rows = [r for r in json.loads(_resolve_model_artifact(Path(next((k for k in cfg['hashes'] if k.endswith('manifest.json'))))).read_text(encoding='utf8')) if r['domain'] == 'VF' and r['split'] == 'test']
frame = pd.DataFrame(rows)
assert len(frame) == 688
for path, expected in cfg['hashes'].items():
    h = hashlib.sha256()
    with open(_resolve_model_artifact(path), 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    assert h.hexdigest() == expected, path
arr = np.stack([_load_model_result_array(H / 'blocks' / f'{i:04}.npy') for i in range(688)])
assert arr.shape == (688, 6, 9, 3) and np.isfinite(arr).all() and np.allclose(arr.sum(3), 1, atol=1e-05)
r = pd.read_csv(_resolve_model_artifact(H / 'results.csv'))
ev = pd.read_csv(_resolve_model_artifact(H / 'events.csv'))
saved = pd.read_csv(_resolve_model_artifact(H / 'predictions.csv'))
y = np.array([labels.index(x) for x in frame.tag])
bases = []
combos = []
for fi in range(6):
    q = saved[saved.fold == fi + 1]
    assert list(q.clip_id) == list(frame.clip_id)
    base = []
    combined = []
    for p in arr[:, fi]:
        c = int(p[0].argmax())
        v = [v for v in p[1:] if v.argmax() == 0]
        base.append(c)
        combined.append(0 if c == 1 and len(v) >= 2 and any((x[0] >= 0.9 for x in v)) else c)
    base = np.array(base)
    combined = np.array(combined)
    assert np.array_equal(base, q.central) and np.array_equal(combined, q.nine) and np.array_equal(base == 2, combined == 2)
    bases.append(base)
    combos.append(combined)
    for name, pred in [('central', base), ('nine', combined)]:
        pr, re, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        a = r[(r.fold == fi + 1) & (r.model == name)].iloc[0]
        for j, l in [(0, 'line'), (2, 'scrum')]:
            assert np.isclose(a[l + '_precision'], pr[j]) and np.isclose(a[l + '_recall'], re[j]) and np.isclose(a[l + '_f1'], f[j]) and (a[l + '_fp'] == sum((y != j) & (pred == j)))
        f = frame.copy()
        f['y_true'] = f.tag
        f['y_final'] = [labels[i] for i in pred]
        p = arr[:, fi]
        f['p_line'] = p[:, 0, 0]
        f['p_scrum'] = p[:, 0, 2]
        if name == 'nine':
            vote = p[:, 1:].argmax(2) == 0
            strength = np.sort(np.where(vote, p[:, 1:, 0], 0), axis=1)[:, -2]
            f.loc[pred != base, 'p_line'] = strength[pred != base]
        for l in ['line', 'scrum']:
            real = events_from_blocks(f, l, prediction_column='y_true', probability_column='p_' + l)
            det = events_from_blocks(f, l, prediction_column='y_final', probability_column='p_' + l)
            matches = match_events(real, det)
            record = ev[(ev.fold == fi + 1) & (ev.model == name) & (ev.label == l)].iloc[0]
            assert record.real == len(real) and record.predicted == len(det) and (record.matched == len(matches)) and (record.unmatched == len(det) - len(matches))
groups = []
sec = frame.section.to_numpy()
bins = np.floor(frame.start_s.to_numpy() / 60).astype(int)
for s in np.unique(sec):
    groups.append([np.flatnonzero((sec == s) & (bins == b)) for b in np.unique(bins[sec == s])])
rng = np.random.default_rng(20261028)
d = []
for _ in range(2000):
    ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g), size=len(g))]) for g in groups])
    yy = y[ix]
    if (yy == 0).sum() == 0:
        continue
    d.append(np.mean([(sum((yy == 0) & (n[ix] == 0)) - sum((yy == 0) & (b[ix] == 0))) / sum(yy == 0) for b, n in zip(bases, combos)]))
res = {'passed': True, 'inputs_hashes_match': True, 'probability_shape': list(arr.shape), 'models': 6, 'predictions_rule_and_block_metrics_independently_recomputed': True, 'event_metrics_recomputed': True, 'scrum_predictions_identical': True, 'line_recall_difference_pp_ci95': (100 * np.quantile(d, [0.025, 0.975])).tolist(), 'bootstrap': '2000 paired, 60-second clusters within section, exploratory; six folds not independent; VF reused'}
(H / 'verification.json').write_text(json.dumps(res, indent=2), encoding='utf8')
print(json.dumps(res, indent=2))
