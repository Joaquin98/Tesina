import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import sys, json, itertools, time
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
H = experiment_directory(__file__)
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
P = ROOT / 'data/m0_h2_d2_allframes_nine_vf_20261027'
D = ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json'
L = ['line', 'other', 'scrum']
assert json.loads(_resolve_model_artifact(P / 'verification.json').read_text())['passed']
rows = [r for r in json.loads(_resolve_model_artifact(D).read_text(encoding='utf8')) if r['domain'] == 'VF' and r['split'] == 'test']
f0 = pd.DataFrame(rows)
y = np.array([L.index(t) for t in f0.tag])
z = np.stack([_load_model_result_array(P / 'blocks' / f'{i:04d}.npy') for i in range(688)])
assert z.shape == (688, 6, 9, 3)
configs = [(h, lo, n, c) for h in [0.5, 0.7, 0.9, 0.95] for lo in [0, 0.6] if h >= lo for n in [2, 3, 4] for c in [0, 0.1]]
protocol = {'selection': 'post-hoc exploratory sweep of VF, no independent tuning data for full frame model; do not report winning score as independent estimate', 'input': 'six previously trained M0-H2-D2 fold models, same VF and 25fps features; average folds not ensemble', 'decision': 'central unless central predicts other; then switch to line if >=N contexts vote line with p>=low, including >=1 with p>=high, and p_line(central)>=central_min. Scrum and line central always preserved', 'high': [0.5, 0.7, 0.9, 0.95], 'low': [0, 0.6], 'minimum_votes': [2, 3, 4], 'central_min': [0, 0.1], 'models': len(configs), 'event_evaluation': 'one-to-one maximal proximity credit; same section, anchor <=35s; group adjacent same label blocks, central for unchanged, Nth context vote confidence for changed'}
(H / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf8')
out = []
details = []
started = time.time()
for t, (high, low, n, cmin) in enumerate(configs):
    for fold in range(6):
        p = z[:, fold]
        base = p[:, 0].argmax(1)
        votes = (p[:, 1:].argmax(2) == 0) & (p[:, 1:, 0] >= low)
        confident = votes & (p[:, 1:, 0] >= high)
        switch = (base == 1) & (votes.sum(1) >= n) & (confident.sum(1) >= 1) & (p[:, 0, 0] >= cmin)
        pred = np.where(switch, 0, base)
        assert np.array_equal(pred == 2, base == 2)
        pr, re, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
        record = {'high': high, 'low': low, 'min_votes': n, 'central_min': cmin, 'fold': fold + 1, 'block_precision': pr[0], 'block_recall': re[0], 'block_f1': f[0], 'block_fp': int(sum((pred == 0) & (y != 0))), 'scrum_precision': pr[2], 'scrum_recall': re[2]}
        frame = f0.copy()
        frame['y_true'] = frame.tag
        frame['y_final'] = [L[i] for i in pred]
        frame['p_line'] = p[:, 0, 0]
        strength = np.sort(np.where(votes, p[:, 1:, 0], 0), axis=1)[:, -n]
        frame.loc[switch, 'p_line'] = strength[switch]
        truth = events_from_blocks(frame, 'line', prediction_column='y_true', probability_column='p_line')
        found = events_from_blocks(frame, 'line', prediction_column='y_final', probability_column='p_line')
        matches = match_events(truth, found)
        m = len(matches)
        record.update(event_truth=len(truth), event_detected=len(found), event_matched=m, event_unmatched=len(found) - m, event_precision=m / len(found) if found else 0, event_recall=m / len(truth) if truth else 0, event_f1=2 * m / (len(found) + len(truth)) if len(found) + len(truth) else 0)
        out.append(record)
    print(f'{t + 1}/{len(configs)} configurations; elapsed {time.time() - started:.0f}s', flush=True)
df = pd.DataFrame(out)
df.to_csv(H / 'metrics_by_fold.csv', index=False)
keys = ['high', 'low', 'min_votes', 'central_min']
summary = df.groupby(keys, as_index=False).mean(numeric_only=True).drop(columns=['fold'])
summary.to_csv(H / 'summary.csv', index=False)
print('TOP EVENT F1')
print(summary.sort_values('event_f1', ascending=False).head(10).to_string(index=False))
print('TOP BLOCK F1')
print(summary.sort_values('block_f1', ascending=False).head(10).to_string(index=False))
(H / 'completion.json').write_text(json.dumps({'complete': True, 'configurations': len(configs), 'fold_evaluations': len(df), 'elapsed_seconds': time.time() - started, 'vf_exploratory': True}, indent=2), encoding='utf8')
