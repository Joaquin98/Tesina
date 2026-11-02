"""Exhaustive line-threshold sweep over every distinct decision state on VF."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from models.common.comparison import metrics as m
HERE = experiment_directory(__file__)
OUT = HERE / 'continuous_sweep'
OUT.mkdir(exist_ok=True)
source = HERE / 'results/derived_predictions.csv.gz'
digest = hashlib.sha256(_resolve_model_artifact(source).read_bytes()).hexdigest()
f = pd.read_csv(_resolve_model_artifact(source))
f = f[f.policy.eq('argmax')].copy().reset_index(drop=True)
assert len(f) == 688
p = m.normalize_probabilities(f[['p_line', 'p_other', 'p_scrum']].to_numpy())
a = p.argmax(1)
labels = np.array(['line', 'other', 'scrum'])
truth = {c: m.events_from_blocks(f, c, prediction_column='y_true', probability_column='p_' + c) for c in ['line', 'scrum']}
scrum = m.temporal_metrics_for_class(f.assign(y_final=labels[a]), 'scrum')

def evaluate(t=None):
    y = a if t is None else np.where(a == 2, 2, np.where(p[:, 0] >= t, 0, 1))
    assert np.array_equal(y == 2, a == 2)
    d = f.assign(y_final=labels[y])
    pr = m.events_from_blocks(d, 'line', prediction_column='y_final', probability_column='p_line')
    pairs = m.match_events(truth['line'], pr)
    tp = len(pairs)
    n = len(pr)
    real = len(truth['line'])
    weighted = 2 * sum((x[3] for x in pairs)) / (n + real)
    return dict(threshold=t, tp=tp, fp=n - tp, fn=real - tp, candidates=n, precision=tp / n if n else 0, recall=tp / real, f1=2 * tp / (n + real), temporal_f1=weighted, macro_temporal_f1=(weighted + scrum['weighted_f1']) / 2, active_blocks=int((y == 0).sum()))
scores = np.unique(p[a != 2, 0])
rows = []
low = 0.0
for t in list(scores) + ([1.0] if scores[-1] < 1 else []):
    rows.append(dict(evaluate(float(t)), lower_exclusive=float(low), upper_inclusive=float(t), lower_inclusive=bool(low == 0)))
    low = float(t)
table = pd.DataFrame(rows)
table.to_csv(OUT / 'all_threshold_states.csv', index=False)
baseline = evaluate()
old = pd.read_csv(_resolve_model_artifact(HERE / 'results/match_metrics.csv'))
for t in [0.45, 0.35, 0.25, 0.15]:
    got = evaluate(t)
    expected = old[np.isclose(old.line_threshold, t)].iloc[0]
    for k, col in [('tp', 'line_tp'), ('fp', 'line_fp'), ('fn', 'line_fn'), ('temporal_f1', 'line_temporal_f1')]:
        assert np.isclose(got[k], expected[col]), (t, k)

def best(metric, minimum_recall=None):
    eligible = table if minimum_recall is None else table[table.recall >= minimum_recall]
    if eligible.empty:
        return None
    b = eligible.sort_values([metric, 'recall', 'fp', 'threshold'], ascending=[False, False, True, False]).iloc[0].to_dict()
    for digits in [2, 3, 4, 6]:
        scale = 10 ** digits
        pick = np.floor(b['upper_inclusive'] * scale) / scale
        if b['lower_exclusive'] < pick <= b['upper_inclusive']:
            b['display_threshold'] = float(pick)
            break
    else:
        b['display_threshold'] = b['threshold']
    assert np.isclose(evaluate(b['display_threshold'])[metric], b[metric])
    return b
summary = dict(model='ALL-DEVELOPMENT-M1-H2-D2-S42', seconds=4804.8, sections=['S01', 'S02', 'S03', 'S04'], states=len(table), blocks=len(f), baseline=baseline, scrum=scrum, best_temporal_f1=best('temporal_f1'), best_event_f1=best('f1'), best_recall90=best('temporal_f1', 0.9), best_recall95=best('temporal_f1', 0.95), best_recall100=best('temporal_f1', 1.0), source_sha256=digest, selection='Exploratory selection on already inspected VF, not independent validation.', matching='Unchanged event grouping within sections, peak-block anchors, Hungarian distance credit, tolerance 35 s.', threshold_domain='All t in [0,1], evaluated at every eligible distinct p_line decision boundary.', policy='Keep argmax scrum. Otherwise line iff normalized p_line >= t, else other.')
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
assert hashlib.sha256(_resolve_model_artifact(source).read_bytes()).hexdigest() == digest
print(json.dumps(summary, ensure_ascii=True, indent=2))
