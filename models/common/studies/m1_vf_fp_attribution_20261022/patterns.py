"""Descriptive diagnostics on VF; no model or production policy changes."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, sys
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from models.common.comparison import metrics as m
HERE = experiment_directory(__file__)
inventory = json.loads(_resolve_model_artifact(ROOT / 'data/m1_argmax_to_line_thresholds_vf_20261022/inventory.json').read_text())
f = pd.concat([pd.read_csv(_resolve_model_artifact(s['probability_path'])) for s in inventory['sources']], ignore_index=True)
labels = np.array(['line', 'other', 'scrum'])
p = m.normalize_probabilities(f[['p_line', 'p_other', 'p_scrum']].to_numpy())
f['y_final'] = labels[p.argmax(1)]
events = {}
truth = {}
scores = {}
diagnostics = []
trials = []
for c in ['line', 'scrum']:
    truth[c] = m.events_from_blocks(f, c, prediction_column='y_true', probability_column='p_' + c)
    events[c] = m.events_from_blocks(f, c, prediction_column='y_final', probability_column='p_' + c)
    scores[c] = [float(f.loc[f.section.eq(e.section) & f.block_index.isin(e.blocks), 'p_' + c].max()) for e in events[c]]
    matched = {j for _, j, _, _ in m.match_events(truth[c], events[c])}
    for j, e in enumerate(events[c]):
        nearby = [(z, t) for z, t in enumerate(events[c]) if z in matched and t.section == e.section and (z != j)]
        other = min(nearby, key=lambda z: abs(z[1].anchor - e.anchor)) if nearby else None
        diagnostics.append(dict(label=c, status='TP' if j in matched else 'FP', section=e.section, start=e.start, end=e.end, anchor=e.anchor, confidence=scores[c][j], blocks=len(e.blocks), nearest_other_tp_anchor_distance=abs(other[1].anchor - e.anchor) if other else None, nearest_other_tp_confidence=scores[c][other[0]] if other else None))
for window in [0, 14, 21, 28, 35]:
    r = dict(suppression_window_seconds=window)
    for c in ['line', 'scrum']:
        kept = []
        for j in sorted(range(len(events[c])), key=lambda j: (-scores[c][j], events[c][j].anchor)):
            e = events[c][j]
            if not any((events[c][k].section == e.section and abs(events[c][k].anchor - e.anchor) <= window for k in kept)):
                kept.append(j)
        pred = [events[c][j] for j in sorted(kept)]
        pairs = m.match_events(truth[c], pred)
        r.update({c + '_tp': len(pairs), c + '_fp': len(pred) - len(pairs), c + '_fn': len(truth[c]) - len(pairs), c + '_temporal_f1': 2 * sum((t[3] for t in pairs)) / (len(truth[c]) + len(pred))})
    r['macro_temporal_f1'] = (r['line_temporal_f1'] + r['scrum_temporal_f1']) / 2
    trials.append(r)
attrib = pd.read_csv(_resolve_model_artifact(HERE / 'results/atribucion_cada_fp.csv'))
alpha_trials = []
for alpha in [0.475, 0.5, 0.525, 0.55, 0.6, 0.65]:
    changed = []
    for r in attrib.itertuples():
        dual = np.array([getattr(r, 'p_dual_' + c) for c in labels])
        geom = np.array([getattr(r, 'p_geometry_' + c) for c in labels])
        fused = (1 - alpha) * np.log(np.clip(dual, 1e-07, 1)) + alpha * np.log(np.clip(geom, 1e-07, 1))
        if labels[fused.argmax()] != r.label:
            changed.append(r.fp_number)
    alpha_trials.append(dict(alpha=alpha, changed_fp_anchor_count=len(changed), fp_numbers=changed))
d = pd.DataFrame(diagnostics)
t = pd.DataFrame(trials)
assert t.iloc[0].line_tp == 23 and t.iloc[0].line_fp == 9 and (t.iloc[0].scrum_tp == 10) and (t.iloc[0].scrum_fp == 2)
summary = d.groupby(['label', 'status']).agg(events=('confidence', 'count'), median_confidence=('confidence', 'median'), min_confidence=('confidence', 'min'), max_confidence=('confidence', 'max'), single_block=('blocks', lambda x: int((x == 1).sum())))
out = HERE / 'patterns_20261024'
out.mkdir(exist_ok=True)
d.to_csv(out / 'event_diagnostics.csv', index=False)
t.to_csv(out / 'temporal_suppression_trials.csv', index=False)
(out / 'alpha_at_fp_anchors.json').write_text(json.dumps(alpha_trials, indent=2))
print('CONFIDENCE:', summary.to_string())
print('TEMPORAL TRIALS:', t.to_string(index=False))
print('ALPHA ANCHORS ONLY:', json.dumps(alpha_trials))
print('FP DETAIL:', d[d.status.eq('FP')].to_string(index=False))
