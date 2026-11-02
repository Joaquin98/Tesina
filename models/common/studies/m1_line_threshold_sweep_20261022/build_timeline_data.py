"""Prepare compact, verified timeline data from the delivered experiment tables."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks
HERE = experiment_directory(__file__)
OUT = HERE / 'results'
VIS = HERE / 'm1-lineout-timelines.html'
pred = pd.read_csv(_resolve_model_artifact(OUT / 'candidate_events.csv.gz'))
frames = pd.read_csv(_resolve_model_artifact(OUT / 'derived_predictions.csv.gz'))
metrics = pd.read_csv(_resolve_model_artifact(OUT / 'metrics_by_match.csv'))
policies = ['original', 'minus_0.05', 'minus_0.15', 'argmax']
data = {}
for match in ['V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'VF']:
    vf = match == 'VF'
    scope = 'VF_single_sections' if vf else 'PV_heldout'
    mask = frames.scope.eq(scope) & (frames.match.str.startswith('VF_') if vf else frames.match.eq(match))
    base = frames[mask & frames.policy.eq('original')]
    group = dict(sections={}, thresholds={}, metrics={})
    for policy in policies:
        mm = metrics[metrics.scope.eq(scope) & metrics.policy.eq(policy) & (metrics.match.str.startswith('VF_') if vf else metrics.match.eq(match))]
        group['thresholds'][policy] = {'line': None if policy == 'argmax' else float(mm.line_threshold.iloc[0]), 'scrum': float(mm.scrum_threshold.iloc[0])}
        group['metrics'][policy] = {c: dict(tp=int(mm[c + '_tp'].sum()), fp=int(mm[c + '_fp'].sum()), fn=int(mm[c + '_fn'].sum()), f1=float(mm[c + '_temporal_f1'].mean())) for c in ['line', 'scrum']}
    for section, f in base.groupby('section', sort=True):
        source_match = f.match.iloc[0]
        sectiondata = {'end': float(f.end_s.max())}
        for c in ['line', 'scrum']:
            alltruth = events_from_blocks(base if not vf else f, c, prediction_column='y_true', probability_column='p_' + c)
            truth = [dict(id=i, start=round(e.start, 3), end=round(e.end, 3), anchor=round(e.anchor, 3)) for i, e in enumerate(alltruth) if e.section == section]
            ep = pred[(pred.scope == scope) & (pred.match == source_match) & (pred.label == c) & (pred.section == section)]
            base_ids = set(ep[(ep.policy == 'original') & (ep.status == 'TP')].truth_id.dropna().astype(int))
            for t in truth:
                t['baseline'] = t['id'] in base_ids
            pp = {}
            for policy in policies:
                pp[policy] = []
                for r in ep[ep.policy == policy].itertuples():
                    tid = None if pd.isna(r.truth_id) else int(r.truth_id)
                    overlap = [] if pd.isna(r.baseline_overlapping_ids) else str(r.baseline_overlapping_ids).split(',')
                    pp[policy].append(dict(start=round(r.start_s, 3), end=round(r.end_s, 3), anchor=round(r.anchor_s, 3), status=r.status, truth=tid, distance=None if pd.isna(r.distance_s) else round(r.distance_s, 3), credit=round(r.weighted_credit, 5), recovered=tid is not None and tid not in base_ids, merged=len(overlap) > 1))
            sectiondata[c] = dict(truth=truth, pred=pp)
        group['sections'][section] = sectiondata
    data[match] = group
payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
(HERE / 'timeline_data.json').write_text(payload, encoding='utf-8')
fragment = _resolve_model_artifact(VIS).read_text(encoding='utf-8')
if '__TIMELINE_DATA__' in fragment:
    fragment = fragment.replace('__TIMELINE_DATA__', payload)
else:
    import re
    fragment = re.sub('(<script type="application/json" id="m1-timeline-data">).*?(</script>)', lambda m: m[1] + payload + m[2], fragment, flags=re.S)
VIS.write_text(fragment, encoding='utf-8')
assert VIS.stat().st_size < 1000000
assert '\\"' not in fragment and '\\n' not in fragment
print(json.dumps(dict(matches=len(data), sections=sum((len(g['sections']) for g in data.values())), fragment_bytes=VIS.stat().st_size)))
