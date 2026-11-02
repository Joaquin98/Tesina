"""Measure video duration to review using saved VF probabilities; no inference."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from models.common.comparison import metrics as m
HERE = experiment_directory(__file__)
SOURCE = ROOT / 'data/m1_argmax_to_line_thresholds_vf_20261022'
inventory = json.loads(_resolve_model_artifact(SOURCE / 'inventory.json').read_text(encoding='utf-8'))
frames = [pd.read_csv(_resolve_model_artifact(s['probability_path'])) for s in inventory['sources']]
frame = pd.concat(frames, ignore_index=True)
assert len(frame) == 688 and (not frame.duplicated(['section', 'block_index']).any())
durations = {s['section']: s['seconds'] for s in inventory['sources']}
duration = sum(durations.values())
assert abs(duration - 4804.8) < 1e-06
assert abs(frame.duration.sum() - duration) < 1e-06
offsets = {}
cursor = 0
for section, seconds in durations.items():
    offsets[section] = cursor
    cursor += seconds

def union_seconds(events, pad=0):
    intervals = sorted(((max(0, offsets[e.section] + e.start - pad), min(duration, offsets[e.section] + e.end + pad)) for e in events))
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum((b - a for a, b in merged))
probs = m.normalize_probabilities(frame[['p_line', 'p_other', 'p_scrum']].to_numpy())
labels = np.array(['line', 'other', 'scrum'])
rules = [('original', 0.5000000000000001), ('argmax', None)] + [('line_threshold', t) for t in [0.45, 0.35, 0.25, 0.15]]
rows, details, section_rows = ([], [], [])
for policy, threshold in rules:
    if policy == 'original':
        y = m.apply_class_thresholds(probs, line=threshold, scrum=0.4500000000000001)
        assert np.array_equal(labels[y], frame.y_final)
    elif policy == 'argmax':
        y = probs.argmax(1)
    else:
        y = np.where(probs.argmax(1) == 2, 2, np.where(probs[:, 0] >= threshold, 0, 1))
    data = frame.assign(y_final=labels[y])
    all_events, false_events, far_events, duplicate_events = ([], [], [], [])
    row = dict(policy=policy, line_threshold=threshold, duration_s=duration)
    for label in ['line', 'scrum']:
        truth = m.events_from_blocks(data, label, prediction_column='y_true', probability_column='p_' + label)
        events = m.events_from_blocks(data, label, prediction_column='y_final', probability_column='p_' + label)
        pairs = m.match_events(truth, events)
        matched = {j for i, j, distance, credit in pairs}
        false = [e for j, e in enumerate(events) if j not in matched]
        far = [e for e in false if not any((t.group == e.group and t.section == e.section and (abs(t.anchor - e.anchor) <= 35) for t in truth))]
        duplicates = [e for e in false if e not in far]
        row.update({label + '_truth': len(truth), label + '_tp': len(pairs), label + '_fn': len(truth) - len(pairs), label + '_fp': len(false), label + '_far_fp': len(far), label + '_duplicates': len(duplicates), label + '_fp_seconds': union_seconds(false), label + '_all_seconds': union_seconds(events)})
        all_events.extend(events)
        false_events.extend(false)
        far_events.extend(far)
        duplicate_events.extend(duplicates)
        for j, e in enumerate(events):
            details.append(dict(policy=policy, line_threshold=threshold, label=label, section=e.section, start_s=e.start, end_s=e.end, duration_s=e.end - e.start, global_start_s=offsets[e.section] + e.start, global_end_s=offsets[e.section] + e.end, status='TP' if j in matched else 'far_FP' if e in far else 'duplicate_FP'))
    for name, events in [('all', all_events), ('fp', false_events), ('far_fp', far_events), ('duplicates', duplicate_events)]:
        row[name + '_count'] = len(events)
        row[name + '_seconds'] = union_seconds(events)
        row[name + '_pct_video'] = union_seconds(events) / duration * 100
        row[name + '_seconds_context5'] = union_seconds(events, 5)
        row[name + '_seconds_context10'] = union_seconds(events, 10)
    row['unmarked_seconds'] = duration - row['all_seconds']
    row['detected_events'] = row['line_tp'] + row['scrum_tp']
    row['missed_events'] = row['line_fn'] + row['scrum_fn']
    assert row['all_seconds'] <= duration and row['fp_seconds'] <= row['all_seconds']
    assert abs(row['fp_seconds'] - row['far_fp_seconds'] - row['duplicates_seconds']) < 1e-06
    for section in durations:
        section_rows.append(dict(policy=policy, line_threshold=threshold, section=section, duration_s=durations[section], all_seconds=union_seconds([e for e in all_events if e.section == section]), fp_seconds=union_seconds([e for e in false_events if e.section == section])))
    rows.append(row)
prior = pd.read_csv(_resolve_model_artifact(ROOT / 'data/m1_vf_lower_thresholds_025_015_20261022/results/match_metrics.csv'))
for row in rows[1:]:
    ref = prior[prior.policy.eq('argmax')] if row['policy'] == 'argmax' else prior[prior.policy.eq('threshold') & np.isclose(prior.line_threshold, row['line_threshold'])]
    assert len(ref) == 1
    ref = ref.iloc[0]
    for label in ['line', 'scrum']:
        for suffix in ['tp', 'fp', 'fn', 'far_fp']:
            assert row[label + '_' + suffix] == ref[label + '_' + suffix]
        assert abs(row[label + '_all_seconds'] - ref[label + '_candidate_seconds']) < 1e-06
existing_fp = pd.read_csv(_resolve_model_artifact(ROOT / 'data/m1_vf_fp_attribution_20261022/fp_inventory.csv'))
for row in rows:
    key = row['policy'] if row['policy'] != 'line_threshold' else f"line_{row['line_threshold']:.2f}"
    ref = existing_fp[existing_fp.policy.eq(key)]
    assert len(ref) == row['fp_count']
    assert abs((ref.end_s - ref.start_s).sum() - row['fp_seconds']) < 1e-06
pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
pd.DataFrame(details).to_csv(HERE / 'events.csv', index=False)
pd.DataFrame(section_rows).to_csv(HERE / 'by_section.csv', index=False)
audit = dict(model=inventory['model']['source_configuration'], coverage='VF S01-S04, 80 min 4.8 s; S05 excluded', matching='Original one-to-one event matching within 35 seconds, separately per section', interpretation='Video duration at 1x, not measured human task completion time; excludes searches for missed events, pauses and editing. Context windows are hypothetical.', source_hashes={s['probability_path']: hashlib.sha256(_resolve_model_artifact(Path(s['probability_path'])).read_bytes()).hexdigest() for s in inventory['sources']}, existing_metrics_reproduced=True, existing_false_positive_inventory_reproduced=True, independence=inventory['independence'], results=rows)
(HERE / 'analysis.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
print(pd.DataFrame(rows)[['policy', 'line_threshold', 'all_count', 'fp_count', 'fp_seconds', 'far_fp_seconds', 'all_seconds', 'all_pct_video', 'detected_events', 'missed_events', 'all_seconds_context5']].to_string(index=False))
