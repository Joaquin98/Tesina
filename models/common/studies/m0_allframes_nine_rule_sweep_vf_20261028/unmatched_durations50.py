import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
ROOT = Path('D:/Tesina/entregas/repositorio_tesina_20261022')
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks, match_events
H = experiment_directory(__file__)
P = ROOT / 'data/m0_h2_d2_allframes_nine_vf_20261027'
M = ROOT / 'data/multicrop_d2_stride25_vf_20261026/manifest.json'
labels = ['line', 'other', 'scrum']
rows = [r for r in json.loads(_resolve_model_artifact(M).read_text(encoding='utf8')) if r['domain'] == 'VF' and r['split'] == 'test']
base = pd.DataFrame(rows)
y = base.tag
probs = np.stack([_load_model_result_array(P / 'blocks' / f'{i:04d}.npy') for i in range(688)])
assert probs.shape == (688, 6, 9, 3)
records = []
summaries = []
for fold in range(6):
    p = probs[:, fold]
    central = p[:, 0].argmax(1)
    votes = p[:, 1:].argmax(2) == 0
    hi = votes & (p[:, 1:, 0] >= 0.5)
    pred = np.where((central == 1) & (votes.sum(1) >= 2) & (hi.sum(1) >= 1), 0, central)
    assert np.array_equal(pred == 2, central == 2)
    frame = base.copy()
    frame['y_true'] = frame.tag
    frame['y_final'] = [labels[i] for i in pred]
    frame['p_line'] = p[:, 0, 0]
    frame['p_scrum'] = p[:, 0, 2]
    score = np.sort(np.where(votes, p[:, 1:, 0], 0), axis=1)[:, -2]
    frame.loc[pred != central, 'p_line'] = score[pred != central]
    per = []
    for lab in ['line', 'scrum']:
        truth = events_from_blocks(frame, lab, prediction_column='y_true', probability_column='p_' + lab)
        det = events_from_blocks(frame, lab, prediction_column='y_final', probability_column='p_' + lab)
        matches = match_events(truth, det)
        matched = {j for _, j, _, _ in matches}
        for j, e in enumerate(det):
            if j not in matched:
                d = {'fold': fold + 1, 'label': lab, 'group': e.group, 'section': e.section, 'start_s': e.start, 'end_s': e.end, 'duration_s': e.end - e.start, 'blocks': len(e.blocks)}
                records.append(d)
                per.append(d)
    for lab in ['line', 'scrum']:
        a = [r for r in per if r['label'] == lab]
        summaries.append({'fold': fold + 1, 'label': lab, 'unmatched_count': len(a), 'duration_sum_s': sum((r['duration_s'] for r in a))})
    duration = 0
    for section in sorted({r['section'] for r in per}):
        spans = sorted(((r['start_s'], r['end_s']) for r in per if r['section'] == section))
        start = end = None
        for a, b in spans:
            if start is None:
                start, end = (a, b)
            elif a <= end:
                end = max(end, b)
            else:
                duration += end - start
                start, end = (a, b)
        if start is not None:
            duration += end - start
    summaries.append({'fold': fold + 1, 'label': 'union_line_scrum', 'unmatched_count': len(per), 'duration_sum_s': duration})
d = pd.DataFrame(records)
s = pd.DataFrame(summaries)
d.to_csv(H / 'rule50_unmatched_events_durations.csv', index=False)
s.to_csv(H / 'rule50_unmatched_duration_summary.csv', index=False)
prior = pd.read_csv(_resolve_model_artifact(H / 'metrics_by_fold.csv'))
prior = prior[(prior.high == 0.5) & (prior.low == 0) & (prior.min_votes == 2) & (prior.central_min == 0)]
for row in s[s.label == 'line'].itertuples():
    assert row.unmatched_count == int(prior[prior.fold == row.fold].iloc[0].event_unmatched)
print(s.to_string(index=False))
print('MEAN')
print(s.groupby('label').mean(numeric_only=True).to_string())
print('distribution')
print(d.groupby('label').duration_s.agg(['count', 'min', 'median', 'max', 'sum']).to_string())
