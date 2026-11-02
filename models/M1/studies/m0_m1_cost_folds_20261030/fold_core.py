"""Pure held-out indices, training-only class costs and section-safe events."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys
import numpy as np
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(source_directory(ROOT / 'data/m1_recall_cost_20261027')))
from core import evaluate_events
from review_coverage import review_coverage
from models.common.comparison.protocol import PV_FOLDS

def fold_indices(frame, fold):
    if frame.block_id.duplicated().any():
        raise ValueError('Duplicate block identity')
    spec = next((f for f in PV_FOLDS if f.fold == fold))
    cv = frame.domain.eq('CV')
    train = frame.domain.eq('CH') | cv & frame.group_id.isin(spec.train_groups)
    dev = cv & frame.group_id.eq(spec.development_group)
    test = cv & frame.group_id.eq(spec.test_group)
    parts = tuple((np.flatnonzero(s.to_numpy()) for s in [train, dev, test]))
    assert all((len(p) for p in parts)) and sum(map(len, parts)) == len(frame)
    assert all((not set(parts[i]) & set(parts[j]) for i, j in [(0, 1), (0, 2), (1, 2)]))
    return parts

def class_weights(y, line, scrum):
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1, 2]).all() or min(line, scrum) < 1:
        raise ValueError('Invalid labels or costs')
    counts = np.bincount(y.astype(int), minlength=3)
    if not counts.all():
        raise ValueError('Missing training class')
    w = len(y) / (3 * counts) * np.array([line, 1, scrum])
    return w / np.average(w, weights=counts)

def selected_indices(global_ids, shuffle, batch, size):
    return np.asarray(global_ids)[np.asarray(shuffle)[batch * size:(batch + 1) * size]]

def section_offsets(frame):
    offsets = {}
    cursor = 0.0
    for section in sorted(frame.section.unique()):
        rows = frame[frame.section == section]
        if rows.start_s.min() < 0 or (rows.end_s <= rows.start_s).any():
            raise ValueError('Invalid block times')
        offsets[section] = cursor
        cursor += float(rows.end_s.max())
    return (offsets, cursor)

def grouped_events(frame, labels, probabilities=None):
    if len(frame) != len(labels):
        raise ValueError('Labels do not align with frame')
    rows = frame.copy()
    rows['_label'] = list(labels)
    rows['_score'] = 1.0 if probabilities is None else np.asarray(probabilities).max(axis=1)
    offsets, _ = section_offsets(frame)
    events = []
    for section in sorted(rows.section.unique()):
        active = None
        for r in rows[rows.section == section].sort_values('block_index', kind='stable').to_dict('records'):
            label = r['_label']
            a = float(r['start_s']) + offsets[section]
            z = float(r['end_s']) + offsets[section]
            if label not in ['line', 'scrum']:
                active = None
                continue
            if active is not None and active['play_type'] == label and (active['last_block'] + 1 == int(r['block_index'])) and (abs(active['end_time'] - a) < 1e-06):
                active.update(end_time=z, duration=z - active['start_time'], last_block=int(r['block_index']), confidence=max(active['confidence'], float(r['_score'])))
                active['blocks'].append(r['block_id'])
            else:
                active = dict(play_type=label, section=section, start_time=a, end_time=z, duration=z - a, first_block=int(r['block_index']), last_block=int(r['block_index']), confidence=float(r['_score']), blocks=[r['block_id']])
                events.append(active)
    return events

def evaluate(frame, truth_labels, predicted_labels, probabilities=None):
    gold = grouped_events(frame, truth_labels)
    truth = [dict(id=f"{e['section']}_{e['first_block']}_{e['play_type']}", play_type=e['play_type'], section=e['section'], start_s=e['start_time'], end_s=e['end_time'], blocks=e['blocks']) for e in gold]
    events = grouped_events(frame, predicted_labels, probabilities)
    result = evaluate_events(truth, events)
    result.update(reference_segments=truth, predictions=events, review_coverage=review_coverage(truth, events), evaluated_video_s=section_offsets(frame)[1])
    return result
