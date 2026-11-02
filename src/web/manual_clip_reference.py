"""Consolidate manual clip decisions without altering their source sessions."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import copy
import hashlib
import json

def consolidate_clip_sessions(sessions):
    """Keep the latest explicit decision for each block on the latest session's grid.

    Callers must supply only clip-generator sessions for the same source video.
    Incompatible block grids remain separate; empty decisions never erase labels.
    Per-block review timestamps prevent a later save of an older session from
    reverting unrelated, newer corrections.
    """
    sessions = [s for s in sessions if any((c.get('final_label') in {'line', 'scrum', 'other'} for c in s.get('candidates', [])))]
    if not sessions:
        return None
    sessions = sorted(sessions, key=lambda s: (str(s.get('updated_at') or ''), str(s.get('session_id') or '')))
    latest = sessions[-1]

    def grid(session):
        return tuple(((int(c.get('candidate_index', i)), float(c.get('start_time', i * 7)), float(c.get('end_time', (i + 1) * 7))) for i, c in enumerate(session.get('candidates', []))))
    expected_grid = grid(latest)
    compatible = [s for s in sessions if grid(s) == expected_grid]
    merged = copy.deepcopy(latest)
    decisions = {}
    for session in compatible:
        for index, candidate in enumerate(session['candidates']):
            if candidate.get('final_label') not in {'line', 'scrum', 'other'}:
                continue
            rank = (str(candidate.get('reviewed_at') or session.get('updated_at') or ''), str(session.get('updated_at') or ''), str(session.get('session_id') or ''))
            if index not in decisions or rank > decisions[index][0]:
                decisions[index] = (rank, candidate, session['session_id'])
    for index, candidate in enumerate(merged['candidates']):
        decision = decisions.get(index)
        candidate['final_label'] = decision[1]['final_label'] if decision else None
        candidate['reviewed_at'] = decision[1].get('reviewed_at') if decision else None
        candidate['source_session_id'] = decision[2] if decision else None
    contributing = {d[2] for d in decisions.values()}
    merged['source_session_ids'] = [s['session_id'] for s in compatible if s['session_id'] in contributing]
    merged['composite_reference'] = len(contributing) > 1
    splits = sorted({s.get('dataset_split', 'train') for s in compatible if s['session_id'] in contributing})
    merged['dataset_split'] = splits[0] if len(splits) == 1 else 'mixed'
    merged['dataset_splits'] = splits
    signature = [(int(c.get('candidate_index', i)), c['final_label']) for i, c in enumerate(merged['candidates'])]
    merged['reference_sha256'] = hashlib.sha256(json.dumps(signature, separators=(',', ':')).encode()).hexdigest()
    merged['reference_id'] = 'labels_' + merged['reference_sha256'][:12]
    return merged
