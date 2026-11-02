import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, hashlib, csv
from PIL import Image
B = experiment_directory(__file__)
training = json.loads(_resolve_model_artifact(B / 'results/training.json').read_text())
assert training['all_checks_pass'] and training['epochs_completed'] == 5 and (training['training_examples'] == 8796)
for p, h in training['reference_hashes_after'].items():
    assert hashlib.sha256(_resolve_model_artifact(Path(p)).read_bytes()).hexdigest() == h
for p, h in training['new_weight_hashes'].items():
    assert hashlib.sha256(_resolve_model_artifact(Path(p)).read_bytes()).hexdigest() == h
qualities = []
for key, members in [('M1-ensemble', 6), ('M1-single', 1)]:
    complete = json.loads(_resolve_model_artifact(B / f'results/{key}_completion.json').read_text())
    assert complete['all_checks_pass'] and complete['hashes_unchanged']
    assert complete['graph_trace_counts'] == [1] * members
    result = json.loads(_resolve_model_artifact(B / f'results/{key}_result.json').read_text())
    t = result['metadata']['timing']
    q = json.loads(_resolve_model_artifact(B / f'results/{key}_quality.json').read_text())
    assert result['metadata']['duration'] == 1201.2 and t['counts']['_on_sampled_frame_calls'] == 30030
    assert t['counts']['recurrent_calls'] == 1195 * members and len(result['window_predictions']) == 1195
    rows = list(csv.DictReader(_resolve_model_artifact(B / f'results/{key}_quality_blocks.csv').open()))
    assert len(rows) == 172
    assert q['duration_seconds'] == 1201.2 and (not q['independent_test_claim_allowed'])
    for label, assignments in q['assignments'].items():
        assert len({a['truth_index'] for a in assignments}) == len(assignments)
        assert len({a['prediction_index'] for a in assignments}) == len(assignments)
        assert all((0 <= a['distance_seconds'] <= 35 and abs(a['credit'] - (1 - a['distance_seconds'] / 35)) < 1e-12 for a in assignments))
    qualities.append(q)
assert all((qualities[0]['events'][c]['truth'] == qualities[1]['events'][c]['truth'] for c in ['line', 'scrum']))
audit = json.loads(_resolve_model_artifact(B / 'figures/image_evidence_v2.json').read_text())
assert audit['axis_full_seconds'] == [0, 1200]
for p, h in audit['sources'].items():
    assert hashlib.sha256(_resolve_model_artifact(Path(p)).read_bytes()).hexdigest() == h
for f in audit['figures']:
    im = _resolve_model_artifact(Image).open(f['png'])
    assert im.size == (2560, 1440)
    assert len(f['rows']) == 3
    assert all((f['axis_seconds'][0] <= e['start_seconds'] < e['end_seconds'] <= f['axis_seconds'][1] for row in f['rows'] for e in row['events_in_display']))
report = {'all_checks_pass': True, 'training_examples': 8796, 'new_dual_and_forest': True, 'epochs': 5, 'graph_trace_counts': {'ensemble': [1] * 6, 'single': [1]}, 'no_original_weight_or_source_changes': True, 'same_video_and_grids': True, 'matching_one_to_one': True, 'human_truth': True, 'image_dimensions': [2560, 1440], 'independent_test_claim_allowed': False, 'feature_extraction_fresh_each_timed_run': True, 'slide_or_thesis_edits_by_this_experiment': False}
(B / 'final_validation.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
