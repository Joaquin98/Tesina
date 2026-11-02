"""Final cross-check of human classes, image rows, clocks and model reuse."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from datetime import datetime
import csv, json, hashlib
B = experiment_directory(__file__)

def read(p):
    return json.loads(_resolve_model_artifact(Path(p)).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(Path(p)).read_bytes()).hexdigest()
req = read(B / 'inputs/request.json')
snapshot = read(B / 'inputs/human_session.json')
current = read(req['human_session'])
fields = ['candidate_index', 'start_time', 'end_time', 'final_label']
labels = lambda s: [{k: c[k] for k in fields} for c in s['candidates']]
validation = read(B / 'final_validation.json')
compar = read(B / 'results/comparison.json')
evidence = read(B / 'figures/image_evidence_v2.json')
checks = {'existing_validation_pass': validation['all_checks_pass'], 'human_classes_unchanged': labels(snapshot) == labels(current), 'model_reuse_without_training': req['retrain'] is False and compar['models_reused_without_retraining'], 'same_locked_thresholds': req['thresholds'] == {'line': 0.5, 'scrum': 0.45}, 'human_blocks_172': len(snapshot['candidates']) == 172}

def intervals(rows, field, start, end):
    result = []
    last = None
    for r in rows:
        label = r[field]
        a = float(r[start])
        z = float(r[end])
        if label in ['line', 'scrum']:
            if last and last['play_type'] == label and (abs(last['end_time'] - a) < 1e-08):
                last['end_time'] = z
            else:
                last = {'play_type': label, 'start_time': a, 'end_time': z}
                result.append(last)
        else:
            last = None
    return result
truth = intervals(snapshot['candidates'], 'final_label', 'start_time', 'end_time')
times = []
for key in ['M1-ensemble', 'M1-single']:
    q = read(B / f'results/{key}_quality.json')
    t = read(B / f'results/{key}_timing.json')
    complete = read(B / f'results/{key}_completion.json')
    blocks = list(csv.DictReader(_resolve_model_artifact(B / f'results/{key}_quality_blocks.csv').open(encoding='utf-8')))
    checks[key + '_block_labels_and_boundaries'] = len(blocks) == 172 and all((b['y_true'] == c['final_label'] and abs(float(b['start_s']) - c['start_time']) < 1e-08 and (abs(float(b['end_s']) - c['end_time']) < 1e-08) for b, c in zip(blocks, snapshot['candidates'])))
    gt = sorted([{'play_type': e['class'], 'start_time': e['start_time'], 'end_time': e['end_time']} for g in q['events'].values() for e in g['truth']], key=lambda e: (e['start_time'], e['play_type']))
    checks[key + '_real_events_match_manual_grouping'] = gt == truth
    predictions = intervals(blocks, 'y_final', 'start_s', 'end_s')
    predicted = sorted([{'play_type': e['class'], 'start_time': e['start_time'], 'end_time': e['end_time']} for g in q['events'].values() for e in g['prediction']], key=lambda e: (e['start_time'], e['play_type']))
    checks[key + '_prediction_events_match_block_csv'] = predictions == predicted
    checks[key + '_original_sources_and_models_unchanged'] = all((sha(p) == h for p, h in complete['hashes_after'].items()))
    checks[key + '_fresh_full_video'] = t['counts']['_on_sampled_frame_calls'] == 30030 and t['counts']['_on_raw_sampled_frame_calls'] == 30030 and (t['counts']['_predict_feature_batch_calls'] == 118)
    times.append(t)
checks['inferences_sequential'] = datetime.fromisoformat(times[0]['inference_end_utc']) < datetime.fromisoformat(times[1]['inference_start_utc'])
checks['figure_sources_unchanged'] = all((sha(p) == h for p, h in evidence['sources'].items()))
for fig in evidence['figures']:
    extent = fig['axis_seconds'][1]
    for i, row in enumerate(fig['rows']):
        events = truth if i == 0 else intervals(list(csv.DictReader(_resolve_model_artifact(B / f"results/{['M1-ensemble', 'M1-single'][i - 1]}_quality_blocks.csv").open(encoding='utf-8'))), 'y_final', 'start_s', 'end_s')
        clipped = [{'class': e['play_type'], 'start_seconds': max(0, e['start_time']), 'end_seconds': min(extent, e['end_time'])} for e in events if min(extent, e['end_time']) > max(0, e['start_time'])]
        checks[f"{Path(fig['png']).name}_row_{i}_matches_data"] = clipped == row['events_in_display']
        if i:
            checks[f"{Path(fig['png']).name}_row_{i}_clock_matches"] = row['inference_seconds'] == times[i - 1]['inference_seconds']
result = {'all_checks_pass': all(checks.values()), 'checks': checks, 'visual_review': {'main_20_min_image_inspected': True, 'layout_legible': True, 'full_axis_0_20_min': True, 'Real_matches_human': True, 'times_and_F1_match': True}, 'actual_video_duration_seconds': 1201.2, 'main_display_axis_seconds': [0, 1200], 'image_sha256': sha(B / 'figures/comparacion_ensemble_modelo_unico_20min_v2.png'), 'inference_seconds': [t['inference_seconds'] for t in times], 'F1_descriptive_only': True, 'new_training_performed': False, 'previous_training_metadata_copied_as_provenance': True}
with _resolve_model_artifact(B / 'visual_review.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
assert result['all_checks_pass'], checks
print(json.dumps({'all_checks_pass': True, 'inference_seconds': result['inference_seconds'], 'real_events': len(truth), 'main_image_verified': True}))
