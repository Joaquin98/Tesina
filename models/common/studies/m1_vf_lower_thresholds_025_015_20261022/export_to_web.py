"""Publish the already evaluated VF variants into the local web result history."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import Counter
import json, hashlib, sys
import numpy as np
import pandas as pd
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from models.common.comparison import metrics as m
from models.common.comparison.export_predictions_to_web import _event, _window
HERE = experiment_directory(__file__)
EXPORT = HERE / 'web_export'
EXPORT.mkdir(exist_ok=True)
RESULTS = ROOT / 'data/results'
COMPARISONS = RESULTS / 'comparisons'
COMPARISONS.mkdir(exist_ok=True)
source = HERE / 'results/derived_predictions.csv.gz'
digest = hashlib.sha256(_resolve_model_artifact(source).read_bytes()).hexdigest()
frame = pd.read_csv(_resolve_model_artifact(source))
metrics = pd.read_csv(_resolve_model_artifact(HERE / 'results/match_metrics.csv'))
base = frame[frame.policy == 'argmax'].copy()
assert len(base) == 688
original = json.loads(_resolve_model_artifact(ROOT / 'data/raw/youtube/Jose_Ricardo_Aguero/manifest_JneZwNgtDYI.json').read_text())
video = ROOT / original['original']['relative_path']
assert _resolve_model_artifact(video).is_file()
offsets = {}
coverage = 0.0
for sec, s in base.groupby('section', sort=True):
    offsets[sec] = coverage
    coverage += float(s.end_s.max())
assert np.isclose(coverage, 4804.8)
manual_labels = []
for label in ['line', 'scrum']:
    for i, e in enumerate(m.events_from_blocks(base, label, prediction_column='y_true', probability_column='p_' + label)):
        manual_labels.append(dict(final_label=label, start_time=offsets[e.section] + e.start, end_time=offsets[e.section] + e.end, source_section=e.section, source_start_time=e.start, source_end_time=e.end, event_id=f'truth_{label}_{i}'))
manual = dict(composite_reference=True, labels=sorted(manual_labels, key=lambda x: x['start_time']), summary=dict(unit='events', reviewed=40, total_candidates=40, final_counts={'line': 30, 'scrum': 10}), provenance='Joined existing verified manual labels for VF S01-S04; no new review session or label changes.', coverage_seconds=coverage)
now = datetime.now(timezone(timedelta(hours=-3)))
record_id = 'm1_vf80_argmax_thresholds_20261022'
comparison_path = COMPARISONS / f'comparison_{record_id}.json'
if _resolve_model_artifact(comparison_path).exists() or _resolve_model_artifact(EXPORT / 'manifest.json').exists():
    raise FileExistsError('This export already exists; refusing to overwrite')
variants = [('argmax', None)] + [('threshold', t) for t in [0.45, 0.35, 0.25, 0.15]]
results = {}
exported = []
for k, (policy, t) in enumerate(variants):
    f = frame[frame.policy == policy].copy()
    metric = metrics[metrics.policy == policy]
    if t is not None:
        f = f[np.isclose(f.line_threshold, t)]
        metric = metric[np.isclose(metric.line_threshold, t)]
    assert len(f) == 688 and len(metric) == 1
    metric = metric.iloc[0]
    name = 'M1-H2-D2 · Argmax · VF 80 min' if t is None else f'M1-H2-D2 · Lineout ≥ {t:.2f} · VF 80 min'
    f['predicted_event_id'] = None
    detections = []
    for label in ['line', 'scrum']:
        ev = m.events_from_blocks(f, label, prediction_column='y_final', probability_column='p_' + label)
        assert len(ev) == int(metric[label + '_candidates'])
        for n, e in enumerate(ev):
            event_id = f'{e.section}_{label}_{n:03d}'
            mask = f.section.eq(e.section) & f.block_index.isin(e.blocks)
            f.loc[mask, 'predicted_event_id'] = event_id
            selected = f.loc[mask].copy()
            selected['start_s'] += offsets[e.section]
            selected['end_s'] += offsets[e.section]
            event = _event(event_id, selected, coverage)
            event.update(source_section=e.section, source_start_time=e.start, source_end_time=e.end, anchor_time=offsets[e.section] + e.anchor, decision_policy=policy, line_threshold=t)
            detections.append(event)
    detections.sort(key=lambda e: (e['start_time'], e['end_time'], e['play_type']))
    f['start_s'] = f.section.map(offsets) + f.start_s
    f['end_s'] = f.section.map(offsets) + f.end_s
    windows = [_window(r, coverage) for _, r in f.iterrows()]
    for w, (_, r) in zip(windows, f.iterrows()):
        w['source_section'] = r.section
    assert all((0 <= w['start_time'] < w['end_time'] <= coverage for w in windows))
    counts = dict(Counter((e['play_type'] for e in detections)))
    avg = sum((e['confidence'] for e in detections)) / len(detections)
    result = dict(status='success', video_path=str(video.resolve()), model_name=name, detections=detections, primary_detections=detections, window_predictions=windows, review_detections_7s=windows, manual_comparison=manual, statistics=dict(avg_confidence=avg, avg_duration=sum((e['duration'] for e in detections)) / len(detections), by_type={label: dict(count=counts.get(label, 0), total_duration=sum((e['duration'] for e in detections if e['play_type'] == label))) for label in ['line', 'scrum']}, block_counts=dict(Counter(f.y_final)), total_windows=688), summary=dict(total_detections=len(detections), plays_by_type=counts, average_confidence=avg, predicted_class='sequence', overall_confidence=avg), metadata=dict(duration=coverage, source_video_duration=original['original']['duration_seconds'], evaluation_coverage_seconds=coverage, coverage_note='VF S01–S04 · 80:05 evaluados. S05 excluido.', source_sections=list(offsets), section_offsets_seconds=offsets, confidence_threshold=0.0, line_confidence_threshold=t, scrum_confidence_threshold=None, saved_decisions_locked=True, analysis_mode='imported_saved_probability_experiment', experimental_result=True, source_configuration='ALL-DEVELOPMENT-M1-H2-D2-S42', decision_policy='argmax' if t is None else 'absolute_line_threshold_preserve_argmax_scrum', source_probabilities_sha256=digest, source_file=str(source.resolve())), event_post_processing=dict(mode='saved_experiment_event_groups', target_labels=['line', 'scrum'], block_duration_seconds=7.0, section_boundaries_preserved=True, line_threshold=t, scrum_policy='preserve_argmax_scrum', note='Exact stored experimental decisions; no generic rethresholding or event merging applied.'), fusion_metadata=dict(model_family='M1', tuning_level='H2', data_regime='D2', architecture='RGB + pose + geometría', version='single all-development refit', section='S01-S04'), evaluation={col: float(metric[col]) for col in metrics.columns if col not in ['policy', 'line_threshold'] and (not pd.isna(metric[col]))})
    rid = (now + timedelta(seconds=k)).strftime('%Y%m%d_%H%M%S')
    path = RESULTS / f'result_{rid}.json'
    if _resolve_model_artifact(path).exists():
        raise FileExistsError(path)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    (EXPORT / f'result_{rid}.json').write_text(encoded, encoding='utf-8')
    path.write_text(encoded, encoding='utf-8')
    exported.append(dict(result_id=rid, name=name, path=str(path), viewer_url=f'http://127.0.0.1:5000/viewer?result_id={rid}', counts=counts, sha256=hashlib.sha256(_resolve_model_artifact(path).read_bytes()).hexdigest()))
    results[name] = result
comparison = dict(id=record_id, created_at=now.isoformat(), video_path=str(video.resolve()), video_name='VF · Jockey Club VT vs Alma Juniors · S01–S04 · 80:05', model_names=list(results), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0, saved_decisions_locked=True, coverage_seconds=coverage), results=results, comparison={name: dict(total=len(r['detections']), avg_confidence=r['statistics']['avg_confidence']) for name, r in results.items()})
encoded = json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False)
(EXPORT / comparison_path.name).write_text(encoded, encoding='utf-8')
comparison_path.write_text(encoded, encoding='utf-8')
manifest = dict(comparison_id=record_id, comparison_url=f'http://127.0.0.1:5000/compare?comparison={record_id}', comparison_path=str(comparison_path), results=exported, source_sha256=digest, seconds=coverage, manual_events={'line': 30, 'scrum': 10}, no_model_or_label_changes=True, no_inference=True)
(EXPORT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
assert hashlib.sha256(_resolve_model_artifact(source).read_bytes()).hexdigest() == digest
print(json.dumps(manifest, ensure_ascii=True, indent=2))
