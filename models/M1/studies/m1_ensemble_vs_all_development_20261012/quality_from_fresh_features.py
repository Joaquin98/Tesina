"""Thesis event metrics on the exact human 7-second blocks, after timing.

Full streaming timing keeps the existing 7 s / 1 s-stride application route.
This separate quality pass uses the fresh features captured in RAM from that
same run, including the final partial human block, and the thesis 7 s grid.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.M1.studies.m1_ensemble_vs_all_development_20261012.runtime import HERE, ROOT, save, sha
from pathlib import Path
import json, time, sys, csv
import numpy as np, pandas as pd
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import temporal_metrics, block_metrics, apply_class_thresholds, events_from_blocks, match_events
from models.common.comparison.protocol import LABELS, LABEL_TO_ID

def evaluate(detector, rgb_batches, key, thresholds, video):
    start = time.perf_counter()
    features = np.concatenate(rgb_batches, axis=0)
    assert features.shape == (30030, 2048)
    p = ROOT / 'data/active_learning_review/sessions/review_session_20260830_112524_0b5e58.json'
    csv_path = ROOT / 'data/active_learning_dataset/sessions/20260830_112524_0b5e58/train_manifest.csv'
    session = json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8'))
    blocks = list(csv.DictReader(_resolve_model_artifact(csv_path).open(encoding='utf-8-sig')))
    assert Path(session['video_path']).resolve() == video.resolve() and session['status'] == 'exported'
    assert len(blocks) == len(session['candidates']) == 172
    candidates = {int(c['candidate_index']): c for c in session['candidates']}
    rows = []
    probabilities = []
    for b in blocks:
        c = candidates[int(b['candidate_index'])]
        assert c['final_label'] == b['correct_label'] and c['manual_labeling'] and c['reviewed_at']
        begin = float(b['start_time'])
        end = float(b['end_time'])
        a = int(round(begin * 25))
        z = min(len(features), int(round(end * 25)))
        length = z - a
        assert 0 < length <= 200
        padded = np.zeros((200, 2048), np.float32)
        padded[:length] = features[a:z]
        mask = np.zeros(200, bool)
        mask[:length] = True
        value = detector._predict_sequence_window(padded, mask, a, z)[0]
        assert np.isfinite(value).all()
        probabilities.append(value)
        rows.append({'block_id': b['dataset_id'], 'group_id': 'VF', 'section': 'S01', 'block_index': int(b['candidate_index']), 'start_s': begin, 'end_s': end, 'duration': end - begin, 'y_true': b['correct_label']})
    frame = pd.DataFrame(rows)
    values = np.asarray(probabilities)
    frame['y_final'] = [LABELS[int(v)] for v in apply_class_thresholds(values, line=thresholds['line'], scrum=thresholds['scrum'])]
    for i, label in enumerate(LABELS):
        frame['p_' + label] = values[:, i]
    frame.to_csv(HERE / f'results/{key}_quality_blocks.csv', index=False)
    metrics = temporal_metrics(frame)
    matches = {}
    exports = {}
    denom = float(frame.end_s.max())
    for label in ['line', 'scrum']:
        truth = events_from_blocks(frame, label, prediction_column='y_true', probability_column='p_' + label)
        predicted = events_from_blocks(frame, label, prediction_column='y_final', probability_column='p_' + label)
        assignments = match_events(truth, predicted)
        weighted_tp = sum((v[3] for v in assignments))
        assert abs(metrics[label]['weighted_f1'] - (2 * weighted_tp / (len(truth) + len(predicted)) if truth or predicted else 1)) < 1e-12
        metrics[label]['weighted_true_positive_credit'] = weighted_tp
        metrics[label]['false_negative_events'] = len(truth) - len(assignments)
        metrics[label]['unmatched_predicted_events'] = len(predicted) - len(assignments)
        metrics[label]['far_false_positive_count'] = round(metrics[label]['far_false_positives_per_hour'] * denom / 3600)
        matches[label] = [{'truth_index': i, 'prediction_index': j, 'distance_seconds': d, 'credit': w} for i, j, d, w in assignments]
        exports[label] = {name: [{'class': e.label, 'start_time': e.start, 'end_time': e.end, 'anchor': e.anchor, 'blocks': list(e.blocks)} for e in events] for name, events in [('truth', truth), ('prediction', predicted)]}
    result = {'temporal': metrics, 'block': block_metrics(frame.y_true.map(LABEL_TO_ID).to_numpy(), values), 'quality_seconds': time.perf_counter() - start, 'grid': '172 exact human blocks, 171 of 7 s plus final 4.2 s', 'timing_grid_is_separate': 'application sliding windows of 7 s with 1 s stride; quality not included in inference stopwatch', 'quality_features': 'fresh video extraction from the same run captured in RAM; no persisted or cross-model video cache', 'thresholds': thresholds, 'distance_tolerance_seconds': 35, 'matching': 'global one-to-one Hungarian credit maximization', 'weighted_F1_formula': '2*sum(max(0,1-distance/35))/(truth_events+predicted_events)', 'truth_anchor': 'midpoint of contiguous real event', 'prediction_anchor': 'midpoint of earliest highest-confidence block in predicted event', 'duration_seconds': denom, 'human_source_hashes': {str(path): sha(path) for path in [p, csv_path]}, 'assignments': matches, 'events': exports, 'independent_test_claim_allowed': False}
    save(f'results/{key}_quality.json', result)
    with _resolve_model_artifact(HERE / f'results/{key}_quality_events.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['source', 'class', 'start_time', 'end_time', 'anchor'])
        for label, groups in exports.items():
            for name, events in groups.items():
                for e in events:
                    w.writerow([name, label, e['start_time'], e['end_time'], e['anchor']])
    print('QUALITY_COMPLETE ' + json.dumps({'model': key, 'f1_mean': metrics['weighted_f1_macro'], 'quality_seconds': result['quality_seconds']}), flush=True)
    return result
