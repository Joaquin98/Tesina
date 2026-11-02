"""Pilot spatial-preprocessing sweep for the deployed RGB+YOLO model.

The historical scale sweep already materialized 224x224 clips at several
spatial scales.  This script reuses those clips, so each scale is evaluated by
the same deployed RGB+YOLO head (including YOLO pose features) without
re-encoding the source videos or retraining a classifier.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import contextlib
import csv
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, f1_score
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from models.M2.predictor import RGBYoloQuantileFusionDetector
SCALE_ROOTS = {'0.40': PROJECT_ROOT / 'data/resize_scale_sweep/scale_040/dataset_040_center224', '0.50': PROJECT_ROOT / 'data/videos/clips_224', '0.525': PROJECT_ROOT / 'data/resize_scale_sweep/scale_0525/dataset_0525_center224', '0.55': PROJECT_ROOT / 'data/resize_scale_sweep/scale_055/dataset_055_center224', '0.575': PROJECT_ROOT / 'data/resize_scale_sweep/scale_0575/dataset_0575_center224', '0.60': PROJECT_ROOT / 'data/resize_scale_sweep/scale_060/dataset_060_center224', '0.75': PROJECT_ROOT / 'data/resize_075_best_model/full/dataset_075_center224', '1.00': PROJECT_ROOT / 'data/resize_scale_sweep/scale_100/dataset_100_center224'}
MANIFEST = PROJECT_ROOT / 'data/resize_scale_sweep/scale_040/test_manifest.csv'
OUT_DIR = PROJECT_ROOT / 'data/joint_multimodal_20260813/preprocess_scale_sweep_rgb_yolo'

def selected_examples(per_class: int, all_cases: bool=False) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with _resolve_model_artifact(MANIFEST).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            rows.append((row['video_name'], row['tag']))
    if all_cases:
        return rows
    selected = []
    for label in ('line', 'scrum', 'other'):
        selected.extend(sorted(((name, tag) for name, tag in rows if tag == label))[:per_class])
    return selected

def evaluate_scale(detector, scale: str, examples: list[tuple[str, str]], quiet_log: io.TextIOBase, checkpoint_path: Path | None=None, max_new_clips: int | None=None) -> dict:
    root = SCALE_ROOTS[scale]
    records = []
    if checkpoint_path is not None and _resolve_model_artifact(checkpoint_path).exists():
        try:
            checkpoint = json.loads(_resolve_model_artifact(checkpoint_path).read_text(encoding='utf-8'))
            records = list(checkpoint.get('records', []))
        except (OSError, json.JSONDecodeError, TypeError):
            records = []
    completed = {str(record.get('video_name')) for record in records}
    processed_this_run = 0
    for video_name, expected in examples:
        if video_name in completed:
            continue
        path = root / video_name
        if not _resolve_model_artifact(path).exists():
            records.append({'scale': float(scale), 'video_name': video_name, 'expected': expected, 'error': 'missing_file'})
            if checkpoint_path is not None:
                checkpoint_path.write_text(json.dumps({'scale': float(scale), 'records': records}, ensure_ascii=False), encoding='utf-8')
            continue
        started = time.time()
        try:
            with contextlib.redirect_stdout(quiet_log), contextlib.redirect_stderr(quiet_log):
                result = detector.analyze_video_segments(str(path), segment_duration=7, confidence_threshold=0.3, temporal_fps_mode='fast_12', feature_batch_size=32, feature_predict_batch_size=4)
            windows = result.get('window_predictions', [])
            if not windows:
                raise RuntimeError('no_window_prediction')
            window = max(windows, key=lambda item: float(item.get('confidence', 0.0)))
            probabilities = window.get('probabilities', {})
            predicted = str(window.get('play_type', 'other'))
            records.append({'scale': float(scale), 'video_name': video_name, 'expected': expected, 'predicted': predicted, 'confidence': float(window.get('confidence', 0.0)), 'true_class_probability': float(probabilities.get(expected, 0.0)), 'line_probability': float(probabilities.get('line', 0.0)), 'scrum_probability': float(probabilities.get('scrum', 0.0)), 'other_probability': float(probabilities.get('other', 0.0)), 'detections': len(result.get('detections', [])), 'seconds': round(time.time() - started, 3)})
        except Exception as exc:
            print(f'[{scale}] {video_name}: ERROR {exc}', file=quiet_log)
            records.append({'scale': float(scale), 'video_name': video_name, 'expected': expected, 'error': repr(exc), 'seconds': round(time.time() - started, 3)})
        completed.add(video_name)
        processed_this_run += 1
        if checkpoint_path is not None:
            checkpoint_path.write_text(json.dumps({'scale': float(scale), 'records': records}, ensure_ascii=False), encoding='utf-8')
        if max_new_clips is not None and processed_this_run >= max_new_clips:
            break
    return {'scale': float(scale), 'records': records}

def summarize(scale_result: dict) -> dict:
    records = [r for r in scale_result['records'] if 'predicted' in r]
    y_true = [r['expected'] for r in records]
    y_pred = [r['predicted'] for r in records]
    labels = ['line', 'other', 'scrum']
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0) if records else {}
    return {'scale': scale_result['scale'], 'n': len(records), 'errors': len(scale_result['records']) - len(records), 'accuracy': float(accuracy_score(y_true, y_pred)) if records else 0.0, 'f1_macro': float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)) if records else 0.0, 'balanced_accuracy': float(sum((report.get(label, {}).get('recall', 0.0) for label in labels)) / len(labels)) if records else 0.0, 'by_class_recall': {label: float(report.get(label, {}).get('recall', 0.0)) for label in labels}, 'by_class_precision': {label: float(report.get(label, {}).get('precision', 0.0)) for label in labels}, 'mean_true_class_probability': float(sum((r['true_class_probability'] for r in records)) / len(records)) if records else 0.0, 'mean_confidence': float(sum((r['confidence'] for r in records)) / len(records)) if records else 0.0, 'mean_detections': float(sum((r['detections'] for r in records)) / len(records)) if records else 0.0}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=5)
    parser.add_argument('--scales', default=None, help='Comma-separated subset, e.g. 1.00')
    parser.add_argument('--out-suffix', default='', help='Suffix for output files, e.g. _25cases')
    parser.add_argument('--all-cases', action='store_true', help='Use every row in the test manifest')
    parser.add_argument('--max-new-clips', type=int, default=None, help='Stop after this many new clips; checkpoint allows chunked runs')
    args = parser.parse_args()
    examples = selected_examples(max(1, args.per_class), all_cases=args.all_cases)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = str(args.out_suffix or '')
    log_path = OUT_DIR / f'run{suffix}.log'
    quiet_log = _resolve_model_artifact(log_path).open('w', encoding='utf-8')
    try:
        detector = RGBYoloQuantileFusionDetector()
        results = []
        scales = list(SCALE_ROOTS)
        if args.scales:
            requested = [item.strip() for item in args.scales.split(',') if item.strip()]
            unknown = [item for item in requested if item not in SCALE_ROOTS]
            if unknown:
                raise ValueError(f'Unknown scales: {unknown}')
            scales = requested
        for scale in scales:
            print(f'Evaluating scale {scale} ({len(examples)} clips)...', flush=True)
            checkpoint_path = OUT_DIR / f"checkpoint_{scale.replace('.', 'p')}{suffix}.json"
            result = evaluate_scale(detector, scale, examples, quiet_log, checkpoint_path=checkpoint_path, max_new_clips=args.max_new_clips)
            result['summary'] = summarize(result)
            results.append(result)
            print(json.dumps(result['summary'], ensure_ascii=False), flush=True)
    finally:
        quiet_log.close()
    summary = [item['summary'] for item in results]
    payload = {'model': 'rgb_yolo_quantile_fusion_all', 'yolo_frames_per_window': 100, 'temporal_fps_mode': 'fast_12', 'selection': {'manifest': str(MANIFEST), 'per_class': args.per_class, 'clips': len(examples)}, 'scale_roots': {scale: str(path) for scale, path in SCALE_ROOTS.items()}, 'summaries': summary, 'results': results}
    (OUT_DIR / f'results{suffix}.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    with _resolve_model_artifact(OUT_DIR / f'summary{suffix}.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved {OUT_DIR / f'results{suffix}.json'}", flush=True)
if __name__ == '__main__':
    main()
