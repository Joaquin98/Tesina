"""Export saved final-comparison predictions as a web-compatible result."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import pandas as pd
ROOT = experiment_directory(__file__).parents[1]
DEFAULT_PREDICTIONS = ROOT / 'data/final_model_comparison_20260827/outputs/runs' / 'PV-M1-H2-D2-PV01-S42/attempt_001/predictions.csv'
DEFAULT_RUN = DEFAULT_PREDICTIONS.with_name('run.json')
DEFAULT_VIDEO = ROOT / 'data/videos/partidos_evaluacion' / 'V_1_S01_F17_15_07_2023_PRIMERA_LOGARITMO_27_VS_24_LA_SALLE.mp4'
DEFAULT_RESULTS = ROOT / 'data/results'
LABELS = ('line', 'other', 'scrum')
TARGET_LABELS = ('line', 'scrum')

def _probabilities(row: pd.Series) -> dict[str, float]:
    return {label: float(row[f'p_{label}']) for label in LABELS}

def _window(row: pd.Series, video_duration: float) -> dict:
    start_time = float(row['start_s'])
    end_time = min(float(row['end_s']), video_duration)
    label = str(row['y_final'])
    probabilities = _probabilities(row)
    return {'play_type': label, 'start_time': start_time, 'end_time': end_time, 'duration': max(0.0, end_time - start_time), 'confidence': probabilities[label], 'frame_start': int(round(start_time * 25.0)), 'frame_end': int(round(end_time * 25.0)), 'probabilities': probabilities, 'ground_truth': str(row['y_true']), 'block_id': str(row['block_id']), 'block_index': int(row['block_index']), 'predicted_event_id': None if pd.isna(row['predicted_event_id']) else str(row['predicted_event_id'])}

def _event(event_id: str, rows: pd.DataFrame, video_duration: float) -> dict:
    label_values = rows['y_final'].astype(str).unique().tolist()
    if len(label_values) != 1 or label_values[0] not in TARGET_LABELS:
        raise ValueError(f'Invalid target event {event_id}: labels={label_values}')
    label = label_values[0]
    start_time = float(rows['start_s'].min())
    end_time = min(float(rows['end_s'].max()), video_duration)
    probabilities = {candidate: float(rows[f'p_{candidate}'].mean()) for candidate in LABELS}
    confidence_values = rows[f'p_{label}'].astype(float)
    peak_index = confidence_values.idxmax()
    peak = rows.loc[peak_index]
    return {'play_type': label, 'start_time': start_time, 'end_time': end_time, 'duration': max(0.0, end_time - start_time), 'confidence': float(confidence_values.max()), 'mean_confidence': float(confidence_values.mean()), 'frame_start': int(round(start_time * 25.0)), 'frame_end': int(round(end_time * 25.0)), 'probabilities': probabilities, 'source_window_count': int(len(rows)), 'predicted_event_id': event_id, 'peak_window': {'start_time': float(peak['start_s']), 'end_time': min(float(peak['end_s']), video_duration), 'confidence': float(peak[f'p_{label}'])}, 'contributing_windows': [{'start_time': float(row['start_s']), 'end_time': min(float(row['end_s']), video_duration), 'confidence': float(row[f'p_{label}'])} for _, row in rows.iterrows()]}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--video', type=Path, default=DEFAULT_VIDEO)
    parser.add_argument('--section', default='S01')
    parser.add_argument('--video-duration', type=float, default=1201.2)
    parser.add_argument('--web-results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--result-id')
    args = parser.parse_args()
    if not _resolve_model_artifact(args.video).exists():
        raise FileNotFoundError(args.video)
    frame = pd.read_csv(_resolve_model_artifact(args.predictions))
    frame = frame[frame['section'].astype(str).eq(args.section)].copy()
    frame = frame.sort_values(['start_s', 'block_index']).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f'No predictions found for section {args.section}')
    if frame['block_id'].duplicated().any():
        raise ValueError('Duplicate block IDs in selected predictions')
    run = json.loads(_resolve_model_artifact(args.run).read_text(encoding='utf-8'))
    windows = [_window(row, args.video_duration) for _, row in frame.iterrows()]
    event_rows = frame.dropna(subset=['predicted_event_id'])
    events = [_event(str(event_id), rows, args.video_duration) for event_id, rows in event_rows.groupby('predicted_event_id', sort=False)]
    events.sort(key=lambda item: (item['start_time'], item['end_time']))
    block_counts = Counter((window['play_type'] for window in windows))
    event_counts = Counter((event['play_type'] for event in events))
    average_confidence = sum((float(event['confidence']) for event in events)) / len(events) if events else 0.0
    by_type = {}
    for label in TARGET_LABELS:
        selected = [event for event in events if event['play_type'] == label]
        if selected:
            by_type[label] = {'count': len(selected), 'total_duration': sum((float(event['duration']) for event in selected))}
    payload = {'status': 'success', 'video_path': str(args.video.resolve()), 'model_name': 'M1-H2-D2 - V1 S01 (resultado experimental)', 'detections': events, 'primary_detections': events, 'review_detections_7s': windows, 'window_predictions': windows, 'event_post_processing': {'mode': 'saved_final_comparison_event_ids', 'target_labels': list(TARGET_LABELS), 'block_duration_seconds': 7.0, 'line_threshold': float(run['thresholds']['line']), 'scrum_threshold': float(run['thresholds']['scrum']), 'note': 'Predicciones guardadas del fold PV01; no se volvió a ejecutar inferencia.'}, 'meta_correction': None, 'fusion_metadata': {'model_family': 'M1', 'tuning_level': 'H2', 'data_regime': 'D2', 'fold': 'PV01', 'test_group': 'V_1', 'section': args.section, 'architecture': 'RGB + pose + geometría'}, 'statistics': {'avg_confidence': average_confidence, 'avg_duration': sum((float(event['duration']) for event in events)) / len(events) if events else 0.0, 'by_type': by_type, 'block_counts': dict(block_counts), 'total_windows': len(windows)}, 'summary': {'total_detections': len(events), 'plays_by_type': dict(event_counts), 'average_confidence': average_confidence, 'predicted_class': 'sequence', 'overall_confidence': average_confidence}, 'metadata': {'duration': float(args.video_duration), 'confidence_threshold': None, 'line_confidence_threshold': float(run['thresholds']['line']), 'scrum_confidence_threshold': float(run['thresholds']['scrum']), 'analysis_mode': 'saved_experiment_predictions', 'temporal_fps_mode': 'source_25_fps_seven_second_blocks', 'source_run_id': str(frame['run_id'].iloc[0]), 'source_section': args.section, 'checkpoint_hash': str(frame['checkpoint_hash'].iloc[0]), 'cache_signature': str(frame['cache_signature'].iloc[0]), 'experimental_result': True}}
    result_id = args.result_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    args.web_results.mkdir(parents=True, exist_ok=True)
    output = args.web_results / f'result_{result_id}.json'
    if _resolve_model_artifact(output).exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'result_id': result_id, 'path': str(output.resolve()), 'windows': len(windows), 'events': len(events), 'block_counts': dict(block_counts), 'event_counts': dict(event_counts)}, ensure_ascii=False))
if __name__ == '__main__':
    main()
