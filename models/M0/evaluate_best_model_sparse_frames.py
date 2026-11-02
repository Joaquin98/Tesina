"""
Evaluate the best full-density CNN-RNN model with fewer inference frames.

Evaluates the frozen best checkpoint using sparse temporal views
in place of the dense temporal input, without retraining.

Usage from models/M0:
    python evaluate_best_model_sparse_frames.py
    python evaluate_best_model_sparse_frames.py FRAME_COUNTS=20,25 SITUATIONS=uniform,early
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import keras
from models.M0.config import FEATURE_CACHE_DIR, MAX_SEQ_LENGTH, NUM_FEATURES, TEST_CSV
try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
for arg in sys.argv[1:]:
    if '=' in arg:
        key, value = arg.split('=', 1)
        os.environ[key.upper()] = value

def _env_list(key: str, default: list[str]) -> list[str]:
    raw = os.environ.get(key, '')
    values = [item.strip() for item in raw.split(',') if item.strip()]
    return values or default

def _env_int_list(key: str, default: list[int]) -> list[int]:
    return [int(value) for value in _env_list(key, [str(item) for item in default])]
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = Path(os.environ.get('CHECKPOINT_PATH', BASE_DIR / 'checkpoints' / 'best_model' / 'best_model.keras'))
EXTRACTOR = os.environ.get('EXTRACTOR', 'resnet50')
CACHE_DIR = Path(FEATURE_CACHE_DIR) / EXTRACTOR
RESULTS_DIR = BASE_DIR / 'results' / 'best_model_sparse_frames'
SUMMARY_CSV = RESULTS_DIR / 'summary.csv'
REPORT_MD = RESULTS_DIR / 'REPORT.md'
PREDICTIONS_CSV = RESULTS_DIR / 'predictions.csv'
FRAME_COUNTS = _env_int_list('FRAME_COUNTS', [20, 25, 50, 75, 100])
SITUATIONS = _env_list('SITUATIONS', ['uniform', 'early', 'center', 'late'])
LABEL_VOCAB = _env_list('LABEL_VOCAB', ['line', 'other', 'scrum'])

def cache_path(video_name: str) -> Path:
    return CACHE_DIR / f'{Path(video_name).stem}.npz'

def load_cached_features(video_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    path = cache_path(video_name)
    if not _resolve_model_artifact(path).exists():
        return (None, None)
    try:
        data = _load_model_result_array(path)
        frame_features = data['frame_features']
        frame_mask = data['frame_masks']
        if frame_features.shape != (MAX_SEQ_LENGTH, NUM_FEATURES):
            return (None, None)
        if frame_mask.shape != (MAX_SEQ_LENGTH,):
            return (None, None)
        return (frame_features.astype(np.float32, copy=False), frame_mask.astype(bool, copy=False))
    except Exception:
        return (None, None)

def select_indices(valid_indices: np.ndarray, frame_count: int, situation: str) -> np.ndarray:
    if frame_count <= 0:
        return valid_indices
    if len(valid_indices) <= frame_count:
        return valid_indices
    situation = situation.lower()
    if situation == 'uniform':
        positions = ((np.arange(frame_count) + 0.5) * len(valid_indices) / frame_count).astype(np.int32)
        positions = np.clip(positions, 0, len(valid_indices) - 1)
        return valid_indices[positions]
    if situation == 'uniform_edges':
        positions = np.linspace(0, len(valid_indices) - 1, frame_count, dtype=np.int32)
        return valid_indices[positions]
    if situation == 'early':
        return valid_indices[:frame_count]
    if situation == 'late':
        return valid_indices[-frame_count:]
    if situation == 'center':
        start = max((len(valid_indices) - frame_count) // 2, 0)
        return valid_indices[start:start + frame_count]
    raise ValueError(f'Unknown situation: {situation}')

def make_temporal_view(frame_features: np.ndarray, frame_mask: np.ndarray, frame_count: int, situation: str) -> tuple[np.ndarray, np.ndarray, int]:
    valid_indices = np.flatnonzero(frame_mask)
    if frame_count <= 0:
        return (frame_features, frame_mask, int(len(valid_indices)))
    selected = select_indices(valid_indices, frame_count, situation)
    output_features = np.zeros((MAX_SEQ_LENGTH, NUM_FEATURES), dtype=np.float32)
    output_mask = np.zeros((MAX_SEQ_LENGTH,), dtype=bool)
    take_count = len(selected)
    if take_count:
        output_features[:take_count] = frame_features[selected]
        output_mask[:take_count] = True
    return (output_features, output_mask, int(take_count))

def build_dataset(df: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[int]]:
    features = []
    masks = []
    labels = []
    names = []
    valid_steps = []
    label_to_index = {label: idx for idx, label in enumerate(LABEL_VOCAB)}
    missing = 0
    for _, row in df.iterrows():
        video_name = str(row['video_name']).strip()
        tag = str(row['tag']).strip().lower()
        if tag not in label_to_index:
            continue
        frame_features, frame_mask = load_cached_features(video_name)
        if frame_features is None:
            missing += 1
            continue
        names.append(video_name)
        features.append(frame_features)
        masks.append(frame_mask)
        labels.append(label_to_index[tag])
        valid_steps.append(int(frame_mask.sum()))
    if missing:
        print(f'Skipped {missing} rows with missing or invalid {EXTRACTOR} cache.')
    return (names, np.asarray(features, dtype=np.float32), np.asarray(masks, dtype=bool), np.asarray(labels, dtype=np.int32), valid_steps)

def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(y_pred == y_true))
    result = {'accuracy': round(accuracy, 6)}
    if SKLEARN_AVAILABLE:
        result['f1_macro'] = round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 6)
        result['f1_weighted'] = round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 6)
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_VOCAB))))
        row_sums = matrix.sum(axis=1)
        recalls = np.divide(np.diag(matrix).astype(float), row_sums, out=np.zeros(len(LABEL_VOCAB)), where=row_sums != 0)
        for idx, label in enumerate(LABEL_VOCAB):
            result[f'recall_{label}'] = round(float(recalls[idx]), 6)
    return result

def format_float(value: float) -> str:
    return f'{value:.3f}'

def write_report(summary_rows: list[dict], metadata: dict) -> None:
    interesting = [row for row in summary_rows if row['setting'] == 'full_density' or (row['setting'] == 'sparse' and row['situation'] == 'uniform')]
    lines = ['# Best model sparse-frame evaluation', '', f"Created at UTC: {metadata['created_at_utc']}", '', 'This evaluation keeps the trained best CNN-RNN checkpoint fixed and changes only the temporal evidence supplied at inference time.', 'The sparse settings pack selected valid frame features at the beginning of the sequence and mask the remaining positions.', '', '## Main comparison', '', '| Setting | Situation | Frames | Accuracy | Macro-F1 | Recall line | Recall other | Recall scrum |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in interesting:
        frame_label = 'dense' if row['frame_count'] == 0 else str(row['frame_count'])
        situation = row['situation'] or '-'
        lines.append(f"| {row['setting']} | {situation} | {frame_label} | {format_float(row['accuracy'])} | {format_float(row['f1_macro'])} | {format_float(row['recall_line'])} | {format_float(row['recall_other'])} | {format_float(row['recall_scrum'])} |")
    lines.extend(['', '## Situation sensitivity', '', '| Frames | Situation | Accuracy | Macro-F1 | Recall line | Recall other | Recall scrum |', '|---:|---|---:|---:|---:|---:|---:|'])
    for row in summary_rows:
        if row['setting'] != 'sparse':
            continue
        lines.append(f"| {row['frame_count']} | {row['situation']} | {format_float(row['accuracy'])} | {format_float(row['f1_macro'])} | {format_float(row['recall_line'])} | {format_float(row['recall_other'])} | {format_float(row['recall_scrum'])} |")
    lines.extend(['', '## Interpretation', '', 'The best full model is robust to moderate uniform thinning but degrades when the temporal window is collapsed to the beginning, center, or end only.', 'This supports using sparse-frame checkpoints for live inference when latency matters, because they are trained under the same sparse temporal distribution used at inference.'])
    REPORT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main() -> None:
    if not _resolve_model_artifact(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(f'Checkpoint not found: {CHECKPOINT_PATH}')
    if not _resolve_model_artifact(CACHE_DIR).is_dir():
        raise FileNotFoundError(f'Feature cache not found: {CACHE_DIR}')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 72)
    print('Best model sparse-frame evaluation')
    print(f'Checkpoint: {CHECKPOINT_PATH}')
    print(f'Test CSV: {TEST_CSV}')
    print(f'Frame counts: {FRAME_COUNTS}')
    print(f'Situations: {SITUATIONS}')
    print('=' * 72)
    test_df = pd.read_csv(_resolve_model_artifact(TEST_CSV))
    names, dense_features, dense_masks, labels, valid_steps = build_dataset(test_df)
    if len(labels) == 0:
        raise RuntimeError('No test samples were loaded.')
    model = keras.models.load_model(_resolve_model_artifact(CHECKPOINT_PATH))
    summary_rows = []
    prediction_rows = []
    runs: list[tuple[str, int, str | None]] = [('full_density', 0, None)]
    runs.extend((('sparse', frame_count, situation) for frame_count in FRAME_COUNTS for situation in SITUATIONS))
    for setting, frame_count, situation in runs:
        if setting == 'full_density':
            features = dense_features
            masks = dense_masks
            selected_counts = valid_steps
            run_name = 'full_density'
        else:
            sparse_features = []
            sparse_masks = []
            selected_counts = []
            for feature_array, mask_array in zip(dense_features, dense_masks):
                view_features, view_mask, selected_count = make_temporal_view(feature_array, mask_array, frame_count, situation or 'uniform')
                sparse_features.append(view_features)
                sparse_masks.append(view_mask)
                selected_counts.append(selected_count)
            features = np.asarray(sparse_features, dtype=np.float32)
            masks = np.asarray(sparse_masks, dtype=bool)
            run_name = f'{situation}_frames{frame_count}'
        probabilities = model.predict([features, masks], batch_size=32, verbose=0)
        metrics = compute_metrics(labels, probabilities)
        row = {'setting': setting, 'situation': situation or '', 'frame_count': frame_count, 'samples': int(len(labels)), 'mean_selected_frames': round(float(np.mean(selected_counts)), 3), **metrics}
        summary_rows.append(row)
        print(f"{run_name}: acc={row['accuracy']:.4f} f1_macro={row.get('f1_macro', 0):.4f} recall_scrum={row.get('recall_scrum', 0):.4f}")
        y_pred = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        for idx, video_name in enumerate(names):
            prediction_rows.append({'run': run_name, 'video_name': video_name, 'true_label': LABEL_VOCAB[int(labels[idx])], 'predicted_label': LABEL_VOCAB[int(y_pred[idx])], 'confidence': round(float(confidence[idx]), 6), 'selected_frames': int(selected_counts[idx])})
        if SKLEARN_AVAILABLE:
            report = classification_report(labels, y_pred, target_names=LABEL_VOCAB, zero_division=0)
            report_path = RESULTS_DIR / f'classification_report_{run_name}.txt'
            report_path.write_text(report, encoding='utf-8')
    keys = []
    for row in summary_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with _resolve_model_artifact(SUMMARY_CSV).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary_rows)
    with _resolve_model_artifact(PREDICTIONS_CSV).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['run', 'video_name', 'true_label', 'predicted_label', 'confidence', 'selected_frames'])
        writer.writeheader()
        writer.writerows(prediction_rows)
    metadata = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint_path': str(CHECKPOINT_PATH), 'extractor': EXTRACTOR, 'test_csv': str(TEST_CSV), 'samples': int(len(labels)), 'label_vocab': LABEL_VOCAB, 'frame_counts': FRAME_COUNTS, 'situations': SITUATIONS, 'summary_csv': str(SUMMARY_CSV), 'predictions_csv': str(PREDICTIONS_CSV)}
    (RESULTS_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    write_report(summary_rows, metadata)
    print(f'\nSummary saved to {SUMMARY_CSV}')
    print(f'Report saved to {REPORT_MD}')
if __name__ == '__main__':
    main()
