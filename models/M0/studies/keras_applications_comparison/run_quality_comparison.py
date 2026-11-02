"""Quick quality comparison for Keras Applications extractors.

This is a deliberately isolated experiment. It reads the existing train/test CSV
and video clips, and uses the shared feature cache and a separate output directory.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames, cached_rgb, auxiliary_directory
from models.common.study_data import study_csv_path
import argparse
import csv
import json
import random
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from models.M0.studies.keras_applications_comparison.compare_keras_applications import build_feature_extractor
ROOT = experiment_directory(__file__).parents[1]
EXPERIMENT_DIR = experiment_directory(__file__)
RESULTS_DIR = auxiliary_directory('keras_quality')
import os
CACHE_DIR = Path(os.environ.get('TESINA_CACHE_DIR', ROOT/'.runtime/cache/features'))/'keras_quality'
DATA_ROOT = ROOT / 'data' / 'videos' / 'clips_224'
TRAIN_CSV = CURRENT_DATASET / 'train.csv'
VALIDATION_CSV = CURRENT_DATASET / 'validation.csv'
TEST_CSV = CURRENT_DATASET / 'test.csv'
PROJECT_FEATURE_CACHE_DIR = ROOT / 'models' / 'M0' / 'feature_cache'
IMG_SIZE = 224
PROJECT_FEATURE_DIM = 2048
CLASSES = ['line', 'other', 'scrum']
LABEL_TO_ID = {label: i for i, label in enumerate(CLASSES)}

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    out = df[['video_name', 'tag']].copy()
    out['video_name'] = out['video_name'].astype(str).str.strip()
    out['tag'] = out['tag'].astype(str).str.strip().str.lower()
    out = out[(out['video_name'] != '') & (out['tag'] != '')]
    return out.reset_index(drop=True)

def load_manifest(csv_path: Path, full: bool, per_class: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(_resolve_model_artifact(study_csv_path(csv_path)))
    df = normalize_manifest(df)
    if full:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rows = []
    for label in CLASSES:
        group = df[df['tag'] == label].sample(n=per_class, random_state=seed, replace=False)
        rows.append(group)
    out = pd.concat(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def crop_center_square(frame: np.ndarray) -> np.ndarray:
    y, x = frame.shape[:2]
    min_dim = min(y, x)
    start_x = x // 2 - min_dim // 2
    start_y = y // 2 - min_dim // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def sample_frame_indices(total_frames: int, source_fps: float, target_fps: float=25.0) -> list[int]:
    """Match the current CNN-RNN temporal sampling convention."""
    if total_frames <= 0:
        return []
    if target_fps <= 0 or source_fps <= 0:
        return list(range(total_frames))
    duration = total_frames / source_fps
    sample_count = max(1, int(np.floor(duration * target_fps)))
    sample_times = np.arange(sample_count, dtype=np.float32) / float(target_fps)
    indices = np.rint(sample_times * float(source_fps)).astype(np.int32)
    indices = np.clip(indices, 0, total_frames - 1)
    return indices.tolist()

def load_video_frames(video_path: Path, max_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total_frames <= 0:
            return np.empty((0, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        sample_indices = sample_frame_indices(total_frames, source_fps, target_fps=25.0)
        sample_counts: dict[int, int] = {}
        for index in sample_indices:
            sample_counts[index] = sample_counts.get(index, 0) + 1
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            repeat_count = sample_counts.get(frame_index, 0)
            if repeat_count > 0:
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                frame = frame[:, :, [2, 1, 0]]
                for _ in range(repeat_count):
                    frames.append(frame.astype(np.float32))
                    if len(frames) >= max_frames:
                        break
            frame_index += 1
            if len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return np.asarray(frames, dtype=np.float32)

def cache_path(model_name: str, split: str, video_name: str, max_frames: int) -> Path:
    stem = Path(video_name).stem
    return CACHE_DIR / model_name / f'frames{max_frames}' / split / f'{stem}.npz'

def load_project_cache_if_available(model_name: str, video_name: str, max_frames: int):
    """Reuse existing full project caches for current extractors when possible."""
    if model_name == 'resnet50':
        features, mask = cached_rgb(video_name)
        return features[:max_frames], mask[:max_frames]
    return None

def extract_split_features(model_name: str, split: str, df: pd.DataFrame, extractor, max_frames: int, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features = []
    masks = []
    labels = []
    names = []
    for row in df.itertuples(index=False):
        video_name = row.video_name
        label = row.tag
        out_path = cache_path(model_name, split, video_name, max_frames)
        project_cached = load_project_cache_if_available(model_name, video_name, max_frames)
        if project_cached is not None:
            clip_features, clip_mask = project_cached
        elif _resolve_model_artifact(study_csv_path(out_path)).exists():
            data = _load_model_result_array(out_path)
            clip_features = data['features']
            clip_mask = data['mask']
        else:
            frames = load_video_frames(DATA_ROOT / video_name, max_frames=max_frames)
            clip_features = np.zeros((max_frames, PROJECT_FEATURE_DIM), dtype=np.float32)
            clip_mask = np.zeros((max_frames,), dtype=bool)
            if len(frames) > 0:
                extracted = extractor.predict(frames, batch_size=batch_size, verbose=0)
                length = min(len(extracted), max_frames)
                clip_features[:length] = extracted[:length]
                clip_mask[:length] = True
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, features=clip_features, mask=clip_mask)
        features.append(clip_features)
        masks.append(clip_mask)
        labels.append(LABEL_TO_ID[label])
        names.append(video_name)
    return (np.asarray(features, dtype=np.float32), np.asarray(masks, dtype=bool), np.asarray(labels, dtype=np.int64), names)

def build_sequence_head(max_frames: int, num_classes: int, learning_rate: float):
    import keras
    features_input = keras.Input((max_frames, PROJECT_FEATURE_DIM), name='features')
    mask_input = keras.Input((max_frames,), dtype='bool', name='mask')
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True))(features_input, mask=mask_input)
    x = keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model([features_input, mask_input], output, name='best_config_bigru_head')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_one_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, max_frames: int, extract_batch_size: int, train_batch_size: int, epochs: int, seed: int, validation_df: pd.DataFrame) -> dict[str, object]:
    import keras
    started = time.perf_counter()
    extractor = None if model_name == 'resnet50' else build_feature_extractor(model_name, weights='imagenet', project_to_2048=True)
    train_x, train_mask, train_y, _ = extract_split_features(model_name, 'train', train_df, extractor, max_frames, extract_batch_size)
    test_x, test_mask, test_y, test_names = extract_split_features(model_name, 'test', test_df, extractor, max_frames, extract_batch_size)
    val_x, val_mask, val_y, _ = extract_split_features(model_name, 'validation', validation_df, extractor, max_frames, extract_batch_size)
    class_weights_values = compute_class_weight(class_weight='balanced', classes=np.arange(len(CLASSES)), y=train_y)
    class_weight = {i: float(w) for i, w in enumerate(class_weights_values)}
    set_seeds(seed)
    head = build_sequence_head(max_frames, len(CLASSES), learning_rate=0.001)
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)]
    callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-06, verbose=0))
    history = head.fit([train_x, train_mask], train_y, validation_data=([val_x, val_mask], val_y), epochs=epochs, batch_size=train_batch_size, verbose=0, class_weight=class_weight, callbacks=callbacks)
    probabilities = head.predict([test_x, test_mask], batch_size=train_batch_size, verbose=0)
    pred_y = np.argmax(probabilities, axis=1)
    report = classification_report(test_y, pred_y, target_names=CLASSES, output_dict=True, zero_division=0)
    result = {'model': model_name, 'status': 'ok', 'train_clips': int(len(train_y)), 'validation_clips': int(len(val_y)), 'test_clips': int(len(test_y)), 'max_frames': int(max_frames), 'training_recipe': 'best_model_resnet50_bigru64_drop03_dense64_lr1e-3', 'epochs_ran': int(len(history.history['loss'])), 'test_accuracy': round(float(accuracy_score(test_y, pred_y)), 4), 'test_f1_macro': round(float(f1_score(test_y, pred_y, average='macro')), 4), 'test_f1_weighted': round(float(f1_score(test_y, pred_y, average='weighted')), 4), 'recall_line': round(float(report['line']['recall']), 4), 'recall_other': round(float(report['other']['recall']), 4), 'recall_scrum': round(float(report['scrum']['recall']), 4), 'duration_s': round(time.perf_counter() - started, 1)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = RESULTS_DIR / f'{model_name}_predictions.csv'
    with _resolve_model_artifact(study_csv_path(detail_path)).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['video_name', 'true_label', 'pred_label', 'p_line', 'p_other', 'p_scrum'])
        writer.writeheader()
        for name, true_id, pred_id, probs in zip(test_names, test_y, pred_y, probabilities):
            writer.writerow({'video_name': name, 'true_label': CLASSES[int(true_id)], 'pred_label': CLASSES[int(pred_id)], 'p_line': float(probs[0]), 'p_other': float(probs[1]), 'p_scrum': float(probs[2])})
    return result

def write_summary(results: list[dict[str, object]], args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'quality_results.json').write_text(json.dumps({'args': vars(args), 'results': results}, indent=2, ensure_ascii=False), encoding='utf-8')
    keys = sorted({key for row in results for key in row})
    with _resolve_model_artifact(study_csv_path(RESULTS_DIR / 'quality_results.csv')).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    ranked = sorted(results, key=lambda r: (r.get('test_f1_macro', 0), r.get('test_accuracy', 0)), reverse=True)
    lines = ['# Quick extractor quality results', '', 'This experiment uses the same sequence-training recipe as the current best model.', '', f"- Train source: `{('full train.csv' if args.full_train else str(args.train_per_class) + ' clips per class')}`", f"- Test source: `{('full test.csv' if args.full_test else str(args.test_per_class) + ' clips per class')}`", f'- Max sampled frames per clip: `{args.max_frames}`', f'- Epoch budget: `{args.epochs}`', '', '| Rank | Model | Accuracy | Macro-F1 | Weighted-F1 | Recall line | Recall other | Recall scrum | Duration s |', '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for rank, row in enumerate(ranked, start=1):
        lines.append(f"| {rank} | `{row['model']}` | {row['test_accuracy']:.4f} | {row['test_f1_macro']:.4f} | {row['test_f1_weighted']:.4f} | {row['recall_line']:.4f} | {row['recall_other']:.4f} | {row['recall_scrum']:.4f} | {row['duration_s']:.1f} |")
    (RESULTS_DIR / 'QUALITY_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', default=['resnet50', 'resnet50v2', 'efficientnetv2b0', 'mobilenetv3large', 'nasnetmobile'])
    parser.add_argument('--train-per-class', type=int, default=12)
    parser.add_argument('--validation-per-class', type=int, default=6)
    parser.add_argument('--full-validation', action='store_true')
    parser.add_argument('--test-per-class', type=int, default=6)
    parser.add_argument('--full-train', action='store_true', help='Use the full train.csv.')
    parser.add_argument('--full-test', action='store_true', help='Use the full test.csv.')
    parser.add_argument('--max-frames', type=int, default=200)
    parser.add_argument('--extract-batch-size', type=int, default=16)
    parser.add_argument('--train-batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    train_df = load_manifest(TRAIN_CSV, args.full_train, args.train_per_class, args.seed)
    validation_df = load_manifest(VALIDATION_CSV, args.full_validation, args.validation_per_class, args.seed)
    test_df = load_manifest(TEST_CSV, args.full_test, args.test_per_class, args.seed + 1)
    results = []
    for model_name in args.models:
        try:
            result = run_one_model(model_name=model_name, train_df=train_df, test_df=test_df, max_frames=args.max_frames, extract_batch_size=args.extract_batch_size, train_batch_size=args.train_batch_size, epochs=args.epochs, seed=args.seed, validation_df=validation_df)
        except Exception as exc:
            result = {'model': model_name, 'status': 'error', 'error': repr(exc)}
        print(json.dumps(result, ensure_ascii=False))
        results.append(result)
        write_summary(results, args)
if __name__ == '__main__':
    main()
