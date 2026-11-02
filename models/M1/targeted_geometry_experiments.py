"""Experimentos de representacion YOLO orientados a scrum y lineout.

La fase usa solamente las caches existentes. Cada variante resume cajas y
articulaciones por clip y se evalua con un Random Forest independiente para
aislar el efecto de los datos geometricos antes de integrarlos en la RNN.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from models.M1.config import LABELS, LABEL_TO_ID, RESULTS_DIR
from models.M1.protocol import build_partition, load_geometry_features, pose_cache_path
VARIANTS = ('y0_current', 'y1_cleaned', 'y2_formation', 'y3_posture', 'y4_targeted', 'y5_targeted_temporal', 'y6_all_plus_targeted')
POSITION_CLEAN_INDICES = [0, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
POSTURE_CLEAN_INDICES = list(range(3, 15)) + list(range(17, 29))

def safe_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else 0.0

def summarize(sequence: np.ndarray, mask: np.ndarray, temporal_extra: bool=False) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    valid = sequence[mask] if np.any(mask) else sequence
    valid = np.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
    deltas = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, sequence.shape[-1]), dtype=np.float32)
    summary = [valid.mean(axis=0), valid.std(axis=0), valid.min(axis=0), valid.max(axis=0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), deltas.mean(axis=0), deltas.std(axis=0), np.abs(deltas).mean(axis=0)]
    if temporal_extra:
        if len(valid) > 1:
            time_axis = np.linspace(-1.0, 1.0, len(valid), dtype=np.float32)
            centered_time = time_axis - time_axis.mean()
            denominator = float(np.sum(centered_time * centered_time)) + 1e-06
            slope = np.sum(centered_time[:, None] * (valid - valid.mean(axis=0)[None, :]), axis=0) / denominator
            first_last = valid[-1] - valid[0]
        else:
            slope = np.zeros(sequence.shape[-1], dtype=np.float32)
            first_last = np.zeros(sequence.shape[-1], dtype=np.float32)
        summary.extend([slope, first_last])
    return np.concatenate(summary).astype(np.float32)

def formation_frame(boxes: np.ndarray, person_conf: np.ndarray, valid: np.ndarray) -> np.ndarray:
    keep = np.asarray(valid, dtype=bool) & (np.asarray(person_conf) >= 0.25)
    selected = np.asarray(boxes, dtype=np.float32)[keep]
    if len(selected) == 0:
        return np.zeros(19, dtype=np.float32)
    centers = np.column_stack(((selected[:, 0] + selected[:, 2]) * 0.5, (selected[:, 1] + selected[:, 3]) * 0.5))
    count = len(centers) / 30.0
    std = centers.std(axis=0)
    extent = np.ptp(centers, axis=0) if len(centers) > 1 else np.zeros(2, dtype=np.float32)
    if len(centers) > 1:
        centered = centers - centers.mean(axis=0, keepdims=True)
        covariance = np.atleast_2d(np.cov(centered, rowvar=False))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        principal = eigenvectors[:, order[0]]
        anisotropy = float(1.0 - eigenvalues[1] / (eigenvalues[0] + 1e-06))
        angle = float(np.arctan2(principal[1], principal[0]))
    else:
        anisotropy, angle = (0.0, 0.0)
    if len(centers) > 1:
        differences = centers[:, None, :] - centers[None, :, :]
        distance_matrix = np.sqrt(np.sum(differences * differences, axis=-1))
        pairwise = distance_matrix[np.triu_indices(len(centers), k=1)]
        distance_matrix[distance_matrix == 0.0] = np.nan
        nearest = np.nanmin(distance_matrix, axis=1)
        nearest = nearest[np.isfinite(nearest)]
    else:
        pairwise = np.empty(0, dtype=np.float32)
        nearest = np.empty(0, dtype=np.float32)
    area = float(extent[0] * extent[1])
    pair_mean = float(pairwise.mean()) if len(pairwise) else 0.0
    pair_std = float(pairwise.std()) if len(pairwise) else 0.0
    nearest_mean = float(nearest.mean()) if len(nearest) else 0.0
    features = np.array([count, float(std[0]), float(std[1]), float(extent[0]), float(extent[1]), anisotropy, float(np.sin(2.0 * angle)), float(np.cos(2.0 * angle)), pair_mean, pair_std, safe_quantile(pairwise, 0.25), safe_quantile(pairwise, 0.5), safe_quantile(pairwise, 0.75), nearest_mean, float(nearest.std()) if len(nearest) else 0.0, safe_quantile(nearest, 0.5), area, float(len(centers) / (area + 0.0001)), float(pair_std / (pair_mean + 0.0001))], dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

def load_clip_variants(video_name: str, pose_model: str) -> dict[str, np.ndarray]:
    path = pose_cache_path(video_name, pose_model)
    with _load_model_result_array(path, allow_pickle=False) as data:
        boxes = np.asarray(data['boxes'], dtype=np.float32)
        person_conf = np.asarray(data['person_conf'], dtype=np.float32)
        keypoints = np.asarray(data['keypoints'], dtype=np.float32)
        keypoint_conf = np.asarray(data['keypoint_conf'], dtype=np.float32)
        valid = np.asarray(data['valid'], dtype=bool)
        positions = np.asarray(data['position_features'], dtype=np.float32)
        poses = np.asarray(data['pose_features'], dtype=np.float32)
        frame_mask = np.asarray(data['frame_mask'], dtype=bool)
    formation = np.stack([formation_frame(box, conf, valid_frame) for box, conf, valid_frame in zip(boxes, person_conf, valid)], axis=0)
    posture = poses[:, POSTURE_CLEAN_INDICES]
    clean = np.concatenate([positions[:, POSITION_CLEAN_INDICES], posture], axis=1)
    targeted = np.concatenate([formation, posture], axis=1)
    current = np.concatenate([positions, poses], axis=1)
    return {'y0_current': summarize(current, frame_mask), 'y1_cleaned': summarize(clean, frame_mask), 'y2_formation': summarize(formation, frame_mask), 'y3_posture': summarize(posture, frame_mask), 'y4_targeted': summarize(targeted, frame_mask), 'y5_targeted_temporal': summarize(targeted, frame_mask, temporal_extra=True), 'y6_all_plus_targeted': np.concatenate([summarize(current, frame_mask), summarize(targeted, frame_mask)])}

def feature_matrix(frame: pd.DataFrame, pose_model: str) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    values = {variant: [] for variant in VARIANTS}
    labels: list[int] = []
    names: list[str] = []
    for row in frame.to_dict('records'):
        name = str(row['video_name'])
        clip = load_clip_variants(name, pose_model)
        for variant in VARIANTS:
            values[variant].append(clip[variant])
        labels.append(LABEL_TO_ID[str(row['tag'])])
        names.append(name)
    return ({variant: np.asarray(rows, dtype=np.float32) for variant, rows in values.items()}, np.asarray(labels, dtype=np.int32), names)

def save_predictions(path: Path, names: list[str], labels: np.ndarray, probabilities: np.ndarray) -> None:
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame({'video_name': names, 'true_label': [LABELS[index] for index in labels], 'predicted_label': [LABELS[index] for index in predictions]})
    for index, label in enumerate(LABELS):
        frame[f'prob_{label}'] = probabilities[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def metrics(labels: np.ndarray, probabilities: np.ndarray, split: str, variant: str, seed: int, dimensions: int, train_seconds: float) -> dict:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=list(range(len(LABELS))), zero_division=0)
    return {'variant': variant, 'seed': int(seed), 'split': split, 'features': int(dimensions), 'accuracy': float(accuracy_score(labels, predictions)), 'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(labels, predictions, average='weighted', zero_division=0)), 'precision_line': float(precision[0]), 'recall_line': float(recall[0]), 'f1_line': float(f1[0]), 'precision_other': float(precision[1]), 'recall_other': float(recall[1]), 'f1_other': float(f1[1]), 'precision_scrum': float(precision[2]), 'recall_scrum': float(recall[2]), 'f1_scrum': float(f1[2]), 'train_seconds': float(train_seconds), 'support_line': int(support[0]), 'support_other': int(support[1]), 'support_scrum': int(support[2])}

def run_seed(seed: int, pose_model: str, trees: int, output_root: Path) -> list[dict]:
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=pose_model)
    train_features, train_labels, train_names = feature_matrix(train_df, pose_model)
    validation_features, validation_labels, validation_names = feature_matrix(validation_df, pose_model)
    test_features, test_labels, test_names = feature_matrix(test_df, pose_model)
    if train_names == validation_names or validation_names == test_names or train_names == test_names:
        raise AssertionError('Se detecto solapamiento de videos entre particiones')
    rows: list[dict] = []
    for variant in VARIANTS:
        start = time.perf_counter()
        classifier = RandomForestClassifier(n_estimators=trees, max_features='sqrt', class_weight='balanced', random_state=seed, n_jobs=-1)
        classifier.fit(train_features[variant], train_labels)
        elapsed = time.perf_counter() - start
        validation_probabilities = classifier.predict_proba(validation_features[variant])
        test_probabilities = classifier.predict_proba(test_features[variant])
        variant_root = output_root / variant / f'seed_{seed}'
        save_predictions(variant_root / 'validation_predictions.csv', validation_names, validation_labels, validation_probabilities)
        save_predictions(variant_root / 'predictions.csv', test_names, test_labels, test_probabilities)
        validation_result = metrics(validation_labels, validation_probabilities, 'validation', variant, seed, train_features[variant].shape[1], elapsed)
        test_result = metrics(test_labels, test_probabilities, 'test', variant, seed, train_features[variant].shape[1], elapsed)
        validation_result['protocol'] = protocol
        test_result['protocol'] = protocol
        result = {'variant': variant, 'seed': int(seed), 'features': int(train_features[variant].shape[1]), 'train_seconds': float(elapsed), 'protocol': protocol, 'validation': validation_result, 'test': test_result}
        (variant_root / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        rows.extend([validation_result, test_result])
        print(json.dumps({key: test_result[key] for key in ('variant', 'seed', 'features', 'accuracy', 'f1_macro', 'recall_line', 'recall_scrum')}, ensure_ascii=False))
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--output-root', default=str(RESULTS_DIR / 'targeted_geometry_rf'))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    results = []
    for seed in seeds:
        print(f'\n=== targeted geometry seed={seed} ===')
        results.extend(run_seed(seed, args.pose_model, args.trees, output_root))
    summary = pd.DataFrame(results)
    summary.to_csv(output_root / 'summary.csv', index=False)
    test_summary = summary[summary['split'] == 'test']
    validation_summary = summary[summary['split'] == 'validation']
    aggregate = test_summary.groupby('variant', as_index=False).agg(runs=('seed', 'count'), features=('features', 'first'), test_accuracy_mean=('accuracy', 'mean'), test_accuracy_std=('accuracy', lambda values: values.std(ddof=0)), test_f1_macro_mean=('f1_macro', 'mean'), test_f1_macro_std=('f1_macro', lambda values: values.std(ddof=0)), test_f1_weighted_mean=('f1_weighted', 'mean'), test_f1_weighted_std=('f1_weighted', lambda values: values.std(ddof=0)), recall_line_mean=('recall_line', 'mean'), recall_other_mean=('recall_other', 'mean'), recall_scrum_mean=('recall_scrum', 'mean'))
    validation_rank = validation_summary.groupby('variant', as_index=False)['f1_macro'].mean().rename(columns={'f1_macro': 'validation_f1_macro_mean'})
    aggregate = aggregate.merge(validation_rank, on='variant', how='left').sort_values('validation_f1_macro_mean', ascending=False)
    aggregate.to_csv(output_root / 'aggregate_metrics.csv', index=False)
    (output_root / 'protocol.json').write_text(json.dumps({'pose_model': args.pose_model, 'trees': args.trees, 'seeds': seeds, 'variants': VARIANTS}, indent=2), encoding='utf-8')
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
