"""Evaluate the high-resolution YOLO ROI in the frozen production protocol.

Only the pose pixels change.  RGB dynamics, labels, five external folds,
historical probabilities, temporal budget, E04/E06 descriptors,
QuantileTransformer, logistic head and 40/60 blend are held fixed.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import QuantileTransformer
try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None
REPO = Path(__file__).resolve().parents[2]
HERE = experiment_directory(__file__)
JOINT = REPO / 'data' / 'joint_multimodal_20260813'
DEFAULT_CACHE = HERE / 'outputs' / 'cache' / 'yolo11n-pose_pt'
DEFAULT_OUTPUT = HERE / 'outputs' / 'quantile_100frames'
LOWRES_MATRIX = JOINT / 'yolo_frame_budget_20260813' / 'e04_e06_100frames.npy'
RGB_FEATURES = JOINT / 'rgb_dynamics_seed42' / 'fusion_features.npz'
LABELS = ('line', 'other', 'scrum')
MODERATE = {0: 1.2, 1: 0.65, 2: 1.8}
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.multimodal import audit_performance as audit
from models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite import make_variant

def metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = probabilities.argmax(axis=1)
    result = audit.metric_dict(y, prediction)
    return result

def load_pose(path: Path) -> dict[str, np.ndarray]:
    with _load_model_result_array(path) as data:
        return {key: data[key] for key in data.files}

def build_highres_matrix(names: np.ndarray, cache_dir: Path, output: Path) -> tuple[np.ndarray, dict]:
    matrix_path = output / 'e04_e06_highres_100frames.npy'
    if _resolve_model_artifact(study_csv_path(matrix_path)).exists():
        matrix = _load_model_result_array(matrix_path, mmap_mode='r')
        if matrix.shape != (len(names), 982):
            raise ValueError(f'Unexpected cached high-resolution matrix shape: {matrix.shape}')
        return (np.asarray(matrix, dtype=np.float32), {'status': 'cached'})
    missing = [name for name in names if not _resolve_model_artifact(study_csv_path(cache_dir / f'{Path(name).stem}.npz')).exists()]
    if missing:
        raise FileNotFoundError(f'High-resolution pose cache is incomplete: {len(missing)} missing; first={missing[0]}')
    started = time.perf_counter()

    def one(name: str) -> tuple[np.ndarray, dict]:
        data = load_pose(cache_dir / f'{Path(name).stem}.npz')
        if data['boxes'].shape[0] != 100:
            raise ValueError(f"{name}: expected 100 pose frames, got {data['boxes'].shape[0]}")
        e04 = make_variant(data, 'E04_tracking_smoothing')
        e06 = make_variant(data, 'E06_rugby_descriptors')
        valid = np.asarray(data['valid'], dtype=bool)
        confidence = np.asarray(data['person_conf'], dtype=np.float32)
        quality = {'frames_with_people': int(np.any(valid, axis=1).sum()), 'mean_people': float(valid.sum(axis=1).mean()), 'mean_confidence': float(confidence[valid].mean()) if np.any(valid) else 0.0}
        return (np.concatenate([e04, e06]).astype(np.float32), quality)
    rows: list[np.ndarray] = []
    qualities: list[dict] = []
    workers = min(8, max(2, (os.cpu_count() or 4) // 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (row, quality) in enumerate(pool.map(one, names.tolist()), start=1):
            rows.append(row)
            qualities.append(quality)
            if index % 500 == 0:
                print(json.dumps({'descriptor_clips': index}), flush=True)
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape != (len(names), 982):
        raise ValueError(f'Unexpected high-resolution descriptor matrix shape: {matrix.shape}')
    np.save(matrix_path, matrix)
    quality = {'status': 'built', 'seconds': float(time.perf_counter() - started), 'mean_frames_with_people': float(np.mean([item['frames_with_people'] for item in qualities])), 'mean_people_per_frame': float(np.mean([item['mean_people'] for item in qualities])), 'mean_person_confidence': float(np.mean([item['mean_confidence'] for item in qualities]))}
    return (matrix, quality)

def external_predictions(x: np.ndarray, y: np.ndarray, folds: list[dict]) -> np.ndarray:
    probabilities = np.zeros((len(y), 3), dtype=np.float32)
    for fold in folds:
        hold = np.asarray(fold['hold'], dtype=np.int32)
        fit = np.ones(len(y), dtype=bool)
        fit[hold] = False
        probabilities[hold] = audit.fit_quantile(x[fit], y[fit], x[hold], C=0.003, class_weight=MODERATE, n_quantiles=256)
    return probabilities

def fold_rows(y: np.ndarray, historical: np.ndarray, lowres_aux: np.ndarray, highres_aux: np.ndarray, folds: list[dict]) -> list[dict]:
    rows = []
    lowres = 0.4 * historical + 0.6 * lowres_aux
    highres = 0.4 * historical + 0.6 * highres_aux
    for fold in folds:
        hold = np.asarray(fold['hold'], dtype=np.int32)
        low = metrics(y[hold], lowres[hold])
        high = metrics(y[hold], highres[hold])
        rows.append({'fold': int(fold['fold']), 'n': int(len(hold)), 'lowres_224_to_640': low, 'highres_source_roi_to_640': high, 'delta_accuracy': high['accuracy'] - low['accuracy'], 'delta_f1_macro': high['f1_macro'] - low['f1_macro']})
    return rows

def paired_test(y: np.ndarray, lowres_probability: np.ndarray, highres_probability: np.ndarray, source: np.ndarray, samples: int) -> dict:
    low = lowres_probability.argmax(axis=1)
    high = highres_probability.argmax(axis=1)
    delta = (high == y).astype(np.float32) - (low == y).astype(np.float32)
    high_only = int(np.sum((high == y) & (low != y)))
    low_only = int(np.sum((high != y) & (low == y)))
    rng = np.random.default_rng(20260813)
    clip_boot = np.empty(samples, dtype=np.float32)
    for index in range(samples):
        selected = rng.integers(0, len(y), len(y))
        clip_boot[index] = delta[selected].mean()
    groups = sorted(set(source.astype(str)))
    group_sizes = np.asarray([np.sum(source == group) for group in groups], dtype=np.float64)
    group_delta = np.asarray([delta[source == group].mean() for group in groups], dtype=np.float64)
    group_boot = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(groups), len(groups))
        group_boot[index] = np.sum(group_sizes[selected] * group_delta[selected]) / np.sum(group_sizes[selected])
    p_value = None
    if binomtest is not None and high_only + low_only:
        p_value = float(binomtest(high_only, high_only + low_only, 0.5, alternative='greater').pvalue)
    return {'delta_accuracy': float(delta.mean()), 'clip_bootstrap_ci95': [float(np.quantile(clip_boot, 0.025)), float(np.quantile(clip_boot, 0.975))], 'source_group_bootstrap_ci95': [float(np.quantile(group_boot, 0.025)), float(np.quantile(group_boot, 0.975))], 'source_groups': int(len(groups)), 'highres_only_correct': high_only, 'lowres_only_correct': low_only, 'mcnemar_one_sided_highres_better_p': p_value}

def natural_check(x_low: np.ndarray, x_high: np.ndarray, names: np.ndarray, y: np.ndarray, historical: np.ndarray) -> list[dict]:
    root = CURRENT_DATASET
    lookup = {name: index for index, name in enumerate(names)}
    split_indices = {}
    for split, filename in (('train', 'train_natural.csv'), ('validation', 'validation.csv'), ('test', 'test_natural.csv')):
        frame = pd.read_csv(_resolve_model_artifact(study_csv_path(root / filename)))
        split_indices[split] = np.asarray([lookup[name] for name in frame.video_name], dtype=np.int32)
    train = split_indices['train']
    learned = {}
    for label, x in (('lowres', x_low), ('highres', x_high)):
        transformer = QuantileTransformer(n_quantiles=min(256, len(train)), output_distribution='normal', random_state=42)
        train_x = transformer.fit_transform(x[train])
        classifier = LogisticRegression(C=0.003, class_weight=MODERATE, max_iter=700, random_state=42).fit(train_x, y[train])
        learned[label] = classifier.predict_proba(transformer.transform(x))
    rows = []
    for split in ('validation', 'test'):
        hold = split_indices[split]
        low = 0.4 * historical[hold] + 0.6 * learned['lowres'][hold]
        high = 0.4 * historical[hold] + 0.6 * learned['highres'][hold]
        low_metrics = metrics(y[hold], low)
        high_metrics = metrics(y[hold], high)
        rows.append({'split': split, 'n': int(len(hold)), 'lowres_224_to_640': low_metrics, 'highres_source_roi_to_640': high_metrics, 'delta_accuracy': high_metrics['accuracy'] - low_metrics['accuracy'], 'delta_f1_macro': high_metrics['f1_macro'] - low_metrics['f1_macro']})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = _load_model_result_array(RGB_FEATURES, allow_pickle=True)
    dynamic = frozen['features'][:, :704].astype(np.float32)
    names = frozen['names'].astype(str)
    y = frozen['labels'].astype(np.int32)
    natural_historical = np.exp(frozen['features'][:, -3:].astype(np.float32))
    lowres_yolo = _load_model_result_array(LOWRES_MATRIX, mmap_mode='r').astype(np.float32)
    highres_yolo, quality = build_highres_matrix(names, args.cache_dir, args.output)
    if lowres_yolo.shape != highres_yolo.shape:
        raise ValueError(f'Descriptor mismatch: low={lowres_yolo.shape}, high={highres_yolo.shape}')
    x_low = np.concatenate([dynamic, lowres_yolo], axis=1)
    x_high = np.concatenate([dynamic, highres_yolo], axis=1)
    folds, external_historical, external_y, source = audit.load_external(names, y)
    if not np.array_equal(external_y, y):
        raise AssertionError('External labels do not match the frozen manifest')
    low_aux = external_predictions(x_low, y, folds)
    high_aux = external_predictions(x_high, y, folds)
    low_fused = 0.4 * external_historical + 0.6 * low_aux
    high_fused = 0.4 * external_historical + 0.6 * high_aux
    external = {'historical': metrics(y, external_historical), 'lowres_224_to_640': metrics(y, low_fused), 'highres_source_roi_to_640': metrics(y, high_fused), 'delta_highres_minus_lowres': {'accuracy': float(metrics(y, high_fused)['accuracy'] - metrics(y, low_fused)['accuracy']), 'balanced_accuracy': float(metrics(y, high_fused)['balanced_accuracy'] - metrics(y, low_fused)['balanced_accuracy']), 'f1_macro': float(metrics(y, high_fused)['f1_macro'] - metrics(y, low_fused)['f1_macro'])}, 'folds': fold_rows(y, external_historical, low_aux, high_aux, folds), 'paired': paired_test(y, low_fused, high_fused, source, args.bootstrap_samples)}
    alpha_rows = []
    for alpha in np.arange(0.0, 1.01, 0.1):
        probability = (1.0 - alpha) * external_historical + alpha * high_aux
        alpha_rows.append({'alpha_highres_auxiliary': round(float(alpha), 1), **metrics(y, probability)})
    result = {'protocol': {'controlled_change': 'YOLO pixels only: upsampled 224x224 versus source-resolution equivalent ROI', 'clips': int(len(y)), 'external_folds': 5, 'yolo_frames_per_clip': 100, 'fixed_features': '704 RGB dynamics + 707 E04 + 275 E06', 'fixed_classifier': 'QuantileTransformer(256, normal) + LogisticRegression(C=.003, moderate weights)', 'fixed_blend': {'historical': 0.4, 'auxiliary': 0.6}, 'selection': 'frozen before evaluation; no high-resolution hyperparameter search'}, 'descriptor_build': quality, 'external_audit': external, 'alpha_sensitivity_highres': alpha_rows, 'natural_train_only': natural_check(x_low, x_high, names, y, natural_historical), 'promotion_rule': 'Promote only if high-resolution improves external macro-F1 and does not reduce external accuracy or target-class recall materially.'}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(args.output / 'paired_predictions.npz', labels=y, names=names, historical=external_historical, lowres_auxiliary=low_aux, highres_auxiliary=high_aux, lowres_fused=low_fused, highres_fused=high_fused)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
