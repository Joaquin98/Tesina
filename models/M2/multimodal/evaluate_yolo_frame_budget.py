"""Evaluate how many cached YOLO frames are needed by the deployed fusion.

The production model was trained from 200 uniformly sampled pose detections per
clip.  This experiment keeps the RGB dynamics and the frozen historical
probabilities unchanged, then recomputes only E04/E06 from uniform subsets of
the cached 200-frame pose sequence.  Every candidate is evaluated with the
same five disjoint external folds and a fresh QuantileTransformer + logistic
regression fitted only on the non-holdout clips.

This is an inference-budget experiment: it does not run YOLO again.  Taking a
subset of the cached detections is equivalent to skipping the other detector
calls at the same temporal positions, and avoids confounding detector changes
with classifier changes.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
HERE = experiment_directory(__file__)
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite import load_clip, make_variant
from models.M2.multimodal import audit_performance as audit
FRAME_BUDGETS = (25, 50, 100, 150, 200)
HISTORICAL_WEIGHT = 0.4
YOLO_WEIGHT = 0.6

def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    pred = probabilities.argmax(axis=1)
    return {'accuracy': float(accuracy_score(y_true, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)), 'f1_macro': float(f1_score(y_true, pred, average='macro', zero_division=0))}

def frame_indices(total: int, count: int) -> np.ndarray:
    if count < 1 or count > total:
        raise ValueError(f'frame budget must be in [1, {total}], got {count}')
    if count == total:
        return np.arange(total, dtype=np.int32)
    indices = np.rint(np.linspace(0, total - 1, count)).astype(np.int32)
    if len(np.unique(indices)) != count:
        raise AssertionError(f'non-unique temporal indices for budget {count}')
    return indices

def downsample_pose(data: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    result = {}
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim and array.shape[0] == 200:
            result[key] = array[indices]
        else:
            result[key] = array
    return result

def build_yolo_matrices(names: np.ndarray, budgets: tuple[int, ...]) -> tuple[dict[int, np.ndarray], float]:
    indices_by_budget = {budget: frame_indices(200, budget) for budget in budgets}
    started = time.perf_counter()
    workers = min(8, max(2, (os.cpu_count() or 4) // 2))

    def one_clip(name: str) -> list[np.ndarray]:
        cached = load_clip(str(name))
        rows = []
        for budget in budgets:
            sampled = downsample_pose(cached, indices_by_budget[budget])
            e04 = make_variant(sampled, 'E04_tracking_smoothing')
            e06 = make_variant(sampled, 'E06_rugby_descriptors')
            rows.append(np.concatenate([e04, e06]).astype(np.float32, copy=False))
        return rows
    rows_by_budget = {budget: [] for budget in budgets}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for position, rows in enumerate(pool.map(one_clip, names), start=1):
            for budget, row in zip(budgets, rows):
                rows_by_budget[budget].append(row)
            if position % 500 == 0:
                print(json.dumps({'processed': position, 'budgets': list(budgets), 'workers': workers}), flush=True)
    elapsed = time.perf_counter() - started
    matrices = {budget: np.asarray(rows, dtype=np.float32) for budget, rows in rows_by_budget.items()}
    for budget, matrix in matrices.items():
        if matrix.shape[1] != 982:
            raise AssertionError(f'unexpected E04+E06 dimension for {budget}: {matrix.shape}')
    return (matrices, elapsed)

def evaluate_budget(budget: int, yolo: np.ndarray, dynamic: np.ndarray, names: np.ndarray, y: np.ndarray, historical: np.ndarray, folds: list[dict]) -> dict:
    x = np.concatenate([dynamic, yolo], axis=1)
    yolo_probabilities = np.zeros((len(y), 3), dtype=np.float32)
    fold_metrics = []
    for fold in folds:
        hold = np.asarray(fold['hold'], dtype=np.int32)
        fit_mask = np.ones(len(y), dtype=bool)
        fit_mask[hold] = False
        yolo_probabilities[hold] = audit.fit_quantile(x[fit_mask], y[fit_mask], x[hold], C=0.003, class_weight=audit.MODERATE)
        fused_hold = HISTORICAL_WEIGHT * historical[hold] + YOLO_WEIGHT * yolo_probabilities[hold]
        fold_metrics.append({'fold': int(fold['fold']), 'n': int(len(hold)), 'fused': metrics(y[hold], fused_hold)})
    fused = HISTORICAL_WEIGHT * historical + YOLO_WEIGHT * yolo_probabilities
    return {'frame_budget': int(budget), 'frames_per_clip_ratio': float(budget / 200.0), 'estimated_detector_calls_reduction': float(1.0 - budget / 200.0), 'yolo_only': metrics(y, yolo_probabilities), 'fused': metrics(y, fused), 'folds': fold_metrics, 'max_abs_difference_from_200': None}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--budgets', nargs='+', type=int, default=list(FRAME_BUDGETS), help='YOLO frame budgets to evaluate (default: 25 50 100 150 200)')
    parser.add_argument('--output', type=Path, default=HERE / 'yolo_frame_budget_20260813', help='directory for metrics and cached reduced descriptors')
    args = parser.parse_args()
    budgets = tuple(sorted(set(args.budgets)))
    if not budgets or any((budget < 1 or budget > 200 for budget in budgets)):
        raise ValueError('budgets must be integers in [1, 200]')
    features = _load_model_result_array(HERE / 'rgb_dynamics_seed42/fusion_features.npz', allow_pickle=True)
    dynamic = features['features'][:, :704].astype(np.float32)
    names = features['names'].astype(str)
    y = features['labels'].astype(int)
    natural_historical = np.exp(features['features'][:, -3:].astype(np.float32))
    folds, external_historical, external_y, _ = audit.load_external(names, y)
    if not np.array_equal(external_y, y):
        raise AssertionError('external labels do not match the feature manifest')
    historical = external_historical.astype(np.float32)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline_yolo = _load_model_result_array(HERE / 'multi_yolo_seed42/E04_tracking_smoothing.npy').astype(np.float32)
    baseline_yolo = np.concatenate([baseline_yolo, _load_model_result_array(HERE / 'invariant_yolo_seed42/E06_rugby_descriptors.npy').astype(np.float32)], axis=1)
    matrices = {}
    pending = []
    for budget in budgets:
        cache_path = output / f'e04_e06_{budget}frames.npy'
        if _resolve_model_artifact(cache_path).exists():
            matrices[budget] = _load_model_result_array(cache_path, mmap_mode='r')
        else:
            pending.append(budget)
    if pending:
        built, feature_seconds = build_yolo_matrices(names, tuple(pending))
        matrices.update(built)
        for budget, matrix in built.items():
            np.save(output / f'e04_e06_{budget}frames.npy', matrix)
    else:
        feature_seconds = 'cached'
    for budget in budgets:
        result = evaluate_budget(budget, matrices[budget], dynamic, names, y, historical, folds)
        result['feature_rebuild_seconds'] = feature_seconds if isinstance(feature_seconds, str) else float(feature_seconds)
        result['feature_rebuild_seconds_per_clip'] = feature_seconds if isinstance(feature_seconds, str) else float(feature_seconds / len(y))
        if budget == 200:
            result['max_abs_difference_from_200'] = float(np.max(np.abs(matrices[budget] - baseline_yolo)))
        rows.append(result)
        (output / f'metrics_{budget}frames.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({'frame_budget': budget, 'fused': result['fused'], 'feature_seconds': result['feature_rebuild_seconds'], 'max_abs_difference_from_production': result['max_abs_difference_from_200']}, ensure_ascii=False), flush=True)
    baseline = next((row for row in rows if row['frame_budget'] == 200), None)
    summary = {'protocol': {'clips': int(len(y)), 'external_folds': 5, 'features': '704 fixed RGB dynamics + E04/E06 recomputed from cached pose detections', 'classifier': 'QuantileTransformer(256, normal) + LogisticRegression(C=.003, moderate weights)', 'blend': {'historical': HISTORICAL_WEIGHT, 'yolo': YOLO_WEIGHT}, 'temporal_sampling': 'uniform subset of the 200 cached YOLO detections, endpoints retained', 'note': 'No YOLO detector rerun; this isolates the effect of skipping detector calls.'}, 'historical_control': metrics(y, external_historical), 'natural_cache_historical_vs_external': {'max_abs_probability_difference': float(np.max(np.abs(natural_historical - external_historical))), 'mean_abs_probability_difference': float(np.mean(np.abs(natural_historical - external_historical)))}, 'production_200_reference': {'audit_fused_accuracy': 0.8407643312101911, 'audit_fused_f1_macro': 0.8140962271208876, 'recomputed_max_abs_feature_difference': baseline['max_abs_difference_from_200'] if baseline is not None else None}, 'results': rows, 'recommendation_rule': 'smallest budget whose fused accuracy drop is <= 0.01 percentage points and whose F1 drop is <= 0.01'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
