"""Search cached YOLO-pose descriptor variants without reopening video clips.

The 4,082 high-resolution pose caches contain 100 temporal samples with boxes,
person confidence, keypoints and keypoint confidence.  This script rebuilds
descriptor primitives from those arrays, screens E01--E09 plus the promoted
E04+E06 representation, and then evaluates a 216-case descriptor grid.

Source-disjoint folds 1--3 fit candidates, fold 4 selects them, and fold 5 is
opened exactly once after selection.  ResNet-50 and YOLO are never executed.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
REPO = Path(__file__).resolve().parents[2]
HERE = experiment_directory(__file__)
JOINT = REPO / 'data' / 'joint_multimodal_20260813'
RGB_FEATURES = JOINT / 'rgb_dynamics_seed42' / 'fusion_features.npz'
POSE_CACHE = HERE / 'outputs' / 'cache' / 'yolo11n-pose_pt'
CURRENT_YOLO = HERE / 'outputs' / 'quantile_100frames' / 'e04_e06_highres_100frames.npy'
DEFAULT_OUTPUT = HERE / 'outputs' / 'cached_descriptor_grid_20260814'
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PLAYER = REPO / 'data' / 'player_geometry_fusion'
if str(PLAYER) not in sys.path:
    sys.path.insert(0, str(source_directory(PLAYER)))
from models.M2.multimodal import audit_performance as audit
from models.M2.studies.yolo_resnet_improvement_suite_20260811 import run_suite as suite
LABELS = ('line', 'other', 'scrum')
WEIGHTS = {0: 1.2, 1: 0.65, 2: 1.8}
REPRESENTATIONS = ('E01_confidence_weighted', 'E02_quality_mask', 'E03_roi_topk', 'E04_tracking_smoothing', 'E05_camera_normalized', 'E06_rugby_descriptors', 'E07_gated_residual', 'E08_dual_temporal', 'E09_player_graph', 'E04_E06_promoted')
PROFILES = {'current': (0.25, 0.2), 'medium': (0.35, 0.3), 'strict': (0.45, 0.4)}
BUDGETS = (25, 50, 75, 100)
SMOOTH_WINDOWS = (1, 3, 5)
POOLINGS = ('global', 'global_halves')

def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def load_inputs():
    with _load_model_result_array(RGB_FEATURES, allow_pickle=True) as data:
        dynamic = data['features'][:, :704].astype(np.float32)
        names = data['names'].astype(str)
        y = data['labels'].astype(np.int64)
    folds, historical, external_y, source = audit.load_external(names, y)
    if not np.array_equal(y, external_y):
        raise AssertionError('Label mismatch between RGB and external folds')
    return (dynamic, names, y, folds, historical.astype(np.float64), source)

def load_clip(name: str) -> dict[str, np.ndarray]:
    path = POSE_CACHE / f'{Path(name).stem}.npz'
    with _load_model_result_array(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}

def topk_formation(data: dict[str, np.ndarray], top_k: int=12) -> np.ndarray:
    rows = []
    for boxes, confidence, valid in zip(data['boxes'], data['person_conf'], data['valid']):
        keep = np.asarray(valid, dtype=bool).copy()
        if keep.any():
            centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            centrality = 1.0 - np.linalg.norm(centers - 0.5, axis=1) / 0.71
            score = confidence * np.clip(centrality, 0.25, 1.0)
            indexes = np.flatnonzero(keep)
            selected = indexes[np.argsort(score[indexes])[::-1][:top_k]]
            keep[:] = False
            keep[selected] = True
        rows.append(suite.formation_frame(boxes, confidence, keep))
    return np.asarray(rows, dtype=np.float32)

def primitive_base(raw: dict[str, np.ndarray]):
    """Build reusable sequences from the already accepted YOLO detections."""
    valid = raw['valid'].astype(bool)
    positions = raw['position_features'].astype(np.float32)
    poses = raw['pose_features'].astype(np.float32)
    frame_mask = raw['frame_mask'].astype(bool)
    prepared = {'boxes': raw['boxes'], 'person_conf': raw['person_conf'], 'keypoints': raw['keypoints'], 'keypoint_conf': raw['keypoint_conf'], 'valid': valid, 'position_features': positions, 'pose_features': poses, 'frame_mask': frame_mask}
    quality, _ = suite.quality_channels(prepared)
    posture = poses[:, suite.POSTURE_CLEAN_INDICES]
    return {'current': np.concatenate([positions, poses], axis=1).astype(np.float32), 'quality': quality.astype(np.float32), 'formation': topk_formation(prepared), 'posture': posture.astype(np.float32), 'rugby': np.stack([suite.rugby_frame(prepared, i) for i in range(100)]).astype(np.float32), 'graph': np.stack([suite.graph_frame(prepared, i) for i in range(100)]).astype(np.float32), 'normalized': np.concatenate([suite.relative_positions(prepared), posture], axis=1).astype(np.float32), 'mask': frame_mask.astype(bool), 'quality_mask': frame_mask.astype(bool)}

def primitive_for_name(name: str):
    """Top-level worker so Windows can parallelize the CPU-bound rebuild."""
    raw = load_clip(name)
    base = primitive_base(raw)
    result = {}
    for profile, (person_threshold, keypoint_threshold) in PROFILES.items():
        arrays = {key: value for key, value in base.items()}
        if profile == 'current':
            quality_mask = base['mask'].copy()
        else:
            quality = base['quality']
            quality_mask = base['mask'] & (quality[:, 0] > 0.0) & (quality[:, 1] >= person_threshold) & (quality[:, 2] >= keypoint_threshold)
        arrays['mask'] = quality_mask.astype(bool)
        arrays['quality_mask'] = quality_mask.astype(bool)
        result[profile] = arrays
    return result

def build_primitive_cache(names: np.ndarray, root: Path, workers: int) -> dict:
    metadata_path = root / 'metadata.json'
    if _resolve_model_artifact(metadata_path).exists():
        metadata = json.loads(_resolve_model_artifact(metadata_path).read_text(encoding='utf-8'))
        if metadata.get('clips') == len(names) and metadata.get('complete') is True:
            return metadata
    root.mkdir(parents=True, exist_ok=True)
    first = primitive_for_name(str(names[0]))
    stores: dict[str, dict[str, np.memmap]] = {}
    shapes: dict[str, dict[str, list[int]]] = {}
    for profile, arrays in first.items():
        profile_dir = root / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        stores[profile] = {}
        shapes[profile] = {}
        for key, value in arrays.items():
            shape = (len(names),) + value.shape
            stores[profile][key] = np.lib.format.open_memmap(profile_dir / f'{key}.npy', mode='w+', dtype=value.dtype, shape=shape)
            shapes[profile][key] = list(shape)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(primitive_for_name, names.tolist(), chunksize=8)):
            for profile, arrays in result.items():
                for key, value in arrays.items():
                    stores[profile][key][index] = value
            if (index + 1) % 250 == 0:
                print(json.dumps({'stage': 'primitive_cache', 'clips': index + 1}), flush=True)
    for profile_stores in stores.values():
        for store in profile_stores.values():
            store.flush()
    metadata = {'complete': True, 'clips': int(len(names)), 'frames': 100, 'profiles': {key: list(value) for key, value in PROFILES.items()}, 'shapes': shapes, 'seconds': float(time.perf_counter() - started), 'source': str(POSE_CACHE.relative_to(REPO)), 'video_or_yolo_recomputed': False}
    dump_json(metadata_path, metadata)
    return metadata

def open_primitives(root: Path, profile: str) -> dict[str, np.ndarray]:
    return {path.stem: _load_model_result_array(path, mmap_mode='r') for path in sorted(_model_artifact_glob(root / profile, '*.npy', recursive=False))}

def sample_positions(budget: int) -> np.ndarray:
    return np.unique(np.rint(np.linspace(0, 99, budget)).astype(np.int32))

def smooth(sequence: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32)
    if window <= 1 or len(values) < 3:
        return values
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode='edge')
    shifted = [padded[offset:offset + len(values)] for offset in range(window)]
    result = np.median(np.stack(shifted, axis=0), axis=0)
    result[~mask] = values[~mask]
    return result.astype(np.float32)

def representation_components(rep: str, arrays: dict[str, np.ndarray], mask: np.ndarray, qmask: np.ndarray, window: int):
    quality = arrays['quality']

    def smoothed(key: str) -> np.ndarray:
        return smooth(arrays[key], mask, window)
    if rep == 'E01_confidence_weighted':
        weight_mask = mask & (quality[:, 1] * quality[:, 2] * quality[:, 3] > 0)
        return [(smoothed('current'), weight_mask, False), (quality, mask, False)]
    if rep == 'E02_quality_mask':
        return [(smoothed('current'), qmask, False), (quality, mask, False)]
    if rep == 'E03_roi_topk':
        return [(smoothed('formation'), mask, True), (smoothed('posture'), mask, False)]
    if rep == 'E04_tracking_smoothing':
        return [(smoothed('current'), mask, True), (quality, mask, False)]
    if rep == 'E05_camera_normalized':
        return [(smoothed('normalized'), mask, True), (quality, mask, False)]
    if rep == 'E06_rugby_descriptors':
        return [(smoothed('rugby'), mask, True)]
    if rep == 'E07_gated_residual':
        gate = 1.0 / (1.0 + np.exp(-8.0 * (quality[:, 1] - 0.35))) * quality[:, 3]
        current = smoothed('current')
        gated = current * gate[:, None]
        return [(current, mask, False), (gated, mask, False), (quality, mask, False)]
    if rep == 'E08_dual_temporal':
        return [(smoothed('current'), mask, True), (smoothed('formation'), mask, True), (smoothed('posture'), mask, True)]
    if rep == 'E09_player_graph':
        return [(smoothed('graph'), mask, True), (quality, mask, True)]
    if rep == 'E04_E06_promoted':
        return [(smoothed('current'), mask, True), (quality, mask, False), (arrays['rugby'], mask, True)]
    raise ValueError(rep)

def pool_components(components, pooling: str) -> np.ndarray:
    length = len(components[0][0])
    segments = [(0, length)]
    if pooling == 'global_halves':
        midpoint = length // 2
        segments.extend([(0, midpoint), (midpoint, length)])
    values = []
    for start, stop in segments:
        for sequence, mask, temporal in components:
            values.append(suite.summarize(sequence[start:stop], mask[start:stop], temporal))
    return np.concatenate(values).astype(np.float32)

def descriptor_matrix(root: Path, config: dict) -> np.ndarray:
    primitives = open_primitives(root, config['profile'])
    selected = sample_positions(config['frame_budget'])
    rows = []
    for index in range(primitives['mask'].shape[0]):
        mask = np.asarray(primitives['mask'][index, selected], dtype=bool)
        qmask = np.asarray(primitives['quality_mask'][index, selected], dtype=bool)
        arrays = {key: np.asarray(value[index, selected]) for key, value in primitives.items() if key not in ('mask', 'quality_mask')}
        components = representation_components(config['representation'], arrays, mask, qmask, config['smooth_window'])
        rows.append(pool_components(components, config['pooling']))
    return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def evaluate_config(config, primitive_root, dynamic, y, historical, train, hold):
    started = time.perf_counter()
    descriptor = descriptor_matrix(primitive_root, config)
    x = np.concatenate([dynamic, descriptor], axis=1)
    transformer = QuantileTransformer(n_quantiles=256, output_distribution='normal', random_state=42)
    train_z = transformer.fit_transform(x[train])
    classifier = LogisticRegression(C=0.003, class_weight=WEIGHTS, max_iter=700, random_state=42).fit(train_z, y[train])
    auxiliary = classifier.predict_proba(transformer.transform(x[hold]))
    fused = 0.4 * historical[hold] + 0.6 * auxiliary
    metrics = audit.metric_dict(y[hold], fused.argmax(axis=1))
    return {**config, 'descriptor_features': int(descriptor.shape[1]), 'total_features': int(x.shape[1]), 'accuracy': metrics['accuracy'], 'balanced_accuracy': metrics['balanced_accuracy'], 'f1_macro': metrics['f1_macro'], 'seconds': float(time.perf_counter() - started)}

def rank_configs(configs, primitive_root, dynamic, y, historical, train, hold, jobs):
    with parallel_config(backend='loky', inner_max_num_threads=1):
        rows = Parallel(n_jobs=jobs, verbose=10, max_nbytes='1M')((delayed(evaluate_config)(config, primitive_root, dynamic, y, historical, train, hold) for config in configs))
    frame = pd.DataFrame(rows).sort_values(['f1_macro', 'balanced_accuracy', 'accuracy'], ascending=False, kind='mergesort').reset_index(drop=True)
    frame.insert(0, 'rank', np.arange(1, len(frame) + 1))
    return frame

def fit_final(config, primitive_root, dynamic, y, historical, train, hold):
    descriptor = descriptor_matrix(primitive_root, config)
    x = np.concatenate([dynamic, descriptor], axis=1)
    transformer = QuantileTransformer(n_quantiles=256, output_distribution='normal', random_state=42)
    classifier = LogisticRegression(C=0.003, class_weight=WEIGHTS, max_iter=700, random_state=42).fit(transformer.fit_transform(x[train]), y[train])
    auxiliary = classifier.predict_proba(transformer.transform(x[hold]))
    fused = 0.4 * historical[hold] + 0.6 * auxiliary
    return (descriptor, transformer, classifier, auxiliary, fused)

def config_from_row(row: pd.Series) -> dict:
    return {'representation': str(row.representation), 'frame_budget': int(row.frame_budget), 'smooth_window': int(row.smooth_window), 'profile': str(row.profile), 'pooling': str(row.pooling)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--jobs', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--cache-workers', type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dynamic, names, y, folds, historical, source = load_inputs()
    train = np.concatenate([folds[index]['hold'] for index in (0, 1, 2)])
    validation = np.asarray(folds[3]['hold'])
    test = np.asarray(folds[4]['hold'])
    if set(source[train]) & set(source[validation]) or set(source[train]) & set(source[test]):
        raise AssertionError('Source-group leakage')
    if set(source[validation]) & set(source[test]):
        raise AssertionError('Source-group leakage')
    primitive_root = args.output / 'primitive_cache'
    metadata = build_primitive_cache(names, primitive_root, args.cache_workers)
    print(json.dumps({'stage': 'primitive_cache_complete', 'seconds': metadata['seconds']}), flush=True)
    current_config = {'representation': 'E04_E06_promoted', 'frame_budget': 100, 'smooth_window': 3, 'profile': 'current', 'pooling': 'global'}
    rebuilt_current = descriptor_matrix(primitive_root, current_config)
    frozen_current = _load_model_result_array(CURRENT_YOLO)
    max_current_delta = float(np.max(np.abs(rebuilt_current - frozen_current)))
    if max_current_delta > 1e-06:
        raise AssertionError(f'Current descriptor reconstruction mismatch: {max_current_delta}')
    del rebuilt_current, frozen_current
    screening_configs = []
    for representation in REPRESENTATIONS:
        screening_configs.append({'representation': representation, 'frame_budget': 100, 'smooth_window': 3 if representation in ('E04_tracking_smoothing', 'E04_E06_promoted') else 1, 'profile': 'current', 'pooling': 'global'})
    screening_path = args.output / 'screening_ranking.csv'
    if _resolve_model_artifact(screening_path).exists() and len(pd.read_csv(_resolve_model_artifact(screening_path))) == len(screening_configs):
        screening = pd.read_csv(_resolve_model_artifact(screening_path))
        finalists = screening.head(3).representation.astype(str).tolist()
        print(json.dumps({'stage': 'screening_reused', 'finalists': finalists}), flush=True)
    else:
        print(json.dumps({'stage': 'screening_start', 'variants': len(screening_configs)}), flush=True)
        screening = rank_configs(screening_configs, primitive_root, dynamic, y, historical, train, validation, args.jobs)
        screening.to_csv(screening_path, index=False)
        finalists = screening.head(3).representation.astype(str).tolist()
        dump_json(args.output / 'screening_finalists.json', finalists)
        print(json.dumps({'stage': 'screening_complete', 'finalists': finalists}), flush=True)
    grid = [{'representation': representation, 'frame_budget': budget, 'smooth_window': window, 'profile': profile, 'pooling': pooling} for representation, budget, window, profile, pooling in itertools.product(finalists, BUDGETS, SMOOTH_WINDOWS, PROFILES, POOLINGS)]
    if len(grid) != 216:
        raise AssertionError(len(grid))
    print(json.dumps({'stage': 'grid_start', 'variants': len(grid)}), flush=True)
    ranking = rank_configs(grid, primitive_root, dynamic, y, historical, train, validation, args.jobs)
    ranking.to_csv(args.output / 'validation_ranking.csv', index=False)
    selected = config_from_row(ranking.iloc[0])
    dump_json(args.output / 'selected_config.json', selected)
    print(json.dumps({'stage': 'selected', 'config': selected}), flush=True)
    fit_train = np.concatenate([train, validation])
    selected_descriptor, transformer, classifier, selected_aux, selected_fused = fit_final(selected, primitive_root, dynamic, y, historical, fit_train, test)
    current_descriptor, _, _, current_aux, current_fused = fit_final(current_config, primitive_root, dynamic, y, historical, fit_train, test)
    selected_metrics = audit.metric_dict(y[test], selected_fused.argmax(axis=1))
    current_metrics = audit.metric_dict(y[test], current_fused.argmax(axis=1))
    historical_metrics = audit.metric_dict(y[test], historical[test].argmax(axis=1))
    selected_correct = selected_fused.argmax(axis=1) == y[test]
    current_correct = current_fused.argmax(axis=1) == y[test]
    selected_only = int(np.sum(selected_correct & ~current_correct))
    current_only = int(np.sum(current_correct & ~selected_correct))
    discordant = selected_only + current_only
    paired = {'selected_only_correct': selected_only, 'current_only_correct': current_only, 'discordant': discordant, 'mcnemar_exact_two_sided_p': float(binomtest(selected_only, discordant, 0.5).pvalue) if discordant else 1.0}
    np.save(args.output / 'selected_descriptor_matrix.npy', selected_descriptor)
    model_dir = args.output / 'selected_model'
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(transformer, model_dir / 'quantile_transformer.joblib')
    joblib.dump(classifier, model_dir / 'logistic_regression.joblib')
    np.savez_compressed(args.output / 'final_test_predictions.npz', names=names[test], labels=y[test], historical=historical[test].astype(np.float32), current_auxiliary=current_aux.astype(np.float32), current_fused=current_fused.astype(np.float32), selected_auxiliary=selected_aux.astype(np.float32), selected_fused=selected_fused.astype(np.float32))
    protocol = {'clips': int(len(y)), 'train_folds': [1, 2, 3], 'validation_fold': 4, 'untouched_test_fold': 5, 'counts': {'train': int(len(train)), 'validation': int(len(validation)), 'test': int(len(test))}, 'selection': 'macro-F1 validation; balanced accuracy and accuracy as ties', 'screening_variants': len(screening_configs), 'grid_variants': len(grid), 'fixed_head': 'QuantileTransformer(256, normal) + LogisticRegression(C=.003, weights 1.2/.65/1.8)', 'fixed_fusion': {'historical': 0.4, 'auxiliary': 0.6}, 'video_or_yolo_recomputed': False, 'primitive_reconstruction_max_abs_delta': max_current_delta}
    summary = {'protocol': protocol, 'screening_top3': screening.head(3).to_dict(orient='records'), 'selected_config': selected, 'best_validation': ranking.iloc[0].to_dict(), 'final_test': {'historical': historical_metrics, 'current': current_metrics, 'selected': selected_metrics, 'selected_vs_current': paired}, 'elapsed_seconds': float(time.perf_counter() - started)}
    dump_json(args.output / 'protocol.json', protocol)
    dump_json(args.output / 'summary.json', summary)
    dump_json(model_dir / 'metadata.json', {'config': selected, 'descriptor_features': int(selected_descriptor.shape[1]), 'total_features': int(dynamic.shape[1] + selected_descriptor.shape[1]), 'labels': list(LABELS), 'fit_clips': int(len(fit_train)), 'warning': 'Experimental selection; test clips are excluded from this fitted artifact.'})
    print(json.dumps({'stage': 'complete', 'elapsed_seconds': summary['elapsed_seconds'], 'selected_test': selected_metrics, 'current_test': current_metrics, 'paired': paired}), flush=True)
if __name__ == '__main__':
    main()
