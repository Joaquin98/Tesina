"""Exhaustive 216-case search for the cached RGB+high-resolution-YOLO head.

The expensive ResNet-50 and YOLO passes are not repeated.  The experiment
uses the five pre-generated source-disjoint folds as fixed partitions:

* folds 1--3: fitting the grid candidates;
* fold 4: hyperparameter selection;
* fold 5: one untouched final test;

The final test is evaluated only after the winning validation configuration
has been selected.  A deployment candidate is then fitted on all 4,082 clips;
that all-data fit is never used to report generalization performance.
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
import argparse
import json
import os
import sys
import time
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
HIGHRES_FEATURES = HERE / 'outputs' / 'quantile_100frames' / 'e04_e06_highres_100frames.npy'
DEFAULT_OUTPUT = HERE / 'outputs' / 'head_grid_216_20260814'
LABELS = ('line', 'other', 'scrum')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.multimodal import audit_performance as audit
FEATURE_COLUMNS = {'rgb_e04': np.arange(0, 1411, dtype=np.int32), 'rgb_e06': np.concatenate([np.arange(0, 704, dtype=np.int32), np.arange(1411, 1686, dtype=np.int32)]), 'rgb_e04_e06': np.arange(0, 1686, dtype=np.int32)}
QUANTILES = (128, 256)
CS = (0.001, 0.003, 0.01)
ALPHAS = (0.5, 0.6, 0.7)
WEIGHTS = {'neutral': {0: 1.0, 1: 1.0, 2: 1.0}, 'current': {0: 1.2, 1: 0.65, 2: 1.8}, 'moderate': {0: 1.1, 1: 0.8, 2: 1.5}, 'conservative': {0: 0.9, 1: 1.0, 2: 1.2}}
CURRENT_CONFIG = {'feature_set': 'rgb_e04_e06', 'n_quantiles': 256, 'C': 0.003, 'alpha_auxiliary': 0.6, 'class_weight_profile': 'current'}

def json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray]:
    with _load_model_result_array(RGB_FEATURES, allow_pickle=True) as frozen:
        dynamic = frozen['features'][:, :704].astype(np.float32)
        names = frozen['names'].astype(str)
        y = frozen['labels'].astype(np.int64)
    highres = _load_model_result_array(HIGHRES_FEATURES).astype(np.float32)
    if dynamic.shape != (4082, 704) or highres.shape != (4082, 982):
        raise ValueError(f'Unexpected feature shapes: RGB={dynamic.shape}, YOLO={highres.shape}')
    x = np.concatenate([dynamic, highres], axis=1)
    folds, historical, external_y, source = audit.load_external(names, y)
    if not np.array_equal(y, external_y):
        raise AssertionError('External labels are not aligned with cached features')
    return (x, y, names, folds, historical.astype(np.float64), source)

def path_probabilities(z_train: np.ndarray, y_train: np.ndarray, z_validation: np.ndarray, columns: np.ndarray, weight_name: str) -> dict:
    """Fit the three C values as one warm-started multinomial path."""
    started = time.perf_counter()
    train = z_train[:, columns]
    validation = z_validation[:, columns]
    classifier = LogisticRegression(C=CS[0], max_iter=700, tol=0.0001, solver='lbfgs', class_weight=WEIGHTS[weight_name], random_state=42, warm_start=True)
    probabilities = []
    iterations = []
    for c_value in CS:
        classifier.set_params(C=c_value)
        classifier.fit(train, y_train)
        probabilities.append(classifier.predict_proba(validation))
        iterations.append(int(np.asarray(classifier.n_iter_).max()))
    return {'weight_name': weight_name, 'cs': list(CS), 'iterations': iterations, 'probabilities': np.asarray(probabilities, dtype=np.float32), 'seconds': float(time.perf_counter() - started)}

def score_grid(x: np.ndarray, y: np.ndarray, historical: np.ndarray, train: np.ndarray, validation: np.ndarray, jobs: int) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    timing: dict[str, dict] = {}
    for n_quantiles in QUANTILES:
        print(json.dumps({'stage': 'quantile', 'n_quantiles': n_quantiles}), flush=True)
        started = time.perf_counter()
        transformer = QuantileTransformer(n_quantiles=min(n_quantiles, len(train)), output_distribution='normal', random_state=42)
        z_train = transformer.fit_transform(x[train]).astype(np.float32)
        z_validation = transformer.transform(x[validation]).astype(np.float32)
        transform_seconds = float(time.perf_counter() - started)
        tasks = [(feature_name, columns, weight_name) for feature_name, columns in FEATURE_COLUMNS.items() for weight_name in WEIGHTS]
        with parallel_config(backend='loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=jobs, verbose=10)((delayed(path_probabilities)(z_train, y[train], z_validation, columns, weight_name) for _, columns, weight_name in tasks))
        timing[str(n_quantiles)] = {'quantile_transform_seconds': transform_seconds, 'paths': []}
        for (feature_name, _, weight_name), result in zip(tasks, results):
            timing[str(n_quantiles)]['paths'].append({'feature_set': feature_name, 'class_weight_profile': weight_name, 'seconds': result['seconds'], 'iterations': result['iterations']})
            for c_index, c_value in enumerate(result['cs']):
                auxiliary = result['probabilities'][c_index].astype(np.float64)
                for alpha in ALPHAS:
                    fused = (1.0 - alpha) * historical[validation] + alpha * auxiliary
                    metric = audit.metric_dict(y[validation], fused.argmax(axis=1))
                    rows.append({'feature_set': feature_name, 'feature_count': int(len(FEATURE_COLUMNS[feature_name])), 'n_quantiles': n_quantiles, 'C': float(c_value), 'class_weight_profile': weight_name, 'weight_line': WEIGHTS[weight_name][0], 'weight_other': WEIGHTS[weight_name][1], 'weight_scrum': WEIGHTS[weight_name][2], 'alpha_auxiliary': alpha, 'alpha_historical': 1.0 - alpha, 'accuracy': metric['accuracy'], 'balanced_accuracy': metric['balanced_accuracy'], 'f1_macro': metric['f1_macro']})
        print(json.dumps({'stage': 'quantile_complete', 'n_quantiles': n_quantiles, 'evaluated': len(rows)}), flush=True)
    ranking = pd.DataFrame(rows).sort_values(['f1_macro', 'balanced_accuracy', 'accuracy'], ascending=False, kind='mergesort')
    ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
    return (ranking.reset_index(drop=True), timing)

def config_from_row(row: pd.Series) -> dict:
    return {'feature_set': str(row['feature_set']), 'n_quantiles': int(row['n_quantiles']), 'C': float(row['C']), 'alpha_auxiliary': float(row['alpha_auxiliary']), 'class_weight_profile': str(row['class_weight_profile'])}

def fit_predict(config: dict, x: np.ndarray, y: np.ndarray, historical: np.ndarray, train: np.ndarray, hold: np.ndarray) -> tuple[np.ndarray, np.ndarray, QuantileTransformer, LogisticRegression]:
    columns = FEATURE_COLUMNS[config['feature_set']]
    transformer = QuantileTransformer(n_quantiles=min(config['n_quantiles'], len(train)), output_distribution='normal', random_state=42)
    train_z = transformer.fit_transform(x[train][:, columns])
    classifier = LogisticRegression(C=config['C'], class_weight=WEIGHTS[config['class_weight_profile']], max_iter=700, random_state=42).fit(train_z, y[train])
    auxiliary = classifier.predict_proba(transformer.transform(x[hold][:, columns]))
    alpha = config['alpha_auxiliary']
    fused = (1.0 - alpha) * historical[hold] + alpha * auxiliary
    return (auxiliary, fused, transformer, classifier)

def evaluate_configuration(name: str, config: dict, x: np.ndarray, y: np.ndarray, historical: np.ndarray, train: np.ndarray, hold: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    started = time.perf_counter()
    auxiliary, fused, _, classifier = fit_predict(config, x, y, historical, train, hold)
    result = {'name': name, 'config': config, 'n_fit': int(len(train)), 'n_hold': int(len(hold)), 'metrics': audit.metric_dict(y[hold], fused.argmax(axis=1)), 'classifier_iterations': np.asarray(classifier.n_iter_).astype(int).tolist(), 'seconds': float(time.perf_counter() - started)}
    return (result, auxiliary, fused)

def paired_comparison(y: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict:
    first_correct = first.argmax(axis=1) == y
    second_correct = second.argmax(axis=1) == y
    second_only = int(np.sum(second_correct & ~first_correct))
    first_only = int(np.sum(first_correct & ~second_correct))
    discordant = first_only + second_only
    return {'selected_only_correct': second_only, 'current_only_correct': first_only, 'discordant': discordant, 'mcnemar_exact_two_sided_p': float(binomtest(second_only, discordant, 0.5).pvalue) if discordant else 1.0}

def fit_deployment_candidate(config: dict, x: np.ndarray, y: np.ndarray, output: Path) -> dict:
    destination = output / 'deployment_candidate'
    destination.mkdir(parents=True, exist_ok=True)
    columns = FEATURE_COLUMNS[config['feature_set']]
    transformer = QuantileTransformer(n_quantiles=min(config['n_quantiles'], len(y)), output_distribution='normal', random_state=42)
    z = transformer.fit_transform(x[:, columns])
    classifier = LogisticRegression(C=config['C'], class_weight=WEIGHTS[config['class_weight_profile']], max_iter=700, random_state=42).fit(z, y)
    joblib.dump(transformer, destination / 'quantile_transformer.joblib')
    joblib.dump(classifier, destination / 'logistic_regression.joblib')
    metadata = {'config': config, 'feature_columns': columns.tolist(), 'feature_count': int(len(columns)), 'fit_clips': int(len(y)), 'labels': list(LABELS), 'warning': 'All-data deployment fit; do not use it to estimate generalization.', 'classifier_iterations': np.asarray(classifier.n_iter_).astype(int).tolist()}
    json_dump(destination / 'metadata.json', metadata)
    return metadata

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--jobs', type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    x, y, names, folds, historical, source = load_data()
    train = np.concatenate([np.asarray(folds[index]['hold']) for index in (0, 1, 2)])
    validation = np.asarray(folds[3]['hold'])
    test = np.asarray(folds[4]['hold'])
    if set(source[train]) & set(source[validation]) or set(source[train]) & set(source[test]):
        raise AssertionError('Source-group leakage between train, validation and test')
    if set(source[validation]) & set(source[test]):
        raise AssertionError('Source-group leakage between validation and test')
    protocol = {'grid_size': 216, 'selection_metric': 'validation f1_macro; ties: balanced_accuracy, accuracy', 'train_folds': [1, 2, 3], 'validation_fold': 4, 'untouched_test_fold': 5, 'counts': {'train': int(len(train)), 'validation': int(len(validation)), 'test': int(len(test))}, 'source_groups': {'train': sorted(set(source[train].tolist())), 'validation': sorted(set(source[validation].tolist())), 'test': sorted(set(source[test].tolist()))}, 'grid': {'feature_sets': {key: int(len(value)) for key, value in FEATURE_COLUMNS.items()}, 'n_quantiles': list(QUANTILES), 'C': list(CS), 'alpha_auxiliary': list(ALPHAS), 'class_weight_profiles': WEIGHTS}, 'cached_inputs': {'rgb': str(RGB_FEATURES.relative_to(REPO)), 'highres_yolo': str(HIGHRES_FEATURES.relative_to(REPO)), 'resnet_or_yolo_recomputed': False}}
    json_dump(args.output / 'protocol.json', protocol)
    print(json.dumps({'stage': 'grid_start', 'counts': protocol['counts']}), flush=True)
    ranking, timing = score_grid(x, y, historical, train, validation, max(1, args.jobs))
    if len(ranking) != 216:
        raise AssertionError(f'Expected 216 grid rows, got {len(ranking)}')
    ranking.to_csv(args.output / 'validation_ranking.csv', index=False)
    json_dump(args.output / 'timing.json', timing)
    selected = config_from_row(ranking.iloc[0])
    json_dump(args.output / 'selected_config.json', selected)
    print(json.dumps({'stage': 'selected', 'config': selected}), flush=True)
    fit_train = np.concatenate([train, validation])
    current_result, current_aux, current_fused = evaluate_configuration('current', CURRENT_CONFIG, x, y, historical, fit_train, test)
    if selected == CURRENT_CONFIG:
        selected_result = dict(current_result)
        selected_result['name'] = 'selected'
        selected_aux, selected_fused = (current_aux, current_fused)
    else:
        selected_result, selected_aux, selected_fused = evaluate_configuration('selected', selected, x, y, historical, fit_train, test)
    historical_metrics = audit.metric_dict(y[test], historical[test].argmax(axis=1))
    comparison = paired_comparison(y[test], current_fused, selected_fused)
    final_test = {'warning': 'Test fold 5 was untouched until after the winning validation row was fixed.', 'historical': historical_metrics, 'current': current_result, 'selected': selected_result, 'selected_vs_current': comparison}
    json_dump(args.output / 'final_test.json', final_test)
    np.savez_compressed(args.output / 'final_test_predictions.npz', names=names[test], labels=y[test], historical=historical[test].astype(np.float32), current_auxiliary=current_aux.astype(np.float32), current_fused=current_fused.astype(np.float32), selected_auxiliary=selected_aux.astype(np.float32), selected_fused=selected_fused.astype(np.float32))
    deployment = fit_deployment_candidate(selected, x, y, args.output)
    summary = {'protocol': protocol, 'selected_config': selected, 'best_validation': ranking.iloc[0].to_dict(), 'current_validation': ranking[(ranking.feature_set == CURRENT_CONFIG['feature_set']) & (ranking.n_quantiles == CURRENT_CONFIG['n_quantiles']) & np.isclose(ranking.C, CURRENT_CONFIG['C']) & np.isclose(ranking.alpha_auxiliary, CURRENT_CONFIG['alpha_auxiliary']) & (ranking.class_weight_profile == CURRENT_CONFIG['class_weight_profile'])].iloc[0].to_dict(), 'final_test': final_test, 'deployment_candidate': deployment, 'elapsed_seconds': float(time.perf_counter() - started)}
    json_dump(args.output / 'summary.json', summary)
    print(json.dumps({'stage': 'complete', 'elapsed_seconds': summary['elapsed_seconds'], 'selected_test': selected_result['metrics'], 'current_test': current_result['metrics']}), flush=True)
if __name__ == '__main__':
    main()
