"""Compara varias reglas de fusiÃ³n sobre las predicciones ya disponibles."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.M1.config import LABELS, RESULTS_DIR
from models.M1.late_fusion import geometry_statistics, rgb_test_probabilities, rgb_validation_probabilities
from models.M1.protocol import build_partition
OUTPUT_DIR = RESULTS_DIR / 'fusion_methods'
METHODS = ('weighted_sum', 'geometric_mean', 'rank_average', 'confidence_gate', 'stacked_logreg')

def normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(probabilities, 1e-07, None)
    return values / values.sum(axis=1, keepdims=True)

def rank_probabilities(probabilities: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(probabilities, dtype=np.float64)
    order = np.argsort(probabilities, axis=1)
    rows = np.arange(len(probabilities))[:, None]
    ranks[rows, order] = np.arange(probabilities.shape[1], dtype=np.float64)[None, :]
    return normalize(ranks + 1.0)

def fuse(prob_rgb: np.ndarray, prob_geometry: np.ndarray, method: str, alpha: float) -> np.ndarray:
    if method == 'weighted_sum':
        return normalize((1.0 - alpha) * prob_rgb + alpha * prob_geometry)
    if method == 'geometric_mean':
        values = np.exp((1.0 - alpha) * np.log(np.clip(prob_rgb, 1e-07, 1.0)) + alpha * np.log(np.clip(prob_geometry, 1e-07, 1.0)))
        return normalize(values)
    if method == 'rank_average':
        return normalize((1.0 - alpha) * rank_probabilities(prob_rgb) + alpha * rank_probabilities(prob_geometry))
    if method == 'confidence_gate':
        rgb_confidence = prob_rgb.max(axis=1)
        geometry_confidence = prob_geometry.max(axis=1)
        return np.where((geometry_confidence >= rgb_confidence)[:, None], prob_geometry, prob_rgb)
    raise ValueError(f'MÃ©todo desconocido: {method}')

def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {'test_accuracy': float(accuracy_score(labels, predictions)), 'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(labels, predictions, average='weighted', zero_division=0))}

def save_predictions(output_dir: Path, names: list[str], labels: np.ndarray, probabilities: np.ndarray) -> None:
    predictions = probabilities.argmax(axis=1)
    frame = pd.DataFrame({'video_name': names, 'true_label': [LABELS[index] for index in labels], 'predicted_label': [LABELS[index] for index in predictions]})
    for index, label in enumerate(LABELS):
        frame[f'prob_{label}'] = probabilities[:, index]
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / 'predictions.csv', index=False)

def run_seed(seed: int, pose_model: str, trees: int, alpha_step: float) -> list[dict]:
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=pose_model)
    train_x, train_y, _ = geometry_statistics(train_df, pose_model, 'both')
    validation_x, validation_y, validation_names = geometry_statistics(validation_df, pose_model, 'both')
    test_x, test_y, test_names = geometry_statistics(test_df, pose_model, 'both')
    geometry = RandomForestClassifier(n_estimators=trees, max_features='sqrt', class_weight='balanced', random_state=seed, n_jobs=-1)
    geometry.fit(train_x, train_y)
    geometry_validation = geometry.predict_proba(validation_x)
    geometry_test = geometry.predict_proba(test_x)
    rgb_validation, rgb_validation_y, rgb_validation_names = rgb_validation_probabilities(seed, validation_df, pose_model)
    if validation_names != rgb_validation_names or not np.array_equal(validation_y, rgb_validation_y):
        raise AssertionError('El orden de validaciÃ³n RGB y geometrÃ\xada no coincide')
    rgb_test = rgb_test_probabilities(seed, test_names)
    results: list[dict] = []
    alpha_values = np.arange(0.0, 1.0 + alpha_step * 0.5, alpha_step)
    for method in METHODS:
        if method == 'stacked_logreg':
            meta_validation = np.concatenate([rgb_validation, geometry_validation], axis=1)
            meta_test = np.concatenate([rgb_test, geometry_test], axis=1)
            meta = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed))
            meta.fit(meta_validation, validation_y)
            probabilities = meta.predict_proba(meta_test)
            validation_f1 = float(f1_score(validation_y, meta.predict(meta_validation), average='macro', zero_division=0))
            alpha = None
        elif method == 'confidence_gate':
            probabilities = fuse(rgb_test, geometry_test, method, 0.0)
            validation_predictions = fuse(rgb_validation, geometry_validation, method, 0.0).argmax(axis=1)
            validation_f1 = float(f1_score(validation_y, validation_predictions, average='macro', zero_division=0))
            alpha = None
        else:
            best = None
            for candidate_alpha in alpha_values:
                candidate = fuse(rgb_validation, geometry_validation, method, float(candidate_alpha))
                candidate_f1 = f1_score(validation_y, candidate.argmax(axis=1), average='macro', zero_division=0)
                if best is None or candidate_f1 > best[0]:
                    best = (float(candidate_f1), float(candidate_alpha))
            validation_f1, alpha = best
            probabilities = fuse(rgb_test, geometry_test, method, alpha)
        output_dir = OUTPUT_DIR / method / f'seed_{seed}'
        save_predictions(output_dir, test_names, test_y, probabilities)
        result = {'model': method, 'seed': int(seed), 'alpha_geometry': alpha, 'validation_f1_macro': validation_f1, 'geometry_features': int(train_x.shape[1]), 'random_forest_trees': int(trees), 'pose_model': pose_model, 'protocol': protocol, **metrics(test_y, probabilities)}
        (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results.append(result)
        print(json.dumps({key: result[key] for key in ('model', 'seed', 'alpha_geometry', 'test_accuracy', 'f1_macro')}, ensure_ascii=False))
    return results

def aggregate(results: list[dict]) -> None:
    frame = pd.DataFrame(results)
    frame.to_csv(OUTPUT_DIR / 'summary.csv', index=False)
    rows = []
    for method, group in frame.groupby('model'):
        row = {'model': method, 'runs': len(group)}
        for metric in ('test_accuracy', 'f1_macro', 'f1_weighted', 'validation_f1_macro'):
            row[f'{metric}_mean'] = float(group[metric].mean())
            row[f'{metric}_std'] = float(group[metric].std(ddof=0))
        rows.append(row)
    pd.DataFrame(rows).sort_values('model').to_csv(OUTPUT_DIR / 'aggregate_metrics.csv', index=False)

def bootstrap_against_m0(results: list[dict], samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for method in METHODS:
        method_results = [item for item in results if item['model'] == method]
        deltas = {'test_accuracy': [], 'f1_macro': [], 'f1_weighted': []}
        for result in method_results:
            run_seed = int(result['seed'])
            candidate = pd.read_csv(_resolve_model_artifact(OUTPUT_DIR / method / f'seed_{run_seed}' / 'predictions.csv'))
            baseline = pd.read_csv(_resolve_model_artifact(RESULTS_DIR / 'm0' / f'seed_{run_seed}' / 'predictions.csv')).set_index('video_name')
            baseline = baseline.loc[candidate['video_name'].tolist()].reset_index()
            for _ in range(samples):
                indexes = rng.integers(0, len(candidate), len(candidate))
                c = candidate.iloc[indexes].reset_index(drop=True)
                b = baseline.iloc[indexes].reset_index(drop=True)
                deltas['test_accuracy'].append(float(np.mean(c['predicted_label'] == c['true_label']) - np.mean(b['predicted_label'] == b['true_label'])))
                deltas['f1_macro'].append(float(f1_score(c['true_label'], c['predicted_label'], average='macro', zero_division=0) - f1_score(b['true_label'], b['predicted_label'], average='macro', zero_division=0)))
                deltas['f1_weighted'].append(float(f1_score(c['true_label'], c['predicted_label'], average='weighted', zero_division=0) - f1_score(b['true_label'], b['predicted_label'], average='weighted', zero_division=0)))
        row = {'model': method, 'baseline': 'm0', 'seeds': ','.join((str(item['seed']) for item in method_results)), 'bootstrap_samples_per_seed': samples}
        for metric, values in deltas.items():
            values = np.asarray(values, dtype=np.float64)
            row[f'{metric}_delta_mean'] = float(values.mean())
            row[f'{metric}_delta_low'] = float(np.quantile(values, 0.025))
            row[f'{metric}_delta_high'] = float(np.quantile(values, 0.975))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'paired_bootstrap.csv', index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default='yolo11n-pose.pt')
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--alpha-step', type=float, default=0.01)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        results.extend(run_seed(seed, args.pose_model, args.trees, args.alpha_step))
    aggregate(results)
    bootstrap_against_m0(results, args.bootstrap_samples, seed=20260807)
    print(f'Resultados: {OUTPUT_DIR}')
if __name__ == '__main__':
    main()
