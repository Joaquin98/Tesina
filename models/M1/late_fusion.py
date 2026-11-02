"""FusiÃ³n tardÃ\xada RGB + geometrÃ\xada rica usando las cachÃ©s existentes.

El RGB se toma del checkpoint M0 ya entrenado. La rama geomÃ©trica se entrena
con un Random Forest sobre estadÃ\xadsticas temporales de cajas y esqueletos. El
peso de la rama geomÃ©trica se elige solamente con validaciÃ³n y se evalÃºa una
sola vez sobre el mismo conjunto de prueba.
"""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.M1.config import LABELS, LABEL_TO_ID, RESULTS_DIR
from models.M1.protocol import build_partition, load_arrays, load_geometry_features
from models.M1.train import build_model, inputs_for_arrays
OUTPUT_DIR = RESULTS_DIR / 'late_fusion'
POSE_MODEL_DEFAULT = 'yolo11n-pose.pt'

def geometry_statistics(df: pd.DataFrame, pose_model: str, kind: str='both') -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convierte detecciones por frame en un vector temporal rico por clip."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    for row in df.to_dict('records'):
        name = str(row['video_name'])
        positions, poses, mask = load_geometry_features(name, pose_model)
        if kind == 'positions':
            sequence = positions
        elif kind == 'poses':
            sequence = poses
        elif kind == 'both':
            sequence = np.concatenate([positions, poses], axis=-1)
        else:
            raise ValueError(f'Tipo geomÃ©trico desconocido: {kind}')
        valid = sequence[mask] if np.any(mask) else sequence
        deltas = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, sequence.shape[-1]), dtype=np.float32)
        vector = np.concatenate([valid.mean(axis=0), valid.std(axis=0), valid.min(axis=0), valid.max(axis=0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), deltas.mean(axis=0), deltas.std(axis=0), np.abs(deltas).mean(axis=0)]).astype(np.float32)
        features.append(vector)
        labels.append(LABEL_TO_ID[str(row['tag'])])
        names.append(name)
    return (np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32), names)

def rgb_validation_probabilities(seed: int, validation_df: pd.DataFrame, pose_model: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Obtiene probabilidades M0 para validaciÃ³n desde su checkpoint."""
    arrays, labels, rows = load_arrays(validation_df, pose_model, 'm0')
    model, _ = build_model('m0', 31, 30)
    checkpoint = RESULTS_DIR / 'm0' / f'seed_{seed}' / 'best.weights.h5'
    if not _resolve_model_artifact(checkpoint).exists():
        raise FileNotFoundError(f'No existe el checkpoint RGB: {checkpoint}')
    model.load_weights(_resolve_model_artifact(checkpoint))
    probabilities = model.predict(inputs_for_arrays(arrays, 'm0'), verbose=0)
    return (probabilities, labels, rows['video_name'].astype(str).tolist())

def rgb_test_probabilities(seed: int, test_names: list[str]) -> np.ndarray:
    path = RESULTS_DIR / 'm0' / f'seed_{seed}' / 'predictions.csv'
    frame = pd.read_csv(_resolve_model_artifact(path)).set_index('video_name')
    columns = [f'prob_{label}' for label in LABELS]
    return np.asarray(frame.loc[test_names][columns], dtype=np.float32)

def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {'test_accuracy': float(accuracy_score(labels, predictions)), 'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(labels, predictions, average='weighted', zero_division=0))}

def run_seed(seed: int, pose_model: str, trees: int, alpha_step: float, fixed_alpha: float | None, output_root: Path) -> dict:
    train_df, validation_df, test_df, protocol = build_partition(seed, require_geometry=True, pose_model=pose_model)
    train_x, train_y, _ = geometry_statistics(train_df, pose_model)
    validation_x, validation_y, validation_names = geometry_statistics(validation_df, pose_model)
    test_x, test_y, test_names = geometry_statistics(test_df, pose_model)
    geometry = RandomForestClassifier(n_estimators=trees, max_features='sqrt', class_weight='balanced', random_state=seed, n_jobs=-1)
    geometry.fit(train_x, train_y)
    geometry_validation = geometry.predict_proba(validation_x)
    geometry_test = geometry.predict_proba(test_x)
    rgb_validation, rgb_validation_y, rgb_validation_names = rgb_validation_probabilities(seed, validation_df, pose_model)
    if validation_names != rgb_validation_names or not np.array_equal(validation_y, rgb_validation_y):
        raise AssertionError('El orden de validaciÃ³n RGB y geometrÃ\xada no coincide')
    rgb_test = rgb_test_probabilities(seed, test_names)
    if fixed_alpha is not None:
        best_alpha = float(fixed_alpha)
        validation_probabilities = (1.0 - best_alpha) * rgb_validation + best_alpha * geometry_validation
        best_validation_f1 = float(f1_score(validation_y, validation_probabilities.argmax(axis=1), average='macro', zero_division=0))
    else:
        best_alpha = 0.0
        best_validation_f1 = -np.inf
        alpha_values = np.arange(0.0, 1.0 + alpha_step * 0.5, alpha_step)
        for alpha in alpha_values:
            validation_probabilities = (1.0 - alpha) * rgb_validation + alpha * geometry_validation
            validation_f1 = f1_score(validation_y, validation_probabilities.argmax(axis=1), average='macro', zero_division=0)
            if validation_f1 > best_validation_f1:
                best_validation_f1 = float(validation_f1)
                best_alpha = float(alpha)
    fused_test = (1.0 - best_alpha) * rgb_test + best_alpha * geometry_test
    output_dir = output_root / f'seed_{seed}'
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = fused_test.argmax(axis=1)
    prediction_frame = pd.DataFrame({'video_name': test_names, 'true_label': [LABELS[index] for index in test_y], 'predicted_label': [LABELS[index] for index in predictions]})
    for index, label in enumerate(LABELS):
        prediction_frame[f'prob_{label}'] = fused_test[:, index]
    prediction_frame.to_csv(output_dir / 'predictions.csv', index=False)
    result = {'model': 'late_fusion_rf', 'seed': int(seed), 'alpha_geometry': best_alpha, 'validation_f1_macro': float(best_validation_f1), 'geometry_features': int(train_x.shape[1]), 'random_forest_trees': int(trees), 'pose_model': pose_model, 'protocol': protocol, **metrics(test_y, fused_test)}
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

def aggregate(results: list[dict], output_root: Path) -> None:
    frame = pd.DataFrame(results)
    frame.to_csv(output_root / 'summary.csv', index=False)
    rows = [{'model': 'late_fusion_rf', 'runs': len(frame)}]
    for metric in ('test_accuracy', 'f1_macro', 'f1_weighted', 'alpha_geometry', 'validation_f1_macro'):
        rows[0][f'{metric}_mean'] = float(frame[metric].mean())
        rows[0][f'{metric}_std'] = float(frame[metric].std(ddof=0))
    pd.DataFrame(rows).to_csv(output_root / 'aggregate_metrics.csv', index=False)

def bootstrap_against_m0(results: list[dict], samples: int, seed: int, output_root: Path) -> None:
    rng = np.random.default_rng(seed)
    deltas = {'test_accuracy': [], 'f1_macro': [], 'f1_weighted': []}
    for result in results:
        run_seed = int(result['seed'])
        candidate = pd.read_csv(_resolve_model_artifact(output_root / f'seed_{run_seed}' / 'predictions.csv'))
        baseline = pd.read_csv(_resolve_model_artifact(RESULTS_DIR / 'm0' / f'seed_{run_seed}' / 'predictions.csv')).set_index('video_name')
        baseline = baseline.loc[candidate['video_name'].tolist()].reset_index()
        for _ in range(samples):
            indexes = rng.integers(0, len(candidate), len(candidate))
            c = candidate.iloc[indexes].reset_index(drop=True)
            b = baseline.iloc[indexes].reset_index(drop=True)
            deltas['test_accuracy'].append(float(np.mean(c['predicted_label'] == c['true_label']) - np.mean(b['predicted_label'] == b['true_label'])))
            deltas['f1_macro'].append(float(f1_score(c['true_label'], c['predicted_label'], average='macro', zero_division=0) - f1_score(b['true_label'], b['predicted_label'], average='macro', zero_division=0)))
            deltas['f1_weighted'].append(float(f1_score(c['true_label'], c['predicted_label'], average='weighted', zero_division=0) - f1_score(b['true_label'], b['predicted_label'], average='weighted', zero_division=0)))
    row = {'model': 'late_fusion_rf', 'baseline': 'm0', 'seeds': ','.join((str(item['seed']) for item in results)), 'bootstrap_samples_per_seed': samples}
    for metric, values in deltas.items():
        values = np.asarray(values, dtype=np.float64)
        row[f'{metric}_delta_mean'] = float(values.mean())
        row[f'{metric}_delta_low'] = float(np.quantile(values, 0.025))
        row[f'{metric}_delta_high'] = float(np.quantile(values, 0.975))
    pd.DataFrame([row]).to_csv(output_root / 'paired_bootstrap.csv', index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--pose-model', default=POSE_MODEL_DEFAULT)
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--alpha-step', type=float, default=0.01)
    parser.add_argument('--fixed-alpha', type=float, default=None, help='Fija el peso geomÃ©trico y evita ajustarlo en validaciÃ³n')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--output-root', default=None, help='Directorio de resultados alternativo')
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = OUTPUT_DIR if args.fixed_alpha is None else RESULTS_DIR / f'late_fusion_fixed_{args.fixed_alpha:.2f}'
    output_root.mkdir(parents=True, exist_ok=True)
    results = [run_seed(seed, args.pose_model, args.trees, args.alpha_step, args.fixed_alpha, output_root) for seed in seeds]
    aggregate(results, output_root)
    bootstrap_against_m0(results, args.bootstrap_samples, seed=20260807, output_root=output_root)
    print(json.dumps(results, indent=2))
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
