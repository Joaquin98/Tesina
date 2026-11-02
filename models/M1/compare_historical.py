"""Compara el ensamble actual contra el mejor ResNet-50 + BiGRU historico.

La comparacion usa los mismos 817 videos de prueba y conserva cada semilla del
ensamble como una corrida independiente. Ademas de las metricas usuales,
calcula McNemar, bootstrap pareado, calidad de probabilidades y errores
compartidos.
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
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from models.M1.config import LABELS, RESULTS_DIR
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
OLD_DEFAULT = RESULTS_DIR.parents[2] / 'data' / 'line_false_positive_review_finetune_1800' / 'full' / 'starting_test_predictions.csv'
NEW_DEFAULT = RESULTS_DIR / 'ensemble_all_residual_pose200_calibrated_fixed'
OUTPUT_DEFAULT = RESULTS_DIR / 'historical_comparison_ensemble_resnet_rnn'

def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve_model_artifact(path))
    required = {'video_name', 'true_label', 'predicted_label', *[f'prob_{label}' for label in LABELS]}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Faltan columnas en {path}: {sorted(missing)}')
    if frame['video_name'].duplicated().any():
        raise ValueError(f'Hay videos duplicados en {path}')
    for column in ['true_label', 'predicted_label', *[f'prob_{label}' for label in LABELS]]:
        if frame[column].isna().any():
            raise ValueError(f'Hay valores nulos en {column} de {path}')
    frame = frame.copy()
    frame['true_id'] = frame['true_label'].map(LABEL_TO_ID)
    frame['pred_id'] = frame['predicted_label'].map(LABEL_TO_ID)
    if frame[['true_id', 'pred_id']].isna().any().any():
        raise ValueError(f'Etiqueta desconocida en {path}')
    probabilities = frame[[f'prob_{label}' for label in LABELS]].to_numpy(dtype=np.float64)
    if np.any(probabilities < -1e-06) or np.any(~np.isfinite(probabilities)):
        raise ValueError(f'Probabilidades invalidas en {path}')
    totals = probabilities.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError(f'Probabilidades con suma no positiva en {path}')
    frame[[f'prob_{label}' for label in LABELS]] = probabilities / totals[:, None]
    return frame.reset_index(drop=True)

def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int=10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    bucket = np.minimum((confidence * bins).astype(int), bins - 1)
    error = 0.0
    for index in range(bins):
        selected = bucket == index
        if not np.any(selected):
            continue
        accuracy = float(np.mean(predictions[selected] == labels[selected]))
        mean_confidence = float(np.mean(confidence[selected]))
        error += float(np.mean(selected)) * abs(accuracy - mean_confidence)
    return error

def metric_values(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    one_hot = np.eye(len(LABELS), dtype=np.float64)[labels]
    return {'accuracy': float(accuracy_score(labels, predictions)), 'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(labels, predictions, average='weighted', zero_division=0)), 'log_loss': float(log_loss(labels, probabilities, labels=list(range(len(LABELS))))), 'brier_multiclass': float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))), 'ece_10bins': float(expected_calibration_error(labels, probabilities))}

def per_class_values(labels: np.ndarray, predictions: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=list(range(len(LABELS))), zero_division=0)
    return pd.DataFrame({'class': LABELS, 'precision': precision, 'recall': recall, 'f1': f1, 'support': support})

def exact_mcnemar(old_correct: np.ndarray, new_correct: np.ndarray) -> dict[str, float | int]:
    old_only = int(np.sum(old_correct & ~new_correct))
    new_only = int(np.sum(~old_correct & new_correct))
    discordant = old_only + new_only
    if discordant == 0:
        p_value = 1.0
    else:
        upper = max(old_only, new_only)
        tail = sum((math.comb(discordant, index) for index in range(upper, discordant + 1))) / 2.0 ** discordant
        p_value = min(1.0, 2.0 * tail)
    return {'old_only_correct': old_only, 'new_only_correct': new_only, 'discordant': discordant, 'mcnemar_exact_p': float(p_value)}

def bootstrap_deltas(labels: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray, old_probabilities: np.ndarray, new_probabilities: np.ndarray, samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    size = len(labels)
    for _ in range(samples):
        indices = rng.integers(0, size, size)
        old_metrics = metric_values(labels[indices], old_predictions[indices], old_probabilities[indices])
        new_metrics = metric_values(labels[indices], new_predictions[indices], new_probabilities[indices])
        rows.append({metric: new_metrics[metric] - old_metrics[metric] for metric in old_metrics})
    return pd.DataFrame(rows)

def paired_error_tables(labels: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_correct = old_predictions == labels
    new_correct = new_predictions == labels
    breakdown = pd.DataFrame([{'case': 'ambos_correctos', 'count': int(np.sum(old_correct & new_correct))}, {'case': 'solo_historico_correcto', 'count': int(np.sum(old_correct & ~new_correct))}, {'case': 'solo_ensamble_correcto', 'count': int(np.sum(~old_correct & new_correct))}, {'case': 'ambos_incorrectos', 'count': int(np.sum(~old_correct & ~new_correct))}])
    rows = []
    for label_id, label in enumerate(LABELS):
        selected = labels == label_id
        rows.append({'class': label, 'support': int(np.sum(selected)), 'historical_correct': int(np.sum(old_correct & selected)), 'ensemble_correct': int(np.sum(new_correct & selected)), 'delta_correct': int(np.sum(new_correct & selected) - np.sum(old_correct & selected))})
    return (breakdown, pd.DataFrame(rows))

def merge_runs(old: pd.DataFrame, new_path: Path) -> pd.DataFrame:
    new = load_predictions(new_path)
    keys = ['video_name', 'true_label']
    merged = old[keys + ['true_id', 'pred_id', *[f'prob_{label}' for label in LABELS]]].merge(new[keys + ['pred_id', *[f'prob_{label}' for label in LABELS]]], on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    if len(merged) != len(old) or len(merged) != len(new):
        raise ValueError(f'Los conjuntos no coinciden: historico={len(old)}, nuevo={len(new)}')
    return merged

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-predictions', default=str(OLD_DEFAULT))
    parser.add_argument('--new-root', default=str(NEW_DEFAULT))
    parser.add_argument('--output-root', default=str(OUTPUT_DEFAULT))
    parser.add_argument('--seeds', default='17,42,2026')
    parser.add_argument('--bootstrap-samples', type=int, default=10000)
    args = parser.parse_args()
    old_path = Path(args.old_predictions)
    new_root = Path(args.new_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    old = load_predictions(old_path)
    seed_values = [int(item.strip()) for item in args.seeds.split(',') if item.strip()]
    labels = old['true_id'].to_numpy(dtype=np.int32)
    old_predictions = old['pred_id'].to_numpy(dtype=np.int32)
    old_probabilities = old[[f'prob_{label}' for label in LABELS]].to_numpy(dtype=np.float64)
    old_metrics = metric_values(labels, old_predictions, old_probabilities)
    old_per_class = per_class_values(labels, old_predictions)
    old_per_class.insert(0, 'model', 'historical_resnet_rnn')
    old_per_class.to_csv(output_root / 'historical_per_class.csv', index=False)
    pd.DataFrame([{'model': 'historical_resnet_rnn', **old_metrics}]).to_csv(output_root / 'historical_metrics.csv', index=False)
    np.savetxt(output_root / 'historical_confusion_matrix.csv', confusion_matrix(labels, old_predictions, labels=list(range(len(LABELS)))), fmt='%d', delimiter=',')
    metric_rows: list[dict[str, float | int | str]] = []
    per_class_rows: list[pd.DataFrame] = []
    paired_rows: list[dict[str, float | int]] = []
    bootstrap_rows: list[dict[str, float | int | str]] = []
    aggregate_error_rows: list[pd.DataFrame] = []
    for seed in seed_values:
        new_path = new_root / f'seed_{seed}' / 'predictions.csv'
        merged = merge_runs(old, new_path)
        new_predictions = merged['pred_id_new'].to_numpy(dtype=np.int32)
        new_probabilities = merged[[f'prob_{label}_new' for label in LABELS]].to_numpy(dtype=np.float64)
        metrics = metric_values(labels, new_predictions, new_probabilities)
        metric_rows.append({'model': 'ensemble_actual', 'seed': seed, **metrics})
        new_per_class = per_class_values(labels, new_predictions)
        new_per_class.insert(0, 'model', 'ensemble_actual')
        new_per_class.insert(1, 'seed', seed)
        per_class_rows.append(new_per_class)
        old_correct = old_predictions == labels
        new_correct = new_predictions == labels
        mc = exact_mcnemar(old_correct, new_correct)
        mc.update({'seed': seed, 'old_accuracy': float(np.mean(old_correct)), 'new_accuracy': float(np.mean(new_correct))})
        paired_rows.append(mc)
        deltas = bootstrap_deltas(labels, old_predictions, new_predictions, old_probabilities, new_probabilities, args.bootstrap_samples, 20260811 + seed)
        for metric in deltas.columns:
            values = deltas[metric].to_numpy(dtype=np.float64)
            bootstrap_rows.append({'seed': seed, 'metric': metric, 'delta_mean_new_minus_old': float(values.mean()), 'ci95_low': float(np.quantile(values, 0.025)), 'ci95_high': float(np.quantile(values, 0.975)), 'bootstrap_probability_delta_le_zero': float((np.sum(values <= 0.0) + 1) / (len(values) + 1))})
        breakdown, by_class = paired_error_tables(labels, old_predictions, new_predictions)
        breakdown.insert(0, 'seed', seed)
        by_class.insert(0, 'seed', seed)
        aggregate_error_rows.append(pd.concat([breakdown.assign(table='overall'), by_class.assign(table='by_class')], ignore_index=True, sort=False))
        np.savetxt(output_root / f'ensemble_confusion_matrix_seed_{seed}.csv', confusion_matrix(labels, new_predictions, labels=list(range(len(LABELS)))), fmt='%d', delimiter=',')
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(output_root / 'metrics_by_seed.csv', index=False)
    pd.concat(per_class_rows, ignore_index=True).to_csv(output_root / 'per_class_by_seed.csv', index=False)
    pd.DataFrame(paired_rows).to_csv(output_root / 'mcnemar_by_seed.csv', index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output_root / 'bootstrap_deltas_by_seed.csv', index=False)
    pd.concat(aggregate_error_rows, ignore_index=True).to_csv(output_root / 'paired_errors_by_seed.csv', index=False)
    aggregate_rows = []
    historical_row: dict[str, float | int | str] = {'model': 'historical_resnet_rnn', 'runs': 1}
    ensemble_row: dict[str, float | int | str] = {'model': 'ensemble_actual', 'runs': len(metric_frame)}
    for metric in old_metrics:
        values = metric_frame[metric].to_numpy(dtype=np.float64)
        historical_row[f'{metric}_mean'] = float(old_metrics[metric])
        historical_row[f'{metric}_std'] = 0.0
        ensemble_row[f'{metric}_mean'] = float(values.mean())
        ensemble_row[f'{metric}_std'] = float(values.std(ddof=0))
    aggregate_rows.extend([historical_row, ensemble_row])
    pd.DataFrame(aggregate_rows).to_csv(output_root / 'metrics_aggregate.csv', index=False)
    summary = {'old_predictions': str(old_path), 'new_root': str(new_root), 'videos': int(len(old)), 'class_counts': {label: int(np.sum(labels == index)) for index, label in enumerate(LABELS)}, 'seeds': seed_values, 'bootstrap_samples_per_seed': int(args.bootstrap_samples), 'historical_metrics': old_metrics, 'ensemble_metrics_mean': {metric: float(metric_frame[metric].mean()) for metric in old_metrics}, 'ensemble_metrics_std': {metric: float(metric_frame[metric].std(ddof=0)) for metric in old_metrics}}
    (output_root / 'audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Resultados: {output_root}')
if __name__ == '__main__':
    main()
