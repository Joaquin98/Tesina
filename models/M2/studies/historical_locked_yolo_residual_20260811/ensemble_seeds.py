"""Ensamble de las correcciones residuales seleccionadas en tres semillas."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data' / 'player_geometry_fusion' / 'pose_cache' / 'frames_200')
if str(HERE) not in sys.path:
    sys.path.insert(0, str(source_directory(HERE)))
from models.M2.studies.historical_locked_yolo_residual_20260811.run_experiment import ACCURACY_TOLERANCE, LABELS, REPRODUCTION, apply_gate, labels, metric_row, paired_bootstrap, stored_probabilities, write_json
SEEDS = (17, 42, 2026)
SHARED_CACHE = HERE / 'results_seed42' / 'feature_cache'
OUTPUT = HERE / 'ensemble_three_seeds'

def load_features(representation: str, split: str) -> np.ndarray:
    with _load_model_result_array(SHARED_CACHE / representation / f'{split}.npz', allow_pickle=True) as data:
        return data['features'].astype(np.float32)

def probabilities_for_seed(seed: int, split: str, base: np.ndarray) -> tuple[np.ndarray, dict]:
    root = HERE / f'results_seed{seed}'
    result = json.loads(_resolve_model_artifact(root / 'selected_result.json').read_text(encoding='utf-8'))
    configuration = result['selected_configuration']
    artifact = joblib.load(_resolve_model_artifact(root / 'selected_model.joblib'))
    features = load_features(configuration['representation'], split)
    transformed = artifact['pca'].transform(artifact['scaler'].transform(features)).astype(np.float32)
    geometry = artifact['model'].predict_proba(transformed)
    probabilities = apply_gate(base, geometry, float(configuration['alpha']), str(configuration['gate']), float(configuration['threshold']))
    return (probabilities, configuration)

def weight_grid(step: float=0.05):
    units = int(round(1.0 / step))
    for first in range(units + 1):
        for second in range(units - first + 1):
            third = units - first - second
            yield (np.asarray([first, second, third], dtype=np.float64) / units)

def combine(probabilities: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    output = sum((weight * probability for weight, probability in zip(weights, probabilities)))
    return output / output.sum(axis=1, keepdims=True)

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(_resolve_model_artifact(HERE / 'results_seed42' / 'validation_manifest.csv'))
    test = pd.read_csv(_resolve_model_artifact(HERE / 'results_seed42' / 'test_manifest.csv'))
    y_validation, y_test = (labels(validation), labels(test))
    base_validation = stored_probabilities(REPRODUCTION / 'checkpoint_validation_predictions.csv', validation)
    base_test = stored_probabilities(REPRODUCTION / 'checkpoint_test_predictions.csv', test)
    baseline_validation = metric_row(y_validation, base_validation)
    baseline_test = metric_row(y_test, base_test)
    validation_probabilities, test_probabilities, configurations = ([], [], [])
    rows = []
    for seed in SEEDS:
        validation_probability, configuration = probabilities_for_seed(seed, 'validation', base_validation)
        test_probability, _ = probabilities_for_seed(seed, 'test', base_test)
        validation_probabilities.append(validation_probability)
        test_probabilities.append(test_probability)
        configurations.append(configuration)
        rows.append({'seed': seed, 'split': 'validation', **metric_row(y_validation, validation_probability)})
        rows.append({'seed': seed, 'split': 'test', **metric_row(y_test, test_probability)})
    pd.DataFrame(rows).to_csv(OUTPUT / 'seed_metrics.csv', index=False)
    candidates = []
    for weights in weight_grid():
        probability = combine(validation_probabilities, weights)
        metrics = metric_row(y_validation, probability)
        candidates.append({'weight_seed17': float(weights[0]), 'weight_seed42': float(weights[1]), 'weight_seed2026': float(weights[2]), **metrics, 'passes_accuracy_guard': metrics['accuracy'] >= baseline_validation['accuracy'] - ACCURACY_TOLERANCE})
    ranking = pd.DataFrame(candidates).sort_values(['passes_accuracy_guard', 'f1_macro', 'accuracy'], ascending=[False, False, False])
    ranking.to_csv(OUTPUT / 'validation_weight_search.csv', index=False)
    selected = ranking.iloc[0].to_dict()
    weights = np.asarray([selected['weight_seed17'], selected['weight_seed42'], selected['weight_seed2026']])
    selected_validation_probability = combine(validation_probabilities, weights)
    selected_test_probability = combine(test_probabilities, weights)
    selected_validation = metric_row(y_validation, selected_validation_probability)
    selected_test = metric_row(y_test, selected_test_probability)
    baseline_prediction = base_test.argmax(axis=1)
    ensemble_prediction = selected_test_probability.argmax(axis=1)
    baseline_correct = baseline_prediction == y_test
    ensemble_correct = ensemble_prediction == y_test
    change_rows = []
    for class_index, label in enumerate(LABELS):
        mask = y_test == class_index
        change_rows.append({'class': label, 'support': int(mask.sum()), 'historical_wrong_ensemble_correct': int(np.sum(mask & ~baseline_correct & ensemble_correct)), 'historical_correct_ensemble_wrong': int(np.sum(mask & baseline_correct & ~ensemble_correct)), 'net_correct': int(np.sum(mask & ~baseline_correct & ensemble_correct) - np.sum(mask & baseline_correct & ~ensemble_correct))})
    pd.DataFrame(change_rows).to_csv(OUTPUT / 'prediction_change_summary.csv', index=False)
    prediction_frame = pd.DataFrame({'video_name': test.video_name.astype(str), 'true_label': test.tag.astype(str), 'historical_prediction': [LABELS[index] for index in baseline_prediction], 'ensemble_prediction': [LABELS[index] for index in ensemble_prediction], 'historical_correct': baseline_correct, 'ensemble_correct': ensemble_correct})
    for index, label in enumerate(LABELS):
        prediction_frame[f'historical_prob_{label}'] = base_test[:, index]
        prediction_frame[f'ensemble_prob_{label}'] = selected_test_probability[:, index]
    prediction_frame.to_csv(OUTPUT / 'test_predictions.csv', index=False)
    result = {'seeds': SEEDS, 'seed_configurations': configurations, 'selected_weights_by_validation': weights, 'historical_validation': baseline_validation, 'ensemble_validation': selected_validation, 'historical_test': baseline_test, 'ensemble_test': selected_test, 'delta_test_accuracy': selected_test['accuracy'] - baseline_test['accuracy'], 'delta_test_f1_macro': selected_test['f1_macro'] - baseline_test['f1_macro'], 'paired_bootstrap': paired_bootstrap(y_test, base_test, selected_test_probability, seed=117, repeats=10000), 'selection': 'weights selected only by validation macro-F1 with accuracy guard'}
    write_json(OUTPUT / 'ensemble_result.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=lambda value: value.tolist()), flush=True)
if __name__ == '__main__':
    main()
