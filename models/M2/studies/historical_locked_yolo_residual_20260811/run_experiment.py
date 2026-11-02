"""FusiÃ³n residual YOLO sobre el mejor ResNet-50 + BiGRU histÃ³rico.

El checkpoint histÃ³rico permanece congelado. Se buscan correcciones pequeÃ±as
basadas en geometrÃ­a YOLO de 200 frames usando exclusivamente entrenamiento y
validaciÃ³n. El conjunto de test se materializa una sola vez, despuÃ©s de fijar
la configuraciÃ³n ganadora por macro-F1 de validaciÃ³n con una guarda de
accuracy.
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
import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data' / 'player_geometry_fusion' / 'pose_cache' / 'frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M0.studies.line_recall_training import run_experiment as recall_model
from models.M0.studies.natural_with_human_review import run_experiment as common
from models.M1.config import LABELS, LABEL_TO_ID, GEOMETRY_LENGTH, POSE_CACHE_DIR
from models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite import feature_matrix
CHECKPOINT = REPO / 'data' / 'line_recall_training' / 'full' / 'training' / 'line_weight_1_25' / 'best_accuracy.keras'
REPRODUCTION = REPO / 'data' / 'historical_resnet_bigru_reproduction_20260811'
REPRESENTATIONS = ('E02_quality_mask', 'E04_tracking_smoothing', 'E06_rugby_descriptors', 'E09_player_graph')
CS = (0.01, 0.05, 0.25, 1.0)
CLASS_WEIGHTS = (None, 'balanced')
ALPHAS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
CONFIDENCE_THRESHOLDS = (0.55, 0.65, 0.75, 0.85)
MARGIN_THRESHOLDS = (0.1, 0.2, 0.3, 0.4)
ACCURACY_TOLERANCE = 0.005

def write_json(path: Path, value) -> None:

    def safe(item):
        if isinstance(item, dict):
            return {str(key): safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(child) for child in item]
        if isinstance(item, np.ndarray):
            return [safe(child) for child in item.tolist()]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, ensure_ascii=False), encoding='utf-8')

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _resolve_model_artifact(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def metric_row(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=np.arange(len(LABELS)), zero_division=0)
    return {'accuracy': float(accuracy_score(labels, predictions)), 'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(labels, predictions, average='weighted', zero_division=0)), 'precision_line': float(precision[0]), 'recall_line': float(recall[0]), 'f1_line': float(f1[0]), 'precision_other': float(precision[1]), 'recall_other': float(recall[1]), 'f1_other': float(f1[1]), 'precision_scrum': float(precision[2]), 'recall_scrum': float(recall[2]), 'f1_scrum': float(f1[2]), 'support_line': int(support[0]), 'support_other': int(support[1]), 'support_scrum': int(support[2])}

def paired_bootstrap(y_true: np.ndarray, baseline_probabilities: np.ndarray, selected_probabilities: np.ndarray, seed: int, repeats: int=5000) -> dict:
    rng = np.random.default_rng(seed)
    baseline_predictions = baseline_probabilities.argmax(axis=1)
    selected_predictions = selected_probabilities.argmax(axis=1)
    accuracy_deltas = np.empty(repeats, dtype=np.float64)
    macro_deltas = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        sample_y = y_true[sample]
        baseline_sample = baseline_predictions[sample]
        selected_sample = selected_predictions[sample]
        accuracy_deltas[index] = accuracy_score(sample_y, selected_sample) - accuracy_score(sample_y, baseline_sample)
        macro_deltas[index] = f1_score(sample_y, selected_sample, average='macro', zero_division=0) - f1_score(sample_y, baseline_sample, average='macro', zero_division=0)
    return {'repeats': repeats, 'accuracy_delta_ci95': [float(np.quantile(accuracy_deltas, 0.025)), float(np.quantile(accuracy_deltas, 0.975))], 'macro_f1_delta_ci95': [float(np.quantile(macro_deltas, 0.025)), float(np.quantile(macro_deltas, 0.975))], 'accuracy_probability_positive': float(np.mean(accuracy_deltas > 0)), 'macro_f1_probability_positive': float(np.mean(macro_deltas > 0))}

def normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-07, 1.0)
    return values / values.sum(axis=1, keepdims=True)

def logits(probabilities: np.ndarray) -> np.ndarray:
    values = normalize(probabilities)
    log_values = np.log(values)
    return log_values - log_values.mean(axis=1, keepdims=True)

def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame['tag'].map(LABEL_TO_ID).to_numpy(dtype=np.int32)

def stored_probabilities(path: Path, frame: pd.DataFrame) -> np.ndarray:
    stored = pd.read_csv(_resolve_model_artifact(path))
    columns = [f'prob_{label}' for label in LABELS]
    indexed = stored.set_index('video_name')
    missing = [name for name in frame.video_name.astype(str) if name not in indexed.index]
    if missing:
        raise KeyError(f'Faltan {len(missing)} probabilidades histÃ³ricas; ejemplos: {missing[:3]}')
    return indexed.loc[frame.video_name.astype(str), columns].to_numpy(dtype=np.float64)

def historical_train_probabilities(frame: pd.DataFrame, output_root: Path) -> np.ndarray:
    cache = output_root / 'historical_train_probabilities.npz'
    names = np.asarray(frame.video_name.astype(str).tolist(), dtype=str)
    if _resolve_model_artifact(cache).exists():
        with _load_model_result_array(cache, allow_pickle=True) as data:
            cached_names = data['names'].astype(str)
            probabilities = data['probabilities'].astype(np.float64)
        if np.array_equal(cached_names, names):
            return probabilities
    store = common.load_base_store()
    rows = recall_model.frame_rows(frame, store)
    model = keras.models.load_model(_resolve_model_artifact(CHECKPOINT), compile=False)
    probabilities = recall_model.predict_probabilities(model, recall_model.FeatureBatches(rows, store, False))
    np.savez_compressed(cache, names=names, probabilities=probabilities)
    del model, store
    keras.backend.clear_session()
    gc.collect()
    return probabilities.astype(np.float64)

def cached_features(frame: pd.DataFrame, split: str, representation: str, output_root: Path):
    root = output_root / 'feature_cache' / representation
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{split}.npz'
    names = np.asarray(frame.video_name.astype(str).tolist(), dtype=str)
    if _resolve_model_artifact(path).exists():
        with _load_model_result_array(path, allow_pickle=True) as data:
            cached_names = data['names'].astype(str)
            values = data['features'].astype(np.float32)
        if np.array_equal(cached_names, names):
            return values
    values, feature_labels, loaded_names = feature_matrix(frame, representation)
    if loaded_names != names.tolist() or not np.array_equal(feature_labels, labels(frame)):
        raise AssertionError(f'Orden o etiquetas incompatibles en {representation}/{split}')
    np.savez_compressed(path, names=names, features=values)
    return values

def transform_fit(train: np.ndarray, validation: np.ndarray, seed: int):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train).astype(np.float32)
    validation_scaled = scaler.transform(validation).astype(np.float32)
    components = min(64, train_scaled.shape[1], train_scaled.shape[0] - 1)
    pca = PCA(n_components=components, random_state=seed)
    train_reduced = pca.fit_transform(train_scaled).astype(np.float32)
    validation_reduced = pca.transform(validation_scaled).astype(np.float32)
    return (train_reduced, validation_reduced, scaler, pca)

def apply_gate(base: np.ndarray, geometry: np.ndarray, alpha: float, gate: str, threshold: float) -> np.ndarray:
    if gate == 'none':
        active = np.ones(len(base), dtype=bool)
    elif gate == 'confidence':
        active = base.max(axis=1) < threshold
    elif gate == 'margin':
        ordered = np.sort(base, axis=1)
        active = ordered[:, -1] - ordered[:, -2] < threshold
    else:
        raise ValueError(gate)
    output = base.copy()
    output[active] = (1.0 - alpha) * base[active] + alpha * geometry[active]
    return normalize(output)

def candidate_row(name: str, params: dict, y_validation: np.ndarray, probabilities: np.ndarray, baseline: dict) -> dict:
    row = {'candidate': name, **params, **metric_row(y_validation, probabilities)}
    row['delta_accuracy_vs_historical'] = row['accuracy'] - baseline['accuracy']
    row['delta_f1_macro_vs_historical'] = row['f1_macro'] - baseline['f1_macro']
    row['passes_accuracy_guard'] = row['accuracy'] >= baseline['accuracy'] - ACCURACY_TOLERANCE
    return row

def fit_geometry(train_x, y_train, c: float, class_weight, seed: int):
    model = LogisticRegression(C=c, max_iter=2000, class_weight=class_weight, random_state=seed, solver='lbfgs')
    model.fit(train_x, y_train)
    return model

def search(seed: int, output_root: Path, cache_root: Path):
    train, validation, test = common.load_base_splits(False, 'natural')
    for split, frame in (('train', train), ('validation', validation), ('test', test)):
        frame.to_csv(output_root / f'{split}_manifest.csv', index=False)
    y_train, y_validation, y_test = (labels(train), labels(validation), labels(test))
    cache_root.mkdir(parents=True, exist_ok=True)
    base_train = historical_train_probabilities(train, cache_root)
    base_validation = stored_probabilities(REPRODUCTION / 'checkpoint_validation_predictions.csv', validation)
    base_test = stored_probabilities(REPRODUCTION / 'checkpoint_test_predictions.csv', test)
    baseline_validation = metric_row(y_validation, base_validation)
    baseline_test = metric_row(y_test, base_test)
    candidates = [candidate_row('historical', {'method': 'baseline'}, y_validation, base_validation, baseline_validation)]
    fitted = {}
    for representation in REPRESENTATIONS:
        print(f'\n=== {representation}: cache 200 frames ===', flush=True)
        train_features = cached_features(train, 'train', representation, cache_root)
        validation_features = cached_features(validation, 'validation', representation, cache_root)
        train_x, validation_x, scaler, pca = transform_fit(train_features, validation_features, seed)
        fitted[representation, 'preprocess'] = (scaler, pca)
        for class_weight in CLASS_WEIGHTS:
            weight_name = 'none' if class_weight is None else str(class_weight)
            for c in CS:
                geometry_model = fit_geometry(train_x, y_train, c, class_weight, seed)
                geometry_validation = geometry_model.predict_proba(validation_x)
                fitted[representation, 'geometry', weight_name, c] = geometry_model
                for alpha in ALPHAS:
                    probability = apply_gate(base_validation, geometry_validation, alpha, 'none', 0.0)
                    candidates.append(candidate_row(f'{representation}_blend', {'method': 'blend', 'representation': representation, 'class_weight': weight_name, 'C': c, 'alpha': alpha, 'gate': 'none', 'threshold': 0.0}, y_validation, probability, baseline_validation))
                    for threshold in CONFIDENCE_THRESHOLDS:
                        probability = apply_gate(base_validation, geometry_validation, alpha, 'confidence', threshold)
                        candidates.append(candidate_row(f'{representation}_confidence_gate', {'method': 'gate', 'representation': representation, 'class_weight': weight_name, 'C': c, 'alpha': alpha, 'gate': 'confidence', 'threshold': threshold}, y_validation, probability, baseline_validation))
                    for threshold in MARGIN_THRESHOLDS:
                        probability = apply_gate(base_validation, geometry_validation, alpha, 'margin', threshold)
                        candidates.append(candidate_row(f'{representation}_margin_gate', {'method': 'gate', 'representation': representation, 'class_weight': weight_name, 'C': c, 'alpha': alpha, 'gate': 'margin', 'threshold': threshold}, y_validation, probability, baseline_validation))
                stack_train = np.concatenate([logits(base_train), train_x], axis=1)
                stack_validation = np.concatenate([logits(base_validation), validation_x], axis=1)
                stack_model = fit_geometry(stack_train, y_train, c, class_weight, seed)
                stack_probability = stack_model.predict_proba(stack_validation)
                fitted[representation, 'stack', weight_name, c] = stack_model
                candidates.append(candidate_row(f'{representation}_stack', {'method': 'stack', 'representation': representation, 'class_weight': weight_name, 'C': c, 'alpha': 0.0, 'gate': 'none', 'threshold': 0.0}, y_validation, stack_probability, baseline_validation))
        del train_features, validation_features
        gc.collect()
    for line_scale in np.arange(0.9, 1.51, 0.05):
        for scrum_scale in np.arange(0.9, 1.41, 0.05):
            probability = base_validation.copy()
            probability[:, LABEL_TO_ID['line']] *= line_scale
            probability[:, LABEL_TO_ID['scrum']] *= scrum_scale
            probability = normalize(probability)
            candidates.append(candidate_row('historical_calibrated', {'method': 'calibration', 'representation': 'none', 'class_weight': 'none', 'C': 0.0, 'alpha': 0.0, 'gate': 'none', 'threshold': 0.0, 'line_scale': float(line_scale), 'scrum_scale': float(scrum_scale)}, y_validation, probability, baseline_validation))
    ranking = pd.DataFrame(candidates).sort_values(['passes_accuracy_guard', 'f1_macro', 'accuracy'], ascending=[False, False, False])
    ranking.to_csv(output_root / 'validation_candidates.csv', index=False)
    eligible = ranking[ranking.passes_accuracy_guard & (ranking.f1_macro > baseline_validation['f1_macro'] + 1e-12)]
    selected = (eligible.iloc[0] if len(eligible) else ranking[ranking.candidate == 'historical'].iloc[0]).to_dict()
    if selected['method'] == 'baseline':
        selected_test_probability = base_test
    elif selected['method'] == 'calibration':
        selected_test_probability = base_test.copy()
        selected_test_probability[:, LABEL_TO_ID['line']] *= float(selected['line_scale'])
        selected_test_probability[:, LABEL_TO_ID['scrum']] *= float(selected['scrum_scale'])
        selected_test_probability = normalize(selected_test_probability)
    else:
        representation = str(selected['representation'])
        test_features = cached_features(test, 'test', representation, cache_root)
        scaler, pca = fitted[representation, 'preprocess']
        test_x = pca.transform(scaler.transform(test_features)).astype(np.float32)
        weight_name = str(selected['class_weight'])
        c = float(selected['C'])
        if selected['method'] == 'stack':
            model = fitted[representation, 'stack', weight_name, c]
            selected_test_probability = model.predict_proba(np.concatenate([logits(base_test), test_x], axis=1))
        else:
            model = fitted[representation, 'geometry', weight_name, c]
            geometry_test = model.predict_proba(test_x)
            selected_test_probability = apply_gate(base_test, geometry_test, float(selected['alpha']), str(selected['gate']), float(selected['threshold']))
        artifact = {'scaler': scaler, 'pca': pca, 'model': model, 'configuration': selected}
        joblib.dump(artifact, output_root / 'selected_model.joblib', compress=3)
    selected_test = metric_row(y_test, selected_test_probability)
    comparison = {'historical_validation': baseline_validation, 'selected_validation': {key: selected[key] for key in metric_row(y_validation, base_validation).keys() if key in selected}, 'historical_test': baseline_test, 'selected_test': selected_test, 'delta_test_accuracy': selected_test['accuracy'] - baseline_test['accuracy'], 'delta_test_f1_macro': selected_test['f1_macro'] - baseline_test['f1_macro'], 'selected_configuration': selected, 'paired_bootstrap': paired_bootstrap(y_test, base_test, selected_test_probability, seed)}
    write_json(output_root / 'selected_result.json', comparison)
    predictions = pd.DataFrame({'video_name': test.video_name.astype(str), 'true_label': test.tag.astype(str), 'historical_prediction': [LABELS[index] for index in base_test.argmax(axis=1)], 'selected_prediction': [LABELS[index] for index in selected_test_probability.argmax(axis=1)]})
    for index, label in enumerate(LABELS):
        predictions[f'historical_prob_{label}'] = base_test[:, index]
        predictions[f'selected_prob_{label}'] = selected_test_probability[:, index]
    predictions.to_csv(output_root / 'selected_test_predictions.csv', index=False)
    write_json(output_root / 'protocol.json', {'seed': seed, 'geometry_length': GEOMETRY_LENGTH, 'pose_cache_dir': POSE_CACHE_DIR, 'representations': REPRESENTATIONS, 'checkpoint': CHECKPOINT, 'checkpoint_sha256': sha256(CHECKPOINT), 'split_sizes': {'train': len(train), 'validation': len(validation), 'test': len(test)}, 'selection': 'maximum validation macro-F1 with validation accuracy no more than 0.005 below historical; test evaluated once after selection', 'candidate_count': len(ranking), 'shared_cache_root': cache_root, 'software': {'python': platform.python_version(), 'keras': keras.__version__, 'numpy': np.__version__}})
    print(json.dumps(comparison, indent=2, ensure_ascii=False), flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', default=str(HERE / 'results_seed42'))
    parser.add_argument('--cache-root', default=str(HERE / 'results_seed42'))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    search(args.seed, output_root, Path(args.cache_root))
    write_json(output_root / 'runtime.json', {'seconds': time.perf_counter() - started})
if __name__ == '__main__':
    main()
