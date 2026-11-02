"""Isolated search for higher line recall with controlled global degradation.

Reads the fixed natural train/validation/test splits and existing feature stores.
Writes only below data/line_recall_training.
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
import json
import sys
import time
from pathlib import Path
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
HERE = experiment_directory(__file__)
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M0.studies.natural_with_human_review import run_experiment as common
REFERENCE_RUN = REPO / 'data' / 'class_imbalance_comparison' / 'full' / 'runs' / 'natural_unweighted'
BALANCED_RUN = REPO / 'data' / 'class_imbalance_comparison' / 'full' / 'runs' / 'balanced_reference'
SEED = common.SEED
LABELS = common.LABELS
LABEL_TO_ID = common.LABEL_TO_ID
BATCH_SIZE = common.BATCH_SIZE
MAX_EPOCHS = common.MAX_EPOCHS
BOOTSTRAP_REPLICATES = 2000
MIN_LINE_PRECISION = 0.72
MAX_GLOBAL_DROP = 0.015
TRAINING_CONFIGS = {'macro_unweighted': {'line_weight': 1.0, 'line_oversample': 1.0}, 'line_weight_1_25': {'line_weight': 1.25, 'line_oversample': 1.0}, 'line_weight_1_50': {'line_weight': 1.5, 'line_oversample': 1.0}, 'line_weight_2_00': {'line_weight': 2.0, 'line_oversample': 1.0}, 'line_oversample_1_50': {'line_weight': 1.0, 'line_oversample': 1.5}}
SELECTIONS = ('best_accuracy', 'best_macro', 'best_guarded_recall')

def json_dump(path: Path, value) -> None:

    def json_safe(item):
        if isinstance(item, dict):
            return {key: json_safe(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_safe(child) for child in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and (not np.isfinite(item)):
            return None
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def frame_rows(frame: pd.DataFrame, store: common.FeatureStore) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        if row.video_name not in store.index:
            raise KeyError(f'Missing feature: {row.video_name}')
        rows.append({'video_name': row.video_name, 'tag': row.tag, 'y': LABEL_TO_ID[row.tag], 'feature_index': store.index[row.video_name], 'is_repeat': False})
    return pd.DataFrame(rows)

def oversample_line(rows: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    if multiplier == 1.0:
        return rows.copy().reset_index(drop=True)
    line = rows[rows.y == LABEL_TO_ID['line']]
    extra_count = int(round((multiplier - 1.0) * len(line)))
    extra = line.sample(n=extra_count, replace=extra_count > len(line), random_state=SEED).copy()
    extra['is_repeat'] = True
    return pd.concat([rows, extra], ignore_index=True)

class FeatureBatches(keras.utils.PyDataset):

    def __init__(self, rows: pd.DataFrame, store: common.FeatureStore, shuffle: bool, line_weight: float=1.0, include_weights: bool=False):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
        self.rows = rows.reset_index(drop=True)
        self.store = store
        self.shuffle = shuffle
        self.line_weight = float(line_weight)
        self.include_weights = include_weights
        self.epoch = 0
        self.schedule = self._make_schedule()

    def _make_schedule(self) -> np.ndarray:
        ids = np.arange(len(self.rows))
        if self.shuffle:
            np.random.default_rng(SEED + self.epoch).shuffle(ids)
        return ids

    def __len__(self) -> int:
        return int(np.ceil(len(self.rows) / BATCH_SIZE))

    def __getitem__(self, index: int):
        ids = self.schedule[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
        batch = self.rows.iloc[ids]
        source_ids = batch.feature_index.to_numpy(dtype=np.int64)
        features = np.asarray(self.store.features[source_ids], dtype=np.float32)
        masks = np.asarray(self.store.masks[source_ids], dtype=bool)
        y = batch.y.to_numpy(dtype=np.int32)
        if not self.include_weights:
            return ((features, masks), y)
        weights = np.ones(len(y), dtype=np.float32)
        weights[y == LABEL_TO_ID['line']] = self.line_weight
        return ((features, masks), y, weights)

    def on_epoch_end(self):
        self.epoch += 1
        self.schedule = self._make_schedule()

def predict_probabilities(model: keras.Model, data: FeatureBatches) -> np.ndarray:
    return np.asarray(model.predict(data, verbose=0), dtype=np.float32)

def metrics_from_probabilities(y_true: np.ndarray, probabilities: np.ndarray, line_alpha: float=1.0) -> dict:
    adjusted = probabilities.copy()
    adjusted[:, LABEL_TO_ID['line']] *= float(line_alpha)
    predicted = np.argmax(adjusted, axis=1)
    metrics = common.metric_bundle(y_true, predicted)
    metrics['line_alpha'] = float(line_alpha)
    return metrics

class ValidationSelector(keras.callbacks.Callback):

    def __init__(self, validation_data: FeatureBatches, y_validation: np.ndarray, run_dir: Path, guards: dict):
        super().__init__()
        self.validation_data = validation_data
        self.y_validation = y_validation
        self.run_dir = run_dir
        self.guards = guards
        self.rows = []
        self.best_accuracy = -np.inf
        self.best_macro = -np.inf
        self.best_guarded = (-np.inf, -np.inf)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probabilities = predict_probabilities(self.model, self.validation_data)
        metrics = metrics_from_probabilities(self.y_validation, probabilities)
        line = metrics['per_class']['line']
        logs['val_macro_f1'] = metrics['f1_macro']
        logs['val_line_precision'] = line['precision']
        logs['val_line_recall'] = line['recall']
        logs['val_line_f1'] = line['f1']
        row = {'epoch': epoch + 1, 'val_accuracy_exact': metrics['accuracy'], 'val_macro_f1': metrics['f1_macro'], 'val_weighted_f1': metrics['f1_weighted'], 'val_line_precision': line['precision'], 'val_line_recall': line['recall'], 'val_line_f1': line['f1'], 'val_other_recall': metrics['per_class']['other']['recall'], 'val_scrum_recall': metrics['per_class']['scrum']['recall']}
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.run_dir / 'validation_by_epoch.csv', index=False)
        if metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = metrics['accuracy']
            self.model.save(self.run_dir / 'best_accuracy.keras', overwrite=True)
        if metrics['f1_macro'] > self.best_macro:
            self.best_macro = metrics['f1_macro']
            self.model.save(self.run_dir / 'best_macro.keras', overwrite=True)
        eligible = line['precision'] >= self.guards['min_line_precision'] and metrics['f1_macro'] >= self.guards['min_macro_f1'] and (metrics['accuracy'] >= self.guards['min_accuracy'])
        guarded_key = (line['recall'], metrics['f1_macro'])
        if eligible and guarded_key > self.best_guarded:
            self.best_guarded = guarded_key
            self.model.save(self.run_dir / 'best_guarded_recall.keras', overwrite=True)
        print(f"val exact: acc={metrics['accuracy']:.4f} macro={metrics['f1_macro']:.4f} line_p={line['precision']:.4f} line_r={line['recall']:.4f} eligible={eligible}", flush=True)

def save_predictions(path: Path, rows: pd.DataFrame, probabilities: np.ndarray, line_alpha: float=1.0) -> tuple[dict, np.ndarray]:
    adjusted = probabilities.copy()
    adjusted[:, 0] *= float(line_alpha)
    predicted = np.argmax(adjusted, axis=1)
    y_true = rows.y.to_numpy(dtype=np.int32)
    metrics = common.metric_bundle(y_true, predicted)
    metrics['line_alpha'] = float(line_alpha)
    output = pd.DataFrame({'video_name': rows.video_name, 'true_label': [LABELS[i] for i in y_true], 'predicted_label': [LABELS[i] for i in predicted], 'correct': y_true == predicted})
    for index, label in enumerate(LABELS):
        output[f'prob_{label}'] = probabilities[:, index]
    output.to_csv(path, index=False)
    return (metrics, predicted)

def reference_probabilities(store: common.FeatureStore, validation_rows: pd.DataFrame, test_rows: pd.DataFrame, output_root: Path) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    cache = output_root / 'reference_probabilities.npz'
    if _resolve_model_artifact(cache).exists():
        with _load_model_result_array(cache) as data:
            val_prob = np.asarray(data['validation'], dtype=np.float32)
            test_prob = np.asarray(data['test'], dtype=np.float32)
    else:
        model = keras.models.load_model(_resolve_model_artifact(REFERENCE_RUN / 'best_model.keras'), compile=False)
        val_data = FeatureBatches(validation_rows, store, False)
        test_data = FeatureBatches(test_rows, store, False)
        val_prob = predict_probabilities(model, val_data)
        test_prob = predict_probabilities(model, test_data)
        np.savez(cache, validation=val_prob, test=test_prob)
        del model
        keras.backend.clear_session()
    val_metrics = metrics_from_probabilities(validation_rows.y.to_numpy(), val_prob)
    test_metrics = metrics_from_probabilities(test_rows.y.to_numpy(), test_prob)
    return (val_prob, test_prob, val_metrics, test_metrics)

def guard_thresholds(reference_validation: dict, reference_test: dict) -> tuple[dict, dict]:
    validation = {'min_line_precision': MIN_LINE_PRECISION, 'min_macro_f1': reference_validation['f1_macro'] - MAX_GLOBAL_DROP, 'min_accuracy': reference_validation['accuracy'] - MAX_GLOBAL_DROP}
    test = {'min_line_precision': MIN_LINE_PRECISION, 'min_macro_f1': reference_test['f1_macro'] - MAX_GLOBAL_DROP, 'min_accuracy': reference_test['accuracy'] - MAX_GLOBAL_DROP}
    return (validation, test)

def run_calibrations(output_root: Path, validation_rows: pd.DataFrame, test_rows: pd.DataFrame, val_prob: np.ndarray, test_prob: np.ndarray, validation_guards: dict, test_guards: dict, quick: bool) -> list[dict]:
    y_val = validation_rows.y.to_numpy(dtype=np.int32)
    grid = np.linspace(0.5, 6.0, 111 if quick else 551)
    candidates = [(float(alpha), metrics_from_probabilities(y_val, val_prob, float(alpha))) for alpha in grid]
    macro_choice = max(candidates, key=lambda item: (item[1]['f1_macro'], -abs(item[0] - 1.0)))
    recall80_pool = [item for item in candidates if item[1]['per_class']['line']['recall'] >= 0.8]
    guarded_pool = [item for item in candidates if item[1]['per_class']['line']['precision'] >= validation_guards['min_line_precision'] and item[1]['f1_macro'] >= validation_guards['min_macro_f1'] and (item[1]['accuracy'] >= validation_guards['min_accuracy'])]
    choices = {'calibration_macro': macro_choice}
    if recall80_pool:
        choices['calibration_recall_080'] = max(recall80_pool, key=lambda item: item[1]['f1_macro'])
    if guarded_pool:
        choices['calibration_guarded_max_recall'] = max(guarded_pool, key=lambda item: (item[1]['per_class']['line']['recall'], item[1]['f1_macro']))
    rows = []
    for name, (alpha, val_metrics) in choices.items():
        run_dir = output_root / 'calibrations' / name
        run_dir.mkdir(parents=True, exist_ok=True)
        test_metrics, _ = save_predictions(run_dir / 'predictions.csv', test_rows, test_prob, alpha)
        json_dump(run_dir / 'validation_metrics.json', val_metrics)
        json_dump(run_dir / 'test_metrics.json', test_metrics)
        rows.append(result_row(name, 'calibration', alpha, val_metrics, test_metrics, 0, 0.0, test_guards))
    return rows

def result_row(name: str, method: str, alpha: float, validation_metrics: dict, test_metrics: dict, epochs: int, seconds: float, test_guards: dict, **extra) -> dict:
    line = test_metrics['per_class']['line']
    val_line = validation_metrics['per_class']['line']
    passes = line['precision'] >= test_guards['min_line_precision'] and test_metrics['f1_macro'] >= test_guards['min_macro_f1'] and (test_metrics['accuracy'] >= test_guards['min_accuracy'])
    return {'experiment': name, 'method': method, 'line_alpha': alpha, 'epochs_ran': epochs, 'training_time_seconds': seconds, 'val_accuracy': validation_metrics['accuracy'], 'val_f1_macro': validation_metrics['f1_macro'], 'val_line_precision': val_line['precision'], 'val_line_recall': val_line['recall'], 'accuracy': test_metrics['accuracy'], 'f1_macro': test_metrics['f1_macro'], 'f1_weighted': test_metrics['f1_weighted'], 'precision_line': line['precision'], 'recall_line': line['recall'], 'f1_line': line['f1'], 'recall_other': test_metrics['per_class']['other']['recall'], 'recall_scrum': test_metrics['per_class']['scrum']['recall'], 'passes_test_guardrails': passes, **extra}

def train_configuration(name: str, config: dict, output_root: Path, train_rows: pd.DataFrame, validation_rows: pd.DataFrame, test_rows: pd.DataFrame, store: common.FeatureStore, validation_guards: dict, test_guards: dict, quick: bool, force: bool) -> list[dict]:
    run_dir = output_root / 'training' / name
    complete_path = run_dir / 'complete.json'
    if _resolve_model_artifact(complete_path).exists() and (not force):
        return json.loads(_resolve_model_artifact(complete_path).read_text(encoding='utf-8'))['results']
    run_dir.mkdir(parents=True, exist_ok=True)
    if config['line_weight'] != 1.0 and config['line_oversample'] != 1.0:
        raise AssertionError('Do not combine sample weighting and oversampling')
    effective_train = oversample_line(train_rows, config['line_oversample'])
    effective_train.to_csv(run_dir / 'train_manifest.csv', index=False)
    json_dump(run_dir / 'configuration.json', {**config, 'seed': SEED, 'base_train_rows': len(train_rows), 'effective_train_rows': len(effective_train), 'repeated_rows': int(effective_train.is_repeat.sum()), 'maximum_epochs': 1 if quick else MAX_EPOCHS, 'batch_size': BATCH_SIZE, 'validation_guards': validation_guards, 'architecture': 'exact natural_unweighted ResNet50-BiGRU'})
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model = common.build_model()
    train_data = FeatureBatches(effective_train, store, True, config['line_weight'], True)
    validation_data = FeatureBatches(validation_rows, store, False)
    selector = ValidationSelector(validation_data, validation_rows.y.to_numpy(dtype=np.int32), run_dir, validation_guards)
    callbacks = [selector, keras.callbacks.EarlyStopping(monitor='val_macro_f1', mode='max', patience=10, restore_best_weights=False, verbose=1), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-06, verbose=1)]
    print(f'Training {name}: rows={len(effective_train)} config={config}', flush=True)
    started = time.perf_counter()
    history_object = model.fit(train_data, validation_data=validation_data, epochs=1 if quick else MAX_EPOCHS, callbacks=callbacks, verbose=2, shuffle=False)
    seconds = time.perf_counter() - started
    epochs = len(history_object.history.get('loss', []))
    pd.DataFrame(history_object.history).to_csv(run_dir / 'history.csv', index=False)
    results = []
    test_data = FeatureBatches(test_rows, store, False)
    for selection in SELECTIONS:
        checkpoint = run_dir / f'{selection}.keras'
        if not _resolve_model_artifact(checkpoint).exists():
            continue
        selected_model = keras.models.load_model(_resolve_model_artifact(checkpoint), compile=False)
        val_prob = predict_probabilities(selected_model, validation_data)
        test_prob = predict_probabilities(selected_model, test_data)
        val_metrics = metrics_from_probabilities(validation_rows.y.to_numpy(), val_prob)
        result_dir = run_dir / selection
        result_dir.mkdir(parents=True, exist_ok=True)
        test_metrics, predicted = save_predictions(result_dir / 'predictions.csv', test_rows, test_prob)
        json_dump(result_dir / 'validation_metrics.json', val_metrics)
        json_dump(result_dir / 'test_metrics.json', test_metrics)
        matrix = confusion_matrix(test_rows.y.to_numpy(), predicted, labels=np.arange(len(LABELS)))
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(result_dir / 'confusion_matrix.csv')
        results.append(result_row(f'{name}/{selection}', 'retraining', 1.0, val_metrics, test_metrics, epochs, seconds, test_guards, line_weight=config['line_weight'], line_oversample=config['line_oversample'], selection=selection))
        del selected_model
    json_dump(complete_path, {'results': results})
    del model, train_data, validation_data, test_data
    keras.backend.clear_session()
    gc.collect()
    return results

def reference_rows(reference_val: dict, reference_test: dict, test_guards: dict) -> list[dict]:
    reference = result_row('natural_unweighted_reference', 'reference', 1.0, reference_val, reference_test, 23, 1415.0774322999932, test_guards)
    metrics = json.loads(_resolve_model_artifact(BALANCED_RUN / 'metrics.json').read_text(encoding='utf-8'))
    balanced = {'experiment': 'balanced_reference', 'method': 'historical_reference', 'line_alpha': 1.0, 'epochs_ran': metrics['epochs_ran'], 'training_time_seconds': metrics['training_time_seconds'], 'val_accuracy': np.nan, 'val_f1_macro': np.nan, 'val_line_precision': np.nan, 'val_line_recall': np.nan, 'accuracy': metrics['accuracy'], 'f1_macro': metrics['f1_macro'], 'f1_weighted': metrics['f1_weighted'], 'precision_line': metrics['per_class']['line']['precision'], 'recall_line': metrics['per_class']['line']['recall'], 'f1_line': metrics['per_class']['line']['f1'], 'recall_other': metrics['per_class']['other']['recall'], 'recall_scrum': metrics['per_class']['scrum']['recall'], 'passes_test_guardrails': False}
    return [reference, balanced]

def prediction_path(output_root: Path, experiment: str) -> Path:
    if experiment == 'natural_unweighted_reference':
        return REFERENCE_RUN / 'predictions.csv'
    if experiment == 'balanced_reference':
        return BALANCED_RUN / 'predictions.csv'
    if experiment.startswith('calibration_'):
        return output_root / 'calibrations' / experiment / 'predictions.csv'
    training, selection = experiment.split('/', 1)
    return output_root / 'training' / training / selection / 'predictions.csv'

def bootstrap_all(output_root: Path, summary: pd.DataFrame, quick: bool) -> None:
    reference = pd.read_csv(_resolve_model_artifact(REFERENCE_RUN / 'predictions.csv'))
    reference_names = reference.video_name.tolist()
    y_true = np.array([LABEL_TO_ID[x] for x in reference.true_label])
    y_reference = np.array([LABEL_TO_ID[x] for x in reference.predicted_label])
    rng = np.random.default_rng(SEED + 9100)
    replicates = 100 if quick else BOOTSTRAP_REPLICATES
    rows = []
    for experiment in summary.experiment:
        if experiment == 'natural_unweighted_reference':
            continue
        path = prediction_path(output_root, experiment)
        if not _resolve_model_artifact(path).exists():
            continue
        current = pd.read_csv(_resolve_model_artifact(path))
        if current.video_name.tolist() != reference_names:
            if quick:
                ref_index = reference.set_index('video_name')
                y_true_current = np.array([LABEL_TO_ID[x] for x in current.true_label])
                y_ref_current = np.array([LABEL_TO_ID[x] for x in ref_index.loc[current.video_name].predicted_label])
                y_current = np.array([LABEL_TO_ID[x] for x in current.predicted_label])
            else:
                raise ValueError(f'Prediction order mismatch: {experiment}')
        else:
            y_true_current, y_ref_current = (y_true, y_reference)
            y_current = np.array([LABEL_TO_ID[x] for x in current.predicted_label])
        samples = {metric: {'estimate': [], 'difference': []} for metric in ('accuracy', 'f1_macro', 'recall_line')}
        for _ in range(replicates):
            ids = rng.integers(0, len(y_true_current), size=len(y_true_current))
            now = common.metric_bundle(y_true_current[ids], y_current[ids])
            ref = common.metric_bundle(y_true_current[ids], y_ref_current[ids])
            current_values = {'accuracy': now['accuracy'], 'f1_macro': now['f1_macro'], 'recall_line': now['per_class']['line']['recall']}
            reference_values = {'accuracy': ref['accuracy'], 'f1_macro': ref['f1_macro'], 'recall_line': ref['per_class']['line']['recall']}
            for metric in samples:
                samples[metric]['estimate'].append(current_values[metric])
                samples[metric]['difference'].append(current_values[metric] - reference_values[metric])
        for metric, metric_samples in samples.items():
            estimates = metric_samples['estimate']
            differences = metric_samples['difference']
            estimate_low, estimate_high = np.percentile(estimates, [2.5, 97.5])
            low, high = np.percentile(differences, [2.5, 97.5])
            p_value = min(1.0, 2 * min(np.mean(np.asarray(differences) <= 0), np.mean(np.asarray(differences) >= 0)))
            rows.append({'experiment': experiment, 'metric': metric, 'replicates': replicates, 'bootstrap_mean': float(np.mean(estimates)), 'metric_ci95_low': estimate_low, 'metric_ci95_high': estimate_high, 'mean_difference_vs_reference': float(np.mean(differences)), 'difference_ci95_low': low, 'difference_ci95_high': high, 'paired_two_sided_p': p_value})
    pd.DataFrame(rows).to_csv(output_root / 'bootstrap_comparisons.csv', index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', choices=TRAINING_CONFIGS, default=list(TRAINING_CONFIGS))
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_root = HERE / ('smoke' if args.quick else 'full')
    output_root.mkdir(parents=True, exist_ok=True)
    train, validation, test = common.load_base_splits(args.quick)
    store = common.load_base_store()
    train_rows = frame_rows(train, store)
    validation_rows = frame_rows(validation, store)
    test_rows = frame_rows(test, store)
    train_rows.to_csv(output_root / 'train_natural.csv', index=False)
    validation_rows.to_csv(output_root / 'validation_fixed.csv', index=False)
    test_rows.to_csv(output_root / 'test_natural_fixed.csv', index=False)
    val_prob, test_prob, reference_val, reference_test = reference_probabilities(store, validation_rows, test_rows, output_root)
    validation_guards, test_guards = guard_thresholds(reference_val, reference_test)
    audit = {'seed': SEED, 'quick': args.quick, 'splits': {'train': len(train), 'validation': len(validation), 'test': len(test)}, 'labels': LABELS, 'minimum_line_precision': MIN_LINE_PRECISION, 'maximum_global_metric_drop': MAX_GLOBAL_DROP, 'validation_guards': validation_guards, 'test_guards': test_guards, 'reference_validation': reference_val, 'reference_test': reference_test, 'training_configs': TRAINING_CONFIGS, 'selection': 'validation only; best accuracy, best macro F1, and highest guarded line recall', 'test_policy': 'test evaluated after checkpoint/calibration selection'}
    json_dump(output_root / 'audit.json', audit)
    if args.audit_only:
        print(f'Audit written to {output_root}')
        return
    results = reference_rows(reference_val, reference_test, test_guards)
    results.extend(run_calibrations(output_root, validation_rows, test_rows, val_prob, test_prob, validation_guards, test_guards, args.quick))
    pd.DataFrame(results).to_csv(output_root / 'comparison_summary.csv', index=False)
    for name in args.configs:
        results.extend(train_configuration(name, TRAINING_CONFIGS[name], output_root, train_rows, validation_rows, test_rows, store, validation_guards, test_guards, args.quick, args.force))
        pd.DataFrame(results).to_csv(output_root / 'comparison_summary.csv', index=False)
    summary = pd.DataFrame(results)
    summary.to_csv(output_root / 'comparison_summary.csv', index=False)
    bootstrap_all(output_root, summary, args.quick)
    eligible = summary[summary.passes_test_guardrails == True].sort_values(['recall_line', 'f1_macro'], ascending=False)
    if len(eligible):
        json_dump(output_root / 'best_candidate.json', eligible.iloc[0].to_dict())
    print(f'Completed line recall experiment in {output_root}', flush=True)
if __name__ == '__main__':
    main()
