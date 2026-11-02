"""Run the grouped nested evaluation using only precomputed CNN features.

The outer evaluation split is loaded only after a candidate has been selected
from inner validation macro-F1. All outputs remain inside this experiment.
"""
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
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
sys.path.insert(0, str(source_directory(experiment_directory(__file__))))
from models.M0.studies.independent_evaluation.protocol import FEATURE_ROOT, LABELS, OUTPUT_DIR, SEED, SPLIT_DIR, assert_disjoint, load_candidates, stable_seed, write_json
MAX_SEQ_LENGTH = 200
NUM_FEATURES = 2048

class FeatureSequence(keras.utils.Sequence):

    def __init__(self, frame: pd.DataFrame, extractor: str, batch_size: int, shuffle: bool, seed: int):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.extractor = extractor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.frame))
        self.label_to_index = {label: i for i, label in enumerate(LABELS)}
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.frame) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _cache_path(self, video_name: str) -> Path:
        safe = str(self.extractor).replace(' ', '_')
        return FEATURE_ROOT / safe / f'{Path(video_name).stem}.npz'

    def __getitem__(self, batch_index: int):
        batch_indices = self.indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        features, masks, labels = ([], [], [])
        for index in batch_indices:
            row = self.frame.iloc[int(index)]
            path = self._cache_path(row.video_name)
            if not _resolve_model_artifact(path).exists():
                raise FileNotFoundError(f'Missing cached feature: {path}')
            with _load_model_result_array(path) as data:
                feature = data['frame_features'].astype('float32', copy=False)
                mask = data['frame_masks'].astype('bool', copy=False)
            if feature.shape != (MAX_SEQ_LENGTH, NUM_FEATURES):
                raise ValueError(f'Unexpected feature shape {feature.shape}: {path}')
            features.append(feature)
            masks.append(mask)
            labels.append(self.label_to_index[str(row.tag)])
        return ((np.stack(features), np.stack(masks)), np.asarray(labels, dtype='int32'))

    def labels_in_iteration_order(self) -> np.ndarray:
        return np.asarray([self.label_to_index[str(self.frame.iloc[int(i)].tag)] for i in self.indices], dtype='int32')

    def names_in_iteration_order(self) -> list[str]:
        return [str(self.frame.iloc[int(i)].video_name) for i in self.indices]

    def groups_in_iteration_order(self) -> list[str]:
        return [str(self.frame.iloc[int(i)].source_group) for i in self.indices]

class MemmapFeatureSequence(keras.utils.Sequence):
    """Batch contiguous feature arrays without materializing an entire split in RAM."""

    def __init__(self, dataset: dict[str, object], batch_size: int, shuffle: bool, seed: int):
        super().__init__()
        self.features = _load_model_result_array(dataset['features_path'], mmap_mode='r')
        self.masks = _load_model_result_array(dataset['masks_path'], mmap_mode='r')
        self.labels = _load_model_result_array(dataset['labels_path'], mmap_mode='r')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, batch_index: int):
        indices = self.indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        return ((np.asarray(self.features[indices]), np.asarray(self.masks[indices])), np.asarray(self.labels[indices]))

@keras.saving.register_keras_serializable(package='independent_evaluation')
class MacroF1Metric(keras.metrics.Metric):
    """Exact epoch-level multiclass macro-F1 from an accumulated confusion matrix."""

    def __init__(self, num_classes: int=len(LABELS), name: str='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.matrix = self.add_weight(name='confusion_matrix', shape=(num_classes, num_classes), initializer='zeros', dtype='float32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        actual = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        matrix = tf.math.confusion_matrix(actual, predicted, num_classes=self.num_classes, dtype=tf.float32)
        self.matrix.assign_add(matrix)

    def result(self):
        true_positive = tf.linalg.diag_part(self.matrix)
        actual_count = tf.reduce_sum(self.matrix, axis=1)
        predicted_count = tf.reduce_sum(self.matrix, axis=0)
        f1 = tf.math.divide_no_nan(2.0 * true_positive, actual_count + predicted_count)
        return tf.reduce_mean(f1)

    def reset_state(self):
        self.matrix.assign(tf.zeros_like(self.matrix))

    def get_config(self):
        return {**super().get_config(), 'num_classes': self.num_classes}

def load_feature_arrays(frame: pd.DataFrame, extractor: str) -> dict[str, object]:
    """Load one split once; reused by every fixed candidate in that outer fold."""
    features = np.empty((len(frame), MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')
    masks = np.empty((len(frame), MAX_SEQ_LENGTH), dtype='bool')
    label_to_index = {label: i for i, label in enumerate(LABELS)}
    labels = np.empty(len(frame), dtype='int32')
    for i, row in enumerate(frame.itertuples(index=False)):
        path = FEATURE_ROOT / extractor / f'{Path(row.video_name).stem}.npz'
        if not _resolve_model_artifact(path).exists():
            raise FileNotFoundError(f'Missing cached feature: {path}')
        with _load_model_result_array(path) as data:
            features[i] = data['frame_features']
            masks[i] = data['frame_masks']
        labels[i] = label_to_index[str(row.tag)]
    return {'inputs': (features, masks), 'labels': labels, 'names': frame.video_name.astype(str).tolist(), 'groups': frame.source_group.astype(str).tolist()}

def create_memmap_dataset(frame: pd.DataFrame, extractor: str, target: Path) -> dict[str, object]:
    """Materialize compressed per-clip caches once as batch-friendly NPY arrays."""
    target.mkdir(parents=True, exist_ok=True)
    features_path = target / 'features.npy'
    masks_path = target / 'masks.npy'
    labels_path = target / 'labels.npy'
    manifest_path = target / 'manifest.csv'
    expected_names = frame.video_name.astype(str).tolist()
    reusable = False
    if all((_resolve_model_artifact(path).exists() for path in (features_path, masks_path, labels_path, manifest_path))):
        existing = pd.read_csv(_resolve_model_artifact(manifest_path))
        reusable = existing.video_name.astype(str).tolist() == expected_names
    if not reusable:
        features = np.lib.format.open_memmap(features_path, mode='w+', dtype='float32', shape=(len(frame), MAX_SEQ_LENGTH, NUM_FEATURES))
        masks = np.lib.format.open_memmap(masks_path, mode='w+', dtype='bool', shape=(len(frame), MAX_SEQ_LENGTH))
        labels = np.lib.format.open_memmap(labels_path, mode='w+', dtype='int32', shape=(len(frame),))
        label_to_index = {label: i for i, label in enumerate(LABELS)}
        for i, row in enumerate(frame.itertuples(index=False)):
            path = FEATURE_ROOT / extractor / f'{Path(row.video_name).stem}.npz'
            if not _resolve_model_artifact(path).exists():
                raise FileNotFoundError(f'Missing cached feature: {path}')
            with _load_model_result_array(path) as data:
                features[i] = data['frame_features']
                masks[i] = data['frame_masks']
            labels[i] = label_to_index[str(row.tag)]
        features.flush()
        masks.flush()
        labels.flush()
        frame[['video_name', 'tag', 'source_group']].to_csv(manifest_path, index=False)
        del features, masks, labels
    return {'storage': 'memmap', 'features_path': str(features_path), 'masks_path': str(masks_path), 'labels_path': str(labels_path), 'labels': _load_model_result_array(labels_path, mmap_mode='r'), 'names': expected_names, 'groups': frame.source_group.astype(str).tolist()}

def build_model(candidate: dict[str, object]) -> keras.Model:
    rnn_class = keras.layers.GRU if candidate['rnn_type'] == 'gru' else keras.layers.LSTM

    def recurrent(units: int, return_sequences: bool):
        layer = rnn_class(units, return_sequences=return_sequences)
        return keras.layers.Bidirectional(layer) if candidate['bidirectional'] else layer
    feature_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES), name='features')
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool', name='mask')
    units = int(candidate['rnn_units'])
    x = recurrent(units, True)(feature_input, mask=mask_input)
    x = recurrent(max(units // 2, 8), False)(x)
    x = keras.layers.Dropout(float(candidate['dropout']))(x)
    if int(candidate['dense_units']) > 0:
        x = keras.layers.Dense(int(candidate['dense_units']), activation='relu')(x)
    output = keras.layers.Dense(len(LABELS), activation='softmax')(x)
    model = keras.Model([feature_input, mask_input], output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(candidate['learning_rate'])), loss='sparse_categorical_crossentropy', metrics=['accuracy', MacroF1Metric()])
    return model

def limited(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame.reset_index(drop=True)
    pieces = []
    per_class = max(1, maximum // len(LABELS))
    for _, group in frame.groupby('tag'):
        pieces.append(group.sample(min(per_class, len(group)), random_state=seed))
    result = pd.concat(pieces, ignore_index=True)
    if len(result) < maximum:
        remaining = frame[~frame.video_name.isin(result.video_name)]
        result = pd.concat([result, remaining.sample(min(maximum - len(result), len(remaining)), random_state=seed)], ignore_index=True)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def class_weights(train: pd.DataFrame) -> tuple[dict[int, float], dict[str, float]]:
    encoded = train.tag.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    indices = np.arange(len(LABELS))
    values = compute_class_weight('balanced', classes=indices, y=encoded)
    by_index = {int(i): float(value) for i, value in zip(indices, values)}
    by_label = {LABELS[i]: by_index[i] for i in indices}
    return (by_index, by_label)

def train_candidate(candidate: dict[str, object], train: pd.DataFrame, validation: pd.DataFrame, candidate_dir: Path, epochs: int, patience: int, seed: int, train_data: dict[str, object], validation_data: dict[str, object]) -> dict[str, object]:
    keras.utils.set_random_seed(seed)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(candidate['batch_size'])
    weights, weights_by_label = class_weights(train)
    model = build_model(candidate)
    using_memmap = train_data.get('storage') == 'memmap'
    if using_memmap:
        train_features = _load_model_result_array(train_data['features_path'], mmap_mode='r')
        train_masks = _load_model_result_array(train_data['masks_path'], mmap_mode='r')
        train_labels = _load_model_result_array(train_data['labels_path'], mmap_mode='r')
        val_features = _load_model_result_array(validation_data['features_path'], mmap_mode='r')
        val_masks = _load_model_result_array(validation_data['masks_path'], mmap_mode='r')
        val_labels = _load_model_result_array(validation_data['labels_path'], mmap_mode='r')
    else:
        fit_args = (train_data['inputs'], train_data['labels'])
        fit_validation = (validation_data['inputs'], validation_data['labels'])
        prediction_input = validation_data['inputs']
    started = time.perf_counter()
    if using_memmap:
        checkpoint = candidate_dir / 'best_inner.weights.h5'
        history_values = {'loss': [], 'accuracy': [], 'macro_f1': [], 'val_loss': [], 'val_accuracy': [], 'val_macro_f1': []}
        rng = np.random.default_rng(seed)
        best_value = -np.inf
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(epochs):
            indices = rng.permutation(len(train_labels))
            model.reset_metrics()
            train_result = None
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                train_result = model.train_on_batch([np.asarray(train_features[batch]), np.asarray(train_masks[batch])], np.asarray(train_labels[batch]), class_weight=weights, return_dict=True)
            model.reset_metrics()
            validation_result = None
            for start in range(0, len(val_labels), batch_size):
                stop = min(start + batch_size, len(val_labels))
                validation_result = model.test_on_batch([np.asarray(val_features[start:stop]), np.asarray(val_masks[start:stop])], np.asarray(val_labels[start:stop]), return_dict=True)
            if train_result is None or validation_result is None:
                raise RuntimeError('Empty train or validation partition')
            for key in ('loss', 'accuracy', 'macro_f1'):
                history_values[key].append(float(train_result[key]))
                history_values[f'val_{key}'].append(float(validation_result[key]))
            current = float(validation_result['macro_f1'])
            print(f"Epoch {epoch + 1}/{epochs} - loss={train_result['loss']:.4f} accuracy={train_result['accuracy']:.4f} macro_f1={train_result['macro_f1']:.4f} val_loss={validation_result['loss']:.4f} val_accuracy={validation_result['accuracy']:.4f} val_macro_f1={current:.4f}", flush=True)
            if current > best_value + 1e-12:
                best_value = current
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                model.save_weights(checkpoint)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
        model.load_weights(_resolve_model_artifact(checkpoint))
        history = type('History', (), {'history': history_values})()
        checkpoint_format = 'weights'
        probabilities = []
        for start in range(0, len(val_labels), batch_size):
            stop = min(start + batch_size, len(val_labels))
            probabilities.append(model.predict_on_batch([np.asarray(val_features[start:stop]), np.asarray(val_masks[start:stop])]))
        probabilities = np.concatenate(probabilities, axis=0)
    else:
        checkpoint = candidate_dir / 'best_inner.keras'
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_macro_f1', mode='max', patience=patience, restore_best_weights=True, verbose=0), keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_macro_f1', mode='max', save_best_only=True, verbose=0)]
        history = model.fit(*fit_args, validation_data=fit_validation, batch_size=batch_size, shuffle=True, epochs=epochs, callbacks=callbacks, class_weight=weights, verbose=2)
        best_epoch = int(np.argmax(history.history['val_macro_f1']) + 1)
        checkpoint_format = 'model'
        probabilities = model.predict(prediction_input, verbose=0)
    duration = time.perf_counter() - started
    predicted = probabilities.argmax(axis=1)
    actual = validation_data['labels']
    result = {**candidate, 'seed': seed, 'epochs_executed': len(history.history['loss']), 'best_epoch': best_epoch, 'duration_seconds': duration, 'inner_validation_accuracy': float(accuracy_score(actual, predicted)), 'inner_validation_f1_macro': float(f1_score(actual, predicted, average='macro')), 'inner_validation_f1_weighted': float(f1_score(actual, predicted, average='weighted')), 'class_weights_by_label': weights_by_label, 'checkpoint': str(checkpoint), 'checkpoint_format': checkpoint_format}
    pd.DataFrame(history.history).to_csv(candidate_dir / 'history.csv', index=False)
    write_json(candidate_dir / 'inner_result.json', result)
    del model
    keras.backend.clear_session()
    gc.collect()
    return result

def metrics_from_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=np.arange(len(LABELS)), zero_division=0)
    return {'accuracy': float(accuracy_score(actual, predicted)), 'f1_macro': float(f1_score(actual, predicted, average='macro')), 'f1_weighted': float(f1_score(actual, predicted, average='weighted')), 'per_class': {label: {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i, label in enumerate(LABELS)}, 'confusion_matrix': confusion_matrix(actual, predicted, labels=np.arange(len(LABELS))).tolist()}

def group_bootstrap(predictions: pd.DataFrame, draws: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    groups = predictions.source_group.unique()
    values = {'accuracy': [], 'f1_macro': [], 'f1_weighted': []}
    for _ in range(draws):
        selected = rng.choice(groups, size=len(groups), replace=True)
        pieces = [predictions[predictions.source_group == group] for group in selected]
        sample = pd.concat(pieces, ignore_index=True)
        actual = sample.actual_index.to_numpy()
        predicted = sample.predicted_index.to_numpy()
        values['accuracy'].append(float(accuracy_score(actual, predicted)))
        values['f1_macro'].append(float(f1_score(actual, predicted, average='macro')))
        values['f1_weighted'].append(float(f1_score(actual, predicted, average='weighted')))
    return {key: [float(np.quantile(series, 0.025)), float(np.quantile(series, 0.975))] for key, series in values.items()}

def evaluate_selected(selected: dict[str, object], evaluation: pd.DataFrame, fold_dir: Path, bootstrap_draws: int, seed: int, evaluation_data: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    if selected.get('checkpoint_format') == 'weights':
        model = build_model(selected)
        model.load_weights(_resolve_model_artifact(selected['checkpoint']))
    else:
        model = keras.models.load_model(_resolve_model_artifact(selected['checkpoint']))
    if evaluation_data.get('storage') == 'memmap':
        evaluation_input = MemmapFeatureSequence(evaluation_data, int(selected['batch_size']), False, seed)
    else:
        evaluation_input = evaluation_data['inputs']
    started = time.perf_counter()
    probabilities = model.predict(evaluation_input, batch_size=None if evaluation_data.get('storage') == 'memmap' else int(selected['batch_size']), verbose=0)
    evaluation_seconds = time.perf_counter() - started
    predicted = probabilities.argmax(axis=1)
    actual = evaluation_data['labels']
    names = evaluation_data['names']
    groups = evaluation_data['groups']
    predictions = pd.DataFrame({'video_name': names, 'source_group': groups, 'actual_index': actual, 'actual_label': [LABELS[i] for i in actual], 'predicted_index': predicted, 'predicted_label': [LABELS[i] for i in predicted]})
    for i, label in enumerate(LABELS):
        predictions[f'probability_{label}'] = probabilities[:, i]
    predictions.to_csv(fold_dir / 'outer_predictions.csv', index=False)
    metrics = metrics_from_predictions(actual, predicted)
    metrics['bootstrap_95_group'] = group_bootstrap(predictions, bootstrap_draws, seed)
    metrics['evaluation_seconds'] = evaluation_seconds
    metrics['classification_report'] = classification_report(actual, predicted, target_names=LABELS, output_dict=True, zero_division=0)
    write_json(fold_dir / 'outer_metrics.json', metrics)
    pd.DataFrame(metrics['confusion_matrix'], index=LABELS, columns=LABELS).to_csv(fold_dir / 'confusion_matrix.csv')
    del model
    keras.backend.clear_session()
    gc.collect()
    return (metrics, predictions)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('smoke', 'pilot', 'full'), default='smoke')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    parser.add_argument('--outer-fold', type=int, action='append')
    parser.add_argument('--candidate-index', type=int)
    parser.add_argument('--inner-only', action='store_true')
    args = parser.parse_args()
    smoke = args.mode == 'smoke'
    pilot = args.mode == 'pilot'
    epochs = args.epochs or (1 if smoke else 12 if pilot else 80)
    patience = args.patience if args.patience is not None else 3 if pilot else 7
    fold_numbers = args.outer_fold or ([1] if smoke else [1, 2] if pilot else [1, 2, 3, 4, 5])
    candidates = load_candidates()
    if smoke:
        candidates = candidates[:1]
    elif pilot:
        candidates = candidates[:2]
    if args.candidate_index is not None:
        if args.candidate_index < 0 or args.candidate_index >= len(candidates):
            raise ValueError(f'candidate-index must be between 0 and {len(candidates) - 1}')
        candidates = [candidates[args.candidate_index]]
    result_root = OUTPUT_DIR / ('integration_smoke' if smoke else f'nested_evaluation_pilot_2fold_e{epochs}_manual' if pilot else 'nested_evaluation')
    result_root.mkdir(parents=True, exist_ok=True)
    fold_results = []
    all_predictions = []
    experiment_started = time.perf_counter()
    for fold_number in fold_numbers:
        split_root = SPLIT_DIR / f'outer_{fold_number:02d}'
        train = pd.read_csv(_resolve_model_artifact(split_root / 'train.csv'))
        validation = pd.read_csv(_resolve_model_artifact(split_root / 'validation.csv'))
        evaluation = pd.read_csv(_resolve_model_artifact(split_root / 'evaluation.csv'))
        assert_disjoint({'train': train, 'validation': validation, 'evaluation': evaluation})
        if smoke:
            train = limited(train, 48, SEED + fold_number)
            validation = limited(validation, 24, SEED + fold_number)
            evaluation = limited(evaluation, 24, SEED + fold_number)
        fold_dir = result_root / f'outer_{fold_number:02d}'
        extractor_names = {str(candidate['extractor']) for candidate in candidates}
        if len(extractor_names) != 1:
            raise RuntimeError('Current memory reuse requires one shared extractor family')
        shared_extractor = next(iter(extractor_names))
        if smoke:
            train_data = load_feature_arrays(train, shared_extractor)
            validation_data = load_feature_arrays(validation, shared_extractor)
        else:
            feature_root = OUTPUT_DIR / 'nested_evaluation_pilot_2fold' / '_feature_arrays' if pilot else result_root / '_feature_arrays'
            cache_root = feature_root / f'outer_{fold_number:02d}'
            train_data = create_memmap_dataset(train, shared_extractor, cache_root / 'train')
            validation_data = create_memmap_dataset(validation, shared_extractor, cache_root / 'validation')
        candidate_results = []
        for candidate in candidates:
            candidate_id = str(candidate['candidate_id'])
            seed = stable_seed(SEED, fold_number, candidate_id)
            result = train_candidate(candidate, train, validation, fold_dir / 'candidates' / candidate_id, epochs, patience, seed, train_data, validation_data)
            candidate_results.append(result)
        if args.inner_only:
            write_json(fold_dir / f'inner_only_candidate_{args.candidate_index}.json', candidate_results[0])
            del train_data, validation_data
            gc.collect()
            continue
        selected = sorted(candidate_results, key=lambda item: (-float(item['inner_validation_f1_macro']), -float(item['inner_validation_accuracy']), str(item['candidate_id'])))[0]
        pd.DataFrame(candidate_results).to_csv(fold_dir / 'inner_candidate_selection.csv', index=False)
        write_json(fold_dir / 'selected_candidate.json', selected)
        evaluation_data = load_feature_arrays(evaluation, str(selected['extractor'])) if smoke else create_memmap_dataset(evaluation, str(selected['extractor']), (OUTPUT_DIR / 'nested_evaluation_pilot_2fold' / '_feature_arrays' if pilot else result_root / '_feature_arrays') / f'outer_{fold_number:02d}' / 'evaluation')
        outer_metrics, predictions = evaluate_selected(selected, evaluation, fold_dir, min(args.bootstrap_draws, 100) if smoke else args.bootstrap_draws, stable_seed(SEED, fold_number, 'bootstrap'), evaluation_data)
        predictions['outer_fold'] = fold_number
        all_predictions.append(predictions)
        fold_result = {'outer_fold': fold_number, 'mode': args.mode, 'selected_candidate': selected['candidate_id'], 'train_clips': len(train), 'train_groups': int(train.source_group.nunique()), 'validation_clips': len(validation), 'validation_groups': int(validation.source_group.nunique()), 'evaluation_clips': len(evaluation), 'evaluation_groups': int(evaluation.source_group.nunique()), 'epochs_executed': selected['epochs_executed'], 'training_seconds_selected': selected['duration_seconds'], 'all_candidate_training_seconds': sum((float(r['duration_seconds']) for r in candidate_results)), **{key: outer_metrics[key] for key in ('accuracy', 'f1_macro', 'f1_weighted', 'per_class', 'bootstrap_95_group')}}
        fold_results.append(fold_result)
        write_json(fold_dir / 'fold_summary.json', fold_result)
        del train_data, validation_data, evaluation_data
        gc.collect()
    if args.inner_only:
        print(json.dumps({'mode': args.mode, 'inner_only': True, 'completed_outer_folds': fold_numbers, 'candidate_index': args.candidate_index}, indent=2))
        return
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(result_root / 'all_outer_predictions.csv', index=False)
    flat = []
    for row in fold_results:
        flat.append({'outer_fold': row['outer_fold'], 'selected_candidate': row['selected_candidate'], 'train_clips': row['train_clips'], 'validation_clips': row['validation_clips'], 'evaluation_clips': row['evaluation_clips'], 'epochs_executed': row['epochs_executed'], 'training_seconds_selected': row['training_seconds_selected'], 'all_candidate_training_seconds': row['all_candidate_training_seconds'], 'accuracy': row['accuracy'], 'f1_macro': row['f1_macro'], 'f1_weighted': row['f1_weighted'], 'recall_line': row['per_class']['line']['recall'], 'recall_other': row['per_class']['other']['recall'], 'recall_scrum': row['per_class']['scrum']['recall']})
    flat_df = pd.DataFrame(flat)
    flat_df.to_csv(result_root / 'fold_metrics.csv', index=False)
    summary = {'mode': args.mode, 'completed_outer_folds': fold_numbers, 'candidate_count': len(candidates), 'epochs_cap': epochs, 'elapsed_seconds': time.perf_counter() - experiment_started, 'metrics_mean': {col: float(flat_df[col].mean()) for col in ('accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum')}, 'metrics_standard_deviation': {col: float(flat_df[col].std(ddof=1)) if len(flat_df) > 1 else None for col in ('accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum')}, 'warning': 'Integration-only metrics; not a performance estimate.' if smoke else 'Reduced two-fold nested grouped estimate. It is methodologically valid for these folds, but uncertainty is larger than in the planned five-fold evaluation.' if pilot else 'Nested grouped estimate. Historical development still influenced the candidate family; outer folds were not used inside this new run.'}
    write_json(result_root / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
