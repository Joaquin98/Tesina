"""Budgeted, validation-only random search on shared ResNet features.

Run with the repository's working TensorFlow Python: python run_search.py.
Missing RGB features are extracted once; the frozen CNN is not trained.
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
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path, RGBFeatureStore
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import gc
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import keras
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
SOURCE = ROOT / 'models/M0'
LABELS = ['line', 'other', 'scrum']
SEED = 20261022
SPLIT_SEED = 42
BATCH = 32
BUDGET = 3600
MAX_TRIALS = 12

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def digest(path):
    return hashlib.sha256(_resolve_model_artifact(study_csv_path(Path(path))).read_bytes()).hexdigest()

def manifest(path):
    df = pd.read_csv(_resolve_model_artifact(study_csv_path(path)))[['video_name', 'tag']]
    df.tag = df.tag.str.strip().str.lower()
    assert set(df.tag) == set(LABELS)
    assert df.groupby('video_name').tag.nunique().max() == 1
    return df.drop_duplicates('video_name').reset_index(drop=True)

def counts(df):
    return {'rows': len(df), 'unique': int(df.video_name.nunique()), 'classes': {k: int(v) for k, v in df.tag.value_counts().items()}}

def prepare():
    train_unique, val, test = (frame[['video_name', 'tag']].copy() for frame in current_frames())
    rng = np.random.default_rng(SPLIT_SEED)
    groups = []
    for label in LABELS:
        group = train_unique[train_unique.tag == label].reset_index(drop=True)
        if len(group) >= 420:
            part = group.iloc[rng.choice(len(group), 420, replace=False)]
        else:
            part = pd.concat([group, group.iloc[rng.choice(len(group), 420 - len(group), replace=True)]])
        groups.append(part)
    train = pd.concat(groups, ignore_index=True)
    train = train.iloc[rng.permutation(len(train))].reset_index(drop=True)
    val = val.reset_index(drop=True)
    parts = {'train_unique': train_unique, 'train': train, 'validation': val, 'test': test}
    intersections = {}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        overlap = sorted(set(parts[a].video_name) & set(parts[b].video_name))
        intersections[a + '_' + b] = overlap
        assert not overlap, (a, b, overlap)
    for name, frame in parts.items():
        frame.to_csv(HERE / (name + '.csv'), index=False)
    audit = {'split_seed': SPLIT_SEED, 'counts': {k: counts(v) for k, v in parts.items()}, 'clip_intersections': intersections, 'group_split': False, 'group_limitation': 'No complete verified clip-to-match mapping in historical manifests. Existing F/R and team-token source grouping is heuristic, not verified match provenance. Fixed current CSV partitions are used; match-level independence is not inferred from clip names.', 'resampling': 'Training only: mixed sampling to 420/class after loading the fixed current partitions.', 'test_policy': 'Only identifiers/labels read for disjointness audit; no test features/predictions until winner frozen. Historical test already used in prior models.common.studies.', 'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in [CURRENT_DATASET / 'train.csv', CURRENT_DATASET / 'validation.csv', CURRENT_DATASET / 'test.csv', SOURCE / 'model.py', SOURCE / 'config.py']}}
    save(HERE / 'split_audit.json', audit)
    return (train, val, test)

class Store(RGBFeatureStore):
    """Batch access to the shared current RGB feature cache."""
    pass

class Data(keras.utils.PyDataset):

    def __init__(self, df, store, shuffle=False, seed=SEED):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)
        self.store = store
        self.rows = np.array([store.index[n] for n in df.video_name])
        self.y = np.array([LABELS.index(t) for t in df.tag], dtype=np.int32)
        self.order = np.arange(len(df))
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return math.ceil(len(self.rows) / BATCH)

    def __getitem__(self, i):
        order = self.order[i * BATCH:(i + 1) * BATCH]
        rows = self.rows[order]
        return ((np.asarray(self.store.features[rows]), np.asarray(self.store.masks[rows], dtype=bool)), self.y[order])

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)

def build(p):
    feat = keras.Input((200, 2048), name='features')
    mask = keras.Input((200,), dtype='bool', name='mask')
    x = keras.layers.Bidirectional(keras.layers.GRU(p['units'], return_sequences=True))(feat, mask=mask)
    x = keras.layers.Bidirectional(keras.layers.GRU(max(p['units'] // 2, 8)))(x)
    x = keras.layers.Dropout(p['dropout'])(x)
    if p['dense']:
        x = keras.layers.Dense(p['dense'], activation='relu')(x)
    out = keras.layers.Dense(3, activation='softmax')(x)
    model = keras.Model([feat, mask], out)
    model.compile(optimizer=keras.optimizers.Adam(p['learning_rate']), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def predict(model, data):
    return np.concatenate([model(x, training=False).numpy() for x, _ in (data[i] for i in range(len(data)))])

def metrics(y, prob):
    pred = prob.argmax(axis=1)
    report = classification_report(y, pred, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0)
    return {'f1_macro': float(f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0)), 'accuracy': float(np.mean(y == pred)), **{f'{label}_{metric}': float(report[label][key]) for label in LABELS for metric, key in [('f1', 'f1-score'), ('recall', 'recall')]}, 'classification_report': report, 'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}

class Monitor(keras.callbacks.Callback):

    def __init__(self, val, directory, deadline):
        super().__init__()
        self.val, self.directory, self.deadline = (val, directory, deadline)
        self.best, self.wait, self.best_epoch = (-1.0, 0, 0)
        self.rows = []
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.monotonic()

    def on_train_batch_end(self, batch, logs=None):
        if time.monotonic() > self.deadline:
            self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        prob = predict(self.model, self.val)
        scores = metrics(self.val.y, prob)
        score = scores['f1_macro']
        if score > self.best:
            self.best, self.best_epoch, self.wait = (score, epoch + 1, 0)
            self.model.save_weights(self.directory / 'best.weights.h5')
            np.save(self.directory / 'validation_probabilities.npy', prob)
            save(self.directory / 'validation_metrics.json', scores)
        else:
            self.wait += 1
        elapsed = time.monotonic() - self.start
        self.epoch_times.append(elapsed)
        self.rows.append({'epoch': epoch + 1, **{k: float(v) for k, v in (logs or {}).items()}, **{f'val_{k}': v for k, v in scores.items() if isinstance(v, float)}, 'duration_s': elapsed})
        pd.DataFrame(self.rows).to_csv(self.directory / 'history.csv', index=False)
        print(f'epoch {epoch + 1}: val_macro_f1={score:.4f}, best={self.best:.4f}, time={elapsed:.1f}s', flush=True)
        if self.wait >= 2 or time.monotonic() + max(self.epoch_times) * 1.15 > self.deadline:
            self.model.stop_training = True

def random_candidates():
    rng = np.random.default_rng(SEED)
    out = []
    while len(out) < MAX_TRIALS:
        p = {'units': int(rng.choice([32, 64, 128])), 'dropout': float(rng.uniform(0.2, 0.5)), 'dense': int(rng.choice([0, 64])), 'learning_rate': float(10 ** rng.uniform(-4, -3))}
        if p not in out:
            out.append(p)
    return out

def main():
    if _resolve_model_artifact(study_csv_path(HERE / 'environment.json')).exists():
        raise RuntimeError('Experiment already started; preserve it and use a new output directory for reruns.')
    train, val, test = prepare()
    store = Store()
    probes = []
    for name in pd.concat([train, val]).video_name.drop_duplicates().iloc[::250]:
        with _load_model_result_array(SOURCE / 'feature_cache/resnet50' / (Path(name).stem + '.npz')) as original:
            row = store.index[name]
            assert np.array_equal(store.features[row], original['frame_features'])
            assert np.array_equal(store.masks[row], original['frame_masks'])
            assert np.isfinite(store.features[row]).all()
            probes.append(name)
    save(HERE / 'environment.json', {'python': sys.executable, 'python_version': sys.version, 'platform': platform.platform(), 'tensorflow': tf.__version__, 'keras': keras.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'devices': [str(x) for x in tf.config.list_physical_devices()], 'intra_threads': 4, 'inter_threads': 1, 'cache': str(STORE), 'cache_shape': list(store.features.shape), 'npz_equality_probes': probes, 'started_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': digest(__file__)})
    candidates = random_candidates()
    save(HERE / 'configuration.json', {'sampling_seed': SEED, 'initialization_seed': SEED, 'split_seed': SPLIT_SEED, 'budget_s': BUDGET, 'max_trials': MAX_TRIALS, 'max_epochs': 10, 'patience': 2, 'monitor': 'val_f1_macro', 'batch_size': BATCH, 'sequence_length': 200, 'extractor': 'resnet50', 'rnn': 'two stacked BiGRU', 'candidates_in_order': candidates, 'pilot_eligible_for_selection': False})
    start = time.monotonic()
    deadline = start + BUDGET
    keras.utils.set_random_seed(SEED)
    pilot_p = {'units': 128, 'dropout': 0.35, 'dense': 64, 'learning_rate': 0.0003}
    model = build(pilot_p)
    pilot_train = Data(train.groupby('tag', group_keys=False).head(32), store, True)
    pilot_val = Data(val.groupby('tag', group_keys=False).head(32), store)
    batch_times = []
    for i in range(len(pilot_train)):
        t = time.monotonic()
        x, y = pilot_train[i]
        result = model.train_on_batch(x, y, return_dict=True)
        batch_times.append(time.monotonic() - t)
        print(f'pilot batch {i + 1}: {batch_times[-1]:.2f}s {result}', flush=True)
    t = time.monotonic()
    predict(model, pilot_val)
    val_batch_s = (time.monotonic() - t) / len(pilot_val)
    epoch_estimate = max(batch_times[1:]) * len(Data(train, store)) + val_batch_s * len(Data(val, store))
    pilot = {'params': pilot_p, 'training_rows': len(pilot_train.y), 'validation_rows': len(pilot_val.y), 'train_batch_times_s': batch_times, 'validation_batch_s': val_batch_s, 'worst_width_epoch_estimate_s': epoch_estimate, 'duration_s': time.monotonic() - start}
    save(HERE / 'pilot.json', pilot)
    del model
    keras.backend.clear_session()
    gc.collect()
    available = deadline - time.monotonic() - 120
    n_trials = 12
    epochs = min(10, int((available / n_trials - batch_times[0]) / (epoch_estimate * 1.2)))
    while epochs < 3 and n_trials > 1:
        n_trials -= 1
        epochs = min(10, int((available / n_trials - batch_times[0]) / (epoch_estimate * 1.2)))
    epochs = max(1, epochs)
    plan = {'planned_trials': n_trials, 'epochs_cap': epochs, 'worst_width_epoch_estimate_s': epoch_estimate, 'estimated_search_s': n_trials * (batch_times[0] + epoch_estimate * epochs * 1.2), 'reserve_final_evaluation_s': 120, 'rule': 'Reduce epochs, then trials until at least three epochs fit; sequential; admission uses conservative remaining-time estimate.'}
    save(HERE / 'budget_plan.json', plan)
    print('BUDGET PLAN ' + json.dumps(plan), flush=True)
    results = []
    for i, p in enumerate(candidates[:n_trials], 1):
        remaining = deadline - time.monotonic()
        required = batch_times[0] + epoch_estimate * epochs * 1.2 + 120
        if remaining < required:
            print(f'Budget admission stopped before trial {i}: remaining={remaining:.1f}s required={required:.1f}s', flush=True)
            break
        directory = HERE / f'trial_{i:02d}'
        directory.mkdir()
        save(directory / 'parameters.json', p)
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)
        model = build(p)
        train_data = Data(train, store, True)
        val_data = Data(val, store)
        monitor = Monitor(val_data, directory, deadline - 120)
        t = time.monotonic()
        print(f'TRIAL {i}/{n_trials} {p}', flush=True)
        model.fit(train_data, epochs=epochs, callbacks=[monitor], verbose=0, shuffle=False)
        model.load_weights(_resolve_model_artifact(study_csv_path(directory / 'best.weights.h5')))
        model.save(directory / 'best.keras')
        scores = json.loads(_resolve_model_artifact(study_csv_path(directory / 'validation_metrics.json')).read_text())
        row = {'trial': i, **p, 'seed': SEED, 'epochs': len(monitor.rows), 'best_epoch': monitor.best_epoch, 'duration_s': time.monotonic() - t, **{f'val_{k}': v for k, v in scores.items() if isinstance(v, float)}}
        results.append(row)
        pd.DataFrame(results).to_csv(HERE / 'results.csv', index=False)
        print('TRIAL RESULT ' + json.dumps(row), flush=True)
        del model, monitor, train_data, val_data
        keras.backend.clear_session()
        gc.collect()
    if not results:
        save(HERE / 'completion.json', {'status': 'no_trial_admitted', 'elapsed_s': time.monotonic() - start})
        return
    winner = max(results, key=lambda row: row['val_f1_macro'])
    save(HERE / 'selected_configuration.json', {'criterion': 'validation macro F1 only; earliest trial breaks exact ties', **winner})
    t = time.monotonic()
    model = keras.models.load_model(_resolve_model_artifact(study_csv_path(HERE / f"trial_{winner['trial']:02d}" / 'best.keras')))
    test_data = Data(test, store)
    prob = predict(model, test_data)
    final = metrics(test_data.y, prob)
    save(HERE / 'historical_test_metrics.json', {'selected_trial': winner['trial'], **final, 'evaluation_duration_s': time.monotonic() - t, 'caveat': 'Historical test consulted in prior models.common.studies. Not a new external test. Exactly one selected configuration evaluated in this search.'})
    output = test.copy()
    output['prediction'] = [LABELS[x] for x in prob.argmax(axis=1)]
    for j, label in enumerate(LABELS):
        output['p_' + label] = prob[:, j]
    output.to_csv(HERE / 'historical_test_predictions.csv', index=False)
    save(HERE / 'completion.json', {'status': 'complete', 'trials_completed': len(results), 'trials_planned': n_trials, 'search_and_evaluation_elapsed_s': time.monotonic() - start, 'training_budget_s': BUDGET, 'second_seed_run': False, 'finished_utc': datetime.now(timezone.utc).isoformat()})
    print('FINAL ' + json.dumps({'winner': winner, 'test': final, 'elapsed_s': time.monotonic() - start}), flush=True)
if __name__ == '__main__':
    main()
