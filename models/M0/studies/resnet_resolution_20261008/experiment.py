"""Controlled paired resolution experiment, isolated and resumable.
Commands: prepare, extract --size S, initial --size S --seed N,
adapt --size S --seed N --fold PVxx [--model M1], summarize, all.
No imported reference entry point is run and no reference output is written.
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
from models.common.study_data import study_csv_path
import argparse, gc, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from models.M0.studies.resnet_resolution_20261008.preflight import CONFIG, Peak, digest, write_json, probe, roi, sample_indices, native_frames
OUT = HERE / 'outputs'
CACHE = OUT / 'cache'
LABELS = ('line', 'other', 'scrum')
import numpy as np
import pandas as pd

def manifest():
    return pd.read_csv(_resolve_model_artifact(study_csv_path(OUT / 'examples.csv')))

def config_hash():
    return digest(HERE / 'config.json')

def initial_dir(size, seed):
    return OUT / 'initial' / f'R{size}_S{seed}'

def run_dir(size, seed, fold, model='M0'):
    return OUT / 'runs' / f'{model}_R{size}_{fold}_S{seed}'

def guard():
    audit = json.loads(_resolve_model_artifact(study_csv_path(OUT / 'source_audit.json')).read_text())
    if audit['status'] != 'ready':
        raise RuntimeError('Source audit blocks extraction or training. See source_audit.json')
    frozen = json.loads(_resolve_model_artifact(study_csv_path(OUT / 'protocol_snapshot.json')).read_text())
    for value in frozen['inputs']:
        if digest(Path(value['path'])) != value['sha256']:
            raise RuntimeError('Protocol input changed since preparation: ' + value['path'])

def prepare():
    from models.common.comparison.data import build_master_manifest
    from models.common.comparison.protocol import PV_FOLDS
    guard_audit = json.loads(_resolve_model_artifact(study_csv_path(OUT / 'source_audit.json')).read_text())
    if guard_audit['status'] != 'ready':
        raise RuntimeError('Cannot freeze plan until sources and masks pass audit')
    m = build_master_manifest(write=False)
    history, final = __import__('preflight').sources()
    source_map = {str(r.video_name): str((ROOT / 'data/7seg unique' / str(r.video_name)).resolve()) for r in history.itertuples()}
    final_map = final.set_index('dataset_id').source_video.to_dict()
    m['source_video'] = [source_map[r.video_name] if r.domain == 'CH' else final_map[r.block_id] for r in m.itertuples()]
    m['cache_index'] = np.arange(len(m))
    stats = {p: Path(p).stat() for p in m.source_video.unique()}
    m['source_bytes'] = [stats[p].st_size for p in m.source_video]
    m['source_mtime_ns'] = [stats[p].st_mtime_ns for p in m.source_video]
    m.to_csv(OUT / 'examples.csv', index=False)
    train = pd.read_csv(_resolve_model_artifact(study_csv_path(CURRENT_DATASET / 'train.csv')))
    names = set(m.loc[m.domain == 'CH', 'video_name'])
    if set(train.video_name) - names:
        raise RuntimeError('Initial training contains unknown source names')
    boundary = int(len(train) * 0.8)
    train['initial_partition'] = ['train'] * boundary + ['validation'] * (len(train) - boundary)
    train.to_csv(OUT / 'initial_split.csv', index=False)
    paths = [HERE / 'config.json', _resolve_model_artifact(study_csv_path(HERE / 'preflight.py')), _resolve_model_artifact(study_csv_path(HERE / 'experiment.py')), _resolve_model_artifact(study_csv_path(HERE / 'pilot.py')), _resolve_model_artifact(study_csv_path(HERE / 'validate_contract.py')), OUT / 'potential_source_overlap.csv', OUT / 'independent_match_inventory.json', OUT / 'source_audit.json', OUT / 'source_audit.csv', OUT / 'examples.csv', OUT / 'initial_split.csv', _resolve_model_artifact(study_csv_path(ROOT / 'models/M0/train_best.py')), _resolve_model_artifact(study_csv_path(ROOT / 'models/M0/model.py')), _resolve_model_artifact(study_csv_path(ROOT / 'models/M0/data_utils.py')), CURRENT_DATASET / 'train.csv', ROOT / 'data/independent_evaluation/outputs/nested_source_manifest.csv', ROOT / 'data/final_tests_feature_cache_20260826/outputs/final_test_feature_manifest.csv', _resolve_model_artifact(study_csv_path(ROOT / 'models/common/comparison/models.py')), _resolve_model_artifact(study_csv_path(ROOT / 'models/common/comparison/data.py')), _resolve_model_artifact(study_csv_path(ROOT / 'models/common/comparison/protocol.py')), _resolve_model_artifact(study_csv_path(ROOT / 'models/common/comparison/metrics.py')), _resolve_model_artifact(study_csv_path(ROOT / 'models/common/spatial.py')), Path.home() / '.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5']
    identities = [dict(path=str(p.resolve()), sha256=digest(p)) for p in paths]
    folds = [dict(fold=f.fold, train_groups=list(f.train_groups), development_group=f.development_group, test_group=f.test_group) for f in PV_FOLDS]
    samples = guard_audit['source_samples']
    smoke = json.loads(_resolve_model_artifact(study_csv_path(OUT / 'smoke/smoke.json')).read_text())
    pilots = [json.loads(_resolve_model_artifact(study_csv_path(OUT / 'compiled_pilot' / f'R{s}.json')).read_text()) for s in CONFIG['resolutions']]
    costs = {str(r['resolution']): samples * r['s_per_frame'] / 3600 for r in pilots}
    write_json(OUT / 'protocol_snapshot.json', dict(config=CONFIG, inputs=identities, folds=folds, initial_rows=len(train), initial_train=boundary, initial_validation=len(train) - boundary, initial_duplicate_names=int(train.video_name.duplicated().sum()), initial_train_validation_overlap=len(set(train.iloc[:boundary].video_name) & set(train.iloc[boundary:].video_name)), examples=len(m), samples=samples, smoke_extraction_projection_hours=costs, projection_warning='Tiny compiled-inference sample in isolated processes. Excludes decode and training. Not a total runtime measurement.', discrepancies=['train_best header says 50 epochs but executed default is 100. New initial training follows code: max100, val_accuracy early stop10, val_loss LR reduction5.', 'Initial Adam weight_decay argument is ignored by build_optimizer(adam). Match Adam with no decay.', 'Initial validation is last20% of ordered balanced_train, not grouped.65 video names are shared across train and validation after balancing. Preserve across sizes and disclose.', 'D2 keeps all CH in every PV fold. Historical F17,F3,F10 codes also occur in filenames of V_1,V_3,V_6. Source-independent separation between CH and CV is not established. Reproduce the explicit existing D2 definition, and interpret results as conditional on this corpus, not proof of unseen-match generalization.', 'Reference 224 videos are recompressed. Re-extract ALL sizes from the SAME native ROI with INTER_LINEAR, including224. New224 is a controlled reconstructed baseline, not an exact replay of old compressed-cache numbers.', 'M0 adaptation follows recorded40 epochs,val_loss patience7,batch16.', 'M1-H2 is a dual RGB/pose head plus500-tree geometry RF and geometric fusion, not just a single RGB head. All YOLO caches remain identical and shared between sizes.'], candidate_rule='M0 mean macro temporal F1 delta >0, positive mean delta in all3 seeds and >=4 of6 matches, class mean deltas >=-0.01. Candidate selection is exploratory on existing six matches. M1 confirmation on these same six is conditional, not independent external validation.', independent_new_match='Not established. Do not invent independent match results.'))
    print('PREPARED', len(m), 'examples', samples, 'frames', flush=True)

def extractor(size):
    import tensorflow as tf, keras
    weights = Path.home() / '.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model = keras.applications.ResNet50(weights=None, include_top=False, pooling='avg', input_shape=(size, size, 3))
    model.load_weights(_resolve_model_artifact(study_csv_path(weights)))
    model.trainable = False

    @tf.function(input_signature=[tf.TensorSpec((None, size, size, 3), tf.float32)])
    def infer(x):
        return model(keras.applications.resnet50.preprocess_input(x), training=False)
    infer(tf.zeros((CONFIG['feature_batch_size'], size, size, 3))).numpy()
    return (model, infer)

def extract(size):
    guard()
    m = manifest()
    folder = CACHE / f'R{size}'
    folder.mkdir(parents=True, exist_ok=True)
    statepath = folder / 'state.json'
    state = json.loads(_resolve_model_artifact(study_csv_path(statepath)).read_text()) if _resolve_model_artifact(study_csv_path(statepath)).exists() else dict(done=0, config_hash=config_hash(), inference_s=0.0, decode_resize_s=0.0, active_s=0.0, frames=0, peak_rss_bytes=0)
    if state['config_hash'] != config_hash():
        raise RuntimeError('Cannot resume cache under changed configuration')
    if state.get('status') == 'complete':
        return
    features = np.lib.format.open_memmap(folder / 'features.npy', mode='r+' if state['done'] else 'w+', dtype=np.float32, shape=(len(m), 200, 2048))
    masks = np.lib.format.open_memmap(folder / 'masks.npy', mode='r+' if state['done'] else 'w+', dtype=bool, shape=(len(m), 200))
    t = time.perf_counter()
    previous_active = state['active_s']
    with Peak() as peak:
        model, infer = extractor(size)
        for idx in range(state['done'], len(m)):
            row = m.iloc[idx]
            tick = time.perf_counter()
            stat = Path(row.source_video).stat()
            if stat.st_size != int(row.source_bytes) or stat.st_mtime_ns != int(row.source_mtime_ns):
                raise RuntimeError('Source changed after protocol preparation: ' + row.source_video)
            info = probe(row.source_video)
            cvrow = type('Times', (), dict(start_time=row.start_s, end_time=row.end_s))()
            indices = sample_indices(info, row.domain, cvrow)
            frames = native_frames(Path(row.source_video), indices, row.domain)
            x = np.asarray([cv2_resize_rgb(frame, size) for frame in frames], dtype=np.float32)
            decode = time.perf_counter() - tick
            tick = time.perf_counter()
            values = []
            for first in range(0, len(x), CONFIG['feature_batch_size']):
                values.append(infer(x[first:first + CONFIG['feature_batch_size']]).numpy())
            values = np.concatenate(values)
            inf = time.perf_counter() - tick
            if values.shape != (len(indices), 2048) or not np.isfinite(values).all():
                raise RuntimeError('Invalid features at ' + row.block_id)
            features[idx] = 0
            features[idx, :len(values)] = values
            masks[idx] = np.arange(200) < len(values)
            features.flush()
            masks.flush()
            state.update(done=idx + 1, inference_s=state['inference_s'] + inf, decode_resize_s=state['decode_resize_s'] + decode, frames=state['frames'] + len(indices), peak_rss_bytes=max(state['peak_rss_bytes'], peak.value))
            state['active_s'] = previous_active + time.perf_counter() - t
            write_json(statepath, state)
            if idx % 10 == 0:
                print(f"EXTRACT R{size} {idx + 1}/{len(m)} frames={state['frames']} inf_s={state['inference_s']:.1f}", flush=True)
            del frames, x, values
    state.update(status='complete', active_s=previous_active + time.perf_counter() - t, peak_rss_bytes=max(state['peak_rss_bytes'], peak.value), features_bytes=(folder / 'features.npy').stat().st_size)
    write_json(statepath, state)

def cv2_resize_rgb(frame, size):
    import cv2
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]

def open_cache(size):
    folder = CACHE / f'R{size}'
    state = json.loads(_resolve_model_artifact(study_csv_path(folder / 'state.json')).read_text())
    if state.get('status') != 'complete':
        raise RuntimeError('Cache extraction incomplete')
    return (_load_model_result_array(folder / 'features.npy', mmap_mode='r'), _load_model_result_array(folder / 'masks.npy', mmap_mode='r'))

def keras_sequence(frame, size, seed, shuffle, dual=False, batch_size=16):
    import keras
    from models.common.comparison.data import load_yolo, geometry_sequence
    x, mask = open_cache(size)

    class Sequence(keras.utils.Sequence):

        def __init__(self):
            super().__init__()
            self.frame = frame.reset_index(drop=True)
            self.indices = np.arange(len(frame))
            self.rng = np.random.default_rng(seed)
            if shuffle:
                self.rng.shuffle(self.indices)

        def __len__(self):
            return int(np.ceil(len(self.frame) / batch_size))

        def __getitem__(self, batch):
            selected = self.indices[batch * batch_size:(batch + 1) * batch_size]
            r = self.frame.iloc[selected]
            ci = r.cache_index.to_numpy(int)
            if dual:
                inputs = dict(rgb=np.asarray(x[ci], np.float32), sequence_mask=np.asarray(mask[ci], bool), pose=np.asarray([geometry_sequence(load_yolo(row)) for row in r.itertuples()], np.float32))
            else:
                inputs = dict(features=np.asarray(x[ci], np.float32), mask=np.asarray(mask[ci], bool))
            return (inputs, np.asarray([LABELS.index(t) for t in r.label], np.int64))

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.indices)
    return Sequence()

def new_head(seed):
    import keras
    keras.utils.set_random_seed(seed)
    f = keras.Input((200, 2048), name='features')
    mask = keras.Input((200,), dtype='bool', name='mask')
    v = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='bidirectional')(f, mask=mask)
    v = keras.layers.Bidirectional(keras.layers.GRU(32), name='bidirectional_1')(v)
    v = keras.layers.Dropout(0.3, name='dropout')(v)
    v = keras.layers.Dense(64, activation='relu', name='dense')(v)
    v = keras.layers.Dense(3, activation='softmax', name='dense_1')(v)
    return keras.Model([f, mask], v, name='best_model')

def weights_digest(model):
    h = hashlib.sha256()
    for w in model.get_weights():
        h.update(w.tobytes())
    return h.hexdigest()

def class_weights(frame):
    from sklearn.utils.class_weight import compute_class_weight
    y = np.asarray([LABELS.index(t) for t in frame.label])
    v = compute_class_weight('balanced', classes=np.arange(3), y=y)
    return dict(enumerate(map(float, v)))

def initial(size, seed):
    import keras
    guard()
    folder = initial_dir(size, seed)
    folder.mkdir(parents=True, exist_ok=True)
    if _resolve_model_artifact(study_csv_path(folder / 'complete.json')).exists():
        return
    m = manifest()
    split = pd.read_csv(_resolve_model_artifact(study_csv_path(OUT / 'initial_split.csv')))
    ch = m[m.domain == 'CH'].set_index('video_name')
    rows = ch.loc[split.video_name].reset_index()
    rows['initial_partition'] = split.initial_partition.to_numpy()
    train = rows[rows.initial_partition == 'train']
    val = rows[rows.initial_partition == 'validation']
    t = time.perf_counter()
    with Peak() as peak:
        model = new_head(seed)
        ih = weights_digest(model)
        model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True), keras.callbacks.ModelCheckpoint(str(folder / 'best.keras'), monitor='val_accuracy', mode='max', save_best_only=True), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-06)]
        history = model.fit(keras_sequence(train, size, seed, True), validation_data=keras_sequence(val, size, seed, False), epochs=100, class_weight=class_weights(rows), callbacks=callbacks, verbose=2)
        write_json(folder / 'history.json', history.history)
    write_json(folder / 'complete.json', dict(size=size, seed=seed, initial_weights_sha256=ih, checkpoint_sha256=digest(folder / 'best.keras'), epochs=len(history.history['loss']), elapsed_s=time.perf_counter() - t, peak_rss_bytes=peak.value, config_hash=config_hash()))

def metric_frame(frame, p):
    result = frame.copy()
    result['y_true'] = result.label
    for i, l in enumerate(LABELS):
        result['p_' + l] = p[:, i]
    return result

def predict(model, frame, size, seed, dual=False):
    from models.common.comparison.metrics import normalize_probabilities
    sequence = keras_sequence(frame, size, seed, False, dual, batch_size=64)
    return normalize_probabilities(np.concatenate([np.asarray(model.predict_on_batch(sequence[b][0])) for b in range(len(sequence))]))

def fixed_geometry(spec, train, dev, test):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from models.common.comparison.data import geometry_statistics, load_yolo
    folder = OUT / 'fixed_yolo' / f'{spec.fold}_S{spec.seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if _resolve_model_artifact(study_csv_path(folder / 'probabilities.npz')).exists():
        with _load_model_result_array(folder / 'probabilities.npz') as p:
            return (p['dev'], p['test'], digest(folder / 'forest.joblib'))

    def matrix(frame):
        return np.asarray([geometry_statistics(load_yolo(r)) for r in frame.itertuples()], np.float32)
    forest = RandomForestClassifier(n_estimators=500, max_features='sqrt', class_weight='balanced', random_state=spec.seed, n_jobs=-1)
    forest.fit(matrix(train), np.asarray([LABELS.index(v) for v in train.label]))
    dp, tp = (forest.predict_proba(matrix(dev)), forest.predict_proba(matrix(test)))
    joblib.dump(forest, folder / 'forest.joblib')
    np.savez_compressed(folder / 'probabilities.npz', dev=dp, test=tp)
    write_json(folder / 'split.json', dict(train=train.block_id.tolist(), dev=dev.block_id.tolist(), test=test.block_id.tolist(), seed=spec.seed))
    return (dp, tp, digest(folder / 'forest.joblib'))

def adapt(size, seed, fold, kind):
    import keras
    from models.common.comparison.protocol import PV_FOLDS, RunSpec
    from models.common.comparison.data import split_for_run
    from models.common.comparison.metrics import threshold_search, apply_class_thresholds, temporal_metrics, block_metrics
    from models.common.comparison.models import build_dual_model, _choose_alpha, _geometric_fuse
    guard()
    folder = run_dir(size, seed, fold, kind)
    folder.mkdir(parents=True, exist_ok=True)
    if _resolve_model_artifact(study_csv_path(folder / 'complete.json')).exists():
        return
    f = next((x for x in PV_FOLDS if x.fold == fold))
    spec = RunSpec('PV', kind, 'H2', 'D2', fold, seed, f.train_groups, f.development_group, f.test_group)
    train, dev, test = split_for_run(manifest(), spec)
    base = initial_dir(size, seed) / 'best.keras'
    if not _resolve_model_artifact(study_csv_path(base.parent / 'complete.json')).exists():
        raise RuntimeError('Initial training incomplete')
    t = time.perf_counter()
    with Peak() as peak:
        keras.utils.set_random_seed(seed)
        model = keras.models.load_model(_resolve_model_artifact(study_csv_path(base)), compile=False) if kind == 'M0' else build_dual_model(base, level='H2')
        model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(keras_sequence(train, size, seed, True, kind == 'M1'), validation_data=keras_sequence(dev, size, seed, False, kind == 'M1'), epochs=40, class_weight=class_weights(train), callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)], verbose=2)
        model.save(folder / 'model.keras')
        write_json(folder / 'history.json', history.history)
        dp = predict(model, dev, size, seed, kind == 'M1')
        tp = predict(model, test, size, seed, kind == 'M1')
        alpha = None
        gh = None
        if kind == 'M1':
            gd, gt, gh = fixed_geometry(spec, train, dev, test)
            alpha = _choose_alpha(np.asarray([LABELS.index(v) for v in dev.label]), dp, gd, geometric=True)
            dp = _geometric_fuse(dp, gd, alpha)
            tp = _geometric_fuse(tp, gt, alpha)
        selected = threshold_search(metric_frame(dev, dp), dp)
        write_json(folder / 'thresholds.json', dict(line=selected['line'], scrum=selected['scrum'], source='validation_only', geometry_alpha=alpha))
        result = metric_frame(test, tp)
        result['y_final'] = [LABELS[v] for v in apply_class_thresholds(tp, line=selected['line'], scrum=selected['scrum'])]
        result['y_argmax'] = [LABELS[v] for v in tp.argmax(1)]
        result.to_csv(folder / 'test_predictions.csv', index=False)
        dv = metric_frame(dev, dp)
        dv.to_csv(folder / 'validation_probabilities.csv', index=False)
        tm = temporal_metrics(result)
        total = float(result.groupby(['group_id', 'section']).end_s.max().sum()) / 3600
        for label in ('line', 'scrum'):
            cm = tm[label]
            cm['all_unmatched_false_positives_per_hour'] = (cm['predicted_events'] - cm['matched_events']) / total if total else 0
        write_json(folder / 'metrics.json', dict(test=tm, validation=selected['temporal'], block_argmax=block_metrics(np.asarray([LABELS.index(v) for v in test.label]), tp)))
    write_json(folder / 'complete.json', dict(size=size, seed=seed, fold=fold, model=kind, train_groups=list(spec.train_groups), development_group=spec.development_group, test_group=spec.test_group, n_train=len(train), n_dev=len(dev), n_test=len(test), epochs=len(history.history['loss']), elapsed_s=time.perf_counter() - t, peak_rss_bytes=peak.value, checkpoint_sha256=digest(folder / 'model.keras'), base_sha256=digest(base), fixed_geometry_sha256=gh, config_hash=config_hash()))
    print('RUN_COMPLETE', kind, size, seed, fold, flush=True)

def summarize():
    rows = []
    for path in sorted(_model_artifact_glob(OUT / 'runs', '*/complete.json', recursive=False)):
        meta = json.loads(_resolve_model_artifact(study_csv_path(path)).read_text())
        metrics = json.loads(_resolve_model_artifact(study_csv_path(path.parent / 'metrics.json')).read_text())['test']
        r = {k: meta[k] for k in ('model', 'size', 'seed', 'fold', 'test_group', 'elapsed_s', 'peak_rss_bytes', 'epochs')}
        r['f1_temporal_macro'] = metrics['weighted_f1_macro']
        for label in ('line', 'scrum'):
            for key in ('weighted_f1', 'weighted_precision', 'weighted_recall', 'far_false_positives_per_hour', 'all_unmatched_false_positives_per_hour'):
                r[label + '_' + key] = metrics[label][key]
        rows.append(r)
    if not rows:
        raise RuntimeError('No completed performance runs. Smoke results do not support accuracy recommendation')
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'per_match_seed.csv', index=False)
    means = frame.groupby(['model', 'size']).mean(numeric_only=True)
    means.to_csv(OUT / 'comparison_means.csv')
    pairs = []
    summaries = []
    rng = np.random.default_rng(20261008)
    for kind in frame.model.unique():
        base = frame[(frame.model == kind) & (frame['size'] == 224)]
        for size in (320, 448):
            candidate = frame[(frame.model == kind) & (frame['size'] == size)]
            merged = base.merge(candidate, on=['model', 'fold', 'seed', 'test_group'], suffixes=('_224', '_candidate'))
            if len(merged) != 18:
                continue
            for score in ['f1_temporal_macro', 'line_weighted_f1', 'scrum_weighted_f1']:
                merged['delta_' + score] = merged[score + '_candidate'] - merged[score + '_224']
            for r in merged.to_dict('records'):
                pairs.append(dict(model=kind, size=size, **{k: v for k, v in r.items() if k in ('fold', 'seed', 'test_group') or k.startswith('delta_')}))
            bymatch = merged.groupby('test_group')['delta_f1_temporal_macro'].mean()
            byseed = merged.groupby('seed')['delta_f1_temporal_macro'].mean()
            boot = np.mean(rng.choice(bymatch.to_numpy(), size=(20000, 6), replace=True), axis=1)
            rec = dict(model=kind, size=size, paired_runs=18, mean_delta=float(merged.delta_f1_temporal_macro.mean()), match_mean_deltas=bymatch.to_dict(), seed_mean_deltas=byseed.to_dict(), cluster_bootstrap_ci95=np.quantile(boot, [0.025, 0.975]).tolist(), positive_matches=int((bymatch > 0).sum()), positive_seeds=int((byseed > 0).sum()), line_mean_delta=float(merged.delta_line_weighted_f1.mean()), scrum_mean_delta=float(merged.delta_scrum_weighted_f1.mean()))
            rec['promising'] = rec['mean_delta'] > 0 and rec['positive_matches'] >= 4 and (rec['positive_seeds'] == 3) and (min(rec['line_mean_delta'], rec['scrum_mean_delta']) >= -0.01)
            summaries.append(rec)
    pd.DataFrame(pairs).to_csv(OUT / 'paired_differences.csv', index=False)
    write_json(OUT / 'paired_summary.json', dict(comparisons=summaries, n_matches=6, n_seeds=3, note='Descriptive conditional uncertainty. Overlapping fold training sets and only six reused matches limit inference. Seeds are paired repeated training, not18 independent matches. M1 candidate selection does not constitute independent external confirmation.'))
    return (frame, summaries)

def worker(args):
    logs = OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    filename = '_'.join(map(str, args)) + '.log'
    write_json(OUT / 'active_worker.json', dict(command=args, log=str(logs / filename), status='running', time_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    print('START_WORKER', filename, flush=True)
    with _resolve_model_artifact(study_csv_path(logs / filename)).open('a', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', str(_resolve_model_artifact(study_csv_path(HERE / 'experiment.py'))), *map(str, args)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Worker failed: ' + filename + ' returncode=' + str(result.returncode))

def run_all():
    guard()
    for size in CONFIG['resolutions']:
        worker(['extract', '--size', size])
        for seed in CONFIG['seeds']:
            worker(['initial', '--size', size, '--seed', seed])
            for fold in [f'PV{i:02d}' for i in range(1, 7)]:
                worker(['adapt', '--size', size, '--seed', seed, '--fold', fold])
    frame, comparisons = summarize()
    promising = [r for r in comparisons if r['model'] == 'M0' and r['promising']]
    if promising:
        chosen = max(promising, key=lambda r: r['mean_delta'])['size']
        write_json(OUT / 'm1_candidate_decision.json', dict(chosen=chosen, rule=json.loads(_resolve_model_artifact(study_csv_path(OUT / 'protocol_snapshot.json')).read_text())['candidate_rule'], based_on='Completed M0, exploratory selection on six reused matches. Training protocol unchanged.'))
        for size in [224, chosen]:
            for seed in CONFIG['seeds']:
                for fold in [f'PV{i:02d}' for i in range(1, 7)]:
                    worker(['adapt', '--size', size, '--seed', seed, '--fold', fold, '--model', 'M1'])
        summarize()
    write_json(OUT / 'execution_complete.json', dict(status='complete', time_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), runs=len(list(_model_artifact_glob(OUT / 'runs', '*/complete.json', recursive=False)))))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'extract', 'initial', 'adapt', 'summarize', 'all'])
    parser.add_argument('--size', type=int, choices=[224, 320, 448])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--fold')
    parser.add_argument('--model', choices=['M0', 'M1'], default='M0')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare()
        elif args.command == 'extract':
            extract(args.size)
        elif args.command == 'initial':
            initial(args.size, args.seed)
        elif args.command == 'adapt':
            adapt(args.size, args.seed, args.fold, args.model)
        elif args.command == 'summarize':
            summarize()
        else:
            run_all()
    except Exception as e:
        write_json(OUT / 'last_failure.json', dict(status='failed', command=vars(args), error=repr(e), traceback=traceback.format_exc()))
        raise
