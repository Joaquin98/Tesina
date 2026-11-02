"""Fit one genuine H2 dual network and forest on all development data.

Frozen feature files and original models are read-only. Hyperparameters are
locked from previous fold development before VF predictions or labels are used.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import models.M1.studies.m1_ensemble_vs_all_development_20261012.runtime as rt
from models.M1.studies.m1_ensemble_vs_all_development_20261012.runtime import HERE, ROOT, save, sha
import sys, time, json, math, csv, statistics
sys.path.insert(0, str(ROOT))

def main():
    import numpy as np, pandas as pd, tensorflow as tf, joblib, keras, torch, cv2
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    from sklearn.ensemble import RandomForestClassifier
    from models.common.comparison import models, data
    base = ROOT / 'data/final_model_comparison_20260827/outputs'
    manifest = json.loads(_resolve_model_artifact(base / 'web_export_m1_h2_d2/manifest.json').read_text())
    selections = []
    references = [base / 'master_examples.csv', data.HISTORICAL_MANIFEST, data.FINAL_MANIFEST, models.M0_PATH, ROOT / 'models/M0/checkpoints/feature_extractor_resnet50.keras', ROOT / 'yolo11n-pose.pt', _resolve_model_artifact(ROOT / 'models/common/comparison/models.py'), _resolve_model_artifact(ROOT / 'models/common/comparison/data.py')]
    for f in manifest['folds']:
        run = ROOT / Path(f['dual_model']).parent
        h = json.loads(_resolve_model_artifact(run / 'dual.history.json').read_text())
        meta = json.loads(_resolve_model_artifact(run / 'run.json').read_text())
        best = min(range(len(h['val_loss'])), key=lambda i: h['val_loss'][i]) + 1
        selections.append({'fold': f['fold'], 'best_epoch': best, 'alpha': meta['model_metadata']['geometry_alpha'], 'thresholds': meta['thresholds']})
        references.extend([run / 'dual.history.json', run / 'run.json', ROOT / f['dual_model'], ROOT / f['geometry_rf']])
    epochs = math.ceil(statistics.median((s['best_epoch'] for s in selections)))
    alpha = statistics.median((s['alpha'] for s in selections))
    thresholds = {label: statistics.median((s['thresholds'][label] for s in selections)) for label in ['line', 'scrum']}
    assert epochs == 5 and abs(alpha - 0.475) < 1e-10
    selection = {'policy': 'locked medians of prior PV development choices; ceil median best epoch; no VF tuning', 'epochs': epochs, 'geometry_alpha': alpha, 'thresholds': thresholds, 'seed': 42, 'batch_size': 16, 'optimizer': 'Adam 1e-4', 'class_weights': 'balanced on all training examples', 'forest': {'n_estimators': 500, 'max_features': 'sqrt', 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}, 'fold_development_inputs': selections, 'frozen_before_VF_prediction_and_labels': True}
    save('inputs/selection.json', selection)
    frame = pd.read_csv(_resolve_model_artifact(base / 'master_examples.csv'))
    assert len(frame) == 8796 and (frame.domain == 'CH').sum() == 4082 and ((frame.domain == 'CV').sum() == 4714)
    assert set(frame[frame.domain == 'CV'].group_id) == {'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6'}
    assert not frame.block_id.duplicated().any()
    for col in ['resnet_path', 'yolo_path']:
        assert all((_resolve_model_artifact(Path(p)).is_file() for p in frame[col]))
    assert all((_resolve_model_artifact(data.PREPARED_FINAL / f'{block}.npz').is_file() for block in frame[frame.domain == 'CV'].block_id))
    frame.to_csv(HERE / 'inputs/all_training_examples.csv', index=False)
    hashes = {str(p): sha(p) for p in set(references)}
    save('inputs/reference_hashes_before.json', hashes)
    final_sources = pd.read_csv(_resolve_model_artifact(data.FINAL_MANIFEST))
    initial = pd.read_csv(_resolve_model_artifact(data.HISTORICAL_MANIFEST))
    known_overlap = any(('JneZwNgtDYI' in str(v) or 'VF_S01' in str(v) for v in final_sources.to_numpy().ravel()))
    known_overlap = known_overlap or any(('JneZwNgtDYI' in str(v) or 'VF_S01' in str(v) for v in initial.to_numpy().ravel()))
    save('inputs/independence_audit.json', {'training_CV_groups': sorted(frame[frame.domain == 'CV'].group_id.unique()), 'known_VF_identifier_overlap': known_overlap, 'VF_session_ids_in_training': any((str(v).startswith('20260830_112524_0b5e58') for v in frame.block_id)), 'legacy_CH_parent_video_provenance_complete': False, 'independent_test_claim_allowed': False, 'claim': 'VF is absent from the declared CV groups and session IDs, but original CH parent-video provenance is incomplete. F1 is descriptive and must not be claimed independent.', 'frozen_pretrained_components': 'ResNet/YOLO unchanged; initialization from original M0 RGB recurrent layers as in H2 recipe', 'initialization_model': str(models.M0_PATH)})
    models.seed_everything(42)
    training_start = time.perf_counter()
    t = time.perf_counter()
    geometry = data.compact_matrix(frame, ('geometry',))
    preparation_seconds = time.perf_counter() - t
    assert geometry.shape == (8796, 549) and np.isfinite(geometry).all()
    labels = data.labels_for(frame)
    t = time.perf_counter()
    forest = RandomForestClassifier(**selection['forest']).fit(geometry, labels)
    forest_seconds = time.perf_counter() - t
    joblib.dump(forest, HERE / 'models/geometry_rf.joblib')
    del geometry
    model = models.build_dual_model(models.M0_PATH, level='H2')
    model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    class Progress(keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):
            (HERE / 'training_progress.json').write_text(json.dumps({'epoch': epoch + 1, 'epochs': epochs, 'elapsed_seconds': time.perf_counter() - training_start, 'logs': logs or {}}, indent=2))
    t = time.perf_counter()
    history = model.fit(models.ResNetCacheSequence(frame, batch_size=16, shuffle=True, seed=42, dual=True), epochs=epochs, class_weight=models._class_weights(labels), callbacks=[Progress()], verbose=2)
    dual_seconds = time.perf_counter() - t
    model.save(HERE / 'models/dual.keras')
    save('models/dual.history.json', history.history)
    elapsed = time.perf_counter() - training_start
    after = {str(p): sha(p) for p in set(references)}
    assert after == hashes
    one = dict(manifest)
    one['folds'] = [{'fold': 'ALL_DEVELOPMENT', 'run_id': 'ALL-M1-H2-D2-S42', 'attempt': 'isolated_new_training', 'test_group': 'VF_descriptive_independence_unverified', 'development_group': 'previous_PV_development_medians', 'dual_model': str((HERE / 'models/dual.keras').relative_to(ROOT)).replace('\\', '/'), 'geometry_rf': str((HERE / 'models/geometry_rf.joblib').relative_to(ROOT)).replace('\\', '/'), 'geometry_alpha': alpha, 'line_threshold': thresholds['line'], 'scrum_threshold': thresholds['scrum']}]
    one['source_configuration'] = 'ALL-DEVELOPMENT-M1-H2-D2-S42'
    one['deployment_policy'] = 'one_new_dual_and_one_new_forest_trained_on_all_8796_examples'
    one['validation_status'] = 'VF_independence_not_fully_verified; F1_descriptive_only'
    one['recommended_web_thresholds'] = {**thresholds, 'source': 'median_existing_PV_development_only'}
    one['cross_validation_metrics'] = {'scope': 'not assigned to new all-data model'}
    save('inputs/M1-single_manifest.json', one)
    ensemble = dict(manifest)
    ensemble['recommended_web_thresholds'] = {**thresholds, 'source': 'existing_PV_development_medians'}
    save('inputs/M1-ensemble_manifest.json', ensemble)
    save('results/training.json', {'all_checks_pass': True, 'new_training': True, 'no_fold_weight_averaging': True, 'training_examples': 8796, 'initial_clips': 4082, 'match_blocks': 4714, 'epochs_completed': len(history.history['loss']), 'cached_feature_training': True, 'frozen_extractors_not_run_or_retrained': True, 'training_seconds': elapsed, 'geometry_preparation_seconds': preparation_seconds, 'forest_fit_seconds': forest_seconds, 'dual_fit_seconds': dual_seconds, 'hashes_unchanged': True, 'reference_hashes_after': after, 'new_weight_hashes': {str(p): sha(p) for p in [HERE / 'models/dual.keras', HERE / 'models/geometry_rf.joblib']}})
    print('TRAINING_COMPLETE ' + json.dumps({'training_seconds': elapsed, 'epochs': epochs, 'examples': len(frame)}), flush=True)
from pathlib import Path
if __name__ == '__main__':
    main()
