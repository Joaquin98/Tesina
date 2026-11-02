"""Validate completed paired evaluations and previous-experiment preservation."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json, sys, csv
import numpy as np, pandas as pd
from models.M0.studies.resnet_fixed_halfscale_crop_20261029.experiment import HERE, OUT, CONFIG, guard, digest, manifest, initial_dir, run_dir
from models.M0.studies.resnet_fixed_halfscale_crop_20261029.preflight import write_json, preservation_inventory
from models.common.comparison.protocol import PV_FOLDS, RunSpec
from models.common.comparison.data import split_for_run
from models.common.comparison.metrics import temporal_metrics, apply_class_thresholds

def verify():
    guard()
    m = manifest()
    pairs = {}
    checked = []
    cache = []
    for size in CONFIG['resolutions']:
        state = json.loads(_resolve_model_artifact(OUT / f'cache/R{size}/state.json').read_text())
        assert state.get('status') == 'complete' and state['done'] == len(m)
        mask = _load_model_result_array(OUT / f'cache/R{size}/masks.npy', mmap_mode='r')
        features = _load_model_result_array(OUT / f'cache/R{size}/features.npy', mmap_mode='r')
        assert features.shape == (len(m), 200, 2048) and mask.shape == (len(m), 200)
        cache.append(dict(size=size, features_sha256=digest(OUT / f'cache/R{size}/features.npy'), masks_sha256=digest(OUT / f'cache/R{size}/masks.npy')))
        if size == 320:
            stage = OUT / 'initial_stage/R320'
            ids = json.loads(_resolve_model_artifact(stage / 'global_indices.json').read_text())
            staged = _load_model_result_array(stage / 'features.npy', mmap_mode='r')
            staged_mask = _load_model_result_array(stage / 'masks.npy', mmap_mode='r')
            for local, global_index in enumerate(ids):
                assert np.array_equal(features[global_index], staged[local]), 'Staged feature reuse differs'
                assert np.array_equal(mask[global_index], staged_mask[local]), 'Staged mask reuse differs'
            cache[-1]['staged_features_sha256'] = digest(stage / 'features.npy')
            cache[-1]['staged_features_reused_exactly'] = len(ids)
            early = json.loads(_resolve_model_artifact(OUT / 'early_check.json').read_text())
            assert early['status'] == 'complete' and early['seed'] == 17
            assert early['conditions']['320']['checkpoint_sha256'] == json.loads(_resolve_model_artifact(initial_dir(320, 17) / 'complete.json').read_text())['checkpoint_sha256']
        for seed in CONFIG['seeds']:
            folder = initial_dir(size, seed)
            init = json.loads(_resolve_model_artifact(folder / 'complete.json').read_text())
            assert digest(folder / 'best.keras') == init['checkpoint_sha256']
            reference = json.loads(_resolve_model_artifact(initial_dir(224, seed) / 'complete.json').read_text())
            assert init['initial_weights_sha256'] == reference['initial_weights_sha256']
            for f in PV_FOLDS:
                folder = run_dir(size, seed, f.fold)
                meta = json.loads(_resolve_model_artifact(folder / 'complete.json').read_text())
                metrics = json.loads(_resolve_model_artifact(folder / 'metrics.json').read_text())
                threshold = json.loads(_resolve_model_artifact(folder / 'thresholds.json').read_text())
                assert meta['model'] == 'M0' and meta['size'] == size and (meta['seed'] == seed) and (meta['fold'] == f.fold)
                assert meta['config_hash'] == digest(HERE / 'config.json') and digest(folder / 'model.keras') == meta['checkpoint_sha256'] and (meta['base_sha256'] == init['checkpoint_sha256'])
                assert threshold['source'] == 'validation_only'
                spec = RunSpec('PV', 'M0', 'H2', 'D2', f.fold, seed, f.train_groups, f.development_group, f.test_group)
                tr, dv, te = split_for_run(m, spec)
                test = pd.read_csv(_resolve_model_artifact(folder / 'test_predictions.csv'))
                dev = pd.read_csv(_resolve_model_artifact(folder / 'validation_probabilities.csv'))
                assert test.block_id.tolist() == te.block_id.tolist() and dev.block_id.tolist() == dv.block_id.tolist()
                assert test.y_true.tolist() == te.label.tolist() and dev.y_true.tolist() == dv.label.tolist()
                for part in (test, dev):
                    p = part[['p_line', 'p_other', 'p_scrum']].to_numpy()
                    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1, atol=1e-05)
                p = test[['p_line', 'p_other', 'p_scrum']].to_numpy()
                expected = np.asarray(['line', 'other', 'scrum'])[apply_class_thresholds(p, line=threshold['line'], scrum=threshold['scrum'])]
                assert test.y_final.tolist() == expected.tolist()
                recomputed = temporal_metrics(test)
                for label in ('line', 'scrum'):
                    for key in ('weighted_f1', 'weighted_precision', 'weighted_recall', 'far_false_positives_per_hour'):
                        assert np.isclose(metrics['test'][label][key], recomputed[label][key], atol=1e-07), (size, seed, f.fold, label, key)
                key = (seed, f.fold)
                identity = (test.block_id.tolist(), test.y_true.tolist())
                if size == 224:
                    pairs[key] = identity
                else:
                    assert pairs[key] == identity
                checked.append(dict(size=size, seed=seed, fold=f.fold, rows=len(test)))
    assert len(checked) == 36 and cache[0]['masks_sha256'] == cache[1]['masks_sha256']
    before = json.loads(_resolve_model_artifact(OUT / 'previous_experiment_inventory.json').read_text())
    after = preservation_inventory()
    assert before == after, 'Previous experiment inventory changed; investigate before reporting preservation'
    result = dict(status='passed', checked_runs=checked, cache_hashes=cache, previous_experiment_unchanged=True, paired_initial_weights_equal=True, paired_masks_identical=True, metrics_recomputed=True, note='Only six matches; descriptive paired comparison, not independent external validation.')
    write_json(OUT / 'final_validation.json', result)
    print('FINAL_VALIDATION_PASSED', len(checked), 'runs', flush=True)
if __name__ == '__main__':
    verify()
