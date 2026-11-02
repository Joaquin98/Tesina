import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.M1.studies.m0_m1_cost_folds_20261030.settings import *
import numpy as np, pandas as pd, importlib.util
from models.M1.studies.m0_m1_cost_folds_20261030.fold_core import fold_indices, evaluate

def main():
    p = read(HERE / 'protocol.json')
    splits = read(HERE / 'splits.json')
    frame = pd.read_csv(_resolve_model_artifact(CSV))
    assert p['source_hashes'] == {f: sha(f) for f in p['source_hashes']}
    stats = read(HERE / 'source_cache_stats.json')
    assert stats == {f: stat(f) for f in stats}
    assert sha(HERE / 'prepared/geometry.npy') == p['geometry_sha256']
    spec = importlib.util.spec_from_file_location('final_fold_independent', _resolve_model_artifact(ROOT / 'models/M1/studies/m1_recall_cost_20261027/publish.py'))
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    proofs = {}
    coverage = {key: [] for key, _, _, _ in CONFIGS}
    initial = {}
    for fold in FOLDS:
        tr, dev, te = fold_indices(frame, fold)
        np.testing.assert_array_equal(te, splits[fold]['test']['indices'])
        test = frame.iloc[te].reset_index(drop=True)
        for key, family, _, _ in CONFIGS:
            out = HERE / 'runs' / fold / key
            v = read(out / 'metrics.json')
            trained = read(out / 'trained.json')
            assert trained['epochs'] == 5 and trained['split_sha256'] == sha(HERE / 'splits.json')
            assert trained['model_hashes'] == {f: sha(f) for f in trained['model_hashes']}
            initial.setdefault(family, set()).add(trained['initial_weights_sha256'])
            prob = _load_model_result_array(out / 'probabilities.npy')
            csv = pd.read_csv(_resolve_model_artifact(out / 'predictions.csv'))
            assert csv.block_id.tolist() == test.block_id.tolist()
            labels = np.array(['line', 'other', 'scrum'])[prob.argmax(1)]
            np.testing.assert_allclose(csv[['p_line', 'p_other', 'p_scrum']].to_numpy(), prob, atol=1e-07)
            again = evaluate(test, test.label.tolist(), labels, prob)
            assert again['metrics'] == v['metrics']
            proof = check.independent_check(again['reference_segments'], again['predictions'], again['metrics'], again['evaluated_video_s'])
            assert all((abs(proof[k] - v['review'][k]) < 1e-06 for k in ['unique_flagged_s', 'flagged_outside_true_segments_s']))
            assert not set(test.block_id) & set(frame.iloc[tr].block_id)
            assert not set(test.block_id) & set(frame.iloc[dev].block_id)
            coverage[key].extend(test.block_id.tolist())
            proofs[fold + '_' + key] = proof
    cv = frame[frame.domain == 'CV'].block_id.tolist()
    assert all((len(set(ids)) == 4714 and sorted(ids) == sorted(cv) for ids in coverage.values())) and all((len(x) == 1 for x in initial.values()))
    assert not list(_model_artifact_glob(HERE, 'error_*.json', recursive=False))
    save(HERE / 'verification.json', dict(status='PASS', runs=24, all4714blocks_held_out_once_per_configuration=True, split_train_dev_test_exclusive=True, initialization_fixed_per_family=True, source_cache_files_unchanged=len(stats), source_hashes_unchanged=True, extractors_executed=0, independent_checks=proofs))
if __name__ == '__main__':
    main()
