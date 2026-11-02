"""Freeze six CV held-out splits and reuse only existing feature tensors."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.M1.studies.m0_m1_cost_folds_20261030.settings import *
import numpy as np, pandas as pd
from models.M1.studies.m0_m1_cost_folds_20261030.fold_core import fold_indices

def main():
    from models.common.comparison import data
    frame = pd.read_csv(_resolve_model_artifact(CSV))
    assert len(frame) == 8796 and (not frame.block_id.duplicated().any())
    manifest = read(ARRAYS / 'manifest.json')
    assert manifest['complete'] and hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() == manifest['frame_sha256']
    master = pd.read_csv(_resolve_model_artifact(data.MASTER_MANIFEST)).set_index('block_id').loc[frame.block_id]
    for column in ['domain', 'group_id', 'section', 'label']:
        assert np.array_equal(frame[column].to_numpy(), master[column].to_numpy()), column
    labels = _load_model_result_array(ARRAYS / 'labels.npy', mmap_mode='r')
    np.testing.assert_array_equal(labels, data.labels_for(frame))
    assert set(frame[frame.domain == 'CV'].group_id) == {f'V_{i}' for i in range(1, 7)}
    files = set([CSV, ARRAYS / 'manifest.json', data.MASTER_MANIFEST, BASE]) | {ARRAYS / f'{k}.npy' for k in ['rgb', 'pose', 'mask', 'labels']}
    for r in frame.itertuples(index=False):
        files.update([Path(r.resnet_path), Path(r.yolo_path)])
        if r.domain == 'CV':
            files.add(data.prepared_final_path(str(r.block_id)))
    missing = [str(p) for p in files if not _resolve_model_artifact(p).is_file()]
    assert not missing, missing[:10]
    stats = {str(p): stat(p) for p in sorted(files)}
    splits = {}
    for fold in FOLDS:
        ids = fold_indices(frame, fold)
        entry = {name: dict(indices=idx.tolist(), blocks=frame.iloc[idx].block_id.tolist(), groups=sorted(frame.iloc[idx].group_id.unique().tolist()), class_counts=np.bincount(np.asarray(labels[idx]), minlength=3).tolist()) for name, idx in zip(['train', 'development', 'test'], ids)}
        from models.common.comparison.protocol import select_run
        reference = data.split_for_run(frame, select_run(f'PV-M0-H2-D2-{fold}-S42'))
        assert all((set(entry[name]['blocks']) == set(part.block_id) for name, part in zip(['train', 'development', 'test'], reference)))
        splits[fold] = entry
        assert not set(entry['test']['blocks']) & set(entry['train']['blocks'])
    assert sorted((i for s in splits.values() for i in s['test']['indices'])) == np.flatnonzero(frame.domain.eq('CV')).tolist()
    out = HERE / 'prepared'
    out.mkdir(exist_ok=True)
    geometry = out / 'geometry.npy'
    if not _resolve_model_artifact(geometry).exists():
        result = []
        for i, row in enumerate(frame.itertuples(index=False)):
            if row.domain == 'CH':
                value = data.geometry_statistics(data.load_yolo(row))
            else:
                with _load_model_result_array(data.prepared_final_path(str(row.block_id)), allow_pickle=False) as z:
                    value = z['geometry'].copy()
            assert value.shape == (549,) and np.isfinite(value).all()
            result.append(value)
            if i % 1000 == 0:
                print(json.dumps(dict(stage='reading_cached_geometry', blocks=i, total=len(frame))), flush=True)
        np.save(geometry, np.asarray(result, dtype=np.float32))
    geom = _load_model_result_array(geometry, mmap_mode='r')
    assert geom.shape == (8796, 549)
    rgb = _load_model_result_array(ARRAYS / 'rgb.npy', mmap_mode='r')
    mask = _load_model_result_array(ARRAYS / 'mask.npy', mmap_mode='r')
    pose = _load_model_result_array(ARRAYS / 'pose.npy', mmap_mode='r')
    assert rgb.shape == (8796, 200, 2048) and mask.shape == (8796, 200) and (pose.shape == (8796, 200, 62))
    for i in [0, 4081, 4082, 5000, 8795]:
        row = frame.iloc[i]
        x, m, _ = data._load_resnet_path.__wrapped__(str(row.resnet_path), str(row.block_id))
        np.testing.assert_array_equal(rgb[i], x)
        np.testing.assert_array_equal(mask[i], m)
        np.testing.assert_array_equal(pose[i], data.geometry_sequence(data.load_yolo(row)))
        np.testing.assert_array_equal(geom[i], data.geometry_statistics(data.load_yolo(row)))
    assert stats == {p: stat(p) for p in stats}
    assert read(PRO / 'verification.json')['status'] == 'PASS' and read(PRO / 'results.json')['completed_configurations'] == 70
    sources = [CSV, BASE, ARRAYS / 'manifest.json', PRO / 'results.json', PRO / 'verification.json', _resolve_model_artifact(HERE / 'run.py'), _resolve_model_artifact(HERE / 'fold_core.py'), _resolve_model_artifact(HERE / 'settings.py'), Path(data.__file__), _resolve_model_artifact(ROOT / 'models/common/comparison/models.py')]
    hashes = {str(p): sha(p) for p in sources}
    protocol = dict(configurations=CONFIGS, folds=FOLDS, panel='PV', new_training_runs=24, seed=42, epochs=5, batch_size=16, optimizer='Adam1e-4', regime='H2-D2; CH + four CV training groups', development='excluded from gradient and final refit; fixed epochs/no selection', class_cost='balanced on training subset only, times[line,1,scrum], empirical mean1', m1_forest=dict(n_estimators=500, max_features='sqrt', random_state=42, n_jobs=8), m1_geometry_alpha=0.475, inputs='existing exact float32 RGB200/pose200/mask memmaps; geometry549 from cached summaries', sampling='full original per-block caches, no sparse simulation', event_matching='same class >=1s overlap one-to-one; group within section only', aggregation='equal-weight fold means primary; micro concatenation secondary', extractors_executed=0, execution='two independent low-priority family workers sharing read-only memory maps; elapsed times not isolated benchmarks', historical_PH='optional scope pending; not part of this six-fold protocol', source_hashes=hashes, source_cache_files=len(stats), geometry_sha256=sha(geometry), provenance='no held-out CV block used in gradient updates; historical initialization provenance remains as prior protocol, not fully recertified; costs chosen on already-observed VF/professional videos, exploratory')
    if _resolve_model_artifact(HERE / 'protocol.json').exists():
        assert read(HERE / 'protocol.json') == json.loads(json.dumps(protocol))
    save(HERE / 'protocol.json', protocol)
    save(HERE / 'splits.json', splits)
    save(HERE / 'source_cache_stats.json', stats)
    save(HERE / 'preflight.json', dict(status='PASS', folds=6, test_blocks=4714, training_examples_available=8796, new_feature_extraction=0, cache_files=len(stats), tensor_equivalence_checks=[0, 4081, 4082, 5000, 8795]))
    print('PREFLIGHT PASS', flush=True)
if __name__ == '__main__':
    main()
