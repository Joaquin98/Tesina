"""Wait only for this experiment's workers, then evaluate and publish."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, subprocess, time, json
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(source_directory(HERE)))
from extract import read, save, N, BLOCK, OLD, sha

def main():
    started = time.perf_counter()
    sources = [*sorted(_model_artifact_glob(OLD / 'results/features', '*.npz', recursive=False)), OLD / 'inputs/config.json', ROOT / 'data/m1_recall_cost_20261027/models/cost_2/dual.keras', ROOT / 'data/m1_recall_cost_20261027/models/cost_2/geometry_rf.joblib', ROOT / 'data/m1_ensemble_vs_all_development_20261012/models/dual.keras', ROOT / 'data/m1_ensemble_vs_all_development_20261012/models/geometry_rf.joblib', _resolve_model_artifact(ROOT / 'models/common/spatial.py'), _resolve_model_artifact(ROOT / 'models/M1/geometry_features.py'), _resolve_model_artifact(ROOT / 'models/M1/pose_cache.py'), ROOT / 'data/sampling_audit_20261026/reconciliation/2023-10-20_reference_v3.json']
    protocol = dict(model='M1 frozen original and x2; same native features', user_authorized_missing_extraction=True, reference_scope='partial automated diagnostic inventory, not human whole-match ground truth', immutable_hashes={str(p): sha(p) for p in sources}, native_frames=N, existing_anchors=5778, new_observations_required=138644, new_training=False)
    dest = HERE / 'protocol.json'
    if _resolve_model_artifact(dest).exists():
        assert read(dest) == protocol
    else:
        save(dest, protocol)
    while len(list(_model_artifact_glob(HERE / 'features', '*.npz', recursive=False))) < len(range(0, N, BLOCK)):
        errors = list(_model_artifact_glob(HERE, 'error_worker*.json', recursive=False))
        if errors:
            raise RuntimeError([read(p) for p in errors])
        states = [read(p) for p in _model_artifact_glob(HERE, 'progress_worker*.json', recursive=False)]
        if len(states) == 2 and any((x['stage'] != 'complete' and time.time() - x['updated_unix'] > 600 for x in states)):
            raise RuntimeError('Extraction worker stopped making progress')
        save(HERE / 'pipeline_progress.json', dict(stage='extraction', completed_chunks=len(list(_model_artifact_glob(HERE / 'features', '*.npz', recursive=False))), total_chunks=len(range(0, N, BLOCK)), elapsed_s=time.perf_counter() - started))
        time.sleep(20)
    for name in ('evaluate.py', 'publish.py'):
        save(HERE / 'pipeline_progress.json', dict(stage=name, elapsed_s=time.perf_counter() - started))
        with _resolve_model_artifact(HERE / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, '-B', str(HERE / name)], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    save(HERE / 'pipeline_complete.json', dict(status='complete', elapsed_s=time.perf_counter() - started))
if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        save(HERE / 'pipeline_error.json', dict(error=traceback.format_exc()))
        raise
