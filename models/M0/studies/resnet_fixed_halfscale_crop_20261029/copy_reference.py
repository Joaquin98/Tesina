"""Resume and verify interrupted reference transfer, preserving source artifacts."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from datetime import datetime, timezone
import json, time, shutil, hashlib, ctypes, traceback
from models.M0.studies.resnet_fixed_halfscale_crop_20261029.preflight import write_json, digest, preservation_inventory
HERE = experiment_directory(__file__)
OUT = HERE / 'outputs'
OLD = HERE.parent / 'resnet_resolution_20261008'
ARCH = HERE / 'archive/reference_reuse_20261029T205032Z'

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    start = time.perf_counter()
    state = ctypes.windll.kernel32.SetThreadExecutionState
    state(2147483649)
    try:
        write_json(OUT / 'reference_maintenance.json', dict(status='checking_reference_copy', time_utc=now(), reason='Compare destination chunks with source hashes and verify integrity. Source read-only.'))
        dst = OUT / 'cache/R224'
        dst.mkdir(exist_ok=True, parents=True)
        src = OLD / 'outputs/cache/R224'
        hashes = {}
        for name in ('features.npy', 'masks.npy'):
            p = src / name
            q = dst / name
            h = hashlib.sha256()
            repaired = 0
            with _resolve_model_artifact(p).open('rb') as a, _resolve_model_artifact(q).open('r+b' if _resolve_model_artifact(q).exists() else 'w+b') as b:
                while True:
                    x = a.read(8 * 1024 * 1024)
                    if not x:
                        break
                    pos = b.tell()
                    y = b.read(len(x))
                    h.update(x)
                    if x != y:
                        b.seek(pos)
                        b.write(x)
                        repaired += len(x)
                b.truncate(p.stat().st_size)
            expected = h.hexdigest()
            assert digest(q) == expected
            hashes[name] = expected
            write_json(OUT / 'reference_copy_progress.json', dict(file=name, status='hash_verified', repaired_bytes=repaired, time_utc=now()))
        shutil.copy2(src / 'state.json', dst / 'state.json')
        for pattern, base in [('R224_S*', 'initial'), ('M0_R224_*', 'runs')]:
            for folder in sorted(_model_artifact_glob(OLD / 'outputs' / base, pattern, recursive=False)):
                if _resolve_model_artifact(folder / 'complete.json').exists():
                    shutil.copytree(folder, OUT / base / folder.name, dirs_exist_ok=True)
        config = json.loads(_resolve_model_artifact(HERE / 'config.json').read_text(encoding='utf-8-sig'))
        config['source_policy'] = config['source_policy'].replace('Both conditions regenerated.', 'Condition320 regenerated. Condition224 reuses the proven byte-equivalent reference.')
        config['reference224_policy'] = 'Reuse copied equivalent224 features, masks, three initial heads and18 evaluations, with historical cost provenance. Source folder remains read-only.'
        write_json(HERE / 'config.json', config)
        ch = digest(HERE / 'config.json')
        originals = ARCH / 'original_metadata'
        originals.mkdir(exist_ok=True)
        for base, pattern in [('initial', 'R224_S*'), ('runs', 'M0_R224_*')]:
            for folder in sorted(_model_artifact_glob(OUT / base, pattern, recursive=False)):
                path = folder / 'complete.json'
                meta = json.loads(_resolve_model_artifact(path).read_text())
                shutil.copy2(path, originals / (folder.name + '_complete.json'))
                meta.update(config_hash=ch, artifact_origin='reused_equivalent_previous224', origin_directory=str(OLD / 'outputs' / base / folder.name), origin_config_hash=meta['config_hash'], cost_measurements_origin='historical_previous_experiment')
                for key in ('training_s', 'prediction_s', 'threshold_and_metrics_s'):
                    meta.setdefault(key, None)
                write_json(path, meta)
        path = dst / 'state.json'
        meta = json.loads(_resolve_model_artifact(path).read_text())
        meta.update(config_hash=ch, artifact_origin='reused_equivalent_previous224', cost_measurements_origin='historical_previous_experiment')
        write_json(path, meta)
        initial = list(_model_artifact_glob(OUT / 'initial', 'R224_S*/complete.json', recursive=False))
        runs = list(_model_artifact_glob(OUT / 'runs', 'M0_R224_*/complete.json', recursive=False))
        assert len(initial) == 3 and len(runs) == 18
        assert json.loads(_resolve_model_artifact(OUT / 'previous_experiment_inventory.json').read_text()) == preservation_inventory()
        reused = dict(status='validated', time_utc=now(), source=str(OLD), features_sha256=hashes['features.npy'], masks_sha256=hashes['masks.npy'], copied_initial=3, copied_runs=18, archive=str(ARCH), proofs=['reference224_equivalence_precheck.json', 'reference224_equivalence_final.json'], copy_validation_s=time.perf_counter() - start, old_experiment_unchanged=True)
        write_json(OUT / 'reference224_reuse.json', reused)
        ledger = OUT / 'worker_timings.jsonl'
        with _resolve_model_artifact(ledger).open('a', encoding='utf-8') as f:
            partial = json.loads(_resolve_model_artifact(ARCH / 'partial_regenerated_R224/state.json').read_text())
            f.write(json.dumps(dict(command=['interrupted_repeated_prefix', '--size', 224], wall_s=partial['active_s'], time_utc=now())) + '\n')
            f.write(json.dumps(dict(command=['reference_copy_validation_and_integrity', '--size', 224], wall_s=time.perf_counter() - start, time_utc=now())) + '\n')
        snap = json.loads(_resolve_model_artifact(OUT / 'protocol_snapshot.json').read_text())
        snap['config'] = config
        snap['reference_reuse'] = config['reference224_policy']
        snap.setdefault('amendments', []).append(dict(time_utc=now(), reason='Exact224 equivalence verified before 320 training/test; equivalent features are reused for the initial stage. No numeric protocol changes.', initial_stage='Extract only the fixed1800-row initial split unique sources, train definitive320 seed17, report clip validation, merge features into full cache and continue all36 evaluations.'))
        paths = [_resolve_model_artifact(HERE / 'early_check.py'), _resolve_model_artifact(HERE / 'test_staged_cache.py'), OUT / 'reference224_equivalence_precheck.json', OUT / 'reference224_equivalence_final.json', OUT / 'reference224_reuse.json']
        for item in snap['inputs']:
            item['sha256'] = digest(Path(item['path']))
        existing = {i['path'] for i in snap['inputs']}
        snap['inputs'] += [dict(path=str(p.resolve()), sha256=digest(p)) for p in paths if str(p.resolve()) not in existing]
        write_json(OUT / 'protocol_snapshot.json', snap)
        write_json(OUT / 'reference_maintenance.json', dict(status='reference_reuse_validated_ready_to_resume320', time_utc=now()))
        print('REFERENCE_COPY_COMPLETE', flush=True)
    except Exception as e:
        write_json(OUT / 'reference_copy_failure.json', dict(error=repr(e), traceback=traceback.format_exc(), time_utc=now()))
        raise
    finally:
        state(2147483648)
if __name__ == '__main__':
    main()
