"""Coordinate frozen model evaluation while fixed training runs separately."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.M1.studies.m0_m1_cost_sampling_20261029.common import *
import subprocess, traceback
PYTHON = sys.executable

def ready(family, key):
    return all((_resolve_model_artifact(HERE / f'evaluation/{date}/{family}_{key}_stride{stride}/metrics.json').exists() for date, stride in [('20250816', 1), ('20250816', 5), ('20250823', 1), ('20231020', 1), ('VF', 25)]))

def run(script, *args):
    with _resolve_model_artifact(HERE / 'pipeline.log').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(script=script, args=args, start=time.time())) + '\n')
        f.flush()
        subprocess.run([PYTHON, '-B', str(HERE / script), *args], cwd=ROOT, stdout=f, stderr=f, check=True)

def main():
    began = time.perf_counter()
    save(HERE / 'pipeline_running.json', dict(pid=os.getpid(), started_unix=time.time()))
    while True:
        errors = list(_model_artifact_glob(HERE, 'training_error.json', recursive=False)) + list(_model_artifact_glob(HERE, 'evaluation_error.json', recursive=False))
        if errors:
            raise RuntimeError('Recorded error: ' + ','.join((str(p) for p in errors)))
        if not ready('m0', 'x2'):
            run('publish.py')
            time.sleep(20)
            continue
        did = False
        for key, _, _ in CONFIGS:
            if _resolve_model_artifact(HERE / f'models/{key}/complete.json').exists() or key == 'x2':
                if not ready('m0', key):
                    run('evaluate.py', '--family', 'm0', '--key', key)
                    did = True
                if not ready('m1', key):
                    run('evaluate.py', '--family', 'm1', '--key', key)
                    did = True
                if did:
                    run('publish.py', '--native')
                    break
        if not did:
            for key, _, _ in CONFIGS:
                if not ready('m1', key):
                    run('evaluate.py', '--family', 'm1', '--key', key)
                    run('publish.py', '--native')
                    did = True
                    break
        if all((ready(f, k) for f in ['m0', 'm1'] for k, _, _ in CONFIGS)) and _resolve_model_artifact(HERE / 'training_complete.json').exists():
            run('verify.py')
            run('publish.py', '--native')
            subprocess.run(['node', str(HERE / 'verify_web.cjs')], cwd=ROOT, check=True)
            run('publish.py')
            save(HERE / 'pipeline_complete.json', dict(status='complete', elapsed_s=time.perf_counter() - began, configurations=70, extractors_executed=0))
            print('Complete70configurations', flush=True)
            return
        run('publish.py')
        if not did:
            p = read(HERE / 'progress.json')
            import psutil
            if not _resolve_model_artifact(HERE / 'training_complete.json').exists() and (not psutil.pid_exists(p['pid'])):
                raise RuntimeError('Training process stopped before completion')
            time.sleep(20)
if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(HERE / 'pipeline_error.json', dict(error=traceback.format_exc()))
        run('publish.py')
        raise
