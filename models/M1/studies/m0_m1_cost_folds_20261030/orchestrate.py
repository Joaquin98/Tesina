"""Keep two independent family workers running and one report writer."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.M1.studies.m0_m1_cost_folds_20261030.settings import *
import subprocess, traceback

def run(script):
    subprocess.run([sys.executable, '-B', str(HERE / script)], cwd=ROOT, check=True)

def main():
    import psutil
    if _resolve_model_artifact(HERE / 'running.json').exists():
        old = read(HERE / 'running.json')['pid']
        if old != os.getpid() and psutil.pid_exists(old) and any(('m0_m1_cost_folds_20261030' in a for a in psutil.Process(old).cmdline())):
            raise RuntimeError('An orchestrator is already active')
    for family in ['m0', 'm1']:
        marker = HERE / f'progress_{family}.json'
        if _resolve_model_artifact(marker).exists():
            pid = read(marker)['pid']
            if psutil.pid_exists(pid) and any(('m0_m1_cost_folds_20261030' in a and 'run.py' in a for a in psutil.Process(pid).cmdline())):
                raise RuntimeError('A family worker is already active; do not duplicate ' + family)
    assert read(HERE / 'preflight.json')['status'] == 'PASS'
    save(HERE / 'running.json', dict(pid=os.getpid(), started_unix=time.time()))
    workers = {}
    handles = []
    for family in ['m0', 'm1']:
        if _resolve_model_artifact(HERE / f'complete_{family}.json').exists():
            continue
        if _resolve_model_artifact(HERE / f'error_{family}.json').exists():
            raise RuntimeError('Resolve recorded worker error before resuming ' + family)
        log = _resolve_model_artifact(HERE / f'{family}.log').open('a', encoding='utf-8')
        handles.append(log)
        workers[family] = subprocess.Popen([sys.executable, '-B', str(_resolve_model_artifact(HERE / 'run.py')), '--family', family], cwd=ROOT, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    while any((p.poll() is None for p in workers.values())):
        run('publish.py')
        failures = [f for f, p in workers.items() if p.poll() not in [None, 0]]
        if failures:
            save(HERE / 'worker_failures.json', dict(families=failures, other_workers_continue=True))
        time.sleep(20)
    assert all((p.returncode == 0 for p in workers.values()))
    run('verify.py')
    run('publish.py')
    subprocess.run(['node', str(HERE / 'verify_web.cjs')], cwd=ROOT, check=True)
    save(HERE / 'complete.json', dict(status='complete', runs=24, folds=6, extractors_executed=0))
    print('COMPLETE24', flush=True)
    for f in handles:
        f.close()
if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(HERE / 'error_pipeline.json', dict(error=traceback.format_exc()))
        run('publish.py')
        raise
