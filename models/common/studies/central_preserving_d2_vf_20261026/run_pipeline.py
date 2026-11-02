import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os, sys, time, subprocess
from common import *

def main():
    with writer_lock('run.lock'):
        hashes = {p.name: digest(p) for p in _model_artifact_glob(HERE, '*.py', recursive=False)}
        hashes.update({n: digest(HERE / n) for n in ['protocol.json', 'manifest.json']})
        if _resolve_model_artifact(HERE / 'implementation_hashes.json').exists():
            assert load(HERE / 'implementation_hashes.json') == hashes
        else:
            save(HERE / 'implementation_hashes.json', hashes)
        check_frozen()
        save(HERE / 'run.json', dict(pid=os.getpid(), python=sys.executable, started_unix=time.time()))
        with _resolve_model_artifact(HERE / 'run.log').open('a', encoding='utf8', buffering=1) as log:
            for script, marker in [('prepare.py', 'prepared.json'), ('train.py', 'training_complete.json'), ('evaluate.py', 'evaluated.json'), ('publish.py', 'publication.json'), ('verify.py', 'completion.json')]:
                if _resolve_model_artifact(HERE / marker).exists():
                    continue
                save(HERE / 'status.json', dict(phase=script.removesuffix('.py')))
                child = subprocess.Popen([sys.executable, '-u', str(HERE / script)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                save(HERE / 'process.json', dict(pipeline_pid=os.getpid(), child_pid=child.pid, script=script, started_unix=time.time()))
                code = child.wait()
                if code:
                    save(HERE / 'status.json', dict(phase='failed', script=script, exit_code=code))
                    raise RuntimeError(script + ' failed; preserve files and inspect run.log')
        save(HERE / 'pipeline_complete.json', dict(complete=True))
if __name__ == '__main__':
    main()
