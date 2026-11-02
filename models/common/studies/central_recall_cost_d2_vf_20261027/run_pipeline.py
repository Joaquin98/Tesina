import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import subprocess, sys, os, time, traceback
from common import *

def main():
    with lock('run.lock'):
        check()
        save(HERE / 'process.json', dict(pipeline_pid=os.getpid(), command=sys.argv, started_unix=time.time()))
        for script, marker in [('train.py', 'training_complete.json'), ('evaluate.py', 'evaluation_complete.json'), ('publish_verify.py', 'completion.json')]:
            if _resolve_model_artifact(HERE / marker).exists():
                continue
            p = subprocess.Popen([sys.executable, '-u', str(HERE / script)], cwd=str(ROOT))
            save(HERE / 'process.json', dict(pipeline_pid=os.getpid(), child_pid=p.pid, script=script, started_unix=time.time()))
            if p.wait() != 0:
                raise RuntimeError(script + ' failed')
        check()
        save(HERE / 'pipeline_complete.json', dict(complete=True, completed_unix=time.time()))
        save(HERE / 'status.json', dict(phase='complete'))
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(HERE / 'error.json', dict(error=repr(e), traceback=traceback.format_exc(), time=time.time()))
        raise
