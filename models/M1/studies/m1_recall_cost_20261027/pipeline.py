import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, os, subprocess, sys, time, msvcrt, traceback
HERE = experiment_directory(__file__)
with _resolve_model_artifact(HERE / 'pipeline.lock').open('a+b') as lock:
    lock.seek(0)
    lock.write(b'0')
    lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    (HERE / 'process.json').write_text(json.dumps(dict(pid=os.getpid(), started_unix=time.time(), stages=['train', 'evaluate', 'publish']), indent=2))
    try:
        for arguments in [('run.py', 'train'), ('run.py', 'evaluate'), ('publish.py',)]:
            completed = subprocess.run([sys.executable, '-B', '-u', str(HERE / arguments[0]), *arguments[1:]], cwd=HERE.parents[1], check=True)
        (HERE / 'pipeline_complete.json').write_text(json.dumps(dict(status='complete', finished_unix=time.time()), indent=2))
    except Exception:
        (HERE / 'pipeline_error.json').write_text(json.dumps(dict(status='error', traceback=traceback.format_exc(), unix=time.time()), indent=2))
        raise
