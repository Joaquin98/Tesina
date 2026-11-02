import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import subprocess, sys
from pathlib import Path
h = experiment_directory(__file__)
for name in ['run.py', 'verify_compare.py']:
    r = subprocess.run([sys.executable, '-u', str(h / name)], cwd='D:/Tesina/entregas/repositorio_tesina_20261022')
    if r.returncode:
        raise SystemExit(r.returncode)
