"""Start the existing app only if its local port is free; never restart it."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import socket, os, sys
root = experiment_directory(__file__).parents[1]
with socket.socket() as sock:
    if sock.connect_ex(('127.0.0.1', 5000)) == 0:
        print('Existing local server is running; leaving it untouched.', flush=True)
        sys.exit(0)
for k, v in {'TF_CPP_MIN_LOG_LEVEL': '2', 'OMP_NUM_THREADS': '2', 'TF_NUM_INTRAOP_THREADS': '2', 'TF_NUM_INTEROP_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}.items():
    os.environ.setdefault(k, v)
sys.path[:0] = [str(source_directory(root / 'src/web')), str(root)]
sys.dont_write_bytecode = True
from app import app
app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)
