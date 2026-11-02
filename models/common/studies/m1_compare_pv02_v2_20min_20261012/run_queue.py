"""Two new 20-minute runs, sequential; write only to this experiment."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, subprocess, sys, time
from datetime import datetime, timezone
B = experiment_directory(__file__)

def state(**values):
    (B / 'queue_state.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), **values}, indent=2), encoding='utf-8')
for name, args in [('RGB + YOLO', ['infer_full.py', '--key', 'M1-H2-D2']), ('Sólo YOLO', ['infer_pose.py'])]:
    state(status='running', model=name, completed=[] if name == 'RGB + YOLO' else ['RGB + YOLO'])
    with _resolve_model_artifact(B / 'logs' / ('full_console.log' if name == 'RGB + YOLO' else 'pose_console.log')).open('x', encoding='utf-8') as log:
        started = time.perf_counter()
        result = subprocess.run([sys.executable, '-B', '-u', str(B / args[0]), *args[1:]], cwd=B, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        state(status='failed', model=name, exit_code=result.returncode)
        raise SystemExit(result.returncode)
state(status='complete', completed=['RGB + YOLO', 'Sólo YOLO'])
