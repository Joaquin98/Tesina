import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import subprocess, sys, json
from datetime import datetime, timezone
B = experiment_directory(__file__)

def state(**values):
    (B / 'queue_state.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), **values}, indent=2))
complete = json.loads(_resolve_model_artifact(B / 'results/M1-H2-D2_completion.json').read_text())
assert complete['all_checks_pass']
state(status='running', model='Sólo YOLO', completed=['RGB + YOLO'], repair='missing pathlib import before video analysis; full timing preserved')
with _resolve_model_artifact(B / 'logs/pose_retry_002.log').open('x', encoding='utf-8') as log:
    r = subprocess.run([sys.executable, '-B', '-u', str(_resolve_model_artifact(B / 'infer_pose.py'))], cwd=B, stdout=log, stderr=subprocess.STDOUT)
if r.returncode:
    state(status='failed', model='Sólo YOLO', exit_code=r.returncode)
    raise SystemExit(r.returncode)
state(status='complete', completed=['RGB + YOLO', 'Sólo YOLO'])
