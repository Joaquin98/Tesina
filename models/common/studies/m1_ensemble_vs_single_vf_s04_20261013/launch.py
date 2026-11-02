"""Launch once, after confirming no other thesis inference is active."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import psutil, json, subprocess, sys
from datetime import datetime, timezone
B = experiment_directory(__file__)
active = []
for p in psutil.process_iter(['name', 'cmdline']):
    cmd = p.info['cmdline'] or []
    if (p.info['name'] or '').lower() not in ['python.exe', 'pythonw.exe']:
        continue
    script = next((v.replace('\\', '/').lower() for v in cmd[1:] if v.endswith('.py')), '')
    if '/tesina/data/' in script and any((v in script.rsplit('/', 1)[-1] for v in ['infer', 'train_all', 'run_experiment'])):
        active.append({'pid': p.pid, 'script': script})
assert not active, active
conditions = {'utc': datetime.now(timezone.utc).isoformat(), 'cpu_percent_one_second': psutil.cpu_percent(interval=1), 'logical_cpus': psutil.cpu_count(), 'other_thesis_inference_processes': active, 'cpu_reserved': False}
with _resolve_model_artifact(B / 'inputs/launch_conditions.json').open('x', encoding='utf-8') as f:
    json.dump(conditions, f, indent=2)
out = _resolve_model_artifact(B / 'logs/launcher.log').open('x', encoding='utf-8')
p = subprocess.Popen([sys.executable, '-B', '-u', str(_resolve_model_artifact(B / 'run_experiment.py'))], cwd=B, stdout=out, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
with _resolve_model_artifact(B / 'inputs/launcher.json').open('x', encoding='utf-8') as f:
    json.dump({'pid': p.pid, 'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable, 'script': str(_resolve_model_artifact(B / 'run_experiment.py'))}, f, indent=2)
print(json.dumps({'pid': p.pid, 'cpu_percent_before': conditions['cpu_percent_one_second'], 'sequential_inference_started': True}))
