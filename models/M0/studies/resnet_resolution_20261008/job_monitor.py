"""Process-scoped awake lease and final report hook for the active experiment.
No power plan is changed. Explicit user sleep/shutdown remains available.
The lease expires when the experiment driver exits, even on failure.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import ctypes, json, subprocess, sys, time
from pathlib import Path
import psutil
HERE = experiment_directory(__file__)
OUT = HERE / 'outputs'
PID = int(sys.argv[1])
proc = psutil.Process(PID)
created = proc.create_time()
state = ctypes.windll.kernel32.SetThreadExecutionState
ES_CONTINUOUS = 2147483648
ES_SYSTEM_REQUIRED = 1
awake = bool(state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
(OUT / 'job_monitor_start.json').write_text(json.dumps(dict(driver_pid=PID, created=created, awake_lease=awake, power_plan_changed=False)), encoding='utf-8')
try:
    while True:
        try:
            if not proc.is_running() or proc.create_time() != created:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(30)
finally:
    state(ES_CONTINUOUS)
completed = _resolve_model_artifact(OUT / 'execution_complete.json').exists()
(OUT / 'job_monitor_exit.json').write_text(json.dumps(dict(status='complete' if completed else 'interrupted_or_failed', time_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(), awake_lease_released=True)), encoding='utf-8')
subprocess.run([sys.executable, str(_resolve_model_artifact(HERE / 'report.py'))], cwd=HERE.parents[1], check=True)
