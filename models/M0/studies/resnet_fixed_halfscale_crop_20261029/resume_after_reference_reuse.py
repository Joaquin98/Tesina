"""Finish reference import asynchronously, then resume the authorized320 run."""
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
from datetime import datetime, timezone
import json, sys, time, subprocess, ctypes, os
import psutil
HERE = experiment_directory(__file__)
OUT = HERE / 'outputs'
copy_pid = int(sys.argv[1])
copy = psutil.Process(copy_pid)
copy_created = copy.create_time()
state = ctypes.windll.kernel32.SetThreadExecutionState
awake = bool(state(2147483648 | 1))

def write(name, data):
    p = OUT / name
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(data, indent=2), encoding='utf-8')
    t.replace(p)

def now():
    return datetime.now(timezone.utc).isoformat()
write('reference_resume_helper.json', dict(status='waiting_for_copy_and_integrity', pid=os.getpid(), copy_pid=copy_pid, copy_created=copy_created, awake_lease=awake, time_utc=now()))
try:
    while True:
        if _resolve_model_artifact(OUT / 'user_stop.json').exists():
            write('reference_resume_helper.json', dict(status='user_stopped', time_utc=now()))
            sys.exit(0)
        p = OUT / 'reference_maintenance.json'
        maintenance = json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig')) if _resolve_model_artifact(p).exists() else {}
        if maintenance.get('status') == 'reference_reuse_validated_ready_to_resume320':
            break
        try:
            alive = copy.is_running() and abs(copy.create_time() - copy_created) < 0.01
        except psutil.NoSuchProcess:
            alive = False
        if not alive:
            raise RuntimeError('Reference copy process ended before verified ready marker; inspect session/archive before retry')
        time.sleep(5)
    while True:
        if _resolve_model_artifact(OUT / 'user_stop.json').exists():
            write('reference_resume_helper.json', dict(status='user_stopped', time_utc=now()))
            sys.exit(0)
        busy = []
        for other in psutil.process_iter(['pid', 'cmdline', 'create_time']):
            try:
                line = ' '.join(other.info['cmdline'] or []).replace('\\', '/').lower()
                if 'm0_m1_cost_sampling_20261029/' in line and any(('/' + n + '.py' in line for n in ('train', 'evaluate', 'pipeline'))):
                    busy.append(dict(pid=other.pid, created=other.info['create_time'], command=other.info['cmdline']))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if not busy:
            break
        write('reference_resume_helper.json', dict(status='queued_for_other_training', pid=os.getpid(), other_processes=busy, awake_lease=awake, time_utc=now()))
        time.sleep(10)
    sys.path.insert(0, str(source_directory(HERE)))
    from models.M0.studies.resnet_fixed_halfscale_crop_20261029.experiment import guard
    guard()
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(process.info['cmdline'] or []).replace('\\', '/').lower()
            if 'resnet_fixed_halfscale_crop_20261029/experiment.py' in cmd:
                write('reference_resume_helper.json', dict(status='experiment_already_active_no_duplicate', active_pid=process.pid, time_utc=now()))
                sys.exit(0)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log = open(_resolve_model_artifact(OUT / ('controller_reuse_' + stamp + '.log')), 'ab', buffering=0)
    err = open(_resolve_model_artifact(OUT / ('controller_reuse_' + stamp + '.stderr.log')), 'ab', buffering=0)
    driver = subprocess.Popen([sys.executable, '-u', str(_resolve_model_artifact(HERE / 'experiment.py')), 'all'], cwd=HERE.parents[1], stdout=log, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    ml = open(_resolve_model_artifact(OUT / ('monitor_reuse_' + stamp + '.log')), 'ab', buffering=0)
    me = open(_resolve_model_artifact(OUT / ('monitor_reuse_' + stamp + '.stderr.log')), 'ab', buffering=0)
    monitor = subprocess.Popen([sys.executable, '-u', str(_resolve_model_artifact(HERE / 'job_monitor.py')), str(driver.pid)], cwd=HERE.parents[1], stdout=ml, stderr=me, creationflags=subprocess.CREATE_NO_WINDOW)
    write('reference_resume_launch.json', dict(status='resumed_with_equivalent224_reference', time_utc=now(), driver_pid=driver.pid, monitor_pid=monitor.pid, remaining_new_condition=320, old_experiment_untouched=True))
    write('reference_resume_helper.json', dict(status='resumed', driver_pid=driver.pid, monitor_pid=monitor.pid, time_utc=now()))
    subprocess.run([sys.executable, str(_resolve_model_artifact(HERE / 'report.py'))], cwd=HERE.parents[1], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
except Exception as error:
    write('reference_resume_failure.json', dict(status='failed', error=repr(error), time_utc=now()))
    raise
finally:
    state(2147483648)
