"""Launch once, after VF_S03 and completed labeling, with no overlap."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from datetime import datetime, timezone
import json, time, sys, subprocess, psutil
B = experiment_directory(__file__)
ROOT = B.parents[1]
PREVIOUS = B.parent / 'm1_ensemble_vs_single_vf_s03_20261013/state.json'
HUMAN = ROOT / 'data/active_learning_review/sessions/review_session_20261013_154318_e823d3.json'
last = None

def read(path):
    try:
        return json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def state(stage, **extra):
    global last
    stamp = {'stage': stage, **extra}
    if stamp == last:
        return
    last = stamp
    tmp = B / 'queue_state.tmp.json'
    tmp.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), **stamp}, indent=2), encoding='utf-8')
    tmp.replace(B / 'queue_state.json')

def active_inference():
    active = []
    for p in psutil.process_iter(['name', 'cmdline']):
        if (p.info['name'] or '').lower() not in ['python.exe', 'pythonw.exe']:
            continue
        script = next((v.replace('\\', '/').lower() for v in (p.info['cmdline'] or [])[1:] if v.endswith('.py')), '')
        if '/tesina/data/' in script and any((v in script.rsplit('/', 1)[-1] for v in ['infer', 'train_all', 'run_experiment'])):
            active.append(p.pid)
    return active
try:
    while True:
        previous = read(PREVIOUS)
        human = read(HUMAN)
        cs = human.get('candidates', [])
        ready = len(cs) == 172 and all((c.get('manual_labeling') and c.get('reviewed_at') and (c.get('final_label') in ['line', 'other', 'scrum']) for c in cs))
        if previous.get('stage') != 'complete':
            state('waiting_for_VF_S03', previous_stage=previous.get('stage'), human_labels_complete=ready)
        elif not ready:
            state('waiting_for_VF_S04_labels', reviewed=sum((c.get('final_label') in ['line', 'other', 'scrum'] for c in cs)))
        elif active_inference():
            state('waiting_for_other_thesis_inference_to_exit')
        else:
            break
        time.sleep(20)
    state('preparing_current_human_snapshot')
    with _resolve_model_artifact(B / 'logs/queue_launch.log').open('x', encoding='utf-8') as f:
        subprocess.run([sys.executable, '-B', str(_resolve_model_artifact(B / 'prepare_experiment.py'))], cwd=B, stdout=f, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, '-B', str(_resolve_model_artifact(B / 'launch.py'))], cwd=B, stdout=f, stderr=subprocess.STDOUT, check=True)
    state('launched', launcher=read(B / 'inputs/launcher.json'))
except Exception as e:
    state('failed', error=str(e))
    raise
