"""Compare the six-fold M1 ensemble with the all-development M1 model."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import time, json, subprocess, sys
from datetime import datetime, timezone
B = experiment_directory(__file__)
ROOT = B.parents[1]
previous = ROOT / 'data/m1_compare_pv02_v2_20min_20261012/delivery_state.json'

def state(stage, **extra):
    (B / 'state.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **extra}, indent=2), encoding='utf-8')

def run(name, args):
    state(name)
    with _resolve_model_artifact(B / 'logs' / f'{name}.log').open('x', encoding='utf-8') as f:
        subprocess.run([sys.executable, '-B', '-u', str(B / args[0]), *args[1:]], cwd=B, stdout=f, stderr=subprocess.STDOUT, check=True)
state('queued_until_current_comparison_and_slides_complete')
while True:
    try:
        d = json.loads(_resolve_model_artifact(previous).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        d = {}
    if d.get('stage') in ['delivery_failed', 'inference_failed']:
        state('waiting_for_current_task_repair', previous=d)
    if d.get('stage') == 'complete':
        break
    time.sleep(20)
try:
    run('training_all_development', ['train_all.py'])
    run('inference_ensemble', ['infer.py', '--key', 'M1-ensemble'])
    run('inference_single', ['infer.py', '--key', 'M1-single'])
    run('generating_figures', ['make_images.py'])
    run('writing_report', ['report.py'])
    run('verifying_outputs', ['verify.py'])
    state('complete', report=str(B / 'README.md'), image=str(B / 'figures/comparacion_ensemble_modelo_unico_20min.png'))
except Exception as error:
    state('failed', error=str(error))
    raise
