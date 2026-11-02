"""Run the two authorized measurements in series; do not train or overwrite."""
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
import json, subprocess, sys
B = experiment_directory(__file__)

def state(stage, **extra):
    tmp = B / 'state.tmp.json'
    tmp.write_text(json.dumps({'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(), **extra}, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(B / 'state.json')

def run(stage, args):
    state(stage)
    with _resolve_model_artifact(B / 'logs' / f'{stage}.log').open('x', encoding='utf-8') as f:
        subprocess.run([sys.executable, '-B', '-u', str(B / args[0]), *args[1:]], cwd=B, stdout=f, stderr=subprocess.STDOUT, check=True)
try:
    run('inference_ensemble', ['infer.py', '--key', 'M1-ensemble'])
    run('inference_single', ['infer.py', '--key', 'M1-single'])
    run('generating_figures', ['make_images.py'])
    run('writing_report', ['report.py'])
    run('verifying_outputs', ['verify.py'])
    state('complete', report=str(B / 'README.md'), image=str(B / 'figures/comparacion_ensemble_modelo_unico_20min_v2.png'), visual_review_pending=True)
except Exception as e:
    state('failed', error=str(e))
    raise
