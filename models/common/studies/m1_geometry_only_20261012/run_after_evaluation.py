"""Keep evaluation and full video timing sequential."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import subprocess, sys, time, json
HERE = experiment_directory(__file__)
while not _resolve_model_artifact(HERE / 'results/evaluation.json').exists():
    log = _resolve_model_artifact(HERE / 'logs/evaluation_console.log').read_text(encoding='utf-8', errors='replace')
    if 'Traceback (most recent call last)' in log:
        raise RuntimeError('Evaluation failed; inspect evaluation console')
    time.sleep(15)
with _resolve_model_artifact(HERE / 'logs/inference_console.log').open('x', encoding='utf-8') as log:
    result = subprocess.run([sys.executable, '-B', '-u', str(_resolve_model_artifact(HERE / 'infer.py'))], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise RuntimeError('Pose-only video run failed; inspect inference console')
print('POSE_ONLY_PIPELINE_COMPLETE', flush=True)
