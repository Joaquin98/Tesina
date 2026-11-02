"""Replicate the verified VF_S02 campaign using completed VF_S03 labels."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import ast, subprocess, sys
B = experiment_directory(__file__)
PREVIOUS = B.parent / 'm1_ensemble_vs_single_vf_s02_20261013'
for name in ['prepare_experiment.py', 'run_experiment.py', 'report.py', 'launch.py', 'check_delivery.py']:
    text = _resolve_model_artifact(PREVIOUS / name).read_text(encoding='utf-8').replace('VF_S02', 'VF_S03').replace('vf_s02', 'vf_s03').replace('20261013_102705_9a03b2', '20261013_110635_bf83de')
    ast.parse(text, filename=name)
    with _resolve_model_artifact(B / name).open('x', encoding='utf-8') as f:
        f.write(text)
subprocess.run([sys.executable, '-B', str(_resolve_model_artifact(B / 'prepare_experiment.py'))], cwd=B, check=True)
subprocess.run([sys.executable, '-B', str(_resolve_model_artifact(B / 'launch.py'))], cwd=B, check=True)
