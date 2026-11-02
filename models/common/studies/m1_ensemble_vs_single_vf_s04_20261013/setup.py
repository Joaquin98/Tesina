"""Prepare scripts for VF_S04 without starting a concurrent inference."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import ast, json
from datetime import datetime, timezone
B = experiment_directory(__file__)
PREVIOUS = B.parent / 'm1_ensemble_vs_single_vf_s02_20261013'
for name in ['prepare_experiment.py', 'run_experiment.py', 'report.py', 'launch.py', 'check_delivery.py']:
    text = _resolve_model_artifact(PREVIOUS / name).read_text(encoding='utf-8').replace('VF_S02', 'VF_S04').replace('vf_s02', 'vf_s04').replace('20261013_102705_9a03b2', '20261013_154318_e823d3').replace("'section':'S02'", "'section':'S04'")
    ast.parse(text, filename=name)
    with _resolve_model_artifact(B / name).open('x', encoding='utf-8') as f:
        f.write(text)
(B / 'inputs').mkdir(exist_ok=True)
(B / 'logs').mkdir(exist_ok=True)
request = {'utc': datetime.now(timezone.utc).isoformat(), 'authorization': 'Ground truth / six-fold M1 ensemble / single M1 comparison on final segment 4.', 'human_session': '20261013_154318_e823d3', 'previous_campaign': 'm1_ensemble_vs_single_vf_s03_20261013', 'sequential': True, 'snapshot_labels_when_launching': True, 'retrain': False, 'thresholds_locked': True, 'fresh_video_each_model': True, 'no_slide_or_thesis_changes': True}
with _resolve_model_artifact(B / 'inputs/queue_request.json').open('x', encoding='utf-8') as f:
    json.dump(request, f, indent=2)
print('VF_S04 scripts prepared; must wait for VF_S03 and complete manual labels.')
