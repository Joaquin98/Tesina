"""Export the exhaustive, verified threshold states for the local interactive page."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import csv, json, hashlib
HERE = experiment_directory(__file__)
source = HERE / 'continuous_sweep'
summary = json.loads(_resolve_model_artifact(source / 'summary.json').read_text(encoding='utf-8'))
with _resolve_model_artifact(source / 'all_threshold_states.csv').open(encoding='utf-8', newline='') as f:
    states = [{k: float(v) for k, v in r.items() if k != 'lower_inclusive'} for r in csv.DictReader(f)]
assert len(states) == summary['states'] == 657
assert states[0]['lower_exclusive'] == 0 and states[-1]['upper_inclusive'] == 1
assert all((a['upper_inclusive'] == b['lower_exclusive'] for a, b in zip(states, states[1:])))
payload = {'summary': summary, 'states': states, 'source_sha256': hashlib.sha256(_resolve_model_artifact(source / 'all_threshold_states.csv').read_bytes()).hexdigest()}
dest = HERE.parents[1] / 'src/web/static/data/m1-vf-threshold-sweep.json'
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(payload, separators=(',', ':'), ensure_ascii=False), encoding='utf-8')
print(dest)
