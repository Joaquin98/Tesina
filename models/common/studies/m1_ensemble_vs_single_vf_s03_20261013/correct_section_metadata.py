"""Correct the isolated quality CSV section before its pending evaluation."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import ast
B = experiment_directory(__file__)
assert not _resolve_model_artifact(B / 'results/M1-ensemble_quality.json').exists()
for name in ['prepare_experiment.py', 'quality_from_fresh_features.py']:
    p = B / name
    original = _resolve_model_artifact(p).read_text(encoding='utf-8')
    corrected = original.replace("'section':'S02'", "'section':'S03'")
    assert corrected != original
    ast.parse(corrected, filename=name)
    with _resolve_model_artifact(B / 'inputs' / f'{name}.before_section_metadata_fix').open('x', encoding='utf-8') as f:
        f.write(original)
    p.write_text(corrected, encoding='utf-8')
print('VF_S03 quality CSV section metadata corrected before evaluation; probabilities and timing route unchanged.')
