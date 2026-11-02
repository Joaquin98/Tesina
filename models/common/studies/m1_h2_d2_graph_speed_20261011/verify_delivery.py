"""Final read-only validation of prior inference references and new evidence."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import ast
import json
from pathlib import Path
import benchmark as isolation
HERE = isolation.HERE
analysis = json.loads(_resolve_model_artifact(HERE / 'results/analysis.json').read_text(encoding='utf-8'))
batch = json.loads(_resolve_model_artifact(HERE / 'results/continuous_batches.json').read_text(encoding='utf-8'))
expected = {}
for file in _model_artifact_glob(HERE / 'results', '*_summary.json', recursive=False):
    summary = json.loads(_resolve_model_artifact(file).read_text(encoding='utf-8'))
    assert summary['reference_content_unchanged']
    assert summary['graph_tracing_counts'] == [1] * 6
    expected.update(summary['hashes_after'])
current = {path: isolation.sha(Path(path)) for path in expected}
assert current == expected
assert all((e['all_pass'] for e in analysis['equivalence']))
assert len(analysis['equivalence']) == 10
assert all((v for check in analysis['additional_checks'] for k, v in check.items() if k not in ['video', 'repeat']))
assert all((('confidence' in d['path'] or 'probabilities' in d['path']) and d['absolute_difference'] <= 1e-06 for d in analysis['event_numeric_differences']))
assert all((c['allclose'] for c in batch['head_checks'] + batch['forest_checks']))
assert all((c['classes_exact'] and c['probabilities_allclose'] and c['events_close'] and c['event_classes_times_exact'] and c['review_events_close'] for c in batch['checks']))
for source in _model_artifact_glob(HERE, '*.py', recursive=False):
    ast.parse(_resolve_model_artifact(source).read_text(encoding='utf-8'))
result = {'current_inference_reference_hashes_match_recorded': True, 'reference_files': len(expected), 'paired_video_runs': 10, 'windows_per_mode': 84, 'all_equivalence_checks_pass': True, 'event_differences_only_confidence_and_probabilities': True, 'batch_combinations_verified': len(batch['checks']), 'all_scripts_syntax_valid': True, 'harness_metadata_note': 'The harness labels model_name using its API identifier and manually assigns the generic engine. Production load_model uses the display name and m1_h2_d2_cross_fold type. The production loader has no extra inference policy after detector construction, and analyze_video uses the same generic branch for both type strings. These descriptive metadata fields are outside prediction equivalence. No production file was edited.', 'current_reference_sha256': current}
isolation.save('results/final_validation.json', result)
print(json.dumps({k: v for k, v in result.items() if k != 'current_reference_sha256'}, indent=2))
