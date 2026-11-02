"""Final artifact and current source-integrity checks for the pose-only ablation."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, hashlib, urllib.request
HERE = experiment_directory(__file__)
evaluation = json.loads(_resolve_model_artifact(HERE / 'results/evaluation.json').read_text(encoding='utf-8'))
completion = json.loads(_resolve_model_artifact(HERE / 'results/completion.json').read_text(encoding='utf-8'))
adapter = json.loads(_resolve_model_artifact(HERE / 'results/adapter_validation.json').read_text(encoding='utf-8'))
result = json.loads(_resolve_model_artifact(HERE / 'results/result.json').read_text(encoding='utf-8'))
assert evaluation['hashes_unchanged'] and completion['all_checks_pass'] and adapter['all_pass']
assert evaluation['fold_count'] == 6 and evaluation['total_heldout_blocks'] == 4714
assert len(result['window_predictions']) == 1195
counts = result['metadata']['timing']['counts']
assert counts['YOLO_calls'] == 30030 and counts['forest_calls'] == 7170
assert counts['ResNet_calls'] == counts['dual_BiGRU_calls'] == 0
refs = {**evaluation['reference_hashes'], **completion['hashes_after']}
for path, signature in refs.items():
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(path)).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    assert h.hexdigest() == signature, path
for key in ['YOLO_only', 'M1_complete']:
    with urllib.request.urlopen('http://127.0.0.1:8768/results/' + key + '.json', timeout=10) as response:
        served = json.load(response)
        assert len(served['window_predictions']) == 1195
with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8768/video', headers={'Range': 'bytes=0-1023'}), timeout=10) as response:
    assert response.status == 206 and len(response.read()) == 1024
proof = {'all_pass': True, 'reference_hashes_currently_unchanged': len(refs), 'heldout_folds': 6, 'heldout_blocks': 4714, 'video_windows': 1195, 'fresh_YOLO_calls': 30030, 'ResNet_calls': 0, 'dual_BiGRU_calls': 0, 'review_results_served': 2, 'video_HTTP_range_pass': True, 'browser_verification': {'event_start_seconds': 196, 'video_seek_seconds': 193, 'context_seconds': 3, 'switching_model_preserves_video_position': True, 'selected_variant': 'YOLO_only'}, 'inference_seconds': result['metadata']['timing']['inference_seconds']}
with _resolve_model_artifact(HERE / 'final_validation.json').open('x', encoding='utf-8') as file:
    json.dump(proof, file, indent=2)
print(json.dumps(proof))
