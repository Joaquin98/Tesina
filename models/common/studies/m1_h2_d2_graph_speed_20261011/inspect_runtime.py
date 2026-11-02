"""Read-only inspection of serialized GRUs and installed loop implementation."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json
from pathlib import Path
import zipfile
import benchmark as isolation
HERE, ROOT = (isolation.HERE, isolation.ROOT)
manifest = json.loads(_resolve_model_artifact(ROOT / 'data/final_model_comparison_20260827/outputs/web_export_m1_h2_d2/manifest.json').read_text(encoding='utf-8'))
models = []
for fold in manifest['folds']:
    with zipfile.ZipFile(ROOT / fold['dual_model']) as archive:
        config = json.loads(archive.read('config.json'))
    found = []

    def visit(value):
        if isinstance(value, dict):
            if value.get('class_name') == 'GRU':
                found.append({key: value['config'].get(key) for key in ['name', 'stateful', 'unroll']})
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(config)
    models.append({'fold': fold['fold'], 'grus': found})
    assert found and all((g['stateful'] is False and g['unroll'] is False for g in found))
site = Path('C:/Users/Joaquin/Downloads/sports-events-detection-main/sports-events-detection-main/.venv/Lib/site-packages')
references = {str(p): isolation.sha(p) for p in [_resolve_model_artifact(site / 'tensorflow/python/ops/while_loop.py'), _resolve_model_artifact(site / 'keras/src/backend/tensorflow/rnn.py')]}
global_keras = Path('C:/Users/Joaquin/.keras/keras.json')
isolated_keras = HERE / 'runtime/keras/keras.json'
result = {'models': models, 'runtime_source_sha256': references, 'tensorflow_eager_python_loop_lines': [482, 487, 488], 'keras_tf_while_loop_lines': [389, 428], 'keras_global_config_read_only': json.loads(_resolve_model_artifact(global_keras).read_text()), 'keras_isolated_config': json.loads(_resolve_model_artifact(isolated_keras).read_text())}
result['keras_config_values_equal'] = result['keras_global_config_read_only'] == result['keras_isolated_config']
isolation.save('results/runtime_inspection.json', result)
print(json.dumps(result, indent=2))
