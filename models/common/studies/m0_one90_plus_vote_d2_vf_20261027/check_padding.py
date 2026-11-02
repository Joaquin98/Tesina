import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import sys, json
from pathlib import Path
import numpy as np
P = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/central_recall_cost_d2_vf_20261027')
D = Path('D:/Tesina/entregas/repositorio_tesina_20261022/data/multicrop_d2_stride25_vf_20261026')
H = experiment_directory(__file__)
sys.path.insert(0, str(source_directory(P)))
from model import build, keras, tf
rows = json.loads(_resolve_model_artifact(D / 'manifest.json').read_text())
idx = np.array([i for i, r in enumerate(rows) if r['domain'] == 'VF'])
assert len(idx) == 688
features = _load_model_result_array(D / 'features/multi.npy', mmap_mode='r')
masks = _load_model_result_array(D / 'features/masks.npy', mmap_mode='r')
ix = np.r_[idx[:12], idx[-8:]]
x = np.asarray(features[ix])
m = np.asarray(masks[ix])
assert x.shape == (20, 8, 9, 2048) and (m[:, 7] == 0).all()
xs = np.transpose(x, (0, 2, 1, 3)).reshape(-1, 8, 2048)
ms = np.repeat(m, 9, axis=0)
records = []
for seed in [17, 42, 2026]:
    model, core, _ = build('fixed')
    model.load_weights(_resolve_model_artifact(P / f'seed_{seed}/cost_4/best.weights.h5'))
    f = keras.Input((7, 2048))
    mask = keras.Input((7,), dtype='bool')
    z = core.get_layer('bigru1')(f, mask=mask)
    z = core.get_layer('bigru2')(z, mask=mask)
    z = core.get_layer('dropout')(z)
    z = core.get_layer('dense')(z)
    z = core.get_layer('classes')(z)
    short = keras.Model([f, mask], z)
    long = np.concatenate([model([xs[i:i + 36], ms[i:i + 36]], training=False).numpy() for i in range(0, len(xs), 36)])
    reduced = np.concatenate([short([xs[i:i + 36, :7], ms[i:i + 36, :7]], training=False).numpy() for i in range(0, len(xs), 36)])
    np.testing.assert_allclose(long, reduced, rtol=1e-05, atol=1e-06)
    assert np.array_equal(long.argmax(1), reduced.argmax(1))
    records.append({'seed': seed, 'clips': len(ix), 'views': 9, 'max_abs_difference': float(np.max(np.abs(long - reduced))), 'identical_predictions': True})
(H / 'padding_equivalence_sample.json').write_text(json.dumps({'results': records, 'scope': '20 VF clips including partial sections, all nine crops, inference only; no retraining'}, indent=2))
print(json.dumps(records, indent=2))
