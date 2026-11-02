import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import sys
from pathlib import Path
sys.path.insert(0, str(source_directory(experiment_directory(__file__).parents[0])))
from common import *
from model import build, core_hash, keras
from costs import class_weights, partitions, decide
import psutil
rows = load(HERE / 'manifest.json')
train, test = partitions(rows)
y = np.array([LABELS.index(rows[i]['tag']) for i in train])
chosen = np.array([train[np.flatnonzero(y == j)[0]] for j in range(3)])
f = _load_model_result_array(PARENT / 'features/multi.npy', mmap_mode='r')
m = _load_model_result_array(PARENT / 'features/masks.npy', mmap_mode='r')
x = np.asarray(f[chosen, :, 0])
mask = np.asarray(m[chosen])
for cost in COSTS:
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    net, core, _ = build('fixed')
    assert core_hash(core) == load(PARENT / 'seed_42/fixed/selected.json')['initial_core_sha256']
    start = core_hash(core)
    cw = {j: float(w) for j, w in enumerate(class_weights(y, cost))}
    net.train_on_batch([x, mask], np.arange(3), class_weight=cw)
    assert core_hash(core) != start
    p = net([x, mask], training=False).numpy()
    decide(p)
    path = HERE / 'analysis' / f'smoke_cost{cost}.weights.h5'
    net.save_weights(path)
    other, _, _ = build('fixed')
    other.load_weights(_resolve_model_artifact(path))
    np.testing.assert_allclose(p, other([x, mask], training=False), atol=1e-07)
procs = []
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmd = ' '.join(p.info['cmdline'] or [])
        if 'python' in (p.info['name'] or '').lower() and any((n in cmd for n in ['run_pipeline.py', 'train_d2.py', 'continue_when_ready.py'])):
            procs.append(p.info)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
save(HERE / 'analysis/smoke_result.json', dict(passed=True, costs=COSTS, train_only_rows=chosen.tolist(), save_reload=True, central_temporal_weights_update=True, initial_parent_match=True, active_related_processes=procs))
print('SMOKE PASSED', json.dumps(procs))
