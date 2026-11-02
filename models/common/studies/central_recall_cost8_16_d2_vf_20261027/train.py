"""Six complete central temporal models; no VF inference during training."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import gc, time
import pandas as pd
from common import *
from costs import class_weights, partitions
from model import build, core_hash, keras

def main():
    rows = load(HERE / 'manifest.json')
    idx, test = partitions(rows)
    assert len(idx) == 8796 and len(test) == 688 and (len(set(idx) & set(test)) == 0)
    assert _load_model_result_array(PARENT / 'features/done.npy').all()
    f = _load_model_result_array(PARENT / 'features/multi.npy', mmap_mode='r')
    mask = _load_model_result_array(PARENT / 'features/masks.npy', mmap_mode='r')
    t = time.perf_counter()
    x = np.asarray(f[idx, :, 0, :])
    m = np.asarray(mask[idx])
    cache_s = time.perf_counter() - t
    y = np.array([LABELS.index(rows[i]['tag']) for i in idx])
    selected = []
    for seed in SEEDS:
        for cost in COSTS:
            out = HERE / f'seed_{seed}' / f'cost_{cost}'
            out.mkdir(parents=True, exist_ok=True)
            if _resolve_model_artifact(out / 'selected.json').exists():
                s = load(out / 'selected.json')
                assert digest(out / 'best.weights.h5') == s['checkpoint_sha256']
                selected.append(s)
                continue
            keras.backend.clear_session()
            gc.collect()
            keras.utils.set_random_seed(seed)
            net, core, _ = build('fixed')
            initial = core_hash(core)
            assert initial == load(PARENT / f'seed_{seed}/fixed/selected.json')['initial_core_sha256']
            w = class_weights(y, cost)
            cw = {i: float(w[i]) for i in range(3)}
            rng = np.random.default_rng(seed)
            history = []
            t = time.perf_counter()
            for epoch in range(1, 16):
                et = time.perf_counter()
                order = rng.permutation(len(idx))
                net.reset_metrics()
                for off in range(0, len(order), 32):
                    b = order[off:off + 32]
                    v = net.train_on_batch([x[b], m[b]], y[b], class_weight=cw, return_dict=True)
                row = dict(epoch=epoch, seconds=time.perf_counter() - et, **{k: float(vv) for k, vv in v.items()})
                history.append(row)
                pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
                save(HERE / 'status.json', dict(phase='training', seed=seed, cost=cost, epochs=15, **row))
                print(seed, cost, row, flush=True)
            net.save_weights(out / 'best.weights.h5')
            s = dict(seed=seed, cost=cost, epochs=15, training_examples=len(idx), class_weights=cw, initial_core_sha256=initial, checkpoint_sha256=digest(out / 'best.weights.h5'), seconds=time.perf_counter() - t, parameters=net.count_params(), selection='final epoch15 predeclared; no VF selection', saved_unix=time.time())
            save(out / 'selected.json', s)
            selected.append(s)
    assert len(selected) == 6
    save(HERE / 'selected_models.json', selected)
    save(HERE / 'training_complete.json', dict(complete=True, models=6, all_saved_before_vf=True, cache_materialization_seconds=cache_s, completed_unix=time.time()))
if __name__ == '__main__':
    with lock('training.lock'):
        main()
