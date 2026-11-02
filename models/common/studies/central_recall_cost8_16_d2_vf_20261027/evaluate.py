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
from costs import partitions, decide, classification_counts
from analysis_math import metrics, changes
from model import build, keras, tf

def bootstrap(y, base, new, section, start, replicates=2000):
    keys = ['line_recall', 'scrum_recall', 'line_precision', 'scrum_precision', 'line_f1', 'scrum_f1', 'f1_macro']
    groups = []
    section = np.asarray(section)
    bins = np.floor(np.asarray(start) / 60).astype(int)
    for s in np.unique(section):
        groups.append([np.flatnonzero((section == s) & (bins == b)) for b in np.unique(bins[section == s])])
    rng = np.random.default_rng(20261027)
    vals = []
    for _ in range(replicates):
        ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g), size=len(g))]) for g in groups])
        if any(((y[ix] == c).sum() == 0 for c in range(3))):
            continue
        a = [metrics(y[ix], p[ix]) for p in base]
        b = [metrics(y[ix], p[ix]) for p in new]
        vals.append([np.mean([bb[k] - aa[k] for aa, bb in zip(a, b)]) for k in keys])
    vals = np.asarray(vals)
    a = [metrics(y, p) for p in base]
    b = [metrics(y, p) for p in new]
    return dict(method='paired fixed60s clusters within section; mean seed metrics; descriptive', requested=replicates, valid=len(vals), seed=20261027, metrics={k: dict(difference=float(np.mean([bb[k] - aa[k] for aa, bb in zip(a, b)])), ci95=np.quantile(vals[:, j], [0.025, 0.975]).tolist()) for j, k in enumerate(keys)})

def main():
    tc = load(HERE / 'training_complete.json')
    assert tc['all_saved_before_vf']
    for s in SEEDS:
        for c in COSTS:
            out = HERE / f'seed_{s}' / f'cost_{c}'
            assert digest(out / 'best.weights.h5') == load(out / 'selected.json')['checkpoint_sha256']
    rows = load(HERE / 'manifest.json')
    _, idx = partitions(rows)
    frame = pd.DataFrame([rows[i] for i in idx])
    y = np.array([LABELS.index(k) for k in frame.tag])
    f = _load_model_result_array(PARENT / 'features/multi.npy', mmap_mode='r')
    mask = _load_model_result_array(PARENT / 'features/masks.npy', mmap_mode='r')
    x = np.asarray(f[idx, :, 0, :])
    m = np.asarray(mask[idx])
    records = []
    preds = {k: [] for k in ['central', 'cost_8', 'cost_16']}
    timings = []
    save(HERE / 'evaluation_started.json', dict(started_unix=time.time(), all_final_checkpoints_saved=True))
    for seed in SEEDS:
        base = None
        for variant in preds:
            out = HERE / f'seed_{seed}' / variant
            out.mkdir(parents=True, exist_ok=True)
            checkpoint = PARENT / f'seed_{seed}/fixed/best.weights.h5' if variant == 'central' else out / 'best.weights.h5'
            keras.backend.clear_session()
            gc.collect()
            keras.utils.set_random_seed(seed)
            net, _, _ = build('fixed')
            net.load_weights(_resolve_model_artifact(checkpoint))

            @tf.function(input_signature=[tf.TensorSpec([None, 8, 2048], tf.float32), tf.TensorSpec([None, 8], tf.bool)])
            def infer(xx, mm):
                return net([xx, mm], training=False)
            t = time.perf_counter()
            p = np.concatenate([infer(x[i:i + 128], m[i:i + 128]).numpy() for i in range(0, len(idx), 128)])
            seconds = time.perf_counter() - t
            pred = decide(p)
            if variant == 'central':
                np.testing.assert_allclose(p, _load_model_result_array(PARENT / f'seed_{seed}/fixed/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
                base = pred
            df = frame.copy()
            df['y_true'] = df.tag
            df['y_final'] = [LABELS[k] for k in pred]
            for j, k in enumerate(LABELS):
                df['p_' + k] = p[:, j]
            df.to_csv(out / 'vf_predictions.csv', index=False)
            np.save(out / 'vf_probabilities.npy', p)
            mm = metrics(y, pred)
            delta = changes(y, base, pred)
            counts = classification_counts(y, pred)
            save(out / 'vf_metrics.json', dict(block=mm, counts=counts, changes=delta, inference_seconds=seconds, checkpoint_sha256=digest(checkpoint)))
            records.append(dict(seed=seed, variant=variant, **mm, corrected=delta['corrected'], broken=delta['broken'], **{k + '_' + t: v for k, cc in counts.items() for t, v in cc.items()}))
            preds[variant].append(pred)
            timings.append(dict(seed=seed, variant=variant, inference_seconds=seconds))
    pd.DataFrame(records).to_csv(HERE / 'results.csv', index=False)
    comp = {k: bootstrap(y, preds['central'], preds[k], frame.section, frame.start_s) for k in ['cost_8', 'cost_16']}
    save(HERE / 'paired_bootstrap.json', comp)
    save(HERE / 'timing.json', dict(training=load(HERE / 'selected_models.json'), inference=timings, cache_materialization_seconds=tc['cache_materialization_seconds'], resnet_cache_reused=True, not_fresh_end_to_end_benchmark=True))
    save(HERE / 'evaluation_complete.json', dict(complete=True, sets=9, vf_blocks=len(idx), means_of_seeds_not_ensemble=True))
if __name__ == '__main__':
    with lock('evaluation.lock'):
        main()
