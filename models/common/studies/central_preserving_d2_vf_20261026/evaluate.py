"""All fixed heads must exist before any VF comparison is computed."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import sys, time
import pandas as pd
from common import *
from analysis_math import split_indices, metrics, changes, success_rule, temporal_bootstrap
from train import STRATEGIES, predict
from model import tf
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import temporal_metrics

def main():
    check_frozen()
    assert load(HERE / 'training_complete.json')['all_saved_before_vf']
    for seed in SEEDS:
        for strategy in STRATEGIES:
            q = HERE / f'seed_{seed}' / strategy
            assert digest(q / 'best.weights.h5') == load(q / 'selected.json')['checkpoint_sha256']
    rows = load(HERE / 'manifest.json')
    _, _, idx = split_indices(rows)
    assert len(idx) == 688
    frame = pd.DataFrame([rows[i] for i in idx])
    y = np.array([LABELS.index(r) for r in frame.tag])
    records = []
    preds = {k: [] for k in ['central'] + STRATEGIES}
    timings = []
    for seed in SEEDS:
        cache = HERE / f'embeddings/seed_{seed}'
        base = tf.nn.softmax(_load_model_result_array(cache / 'logits.npy', mmap_mode='r')[idx, 0]).numpy()
        np.testing.assert_allclose(base, _load_model_result_array(CACHE / f'seed_{seed}/fixed/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
        for strategy in ['central'] + STRATEGIES:
            out = HERE / f'seed_{seed}' / strategy
            out.mkdir(parents=True, exist_ok=True)
            t = time.perf_counter()
            if strategy == 'central':
                p = base
                attention = np.zeros((len(idx), 9), dtype='float32')
                attention[:, 0] = 1
            else:
                p, attention = predict(strategy, out, cache, idx)
            infer_s = time.perf_counter() - t
            assert p.shape == (688, 3) and np.isfinite(p).all() and (p >= 0).all()
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-06)
            assert attention.shape == (688, 9) and np.isfinite(attention).all() and (attention >= 0).all()
            np.testing.assert_allclose(attention.sum(1), 1, atol=1e-06)
            if strategy == 'late_mix':
                assert (attention[:, 0] >= 0.5).all()
            prediction = p.argmax(1)
            df = frame.copy()
            df['y_true'] = df.tag
            df['y_final'] = [LABELS[i] for i in prediction]
            for j, k in enumerate(LABELS):
                df['p_' + k] = p[:, j]
            df.to_csv(out / 'vf_predictions.csv', index=False)
            np.save(out / 'vf_probabilities.npy', p)
            np.save(out / 'vf_attention.npy', attention)
            values = metrics(y, prediction)
            delta = changes(y, base.argmax(1), prediction)
            events = temporal_metrics(df)
            save(out / 'vf_metrics.json', dict(block=values, changes=delta, temporal=events, inference_from_embeddings_seconds=infer_s, rule='argmax without thresholds'))
            records.append(dict(seed=seed, strategy=strategy, **values, corrected=delta['corrected'], broken=delta['broken']))
            preds[strategy].append(prediction)
            timings.append(dict(seed=seed, strategy=strategy, inference_from_embeddings_seconds=infer_s))
    results = pd.DataFrame(records)
    results.to_csv(HERE / 'results.csv', index=False)
    means = results.groupby('strategy').mean(numeric_only=True)
    comparisons = {}
    for strategy in STRATEGIES:
        b = paired_bootstrap(y, preds['central'], preds[strategy], replicates=2000, seed=20261026)
        temporal = temporal_bootstrap(y, preds['central'], preds[strategy], frame.section.to_numpy(), frame.start_s.to_numpy())
        improvements = [metrics(y, n)['f1_macro'] > metrics(y, a)['f1_macro'] for a, n in zip(preds['central'], preds[strategy])]
        evidence = success_rule(means.loc['central'].to_dict(), means.loc[strategy].to_dict(), b['ci95'], improvements, temporal['ci95'])
        comparisons[strategy] = dict(stratified_bootstrap=b, temporal_bootstrap=temporal, evidence_rule=evidence)
    save(HERE / 'comparisons.json', dict(comparisons=comparisons, exploratory=True, multiple_strategies=3, independent_test=False, mean_seeds_not_ensemble=True))
    timing = dict(pilot_central=load(HERE / 'pilot/central/selected.json')['seconds'], pilot_heads={k: load(HERE / 'pilot' / k / 'selected.json')['seconds'] for k in STRATEGIES}, final_heads=[dict(seed=s, strategy=k, seconds=load(HERE / f'seed_{s}' / k / 'selected.json')['seconds']) for s in SEEDS for k in STRATEGIES], embeddings={k: load(HERE / 'embeddings' / k / 'timing.json') for k in ['pilot'] + [f'seed_{s}' for s in SEEDS]}, inference=timings, cache_reused=9484, not_fresh_benchmark=True)
    save(HERE / 'timing.json', timing)
    save(HERE / 'evaluated.json', dict(complete=True, prediction_sets=12, vf_blocks=688))
if __name__ == '__main__':
    main()
