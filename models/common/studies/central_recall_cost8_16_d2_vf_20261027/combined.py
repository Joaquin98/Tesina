"""Verify all five costs and add descriptive comparisons to the report."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from common import *
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
OLD = ROOT / 'data/central_recall_cost_d2_vf_20261027'
VARIANTS = ['central', 'cost_2', 'cost_4', 'cost_8', 'cost_16']

def combine(report):
    from evaluate import bootstrap
    check()
    tables = {str(root): pd.read_csv(_resolve_model_artifact(root / 'results.csv')) for root in [OLD, HERE]}
    frame = None
    preds = {v: [] for v in VARIANTS}
    records = []
    for seed in SEEDS:
        for variant in VARIANTS:
            root = OLD if variant in ['cost_2', 'cost_4'] else HERE
            out = root / f'seed_{seed}' / variant
            df = pd.read_csv(_resolve_model_artifact(out / 'vf_predictions.csv'))
            p = _load_model_result_array(out / 'vf_probabilities.npy')
            if frame is None:
                frame = df
            else:
                for col in ['clip_id', 'y_true', 'section', 'start_s', 'end_s']:
                    assert df[col].tolist() == frame[col].tolist(), col
            assert len(df) == 688 and df.clip_id.is_unique
            y = np.array([LABELS.index(k) for k in df.y_true])
            pred = p.argmax(1)
            np.testing.assert_allclose(p, df[['p_' + k for k in LABELS]], atol=1e-07)
            assert df.y_final.tolist() == [LABELS[k] for k in pred]
            pr, re, f, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
            mm = dict(f1_macro=float(f.mean()), accuracy=float(accuracy_score(y, pred)))
            cm = confusion_matrix(y, pred, labels=[0, 1, 2])
            for j, k in enumerate(LABELS):
                mm.update({k + '_precision': float(pr[j]), k + '_recall': float(re[j]), k + '_f1': float(f[j]), k + '_tp': int(cm[j, j]), k + '_fn': int(support[j] - cm[j, j]), k + '_fp': int(cm[:, j].sum() - cm[j, j])})
            source = tables[str(root)]
            r = source[(source.seed == seed) & (source.variant == variant)].iloc[0]
            for k, v in mm.items():
                np.testing.assert_allclose(r[k], v, atol=1e-12)
            if variant == 'central':
                np.testing.assert_allclose(p, _load_model_result_array(OLD / f'seed_{seed}/central/vf_probabilities.npy'), atol=1e-06, rtol=1e-05)
            preds[variant].append(pred)
            records.append(dict(seed=seed, variant=variant, **mm))
    results = pd.DataFrame(records)
    results.to_csv(HERE / 'combined_results.csv', index=False)
    comp = {v + '_vs_cost_4': bootstrap(y, preds['cost_4'], preds[v], frame.section, frame.start_s) for v in ['cost_8', 'cost_16']}
    sections = frame.section.to_numpy()
    bins = (frame.start_s.to_numpy() // 60).astype(int)
    groups = [[np.flatnonzero((sections == s) & (bins == b)) for b in np.unique(bins[sections == s])] for s in np.unique(sections)]
    for variant in ['cost_8', 'cost_16']:
        rng = np.random.default_rng(20261027)
        delta = []
        for _ in range(2000):
            ii = np.concatenate([np.concatenate([g[j] for j in rng.integers(len(g), size=len(g))]) for g in groups])
            if any((np.sum(y[ii] == j) == 0 for j in range(3))):
                continue
            delta.append([np.mean([np.mean(n[ii[y[ii] == j]] == j) - np.mean(b[ii[y[ii] == j]] == j) for n, b in zip(preds[variant], preds['cost_4'])]) for j in [0, 2]])
        for j, k in enumerate(['line', 'scrum']):
            np.testing.assert_allclose(np.quantile(np.asarray(delta)[:, j], [0.025, 0.975]), comp[variant + '_vs_cost_4']['metrics'][k + '_recall']['ci95'], atol=1e-12)
    save(HERE / 'combined_bootstrap.json', comp)
    means = results.groupby('variant').mean(numeric_only=True).reindex(VARIANTS)
    keys = ['line_precision', 'line_recall', 'line_f1', 'scrum_precision', 'scrum_recall', 'scrum_f1', 'f1_macro', 'accuracy']
    counts = ['line_tp', 'line_fn', 'line_fp', 'scrum_tp', 'scrum_fn', 'scrum_fp']
    ci = [dict(comparison=c, metric=k, delta_pp=100 * v['difference'], lower_pp=100 * v['ci95'][0], upper_pp=100 * v['ci95'][1]) for c, metrics in comp.items() for k, v in metrics['metrics'].items()]
    fragment = '<h2>Comparacion completa: costos 1 / 2 / 4 / 8 / 16</h2><p>Central equivale a costo1. Promedio de tres ejecuciones separadas; no ensamble. Costos8/16 son una extension decidida despues de ver VF anterior: comparaciones exploratorias, sin garantia de generalizacion.</p>'
    fragment += (100 * means[keys]).to_html(float_format=lambda x: f'{x:.2f}')
    fragment += '<h3>TP / FN / FP medios por ejecucion</h3><p>Denominadores: 68 bloques line, 39 scrum. Precision media por semilla; no cociente de los conteos medios.</p>' + means[counts].to_html(float_format=lambda x: f'{x:.2f}')
    fragment += '<h3>Diferencias frente a costo4: IC95 descriptivos</h3>' + pd.DataFrame(ci).to_html(index=False, float_format=lambda x: f'{x:.2f}')
    perseed = results[['seed', 'variant'] + keys].copy()
    perseed[keys] *= 100
    fragment += '<h3>Cinco costos por semilla (%)</h3>' + perseed.to_html(index=False, float_format=lambda x: f'{x:.2f}')
    fragment += '<h3>Conteos por semilla</h3>' + results[['seed', 'variant'] + counts].to_html(index=False)
    path = report / 'index.html'
    doc = _resolve_model_artifact(path).read_text(encoding='utf-8')
    path.write_text(doc.replace('</html>', fragment + '</html>'), encoding='utf-8')
    save(HERE / 'combined_verification.json', dict(passed=True, prediction_sets=15, blocks=688, metrics_recomputed_sklearn=True, same_order_labels_boundaries=True, central_probabilities_reproduced=True, recall_ci_independent=True, mean_seeds_not_ensemble=True, adaptive_exploratory=True))
