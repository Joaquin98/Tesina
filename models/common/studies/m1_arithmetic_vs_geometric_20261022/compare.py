"""Compare frozen branch fusion with the original alpha on each fold test.

Fixed decision rules isolate the fusion change. Grid rankings are descriptive
test comparisons, not new independent validation or development selection.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json
sys.path.insert(0, str(source_directory(experiment_directory(__file__).parents[1])))
from models.common.studies.m1_argmax_grid_20261022.run import np, pd, m, Evaluator, decide, GRID, RUNS, LABELS, sha
from sklearn.metrics import f1_score
HERE = experiment_directory(__file__)
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
rows = []
fixed = []
hashes = {}
reconstruction = []
for i in range(1, 7):
    fold = f'PV{i:02d}'
    folder = RUNS / f'PV-M1-H2-D2-{fold}-S42' / 'attempt_001'
    for filename in ['run.json', 'predictions.csv']:
        path = folder / filename
        hashes[str(path)] = sha(path)
    run = json.loads(_resolve_model_artifact(folder / 'run.json').read_text())
    f = pd.read_csv(_resolve_model_artifact(folder / 'predictions.csv'))
    f['group_id'] = f.test_group
    alpha = float(run['model_metadata']['geometry_alpha'])
    dual = f[['p_dual_' + c for c in LABELS]].to_numpy()
    geom = f[['p_geometry_' + c for c in LABELS]].to_numpy()
    assert np.isfinite(dual).all() and np.isfinite(geom).all()
    assert np.allclose(dual.sum(1), 1) and np.allclose(geom.sum(1), 1)
    reconstructed = m.normalize_probabilities(np.exp((1 - alpha) * np.log(np.clip(dual, 1e-07, 1)) + alpha * np.log(np.clip(geom, 1e-07, 1))))
    stored = f[['p_' + c for c in LABELS]].to_numpy()
    error = float(np.abs(reconstructed - stored).max())
    assert np.allclose(reconstructed, stored, atol=1e-12, rtol=1e-10), (fold, error)
    reconstruction.append(dict(fold=fold, alpha=alpha, max_absolute_error=error))
    arithmetic = m.normalize_probabilities((1 - alpha) * dual + alpha * geom)
    rules = [('argmax', 'argmax', 'argmax'), ('line_argmax_scrum_035', 'argmax', 0.35), ('line_050_scrum_035', 0.5, 0.35), ('original_fold_thresholds', run['thresholds']['line'], run['thresholds']['scrum'])]
    for fusion, p in [('geometric', stored), ('arithmetic', arithmetic)]:
        frame = f.copy()
        for j, c in enumerate(LABELS):
            frame['p_' + c] = p[:, j]
        e = Evaluator(frame)
        for policy, line, scrum in rules:
            y = decide(e.p, line, scrum)
            e.verify(y)
            fixed.append(dict(fold=fold, test_group=run['test_group'], alpha=alpha, fusion=fusion, policy=policy, line_rule=line, scrum_rule=scrum, block_macro_f1=f1_score(e.ytrue, y, labels=[0, 1, 2], average='macro', zero_division=0), **e.evaluate(y)))
        for line in GRID:
            for scrum in GRID:
                rows.append(dict(fold=fold, test_group=run['test_group'], alpha=alpha, fusion=fusion, line_rule=line, scrum_rule=scrum, **e.evaluate(decide(e.p, line, scrum))))
    print(f'{fold}: both fusions evaluated; alpha={alpha:.2f}; reconstructed error={error:.3g}', flush=True)
fixed = pd.DataFrame(fixed)
grid = pd.DataFrame(rows)
fixed.to_csv(OUT / 'fixed_rules_by_fold.csv', index=False)
grid.to_csv(OUT / 'grid_by_fold.csv', index=False)

def aggregate(table, keys):
    result = []
    for values, g in table.groupby(keys, sort=False):
        r = dict(zip(keys, values))
        r['mean_macro_temporal_f1'] = float(g.macro_temporal_f1.mean())
        if 'block_macro_f1' in g:
            r['mean_block_macro_f1'] = float(g.block_macro_f1.mean())
        for c in ['line', 'scrum']:
            tp, fp, fn = (int(g[c + '_' + k].sum()) for k in ['tp', 'fp', 'fn'])
            r.update({c + '_tp': tp, c + '_fp': fp, c + '_fn': fn, c + '_event_f1': 2 * tp / (2 * tp + fp + fn), c + '_mean_temporal_f1': float(g[c + '_temporal_f1'].mean())})
        result.append(r)
    return pd.DataFrame(result)
fa = aggregate(fixed, ['fusion', 'policy'])
ga = aggregate(grid, ['fusion', 'line_rule', 'scrum_rule']).sort_values('mean_macro_temporal_f1', ascending=False, kind='stable')
fa.to_csv(OUT / 'fixed_rules_aggregate.csv', index=False)
ga.to_csv(OUT / 'grid_aggregate.csv', index=False)
old = pd.read_csv(_resolve_model_artifact(HERE.parent / 'm1_argmax_grid_20261022/cached_tests/aggregate_361_combinations.csv'))

def key(line, scrum):
    return tuple((v if v == 'argmax' else round(float(v), 6) for v in [line, scrum]))
reference = {key(r.line_rule, r.scrum_rule): r.mean_macro_temporal_f1 for r in old.itertuples()}
for r in ga[ga.fusion.eq('geometric')].itertuples():
    assert np.isclose(r.mean_macro_temporal_f1, reference[key(r.line_rule, r.scrum_rule)], atol=1e-12, rtol=0)
assert len(fixed) == 48 and len(grid) == 4332 and (len(ga) == 722)
assert all((sha(p) == h for p, h in hashes.items()))
assert not any((k == 'keras' or k.startswith('tensorflow') for k in sys.modules))
verification = dict(passed=True, folds=6, same_original_alpha=True, no_inference=True, no_retraining=True, keras_loaded=False, geometric_reconstruction=reconstruction, original_grid_reproduced=True, metrics_crosschecked_with_original=True, grid_ranking='descriptive test comparison only; not independent selection', source_sha256=hashes)
(HERE / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print('\nFixed rules:', flush=True)
print(fa.to_string(index=False), flush=True)
print('\nBest descriptive combination per fusion:', flush=True)
print(ga.groupby('fusion', sort=False).head(1).to_string(index=False), flush=True)
print('\nArgmax per fold:', flush=True)
print(fixed[fixed.policy.eq('argmax')][['fold', 'alpha', 'fusion', 'macro_temporal_f1', 'block_macro_f1']].to_string(index=False), flush=True)
