"""Descriptive 361-policy test comparison using saved probabilities only.

No inference, training, or selection on development in this script.
Ranking here describes these six tests and is not a new independent score.
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
from models.common.studies.m1_argmax_grid_20261022.run import pd, np, Evaluator, decide, GRID, RUNS, sha, HERE
out = HERE / 'cached_tests'
out.mkdir(exist_ok=True)
rows = []
hashes = {}
for i in range(1, 7):
    fold = f'PV{i:02d}'
    folder = RUNS / f'PV-M1-H2-D2-{fold}-S42' / 'attempt_001'
    path = folder / 'predictions.csv'
    hashes[str(path)] = sha(path)
    f = pd.read_csv(_resolve_model_artifact(path))
    f['group_id'] = f.test_group
    e = Evaluator(f)
    for line, scrum in [('argmax', 'argmax'), (0.35, 'argmax'), ('argmax', 0.35), (0.35, 0.35)]:
        e.verify(decide(e.p, line, scrum))
    for line in GRID:
        for scrum in GRID:
            family = 'both_argmax' if line == scrum == 'argmax' else 'line_argmax' if line == 'argmax' else 'scrum_argmax' if scrum == 'argmax' else 'numeric_numeric'
            rows.append(dict(fold=fold, test_group=f.test_group.iloc[0], line_rule=line, scrum_rule=scrum, family=family, **e.evaluate(decide(e.p, line, scrum))))
    print(f'{fold}: evaluated all 361 policies on saved test probabilities', flush=True)
table = pd.DataFrame(rows)
table.to_csv(out / 'per_fold.csv', index=False)
aggregate = []
for (line, scrum, family), g in table.groupby(['line_rule', 'scrum_rule', 'family'], sort=False):
    r = dict(line_rule=line, scrum_rule=scrum, family=family, mean_macro_temporal_f1=float(g.macro_temporal_f1.mean()))
    for c in ['line', 'scrum']:
        tp, fp, fn = (int(g[c + '_' + k].sum()) for k in ['tp', 'fp', 'fn'])
        r.update({c + '_tp': tp, c + '_fp': fp, c + '_fn': fn, c + '_precision': tp / (tp + fp) if tp + fp else 0, c + '_recall': tp / (tp + fn), c + '_event_f1': 2 * tp / (2 * tp + fp + fn), c + '_mean_temporal_f1': float(g[c + '_temporal_f1'].mean())})
    aggregate.append(r)
a = pd.DataFrame(aggregate).sort_values('mean_macro_temporal_f1', ascending=False, kind='stable')
a.to_csv(out / 'aggregate_361_combinations.csv', index=False)
assert len(table) == 2166 and len(a) == 361
assert all((sha(p) == h for p, h in hashes.items()))
assert not any((k == 'keras' or k.startswith('tensorflow') for k in sys.modules))
(out / 'verification.json').write_text(json.dumps(dict(passed=True, inference=False, keras_loaded=False, folds=6, combinations_per_fold=361, selection='none; descriptive test comparison only', original_metrics_crosschecked=True, source_sha256=hashes), indent=2), encoding='utf-8')
columns = ['line_rule', 'scrum_rule', 'line_tp', 'line_fp', 'line_fn', 'scrum_tp', 'scrum_fp', 'scrum_fn', 'mean_macro_temporal_f1']
print('Top descriptive combinations (not independent selection):', flush=True)
print(a[columns].head(12).to_string(index=False), flush=True)
print('Best descriptive combination per family:', flush=True)
print(a.groupby('family', sort=False).head(1)[['family'] + columns].to_string(index=False), flush=True)
