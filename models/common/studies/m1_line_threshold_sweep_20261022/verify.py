"""Verify the delivered tables and unchanged input hashes without running networks."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
HERE = experiment_directory(__file__)
OUT = HERE / 'results'

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(p)).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()
validation = json.loads(_resolve_model_artifact(HERE / 'validation.json').read_text(encoding='utf-8'))
assert validation['sources_unchanged']
assert all((sha(p) == h for p, h in validation['source_hashes'].items()))
m = pd.read_csv(_resolve_model_artifact(OUT / 'metrics_by_match.csv'))
c = pd.read_csv(_resolve_model_artifact(OUT / 'changes_by_match.csv'))
d = pd.read_csv(_resolve_model_artifact(OUT / 'derived_predictions.csv.gz'))
a = pd.read_csv(_resolve_model_artifact(OUT / 'aggregate_equal_weight.csv'))
assert len(m) == 180 and len(a) == 36 and (len(d) == (4714 + 688) * 18)
assert not m.duplicated(['scope', 'match', 'policy']).any()
for (scope, match), g in d.groupby(['scope', 'match']):
    base = g[g.policy == 'original'].sort_values(['section', 'block_index'])
    for policy, h in g.groupby('policy'):
        h = h.sort_values(['section', 'block_index'])
        assert np.array_equal(base.block_id, h.block_id)
        assert np.array_equal(base.y_true, h.y_true)
        assert np.array_equal(base[['p_line', 'p_other', 'p_scrum']], h[['p_line', 'p_other', 'p_scrum']])
        assert np.array_equal(base.scrum_threshold, h.scrum_threshold)
        if policy != 'argmax':
            assert (h.loc[base.y_final.to_numpy() == 'line', 'y_final'] == 'line').all()
for r in m.itertuples():
    for label in ['line', 'scrum']:
        assert getattr(r, label + '_tp') + getattr(r, label + '_fn') == getattr(r, label + '_real')
        assert getattr(r, label + '_tp') + getattr(r, label + '_fp') == getattr(r, label + '_candidates')
        assert getattr(r, label + '_far_fp') + getattr(r, label + '_duplicates') == getattr(r, label + '_fp')
    assert np.isclose(r.temporal_f1_macro, (r.line_temporal_f1 + r.scrum_temporal_f1) / 2)
    base = m[(m.scope == r.scope) & (m.match == r.match) & (m.policy == 'original')].iloc[0]
    cr = c[(c.scope == r.scope) & (c.match == r.match) & (c.policy == r.policy)].iloc[0]
    assert r.line_tp - base.line_tp == cr.line_recovered - cr.line_lost
    assert r.line_fp - base.line_fp == cr.line_net_fp
for r in a.itertuples():
    g = m[(m.scope == r.scope) & (m.policy == r.policy)]
    assert len(g) == (6 if r.scope == 'PV_heldout' else 4)
    for k in ['line_recall', 'line_precision', 'line_f1', 'line_temporal_f1', 'scrum_temporal_f1', 'temporal_f1_macro', 'all_fp_per_hour', 'all_far_fp_per_hour']:
        assert np.isclose(getattr(r, 'mean_' + k), g[k].mean())
    assert r.sum_line_tp == g.line_tp.sum() and r.sum_line_fp == g.line_fp.sum()
result = dict(passed=True, metric_rows=len(m), aggregate_rows=len(a), derived_prediction_rows=len(d), original_hashes_still_match=True, probabilities_and_labels_unchanged_across_policies=True, event_count_and_change_accounting=True, equal_weight_aggregates=True, output_hashes={p.name: sha(p) for p in sorted(_model_artifact_glob(OUT, '*', recursive=False)) if _resolve_model_artifact(p).is_file()})
(HERE / 'delivery_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'output_hashes'}, indent=2))
