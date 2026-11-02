"""Pure independent metrics, split policy and predeclared evidence rules."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np
LABELS = ['line', 'other', 'scrum']

def split_indices(rows):
    train = []
    val = []
    test = []
    for i, r in enumerate(rows):
        if r['domain'] == 'VF':
            assert r['split'] == 'test'
            test.append(i)
        else:
            assert r['split'] == 'train'
            (val if r['domain'] == 'CV' and r['group_id'] == 'V_6' else train).append(i)
    return (np.array(train, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int))

def metrics(y, p):
    c = np.bincount(np.asarray(y) * 3 + np.asarray(p), minlength=9).reshape(3, 3)
    d = c.diagonal()
    pr = np.divide(d, c.sum(0), out=np.zeros(3), where=c.sum(0) > 0)
    re = np.divide(d, c.sum(1), out=np.zeros(3), where=c.sum(1) > 0)
    f = np.divide(2 * d, c.sum(0) + c.sum(1), out=np.zeros(3), where=c.sum(0) + c.sum(1) > 0)
    out = {'f1_macro': float(f.mean()), 'accuracy': float(d.sum() / max(c.sum(), 1))}
    for i, k in enumerate(LABELS):
        out.update({k + '_precision': float(pr[i]), k + '_recall': float(re[i]), k + '_f1': float(f[i])})
    return out

def changes(y, base, new):
    a = base == y
    b = new == y
    out = {'corrected': int((~a & b).sum()), 'broken': int((a & ~b).sum()), 'both_correct': int((a & b).sum()), 'both_wrong': int((~a & ~b).sum()), 'by_true_class': {}}
    for i, k in enumerate(LABELS):
        mask = y == i
        out['by_true_class'][k] = {'corrected': int((~a & b & mask).sum()), 'broken': int((a & ~b & mask).sum()), 'support': int(mask.sum())}
    return out

def success_rule(base, new, ci, seed_improvements, temporal_ci=None):
    checks = {'macro_gain_at_least_3pp': new['f1_macro'] - base['f1_macro'] >= 0.03, 'both_event_class_f1_improve': all((new[k + '_f1'] > base[k + '_f1'] for k in ['line', 'scrum'])), 'no_precision_or_recall_drop_over_2pp': all((new[k + '_' + m] >= base[k + '_' + m] - 0.02 for k in ['line', 'scrum'] for m in ['precision', 'recall'])), 'at_least_two_seeds_improve': sum(seed_improvements) >= 2, 'stratified_ci_positive': ci[0] > 0}
    if temporal_ci is not None:
        checks['temporal_block_ci_positive'] = temporal_ci[0] > 0
    return {'passed': all(checks.values()), 'checks': checks}

def temporal_bootstrap(y, base, new, sections, starts, replicates=2000, seed=20261026):
    rng = np.random.default_rng(seed)
    sections = np.asarray(sections)
    starts = np.asarray(starts)
    groups = []
    for section in np.unique(sections):
        bins = np.floor(starts / 60).astype(int)
        groups.append([np.flatnonzero((sections == section) & (bins == b)) for b in np.unique(bins[sections == section])])
    d = []
    for _ in range(replicates):
        ix = np.concatenate([np.concatenate([g[j] for j in rng.integers(0, len(g), len(g))]) for g in groups])
        d.append(np.mean([metrics(y[ix], n[ix])['f1_macro'] - metrics(y[ix], a[ix])['f1_macro'] for a, n in zip(base, new)]))
    delta = np.mean([metrics(y, n)['f1_macro'] - metrics(y, a)['f1_macro'] for a, n in zip(base, new)])
    return dict(mean_difference=float(delta), ci95=np.quantile(d, [0.025, 0.975]).tolist(), replicates=replicates, seed=seed, cluster_seconds=60, method='resample contiguous fixed60s clusters within section; paired predictions, seed mean; exploratory')
