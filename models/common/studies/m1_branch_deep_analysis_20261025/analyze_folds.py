"""Complete saved-test branch ablations and predefined simple interventions.

All new interventions are exploratory. No new tuning on development and no
independent performance claim; original out-of-fold model weights stay fixed.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json, hashlib
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from models.common.studies.m1_argmax_grid_20261022.run import np, pd, m, Evaluator, decide, RUNS, LABELS
from sklearn.metrics import f1_score, confusion_matrix
HERE = experiment_directory(__file__)
OUT = HERE / 'folds'
OUT.mkdir(exist_ok=True)
metrics = []
sweeps = []
events = []
truthrows = []
blocks = []
calibration = []
confusions = []
source_hashes = {}
sha = lambda p: hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()

def fuse(d, g, a, mode='geometric'):
    return m.normalize_probabilities(np.exp((1 - a) * np.log(np.clip(d, 1e-07, 1)) + a * np.log(np.clip(g, 1e-07, 1))) if mode == 'geometric' else (1 - a) * d + a * g)

def frame_for(f, p, y):
    q = f.assign(y_final=LABELS[y]).copy()
    for k, c in enumerate(LABELS):
        q['p_' + c] = p[:, k]
    return q

def event_data(f, p, y, suppress=0):
    q = frame_for(f, p, y)
    out = {}
    for c in ['line', 'scrum']:
        tr = m.events_from_blocks(q, c, prediction_column='y_true', probability_column='p_' + c)
        pr = m.events_from_blocks(q, c, prediction_column='y_final', probability_column='p_' + c)
        scores = []
        for e in pr:
            scores.append(float(q.loc[q.section.eq(e.section) & q.block_index.isin(e.blocks), 'p_' + c].max()))
        if suppress:
            keep = []
            for j in sorted(range(len(pr)), key=lambda j: (-scores[j], pr[j].anchor)):
                if not any((pr[j].section == pr[k].section and abs(pr[j].anchor - pr[k].anchor) <= suppress for k in keep)):
                    keep.append(j)
            keep = sorted(keep)
            pr = [pr[j] for j in keep]
            scores = [scores[j] for j in keep]
        pairs = m.match_events(tr, pr)
        out[c] = (tr, pr, pairs, scores)
    return out

def event_score(data):
    r = {}
    for c, (tr, pr, pairs, scores) in data.items():
        matched = {j for _, j, _, _ in pairs}
        near = sum((j not in matched and any((t.section == e.section and abs(t.anchor - e.anchor) <= 35 for t in tr)) for j, e in enumerate(pr)))
        r.update({c + '_tp': len(pairs), c + '_fp': len(pr) - len(pairs), c + '_fn': len(tr) - len(pairs), c + '_duplicates': near, c + '_far_fp': len(pr) - len(pairs) - near, c + '_temporal_f1': 2 * sum((z[3] for z in pairs)) / (len(tr) + len(pr))})
    r['macro_temporal_f1'] = (r['line_temporal_f1'] + r['scrum_temporal_f1']) / 2
    return r
for i in range(1, 7):
    fold = f'PV{i:02d}'
    folder = RUNS / f'PV-M1-H2-D2-{fold}-S42' / 'attempt_001'
    for name in ['run.json', 'predictions.csv']:
        source_hashes[str(folder / name)] = sha(folder / name)
    run = json.loads(_resolve_model_artifact(folder / 'run.json').read_text())
    f = pd.read_csv(_resolve_model_artifact(folder / 'predictions.csv'))
    f['group_id'] = f.test_group
    assert run['test_group'] not in run['train_groups'] + [run['development_group']]
    d = f[['p_dual_' + c for c in LABELS]].to_numpy()
    g = f[['p_geometry_' + c for c in LABELS]].to_numpy()
    p = f[['p_' + c for c in LABELS]].to_numpy()
    a = run['model_metadata']['geometry_alpha']
    assert np.allclose(fuse(d, g, a), p, atol=1e-12)
    yt = np.array([list(LABELS).index(x) for x in f.y_true])
    dy = d.argmax(1)
    gy = g.argmax(1)
    fy = p.argmax(1)
    packs = {}
    for name, prob in [('dual', d), ('geometry', g), ('fusion', p)]:
        pred = prob.argmax(1)
        bm = m.block_metrics(yt, prob)
        data = event_data(f, prob, pred)
        packs[name] = data
        row = dict(fold=fold, policy=name, alpha=a, block_macro_f1=bm['f1_macro'], accuracy=bm['accuracy'], nll=bm['log_loss'], brier=bm['brier_multiclass'], ece=bm['ece_15'], **event_score(data))
        metrics.append(row)
        ref = m.temporal_metrics(frame_for(f, prob, pred))
        assert np.isclose(row['macro_temporal_f1'], ref['weighted_f1_macro'])
        cf = confusion_matrix(yt, pred, labels=[0, 1, 2])
        for u, c in enumerate(LABELS):
            for v, cl in enumerate(LABELS):
                confusions.append(dict(fold=fold, branch=name, truth=c, prediction=cl, count=int(cf[u, v])))
        for lo in np.arange(0, 1, 0.1):
            conf = prob.max(1)
            mask = (conf >= lo) & (conf < lo + 0.1 if lo < 0.9 else conf <= 1)
            calibration.append(dict(fold=fold, branch=name, bin_lower=lo, n=int(mask.sum()), confidence_sum=float(conf[mask].sum()), correct=int((pred[mask] == yt[mask]).sum())))
        for c, (tr, pr, pairs, scores) in data.items():
            mt = {j: i for i, j, _, _ in pairs}
            for j, e in enumerate(pr):
                ids = f.index[f.section.eq(e.section) & f.block_index.isin(e.blocks)].to_numpy()
                k = list(LABELS).index(c)
                idx = int(ids[np.argmax(prob[ids, k])])
                near = min((abs(e.anchor - t.anchor) for t in tr if t.section == e.section), default=float('inf'))
                events.append(dict(fold=fold, branch=name, label=c, section=e.section, start=e.start, end=e.end, anchor=e.anchor, blocks=len(e.blocks), status='TP' if j in mt else 'duplicate_FP' if near <= 35 else 'far_FP', confidence=scores[j], truth_at_anchor=f.loc[idx, 'y_true'], dual_decision=LABELS[dy[idx]], geometry_decision=LABELS[gy[idx]], block_id=f.loc[idx, 'block_id']))
    for c in ['line', 'scrum']:
        matches = {name: {i for i, _, _, _ in packs[name][c][2]} for name in packs}
        for j, e in enumerate(packs['fusion'][c][0]):
            truthrows.append(dict(fold=fold, label=c, section=e.section, start=e.start, end=e.end, **{name: j in matches[name] for name in matches}))
    for idx, r in enumerate(f.itertuples()):
        dc = dy[idx] == yt[idx]
        gc = gy[idx] == yt[idx]
        category = 'both_correct' if dc and gc else 'dual_only_correct' if dc else 'geometry_only_correct' if gc else 'both_wrong_same' if dy[idx] == gy[idx] else 'both_wrong_different'
        blocks.append(dict(fold=fold, block_id=r.block_id, truth=r.y_true, dual=LABELS[dy[idx]], geometry=LABELS[gy[idx]], fusion=LABELS[fy[idx]], category=category, fusion_correct=bool(fy[idx] == yt[idx]), dual_confidence=float(d[idx].max()), geometry_confidence=float(g[idx].max()), fusion_confidence=float(p[idx].max())))
    policies = []
    for inc in [0.05, 0.075, 0.125]:
        policies.append((f'alpha_plus_{inc:.3f}', fuse(d, g, min(1, a + inc)), None, 0))
    policies.append(('arithmetic_same_alpha', fuse(d, g, a, 'arithmetic'), None, 0))
    for window in [14, 21, 28, 35]:
        policies.append((f'nms_{window}', p, None, window))
    for gate in [0.15, 0.25]:
        pred = fy.copy()
        veto = (pred != 1) & (g[np.arange(len(g)), pred] < gate)
        pred[veto] = 1
        policies.append((f'geometry_gate_{gate:.2f}', p, pred, 0))
    policies.append(('geometry_agreement', p, np.where((fy != 1) & (gy != fy), 1, fy), 0))
    policies.append(('original_dev_thresholds', p, decide(p, run['thresholds']['line'], run['thresholds']['scrum']), 0))
    baseline_truth = {c: {i for i, _, _, _ in packs['fusion'][c][2]} for c in ['line', 'scrum']}
    for row in metrics:
        if row['fold'] == fold and row['policy'] in packs:
            for c in ['line', 'scrum']:
                ids = {j for j, _, _, _ in packs[row['policy']][c][2]}
                row[c + '_lost_vs_fusion'] = len(baseline_truth[c] - ids)
                row[c + '_recovered_vs_fusion'] = len(ids - baseline_truth[c])
    for name, prob, y, window in policies:
        if y is None:
            y = prob.argmax(1)
        data = event_data(f, prob, y, window)
        row = dict(fold=fold, policy=name, alpha=a, **event_score(data))
        for c in ['line', 'scrum']:
            ids = {i for i, _, _, _ in data[c][2]}
            row[c + '_lost_vs_fusion'] = len(baseline_truth[c] - ids)
            row[c + '_recovered_vs_fusion'] = len(ids - baseline_truth[c])
        metrics.append(row)
    for alpha in np.arange(0, 1.0001, 0.05):
        prob = fuse(d, g, alpha)
        q = frame_for(f, prob, prob.argmax(1))
        ev = Evaluator(q)
        sweeps.append(dict(fold=fold, alpha=float(alpha), **ev.evaluate(prob.argmax(1))))
    print(f'{fold}: branches, decisions, events, calibration and interventions complete', flush=True)
table = pd.DataFrame(metrics)
table.to_csv(OUT / 'metrics_by_fold.csv', index=False)
agg = []
base = table[table.policy.eq('fusion')].set_index('fold')
for policy, q in table.groupby('policy', sort=False):
    delta = q.set_index('fold').macro_temporal_f1 - base.macro_temporal_f1
    r = dict(policy=policy, mean_temporal_f1=float(q.macro_temporal_f1.mean()), delta_vs_fusion=float(delta.mean()), folds_improved=int((delta > 1e-10).sum()), folds_worse=int((delta < -1e-10).sum()), worst_fold_delta=float(delta.min()), best_fold_delta=float(delta.max()))
    for c in ['line', 'scrum']:
        for k in ['tp', 'fp', 'fn', 'duplicates', 'far_fp', 'lost_vs_fusion', 'recovered_vs_fusion']:
            col = c + '_' + k
            if col in q:
                r[col] = int(q[col].sum())
        r[c + '_mean_temporal_f1'] = float(q[c + '_temporal_f1'].mean())
    for k in ['block_macro_f1', 'accuracy', 'nll', 'brier', 'ece']:
        if q[k].notna().any():
            r['mean_' + k] = float(q[k].mean())
    agg.append(r)
pd.DataFrame(agg).to_csv(OUT / 'aggregate.csv', index=False)
for name, rows in [('alpha_sweep', sweeps), ('events', events), ('truth_event_outcomes', truthrows), ('block_decisions', blocks), ('calibration_bins', calibration), ('confusion', confusions)]:
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False)
assert len(blocks) == 4714 and len(truthrows) == 295
assert all((sha(Path(p)) == h for p, h in source_hashes.items()))
assert not any((x == 'keras' or x.startswith('tensorflow') for x in sys.modules))
(HERE / 'fold_verification.json').write_text(json.dumps(dict(passed=True, blocks=4714, true_events=295, folds=6, source_sha256=source_hashes, no_inference=True, no_training=True, new_interventions='predefined exploratory test comparisons; no development selection; not independent validation', all_base_metrics_crosschecked=True), indent=2))
print(pd.DataFrame(agg)[['policy', 'mean_temporal_f1', 'delta_vs_fusion', 'folds_improved', 'folds_worse', 'line_tp', 'line_fp', 'scrum_tp', 'scrum_fp']].to_string(index=False), flush=True)
