"""Read-only probability re-evaluation. Run from any directory; no model imports."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, hashlib, time, platform
ROOT = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.common.comparison import metrics as m
from models.common.comparison.protocol import LABELS, PV_FOLDS
BASE = ROOT / 'data/final_model_comparison_20260827'
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
(HERE / 'figures').mkdir(exist_ok=True)
sources = {}

def sha(path):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(path)).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def track(path):
    path = Path(path)
    sources[str(path)] = sha(path)
    return path

def readcsv(path):
    return pd.read_csv(_resolve_model_artifact(track(path)))

def readjson(path):
    return json.loads(_resolve_model_artifact(track(path)).read_text(encoding='utf-8-sig'))

def savejson(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def timestamp(s):
    return f'{int(s) // 60:02d}:{s % 60:04.1f}'
start = time.perf_counter()
track(Path(m.__file__))
track(_resolve_model_artifact(BASE / 'protocol.py'))
track(_resolve_model_artifact(BASE / 'runner.py'))
manifest = readcsv(ROOT / 'data/final_tests_feature_cache_20260826/outputs/final_test_feature_manifest.csv')
master = readcsv(BASE / 'outputs/master_examples.csv')
historical_audit = readjson(ROOT / 'data/final_tests_feature_cache_20260826/outputs/training_readiness_audit.json')
assert historical_audit['valid']
summary_historical = readcsv(BASE / 'outputs/comparison_summary.csv')
datasets, inventory, checks, session_inventory = ([], [], [], [])
for fold in PV_FOLDS:
    rid = f'PV-M1-H2-D2-{fold.fold}-S42'
    runs = sorted(_model_artifact_glob(BASE / 'outputs/runs' / rid, 'attempt_*/run.json', recursive=False))
    assert len(runs) == 1, (rid, 'ambiguous attempt', runs)
    meta = readjson(runs[0])
    folder = runs[0].parent
    saved = readjson(folder / 'metrics.json')
    f = readcsv(folder / 'predictions.csv').rename(columns={'test_group': 'group_id'})
    assert meta['smoke_limit'] is None and meta['model_metadata']['variant'] == 'M1-100-H2'
    assert meta['test_group'] == fold.test_group and meta['development_group'] == fold.development_group
    assert meta['train_groups'] == list(fold.train_groups)
    assert fold.test_group not in meta['train_groups'] and fold.test_group != meta['development_group']
    assert f.checkpoint_hash.eq(meta['checkpoint_hash']).all()
    assert f.cache_signature.eq(meta['cache_signature']).all()
    weights = {}
    for p, h in meta['artifacts'].items():
        actual = sha(track(p))
        assert actual == h, p
        weights[p] = actual
    mf = manifest[manifest.match == fold.test_group].set_index('dataset_id').loc[f.block_id]
    assert len(f) == len(mf) == meta['n_test']
    assert np.array_equal(f.y_true, mf.true_label)
    assert np.array_equal(f.section, mf.section)
    assert np.array_equal(f.block_index, mf.block_index)
    assert np.allclose(f.start_s, mf.start_time) and np.allclose(f.end_s, mf.end_time)
    for section, sf in mf.reset_index().groupby('section', sort=False):
        paths = sf.session_file.unique()
        assert len(paths) == 1
        session = readjson(paths[0])
        candidates = {int(c['candidate_index']): c for c in session['candidates']}
        assert len(candidates) == len(sf)
        for r in sf.itertuples():
            c = candidates[int(r.block_index)]
            assert c['final_label'] == r.true_label, (paths[0], r.block_index)
            assert np.isclose(c['start_time'], r.start_time) and np.isclose(c['end_time'], r.end_time)
        ordered = sf.sort_values('block_index')
        assert ordered.iloc[0].start_time == 0
        assert np.allclose(ordered.start_time.to_numpy()[1:], ordered.end_time.to_numpy()[:-1])
        assert np.isclose(ordered.iloc[-1].end_time, session['video_duration'] - session.get('ignored_tail_seconds', 0), atol=0.01)
        session_inventory.append(dict(match=fold.test_group, section=section, session=paths[0], video_name=session['video_name'], seconds=float(ordered.iloc[-1].end_time), video_seconds=float(session['video_duration']), ignored_tail_seconds=session.get('ignored_tail_seconds', 0), blocks=len(sf), verified=True))
    p = f[['p_line', 'p_other', 'p_scrum']].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-06)
    baseline = m.apply_class_thresholds(p, line=meta['thresholds']['line'], scrum=meta['thresholds']['scrum'])
    assert np.array_equal(np.asarray(LABELS)[baseline], f.y_final)
    assert np.array_equal(np.asarray(LABELS)[m.normalize_probabilities(p).argmax(1)], f.y_argmax)
    tm = m.temporal_metrics(f)
    for c in ['line', 'scrum']:
        assert np.isclose(tm[c]['weighted_f1'], saved['temporal'][c]['weighted_f1'], atol=1e-12)
    checks.append(dict(scope='PV_heldout', match=fold.test_group, baseline_decisions=True, baseline_metrics=True, human_labels_verified=True))
    saved_files = sorted((str(x.relative_to(folder)) for x in _model_artifact_glob(folder, '*', recursive=True) if _resolve_model_artifact(x).is_file()))
    inventory.append(dict(scope='PV_heldout', match=fold.test_group, run_id=rid, attempt=meta['attempt'], probability_path=str(folder / 'predictions.csv'), training_groups=meta['train_groups'], development_group=meta['development_group'], test_group=meta['test_group'], out_of_declared_training=True, initial_historical_source_independence='not newly established', checkpoint_hash=meta['checkpoint_hash'], weight_hashes=weights, cache_signature=meta['cache_signature'], thresholds=meta['thresholds'], model_metadata=meta['model_metadata'], sections=f.section.nunique(), blocks=len(f), seconds=float(f.duration.sum()), development_probabilities_available=False, files_in_attempt=saved_files, unavailable_artifact=f'{folder}/development_predictions.csv (not persisted by runner; no alternate matching file found)', label_provenance='all rows checked against frozen feature manifest and 31 original human review sessions'))
    datasets.append(dict(scope='PV_heldout', match=fold.test_group, frame=f, line=meta['thresholds']['line'], scrum=meta['thresholds']['scrum'], saved=saved['temporal']))
vfroot = ROOT / 'data/m1_ensemble_vs_all_development_20261012'
vfmeta = readjson(vfroot / 'inputs/M1-single_manifest.json')
vf_independence = readjson(vfroot / 'inputs/independence_audit.json')
vfweights = {str(ROOT / vfmeta['folds'][0][k]): sha(track(ROOT / vfmeta['folds'][0][k])) for k in ['dual_model', 'geometry_rf']}
vf_hashes = None
for n in range(1, 5):
    campaign = vfroot if n == 1 else ROOT / f'data/m1_ensemble_vs_single_vf_s0{n}_20261013'
    f = readcsv(campaign / 'results/M1-single_quality_blocks.csv')
    startup = readjson(campaign / 'results/M1-single_startup.json')
    saved = readjson(campaign / 'results/M1-single_quality.json')
    startup_weights = {p: h for p, h in startup['hashes_before'].items() if p in vfweights}
    assert startup_weights == vfweights, (n, startup_weights, vfweights)
    human_path = vfroot.parent / f'm1_ensemble_vs_single_vf_s0{n}_20261013/inputs/human_blocks.csv'
    if n == 1:
        human_path = ROOT / 'data/active_learning_dataset/sessions/20260830_112524_0b5e58/train_manifest.csv'
    human = readcsv(human_path).rename(columns={'dataset_id': 'block_id', 'candidate_index': 'block_index', 'correct_label': 'y_true', 'start_time': 'start_s', 'end_time': 'end_s'})
    for c in ['block_id', 'block_index', 'y_true']:
        assert np.array_equal(f[c], human[c])
    for c in ['start_s', 'end_s']:
        assert np.allclose(f[c], human[c])
    p = f[['p_line', 'p_other', 'p_scrum']].to_numpy()
    assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-06)
    line, scrum = (startup['thresholds']['line'], startup['thresholds']['scrum'])
    assert np.array_equal(np.asarray(LABELS)[m.apply_class_thresholds(p, line=line, scrum=scrum)], f.y_final)
    assert np.isclose(m.temporal_metrics(f)['weighted_f1_macro'], saved['temporal']['weighted_f1_macro'])
    datasets.append(dict(scope='VF_single_sections', match=f'VF_S0{n}', frame=f, line=line, scrum=scrum, saved=saved['temporal']))
    inventory.append(dict(scope='VF_single_sections', match=f'VF_S0{n}', run_id='ALL-M1-H2-D2-S42', probability_path=str(campaign / 'results/M1-single_quality_blocks.csv'), training_groups=['V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6'], independent_test_claim_allowed=False, independence_audit=vf_independence, weight_hashes=vfweights, thresholds=startup['thresholds'], blocks=len(f), seconds=float(f.duration.sum()), label_path=str(human_path), complete_match=False))
policies = [('original', 'original', 0.0)] + [(f'minus_{d:.2f}', 'subtract', float(d)) for d in np.arange(0.05, 0.651, 0.05)] + [('times_0.75', 'multiply', 0.75), ('times_0.50', 'multiply', 0.5), ('times_0.25', 'multiply', 0.25), ('argmax', 'argmax', 0.0)]
config = dict(model='M1-H2-D2', labels=list(LABELS), policies=[dict(id=i, kind=k, value=v) for i, k, v in policies], minimum_line_threshold=0.05, scrum_threshold='unchanged per fold / VF section', tolerance_seconds=35, rule='Original apply_class_thresholds; target probabilities divided by their own thresholds; line >= scrum wins ties; both must meet own threshold. No argmax gate; other probability not compared.', argmax='separate control; normalized line/other/scrum argmax, first index line then other then scrum on ties; has no class thresholds', grouping='consecutive block indices within (group_id,section); section boundaries never merged', truth_anchor='midpoint of true event', predicted_anchor='midpoint of earliest block with maximal class probability in the event', matching='original Hungarian assignment maximizing sum(max(0,1-|anchor difference|/35)); one-to-one same group/section; distance <=35s; zero credit at exactly 35s retained and reported', integer_metrics='TP=count of original assigned pairs, FP=unmatched predictions including duplicates, FN=unmatched truth. These are distinct from weighted credit.', aggregation='arithmetic mean per match for PV, equal section weight for VF diagnostic only; integer counts also summed with sum_ prefix', event_changes='recovered/lost use stable true-event indices; new FP without any overlapping baseline FP event; removed FP analogous. Every prediction tracks overlap with all baseline candidates.', exploration='test labels used to compare policies; no newly selected threshold has an independent test estimate', validation='same-fold development probabilities not persisted; no inference or training to reconstruct them; another fold test prediction uses different weights and is not a substitute', source_policy='input files and weights hashed before/after; originals never modified')
savejson(HERE / 'config.json', config)
savejson(HERE / 'inventory.json', inventory)
pd.DataFrame(session_inventory).to_csv(HERE / 'manual_sessions.csv', index=False)
metrics_rows, all_predictions, event_rows, truth_changes, pred_changes, change_rows = ([], [], [], [], [], [])

def overlaps(a, b):
    return a.group == b.group and a.section == b.section and (min(a.end, b.end) > max(a.start, b.start))

def eventdict(e):
    return dict(section=e.section, start_s=e.start, end_s=e.end, anchor_s=e.anchor, start_time=timestamp(e.start), end_time=timestamp(e.end), anchor_time=timestamp(e.anchor), blocks=','.join(map(str, e.blocks)))

def detail(f, c):
    truth = m.events_from_blocks(f, c, prediction_column='y_true', probability_column='p_' + c)
    pred = m.events_from_blocks(f, c, prediction_column='y_final', probability_column='p_' + c)
    pairs = m.match_events(truth, pred)
    return (truth, pred, pairs)
for ds in datasets:
    f = ds['frame']
    p = f[['p_line', 'p_other', 'p_scrum']].to_numpy()
    common = dict(scope=ds['scope'], match=ds['match'])
    bases = {c: detail(f, c) for c in ['line', 'scrum']}
    hours = float(f.duration.sum() / 3600)
    for policy, kind, value in policies:
        t = ds['line'] if kind == 'original' else max(0.05, ds['line'] - value) if kind == 'subtract' else ds['line'] * value
        if kind == 'multiply':
            t = max(0.05, t)
        if kind == 'argmax':
            t = None
        y = m.normalize_probabilities(p).argmax(1) if kind == 'argmax' else m.apply_class_thresholds(p, line=t, scrum=ds['scrum'])
        candidate = f.copy()
        candidate['y_final'] = np.asarray(LABELS)[y]
        all_predictions.append(candidate[['block_id', 'group_id', 'section', 'block_index', 'start_s', 'end_s', 'duration', 'y_true', 'p_line', 'p_other', 'p_scrum', 'y_final']].assign(**common, policy=policy, line_threshold=t, scrum_threshold=ds['scrum']))
        temporal = m.temporal_metrics(candidate)
        row = dict(**common, policy=policy, line_threshold=t, scrum_threshold=ds['scrum'], hours=hours, temporal_f1_macro=temporal['weighted_f1_macro'])
        changes = dict(**common, policy=policy, line_threshold=t)
        for c in ['line', 'scrum']:
            truth, pred, pairs = detail(candidate, c)
            bt, bp, bpairs = bases[c]
            matched_t = {a: b for a, b, d, w in pairs}
            matched_p = {b: a for a, b, d, w in pairs}
            base_t = {a: b for a, b, d, w in bpairs}
            base_p = {b: a for a, b, d, w in bpairs}
            fp = [j for j in range(len(pred)) if j not in matched_p]
            bfp = [j for j in range(len(bp)) if j not in base_p]
            far = [j for j in fp if not any((e.group == pred[j].group and e.section == pred[j].section and (abs(e.anchor - pred[j].anchor) <= 35) for e in truth))]
            tp = len(pairs)
            n = len(truth)
            npred = len(pred)
            vals = dict(real=n, tp=tp, fp=len(fp), fn=n - tp, candidates=npred, precision=tp / npred if npred else 0.0, recall=tp / n if n else 0.0, f1=2 * tp / (n + npred) if n + npred else 1.0, temporal_f1=temporal[c]['weighted_f1'], weighted_tp_credit=sum((x[3] for x in pairs)), zero_credit_pairs=sum((x[3] == 0 for x in pairs)), fp_per_hour=len(fp) / hours, far_fp=len(far), far_fp_per_hour=len(far) / hours, duplicates=len(fp) - len(far), candidate_seconds=sum((e.end - e.start for e in pred)), fusions=temporal[c]['fusions'], fragmentations=temporal[c]['fragmentations'])
            row.update({c + '_' + k: v for k, v in vals.items()})
            recovered = set(matched_t) - set(base_t)
            lost = set(base_t) - set(matched_t)
            newfp = [j for j in fp if not any((overlaps(pred[j], bp[k]) for k in bfp))]
            removedfp = [j for j in bfp if not any((overlaps(bp[j], pred[k]) for k in fp))]
            merged = sum((sum((overlaps(e, b) for b in bp)) > 1 for e in pred))
            changed = sum((not any((eventdict(e) == eventdict(b) for b in bp)) for e in pred))
            changes.update({c + '_' + k: v for k, v in dict(recovered=len(recovered), lost=len(lost), net_tp=tp - len(bpairs), new_fp_nonoverlap=len(newfp), removed_fp_nonoverlap=len(removedfp), net_fp=len(fp) - len(bfp), merged_candidates=merged, changed_candidates=changed, net_fusions=temporal[c]['fusions'] - ds['saved'][c]['fusions'], delta_temporal_f1=temporal[c]['weighted_f1'] - ds['saved'][c]['weighted_f1']).items()})
            for i, e in enumerate(truth):
                if policy != 'original':
                    effect = 'recovered' if i in recovered else 'lost' if i in lost else 'still_detected' if i in matched_t else 'still_missed'
                    truth_changes.append(dict(**common, policy=policy, label=c, truth_id=i, effect=effect, **eventdict(e), baseline_candidate=base_t.get(i), candidate=matched_t.get(i), predicted_anchor_s=pred[matched_t[i]].anchor if i in matched_t else None))
            for j, e in enumerate(pred):
                pair = next((x for x in pairs if x[1] == j), None)
                overlap_base = [k for k, b in enumerate(bp) if overlaps(e, b)]
                status = 'TP' if j in matched_p else 'far_FP' if j in far else 'duplicate_FP'
                event_rows.append(dict(**common, policy=policy, label=c, event_id=j, status=status, **eventdict(e), truth_id=matched_p.get(j), distance_s=pair[2] if pair else None, weighted_credit=pair[3] if pair else 0, baseline_overlapping_ids=','.join(map(str, overlap_base)), new_fp_nonoverlap=j in newfp))
                if policy != 'original' and (j in newfp or len(overlap_base) > 1 or (not any((eventdict(e) == eventdict(b) for b in bp)))):
                    pred_changes.append(dict(**common, policy=policy, label=c, event_id=j, status=status, **eventdict(e), baseline_overlapping_ids=','.join(map(str, overlap_base)), merged=len(overlap_base) > 1, new_fp_nonoverlap=j in newfp))
        row['all_fp_per_hour'] = row['line_fp_per_hour'] + row['scrum_fp_per_hour']
        row['all_far_fp_per_hour'] = row['line_far_fp_per_hour'] + row['scrum_far_fp_per_hour']
        changes['scrum_to_line_blocks'] = int(((f.y_final == 'scrum') & (candidate.y_final == 'line')).sum())
        changes['changed_blocks'] = int((f.y_final != candidate.y_final).sum())
        metrics_rows.append(row)
        change_rows.append(changes)
    print('evaluated', ds['match'], len(f), 'blocks', flush=True)
metrics = pd.DataFrame(metrics_rows)
changes = pd.DataFrame(change_rows)
metrics.to_csv(OUT / 'metrics_by_match.csv', index=False)
changes.to_csv(OUT / 'changes_by_match.csv', index=False)
pd.concat(all_predictions, ignore_index=True).to_csv(OUT / 'derived_predictions.csv.gz', index=False, compression='gzip')
pd.DataFrame(event_rows).to_csv(OUT / 'candidate_events.csv.gz', index=False, compression='gzip')
tc = pd.DataFrame(truth_changes)
pc = pd.DataFrame(pred_changes)
tc.to_csv(OUT / 'truth_event_changes.csv.gz', index=False, compression='gzip')
tc[tc.effect.isin(['recovered', 'lost'])].to_csv(OUT / 'event_changes.csv', index=False)
pc.to_csv(OUT / 'changed_candidates.csv', index=False)
pc[pc.new_fp_nonoverlap].to_csv(OUT / 'added_false_alarms.csv', index=False)
agg = []
for (scope, policy), g in metrics.groupby(['scope', 'policy'], sort=False):
    r = dict(scope=scope, policy=policy, units=len(g))
    for col in g.select_dtypes(include='number').columns:
        r['mean_' + col] = float(g[col].mean())
    for c in ['line', 'scrum']:
        for col in ['real', 'tp', 'fp', 'fn', 'candidates', 'far_fp', 'duplicates', 'zero_credit_pairs', 'fusions', 'fragmentations']:
            r['sum_' + c + '_' + col] = int(g[c + '_' + col].sum())
    cg = changes[(changes.scope == scope) & (changes.policy == policy)]
    for c in ['line', 'scrum']:
        for col in ['recovered', 'lost', 'net_tp', 'new_fp_nonoverlap', 'removed_fp_nonoverlap', 'net_fp', 'merged_candidates', 'changed_candidates', 'net_fusions']:
            r['sum_' + c + '_' + col] = int(cg[c + '_' + col].sum())
    agg.append(r)
aggregate = pd.DataFrame(agg)
aggregate.to_csv(OUT / 'aggregate_equal_weight.csv', index=False)
historical = summary_historical.query("panel=='PV' and model=='M1' and level=='H2' and data_regime=='D2'").iloc[0]
pvbase = aggregate.query("scope=='PV_heldout' and policy=='original'").iloc[0]
assert np.isclose(pvbase.mean_temporal_f1_macro, historical.f1w_macro, atol=1e-12)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
pv = aggregate.query("scope=='PV_heldout'")
sweep = pv[pv.policy.str.startswith('minus_') | pv.policy.eq('original')]
axs[0].plot(sweep.mean_line_fp_per_hour, sweep.mean_line_recall, 'o-', label='Umbrales originales − reducción')
axs[0].set(xlabel='FP de lineout por hora (todos)', ylabel='Sensibilidad por evento', title='Seis partidos, igual peso')
axs[1].plot(sweep.mean_line_precision, sweep.mean_line_recall, 'o-')
axs[1].set(xlabel='Precisión por evento', ylabel='Sensibilidad por evento', title='Lineout: precisión y sensibilidad')
axs[2].plot(sweep.mean_line_threshold, sweep.mean_line_temporal_f1, 'o-', label='Lineout')
axs[2].plot(sweep.mean_line_threshold, sweep.mean_scrum_temporal_f1, 'o-', label='Scrum')
axs[2].plot(sweep.mean_line_threshold, sweep.mean_temporal_f1_macro, 'o-', label='Media ambas clases')
axs[2].set(xlabel='Umbral medio efectivo de lineout', ylabel='F1 temporal ponderado', title='Efecto sobre las dos clases')
for policy in ['original', 'minus_0.05', 'minus_0.10', 'minus_0.20', 'minus_0.30', 'minus_0.65']:
    r = pv[pv.policy == policy].iloc[0]
    axs[0].annotate(policy.replace('minus_', '−').replace('original', 'Original'), (r.mean_line_fp_per_hour, r.mean_line_recall), fontsize=8, xytext=(3, 3), textcoords='offset points')
a = pv[pv.policy == 'argmax'].iloc[0]
axs[0].scatter([a.mean_line_fp_per_hour], [a.mean_line_recall], marker='x', s=80, label='Argmax (control separado)')
for ax in axs:
    ax.grid(alpha=0.2)
axs[0].legend(fontsize=8)
axs[2].legend(fontsize=8)
fig.suptitle('M1-H2-D2 · barrido exploratorio sin entrenamiento', fontsize=16)
fig.tight_layout()
fig.savefig(HERE / 'figures/sensitivity_false_alarms.png', dpi=150)
plt.close(fig)
savejson(HERE / 'validation.json', dict(baseline_checks=checks, baseline_mean_matches_historical=True, historical_mean=float(historical.f1w_macro), reconstructed_mean=float(pvbase.mean_temporal_f1_macro), manual_sessions=len(session_inventory), manual_blocks=sum((s['blocks'] for s in session_inventory)), unavailable_development_probabilities=[i['unavailable_artifact'] for i in inventory if i['scope'] == 'PV_heldout'], python=sys.version, platform=platform.platform(), analysis_seconds=time.perf_counter() - start, sources_unchanged=all((sha(p) == h for p, h in sources.items())), source_hashes=sources))
assert all((sha(p) == h for p, h in sources.items()))
print(aggregate[aggregate.scope == 'PV_heldout'][['policy', 'mean_line_recall', 'mean_line_precision', 'mean_line_temporal_f1', 'mean_scrum_temporal_f1', 'mean_temporal_f1_macro', 'sum_line_tp', 'sum_line_fp', 'sum_line_recovered', 'sum_line_lost', 'mean_line_fp_per_hour', 'mean_all_far_fp_per_hour']].to_string(index=False))
