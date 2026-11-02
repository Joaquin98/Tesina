"""Post-hoc sensitivity. Reads saved predictions only; writes beside this file.
Run: python models/common/studies/sensibilidad_tolerancia_temporal/analyze.py audit|sensitivity
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys, json, hashlib, platform, itertools
from dataclasses import asdict
from pathlib import Path
sys.dont_write_bytecode = True
OUT = experiment_directory(__file__)
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, kendalltau
from models.common.comparison import metrics as m
BASE = ROOT / 'data/final_model_comparison_20260827/outputs'
TAUS = [7, 14, 21, 28, 35]
MAIN = ['M0-H2-D2', 'M1-H2-D2', 'M2-H2-D2', 'M3-H2-D2', 'M3-H1-D2']

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def sha(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()

def csv(name, rows):
    f = pd.DataFrame(rows)
    f.to_csv(OUT / name, index=False, float_format='%.15g')
    return f

def chosen():
    r = pd.read_csv(_resolve_model_artifact(BASE / 'run_registry.csv'))
    r = r[(r.status == 'completa') & (r.panel == 'PV') & r.model.isin(['M0', 'M1', 'M2', 'M3']) & r.level.isin(['H0', 'H1', 'H2'])]
    r = r[r.run_id.str.match('^PV-M[0-3]-H[0-2]-D[0-2]-PV0[1-6]-S42$')]
    assert len(r) == 216 and (not r.run_id.duplicated().any()), (len(r), 'ambiguous/missing completed runs')
    return r.sort_values('run_id')

def runpath(row):
    return BASE / 'runs' / row.run_id / f'attempt_{int(row.attempt):03d}'

def snapshot(r):
    paths = [BASE / x for x in ['run_registry.csv', 'comparison_summary.csv', 'completed_fold_results.csv', 'master_examples.csv']]
    paths += list(_model_artifact_glob(ROOT / 'data/final_model_comparison_20260827', '*.py', recursive=False))
    paths += list(_model_artifact_glob(ROOT / 'docs/tesina_con_correcciones', '*.tex', recursive=True))
    paths += [ROOT / 'docs/tesina_con_correcciones/comment_history.md', ROOT / 'docs/PROTOCOLO_MEDICION_Y_REGISTRO_AFINADO_20260827.md', ROOT / 'docs/PLAN_EXPERIMENTAL_AFINADO_CINCO_MODELOS_20260827.md']
    paths += [runpath(row) / x for row in r.itertuples() for x in ['run.json', 'metrics.json', 'predictions.csv']]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}

def scores(tp, nt, npred):
    return dict(precision=tp / npred if npred else 0.0, recall=tp / nt if nt else 0.0, f1=2 * tp / (nt + npred) if nt + npred else 1.0)

def audit():
    r = chosen()
    before = snapshot(r)
    dump('protected_before.json', before)
    master = pd.read_csv(_resolve_model_artifact(BASE / 'master_examples.csv')).set_index('block_id')
    detail = pd.read_csv(_resolve_model_artifact(BASE / 'completed_fold_results.csv')).set_index('run_id')
    checks = []
    events = []
    provenance = []
    original = []
    truth_by_group = {}
    for row in r.itertuples():
        path = runpath(row)
        meta = json.loads(_resolve_model_artifact(path / 'run.json').read_text())
        saved = json.loads(_resolve_model_artifact(path / 'metrics.json').read_text())
        f = pd.read_csv(_resolve_model_artifact(path / 'predictions.csv'), float_precision='round_trip')
        f['group_id'] = f.test_group
        assert meta['smoke_limit'] is None and len(f) == meta['n_test']
        assert set(f.run_id) == {row.run_id} and set(f.test_group) == {meta['test_group']}
        assert meta['test_group'] not in meta['train_groups'] + [meta['development_group']]
        assert not f.duplicated(['group_id', 'section', 'block_index']).any()
        expected = master.loc[f.block_id]
        assert np.array_equal(f.y_true.to_numpy(), expected.label.to_numpy())
        for col in ['start_s', 'end_s', 'block_index', 'section']:
            assert np.array_equal(f[col].to_numpy(), expected[col].to_numpy()), col
        key = meta['test_group']
        truth = f[['block_id', 'y_true', 'start_s', 'end_s', 'section', 'block_index']].to_csv(index=False)
        assert key not in truth_by_group or truth_by_group[key] == truth
        truth_by_group[key] = truth
        value = m.temporal_metrics(f)
        conf = f'{row.model}-{row.level}-{row.data_regime}'
        original.append(dict(configuration=conf, match=key, macro=value['weighted_f1_macro'], line=value['line']['weighted_f1'], scrum=value['scrum']['weighted_f1']))
        for label in ['line', 'scrum']:
            for metric in ['weighted_f1', 'weighted_precision', 'weighted_recall', 'far_false_positives_per_hour', 'true_events', 'predicted_events', 'matched_events']:
                a = value[label][metric]
                b = saved['temporal'][label][metric]
                checks.append(dict(run_id=row.run_id, label=label, metric=metric, reconstructed=a, saved=b, absolute_difference=abs(a - b)))
            assert abs(value[label]['weighted_f1'] - detail.loc[row.run_id, f'f1w_{label}']) < 1e-12
            t = m.events_from_blocks(f, label, prediction_column='y_true', probability_column=f'p_{label}')
            p = m.events_from_blocks(f, label, prediction_column='y_final', probability_column=f'p_{label}')
            duration = float(f.groupby(['group_id', 'section'], sort=False).end_s.max().sum())
            events.append(dict(configuration=conf, run_id=row.run_id, match=key, label=label, seconds=duration, truth=[asdict(e) for e in t], prediction=[asdict(e) for e in p], historical_hard_f1=value[label]['f1_at_seconds']))
        provenance.append(dict(run_id=row.run_id, attempt=int(row.attempt), rows=len(f), test=key, development=meta['development_group'], train=meta['train_groups'], thresholds=meta['thresholds'], checkpoint_hash=meta['checkpoint_hash'], cache_signature=meta['cache_signature'], model_metadata=meta.get('model_metadata'), prediction_sha256=sha(path / 'predictions.csv')))
    checks = csv('audit_35_per_run.csv', checks)
    orig = csv('audit_35_reconstruction.csv', original)
    mean = orig.groupby('configuration')[['macro', 'line', 'scrum']].mean()
    summary = pd.read_csv(_resolve_model_artifact(BASE / 'comparison_summary.csv'))
    summary = summary[(summary.panel == 'PV') & summary.model.isin(['M0', 'M1', 'M2', 'M3']) & summary.level.isin(['H0', 'H1', 'H2'])]
    summary['configuration'] = summary.model + '-' + summary.level + '-' + summary.data_regime
    comparisons = []
    for row in summary.itertuples():
        for col in ['macro', 'line', 'scrum']:
            a = mean.loc[row.configuration, col]
            b = getattr(row, f'f1w_{col}')
            comparisons.append(dict(configuration=row.configuration, metric=col, reconstructed=a, historical_summary=b, absolute_difference=abs(a - b), rounded_3=f'{a:.3f}'))
    comparisons = csv('audit_35_summary.csv', comparisons)
    assert checks.absolute_difference.max() < 1e-12
    assert comparisons.absolute_difference.max() < 1e-12
    dump('frozen_events.json', events)
    dump('provenance_runs.json', provenance)
    dump('parameters.json', dict(tolerances_seconds=TAUS, models=['M0', 'M1', 'M2', 'M3'], levels=['H0', 'H1', 'H2'], regimes=['D0', 'D1', 'D2'], matches=6, seed_original=42, main_configurations=MAIN, policy='All saved y_final, probabilities, thresholds, verifier decisions, groups and anchors frozen. No training, inference or threshold search.', hard_matching='maximum cardinality with distance <= tau; ties minimize total distance', interval_matching='maximum cardinality with tIoU >= threshold; ties maximize summed tIoU', uncertainty='No confidence intervals; descriptive six-match paired and leave-one-match-out analysis.'))
    dump('audit_status.json', dict(passed=True, runs=len(r), configurations=len(mean), max_abs_error_saved_metrics=float(checks.absolute_difference.max()), max_abs_error_summary=float(comparisons.absolute_difference.max()), python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    assert snapshot(r) == before
    print(json.dumps(json.loads(_resolve_model_artifact(OUT / 'audit_status.json').read_text()), indent=2))

def distances(t, p):
    return np.array([[abs(a.anchor - b.anchor) if (a.group, a.section) == (b.group, b.section) else np.inf for b in p] for a in t], dtype=float).reshape(len(t), len(p))

def cardinality(eligible, tie):
    if not eligible.size:
        return []
    reward = eligible.astype(float) * (min(eligible.shape) + 1 + tie)
    i, j = linear_sum_assignment(-reward)
    return [(int(a), int(b)) for a, b in zip(i, j) if eligible[a, b]]

def sensitivity():
    assert json.loads(_resolve_model_artifact(OUT / 'audit_status.json').read_text())['passed']
    assert snapshot(chosen()) == json.loads(_resolve_model_artifact(OUT / 'protected_before.json').read_text())
    rows = []
    pairs = []
    interval = []
    old_hard = []
    event_stats = []
    for entry in json.loads(_resolve_model_artifact(OUT / 'frozen_events.json').read_text()):
        t = [m.Event(**v) for v in entry['truth']]
        p = [m.Event(**v) for v in entry['prediction']]
        d = distances(t, p)
        nt = len(t)
        npred = len(p)
        base = {k: entry[k] for k in ['configuration', 'run_id', 'match', 'label']}
        for kind, ev in [('truth', t), ('prediction', p)]:
            event_stats.extend((dict(**base, kind=kind, duration=e.end - e.start, blocks=len(e.blocks)) for e in ev))
        for tau in TAUS:
            weighted = m.match_events(t, p, max_distance=float(tau))
            hard = cardinality(d <= tau, np.where(np.isfinite(d), np.maximum(0, 1 - d / tau), 0))
            far = sum((not any(d[:, j] <= tau) for j in range(npred)))
            for kind, matched in [('weighted', weighted), ('hard', [(a, b, float(d[a, b]), 1.0) for a, b in hard])]:
                tp = sum((v[3] for v in matched))
                rows.append(dict(**base, tau=tau, metric=kind, true_events=nt, predicted_events=npred, tp=tp, fp=npred - tp, fn=nt - tp, far_fp=far, far_fp_per_hour=far / (entry['seconds'] / 3600), **scores(tp, nt, npred)))
                pairs.extend((dict(**base, tau=tau, metric=kind, truth_index=a, prediction_index=b, distance=dist, credit=w) for a, b, dist, w in matched))
            if str(tau) in entry['historical_hard_f1']:
                old_hard.append(dict(**base, tau=tau, historical_f1=entry['historical_hard_f1'][str(tau)], new_f1=scores(len(hard), nt, npred)['f1']))
        iou = np.array([[m._iou(a, b) if (a.group, a.section) == (b.group, b.section) else 0 for b in p] for a in t]).reshape(nt, npred)
        for threshold in [0.1, 0.25, 0.5]:
            matched = cardinality(iou >= threshold, iou)
            interval.append(dict(**base, tiou=threshold, true_events=nt, predicted_events=npred, tp=len(matched), **scores(len(matched), nt, npred)))
    df = csv('per_match_per_class.csv', rows)
    csv('assignments.csv', pairs)
    csv('event_durations.csv', event_stats)
    old = csv('historical_hard_comparison.csv', old_hard)
    csv('tiou_per_match_per_class.csv', interval)
    agg = df.groupby(['configuration', 'tau', 'metric', 'label'])[['precision', 'recall', 'f1', 'far_fp_per_hour']].mean().reset_index()
    csv('per_class_means.csv', agg)
    fold = df.groupby(['configuration', 'tau', 'metric', 'match']).agg(f1_macro=('f1', 'mean'), far_fp_per_hour=('far_fp_per_hour', 'sum')).reset_index()
    csv('per_match_macro.csv', fold)
    overall = fold.groupby(['configuration', 'tau', 'metric'])[['f1_macro', 'far_fp_per_hour']].mean().reset_index()
    overall['rank'] = overall.groupby(['tau', 'metric']).f1_macro.rank(ascending=False, method='min').astype(int)
    ranks35 = overall[overall.tau == 35].set_index(['configuration', 'metric'])['rank']
    overall['rank_at_35'] = [ranks35.loc[c, k] for c, k in zip(overall.configuration, overall.metric)]
    overall['rank_change_vs_35'] = overall.rank_at_35 - overall['rank']
    csv('ranking_all_36.csv', overall.sort_values(['metric', 'tau', 'rank']))
    csv('main_models_per_match.csv', df[df.configuration.isin(MAIN)])
    interval_df = pd.DataFrame(interval)
    csv('tiou_class_means.csv', interval_df.groupby(['configuration', 'tiou', 'label'])[['precision', 'recall', 'f1']].mean().reset_index())
    csv('tiou_macro_means.csv', interval_df.groupby(['configuration', 'tiou']).f1.mean().reset_index(name='f1_macro'))
    w = overall[overall.metric == 'weighted']
    ref = w[w.tau == 35].set_index('configuration')
    stability = []
    for tau in TAUS:
        a = w[w.tau == tau].set_index('configuration').loc[ref.index]
        stability.append(dict(tau=tau, spearman=float(spearmanr(ref.f1_macro, a.f1_macro).statistic), kendall=float(kendalltau(ref.f1_macro, a.f1_macro).statistic), top5_overlap=len(set(ref.nlargest(5, 'f1_macro').index) & set(a.nlargest(5, 'f1_macro').index)), leader=a.f1_macro.idxmax(), max_absolute_rank_change=int((ref['rank'] - a['rank']).abs().max())))
    csv('rank_stability.csv', stability)
    paired = []
    loo = []
    for tau in TAUS:
        x = fold[(fold.tau == tau) & (fold.metric == 'weighted')].pivot(index='match', columns='configuration', values='f1_macro')
        delta = x['M1-H2-D2'] - x['M3-H2-D2']
        paired.extend((dict(tau=tau, match=k, m1_minus_m3=v) for k, v in delta.items()))
        loo.extend((dict(tau=tau, omitted_match=k, m1_minus_m3_mean=float(delta.drop(k).mean())) for k in delta.index))
    csv('paired_m1_m3.csv', paired)
    csv('leave_one_match_out_m1_m3.csv', loo)
    assert df.groupby(['configuration', 'match', 'label', 'metric']).true_events.nunique().max() == 1
    assert df.groupby(['configuration', 'match', 'label', 'metric']).predicted_events.nunique().max() == 1
    assert df.sort_values('tau').groupby(['configuration', 'match', 'label', 'metric']).f1.diff().dropna().min() > -1e-12
    assert (old.new_f1 >= old.historical_f1 - 1e-12).all()
    after = snapshot(chosen())
    before = json.loads(_resolve_model_artifact(OUT / 'protected_before.json').read_text())
    dump('protected_verification.json', dict(passed=after == before, files=len(after), changes=[p for p in before if before[p] != after[p]]))
    assert after == before
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    colors = ['#777777', '#0072B2', '#009E73', '#D55E00', '#CC79A7']
    for conf, color in zip(MAIN, colors):
        data = agg[(agg.configuration == conf) & (agg.metric == 'weighted')]
        a = w[w.configuration == conf].sort_values('tau')
        axes[0].plot(a.tau, a.f1_macro, 'o-', color=color, label=conf, lw=2)
        for ax, label in zip(axes[1:], ['line', 'scrum']):
            b = data[data.label == label].sort_values('tau')
            ax.plot(b.tau, b.f1, 'o-', color=color, lw=2)
    for ax, title in zip(axes, ['Promedio de las dos clases', 'Lineout', 'Scrum']):
        ax.set(title=title, xlabel='Tolerancia máxima (s)', xticks=TAUS, ylim=(0.15, 0.7))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel('F1 temporal ponderado')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Sensibilidad posterior con detecciones fijas · media de seis partidos')
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(OUT / 'sensibilidad_principales.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'sensibilidad_principales.svg', bbox_inches='tight')
    plt.close(fig)
    print(w[w.configuration.isin(MAIN)].pivot(index='configuration', columns='tau', values='f1_macro').round(6).to_string())
    print(pd.DataFrame(stability).to_string(index=False))
    print('Historical hard differences:', int((old.new_f1 > old.historical_f1 + 1e-12).sum()))
if __name__ == '__main__':
    {'audit': audit, 'sensitivity': sensitivity}[sys.argv[1]]()
