"""Two additional absolute lineout thresholds on VF S01-S04, with argmax scrum preserved."""
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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from models.common.comparison import metrics as m
HERE = experiment_directory(__file__)
OUT = HERE / 'results'
FIG = HERE / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
PREVIOUS = ROOT / 'data/m1_argmax_to_line_thresholds_vf_20261022'
verification = json.loads(_resolve_model_artifact(PREVIOUS / 'verification.json').read_text())
inventory = json.loads(_resolve_model_artifact(PREVIOUS / 'inventory.json').read_text())
source_hashes = {}
frames = []
for item in inventory['sources']:
    path = Path(item['probability_path'])
    digest = hashlib.sha256(_resolve_model_artifact(path).read_bytes()).hexdigest()
    assert digest == verification['source_hashes'][str(path)]
    source_hashes[str(path)] = digest
    f = pd.read_csv(_resolve_model_artifact(path))
    assert f.section.eq(item['section']).all()
    frames.append(f)
f = pd.concat(frames, ignore_index=True)
assert len(f) == 688 and set(f.section) == {'S01', 'S02', 'S03', 'S04'}
for path in inventory['model']['folds']:
    for key in ['dual_model', 'geometry_rf']:
        pth = ROOT / path[key]
        digest = hashlib.sha256(_resolve_model_artifact(pth).read_bytes()).hexdigest()
        assert digest == verification['source_hashes'][str(pth)]
        source_hashes[str(pth)] = digest
p = m.normalize_probabilities(f[['p_line', 'p_other', 'p_scrum']].to_numpy())
argmax = p.argmax(1)
labels = np.array(['line', 'other', 'scrum'])
hours = float(f.duration.sum() / 3600)
offsets = {}
seconds = 0.0
for sec, g in f.groupby('section', sort=True):
    offsets[sec] = seconds
    seconds += float(g.end_s.max())
assert np.isclose(seconds, 4804.8)
truth = {c: m.events_from_blocks(f, c, prediction_column='y_true', probability_column='p_' + c) for c in ['line', 'scrum']}
RULES = [('argmax', None), ('threshold', 0.45), ('threshold', 0.35), ('threshold', 0.25), ('threshold', 0.15)]
config = dict(model=inventory['model']['source_configuration'], match='VF', sections=['S01', 'S02', 'S03', 'S04'], S05=False, new_thresholds=[0.25, 0.15], reference='argmax', controls=[0.45, 0.35], policy='Preserve every argmax scrum. On other blocks, line iff p_line >= t, else other. No requirement that line beats other.', invariants='fixed probabilities and labels, same weights, grouping and anchors, max temporal-credit one-to-one assignment at 35 s', interpretation='Further exploration on already inspected VF labels; not an independently validated selection.')
(HERE / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
rows = []
section_rows = []
events = []
truth_changes = []
candidate_changes = []
derived = []
states = []
base_sets = {}
previous_sets = {}
for policy, t in RULES:
    if policy == 'argmax':
        y = argmax.copy()
    else:
        y = np.full(len(f), 1, dtype=int)
        y[argmax == 2] = 2
        y[(p[:, 0] >= t) & (argmax != 2)] = 0
    assert np.array_equal(y == 2, argmax == 2)
    d = f.copy()
    d['y_argmax'] = labels[argmax]
    d['y_final'] = labels[y]
    derived.append(d.assign(policy=policy, line_threshold=t))
    tm = m.temporal_metrics(d)
    row = dict(policy=policy, line_threshold=t, hours=hours, temporal_f1_macro=tm['weighted_f1_macro'])
    details = {}
    for c in ['line', 'scrum']:
        tr = truth[c]
        pr = m.events_from_blocks(d, c, prediction_column='y_final', probability_column='p_' + c)
        pa = m.match_events(tr, pr)
        mt = {z[0] for z in pa}
        mp = {z[1]: z for z in pa}
        if policy == 'argmax':
            base_sets[c] = mt
            previous_sets[c] = mt
        recovered = mt - base_sets[c]
        lost = base_sets[c] - mt
        step_recovered = mt - previous_sets[c]
        step_lost = previous_sets[c] - mt
        tp = len(pa)
        fp = len(pr) - tp
        fn = len(tr) - tp
        far = fp - tm[c]['duplicates']
        values = dict(real=len(tr), tp=tp, fp=fp, fn=fn, precision=tp / len(pr) if pr else 0, recall=tp / len(tr) if tr else 0, f1=2 * tp / (len(tr) + len(pr)) if len(tr) + len(pr) else 1, temporal_f1=tm[c]['weighted_f1'], weighted_tp=sum((z[3] for z in pa)), fp_per_hour=fp / hours, far_fp=far, far_fp_per_hour=far / hours, duplicates=tm[c]['duplicates'], candidates=len(pr), candidate_seconds=sum((e.end - e.start for e in pr)), fusions=tm[c]['fusions'], fragmentations=tm[c]['fragmentations'], zero_credit_pairs=sum((z[3] == 0 for z in pa)), recovered_vs_argmax=len(recovered), lost_vs_argmax=len(lost), recovered_vs_previous=len(step_recovered), lost_vs_previous=len(step_lost))
        row.update({c + '_' + k: v for k, v in values.items()})
        for tid, e in enumerate(tr):
            truth_changes.append(dict(policy=policy, line_threshold=t, label=c, truth_id=tid, section=e.section, start_s=e.start, end_s=e.end, global_start_s=offsets[e.section] + e.start, global_end_s=offsets[e.section] + e.end, detected=tid in mt, recovered_vs_argmax=tid in recovered, lost_vs_argmax=tid in lost, recovered_vs_previous=tid in step_recovered, lost_vs_previous=tid in step_lost))
        for j, e in enumerate(pr):
            pair = mp.get(j)
            near = any((a.section == e.section and abs(a.anchor - e.anchor) <= 35 for a in tr))
            events.append(dict(policy=policy, line_threshold=t, label=c, event_id=j, section=e.section, start_s=e.start, end_s=e.end, anchor_s=e.anchor, global_start_s=offsets[e.section] + e.start, global_end_s=offsets[e.section] + e.end, status='TP' if pair else 'duplicate_FP' if near else 'far_FP', truth_id=pair[0] if pair else None, distance_s=pair[2] if pair else None, weighted_credit=pair[3] if pair else 0))
        details[c] = (pr, pa, mt, mp)
        previous_sets[c] = mt
    for sec, g in d.groupby('section', sort=True):
        stm = m.temporal_metrics(g)
        r = dict(section=sec, policy=policy, line_threshold=t, temporal_f1_macro=stm['weighted_f1_macro'])
        for c in ['line', 'scrum']:
            st = [a for a in truth[c] if a.section == sec]
            sp = [a for a in details[c][0] if a.section == sec]
            sm = m.match_events(st, sp)
            r.update({c + '_real': len(st), c + '_tp': len(sm), c + '_fp': len(sp) - len(sm), c + '_fn': len(st) - len(sm), c + '_temporal_f1': stm[c]['weighted_f1']})
        section_rows.append(r)
    row['all_fp_per_hour'] = row['line_fp_per_hour'] + row['scrum_fp_per_hour']
    row['all_far_fp_per_hour'] = row['line_far_fp_per_hour'] + row['scrum_far_fp_per_hour']
    states.append((policy, t, details['line']))
    rows.append(row)
metrics = pd.DataFrame(rows)
old = pd.read_csv(_resolve_model_artifact(PREVIOUS / 'results/match_metrics.csv'))
for r in rows[:3]:
    o = old[old.policy == ('argmax' if r['policy'] == 'argmax' else 'scrum_priority')]
    if r['line_threshold'] is not None:
        o = o[np.isclose(o.line_threshold, r['line_threshold'])]
    assert len(o) == 1
    for col in ['line_tp', 'line_fp', 'line_fn', 'line_temporal_f1', 'scrum_temporal_f1', 'temporal_f1_macro']:
        assert np.isclose(r[col], o.iloc[0][col]), col
for r in rows:
    assert r['line_tp'] + r['line_fn'] == 30 and r['line_tp'] + r['line_fp'] == r['line_candidates']
    assert np.isclose(r['scrum_temporal_f1'], rows[0]['scrum_temporal_f1'])
    assert r['scrum_tp'] == rows[0]['scrum_tp'] and r['scrum_fp'] == rows[0]['scrum_fp']
metrics.to_csv(OUT / 'match_metrics.csv', index=False)
pd.DataFrame(section_rows).to_csv(OUT / 'metrics_by_section.csv', index=False)
changes = pd.DataFrame(truth_changes)
changes.to_csv(OUT / 'truth_event_outcomes.csv', index=False)
changes[changes.recovered_vs_argmax | changes.lost_vs_argmax].to_csv(OUT / 'recovered_lost_vs_argmax.csv', index=False)
changes[changes.recovered_vs_previous | changes.lost_vs_previous].to_csv(OUT / 'changes_vs_previous_threshold.csv', index=False)
pd.DataFrame(events).to_csv(OUT / 'candidate_events.csv', index=False)
pd.concat(derived, ignore_index=True).to_csv(OUT / 'derived_predictions.csv.gz', index=False, compression='gzip')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'text.color': '#202a35'})
colors = dict(real='#64748b', tp='#1679a7', new='#25865d', fp='#dc7a18', miss='#be3445')
fig, ax = plt.subplots(figsize=(21, 10))
n = len(RULES)
for y in range(n + 1):
    ax.axhline(y, color='#f0f3f6', linewidth=28, zorder=0)
for e in truth['line']:
    ax.broken_barh([((offsets[e.section] + e.start) / 60, (e.end - e.start) / 60)], (n - 0.17, 0.34), facecolors=colors['real'])
    ax.scatter([(offsets[e.section] + e.anchor) / 60], [n], s=24, color=colors['real'])
for k, ((policy, t, (pr, pa, mt, mp)), r) in enumerate(zip(states, rows)):
    y = n - 1 - k
    for j, e in enumerate(pr):
        color = colors['new' if j in mp and mp[j][0] not in base_sets['line'] else 'tp' if j in mp else 'fp']
        start = (offsets[e.section] + e.start) / 60
        anchor = (offsets[e.section] + e.anchor) / 60
        ax.broken_barh([(start, (e.end - e.start) / 60)], (y - 0.17, 0.34), facecolors=color)
        ax.scatter([anchor], [y], s=24, color=color)
    for j, e in enumerate(truth['line']):
        if j not in mt:
            ax.scatter([(offsets[e.section] + e.anchor) / 60], [y], marker='x', s=64, color=colors['miss'], linewidths=1.8)
    ax.text(seconds / 60 + 0.7, y, f"{r['line_tp']}/30 detectados\n{r['line_fp']} falsas alarmas\n{r['line_fn']} omitidos", va='center', fontsize=12)
for i, (sec, left) in enumerate(offsets.items()):
    right = list(offsets.values())[i + 1] if i < 3 else seconds
    if left:
        ax.axvline(left / 60, color='#aab5c1', linestyle='--', linewidth=1)
    ax.text((left + right) / 120, n + 0.57, sec, ha='center', fontsize=12, color=colors['real'])

def fmt(s):
    v = round(s)
    return f'{v // 60:02d}:{v % 60:02d}'
ax.set_yticks(list(range(n, -1, -1)), ['Lineouts reales', 'Argmax', 'Umbral 0.45', 'Umbral 0.35', 'Umbral 0.25 · nuevo', 'Umbral 0.15 · nuevo'], fontsize=14)
ax.tick_params(axis='y', length=0, pad=12)
ax.set(xlim=(0, seconds / 60), ylim=(-0.5, n + 1), xlabel='Tiempo acumulado del video (min:s)')
ticks = list(np.arange(0, 80, 5)) + [seconds / 60]
ax.set_xticks(ticks, [fmt(v * 60) for v in ticks], fontsize=11)
ax.xaxis.grid(True, color='#e0e6ed', linewidth=0.8)
ax.spines[['left', 'right', 'top']].set_visible(False)
ax.spines['bottom'].set_color('#dce3ea')
fig.suptitle('VF · 80:05 · dos pruebas adicionales: umbrales 0.25 y 0.15', x=0.035, y=0.975, ha='left', fontsize=24)
fig.text(0.035, 0.925, 'Mismo M1-H2-D2 · probabilidades y etiquetas fijas · scrum conservado · S01–S04 · sin S05', fontsize=14, color=colors['real'])
fig.legend(handles=[Patch(facecolor=colors['real'], label='Lineout real'), Patch(facecolor=colors['tp'], label='Detectado'), Patch(facecolor=colors['new'], label='Recuperado frente a argmax'), Patch(facecolor=colors['fp'], label='Falsa alarma'), Line2D([], [], marker='x', color=colors['miss'], linestyle='None', label='Omitido')], ncol=5, frameon=False, loc='lower center', bbox_to_anchor=(0.51, 0.055), fontsize=12)
fig.text(0.035, 0.018, 'Comparación exploratoria. Eventos y emparejamientos se recalculan en cada umbral; tolerancia temporal de 35 s.', fontsize=12, color=colors['real'])
fig.subplots_adjust(left=0.16, right=0.81, top=0.88, bottom=0.17)
fig.savefig(FIG / 'VF_80min_lower_025_015.png', dpi=180, facecolor='white')
plt.close(fig)
assert all((hashlib.sha256(_resolve_model_artifact(Path(p)).read_bytes()).hexdigest() == h for p, h in source_hashes.items()))
(HERE / 'verification.json').write_text(json.dumps(dict(passed=True, controls_reproduced=True, probabilities_and_weights_unchanged=True, scrum_decisions_unchanged=True, blocks=688, seconds=seconds, S05=False, source_hashes=source_hashes), indent=2), encoding='utf-8')
cols = ['policy', 'line_threshold', 'line_real', 'line_tp', 'line_fp', 'line_fn', 'line_precision', 'line_recall', 'line_f1', 'line_temporal_f1', 'temporal_f1_macro', 'line_fp_per_hour', 'line_far_fp_per_hour', 'line_recovered_vs_argmax', 'line_lost_vs_argmax', 'line_recovered_vs_previous', 'line_lost_vs_previous']
report = ['VF: DOS PRUEBAS ADICIONALES DE UMBRAL', '', 'Se añaden 0.25 y 0.15 a la regla de umbral absoluto para lineout. La referencia sigue siendo argmax. Se conserva cada scrum de argmax; en los otros bloques se acepta lineout cuando p_line >= t, aunque other tenga una probabilidad mayor.', 'Se reutilizan los 688 bloques (4804.8 s) del mismo modelo único ALL-M1-H2-D2-S42 y sus cuatro secciones VF S01-S04. S05 queda fuera del alcance de esta evaluación. No se entrenó ni se ejecutó inferencia.', 'Se mantienen agrupación por sección, anclas, asignación uno a uno y tolerancia de 35 s. El F1 temporal usa crédito por distancia; los TP/FP/FN de la tabla son enteros. Los controles argmax, 0.45 y 0.35 reproducen exactamente el análisis anterior.', '', metrics[cols].to_string(index=False), '', 'La comparación sigue siendo exploratoria: los mismos datos VF ya se han inspeccionado y sus etiquetas se utilizan para describir el intercambio entre recuperación y falsas alarmas. No se presenta como validación independiente de un nuevo umbral.']
(HERE / 'reporte.txt').write_text('\n'.join(report), encoding='utf-8')
print(metrics[cols].to_string(index=False))
