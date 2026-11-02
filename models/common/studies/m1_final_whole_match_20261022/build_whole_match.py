"""One continuous whole-video timeline, retaining honest label coverage for S05."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, sys, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from models.common.comparison import metrics as m
HERE = experiment_directory(__file__)
OUT = HERE / 'results'
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
SOURCE = ROOT / 'data/m1_argmax_to_line_thresholds_vf_20261022'
derived = pd.read_csv(_resolve_model_artifact(SOURCE / 'results/derived_predictions.csv.gz'))
old_metrics = pd.read_csv(_resolve_model_artifact(SOURCE / 'results/match_metrics.csv'))
tail = pd.read_csv(_resolve_model_artifact(OUT / 'S05_probabilities.csv'))
completion = json.loads(_resolve_model_artifact(OUT / 'completion.json').read_text())
assert completion['success'] and completion['same_weights_and_implementation_as_S01_S04']
manifest = json.loads(_resolve_model_artifact(ROOT / 'data/raw/youtube/Jose_Ricardo_Aguero/manifest_JneZwNgtDYI.json').read_text())
sections = [f'S0{i}' for i in range(1, 6)]
offsets = {}
cursor = 0
for s in sections:
    offsets[s] = cursor
    cursor += float(derived[(derived.section == s) & (derived.policy == 'argmax')].end_s.max()) if s != 'S05' else float(tail.end_s.max())
duration = cursor
assert abs(duration - manifest['original']['duration_seconds']) < 0.1
base = derived[derived.policy == 'argmax'].copy()
truth = m.events_from_blocks(base, 'line', prediction_column='y_true', probability_column='p_line')
policies = [('argmax', None), ('scrum_priority', 0.45), ('scrum_priority', 0.35)]
rows = []
events = []
decisions = []
states = []
for policy, t in policies:
    known = derived[derived.policy == policy].copy()
    metric = old_metrics[old_metrics.policy == policy]
    if t is not None:
        known = known[np.isclose(known.line_threshold, t)]
        metric = metric[np.isclose(metric.line_threshold, t)]
    assert len(known) == 688 and len(metric) == 1
    metric = metric.iloc[0]
    p = m.normalize_probabilities(tail[['p_line', 'p_other', 'p_scrum']].to_numpy())
    argmax = p.argmax(1)
    if policy == 'argmax':
        y = argmax
    else:
        y = np.full(len(tail), 1, dtype=int)
        y[argmax == 2] = 2
        y[(p[:, 0] >= t) & (argmax != 2)] = 0
    td = tail.copy()
    td['y_final'] = np.array(['line', 'other', 'scrum'])[y]
    td['policy'] = policy
    td['line_threshold'] = t
    assert np.array_equal(td.y_final.eq('scrum'), np.array(['line', 'other', 'scrum'])[argmax] == 'scrum')
    pred = m.events_from_blocks(known, 'line', prediction_column='y_final', probability_column='p_line')
    pairs = m.match_events(truth, pred)
    tailpred = m.events_from_blocks(td, 'line', prediction_column='y_final', probability_column='p_line')
    assert len(pairs) == metric.line_tp and len(pred) - len(pairs) == metric.line_fp
    matched_t = {z[0] for z in pairs}
    matched_p = {z[1] for z in pairs}
    for j, e in enumerate(pred):
        events.append(dict(policy=policy, line_threshold=t, section=e.section, start_s=e.start, end_s=e.end, global_start_s=offsets[e.section] + e.start, global_end_s=offsets[e.section] + e.end, global_anchor_s=offsets[e.section] + e.anchor, status='TP' if j in matched_p else 'FP', human_label_available=True))
    for e in tailpred:
        events.append(dict(policy=policy, line_threshold=t, section=e.section, start_s=e.start, end_s=e.end, global_start_s=offsets[e.section] + e.start, global_end_s=offsets[e.section] + e.end, global_anchor_s=offsets[e.section] + e.anchor, status='unverified_candidate', human_label_available=False))
    for frame in [known, td]:
        frame = frame.copy()
        frame['global_start_s'] = frame.section.map(offsets) + frame.start_s
        frame['global_end_s'] = frame.section.map(offsets) + frame.end_s
        frame['label_status'] = np.where(frame.section == 'S05', 'unlabeled', 'human_verified_source')
        decisions.append(frame)
    rows.append(dict(policy=policy, line_threshold=t, whole_video_seconds=duration, total_candidates=len(pred) + len(tailpred), labeled_seconds=offsets['S05'], labeled_real_lineouts=len(truth), labeled_tp=len(pairs), labeled_fp=len(pred) - len(pairs), labeled_fn=len(truth) - len(pairs), unlabeled_seconds=float(tail.duration.sum()), unverified_tail_candidates=len(tailpred), full_match_recall=None, full_match_precision=None, reason='S05 lacks human truth; candidate counts cover all video, event accuracy only covers S01-S04.'))
    states.append((policy, t, pred, tailpred, matched_t, matched_p))
pd.DataFrame(rows).to_csv(OUT / 'whole_match_summary.csv', index=False)
pd.DataFrame(events).to_csv(OUT / 'whole_match_candidate_events.csv', index=False)
pd.concat(decisions, ignore_index=True).to_csv(OUT / 'whole_match_decisions.csv.gz', index=False, compression='gzip')
pd.DataFrame([dict(section=e.section, start_s=e.start, end_s=e.end, global_start_s=offsets[e.section] + e.start, global_end_s=offsets[e.section] + e.end, global_anchor_s=offsets[e.section] + e.anchor) for e in truth]).to_csv(OUT / 'human_truth_global.csv', index=False)

def fmt(s):
    n = round(s)
    return f'{n // 60:02d}:{n % 60:02d}'
colors = dict(real='#64748b', tp='#1679a7', fp='#dc7a18', miss='#be3445', unverified='#8054a3')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'text.color': '#202a35'})
fig, ax = plt.subplots(figsize=(22, 8))
end = duration / 60
tailstart = offsets['S05'] / 60
ax.axvspan(tailstart, end, color=colors['unverified'], alpha=0.08, zorder=0)
for y in range(4):
    ax.axhline(y, color='#f0f3f6', linewidth=26, zorder=0)
for e in truth:
    start = (offsets[e.section] + e.start) / 60
    anchor = (offsets[e.section] + e.anchor) / 60
    ax.broken_barh([(start, (e.end - e.start) / 60)], (2.82, 0.36), facecolors=colors['real'], zorder=3)
    ax.scatter([anchor], [3], s=20, color=colors['real'], zorder=4)
ax.text((tailstart + end) / 2, 3.28, 'Sin etiquetas', ha='center', fontsize=10, color=colors['unverified'])
for k, ((policy, t, pred, tailpred, mt, mp), r) in enumerate(zip(states, rows)):
    y = 2 - k
    for j, e in enumerate(pred):
        start = (offsets[e.section] + e.start) / 60
        anchor = (offsets[e.section] + e.anchor) / 60
        color = colors['tp' if j in mp else 'fp']
        ax.broken_barh([(start, (e.end - e.start) / 60)], (y - 0.18, 0.36), facecolors=color, zorder=3)
        ax.scatter([anchor], [y], s=17, color=color, zorder=4)
    for j, e in enumerate(truth):
        if j not in mt:
            ax.scatter([(offsets[e.section] + e.anchor) / 60], [y], marker='x', s=57, linewidths=1.7, color=colors['miss'], zorder=5)
    for e in tailpred:
        start = (offsets['S05'] + e.start) / 60
        anchor = (offsets['S05'] + e.anchor) / 60
        ax.broken_barh([(start, (e.end - e.start) / 60)], (y - 0.18, 0.36), facecolors=colors['unverified'], zorder=3)
        ax.scatter([anchor], [y], s=19, color=colors['unverified'], zorder=4)
    ax.text(end + 0.65, y, f"{r['labeled_tp']}/30 detectados · {r['labeled_fp']} FP\n+ {r['unverified_tail_candidates']} candidatos sin revisar", va='center', fontsize=11)
for s in sections:
    left = offsets[s] / 60
    right = offsets[sections[sections.index(s) + 1]] / 60 if s != 'S05' else end
    if left:
        ax.axvline(left, color='#abb6c1', linewidth=0.8, linestyle='--', zorder=1)
    ax.text((left + right) / 2, 3.72, s, ha='center', fontsize=11, color=colors['real'])
ax.set_yticks([3, 2, 1, 0], ['Lineouts reales', 'Argmax', 'Umbral 0.45', 'Umbral 0.35'], fontsize=14)
ax.tick_params(axis='y', length=0, pad=12)
ax.set(xlim=(0, end), ylim=(-0.52, 4.03), xlabel='Tiempo acumulado del video (min:s)')
ticks = list(np.arange(0, 81, 5)) + [end]
ax.set_xticks(ticks, [fmt(x * 60) for x in ticks], fontsize=11)
ax.xaxis.grid(True, color='#e1e6ec', linewidth=0.8)
ax.spines[['left', 'right', 'top']].set_visible(False)
ax.spines['bottom'].set_color('#dce3ea')
fig.suptitle(f'Partido final completo · {fmt(duration)} · una única línea de tiempo', x=0.035, y=0.97, ha='left', fontsize=24)
fig.text(0.035, 0.9, 'Jockey Club VT vs Alma Juniors · M1 único H2-D2 · mismas probabilidades por regla · scrum conservado', fontsize=14, color=colors['real'])
fig.legend(handles=[Patch(facecolor=colors['real'], label='Lineout real'), Patch(facecolor=colors['tp'], label='Detectado'), Patch(facecolor=colors['fp'], label='Falsa alarma'), Line2D([], [], marker='x', color=colors['miss'], linestyle='None', label='Omitido'), Patch(facecolor=colors['unverified'], label='Candidato sin etiqueta (S05)')], ncol=5, frameon=False, loc='lower center', bbox_to_anchor=(0.51, 0.06), fontsize=12)
fig.text(0.035, 0.02, 'S01–S04: referencia manual disponible. S05: probabilidades nuevas del mismo modelo; sin etiquetas humanas, sus candidatos no se clasifican como aciertos o errores.', fontsize=12, color=colors['real'])
fig.subplots_adjust(left=0.13, right=0.81, top=0.84, bottom=0.22)
fig.savefig(FIG / 'VF_whole_83min_continuous_timeline.png', dpi=180, facecolor='white')
plt.close(fig)
fig, ax = plt.subplots(figsize=(11, 4.8))
for k, (_, t, _, tailpred, _, _) in enumerate(states):
    y = 2 - k
    ax.axhline(y, color='#f0f3f6', linewidth=24, zorder=0)
    for e in tailpred:
        ax.broken_barh([(e.start, e.end - e.start)], (y - 0.18, 0.36), facecolors=colors['unverified'])
    ax.text(float(tail.end_s.max()) + 3, y, f'{len(tailpred)} candidatos', va='center', fontsize=12)
ax.set_yticks([2, 1, 0], ['Argmax', 'Umbral 0.45', 'Umbral 0.35'], fontsize=12)
ax.tick_params(axis='y', length=0, pad=12)
ax.set(xlim=(0, float(tail.end_s.max())), ylim=(-0.5, 2.5), xlabel='Tiempo acumulado del video (min:s)')
tick = np.arange(0, float(tail.end_s.max()), 30)
ax.set_xticks(tick, [fmt(offsets['S05'] + v) for v in tick])
ax.xaxis.grid(True, color='#dce3ea')
ax.spines[['left', 'right', 'top']].set_visible(False)
fig.suptitle('Últimos 2:49 · S05 completo', x=0.04, y=0.96, ha='left', fontsize=19)
fig.text(0.04, 0.87, 'Mismo M1 único · candidatos sin referencia manual · todavía no se pueden contar aciertos ni falsas alarmas', fontsize=11, color=colors['unverified'])
fig.subplots_adjust(left=0.17, right=0.8, top=0.79, bottom=0.19)
fig.savefig(FIG / 'VF_S05_missing_tail_predictions.png', dpi=170, facecolor='white')
plt.close(fig)
report = ['PARTIDO FINAL COMPLETO: VF', '', f'Duración de video cubierta: {duration:.3f} s ({fmt(duration)}). Las secciones S01–S05 se muestran consecutivas en el reloj del archivo original. No se unen eventos entre límites de sección ni se altera la política de evaluación histórica.', 'S01–S04 reutilizan exactamente las probabilidades ya guardadas. Sólo se ejecutó el modelo congelado en S05, que carecía de una secuencia completa de probabilidades. No hubo entrenamiento y los hashes de modelos/extractores/implementación son los mismos que en las ejecuciones originales.', 'S05 no dispone de referencia manual. Sus predicciones se muestran como candidatos sin verificar, nunca como verdaderos o falsos positivos. Las métricas de acierto/omisión se restringen explícitamente a S01–S04. Se dejó una plantilla de revisión de sus 25 bloques.', '', pd.DataFrame(rows).to_string(index=False), '', f"Inicio global de S05: {offsets['S05']:.3f} s ({fmt(offsets['S05'])}). Duración del último tramo con cuadros: {float(tail.duration.sum()):.3f} s. Diferencia con duración audiovisual declarada: {manifest['original']['duration_seconds'] - duration:.3f} s, menor a 0.1 s.", 'La comparación usa argmax como referencia y umbrales absolutos 0.45/0.35 para lineout, conservando scrum cuando es argmax. Los umbrales no se volvieron a seleccionar con el final. La independencia respecto de todas las fuentes históricas CH sigue sin estar completamente verificada.']
(HERE / 'reporte_partido_completo.txt').write_text('\n'.join(report), encoding='utf-8')
(OUT / 'whole_match_coverage.json').write_text(json.dumps(dict(duration_seconds=duration, source_duration_seconds=manifest['original']['duration_seconds'], section_offsets_seconds=offsets, all_five_sections_with_probabilities=True, human_labeled_sections=['S01', 'S02', 'S03', 'S04'], unlabeled_sections=['S05'], full_accuracy_metrics_available=False, source_integrity_verified=completion['source_hashes_unchanged']), indent=2), encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
