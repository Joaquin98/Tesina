"""Continuous S01-S04 timeline, excluding S05 according to the recorded evaluation scope."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from models.common.comparison import metrics as m
HERE = experiment_directory(__file__)
SOURCE = ROOT / 'data/m1_argmax_to_line_thresholds_vf_20261022'
OUT = HERE / 'results'
FIG = HERE / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
frames = pd.read_csv(_resolve_model_artifact(SOURCE / 'results/derived_predictions.csv.gz'))
summary = pd.read_csv(_resolve_model_artifact(SOURCE / 'results/match_metrics.csv'))
baseline = frames[frames.policy == 'argmax'].copy()
assert len(baseline) == 688 and set(baseline.section) == {'S01', 'S02', 'S03', 'S04'}
offsets = {}
cursor = 0.0
for section, s in baseline.groupby('section', sort=True):
    offsets[section] = cursor
    cursor += s.end_s.max()
assert np.isclose(cursor, 4804.8)
truth = m.events_from_blocks(baseline, 'line', prediction_column='y_true', probability_column='p_line')
bp = m.events_from_blocks(baseline, 'line', prediction_column='y_final', probability_column='p_line')
basepairs = m.match_events(truth, bp)
baseline_truth = {z[0] for z in basepairs}
assert len(truth) == 30 and len(basepairs) == 23
colors = {'real': '#64748b', 'tp': '#1679a7', 'new': '#25865d', 'fp': '#dc7a18', 'miss': '#be3445'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'text.color': '#202a35'})
fig, ax = plt.subplots(figsize=(21, 8))
end = cursor / 60
for y in range(4):
    ax.axhline(y, color='#f0f3f6', linewidth=30, zorder=0)
for e in truth:
    ax.broken_barh([((offsets[e.section] + e.start) / 60, (e.end - e.start) / 60)], (2.82, 0.36), facecolors=colors['real'], zorder=3)
    ax.scatter([(offsets[e.section] + e.anchor) / 60], [3], s=25, color=colors['real'], zorder=4)
policies = [('argmax', None), ('scrum_priority', 0.45), ('scrum_priority', 0.35)]
global_events = []
exported = []

def fmt(seconds):
    n = round(seconds)
    return f'{n // 60:02d}:{n % 60:02d}'
for k, (policy, t) in enumerate(policies):
    f = frames[frames.policy == policy].copy()
    r = summary[summary.policy == policy]
    if t is not None:
        f = f[np.isclose(f.line_threshold, t)]
        r = r[np.isclose(r.line_threshold, t)]
    assert len(f) == 688 and len(r) == 1
    r = r.iloc[0]
    pred = m.events_from_blocks(f, 'line', prediction_column='y_final', probability_column='p_line')
    pairs = m.match_events(truth, pred)
    mt = {z[0] for z in pairs}
    mp = {z[1]: z[0] for z in pairs}
    assert len(pairs) == r.line_tp and len(pred) - len(pairs) == r.line_fp
    y = 2 - k
    for j, e in enumerate(pred):
        status = 'recovered' if j in mp and mp[j] not in baseline_truth else 'TP' if j in mp else 'FP'
        color = colors['new' if status == 'recovered' else 'tp' if status == 'TP' else 'fp']
        start = (offsets[e.section] + e.start) / 60
        anchor = (offsets[e.section] + e.anchor) / 60
        ax.broken_barh([(start, (e.end - e.start) / 60)], (y - 0.18, 0.36), facecolors=color, zorder=3)
        ax.scatter([anchor], [y], s=25 if status != 'recovered' else 55, color=color, zorder=4)
        if status == 'recovered':
            ax.annotate('+ ' + fmt(anchor * 60), (anchor, y), xytext=(0, 25), textcoords='offset points', ha='center', fontsize=11, color=colors['new'])
        global_events.append(dict(policy=policy, line_threshold=t, section=e.section, start_s=e.start, end_s=e.end, global_start_s=offsets[e.section] + e.start, global_end_s=offsets[e.section] + e.end, global_anchor_s=anchor * 60, status=status))
    for j, e in enumerate(truth):
        if j not in mt:
            ax.scatter([(offsets[e.section] + e.anchor) / 60], [y], marker='x', s=64, linewidths=1.8, color=colors['miss'], zorder=5)
    ax.text(end + 0.7, y, f'{int(r.line_tp)}/30 detectados\n{int(r.line_fp)} falsas alarmas\n{int(r.line_fn)} omitidos', va='center', fontsize=12)
    f['global_start_s'] = f.section.map(offsets) + f.start_s
    f['global_end_s'] = f.section.map(offsets) + f.end_s
    exported.append(f)
for i, (sec, left_s) in enumerate(offsets.items()):
    left = left_s / 60
    right = list(offsets.values())[i + 1] / 60 if i < 3 else end
    if left:
        ax.axvline(left, color='#aab5c1', linestyle='--', linewidth=1, zorder=1)
    ax.text((left + right) / 2, 3.65, sec, ha='center', fontsize=12, color=colors['real'])
ax.set_yticks([3, 2, 1, 0], ['Lineouts reales', 'Argmax', 'Umbral 0.45', 'Umbral 0.35'], fontsize=15)
ax.tick_params(axis='y', length=0, pad=12)
ax.set(xlim=(0, end), ylim=(-0.5, 4.0), xlabel='Tiempo acumulado del video (min:s)')
ticks = list(np.arange(0, 80, 5)) + [end]
ax.set_xticks(ticks, [fmt(v * 60) for v in ticks], fontsize=11)
ax.xaxis.grid(True, color='#e0e6ed', linewidth=0.8)
ax.spines[['left', 'right', 'top']].set_visible(False)
ax.spines['bottom'].set_color('#dce3ea')
fig.suptitle('Partido final · 80:05 en una única línea de tiempo', x=0.035, y=0.97, ha='left', fontsize=24)
fig.text(0.035, 0.9, 'Jockey Club VT vs Alma Juniors · S01–S04 · mismo M1-H2-D2 y probabilidades · scrum conservado · S05 excluido', fontsize=14, color=colors['real'])
fig.legend(handles=[Patch(facecolor=colors['real'], label='Lineout real'), Patch(facecolor=colors['tp'], label='Detectado'), Patch(facecolor=colors['new'], label='Recuperado frente a argmax'), Patch(facecolor=colors['fp'], label='Falsa alarma'), Line2D([], [], marker='x', color=colors['miss'], linestyle='None', label='Omitido')], ncol=5, frameon=False, loc='lower center', bbox_to_anchor=(0.51, 0.065), fontsize=12)
fig.text(0.035, 0.02, 'Las líneas verticales separan secciones del mismo video. Cada marca conserva su evaluación original por evento, con tolerancia de 35 s.', fontsize=12, color=colors['real'])
fig.subplots_adjust(left=0.13, right=0.81, top=0.84, bottom=0.22)
path = FIG / 'VF_S01_S04_80min_continuous.png'
fig.savefig(path, dpi=180, facecolor='white')
plt.close(fig)
pd.DataFrame(global_events).to_csv(OUT / 'S01_S04_global_events.csv', index=False)
pd.concat(exported, ignore_index=True).to_csv(OUT / 'S01_S04_global_decisions.csv.gz', index=False, compression='gzip')
(OUT / 'S01_S04_coverage.json').write_text(json.dumps(dict(duration_seconds=cursor, sections=list(offsets), section_offsets_seconds=offsets, all_probabilities_reused=True, S05_excluded=True, tail_inference_stopped=True, lineout_events=30, global_time_is_video_clock=True, event_matching_still_separate_per_section=True), indent=2), encoding='utf-8')
(HERE / 'S05_cancelled.json').write_text(json.dumps(dict(status='cancelled', reason='S05 is excluded from this evaluation scope.', existing_results_modified=False, tail_predictions_included=False), indent=2), encoding='utf-8')
print(path)
