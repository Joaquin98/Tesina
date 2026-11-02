"""Show both target classes on the same continuous VF clock without changing decisions."""
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
OUT = HERE / 'results'
FIG = HERE / 'figures'
d = pd.read_csv(_resolve_model_artifact(OUT / 'derived_predictions.csv.gz'))
metrics = pd.read_csv(_resolve_model_artifact(OUT / 'match_metrics.csv'))
base = d[d.policy == 'argmax'].copy()
assert len(base) == 688
offsets = {}
seconds = 0.0
for sec, g in base.groupby('section', sort=True):
    offsets[sec] = seconds
    seconds += float(g.end_s.max())
truth = {c: m.events_from_blocks(base, c, prediction_column='y_true', probability_column='p_' + c) for c in ['line', 'scrum']}
assert len(truth['line']) == 30 and len(truth['scrum']) == 10
configs = [('Argmax', 'argmax', None), ('Umbral lineout 0.35', 'threshold', 0.35), ('Umbral lineout 0.25', 'threshold', 0.25), ('Umbral lineout 0.15', 'threshold', 0.15)]
colors = dict(line='#1679a7', scrum='#8054a3', new='#25865d', fp='#dc7a18', miss='#be3445')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'text.color': '#202a35'})
fig, ax = plt.subplots(figsize=(22, 13))
gap = 2.8
top = gap * len(configs)
rows = []
labels = []
positions = []
checks = []
base_matched = {}
for c, y in [('line', top), ('scrum', top - 1)]:
    ax.axhline(y, color='#f0f3f6', linewidth=24, zorder=0)
    for e in truth[c]:
        start = (offsets[e.section] + e.start) / 60
        anchor = (offsets[e.section] + e.anchor) / 60
        ax.broken_barh([(start, (e.end - e.start) / 60)], (y - 0.16, 0.32), facecolors=colors[c], alpha=0.55, zorder=3)
        ax.scatter([anchor], [y], color=colors[c], alpha=0.55, s=22, zorder=4)
    labels.append('Reales · ' + ('Lineout' if c == 'line' else 'Scrum'))
    positions.append(y)
    ax.text(seconds / 60 + 0.6, y, f'{len(truth[c])} jugadas reales', va='center', fontsize=12)
for k, (title, policy, t) in enumerate(configs):
    frame = d[d.policy == policy].copy()
    row = metrics[metrics.policy == policy]
    if t is not None:
        frame = frame[np.isclose(frame.line_threshold, t)]
        row = row[np.isclose(row.line_threshold, t)]
    assert len(frame) == 688 and len(row) == 1
    row = row.iloc[0]
    if policy != 'argmax':
        assert np.array_equal(frame.y_final.eq('scrum').to_numpy(), base.y_final.eq('scrum').to_numpy())
    center = top - gap * (k + 1)
    for c, y in [('line', center), ('scrum', center - 1)]:
        ax.axhline(y, color='#f0f3f6', linewidth=24, zorder=0)
        pred = m.events_from_blocks(frame, c, prediction_column='y_final', probability_column='p_' + c)
        pairs = m.match_events(truth[c], pred)
        mp = {z[1]: z[0] for z in pairs}
        mt = {z[0] for z in pairs}
        if policy == 'argmax':
            base_matched[c] = mt
        assert len(pairs) == row[c + '_tp'] and len(pred) - len(pairs) == row[c + '_fp']
        for j, e in enumerate(pred):
            recovered = j in mp and mp[j] not in base_matched[c]
            color = colors['new'] if recovered else colors[c] if j in mp else colors['fp']
            start = (offsets[e.section] + e.start) / 60
            anchor = (offsets[e.section] + e.anchor) / 60
            ax.broken_barh([(start, (e.end - e.start) / 60)], (y - 0.16, 0.32), facecolors=color, zorder=3)
            ax.scatter([anchor], [y], s=22, color=color, zorder=4)
        for j, e in enumerate(truth[c]):
            if j not in mt:
                ax.scatter([(offsets[e.section] + e.anchor) / 60], [y], marker='x', s=55, color=colors['miss'], linewidths=1.7, zorder=5)
        labels.append(title + ' · ' + ('Lineout' if c == 'line' else 'Scrum'))
        positions.append(y)
        ax.text(seconds / 60 + 0.6, y, f'{len(pairs)}/{len(truth[c])} detectados · {len(pred) - len(pairs)} FP', va='center', fontsize=12)
        rows.append(dict(rule=title, label=c, real=len(truth[c]), detected=len(pairs), false_alarms=len(pred) - len(pairs), omitted=len(truth[c]) - len(pairs)))
    ax.axhline(center - 1.9, color='#e6eaf0', linewidth=0.7, zorder=0)
for i, (sec, left) in enumerate(offsets.items()):
    right = list(offsets.values())[i + 1] if i < 3 else seconds
    if left:
        ax.axvline(left / 60, color='#aab5c1', linestyle='--', linewidth=0.9, zorder=1)
    ax.text((left + right) / 120, top + 1.0, sec, ha='center', fontsize=12, color='#64748b')

def fmt(s):
    n = round(s)
    return f'{n // 60:02d}:{n % 60:02d}'
ax.set_yticks(positions, labels, fontsize=12)
ax.tick_params(axis='y', length=0, pad=12)
ax.set(xlim=(0, seconds / 60), ylim=(min(positions) - 0.7, top + 1.6), xlabel='Tiempo acumulado del video (min:s)')
ticks = list(np.arange(0, 80, 5)) + [seconds / 60]
ax.set_xticks(ticks, [fmt(v * 60) for v in ticks], fontsize=11)
ax.xaxis.grid(True, color='#e0e6ed', linewidth=0.8)
ax.spines[['left', 'right', 'top']].set_visible(False)
ax.spines['bottom'].set_color('#dce3ea')
fig.suptitle('VF · 80:05 · lineout y scrum en el mismo eje temporal', x=0.035, y=0.978, ha='left', fontsize=24)
fig.text(0.035, 0.936, 'Mismo M1-H2-D2 · sólo varía el umbral de lineout · scrum conserva exactamente las decisiones de argmax · sin S05', fontsize=14, color='#64748b')
fig.legend(handles=[Patch(facecolor=colors['line'], label='Lineout detectado'), Patch(facecolor=colors['scrum'], label='Scrum detectado'), Patch(facecolor=colors['new'], label='Recuperado frente a argmax'), Patch(facecolor=colors['fp'], label='Falsa alarma (de la clase de esa fila)'), Line2D([], [], marker='x', color=colors['miss'], linestyle='None', label='Omitido')], ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.52, 0.035), fontsize=12)
fig.text(0.035, 0.012, 'Referencias reales en tonos suaves. FP = eventos sin pareja. Misma agrupación, anclas y tolerancia de 35 s en todas las filas.', fontsize=12, color='#64748b')
fig.subplots_adjust(left=0.215, right=0.805, top=0.9, bottom=0.14)
path = FIG / 'VF_80min_line_and_scrum.png'
fig.savefig(path, dpi=180, facecolor='white')
plt.close(fig)
pd.DataFrame(rows).to_csv(OUT / 'line_and_scrum_plot_counts.csv', index=False)
(OUT / 'line_and_scrum_plot_verification.json').write_text(json.dumps(dict(seconds=seconds, blocks=688, real_lineouts=30, real_scrums=10, all_scrum_predictions_identical=True, all_counts_match_previously_evaluated_results=True, no_inference_or_training=True, S05=False), indent=2), encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
