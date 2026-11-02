"""A single, unambiguous comparison: identical M1-H2-D2 probabilities, lower line thresholds."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT = experiment_directory(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import apply_class_thresholds, events_from_blocks, match_events
HERE = experiment_directory(__file__)
OUT = HERE / 'figures/timelines'
source = ROOT / 'data/final_model_comparison_20260827/outputs/runs/PV-M1-H2-D2-PV04-S42/attempt_001/predictions.csv'
meta = json.loads(_resolve_model_artifact(source.parent / 'run.json').read_text())
before = hashlib.sha256(_resolve_model_artifact(source).read_bytes()).hexdigest()
raw = pd.read_csv(_resolve_model_artifact(source)).rename(columns={'test_group': 'group_id'})
f = raw[raw.section == 'S03'].copy()
assert len(f) == 172 and np.isclose(f.duration.sum(), 1200)
assert f.checkpoint_hash.eq(meta['checkpoint_hash']).all()
prob = f[['p_line', 'p_other', 'p_scrum']].to_numpy()
truth = events_from_blocks(f, 'line', prediction_column='y_true', probability_column='p_line')
assert len(truth) == 10
baseline = meta['thresholds']['line']
scrum = meta['thresholds']['scrum']
thresholds = [baseline, 0.65, 0.55, 0.4, 0.2, 0.05]
reference = pd.read_csv(_resolve_model_artifact(HERE / 'audit_20261022/V4_S03_independent_thresholds.csv'))
rows = []
derived = []
states = []
for threshold in thresholds:
    d = f.copy()
    labels = apply_class_thresholds(prob, line=threshold, scrum=scrum)
    d['y_final'] = np.array(['line', 'other', 'scrum'])[labels]
    if threshold == baseline:
        assert np.array_equal(d.y_final, f.y_final)
    pred = events_from_blocks(d, 'line', prediction_column='y_final', probability_column='p_line')
    pairs = match_events(truth, pred)
    mt = {i for i, j, dist, credit in pairs}
    mp = {j for i, j, dist, credit in pairs}
    row = dict(threshold=threshold, real=len(truth), detected=len(pairs), false_alarms=len(pred) - len(pairs), omitted=len(truth) - len(pairs))
    check = reference[(reference.rule == 'thresholds') & np.isclose(reference.threshold, threshold)].iloc[0]
    assert row['detected'] == check.tp and row['false_alarms'] == check.fp and (row['omitted'] == check.fn)
    rows.append(row)
    states.append((threshold, pred, mt, mp))
    d.assign(line_threshold=threshold, scrum_threshold=scrum).to_csv(OUT / f'V4_S03_same_model_t{threshold:.2f}.csv', index=False)
    for i, e in enumerate(truth):
        derived.append(dict(threshold=threshold, event=i + 1, start_s=e.start, end_s=e.end, anchor_s=e.anchor, detected=i in mt))
pd.DataFrame(rows).to_csv(OUT / 'V4_S03_same_model_counts.csv', index=False)
pd.DataFrame(derived).to_csv(OUT / 'V4_S03_same_model_truth_outcomes.csv', index=False)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'text.color': '#202a35', 'axes.labelcolor': '#202a35'})
colors = dict(real='#64748b', tp='#1679a7', fp='#dc7a18', miss='#be3445')
fig, ax = plt.subplots(figsize=(16, 10))
n = len(states)
for e in truth:
    ax.axvspan(e.start / 60, e.end / 60, color=colors['real'], alpha=0.045, zorder=0)
ax.axhline(n, color='#eef2f6', linewidth=24, zorder=0)
for i, e in enumerate(truth, 1):
    ax.broken_barh([(e.start / 60, (e.end - e.start) / 60)], (n - 0.16, 0.32), facecolors=colors['real'], zorder=3)
    ax.annotate(f'#{i}', (e.anchor / 60, n), xytext=(0, 19), textcoords='offset points', ha='center', fontsize=11, color=colors['real'])
for k, ((threshold, pred, mt, mp), r) in enumerate(zip(states, rows)):
    y = n - 1 - k
    ax.axhline(y, color='#f1f4f7', linewidth=24, zorder=0)
    for j, e in enumerate(pred):
        ax.broken_barh([(e.start / 60, (e.end - e.start) / 60)], (y - 0.16, 0.32), facecolors=colors['tp' if j in mp else 'fp'], zorder=3)
    for i, e in enumerate(truth):
        if i not in mt:
            ax.scatter([e.anchor / 60], [y], marker='x', s=64, linewidths=1.8, color=colors['miss'], zorder=5)
    ax.text(20.35, y, f"{r['detected']} de 10       {r['false_alarms']}", va='center', fontsize=13)
ax.text(20.35, n, 'Detectados     Falsas', fontsize=12, va='center')
ax.set_yticks(list(range(n, -1, -1)), ['Lineouts reales'] + [f'Umbral {t:.2f}' + (' · original' if k == 0 else '') for k, (t, *_) in enumerate(states)], fontsize=14)
ax.tick_params(axis='y', length=0, pad=12)
ax.set_xlim(0, 20)
ax.set_ylim(-0.5, n + 0.6)
ax.set_xticks(np.arange(0, 21, 2), [f'{i:02d}:00' for i in range(0, 21, 2)], fontsize=12)
ax.set_xlabel('Tiempo de la sección (min:s)', fontsize=13, labelpad=13)
ax.xaxis.grid(True, color='#dce3ea', linewidth=0.8)
for side in ['left', 'right', 'top']:
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color('#dce3ea')
fig.suptitle('Mismo modelo. Mismos 20 minutos. Sólo baja el umbral.', x=0.035, y=0.98, ha='left', fontsize=22)
fig.text(0.035, 0.93, 'M1-H2-D2 · pliegue PV04 · V4, sección S03 · umbral de scrum fijo en 0.40', fontsize=13, color=colors['real'])
fig.legend(handles=[Patch(facecolor=colors['real'], label='Lineout real'), Patch(facecolor=colors['tp'], label='Detectado'), Patch(facecolor=colors['fp'], label='Falsa alarma'), Line2D([], [], marker='x', color=colors['miss'], linestyle='None', markersize=9, label='Omitido')], ncol=4, loc='lower center', bbox_to_anchor=(0.52, 0.065), frameon=False, fontsize=12)
fig.text(0.035, 0.025, 'Se recalculan los eventos con las probabilidades originales. Emparejamiento uno a uno, tolerancia de 35 s. Barrido exploratorio.', fontsize=11, color=colors['real'])
fig.subplots_adjust(left=0.18, right=0.83, top=0.88, bottom=0.17)
path = OUT / 'V4_S03_same_model_thresholds_20min.png'
fig.savefig(path, dpi=170, facecolor='white')
plt.close(fig)
assert before == hashlib.sha256(_resolve_model_artifact(source).read_bytes()).hexdigest()
(OUT / 'V4_S03_same_model_verification.json').write_text(json.dumps(dict(source=str(source), source_sha256=before, checkpoint_hash=meta['checkpoint_hash'], same_probabilities_for_every_row=True, original_decisions_reproduced=True, independent_count_audit_matches=True, scrum_threshold=scrum, section='S03', seconds=1200, counts=rows), indent=2), encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
