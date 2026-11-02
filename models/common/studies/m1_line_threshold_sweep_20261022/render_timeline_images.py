"""Static timelines from saved probabilities/evaluations; no inference."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = experiment_directory(__file__)
OUT = HERE / 'figures/timelines'
OUT.mkdir(exist_ok=True)
data = json.loads(_resolve_model_artifact(HERE / 'timeline_data.json').read_text(encoding='utf-8'))
REAL = '#64748b'
TP = '#1679a7'
FP = '#dc7a18'
MISS = '#be3445'
INK = '#202a35'
GRID = '#e0e5eb'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK, 'savefig.facecolor': 'white'})

def fmt(v):
    n = int(v)
    return f'{n // 60:02d}:{n % 60:02d}'

def fexact(v):
    return fmt(v) + (f'.{int(round(v % 1 * 10))}' if v % 1 else '')

def paired(events, tid):
    return any((e['status'] == 'TP' and e['truth'] == tid for e in events))
legend = [Patch(facecolor=REAL, label='Lineout real'), Patch(facecolor=TP, label='Detectado'), Patch(facecolor=FP, label='Falsa alarma'), Line2D([], [], marker='x', color=MISS, linestyle='None', markersize=8, label='Omitido')]

def focused(name, title, match, section, policy, lo, hi, explanation):
    g = data[match]
    d = g['sections'][section]['line']
    truth = d['truth']
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    visible = lambda e: e['end'] > lo and e['start'] < hi
    rows = [('Reales', truth), (f"Original\numbral {g['thresholds']['original']['line']:.2f}", d['pred']['original']), (f"Reducido\numbral {g['thresholds'][policy]['line']:.2f}", d['pred'][policy])]
    for i, (label, events) in enumerate(rows):
        y = 2 - i
        ax.axhline(y, color='#f0f3f6', linewidth=19, zorder=0)
        active = [e for e in events if visible(e)]
        for e in active:
            color = REAL if i == 0 else TP if e['status'] == 'TP' else FP
            ax.broken_barh([(max(lo, e['start']), min(hi, e['end']) - max(lo, e['start']))], (y - 0.14, 0.28), facecolors=color, zorder=3)
            center = (max(lo, e['start']) + min(hi, e['end'])) / 2
            if i == 0:
                text = f"Real {fmt(e['start'])}–{fmt(e['end'])}"
            elif e['status'] == 'TP':
                text = 'Adicional' if i == 2 and (not paired(d['pred']['original'], e['truth'])) else 'Detectado'
            else:
                text = 'Falsa alarma'
            ax.annotate(text, (center, y), xytext=(0, 19), textcoords='offset points', ha='center', fontsize=11)
            if name == '03_union' and i == 2 and (e['start'] < 763 < e['end']):
                ax.scatter([e['anchor']], [y], s=45, color=INK, zorder=5)
                ax.annotate(f"Instante elegido: {fexact(e['anchor'])}", (e['anchor'], y), xytext=(0, -28), textcoords='offset points', ha='center', fontsize=11)
        if i:
            for t in truth:
                if visible(t) and (not paired(events, t['id'])):
                    ax.scatter([t['anchor']], [y], marker='x', s=85, color=MISS, zorder=6, linewidth=2)
                    ax.annotate('Omitido', (t['anchor'], y), xytext=(0, 19) if i == 1 else (0, -26), textcoords='offset points', ha='center', color=MISS, fontsize=11)
    ax.set_yticks([2, 1, 0], [r[0] for r in rows])
    ax.tick_params(axis='y', length=0, pad=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.6, 2.62)
    ticks = np.arange(lo, hi + 0.1, 30)
    ax.set_xticks(ticks, [fmt(v) for v in ticks])
    ax.set_xlabel('Tiempo de la sección (min:s)', labelpad=8)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    fig.suptitle(title, x=0.04, y=0.985, ha='left', fontsize=17, fontweight='medium')
    fig.text(0.04, 0.91, f"M1-H2-D2 · {match.replace('_', '')} · sección {section} · {fmt(lo)}–{fmt(hi)}", fontsize=11, color=REAL)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(0.53, 0.09), ncol=4, frameon=False, fontsize=10)
    fig.text(0.04, 0.025, explanation, fontsize=10, color=INK)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.29, top=0.8)
    path = OUT / (name + '.png')
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path
focused('01_evento_adicional', 'Una detección adicional', 'V_4', 'S03', 'minus_0.05', 840, 960, 'Al bajar de 0.70 a 0.65 aparece la detección del lineout de 15:38–15:52.')
focused('02_falsa_alarma', 'El costo: aparece una falsa alarma', 'V_1', 'S02', 'minus_0.05', 360, 480, 'El candidato de 06:11–06:18 no tiene pareja real dentro de los 35 segundos permitidos.')
focused('03_union', 'Una unión puede hacer perder una detección', 'V_6', 'S04', 'minus_0.15', 720, 840, 'Centro real 12:43.0 → instante elegido 13:21.5: distancia 38.5 s, por encima de la tolerancia de 35 s.')
for match in ['V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'VF']:
    g = data[match]
    sections = list(g['sections'])
    fig, axes = plt.subplots(len(sections), 1, figsize=(13, len(sections) * 2.55), squeeze=False)
    for ax, (sec, v) in zip(axes[:, 0], g['sections'].items()):
        d = v['line']
        rows = [('Reales', d['truth'])]
        for pol, label in [('original', 'Original'), ('minus_0.05', 'Bajar 0.05'), ('minus_0.15', 'Bajar 0.15')]:
            rows.append((f"{label} · t={g['thresholds'][pol]['line']:.2f}", d['pred'][pol]))
        for i, (label, events) in enumerate(rows):
            y = 3 - i
            ax.axhline(y, color='#f1f4f7', linewidth=12, zorder=0)
            for e in events:
                color = REAL if i == 0 else TP if e['status'] == 'TP' else FP
                ax.broken_barh([(e['start'] / 60, (e['end'] - e['start']) / 60)], (y - 0.18, 0.36), facecolors=color, zorder=3)
            if i:
                for t in d['truth']:
                    if not paired(events, t['id']):
                        ax.scatter([t['anchor'] / 60], [y], marker='x', s=23, color=MISS, zorder=5, linewidths=1.2)
        ax.set_yticks([3, 2, 1, 0], [r[0] for r in rows], fontsize=10)
        ax.tick_params(axis='y', length=0, pad=10)
        ax.set_xlim(0, v['end'] / 60)
        ax.set_ylim(-0.5, 3.5)
        ax.set_xticks(np.arange(0, v['end'] / 60 + 0.01, 2))
        ax.set_xlabel('Minutos desde el comienzo de la sección', fontsize=10, labelpad=3)
        ax.set_title(f"{sec} · duración {fmt(v['end'])}", loc='left', fontsize=12, pad=9)
        ax.xaxis.grid(True, color=GRID, zorder=0)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
    fig.suptitle(f"{match.replace('_', '')} · todos los lineouts y candidatos", x=0.03, y=0.99, ha='left', fontsize=18, fontweight='medium')
    fig.legend(handles=legend, ncol=4, loc='upper center', bbox_to_anchor=(0.56, 0.965), frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.16, right=0.975, top=0.92, bottom=0.065, hspace=0.65)
    fig.text(0.03, 0.012, 'Probabilidades existentes · tolerancia de 35 s · cada sección se evalúa por separado · sin nuevos entrenamientos', fontsize=10, color=REAL)
    fig.savefig(OUT / f'{match}_completo.png', dpi=160)
    plt.close(fig)
print('Imágenes guardadas en', OUT)
