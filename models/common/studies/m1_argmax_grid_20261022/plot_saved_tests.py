"""Summary image for the descriptive test-only comparison."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
here = experiment_directory(__file__) / 'cached_tests'
a = pd.read_csv(_resolve_model_artifact(here / 'aggregate_361_combinations.csv'), dtype={'line_rule': str, 'scrum_rule': str})

def label(v):
    return 'argmax' if v == 'argmax' else f'{float(v):.2f}'
a['line'] = a.line_rule.map(label)
a['scrum'] = a.scrum_rule.map(label)
order = [f'{v:.2f}' for v in np.arange(0.1, 0.951, 0.05)] + ['argmax']
matrix = a.pivot(index='scrum', columns='line', values='mean_macro_temporal_f1').loc[order, order]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig = plt.figure(figsize=(14, 14), facecolor='#f5f7fb')
gs = fig.add_gridspec(2, 1, height_ratios=[4.6, 1.45], hspace=0.27)
ax = fig.add_subplot(gs[0])
ax.set_facecolor('white')
im = ax.imshow(matrix.to_numpy(), origin='upper', cmap='YlGnBu', vmin=0, vmax=0.65, aspect='equal')
ax.set_xticks(range(19), order, rotation=45, ha='right')
ax.set_yticks(range(19), order)
ax.set_xlabel('Regla para LINE', labelpad=12, fontweight='bold')
ax.set_ylabel('Regla para SCRUM', labelpad=12, fontweight='bold')
ax.axvline(17.5, color='white', lw=3)
ax.axhline(17.5, color='white', lw=3)
for line, scrum in [('0.50', '0.35'), ('argmax', '0.35'), ('0.50', 'argmax'), ('argmax', 'argmax')]:
    x, y = (order.index(line), order.index(scrum))
    ax.add_patch(Rectangle((x - 0.47, y - 0.47), 0.94, 0.94, fill=False, edgecolor='#ffbf47', lw=2.6))
    ax.text(x, y, f'{matrix.loc[scrum, line]:.3f}', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
cb.set_label('F1 temporal macro · promedio de los 6 tests')
fig.suptitle('M1 · Umbrales + argmax en los seis tests', x=0.08, ha='left', y=0.98, fontsize=23, fontweight='bold', color='#152641')
fig.text(0.08, 0.945, '361 combinaciones por fold · Cada modelo en su propio test · Probabilidades guardadas', fontsize=12, color='#455572')
tabax = fig.add_subplot(gs[1])
tabax.axis('off')
families = a.groupby('family', sort=False).head(1).copy()
families['priority'] = families.family.map({'both_argmax': 0, 'scrum_argmax': 1, 'line_argmax': 2, 'numeric_numeric': 3})
families = families.sort_values('priority')
cells = []
for r in families.itertuples():
    cells.append([r.line, r.scrum, f'{r.line_tp} / {r.line_fp} / {r.line_fn}', f'{r.scrum_tp} / {r.scrum_fp} / {r.scrum_fn}', f'{r.mean_macro_temporal_f1:.4f}'])
tabax.text(0, 1.05, 'Mejor combinación observada por familia', fontweight='bold', fontsize=15, transform=tabax.transAxes)
table = tabax.table(cellText=cells, colLabels=['Line', 'Scrum', 'Line: TP / FP / FN', 'Scrum: TP / FP / FN', 'F1 temporal'], cellLoc='center', bbox=[0, 0.18, 1, 0.77], colWidths=[0.14, 0.14, 0.25, 0.25, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(12)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#dce3ee')
    cell.set_facecolor('#193454' if row == 0 else '#e9f0f8' if row % 2 else 'white')
    if row == 0:
        cell.set_text_props(color='white', weight='bold')
tabax.text(0, 0.04, 'TP = aciertos · FP = falsas detecciones · FN = jugadas omitidas. Totales: 188 line y 107 scrum.', fontsize=10.5, transform=tabax.transAxes, color='#455572')
fig.text(0.08, 0.028, 'Comparación exploratoria sobre test. Elegir aquí la mejor combinación no constituye una evaluación independiente.\nEn modos mixtos se conserva la clase con argmax y se aplica el umbral a la otra en los bloques restantes.', fontsize=10, color='#455572')
fig.subplots_adjust(left=0.08, right=0.93, top=0.91, bottom=0.085)
fig.savefig(here / 'comparacion_umbrales_argmax.png', dpi=145, facecolor=fig.get_facecolor())
plt.close(fig)
print(here / 'comparacion_umbrales_argmax.png')
