"""Prediction timelines generated from recorded data."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import models.M1.studies.m1_ensemble_vs_all_development_20261012.runtime as rt
from models.M1.studies.m1_ensemble_vs_all_development_20261012.runtime import HERE, ROOT, sha
from pathlib import Path
import json, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager
OUTPUT = HERE / 'figures'
OUTPUT.mkdir(exist_ok=True)
COLORS = {'line': '#3D8DFF', 'scrum': '#47E2C2', 'background': '#EAEAEA', 'ink': '#121212', 'navy': '#062E55', 'gray': '#555555'}
for file in ['arial.ttf', 'arialbd.ttf']:
    font_manager.fontManager.addfont('C:/Windows/Fonts/' + file)
plt.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none', 'axes.unicode_minus': False})
paths = [HERE / 'results/M1-ensemble_result.json', HERE / 'results/M1-single_result.json']
data = [json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8')) for p in paths]
quality_paths = [HERE / 'results/M1-ensemble_quality.json', HERE / 'results/M1-single_quality.json']
qualities = [json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8')) for p in quality_paths]
assert data[0]['metadata']['duration'] == data[1]['metadata']['duration'] == 1201.2
truths = []
for q in qualities:
    truths.append(sorted([{'play_type': e['class'], 'start_time': e['start_time'], 'end_time': e['end_time']} for label, g in q['events'].items() for e in g['truth']], key=lambda e: (e['start_time'], e['play_type'])))
assert truths[0] == truths[1] and len(truths[0]) == 13
rows = [{'label': 'Real', 'subtitle': 'Etiquetado manual Â· bloques de 7 s', 'result': {'detections': truths[0]}, 'real': True}]
for label, subtitle, q, d in zip(['Ensemble (6)', 'Modelo Ãºnico'], ['M1-H2-D2 Â· seis folds', 'M1-H2-D2 Â· 8796 ejemplos'], qualities, data):
    events = sorted([{'play_type': e['class'], 'start_time': e['start_time'], 'end_time': e['end_time']} for tag, g in q['events'].items() for e in g['prediction']], key=lambda e: (e['start_time'], e['play_type']))
    rows.append({'label': label, 'subtitle': subtitle, 'result': {'detections': events, 'metadata': d['metadata']}, 'f1': q['temporal']['weighted_f1_macro']})

def clock(seconds):
    value = round(seconds)
    return f'{value // 3600} h {value % 3600 // 60:02} min {value % 60:02} s' if value >= 3600 else f'{value // 60} min {value % 60:02} s'

def draw(name, extent, title, subtitle):
    png = OUTPUT / (name + '.png')
    svg = OUTPUT / (name + '.svg')
    assert not _resolve_model_artifact(png).exists() and (not _resolve_model_artifact(svg).exists())
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.text(0.04, 0.92, title, fontsize=26, fontweight='bold', color=COLORS['ink'])
    fig.text(0.04, 0.845, subtitle, fontsize=13, color=COLORS['gray'])
    fig.text(0.94, 0.783, 'Tiempo de inferencia', ha='right', fontsize=12, color=COLORS['gray'])
    evidence = []
    for index, (row, y) in enumerate(zip(rows, [0.705, 0.515, 0.325])):
        result = row['result']
        real = row.get('real', False)
        t = None if real else result['metadata']['timing']['inference_seconds']
        fig.text(0.045, y + 0.025, row['label'], fontsize=17, fontweight='bold', color=COLORS['navy'])
        fig.text(0.045, y - 0.02, row['subtitle'], fontsize=10, color=COLORS['gray'])
        ax = fig.add_axes([0.205, y - 0.032, 0.6, 0.072])
        ax.set_xlim(0, extent)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(Rectangle((0, 0.03), extent, 0.94, facecolor=COLORS['background'], edgecolor='#D8D8D8', linewidth=0.7))
        plotted = []
        for event in result['detections']:
            label = event['play_type']
            start = max(0, float(event['start_time']))
            end = min(extent, float(event['end_time']))
            assert label in ['line', 'scrum']
            if end <= start:
                continue
            ax.add_patch(Rectangle((start, 0.03), end - start, 0.94, facecolor=COLORS[label], edgecolor='none'))
            plotted.append({'class': label, 'start_seconds': start, 'end_seconds': end})
        fig.text(0.94, y + 0.02, 'Etiquetado manual' if real else clock(t), ha='right', fontsize=16 if real else 19, fontweight='bold', color=COLORS['navy'] if real else COLORS['line'] if index == 1 else '#079B83')
        visible = [e for e in result['detections'] if e['start_time'] < extent and e['end_time'] > 0]
        count_line = sum((e['play_type'] == 'line' for e in visible))
        count_scrum = sum((e['play_type'] == 'scrum' for e in visible))
        fig.text(0.94, y - 0.029, f'{count_line} lineout Â· {count_scrum} scrum', ha='right', fontsize=12, color=COLORS['gray'])
        if not real:
            fig.text(0.94, y - 0.058, f"F1 temporal medio: {row['f1']:.3f}", ha='right', fontsize=11, color=COLORS['navy'])
        evidence.append({'model': row['label'], 'events_in_display': plotted, 'inference_seconds': t})
    axis = fig.add_axes([0.205, 0.233, 0.6, 0.015])
    axis.set_xlim(0, extent)
    axis.set_ylim(0, 1)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.spines['bottom'].set_visible(True)
    axis.spines['bottom'].set_color('#888888')
    axis.set_yticks([])
    step = 120 if extent == 1200 else 60
    axis.set_xticks(list(range(0, int(extent) + 1, step)), [str(v // 60) for v in range(0, int(extent) + 1, step)])
    axis.tick_params(axis='x', labelsize=10, length=7, color='#888888', labelcolor=COLORS['gray'], pad=8)
    fig.text(0.505, 0.18, 'Minuto del video', ha='center', fontsize=12, color=COLORS['gray'])
    for x, label, color in [(0.285, 'lineout', COLORS['line']), (0.425, 'scrum', COLORS['scrum']), (0.565, 'otro / sin evento', COLORS['background'])]:
        fig.add_artist(Rectangle((x, 0.123), 0.022, 0.024, transform=fig.transFigure, facecolor=color, edgecolor='#D8D8D8' if color == COLORS['background'] else 'none'))
        fig.text(x + 0.033, 0.131, label, va='center', fontsize=12, color=COLORS['gray'])
    fig.text(0.04, 0.069, 'Real y predicho: bloques de 7 s agrupados por clase. Tiempos y F1: VF_S01 completo (1201,2 s).', fontsize=12, fontweight='bold', color=COLORS['navy'])
    fig.text(0.04, 0.032, 'Ambos umbrales lineout / scrum: 0,50 / 0,45. F1 descriptivo: independencia de VF no verificada por completo.', fontsize=10, color=COLORS['gray'])
    if extent != 1200:
        fig.text(0.94, 0.032, 'Los tiempos corresponden a los 20 min completos.', ha='right', fontsize=10, color=COLORS['gray'])
    fig.savefig(png, dpi=160, facecolor='white')
    fig.savefig(svg, facecolor='white')
    plt.close(fig)
    return {'png': str(png), 'svg': str(svg), 'axis_seconds': [0, extent], 'rows': evidence}
figures = [draw('comparacion_ensemble_modelo_unico_20min', 1200, 'Ensemble y modelo Ãºnico entrenado con todos los datos', 'VF_S01 Â· comparaciÃ³n completa de 20 min Â· ambos en grafo Â· pesos y umbrales fijados con desarrollo'), draw('comparacion_ensemble_modelo_unico_detalle_6min', 360, 'Ensemble y modelo Ãºnico: primeros seis minutos', 'AmpliaciÃ³n de la misma comparaciÃ³n Â· mÃ©tricas y cronÃ³metros del video completo')]
audit = {'sources': {str(p): sha(p) for p in paths + quality_paths + [HERE / 'inputs/selection.json', HERE / 'inputs/independence_audit.json']}, 'video_duration_seconds': 1201.2, 'axis_full_seconds': [0, 1200], 'colors': COLORS, 'figures': figures, 'truth': 'final human labels, never model output', 'prediction_grid': 'same exact manual 7-second blocks and final 4.2s tail', 'independent_test_claim_allowed': False}
with _resolve_model_artifact(OUTPUT / 'image_evidence.json').open('x', encoding='utf-8') as file:
    json.dump(audit, file, indent=2)
print(json.dumps({'images': [f['png'] for f in figures]}))
