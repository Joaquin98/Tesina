"""Prediction timelines generated from recorded data."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import runtime as rt
from runtime import HERE, ROOT, sha
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
paths = [ROOT / 'data/top5_vf_sequential_20261012/results/M1-H2-D2_result.json', HERE / 'results/result.json']
data = [json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8')) for p in paths]
assert data[0]['metadata']['duration'] == data[1]['metadata']['duration'] == 1201.2
rows = [{'label': 'RGB + YOLO', 'subtitle': 'M1-H2-D2 completo', 'result': data[0]}, {'label': 'Sólo YOLO', 'subtitle': 'Geometría de M1-H2-D2', 'result': data[1]}]
session_id = '20260830_112524_0b5e58'
truth_source = ROOT / f'data/active_learning_dataset/sessions/{session_id}/train_manifest.csv'
truth_review = ROOT / f'data/active_learning_dataset/sessions/{session_id}/train_review_log.csv'
truth_session = ROOT / f'data/active_learning_review/sessions/review_session_{session_id}.json'
blocks = list(csv.DictReader(_resolve_model_artifact(truth_source).open(encoding='utf-8-sig')))
review = list(csv.DictReader(_resolve_model_artifact(truth_review).open(encoding='utf-8-sig')))
session = json.loads(_resolve_model_artifact(truth_session).read_text(encoding='utf-8'))
assert session['status'] == 'exported' and session['review_source'] == 'manual_fixed_blocks'
assert len(blocks) == len(review) == len(session['candidates']) == 172
assert session['video_duration'] == data[0]['metadata']['duration'] == 1201.2
assert Path(session['video_path']).resolve() == Path(data[0]['video_path']).resolve() == Path(data[1]['video_path']).resolve()
assert [(b['dataset_id'], b['correct_label'], b['start_time'], b['end_time']) for b in blocks] == [(b['dataset_id'], b['correct_label'], b['start_time'], b['end_time']) for b in review]
candidates = {int(c['candidate_index']): c for c in session['candidates']}
for b in blocks:
    assert Path(b['source_video']).resolve() == Path(session['video_path']).resolve()
    c = candidates[int(b['candidate_index'])]
    assert c['final_label'] == b['correct_label'] and c['manual_labeling'] and c['reviewed_at']
    assert c['start_time'] == float(b['start_time']) and c['end_time'] == float(b['end_time'])
blocks.sort(key=lambda b: float(b['start_time']))
assert float(blocks[0]['start_time']) == 0 and float(blocks[-1]['end_time']) == 1201.2
assert all((abs(float(a['end_time']) - float(b['start_time'])) < 1e-08 for a, b in zip(blocks, blocks[1:])))
spans = []
for b in blocks:
    label = b['correct_label']
    start = float(b['start_time'])
    end = float(b['end_time'])
    assert label in ['line', 'other', 'scrum'] and end > start
    if spans and spans[-1]['play_type'] == label:
        spans[-1]['end_time'] = end
    else:
        spans.append({'play_type': label, 'start_time': start, 'end_time': end})
truth_events = [s for s in spans if s['play_type'] != 'other']
assert len(truth_events) == 13
rows.insert(0, {'label': 'Real', 'subtitle': 'Etiquetado manual · bloques de 7 s', 'result': {'detections': truth_events}, 'real': True})

def clock(seconds):
    value = round(seconds)
    return f'{value // 3600} h {value % 3600 // 60:02} min {value % 60:02} s' if value >= 3600 else f'{value // 60} min {value % 60:02} s'

def draw(name, begin, extent, title, subtitle):
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
        ax.set_xlim(begin, extent)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(Rectangle((begin, 0.03), extent - begin, 0.94, facecolor=COLORS['background'], edgecolor='#D8D8D8', linewidth=0.7))
        plotted = []
        for event in result['detections']:
            label = event['play_type']
            start = max(begin, float(event['start_time']))
            end = min(extent, float(event['end_time']))
            assert label in ['line', 'scrum']
            if end <= start:
                continue
            ax.add_patch(Rectangle((start, 0.03), end - start, 0.94, facecolor=COLORS[label], edgecolor='none'))
            plotted.append({'class': label, 'start_seconds': start, 'end_seconds': end})
        fig.text(0.94, y + 0.02, 'Etiquetado manual' if real else clock(t), ha='right', fontsize=16 if real else 19, fontweight='bold', color=COLORS['navy'] if real else COLORS['line'] if index == 1 else '#079B83')
        visible = [e for e in result['detections'] if e['start_time'] < extent and e['end_time'] > begin]
        count_line = sum((e['play_type'] == 'line' for e in visible))
        count_scrum = sum((e['play_type'] == 'scrum' for e in visible))
        fig.text(0.94, y - 0.029, f'{count_line} lineout · {count_scrum} scrum', ha='right', fontsize=12, color=COLORS['gray'])
        evidence.append({'model': row['label'], 'events_in_display': plotted, 'inference_seconds': t})
    axis = fig.add_axes([0.205, 0.233, 0.6, 0.015])
    axis.set_xlim(begin, extent)
    axis.set_ylim(0, 1)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.spines['bottom'].set_visible(True)
    axis.spines['bottom'].set_color('#888888')
    axis.set_yticks([])
    step = 120 if extent == 1200 else 60
    axis.set_xticks(list(range(int(begin), int(extent) + 1, step)), [str(v // 60) for v in range(int(begin), int(extent) + 1, step)])
    axis.tick_params(axis='x', labelsize=10, length=7, color='#888888', labelcolor=COLORS['gray'], pad=8)
    fig.text(0.505, 0.18, 'Minuto del video', ha='center', fontsize=12, color=COLORS['gray'])
    for x, label, color in [(0.285, 'lineout', COLORS['line']), (0.425, 'scrum', COLORS['scrum']), (0.565, 'otro / sin evento', COLORS['background'])]:
        fig.add_artist(Rectangle((x, 0.123), 0.022, 0.024, transform=fig.transFigure, facecolor=color, edgecolor='#D8D8D8' if color == COLORS['background'] else 'none'))
        fig.text(x + 0.033, 0.131, label, va='center', fontsize=12, color=COLORS['gray'])
    fig.text(0.04, 0.069, 'Real: bloques manuales contiguos de la misma clase. Modelos: eventos predichos agrupados.', fontsize=12, fontweight='bold', color=COLORS['navy'])
    fig.text(0.04, 0.032, 'Umbrales de lineout / scrum: RGB + YOLO 0,80 / 0,75; sólo YOLO 0,30 / 0,45.', fontsize=10, color=COLORS['gray'])
    if extent != 1200:
        fig.text(0.94, 0.032, 'Los tiempos corresponden a los 20 min completos.', ha='right', fontsize=10, color=COLORS['gray'])
    fig.savefig(png, dpi=160, facecolor='white')
    fig.savefig(svg, facecolor='white')
    plt.close(fig)
    return {'png': str(png), 'svg': str(svg), 'axis_seconds': [begin, extent], 'rows': evidence}
figures = [draw('comparacion_real_inferencias_fragmento_6a12min', 360, 720, 'Anotaciones reales y predicciones: otro fragmento', 'VF_S01 · minutos 06:00 a 12:00 · misma escala para las tres bandas')]
audit = {'sources': {str(p): sha(p) for p in paths + [truth_source, truth_review, truth_session]}, 'video_duration_seconds': 1201.2, 'display_full_axis_seconds': 1200, 'colors': COLORS, 'based_on': 'manual correct_label/final_label for Real; saved merged detections for predictions', 'ground_truth_validation': {'session_id': session_id, 'blocks': 172, 'coverage_seconds': [0, 1201.2], 'same_source_video': True, 'manifest_matches_review_log_and_session': True, 'real_events': 13, 'line': 8, 'scrum': 5, 'event_boundaries_from_7s_blocks': True}, 'style_reference': 'docs/presentacion_guion_20260908/.build/v9_times_round_header/rendered/slide-33.png', 'original_slide_decks_modified': False, 'figures': figures}
with _resolve_model_artifact(OUTPUT / 'image_fragment_6a12_evidence.json').open('x', encoding='utf-8') as file:
    json.dump(audit, file, indent=2)
print(json.dumps({'images': [f['png'] for f in figures], 'source_event_counts': [13] + [len(d['detections']) for d in data]}))
