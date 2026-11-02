"""Evaluate frozen alerts against completed human seven-second labels."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from collections import Counter
import hashlib
import html
import json
import os
import shutil
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
OUT = HERE / 'human_validation'
SESSION_ID = '20261026_212724_97d95f'
RESULT_ID = '20261025_argentina_allblacks_20250816_argmax'
SESSION = ROOT / f'data/active_learning_review/sessions/review_session_{SESSION_ID}.json'
RESULT = ROOT / f'data/results/result_{RESULT_ID}.json'
OUT.mkdir(exist_ok=True)

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(p):
    return hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()

def overlap(a, b):
    return max(0.0, min(a['end_s'], b['end_s']) - max(a['start_s'], b['start_s']))

def match(truth, prediction, minimum_overlap):
    if not truth or not prediction:
        return []
    seconds = np.array([[overlap(t, p) for p in prediction] for t in truth])
    valid = seconds >= minimum_overlap
    score = valid.astype(float) + np.where(valid, seconds, 0.0) / ((seconds.max() + 1.0) * (min(seconds.shape) + 1.0))
    ii, jj = linear_sum_assignment(-score)
    return [(int(i), int(j), float(seconds[i, j])) for i, j in zip(ii, jj) if valid[i, j]]
hashes = {str(p): digest(p) for p in (SESSION, RESULT)}
s, r = (read(SESSION), read(RESULT))
assert os.path.samefile(s['video_path'], r['video_path'])
blocks = s['candidates']
assert s['dataset_split'] == 'test' and len(blocks) == 881
assert all((b['final_label'] in {'line', 'scrum', 'other'} for b in blocks))
assert all((abs(b['start_time'] - i * 7.0) < 1e-06 for i, b in enumerate(blocks)))
assert all((abs(a['end_time'] - b['start_time']) < 1e-06 for a, b in zip(blocks, blocks[1:])))
assert abs(blocks[-1]['end_time'] - s['video_duration']) < 1e-06
truth = []
for b in blocks:
    label = b['final_label']
    if label == 'other':
        continue
    if truth and truth[-1]['play_type'] == label and (truth[-1]['end_s'] == b['start_time']):
        truth[-1]['end_s'] = b['end_time']
        truth[-1]['blocks'].append(b['candidate_index'])
    else:
        truth.append(dict(id=f'human_{label}_{len(truth) + 1}', play_type=label, start_s=b['start_time'], end_s=b['end_time'], blocks=[b['candidate_index']]))
predictions = [dict(id=i + 1, play_type=p['play_type'], start_s=p['start_time'], end_s=p['end_time']) for i, p in enumerate(r['detections'])]
assert len(predictions) == 105
metrics, diagnostics, matched, missed, false_alerts = ({}, {}, [], [], [])
for label in ('line', 'scrum'):
    ts = [t for t in truth if t['play_type'] == label]
    ps = [p for p in predictions if p['play_type'] == label]
    pairs = match(ts, ps, 1.0)
    tp, nt, npred = (len(pairs), len(ts), len(ps))
    metrics[label] = dict(tp=tp, fp=npred - tp, fn=nt - tp, reference_segments=nt, alerts=npred, precision=tp / npred, recall=tp / nt, f1=2 * tp / (nt + npred))
    selected_t, selected_p = ({i for i, _, _ in pairs}, {j for _, j, _ in pairs})
    for i, j, seconds in pairs:
        matched.append(dict(reference_id=ts[i]['id'], prediction_id=ps[j]['id'], overlap_s=seconds))
    missed.extend((t for i, t in enumerate(ts) if i not in selected_t))
    false_alerts.extend((p for j, p in enumerate(ps) if j not in selected_p))
    diagnostics[label] = {}
    for threshold in (1e-06, 1.0, 3.5):
        n = len(match(ts, ps, threshold))
        diagnostics[label][str(threshold)] = dict(tp=n, precision=n / npred, recall=n / nt)
    inclusive = sum((any((overlap(t, p) >= 1.0 for t in ts)) for p in ps))
    diagnostics[label]['alerts_overlapping_reference_without_deduplication'] = dict(correct=inclusive, alerts=npred, precision=inclusive / npred)
tp = sum((m['tp'] for m in metrics.values()))
nt = len(truth)
npred = len(predictions)
metrics['total'] = dict(tp=tp, fp=npred - tp, fn=nt - tp, reference_segments=nt, alerts=npred, precision=tp / npred, recall=tp / nt, f1=2 * tp / (nt + npred))
assert [(metrics[k]['reference_segments'], metrics[k]['tp']) for k in ('line', 'scrum')] == [(26, 22), (15, 14)]
assert metrics['total']['tp'] + metrics['total']['fn'] == len(truth)
assert metrics['total']['tp'] + metrics['total']['fp'] == len(predictions)
assert len({p['prediction_id'] for p in matched}) == len(matched)
assert len({p['reference_id'] for p in matched}) == len(matched)
reconciled = read(HERE / 'reconciliation/2025-08-16_reference_v3.json')['events']
visual_without_human = [e for e in reconciled if not any((e['play_type'] == t['play_type'] and overlap(e, t) > 0 for t in truth))]
report = dict(status='complete_against_saved_human_block_labels', date='2026-10-26', session_id=SESSION_ID, result_id=RESULT_ID, video_duration_s=s['video_duration'], reviewed_blocks=len(blocks), block_counts=dict(Counter((b['final_label'] for b in blocks))), method='Etiquetas humanas de toda la transmisión. Se unen exclusivamente bloques contiguos de la misma clase. Emparejamiento uno a uno por clase, maximizando número de pares con al menos1s de solapamiento; desempate por solapamiento. Una detección repetida sobre el mismo segmento no obtiene otro acierto. Sin tolerancia de35s, sin filtrar alertas por una revisión visual automática, sin inferencia nueva ni ajuste de umbrales.', limitations=['Los segmentos humanos no distinguen vivo/replay ni otorgamientos/rearmados. No equivalen necesariamente a reinicios únicos de fase ni a totales oficiales.', 'Resolución de referencia de7s. Los límites exactos de la jugada no están anotados; se muestra sensibilidad al exigir3,5s de solapamiento.', 'Los porcentajes anteriores por revisión diagnóstica automática usaban otra referencia y permitían duplicados en precisión; no son una comparación controlada con estos porcentajes.', 'Este partido fue seleccionado después de inspeccionar resultados; la revisión humana no lo convierte en un conjunto externo elegido a ciegas.'], metrics=metrics, overlap_diagnostics=diagnostics, reference_segments=truth, matches=matched, missed_segments=sorted(missed, key=lambda t: t['start_s']), unmatched_alerts=false_alerts, previous_visual_intervals_without_human_same_class=visual_without_human, provenance=[dict(path=p, sha256=h) for p, h in hashes.items()], model_result_url=f'/viewer?result_id={RESULT_ID}', labeling_url=f'/clip-generator?session_id={SESSION_ID}')
dump(OUT / 'metrics.json', report)
dump(OUT / 'human_labels_snapshot.json', dict(session_id=SESSION_ID, video_path=s['video_path'], candidates=blocks))

def percent(v):
    return f'{100 * v:.1f}%'.replace('.', ',')

def clock(v):
    return f'{int(v) // 60:02d}:{int(v) % 60:02d}'
table = ''.join((f"<tr><td>{html.escape(k)}</td><td>{m['reference_segments']}</td><td>{m['alerts']}</td><td>{m['tp']}</td><td>{m['fp']}</td><td>{m['fn']}</td><td>{percent(m['precision'])}</td><td>{percent(m['recall'])}</td><td>{percent(m['f1'])}</td></tr>" for k, m in metrics.items()))
missing_table = ''.join((f"<tr><td>{t['play_type']}</td><td>{clock(t['start_s'])}–{clock(t['end_s'])}</td><td>{', '.join((str(i + 1) for i in t['blocks']))}</td></tr>" for t in report['missed_segments']))
limits = ''.join(('<li>' + html.escape(v) + '</li>' for v in report['limitations']))
document = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Argentina–NZ16/08 · validación humana</title><style>body{font:16px system-ui;line-height:1.6;max-width:1200px;margin:30px auto;padding:0 18px;color:#243447}table{width:100%;border-collapse:collapse;margin:20px 0}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow:auto}aside{background:#edf3f8;padding:18px}a{color:#126b84}</style>\n<h1>Argentina24–41 Nueva Zelanda ·16/08/2025</h1><p>Validación con las etiquetas humanas guardadas el26/10/2026.881/881 bloques revisados:38 line,43 scrum y800 otro.880 bloques de7s y último de6,44s; cobertura de toda la transmisión de102:46,44.</p>\n<aside><strong>Referencia:</strong> al unir bloques contiguos quedan26 segmentos de line y15 de scrum. Se comparan con las105 alertas originales congeladas. Cada pareja debe coincidir en clase y solaparse al menos1s; cada alerta y segmento se usan una sola vez. Estos segmentos incluyen lo que se etiquetó en la transmisión y no se presentan como conteo oficial de jugadas únicas.</aside>\n<div class="scroll"><table data-table="human-metrics"><thead><tr><th>Clase</th><th>Segmentos humanos</th><th>Alertas</th><th>Aciertos</th><th>Sin pareja</th><th>Omitidos</th><th>Precisión</th><th>Sensibilidad</th><th>F1</th></tr></thead><tbody>' + table + '</tbody></table></div>\n<p>"Sin pareja" cuenta alertas que no coinciden con un segmento humano de la misma clase, además de duplicados sobre uno ya emparejado. Sin descontar el duplicado de scrum, la precisión por alerta sería15/54 (27,8%) y37/105 (35,2%) total; la tabla usa un único protocolo uno a uno.</p>\n<h2>Segmentos humanos sin detección de la misma clase</h2><div class="scroll"><table><thead><tr><th>Clase</th><th>Tiempo del video</th><th>Clips (desde1)</th></tr></thead><tbody>' + missing_table + '</tbody></table></div>\n<p>Al exigir3,5s de solapamiento, line pasa a20/26 (76,9%) y scrum se mantiene14/15 (93,3%). Con cualquier solapamiento positivo, los aciertos son iguales que con1s.</p>\n<h2>Alcance de la comparación</h2><ul>' + limits + '</ul><p><a href="' + report['labeling_url'] + '">Volver a las anotaciones manuales</a> · <a href="' + report['model_result_url'] + '">Ver detecciones originales</a> · <a href="metrics.json">Métricas y emparejamientos</a> · <a href="human_labels_snapshot.json">Copia de las etiquetas usadas</a> · <a href="../sampling_audit_20261026/index.html#reconciliacion">Revisión visual anterior</a></p></html>'
from human_reference_comparison import append_reference_comparison
document = append_reference_comparison(document, report)
from game1_sampling_comparison import append_sampling_comparison
document = append_sampling_comparison(document, report)
from game1_threeway_comparison import append_threeway_comparison
document = append_threeway_comparison(document, report)
from game1_fourway_comparison import append_fourway_comparison
document = append_fourway_comparison(document, report)
from recompute_corrected_labels import append_corrected_reference_link
document = append_corrected_reference_link(document, report)
(OUT / 'index.html').write_text(document, encoding='utf-8')
web = ROOT / 'src/web/static/reports/professional_rugby_human_20261026'
web.mkdir(exist_ok=True)
for name in ('metrics.json', 'human_labels_snapshot.json', 'index.html'):
    shutil.copyfile(OUT / name, web / name)
for path, expected in hashes.items():
    assert digest(Path(path)) == expected, 'A source changed during evaluation'
print(json.dumps(dict(status=report['status'], blocks=report['block_counts'], metrics=metrics, missed=report['missed_segments'], previous_visual_intervals_without_human=len(visual_without_human)), ensure_ascii=True))
