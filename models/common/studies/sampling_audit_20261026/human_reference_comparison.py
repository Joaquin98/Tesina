"""Append manual labels, automated frame review and the external comparison to the report."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import hashlib
import html
import json
import re
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
WEB = ROOT / 'src/web/static/reports/professional_rugby_human_20261026'

def read(path):
    return json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8-sig'))

def pct(value):
    return f'{100 * value:.1f}%'.replace('.', ',')

def same_protocol(reference, predictions):
    result = {}
    for label in ('line', 'scrum'):
        ts = [t for t in reference if t['play_type'] == label]
        ps = [p for p in predictions if p['play_type'] == label]
        seconds = np.array([[max(0.0, min(t['end_s'], p['end_time']) - max(t['start_s'], p['start_time'])) for p in ps] for t in ts])
        valid = seconds >= 1.0
        score = valid.astype(float) + np.where(valid, seconds, 0.0) / ((seconds.max() + 1.0) * (min(seconds.shape) + 1.0))
        ii, jj = linear_sum_assignment(-score)
        tp = int(sum((valid[i, j] for i, j in zip(ii, jj))))
        result[label] = dict(tp=tp, reference_segments=len(ts), alerts=len(ps), precision=tp / len(ps), recall=tp / len(ts))
    nt = sum((v['reference_segments'] for v in result.values()))
    npred = sum((v['alerts'] for v in result.values()))
    tp = sum((v['tp'] for v in result.values()))
    result['total'] = dict(tp=tp, reference_segments=nt, alerts=npred, precision=tp / npred, recall=tp / nt)
    return result

def is_inventory_exclusion(action,aliases):
    return aliases.get(action,action)=='exclude_from_diagnostic_inventory'


def append_reference_comparison(document, human):
    sources = [HERE / 'reconciliation/reconciliation.json', HERE / 'official_sources/counts.json', HERE / 'reconciliation/2025-08-16_reference_v3.json', ROOT / f"data/results/result_{human['result_id']}.json"]
    recon, official, diagnostic_reference, original = [read(p) for p in sources]
    diagnostic = next((m for m in recon['matches'] if m['date'] == '2025-08-16'))
    correction_path = HERE / 'human_reference_user_corrections.json'
    corrections = read(correction_path)['corrections'] if _resolve_model_artifact(correction_path).exists() else []
    original_diagnostic_counts = dict(diagnostic['counts'])
    confirmations = read(correction_path).get('visible_event_confirmations', []) if _resolve_model_artifact(correction_path).exists() else []
    aliases=read(ROOT/'data/report_field_aliases.json')['inventory_actions'] if corrections else {}
    for correction in corrections:
        assert is_inventory_exclusion(correction['action'],aliases)
        event = next((e for e in diagnostic_reference['events'] if e['id'] == correction['reference_id']))
        assert (event['start_s'], event['end_s']) == (correction['start_s'], correction['end_s'])
        diagnostic_reference['events'].remove(event)
        diagnostic['counts'][event['play_type']] -= 1
    if corrections:
        sources.append(correction_path)
    internet = next((m for m in official['matches'] if m['date'] == '2025-08-16'))
    current = same_protocol(human['reference_segments'], original['detections'])
    for label in ('line', 'scrum', 'total'):
        for key in ('tp', 'reference_segments', 'alerts', 'precision', 'recall'):
            assert abs(current[label][key] - human['metrics'][label][key]) < 1e-10, 'Human metrics changed; do not silently replace the reference.'
    diagnostic_matched = same_protocol(diagnostic_reference['events'], original['detections'])
    assert [diagnostic_matched[k]['tp'] for k in ('line', 'scrum', 'total')] == [28, 12, 40]
    data = dict(match_date='2025-08-16', human_session_id=human['session_id'], result_id=human['result_id'], human_block_counts=human['block_counts'], human_segments={k: current[k]['reference_segments'] for k in ('line', 'scrum', 'total')}, diagnostic_visual_inventory=diagnostic['counts'], user_supplied_internet_counts=internet['user_supplied_totals'], verified_official_pdf_counts=internet['official_totals'], common_protocol='Same frozen original alerts; same-class one-to-one matching, overlap >=1s. No prior diagnostic correct/replay flags used to filter predictions.', human_metrics_common_protocol=current, diagnostic_metrics_common_protocol=diagnostic_matched, historical_diagnostic_metrics=diagnostic['summary'], historical_diagnostic_counts_before_user_corrections=original_diagnostic_counts, user_corrections=corrections, corrected_diagnostic_reference_events=diagnostic_reference['events'], visible_event_confirmations=confirmations, internet_metrics=None, internet_metrics_reason='Aggregate counts do not locate events or determine true positives / false negatives.', provenance=[dict(path=str(p), sha256=hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()) for p in sources])
    counts = [('Tus clips etiquetados', human['block_counts']['line'], human['block_counts']['scrum'], 'Bloques de hasta 7 segundos; una jugada puede ocupar varios clips.'), ('Tus segmentos agrupados', current['line']['reference_segments'], current['scrum']['reference_segments'], 'Sólo se unen clips contiguos de la misma clase. Sin distinción anotada de vivo, repetición o rearmado.'), ('Inventario diagnóstico automático por fotogramas, revisado', diagnostic['counts']['line'], diagnostic['counts']['scrum'], 'Inventario exploratorio de jugadas únicas en vivo; rearmados agrupados. Muestreo cada 2 s y ampliaciones selectivas.'), ('Conteo de Internet que pasaste', internet['user_supplied_totals']['line'], internet['user_supplied_totals']['scrum'], 'Se conserva tu tabla original. Para este partido coincide con los reinicios propios ganados, sin sumar los perdidos.'), ('PDF de la competición verificado', internet['official_totals']['line'], internet['official_totals']['scrum'], 'Reinicios propios ganados más perdidos de ambos equipos; no se vuelven a sumar los robados.')]
    count_rows = ''.join((f'<tr><td>{html.escape(name)}</td><td>{line}</td><td>{scrum}</td><td>{line + scrum}</td><td>{html.escape(note)}</td></tr>' for name, line, scrum, note in counts))
    metric_rows = ''
    for label in ('line', 'scrum', 'total'):
        h, a = (current[label], diagnostic_matched[label])
        metric_rows += f"<tr><td>{label}</td><td>{h['tp']}/{h['alerts']} ({pct(h['precision'])})</td><td>{h['tp']}/{h['reference_segments']} ({pct(h['recall'])})</td><td>{a['tp']}/{a['alerts']} ({pct(a['precision'])})</td><td>{a['tp']}/{a['reference_segments']} ({pct(a['recall'])})</td></tr>"
    all_matches = []
    all_rows = ''
    for m in official['matches']:
        v = next((v for v in recon['matches'] if v['date'] == m['date']))
        u, o = (m['user_supplied_totals'], m['official_totals'])
        all_matches.append(dict(date=m['date'], diagnostic_visual=v['counts'], user_internet=u, verified_pdf=o, official_source=m['source_url']))
        all_rows += f"<tr><td>{m['date']}</td><td>{u['line']} / {u['scrum']}</td><td>{v['counts']['line']} / {v['counts']['scrum']}</td><td>{o['line']} / {o['scrum']}</td><td><a href='../sampling_audit_20261026/official_{m['pdf']}'>PDF verificado</a></td></tr>"
    data['three_match_count_comparison'] = all_matches
    section = '<section id="comparacion-referencias"><h2>Comparación: las anotaciones manuales, revisión diagnóstica automática por fotogramas y datos de Internet</h2>\n<p>La revisión por fotos es la evaluación diagnóstica automática, distinta de las detecciones del modelo. La siguiente tabla conserva los conteos de cada fuente y explica su unidad.</p>\n<div class="scroll"><table data-table="reference-counts"><thead><tr><th>Referencia · 16/08/2025</th><th>Line</th><th>Scrum</th><th>Suma</th><th>Qué cuenta</th></tr></thead><tbody>' + count_rows + '</tbody></table></div>\n<p><strong>La diferencia 29/10 frente a 32/12:</strong> en el PDF, Argentina ganó 14 lines propios y perdió 3; NZ ganó 15 y perdió 0. El total es 14+3+15+0 = <strong>32</strong>. En scrum, Argentina 6 ganados y 1 perdido; NZ 4 y 1: 6+1+4+1 = <strong>12</strong>. Los 29 y 10 de tu consulta corresponden a sumar únicamente los ganados. <a href="../sampling_audit_20261026/official_20250816.pdf">Ver PDF de la competición</a>.</p>\n<h3>Mismas 105 alertas originales evaluadas con dos referencias</h3>\n<p>Para comparar las métricas se recalcularon ambas con el mismo criterio: misma clase, al menos 1 segundo de solapamiento y emparejamiento uno a uno. No se filtran las alertas con los veredictos de la revisión diagnóstica anterior. Aun así, las referencias tienen unidades distintas: tus segmentos contiguos frente al inventario diagnóstico automático de jugadas únicas en vivo.</p>\n<div class="scroll"><table data-table="reference-metrics"><thead><tr><th>Clase</th><th>Precisión · las anotaciones manuales</th><th>Sensibilidad · las anotaciones manuales</th><th>Precisión · inventario diagnóstico automático</th><th>Sensibilidad · inventario diagnóstico automático</th></tr></thead><tbody>' + metric_rows + '</tbody></table></div>\n<p>La precisión histórica de la revisión diagnóstica automática era 56,9% en line, 29,6% en scrum y 42,9% total: contaba alertas correctas incluso repetidas sobre una misma jugada. Aquí se muestran 54,9%, 22,2% y 38,1% al usar una única pareja por jugada. La sensibilidad actual se calcula con el inventario corregido que muestra la tabla. Tus métricas guardadas se conservan.</p>\n<p><strong>Internet aporta totales, no precisión o sensibilidad del modelo.</strong> Sin tiempos de cada jugada no permite saber qué detectó u omitió el modelo; no se calculan porcentajes cambiando sólo el denominador.</p>\n<h3>Tu tabla de Internet y la revisión diagnóstica automática de los tres partidos</h3><p>Valores expresados como <strong>line / scrum</strong>. La revisión diagnóstica automática es la última reconciliada, no el primer inventario.</p>\n<div class="scroll"><table data-table="internet-three-matches"><thead><tr><th>Partido</th><th>Internet: aporte original tuyo</th><th>Revisión por fotogramas automáticos</th><th>PDF de competición</th><th>Fuente</th></tr></thead><tbody>' + all_rows + '</tbody></table></div>\n<p>23/08: faltan localizar dos lines respecto del PDF. Los 9 scrums de tu consulta también aparecían en ESPN, frente a 7 del PDF; el video tiene dos rearmados. Contarlos aparte es una explicación compatible, pero la definición del proveedor no está confirmada. 2023: quedan dos lines sin localizar respecto del PDF de World Rugby; también se registró un conflicto con SA Rugby, que informa 27. Se conservan las diferencias pendientes.</p>\n<p><a href="reference_comparison.json">Datos de esta comparación y trazabilidad</a> · <a href="../sampling_audit_20261026/index.html#reconciliacion">Fotogramas y reconciliación anterior</a></p></section>'
    if corrections:
        correction_note = '<aside data-note="manual-correction"><strong>Corrección de la anotación manual:</strong> en 32:00–32:06 (clips 275–276) no hay line. Se excluye line09 del inventario diagnóstico automático: pasa de 32 a 31 lines y conserva 12 scrums. Queda una diferencia de un line respecto de los 32 del PDF, todavía sin resolver. Anotaciones manuales y las detecciones del modelo no cambian. En el cruce entre inventarios quedan seis lines sólo en el inventario diagnóstico y uno sólo en las anotaciones manuales: 31−6+1 = 26.</aside>'
        section = section.replace('<div class="scroll"><table data-table="reference-counts">', correction_note + '<div class="scroll"><table data-table="reference-counts">', 1)
    if confirmations:
        replay_note = '<p data-note="confirmed-scrum-replay"><strong>43:45–43:52, clip 376: sí hay scrum.</strong> Tu etiqueta scrum es correcta. Los fotogramas muestran una repetición, ya registrada así en la revisión diagnóstica automática del candidato 45. El modelo también lo detectó (43:41–43:57). El inventario diagnóstico automático de jugadas únicas no lo sumó de nuevo: el scrum en vivo ya figura en 42:44–42:58. Este caso es una diferencia de alcance entre referencias, no una omisión del modelo ni un motivo para cambiar tu etiqueta.</p>'
        section = section.replace('<h3>Mismas 105 alertas', replay_note + '<h3>Mismas 105 alertas', 1)
    if 'id="comparacion-referencias"' in document:
        document, n = re.subn('<section id="comparacion-referencias">.*?</section>', lambda _: section, document, count=1, flags=re.S)
        assert n == 1
    else:
        marker = '<h2>Alcance de la comparación</h2>'
        assert marker in document, 'Report structure changed: insert comparison explicitly.'
        document = document.replace(marker, section + marker, 1)
    for folder in (HERE / 'human_validation', WEB):
        assert _resolve_model_artifact(folder).is_dir()
        (folder / 'reference_comparison.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return document
if __name__ == '__main__':
    human = read(HERE / 'human_validation/metrics.json')
    for path in (HERE / 'human_validation/index.html', WEB / 'index.html'):
        before = _resolve_model_artifact(path).read_bytes()
        revised = append_reference_comparison(before.decode('utf-8'), human)
        assert _resolve_model_artifact(path).read_bytes() == before, 'Concurrent report update: retry instead of overwriting.'
        temp = path.with_suffix('.references.tmp')
        temp.write_text(revised, encoding='utf-8')
        temp.replace(path)
    print(json.dumps(dict(status='updated', sources=['human_labels', 'diagnostic_frame_review', 'user_internet_counts', 'official_PDF'], model_inference_executed=False)))
