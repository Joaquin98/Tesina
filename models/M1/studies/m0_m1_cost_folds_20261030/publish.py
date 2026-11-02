"""One publisher for fold results and previously verified professional results."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.M1.studies.m0_m1_cost_folds_20261030.settings import *
import html, numpy as np
WEB = ROOT / 'src/web/static/reports/m0_m1_cost_folds_20261030'
NAMES = {'m0_control': 'M0 control Ã—1', 'm0_line4_scrum2': 'M0 line Ã—4 / scrum Ã—2', 'm1_control': 'M1 control Ã—1', 'm1_x2': 'M1 Ã—2'}

def pct(x):
    return f'{100 * x:.1f}%'.replace('.', ',')

def tm(x):
    n = round(x)
    return f'{n // 60}:{n % 60:02d}'

def cells(v):
    m = v['metrics']
    r = v['review']
    return ''.join(('<td>' + s + '</td>' for s in [f"{m['line']['tp']}/{m['line']['reference_segments']}", pct(m['line']['recall']), pct(m['line']['precision']), f"{m['scrum']['tp']}/{m['scrum']['reference_segments']}", pct(m['scrum']['recall']), pct(m['scrum']['precision']), pct(m['total']['recall']), pct(m['total']['precision']), tm(r['unique_flagged_s']), tm(r['flagged_outside_true_segments_s'])]))
HEAD = '<tr><th>Modelo</th><th>Lines</th><th>Sens. line</th><th>Prec. line</th><th>Scrums</th><th>Sens. scrum</th><th>Prec. scrum</th><th>Sens. total</th><th>Prec. total</th><th>RevisiÃ³n</th><th>Fuera de referencia</th></tr>'

def table(rows):
    return '<div class="scroll"><table><thead>' + HEAD + '</thead><tbody>' + rows + '</tbody></table></div>'

def main():
    WEB.mkdir(parents=True, exist_ok=True)
    variants = [read(p) for p in _model_artifact_glob(HERE / 'runs', '*/*/metrics.json', recursive=False)] if _resolve_model_artifact(HERE / 'runs').exists() else []
    professional = []
    source = read(PRO / 'results.json')
    for date in ['20250823', '20231020']:
        for key, family, _, _ in CONFIGS:
            oldkey = 'original' if key.endswith('control') else 'old_line4' if family == 'm0' else 'x2'
            v = next((v for v in source['variants'] if v['date'] == date and v['family'] == family and (v['key'] == oldkey) and (v['stride'] == 1)))
            professional.append(dict(configuration=key, **v))
    aggregate = {}
    for key, _, _, _ in CONFIGS:
        chosen = [v for v in variants if v['key'] == key]
        if not chosen:
            continue
        macro = {label: {metric: float(np.mean([v['metrics'][label][metric] for v in chosen])) for metric in ['precision', 'recall', 'f1']} for label in ['line', 'scrum', 'total']}
        counts = {label: {metric: sum((v['metrics'][label][metric] for v in chosen)) for metric in ['tp', 'fp', 'fn', 'alerts', 'reference_segments']} for label in ['line', 'scrum', 'total']}
        for label, m in counts.items():
            m.update(precision=m['tp'] / m['alerts'] if m['alerts'] else 0, recall=m['tp'] / m['reference_segments'] if m['reference_segments'] else 0, f1=2 * m['tp'] / (m['alerts'] + m['reference_segments']) if m['alerts'] + m['reference_segments'] else 0)
        aggregate[key] = dict(folds=len(chosen), complete=len(chosen) == 6, macro=macro, micro=counts, total_review_s=sum((v['review']['unique_flagged_s'] for v in chosen)), total_video_s=sum((v['evaluated_video_s'] for v in chosen)), mean_video_fraction=float(np.mean([v['review']['unique_flagged_s'] / v['evaluated_video_s'] for v in chosen])))
    errors = {p.name: read(p) for p in _model_artifact_glob(HERE, 'error_*.json', recursive=False)}
    final = _resolve_model_artifact(HERE / 'verification.json').exists() and read(HERE / 'verification.json').get('status') == 'PASS' and (len(variants) == 24) and (not errors)
    result = dict(status='error' if errors else 'complete' if final else 'running', completed_configurations=len(variants), expected_configurations=24, variants=variants, aggregate=aggregate, professional_partial_diagnostic=professional, errors=errors, extractors_executed=0, progress={f: read(HERE / f'progress_{f}.json') if _resolve_model_artifact(HERE / f'progress_{f}.json').exists() else {} for f in ['m0', 'm1']}, note='Six PV folds; fixed5epochs; controls refitted identically; independent held-out gradient separation, exploratory cost selection; historical PH scope pending.')
    save(HERE / 'results.json', result)
    save(WEB / 'results.json', result)
    sections = ''
    for fold in FOLDS:
        rows = ''
        for key, _, _, _ in CONFIGS:
            v = next((v for v in variants if v['fold'] == fold and v['key'] == key), None)
            rows += '<tr data-fold="' + fold + '" data-key="' + key + '"><th>' + NAMES[key] + '</th>' + (cells(v) if v else '<td colspan="10">Pendiente</td>') + '</tr>'
        sections += '<section class="fold"><h2>' + fold + ' Â· partido reservado V' + str(int(fold[-2:])) + '</h2>' + table(rows) + '</section>'
    pro = ''
    for date, label in [('20250823', 'Argentina-NZ Â· 23/08/2025'), ('20231020', 'Argentina-NZ Â· 20/10/2023')]:
        rows = ''.join(('<tr><th>' + NAMES[v['configuration']] + '</th>' + cells(v) + '</tr>' for v in professional if v['date'] == date))
        pro += '<section><h2>' + label + ' Â· referencia diagnÃ³stica automÃ¡tica parcial</h2>' + table(rows) + '</section>'
    agg = ''
    for key, a in aggregate.items():
        m = a['macro']
        agg += '<tr><th>' + NAMES[key] + '</th><td>' + str(a['folds']) + '/6</td><td>' + pct(m['line']['recall']) + '</td><td>' + pct(m['line']['precision']) + '</td><td>' + pct(m['scrum']['recall']) + '</td><td>' + pct(m['scrum']['precision']) + '</td><td>' + pct(m['total']['recall']) + '</td><td>' + pct(m['total']['precision']) + '</td><td>' + pct(a['mean_video_fraction']) + '</td></tr>'
    progress = '\n'.join((f.upper() + ': ' + str(result['progress'][f].get('fold', '')) + ' ' + str(result['progress'][f].get('key', '')) + ' Â· ' + str(result['progress'][f].get('stage', 'preparaciÃ³n')) + ' Â· Ã©poca ' + str(result['progress'][f].get('epoch', '-')) + '/5 Â· lote ' + str(result['progress'][f].get('batch', '-')) + '/' + str(result['progress'][f].get('batches', '-')) for f in ['m0', 'm1']))
    state = {'running': 'en curso', 'error': 'error registrado', 'complete': 'completo'}[result['status']]
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M0/M1: folds y profesionales</title><style>body{font:16px/1.5 system-ui;color:#243b4b;max-width:1450px;margin:28px auto;padding:0 18px}h1,h2{color:#142d42}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{padding:9px;text-align:left;border-bottom:1px solid #d6e2e8;vertical-align:top}thead{background:#142d42;color:white}aside{background:#edf5f6;padding:16px}section{margin:32px 0}a,summary{color:#087f8c}pre{white-space:pre-wrap;word-break:break-word;font-size:12px}</style><h1>M0 y M1 en partidos reservados</h1><aside><strong>' + f'{len(variants)}/24 evaluaciones Â· {state}' + '</strong><p>CachÃ© existente, sin extracciÃ³n nueva. La tabla se actualiza al recargar.</p><details><summary>Progreso de entrenamiento</summary><pre>' + html.escape(progress) + '</pre></details></aside><p>Se prueban seis folds de partidos V1-V6. En cada uno quedan fuera del reajuste el partido de desarrollo y el partido de prueba. Se reentrenan cuatro recetas durante cinco Ã©pocas fijas: M0 control, M0 line Ã—4/scrum Ã—2, M1 control y M1 Ã—2. Semilla 42, lote 16, Adam 0,0001. Balanceo y bosque ajustados sÃ³lo con entrenamiento; fusiÃ³n M1 fija. Se mantiene la resoluciÃ³n temporal original de las cachÃ©s, sin simular cada 25.</p><p>Se exige la misma clase y al menos 1 s de solapamiento, uno a uno. Las secciones no se fusionan. RevisiÃ³n como uniÃ³n temporal a 1Ã—, sin pausas. Fuera de referencia incluye falsas alertas y mÃ¡rgenes. TambiÃ©n se guardan mÃ©tricas por bloque y el criterio temporal histÃ³rico de 35 s, como secundarias. Los costos se eligieron tras observar VF/profesionales: prueba exploratoria, con procedencia histÃ³rica de inicializaciÃ³n parcialmente documentada.</p><h2>Promedio con igual peso por fold</h2><p>Los agregados con menos de 6/6 folds son parciales. El concatenado se conserva por separado en los datos.</p><div class="scroll"><table><thead><tr><th>Modelo</th><th>Folds</th><th>Sens. line</th><th>Prec. line</th><th>Sens. scrum</th><th>Prec. scrum</th><th>Sens. total</th><th>Prec. total</th><th>Video a revisar</th></tr></thead><tbody>' + agg + '</tbody></table></div>' + sections + '<h1>Otros profesionales: resultados ya calculados</h1><p>Inventarios diagnÃ³sticos automÃ¡ticos parciales: la sensibilidad es cobertura del inventario localizado. Las alertas sin pareja no son falsos positivos confirmados y la precisiÃ³n tabulada es sÃ³lo respecto de esa referencia. Estos resultados no equivalen a etiquetas humanas completas ni a los folds reservados. Se reutilizaron las evaluaciones verificadas del 29/10, sin repetir inferencia.</p>' + pro + '<p><a href="results.json">Datos completos, predicciones y tiempos</a> Â· <a href="/static/reports/m0_m1_cost_sampling_20261029/index.html">ComparaciÃ³n anterior: profesional humano, VF y todos los costos</a></p></html>'
    doc = doc.replace('<h1>Otros profesionales:', '<h1 id="profesionales">Otros profesionales:')
    if _resolve_model_artifact(HERE / 'scope_confirmation.json').exists():
        result['note'] = 'Six PV folds V1-V6 are included; five historical PH folds are excluded. The numerical protocol is fixed.'
        save(HERE / 'results.json', result)
        save(WEB / 'results.json', result)
        doc = doc.replace('Se prueban seis folds de partidos V1-V6.', 'Alcance confirmado: sÃ³lo los seis folds de partidos V1-V6; sin folds histÃ³ricos.')
    doc = doc.replace('<h2>Promedio con igual peso por fold</h2>', '<p><a href="#profesionales">Ver los resultados disponibles de los otros profesionales</a></p><h2>Promedio con igual peso por fold</h2>')
    (WEB / 'index.html').write_text(doc, encoding='utf-8')
    print(json.dumps(dict(status=result['status'], fold_runs=len(variants), professional_results=len(professional))), flush=True)
if __name__ == '__main__':
    main()
