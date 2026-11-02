"""Publish fixed comparisons without changing source results or live models."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.source_layout import source_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, json, copy, html, urllib.request, hashlib, importlib.util
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(ROOT / 'data/m1_recall_cost_20261027')), str(ROOT)]
spec = importlib.util.spec_from_file_location('prior_cost_publication', _resolve_model_artifact(ROOT / 'models/M1/studies/m1_recall_cost_20261027/publish.py'))
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
from core import evaluate_events
WEB = ROOT / 'src/web/static/reports/m1_cost8_other_professionals_20261027'
BASE = 'http://127.0.0.1:5000'

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(p).open('rb') as f:
        for c in iter(lambda: f.read(1048576), b''):
            h.update(c)
    return h.hexdigest()

def pct(v):
    return f'{100 * v:.2f}%'.replace('.', ',')

def tm(v):
    return f'{int(round(v)) // 60}:{int(round(v)) % 60:02d}'

def post(url, payload):
    request = urllib.request.Request(BASE + url, data=json.dumps(payload, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request) as r:
        return json.load(r)

def get(url):
    with urllib.request.urlopen(BASE + url) as r:
        return json.load(r)

def timeline(label, events, duration, truth=False):
    rects = ''.join((f'''<rect x="{1000 * e['start_s' if truth else 'start_time'] / duration:.3f}" y="2" width="{max(0.3, 1000 * (e['end_s' if truth else 'end_time'] - e['start_s' if truth else 'start_time']) / duration):.3f}" height="18" fill="{('#168b88' if e['play_type'] == 'line' else '#d48a23')}"><title>{e['play_type']} {tm(e['start_s' if truth else 'start_time'])}–{tm(e['end_s' if truth else 'end_time'])}</title></rect>''' for e in events))
    return f'<div class="timeline"><strong>{html.escape(label)}</strong><svg viewBox="0 0 1000 22" role="img" aria-label="{html.escape(label)}"><rect width="1000" height="22" fill="#edf1f4"/>{rects}</svg></div>'

def main():
    completion = read(HERE / 'completion.json')
    protocol = read(HERE / 'protocol.json')
    assert completion['status'] == 'complete' and len(completion['records']) == 3
    assert completion['extractors_executed'] == 0 and completion['source_hashes_unchanged'] and (not completion['new_training'])
    for p, h in protocol['source_hashes'].items():
        assert sha(Path(p)) == h
    records = completion['records']
    proofs = {}
    games = {}
    summary = []
    for record in records:
        key = record['job']
        date = record['date']
        stride = record['stride']
        truth = record['reference_segments']
        baseline = read(ROOT / f"data/results/result_{record['baseline_result_id']}.json")
        predicted = read(HERE / key / 'events.json')
        rows = [json.loads(x) for x in _resolve_model_artifact(HERE / key / 'windows.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows) == record['windows']
        for mode, es in [('original', baseline['detections']), ('cost8', predicted)]:
            result = record[mode]
            checked = evaluate_events(truth, es)
            assert checked['metrics'] == result['metrics']
            proof = prior.independent_check(truth, es, result['metrics'], record['duration_s'])
            for k in ('unique_flagged_s', 'flagged_outside_true_segments_s'):
                assert abs(proof[k] - result['review'][k]) < 1e-06
            proofs[key + '_' + mode] = proof
        for p, h in record['model_hashes'].items():
            assert sha(Path(p)) == h
        sparse_name = 'completo' if stride == 1 else 'cada 25'
        for mode, es, ws in [('original', baseline['detections'], baseline['window_predictions']), ('cost8', predicted, rows)]:
            title = ('Original' if mode == 'original' else 'Costo ×8') + ' · ' + sparse_name
            payload = copy.deepcopy(baseline)
            payload.setdefault('result_id', record['baseline_result_id'])
            for drop in ('sampling_comparison', 'exploratory_review', 'processing_timing', 'manual_comparison', 'omission_review'):
                payload.pop(drop, None)
            if mode == 'cost8':
                rid = f'20261027_argentina_allblacks_{date}_m1_cost8_stride{stride}'
                payload.update(result_id=rid, model_name='M1 · ' + title, detections=es, primary_detections=copy.deepcopy(es), window_predictions=ws)
                by = {label: dict(count=sum((e['play_type'] == label for e in es)), total_duration=sum((e['end_time'] - e['start_time'] for e in es if e['play_type'] == label))) for label in ('line', 'scrum')}
                payload['statistics'] = dict(by_type=by, avg_confidence=sum((e['confidence'] for e in es)) / len(es) if es else 0, avg_duration=sum((e['end_time'] - e['start_time'] for e in es)) / len(es) if es else 0)
                payload['summary'] = dict(total_detections=len(es), by_type={k: v['count'] for k, v in by.items()})
                payload['metadata'].update(source_configuration='ALL-DEVELOPMENT-M1-H2-D2-S42-COST8', training_class_cost=8, training_models=record['model_hashes'], source_result_id=record['baseline_result_id'], new_training=False, experimental_sampling_stride=stride, saved_decisions_locked=True, extractors_executed=0, cached_evaluation_seconds=record['timing']['cached_cost8_inference_s'], human_labels_available=False, reference_scope='partial automated diagnostic inventory; exploratory temporal matching, not certified human/full-match accuracy', experiment_report_url=BASE + '/static/reports/m1_cost8_other_professionals_20261027/index.html')
                path = ROOT / f'data/results/result_{rid}.json'
                if _resolve_model_artifact(path).exists():
                    assert read(path) == payload
                else:
                    save(path, payload)
            rid = payload['result_id']
            games.setdefault(date, dict(video_path=payload['video_path'], results={}))
            games[date]['results'][title] = payload
            summary.append(dict(key=key + '_' + mode, date=date, title=title, stride=stride, mode=mode, viewer_url='/viewer?result_id=' + rid, metrics=record[mode]['metrics'], review=record[mode]['review'], missed_segments=record[mode]['missed_segments'], review_coverage=record[mode]['review_coverage'], cached_inference_s=record['timing']['cached_cost8_inference_s'] if mode == 'cost8' else None))
    for date, game in games.items():
        cid = f'20261027_argentina_allblacks_{date}_original_vs_cost8'
        history = dict(id=cid, video_path=game['video_path'], video_name=f'Argentina–NZ · {date} · Original vs ×8', model_names=list(game['results']), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=game['results'], comparison={name: dict(total=len(p['detections']), by_type={label: sum((e['play_type'] == label for e in p['detections'])) for label in ('line', 'scrum')}, avg_confidence=p['statistics']['avg_confidence']) for name, p in game['results'].items()})
        assert post('/api/comparison-history', history)['success']
        assert get('/api/comparison-history/' + cid)['comparison']['results'] == game['results']
        game['comparison_url'] = '/compare?comparison=' + cid
    result = dict(status='complete', variants=summary, protocol=protocol, elapsed_s=completion['elapsed_s'], extractors_executed=0, new_training=False, comparisons={d: g['comparison_url'] for d, g in games.items()}, reference_scope=protocol['reference_scope'])
    save(HERE / 'results.json', result)
    save(WEB / 'results.json', result)
    esc = html.escape
    sections = []
    for date, game in games.items():
        variants = [v for v in summary if v['date'] == date]
        record = next((r for r in records if r['date'] == date))
        truth = record['reference_segments']
        title = 'Argentina 29–23 Nueva Zelanda · 23/08/2025' if date == '20250823' else 'Argentina 6–44 Nueva Zelanda · 20/10/2023'
        rows = ''
        classes = ''
        misses = ''
        for v in variants:
            m = v['metrics']['total']
            l = v['metrics']['line']
            s = v['metrics']['scrum']
            rev = v['review']
            rows += f'''<tr><th>{esc(v['title'])}</th><td>{l['tp']}/{l['reference_segments']}</td><td>{s['tp']}/{s['reference_segments']}</td><td>{pct(m['recall'])}</td><td>{pct(m['precision'])}</td><td>{m['alerts']}</td><td>{m['fp']}</td><td>{tm(rev['unique_flagged_s'])}</td><td>{tm(rev['flagged_outside_true_segments_s'])}</td><td>{(tm(v['cached_inference_s']) if v['cached_inference_s'] is not None else 'histórico')}</td><td><a href="{esc(v['viewer_url'])}">Ver</a></td></tr>'''
            for label in ('line', 'scrum'):
                cm = v['metrics'][label]
                classes += f"<tr><th>{esc(v['title'])}</th><td>{label}</td><td>{cm['tp']}/{cm['reference_segments']}</td><td>{pct(cm['recall'])}</td><td>{pct(cm['precision'])}</td><td>{pct(cm['f1'])}</td></tr>"
            missed = ', '.join((f"{t['play_type']} {tm(t['start_s'])}–{tm(t['end_s'])}" for t in v['missed_segments'])) or 'Ninguno del inventario'
            misses += f"<tr><th>{esc(v['title'])}</th><td>{esc(missed)}</td></tr>"
        lines = timeline('Inventario visual diagnóstico automático · incompleto', truth, record['duration_s'], True)
        for name, p in game['results'].items():
            lines += timeline(name, p['detections'], record['duration_s'])
        restriction = 'Sólo existe caché cada 25; no se ejecutó una versión completa.' if date == '20231020' else 'Caché completo: mismas ventanas, dos frecuencias de entrada.'
        sections.append(f'''<section data-game="{date}"><h2>{title}</h2><p>{restriction} Referencia: {sum((t['play_type'] == 'line' for t in truth))} line y {sum((t['play_type'] == 'scrum' for t in truth))} scrum únicos en vivo. Faltan dos lines por localizar frente al PDF de competición.</p><p><a href="{esc(game['comparison_url'])}">Comparar detecciones con video en la herramienta</a></p><div class="scroll"><table data-table="totals"><thead><tr><th>Variante</th><th>Line</th><th>Scrum</th><th>Sensibilidad sobre inventario</th><th>Precisión temporal exploratoria</th><th>Alertas</th><th>Sin pareja</th><th>Revisión única</th><th>Fuera del inventario</th><th>Inferencia caché</th><th>Video</th></tr></thead><tbody>{rows}</tbody></table></div><h3>Timelines · verde line / naranja scrum</h3>{lines}<h3>Por clase</h3><div class="scroll"><table><thead><tr><th>Variante</th><th>Clase</th><th>Encontrados</th><th>Sensibilidad</th><th>Precisión temporal</th><th>F1</th></tr></thead><tbody>{classes}</tbody></table></div><h3>Sin pareja correcta en el inventario</h3><div class="scroll"><table><thead><tr><th>Variante</th><th>Tiempos absolutos de transmisión</th></tr></thead><tbody>{misses}</tbody></table></div></section>''')
    observations = []
    for record in records:
        a = record['original']
        b = record['cost8']
        am = a['metrics']['total']
        bm = b['metrics']['total']
        name = ('23/08/2025' if record['date'] == '20250823' else '20/10/2023') + (' completo' if record['stride'] == 1 else ' cada 25')
        observations.append(f"<li><strong>{name}:</strong> original {am['tp']}/{am['reference_segments']}, ×8 {bm['tp']}/{bm['reference_segments']}; revisión {tm(a['review']['unique_flagged_s'])} → {tm(b['review']['unique_flagged_s'])}; alertas {am['alerts']} → {bm['alerts']}.</li>")
    overview = '<h2>Resultado de las tres pruebas</h2><ul>' + ''.join(observations) + '</ul>'
    document = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 ×8 · otros profesionales</title><style>body{font:16px system-ui;line-height:1.5;color:#243447;max-width:1500px;margin:28px auto;padding:0 18px}h1{font-size:1.9rem}aside{padding:16px;background:#fff0d6}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;margin:18px 0}td,th{padding:10px;text-align:left;border-bottom:1px solid #dbe2e9}thead{background:#edf3f8}a{color:#126b84}.timeline{margin:10px 0}.timeline svg{display:block;width:100%;height:26px}section{margin:36px 0}code{overflow-wrap:anywhere}</style><h1>M1 ×8 en los otros partidos profesionales</h1><p>27/10/2026. Mismo modelo ×8 ya entrenado, argmax y agrupación temporal originales. Sólo caché existente: cero extractores, sin abrir videos durante la inferencia y sin nuevo entrenamiento.</p><aside><strong>Alcance de las métricas.</strong> Estos dos partidos todavía no tienen las anotaciones manuales completas. Se usan inventarios visuales diagnósticos automáticos reconciliados, que siguen incompletos. Sensibilidad es sobre ese inventario; precisión es la proporción de alertas con pareja de la misma clase y al menos 1 segundo de solapamiento, con una pareja por jugada. Las alertas sin pareja NO son todas falsas confirmadas: pueden incluir repeticiones, rearmados o jugadas ausentes del inventario. Este criterio reemplaza, para esta comparación, la precisión visual histórica y no debe mezclarse con ella. La revisión única es tiempo marcado sin solapamientos, a velocidad 1× y sin pausas. El tiempo fuera del inventario tampoco es costo falso confirmado ni garantiza revisar completa cada jugada.</aside><p><a href="results.json">Resultados y trazabilidad</a> · <a href="verification.json">Verificación independiente</a> · <a href="/static/reports/m1_recall_cost_20261027/index.html">Prueba anterior con las anotaciones manuales del 16/08/2025</a></p>' + overview + ''.join(sections) + '<h2>Verificación y límites</h2><p>Se reprodujeron controles originales distribuidos por cada transmisión antes de usar ×8, y se verificó el posprocesado de todas las ventanas originales. Se conservaron la selección, interpolación float32, 175 posiciones válidas más 25 ceros, geometría rint100 y fusión 0,475. Los pesos y cachés fuente permanecen intactos. Los tiempos son de evaluación desde caché, no de procesamiento nuevo de video. No se ajustaron pesos, umbrales o hiperparámetros usando estos resultados. Las diferencias de inventario y muestreo impiden afirmar una sensibilidad completa certificada o superioridad general en profesionales.</p></html>'
    for path in (HERE / 'report.html', WEB / 'index.html'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding='utf-8')
    proof = dict(status='PASS', independent_matching_and_interval_checks=proofs, source_hashes_unchanged=True, all_baseline_controls=[r['baseline_validation'] for r in records], native_comparison_roundtrip=True, configurations=3, source_cache_stats_unchanged=completion['source_cache_stats_unchanged'], extractors_executed=0, report_url=BASE + '/static/reports/m1_cost8_other_professionals_20261027/index.html', publisher_sha256=sha(Path(__file__)))
    save(HERE / 'verification.json', proof)
    save(WEB / 'verification.json', proof)
    print(json.dumps(dict(status='published', report_url=proof['report_url'], variants=[dict(title=v['title'], date=v['date'], metrics=v['metrics']['total']) for v in summary])))
if __name__ == '__main__':
    main()
