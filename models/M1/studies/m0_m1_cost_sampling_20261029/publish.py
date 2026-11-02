"""Publish verified results, clear pending rows, and paired native viewers."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.M1.studies.m0_m1_cost_sampling_20261029.common import *
import html, copy, urllib.request, argparse
from core import evaluate_events
WEB = ROOT / 'src/web/static/reports/m0_m1_cost_sampling_20261029'
BASE = 'http://127.0.0.1:5000'
PUBLICATION_SCHEMA = 2
NAMES = {'original': 'Control Ã—1', 'x2': 'Line Ã—2 / scrum Ã—2', 'old_line4': 'Line Ã—4 / scrum Ã—2', 'old_line8': 'Line Ã—8 / scrum Ã—2', 'line2': 'Line Ã—2 / scrum Ã—1', 'line4': 'Line Ã—4 / scrum Ã—1', 'line8': 'Line Ã—8 / scrum Ã—1'}

def post(path, value):
    request = urllib.request.Request(BASE + path, json.dumps(value).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=60) as r:
        return json.load(r)

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.load(r)

def tm(t):
    n = int(round(t))
    return f'{n // 60}:{n % 60:02d}'

def pct(v):
    return f'{100 * v:.2f}%'.replace('.', ',')

def payload(job, key, stride, family):
    folder = HERE / f"evaluation/{job['date']}/{family}_{key}_stride{stride}"
    v = read(folder / 'metrics.json')
    d = copy.deepcopy(read(ROOT / f"data/results/result_{job['baseline']}.json"))
    events = read(folder / 'events.json')
    rid = f"20261029_argentina_allblacks_{job['date']}_{family}_{key}_stride{stride}"
    d.pop('omission_review', None)
    d.update(result_id=rid, model_name=f'{family.upper()} Â· {NAMES[key]} Â· ' + ('completo' if stride == 1 else 'cada 5'), detections=events, primary_detections=copy.deepcopy(events), window_predictions=[json.loads(x) for x in _resolve_model_artifact(folder / 'windows.jsonl').read_text(encoding='utf-8').splitlines()], human_evaluation=v['metrics'])
    meta = d.setdefault('metadata', {})
    meta.update(source_configuration=f'{family.upper()}-H2-D2-S42-{key}', saved_decisions_locked=True, experimental_sampling_stride=stride, source_model_hashes=v.get('model_hashes', {}), source_result=v.get('source_result'), uses_yolo=family == 'm1', uses_geometry_forest=family == 'm1', protocol='RGB200;175 valid+25padding; no pose/RF' if family == 'm0' else 'M1 dual temporal+geometry RF', training_class_multipliers=[next((c[1] for c in CONFIGS if c[0] == key)), 1, next((c[2] for c in CONFIGS if c[0] == key))], extractors_executed=0, human_evaluation_report='/static/reports/m0_m1_cost_sampling_20261029/index.html')
    for field in ['training_class_cost', 'source_result_id', 'imported_from_completed_experiment', 'imported_at']:
        meta.pop(field, None)
    hashes = v.get('model_hashes') or {str(p): sha(p) for p in ([model_path(key)] if family == 'm0' else m1_paths(key))}
    meta.update(publication_schema=PUBLICATION_SCHEMA, source_model_hashes=hashes, training_models=hashes, new_training=family == 'm0' and key != 'x2', event_policy_source_result_id=job['baseline'], experimental_interpolation='none' if stride == 1 else 'linear_selected_cached_features', sampling_note=('RGB only' if family == 'm0' else 'RGB and pose') + f': stride {stride}, replayed from native cache; no extractor executed.')
    counts = {k: sum((e['play_type'] == k for e in events)) for k in ['line', 'scrum']}
    d['summary'] = dict(total_detections=len(events), by_type=counts)
    d['statistics'] = dict(avg_confidence=sum((e['confidence'] for e in events)) / len(events) if events else 0, avg_duration=sum((e['duration'] for e in events)) / len(events) if events else 0, by_type={k: dict(count=counts[k], total_duration=sum((e['duration'] for e in events if e['play_type'] == k))) for k in counts})
    if job['kind'] == 'human':
        ref = read(job['reference'])
        labels = read(job['reference'].parent / 'labels.json')['labels']
        d['manual_comparison'] = dict(reference_session_pinned=True, composite_reference=True, reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], session_id=ref['session_ids'][0], source_session_ids=ref['session_ids'], labels=labels, summary=dict(unit='blocks', reviewed=881, total_candidates=881, pending=0, final_counts=ref['block_counts']))
    else:
        d.pop('manual_comparison', None)
    dest = ROOT / f'data/results/result_{rid}.json'
    if _resolve_model_artifact(dest).exists():
        previous = read(dest)
        if previous != d:
            assert previous.get('metadata', {}).get('publication_schema', 0) < PUBLICATION_SCHEMA
            assert {k: v for k, v in previous.items() if k != 'metadata'} == {k: v for k, v in d.items() if k != 'metadata'}
            save(dest, d)
    else:
        save(dest, d)
    return d

def native_pairs():
    pairs = {}
    for job in JOBS:
        for stride in [1, 5] if job['kind'] == 'human' else [1]:
            for key, _, _ in CONFIGS:
                paths = [HERE / f"evaluation/{job['date']}/{family}_{key}_stride{stride}/metrics.json" for family in ['m0', 'm1']]
                if not all((_resolve_model_artifact(p).exists() for p in paths)):
                    continue
                cid = f"20261029_argentina_allblacks_{job['date']}_m0_vs_m1_{key}_stride{stride}"
                marker = HERE / 'published' / f'{cid}.json'
                if _resolve_model_artifact(marker).exists() and read(marker).get('publication_schema') == PUBLICATION_SCHEMA:
                    pairs[f"{job['date']}_{key}_{stride}"] = read(marker)['url']
                    continue
                ds = [payload(job, key, stride, f) for f in ['m0', 'm1']]
                records = {d['model_name']: d for d in ds}
                ref = read(job['reference'])
                truth = ref['reference_segments'] if job['kind'] == 'human' else ref['events']
                matched = {}
                for name, d in records.items():
                    matched[name] = {x['reference_id'] for x in evaluate_events(truth, d['detections'])['assignments']}
                if job['kind'] == 'human':
                    names = list(records)
                    ds[0]['omission_review'] = dict(reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], context_s=3, method='same-class one-to-one overlap>=1s', models=[dict(key=f'model_{i}', title=name, missed=records[name]['human_evaluation']['total']['fn'], prediction_signature=[[e['play_type'], e['start_time'], e['end_time']] for e in records[name]['detections']]) for i, name in enumerate(names)], segments=[dict(**t, detected={f'model_{i}': t['id'] in matched[name] for i, name in enumerate(names)}) for t in truth if not all((t['id'] in m for m in matched.values()))])
                history = dict(id=cid, video_path=ds[0]['video_path'], video_name=f"Argentina-NZ {job['date']} Â· M0 vs M1 Â· {NAMES[key]} Â· stride{stride}", model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={n: dict(total=len(d['detections']), by_type=d['summary']['by_type'], avg_confidence=d['statistics']['avg_confidence']) for n, d in records.items()})
                assert post('/api/comparison-history', history)['success']
                assert get('/api/comparison-history/' + cid)['comparison']['results'] == records
                url = '/compare?comparison=' + cid + ('#omisiones' if job['kind'] == 'human' else '')
                save(marker, dict(url=url, result_ids=[d['result_id'] for d in ds], publication_schema=PUBLICATION_SCHEMA))
                pairs[f"{job['date']}_{key}_{stride}"] = url
    return pairs

def main(native=False):
    WEB.mkdir(parents=True, exist_ok=True)
    pairs = native_pairs() if native else {}
    if not native:
        for job in JOBS:
            for stride in [1, 5] if job['kind'] == 'human' else [1]:
                for key, _, _ in CONFIGS:
                    marker = HERE / 'published' / f"20261029_argentina_allblacks_{job['date']}_m0_vs_m1_{key}_stride{stride}.json"
                    if _resolve_model_artifact(marker).exists():
                        pairs[f"{job['date']}_{key}_{stride}"] = read(marker)['url']
    variants = []
    for file in _model_artifact_glob(HERE / 'evaluation', '*/*/metrics.json', recursive=False):
        value = read(file)
        value['result_folder'] = str(file.parent)
        variants.append(value)
    differences = []
    for date, stride in [('20250816', 1), ('20250816', 5), ('20250823', 1), ('20231020', 1), ('VF', 25)]:
        if date == 'VF':
            truth = read(ROOT / 'data/m1_recall_cost_20261027/vf25_results.json')['reference_segments']
        else:
            job = next((j for j in JOBS if j['date'] == date))
            ref = read(job['reference'])
            truth = ref['reference_segments'] if job['kind'] == 'human' else ref['events']
        byid = {t['id']: t for t in truth}
        for key, _, _ in CONFIGS:
            pair = [next((v for v in variants if v['date'] == date and v['stride'] == stride and (v['key'] == key) and (v['family'] == family)), None) for family in ['m0', 'm1']]
            if any((v is None for v in pair)):
                continue
            m0, m1 = pair
            s0 = {a['reference_id'] for a in m0['assignments']}
            s1 = {a['reference_id'] for a in m1['assignments']}
            gained = [byid[i] for i in sorted(s0 - s1)]
            lost = [byid[i] for i in sorted(s1 - s0)]
            for label in ['line', 'scrum']:
                assert sum((t['play_type'] == label for t in gained)) - sum((t['play_type'] == label for t in lost)) == m0['metrics'][label]['tp'] - m1['metrics'][label]['tp']
            differences.append(dict(date=date, stride=stride, key=key, m0_recovers=gained, m0_loses=lost, m0_minus_m1_review_s=m0['review']['unique_flagged_s'] - m1['review']['unique_flagged_s'], m0_minus_m1_extra_s=m0['review']['flagged_outside_true_segments_s'] - m1['review']['flagged_outside_true_segments_s']))
    training = read(HERE / 'training_status.json') if _resolve_model_artifact(HERE / 'training_status.json').exists() else dict(status='running', models=[read(p) for p in sorted(_model_artifact_glob(HERE / 'models', '*/complete.json', recursive=False))])
    p = read(HERE / 'progress.json') if _resolve_model_artifact(HERE / 'progress.json').exists() else {}
    ep = read(HERE / 'evaluation_progress.json') if _resolve_model_artifact(HERE / 'evaluation_progress.json').exists() else {}
    errors = {f.name: read(f) for f in _model_artifact_glob(HERE, '*_error.json', recursive=False)}
    all_complete = len(variants) == 70 and _resolve_model_artifact(HERE / 'training_complete.json').exists() and (not errors)
    qa = read(HERE / 'web_verification.json') if _resolve_model_artifact(HERE / 'web_verification.json').exists() else {}
    result = dict(status='error' if errors else 'complete' if all_complete and qa.get('status') == 'PASS' else 'validating' if all_complete else 'running', expected_configurations=70, completed_configurations=len(variants), training=training, progress=p, evaluation_progress=ep, errors=errors, variants=variants, pairs=pairs, extractors_executed=0, reference_note='16/08 human fd96032b00fa;23/08 and2023 partial automated diagnostic reference; VF human blocks central every25')
    result['paired_differences'] = differences
    save(HERE / 'results.json', result)
    save(WEB / 'results.json', result)
    sections = ''
    for date, stride, label in [('20250816', 1, '16/08/2025 Â· las anotaciones manuales Â· completo'), ('20250816', 5, '16/08/2025 Â· las anotaciones manuales Â· cada 5'), ('20250823', 1, '23/08/2025 Â· inventario diagnÃ³stico automÃ¡tico parcial Â· completo'), ('20231020', 1, '20/10/2023 Â· inventario diagnÃ³stico automÃ¡tico parcial Â· completo'), ('VF', 25, 'VF amateur Â· S01-S04 Â· central cada 25')]:
        rows = ''
        omissions = ''
        for key, _, _ in CONFIGS:
            for family in ['m0', 'm1']:
                v = next((v for v in variants if v['date'] == date and v['stride'] == stride and (v['key'] == key) and (v['family'] == family)), None)
                name = family.upper() + ' Â· ' + NAMES[key]
                if not v:
                    rows += f'<tr class="pending"><th>{html.escape(name)}</th><td colspan="14">Pendiente</td></tr>'
                    continue
                m = v['metrics']
                rv = v['review']
                pair = pairs.get(f'{date}_{key}_{stride}')
                if native and date != 'VF':
                    pair = next((url for k, url in pairs.items() if k == f'{date}_{key}_{stride}'), None)
                else:
                    marker = HERE / 'published' / f'20261029_argentina_allblacks_{date}_m0_vs_m1_{key}_stride{stride}.json'
                    pair = read(marker)['url'] if _resolve_model_artifact(marker).exists() else None
                actions = f'<a href="{pair}">Video y omisiones</a>' if pair else 'ComparaciÃ³n pendiente' if date != 'VF' else 'Bloques VF'
                rows += f"<tr><th>{html.escape(name)}</th><td>{m['line']['tp']}/{m['line']['reference_segments']}</td><td>{pct(m['line']['recall'])}</td><td>{pct(m['line']['precision'])}</td><td>{m['scrum']['tp']}/{m['scrum']['reference_segments']}</td><td>{pct(m['scrum']['recall'])}</td><td>{pct(m['scrum']['precision'])}</td><td>{pct(m['total']['recall'])}</td><td>{pct(m['total']['precision'])}</td><td>{m['total']['alerts']}</td><td>{m['total']['fn']}</td><td>{pct(m['total']['f1'])}</td><td>{tm(rv['unique_flagged_s'])}</td><td>{tm(rv['flagged_outside_true_segments_s'])}</td><td>{actions}</td></tr>"
                miss = ', '.join((f"{t['play_type']} {tm(t['start_s'])}-{tm(t['end_s'])}" for t in v['missed_segments'])) or 'Ninguna'
                omissions += f'<tr><th>{html.escape(name)}</th><td>{miss}</td></tr>'
        sections += f'<section><h2>{label}</h2><div class="scroll"><table><thead><tr><th>Modelo y costos</th><th>Line</th><th>Sens. line</th><th>Prec. line</th><th>Scrum</th><th>Sens. scrum</th><th>Prec. scrum</th><th>Sens. total</th><th>Prec. total</th><th>Alertas</th><th>Omisiones</th><th>F1 total</th><th>RevisiÃ³n</th><th>Fuera de referencia</th><th>Ver</th></tr></thead><tbody>{rows}</tbody></table></div><details><summary>Jugadas omitidas</summary><div class="scroll"><table><tbody>{omissions}</tbody></table></div></details></section>'
        changes = ''
        for d in (d for d in differences if d['date'] == date and d['stride'] == stride):

            def listing(items):
                return ', '.join((t['play_type'] + ' ' + tm(t['start_s']) + '-' + tm(t['end_s']) for t in items)) or 'Ninguna'
            delta = d['m0_minus_m1_review_s']
            changes += f"<tr><th>{NAMES[d['key']]}</th><td>{listing(d['m0_recovers'])}</td><td>{listing(d['m0_loses'])}</td><td>{('+' if delta >= 0 else '-')}{tm(abs(delta))}</td></tr>"
        detail = f'<details><summary>QuÃ© recupera y quÃ© pierde M0 frente a M1 con los mismos costos</summary><div class="scroll"><table><thead><tr><th>Costos</th><th>M0 recupera</th><th>M0 pierde</th><th>RevisiÃ³n M0 menos M1</th></tr></thead><tbody>{changes}</tbody></table></div></details>'
        sections = sections[:-10] + detail + '</section>'
    status = f'{len(variants)}/70 configuraciones evaluadas. ' + ('Completo.' if all_complete else 'Prueba en curso.')
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M0 vs M1: costos y muestreo</title><style>body{font:16px/1.5 system-ui;color:#243b4b;max-width:1550px;margin:28px auto;padding:0 18px}h1,h2{color:#142d42}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;margin:16px 0}td,th{padding:9px;text-align:left;border-bottom:1px solid #d6e2e8;vertical-align:top}thead{background:#142d42;color:white}aside{background:#edf5f6;padding:16px;border-left:4px solid #087f8c}.pending{color:#7b8791}a,summary{color:#087f8c}summary{cursor:pointer}section{margin-top:36px}#progress{white-space:pre-wrap;font-size:14px}</style><h1>M0 frente a M1: pesos de clase y muestreo</h1><aside><strong id="state">' + status + '</strong><p id="progress"></p></aside><p>M0: RGB200 histÃ³rico reajustado H2-D2, sin YOLO ni bosque. M1: red RGB/pose y bosque geomÃ©trico. Mismos8.796ejemplos amateur, semilla42,5Ã©pocas, lote16 y Adam0,0001. Multiplicadores [line,other,scrum] por encima del balanceo por frecuencia, media empÃ­rica1. El controlÃ—1 de M0 se entrena con esa misma receta. Ã—2 de M0 reutiliza el checkpoint ya verificado. No se extraen caracterÃ­sticas nuevas.</p><p>16/08: las anotaciones manuales corregidas,32line+15scrum.23/08 y Mundial: inventarios diagnÃ³sticos automÃ¡ticos parciales; las alertas sin pareja no son falsos positivos confirmados y no certifican sensibilidad de todo el partido. VF: referencia humana por bloques,30line+10scrum, S01-S04,688bloques; sÃ³lo cachÃ© central cada25. Sin multicrop. No se mezclan con los M0 de seis pliegues de otros experimentos.</p><p>Coincidencia uno a uno, misma clase yâ‰¥1s de solapamiento. RevisiÃ³n en minutos:segundos, reproducciÃ³n1Ã— sin pausas, uniÃ³n temporal sin duplicados. Fuera de referencia incluye mÃ¡rgenes de eventos correctos. Cada5 y cada25 sÃ³lo reducen la frecuencia de observaciones en inferencia; no se reentrenÃ³ con esa frecuencia. Los tiempos desde cachÃ© y con procesos concurrentes no son benchmarks de video sin cachÃ©. Resultados exploratorios sobre videos ya observados.</p><p><a href="results.json">Datos y tiempos completos</a> Â· <a href="/static/reports/m0_cost2_professional_human_20261027/index.html">ComparaciÃ³n Ã—2 anterior</a></p>' + sections + "<script>async function refresh(){try{const r=await fetch('results.json?v='+Date.now());const d=await r.json();document.getElementById('state').textContent=d.completed_configurations+'/70 configuraciones evaluadas Â· '+d.status;const lr=await fetch('live_progress.json?v='+Date.now());const live=await lr.json();const p=live.training,e=live.evaluation;document.getElementById('progress').textContent='Entrenamiento: '+(p.key||'')+' Â· Ã©poca '+(p.epoch||'?')+'/'+(p.epochs||5)+' Â· lote '+(p.batch||'?')+'/'+(p.batches||550)+'\\nEvaluaciÃ³n: '+(e.date||'')+' '+(e.family||'')+' '+(e.key||'')+' Â· '+(e.windows||'?')+'/'+(e.total||'?')+' ventanas. Las tablas se actualizan al recargar.';}catch(e){}}refresh();setInterval(refresh,20000);</script></html>"
    for before, after in {'Mismos8.796ejemplos amateur, semilla42,5Ã©pocas, lote16 y Adam0,0001.': 'Mismos 8.796 ejemplos amateur, semilla 42, 5 Ã©pocas, lote de 16 y Adam con tasa 0,0001.', '[line,other,scrum]': '[line, otro, scrum]', 'media empÃ­rica1': 'media empÃ­rica 1', 'controlÃ—1': 'control Ã—1', 'corregidas,32line+15scrum.23/08': 'corregidas: 32 lines y 15 scrums. El 23/08', 'por bloques,30line+10scrum, S01-S04,688bloques; sÃ³lo cachÃ© central cada25.': 'por bloques: 30 lines y 10 scrums, en los 688 bloques de S01â€“S04; sÃ³lo cachÃ© central cada 25.', 'clase yâ‰¥1s': 'clase y al menos 1 s', 'reproducciÃ³n1Ã—': 'reproducciÃ³n a 1Ã—', 'Cada5 y cada25': 'Cada 5 y cada 25', '+d.status;': "+({running:'en curso',validating:'en verificaciÃ³n',complete:'completo',error:'error'}[d.status]||d.status);"}.items():
        doc = doc.replace(before, after)
    doc = doc.replace('const p=live.training,e=live.evaluation;', "const p=live.training,e=live.evaluation;if(d.status==='complete'){document.getElementById('progress').textContent='7/7 modelos preparados Â· 70/70 evaluaciones terminadas Â· controles numÃ©ricos y de video verificados.';return;}")
    if _resolve_model_artifact(HERE / 'requested_comparison.json').exists():
        links = read(HERE / 'requested_comparison.json')['links']
        requested = '<p><a href="' + html.escape(links['1']) + '">M0 line Ã—8 / scrum Ã—1 frente a M1 Ã—2: completo y omisiones</a> Â· <a href="' + html.escape(links['5']) + '">Las mismas variantes cada 5</a></p>'
        doc = doc.replace('<p><a href="results.json">', requested + '<p><a href="results.json">')
    (WEB / 'index.html').write_text(doc, encoding='utf-8')
    print(json.dumps(dict(status=result['status'], evaluated=len(variants), native_pairs=len(pairs))), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--native', action='store_true')
    a = p.parse_args()
    main(a.native)
