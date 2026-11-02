"""Verify frozen-cost comparisons and publish separate reports and saved viewers."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import json, hashlib, time, copy, urllib.request, html, subprocess, sys, importlib.metadata
import numpy as np
from models.M1.studies.m1_recall_cost_20261027.core import evaluate_events
from models.M1.studies.m1_recall_cost_20261027.review_coverage import review_coverage
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
WEB = ROOT / 'src/web/static/reports/m1_recall_cost_20261027'
BASEURL = 'http://127.0.0.1:5000'
COMPARISON_ID = '20261027_argentina_allblacks_20250816_m1_recall_costs'

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')

def get(path):
    with urllib.request.urlopen(BASEURL + path) as r:
        return json.load(r)

def independent_check(truth, pred, metrics, video_duration=6166.44):
    proof = {}
    for label in ('line', 'scrum'):
        ts = [t for t in truth if t['play_type'] == label]
        ps = [p for p in pred if p['play_type'] == label]
        used = {}
        edges = [[j for j, p in enumerate(ps) if min(t['end_s'], p['end_time']) - max(t['start_s'], p['start_time']) >= 1] for t in ts]

        def visit(i, seen):
            for j in edges[i]:
                if j in seen:
                    continue
                seen.add(j)
                if j not in used or visit(used[j], seen):
                    used[j] = i
                    return True
            return False
        for i in range(len(ts)):
            visit(i, set())
        assert len(used) == metrics[label]['tp']
        proof[label] = len(used)
    boundaries = sorted({0.0, video_duration} | {float(x) for t in truth for x in (t['start_s'], t['end_s'])} | {float(x) for p in pred for x in (p['start_time'], p['end_time'])})
    weights = np.diff(boundaries)
    grid = (np.asarray(boundaries[:-1]) + np.asarray(boundaries[1:])) / 2
    gt = np.zeros(len(grid), bool)
    flag = np.zeros(len(grid), bool)
    for t in truth:
        gt |= (grid >= t['start_s']) & (grid < t['end_s'])
    for p in pred:
        flag |= (grid >= p['start_time']) & (grid < p['end_time'])
    return dict(tp=proof, unique_flagged_s=float(weights[flag].sum()), flagged_outside_true_segments_s=float(weights[flag & ~gt].sum()))

def main():
    trained = read(HERE / 'training_complete.json')
    evaluated = read(HERE / 'evaluation_complete.json')
    protocol = read(HERE / 'protocol.json')
    assert trained['status'] == evaluated['status'] == 'complete' and len(evaluated['records']) == 6
    if not _resolve_model_artifact(HERE / 'vf25_results.json').exists():
        subprocess.run([sys.executable, '-B', '-u', str(_resolve_model_artifact(HERE / 'evaluate_vf25.py'))], cwd=ROOT, check=True)
    vf = read(HERE / 'vf25_results.json')
    assert vf['status'] == 'complete' and vf['extractors_executed'] == 0 and vf['baseline_all688_argmax_reproduced']
    assert len(vf['variants']) == 5
    for variant in vf['variants']:
        for p, h in variant.get('model_hashes', {}).items():
            assert sha(Path(p)) == h
    from models.M1.studies.m1_recall_cost_20261027.evaluate_vf25 import events as vf_events
    vfproof = {}
    for variant in vf['variants']:
        vr = variant.get('rows') or read(ROOT / 'data/vf_full_sampling_20261025/timelines.json')['rows']
        ve = vf_events(vr, 'predicted' if 'rows' in variant else 'original')
        vp = [dict(play_type=e['play_type'], start_time=e['start_s'], end_time=e['end_s']) for e in ve]
        vpcheck = independent_check(vf['reference_segments'], vp, variant['metrics'], vf['duration_s'])
        for key in ('unique_flagged_s', 'flagged_outside_true_segments_s'):
            assert abs(vpcheck[key] - variant['review'][key]) < 1e-06
        vfproof[variant['title']] = vpcheck
    assert max((r['saved_unix'] for r in trained['models'])) <= trained['all_models_saved_before_evaluation_unix']
    for p, h in protocol['source_hashes'].items():
        assert sha(Path(p)) == h
    for r in trained['models']:
        for p, h in r['model_hashes'].items():
            assert sha(Path(p)) == h
    for p, stat in read(HERE / 'cache_stats_before.json').items():
        assert [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == stat
    refdir = ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa'
    reference = read(refdir / 'metrics.json')
    truth = reference['reference_segments']
    labels = read(refdir / 'labels.json')['labels']
    for source in reference['provenance']:
        assert sha(Path(source['path'])) == source['sha256']
    pinned = dict(reference_session_pinned=True, composite_reference=True, reference_id=reference['revision_id'], reference_sha256=reference['labels_sha256'], session_id=reference['session_ids'][0], source_session_ids=reference['session_ids'], labels=labels, summary=dict(unit='blocks', reviewed=881, total_candidates=881, pending=0, final_counts=reference['block_counts']))
    records = {}
    summaries = []
    proofs = {}
    matches = {}
    for key, title in [('original', 'Original Â· completo'), ('every5', 'Original Â· cada 5')]:
        payload = read(ROOT / f"data/results/result_{reference['variants'][key]['result_id']}.json")
        payload['manual_comparison'] = copy.deepcopy(pinned)
        records[title] = payload
        value = evaluate_events(truth, payload['detections'])
        payload['human_evaluation'] = value['metrics']
        payload['metadata']['human_evaluation_report'] = '/static/reports/m1_recall_cost_20261027/index.html'
        value.update(key=key, title=title, cost=1, stride=1 if key == 'original' else 5, viewer_url='/viewer?result_id=' + payload['result_id'])
        summaries.append(value)
    for cost in (2, 4, 8):
        for stride in (1, 5):
            folder = HERE / 'evaluation' / f'cost_{cost}_stride_{stride}'
            metrics = read(folder / 'metrics.json')
            events = read(folder / 'events.json')
            windows = [json.loads(s) for s in _resolve_model_artifact(folder / 'windows.jsonl').read_text().splitlines()]
            assert len(windows) == 6160 and metrics['reference_sha256'] == reference['labels_sha256']
            assert max((trained_model['saved_unix'] for trained_model in trained['models'])) <= trained['all_models_saved_before_evaluation_unix']
            value = evaluate_events(truth, events)
            assert value['metrics'] == metrics['metrics']
            label = 'completo' if stride == 1 else 'cada 5'
            title = f'Costo Ã—{cost} Â· {label}'
            key = f'cost_{cost}_stride_{stride}'
            rid = f'20261027_argentina_allblacks_20250816_m1_cost{cost}_stride{stride}'
            baseline = records['Original Â· completo']
            stats = {}
            for cls in ('line', 'scrum'):
                alerts = [e for e in events if e['play_type'] == cls]
                stats[cls] = dict(count=len(alerts), total_duration=sum((e['end_time'] - e['start_time'] for e in alerts)))
            metadata = copy.deepcopy(baseline['metadata'])
            metadata.update(source_configuration=f'ALL-DEVELOPMENT-M1-H2-D2-S42-COST{cost}', experimental_sampling_stride=stride, experimental_interpolation='none' if stride == 1 else 'linear RGB/geometry on global native frame grid; nearest pose-valid mask', source_result_id=baseline['result_id'], saved_decisions_locked=True, human_evaluation_report='/static/reports/m1_recall_cost_20261027/index.html', training_class_cost=cost, decision_rule='raw argmax; no threshold changes', new_training=True, training_models=metrics['model_hashes'], reference_annotation='human corrected labels_fd96032b00fa', sampling_note=f'Both extractors stride{stride} replayed from native cache; no extractor executed.')
            payload = dict(status='success', result_id=rid, video_path=baseline['video_path'], model_name='M1 Â· ' + title, detections=events, primary_detections=copy.deepcopy(events), window_predictions=windows, event_post_processing=copy.deepcopy(baseline['event_post_processing']), statistics=dict(avg_confidence=sum((e['confidence'] for e in events)) / len(events) if events else 0, avg_duration=sum((e['end_time'] - e['start_time'] for e in events)) / len(events) if events else 0, by_type=stats), summary=dict(total_detections=len(events), by_type={k: v['count'] for k, v in stats.items()}), metadata=metadata, human_evaluation=value['metrics'], manual_comparison=copy.deepcopy(pinned))
            path = ROOT / f'data/results/result_{rid}.json'
            if _resolve_model_artifact(path).exists():
                assert read(path) == payload
            else:
                save(path, payload)
            records[title] = payload
            value.update(key=key, title=title, cost=cost, stride=stride, viewer_url='/viewer?result_id=' + rid, cached_evaluation_seconds=metrics['cached_evaluation_seconds'])
            summaries.append(value)
    for value in summaries:
        pred = records[value['title']]['detections']
        proof = independent_check(truth, pred, value['metrics'])
        value['review_coverage'] = review_coverage(truth, pred)
        assert all((proof[k] == value['review'][k] for k in ('unique_flagged_s', 'flagged_outside_true_segments_s')))
        used_predictions = set()
        used_references = set()
        for pair in value['assignments']:
            assert pair['prediction_index'] not in used_predictions and pair['reference_id'] not in used_references
            used_predictions.add(pair['prediction_index'])
            used_references.add(pair['reference_id'])
        proofs[value['key']] = proof
        matches[value['key']] = used_references
    models = [dict(key=v['key'], title=v['title'], missed=v['metrics']['total']['fn'], prediction_signature=[[p['play_type'], p['start_time'], p['end_time']] for p in records[v['title']]['detections']]) for v in summaries]
    omissions = []
    for t in truth:
        detected = {v['key']: t['id'] in matches[v['key']] for v in summaries}
        if not all(detected.values()):
            omissions.append(dict(**t, detected=detected))
    records['Original Â· completo']['omission_review'] = dict(reference_id=reference['revision_id'], reference_sha256=reference['labels_sha256'], models=models, segments=omissions, context_s=3, method='same-class one-to-one overlap>=1s')
    history = dict(id=COMPARISON_ID, video_path=records['Original Â· completo']['video_path'], video_name='Argentinaâ€“NZ Â· 16/08/2025 Â· M1 costos de sensibilidad', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={title: dict(total=len(r['detections']), by_type={k: v['count'] for k, v in r['statistics']['by_type'].items()}, avg_confidence=r['statistics']['avg_confidence']) for title, r in records.items()})
    progress_temp = HERE / 'progress.publish.tmp'
    save(progress_temp, dict(stage='publishing_verified_results', updated_unix=time.time(), configurations=8, vf_configurations=5))
    progress_temp.replace(HERE / 'progress.json')
    request = urllib.request.Request(BASEURL + '/api/comparison-history', data=json.dumps(history, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request) as response:
        assert json.load(response)['success']
    saved = get('/api/comparison-history/' + COMPARISON_ID)['comparison']
    assert saved['results'] == records
    attempts = {}
    for name in ('training_attempt_001.json', 'training_attempt_002.json'):
        path = HERE / name
        if _resolve_model_artifact(path).exists():
            attempt = read(path)
            attempts[name] = attempt.get('progress', attempt).get('elapsed_s')
    versions = {'python': sys.version}
    for package in ('numpy', 'keras', 'tensorflow', 'scikit-learn', 'joblib'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not found by distribution name'
    result = dict(status='complete', match='Argentina 24â€“41 Nueva Zelanda Â· 16/08/2025', reference_id=reference['revision_id'], reference_sha256=reference['labels_sha256'], reference_counts=dict(line=32, scrum=15, total=47), duration_s=6166.44, protocol=protocol, runtime_versions=versions, variants=summaries, training=trained, aborted_attempt_elapsed_s=attempts, common_criterion='same-class one-to-one, overlap>=1s, maximum cardinality then overlap', vf_status='Cost variants evaluated at every25 from existing complete native cache; full and every5 new models remain pending', vf25=vf, extractors_executed=0, comparison_url='/compare?comparison=' + COMPARISON_ID + '#omisiones', source_models_and_caches_unchanged=True, independent_test_claim_allowed=False)
    WEB.mkdir(parents=True, exist_ok=True)
    save(HERE / 'results.json', result)
    save(WEB / 'results.json', result)
    esc = html.escape

    def pct(v):
        return f'{v * 100:.2f}%'.replace('.', ',')

    def tm(v):
        return f'{int(round(v)) // 60}:{int(round(v)) % 60:02d}'
    rows = ''
    for v in summaries:
        m = v['metrics']['total']
        review = v['review']
        rows += f"<tr><th>{esc(v['title'])}</th><td>{m['tp']}/47</td><td>{m['alerts']}</td><td>{m['fp']}</td><td>{m['fn']}</td><td>{pct(m['precision'])}</td><td>{pct(m['recall'])}</td><td>{pct(m['f1'])}</td><td>{v['review_coverage']['any_alert_segments']}/47</td><td>{tm(review['unique_flagged_s'])}</td><td>{tm(review['flagged_outside_true_segments_s'])}</td><td><a href='{esc(v['viewer_url'])}'>Ver alertas</a></td></tr>"
    byclass = ''
    for v in summaries:
        for cls in ('line', 'scrum'):
            m = v['metrics'][cls]
            byclass += f"<tr><th>{esc(v['title'])}</th><td>{cls}</td><td>{m['tp']}/{m['reference_segments']}</td><td>{m['fp']}</td><td>{m['fn']}</td><td>{pct(m['precision'])}</td><td>{pct(m['recall'])}</td><td>{pct(m['f1'])}</td></tr>"
    misses = ''
    for v in summaries:
        covered = set(v['review_coverage']['any_alert_ids'])
        text = ', '.join((f"{t['play_type']} {tm(t['start_s'])}â€“{tm(t['end_s'])}" + (' (incluido en una alerta, sin pareja)' if t['id'] in covered else ' (fuera de alertas)') for t in v['missed_segments'])) or 'Ninguno con esta referencia'
        misses += f"<tr><th>{esc(v['title'])}</th><td>{esc(text)}</td></tr>"
    timing_rows = ''.join((f"<tr><th>Ã—{v['cost']}</th><td>{tm(v['forest_s'])}</td><td>{tm(v['dual_s'])}</td><td>{tm(v['forest_s'] + v['dual_s'])}</td></tr>" for v in trained['models']))
    timing_section = '<h2>Tiempos medidos de entrenamiento</h2><div class="scroll"><table><thead><tr><th>Costo</th><th>Bosque</th><th>Red Â· cinco Ã©pocas</th><th>Suma</th></tr></thead><tbody>' + timing_rows + '</tbody></table></div><p>La preparaciÃ³n de entradas desde cachÃ©, importaciones y publicaciÃ³n son costos adicionales. Los intentos descartados se conservan por separado en los datos completos. Los tiempos de evaluaciÃ³n de cada variante son desde cachÃ©, sin extracciÃ³n de video.</p>'
    vfrows = ''
    for v in vf['variants']:
        m = v['metrics']['total']
        line = v['metrics']['line']
        scrum = v['metrics']['scrum']
        vfrows += f"<tr><th>{esc(v['title'])}</th><td>{line['tp']}/30 ({pct(line['recall'])})</td><td>{scrum['tp']}/10 ({pct(scrum['recall'])})</td><td>{pct(m['precision'])}</td><td>{pct(m['recall'])}</td><td>{pct(m['f1'])}</td><td>{m['fp']}</td><td>{tm(v['review']['unique_flagged_s'])}</td></tr>"
    vf_section = '<h2>VF amateur: costos evaluados con el cachÃ© cada25</h2><p>Jockey Club VTâ€“Alma Juniors, S01â€“S04:80:04,8,688 bloques y40 segmentos positivos (30line/10scrum). Se reutilizÃ³ el cachÃ© nativo completo cada25, sin extractores ni video. El original cada25 se reprodujo sobre los688 bloques con las mismas decisiones antes de evaluar los costos. Misma coincidencia de clase, solapamiento mÃ­nimo1s y uno a uno; sin tolerancia de35s. Cada5 y la versiÃ³n completa de los modelos nuevos en VF siguen pendientes por falta de entradas nativas. No se mezclan esas variantes con cada25.</p><div class="scroll"><table data-table="vf-cost-total"><thead><tr><th>Variante</th><th>Line encontrados</th><th>Scrum encontrados</th><th>PrecisiÃ³n total</th><th>Sensibilidad total</th><th>F1 total</th><th>FP</th><th>RevisiÃ³n Ãºnica</th></tr></thead><tbody>' + vfrows + '</tbody></table></div>'
    document = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 Â· costo de omitir jugadas</title><style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:30px auto;padding:0 18px;color:#243447}h1{font-size:1.8rem}table{width:100%;border-collapse:collapse;margin:16px 0}td,th{text-align:left;padding:10px;border-bottom:1px solid #dbe2e9}thead{background:#edf3f8}.scroll{overflow:auto}aside{padding:16px;background:#edf3f8}a{color:#126b84}code{overflow-wrap:anywhere}</style><h1>M1: costo de omitir line y scrum</h1><p>Argentina24â€“41 Nueva Zelanda Â· 16/08/2025. Tres entrenamientos nuevos, costos relativosÃ—2/Ã—4/Ã—8, comparados con el original. Prioridad elegida: sensibilidad, mostrando toda la pÃ©rdida de precisiÃ³n y revisiÃ³n adicional.</p><aside>Referencia corregida:32 segmentos line y15 scrum. Misma clase, solapamiento mÃ­nimo1s, emparejamiento uno a uno. Pesos de entrenamiento normalizados a media1; mismo M1-H2-D2,8796 ejemplos, semilla42,5Ã©pocas y fusiÃ³n0.475. Se entrenaron red y bosque. Inferencia argmax, sin umbrales nuevos ni entrenamiento con las etiquetas del partido.</aside><p><a href="' + result['comparison_url'] + '">Abrir comparaciÃ³n con video, timelines y segmentos omitidos</a> Â· <a href="results.json">Datos completos y trazabilidad</a></p><h2>Resultado total y carga de revisiÃ³n</h2><p>Tiempos de reproducciÃ³n a1Ã—, sin pausas ni navegaciÃ³n. La revisiÃ³n Ãºnica elimina solapamientos entre alertas. Â«ExtraÂ» es tiempo marcado fuera de cualquier segmento positivo humano; incluye mÃ¡rgenes de alertas correctas. No es idÃ©ntico al nÃºmero de falsos positivos. Incluidos en revisiÃ³n cuenta segmentos tocados por cualquier alerta durante al menos1s, aunque tenga otra clase; se distingue de la sensibilidad uno a uno. Una alerta larga puede incluir mÃ¡s de un segmento y obtener sÃ³lo una pareja.</p><div class="scroll"><table data-table="cost-total"><thead><tr><th>Variante</th><th>Encontrados</th><th>Alertas</th><th>FP</th><th>Sin pareja</th><th>PrecisiÃ³n</th><th>Sensibilidad</th><th>F1</th><th>Incluidos en revisiÃ³n</th><th>RevisiÃ³n Ãºnica</th><th>Extra</th><th>Video</th></tr></thead><tbody>' + rows + '</tbody></table></div><h2>Line y scrum por separado</h2><div class="scroll"><table data-table="cost-classes"><thead><tr><th>Variante</th><th>Clase</th><th>Encontrados</th><th>FP</th><th>Sin pareja</th><th>PrecisiÃ³n</th><th>Sensibilidad</th><th>F1</th></tr></thead><tbody>' + byclass + '</tbody></table></div><h2>Segmentos omitidos</h2><p>Los tiempos corresponden a la transmisiÃ³n. Pueden incluir repeticiones o rearmados.</p><div class="scroll"><table><thead><tr><th>Variante</th><th>Omisiones</th></tr></thead><tbody>' + misses + '</tbody></table></div>' + timing_section + vf_section + '<h2>Alcance</h2><p>Las tres variantes se fijaron antes de evaluar y se publican completas; no se elige una usando este partido. Ã‰poca final5, sin ajuste de umbrales. Extractores ejecutados:0. Los tiempos registrados de evaluaciÃ³n son desde cachÃ©, no procesamiento nuevo de video. En VF se evalÃºa cada25 desde el cachÃ© existente; cada5/completo nuevos quedan pendientes y no se extrajo nada nuevo. El partido ya fue observado y la procedencia de parte del entrenamiento histÃ³rico no estÃ¡ completamente certificada: resultados exploratorios, sin afirmar validaciÃ³n externa independiente ni sensibilidad garantizada en otro partido.</p></html>'
    conclusion = '<aside><strong>Lectura de esta prueba.</strong> En el partido profesional, costo Ã—2 completo recupera 46/47 segmentos, frente a 42/47 del original, y aumenta la revisiÃ³n de 18:15 a 23:20 (+5:05). Encuentra 31/32 line y 15/15 scrum. Ã—4 y Ã—8 completos no mejoran esa sensibilidad y generan mÃ¡s falsas alarmas. El line de 82:22â€“82:43 queda fuera de las alertas de Ã—2; Ã—4 sÃ³lo toca sus Ãºltimos 2 segundos mediante una alerta de scrum. Que 47/47 estÃ©n incluidos por algÃºn solapamiento no significa que estÃ©n completos ni bien clasificados. En VF cada 25, ningÃºn costo mejora la sensibilidad del original cada 25: Ã—2 obtiene 70%, Ã—4 65% y Ã—8 77,5%, frente a 77,5%. Estos resultados no demuestran una mejora general en videos nuevos.</aside>'
    document = document.replace('<h2>Resultado total y carga de revisiÃ³n</h2>', conclusion + '<h2>Resultado total y carga de revisiÃ³n</h2>')
    for before, after in {'Argentina24': 'Argentina 24', 'relativosÃ—': 'relativos Ã—', 'corregida:32': 'corregida: 32', 'line y15': 'line y 15', 'mÃ­nimo1s': 'mÃ­nimo 1 s', 'media1;': 'media 1;', 'M1-H2-D2,8796': 'M1-H2-D2, 8796', 'semilla42,5Ã©pocas': 'semilla 42, 5 Ã©pocas', 'fusiÃ³n0.475': 'fusiÃ³n 0,475', 'a1Ã—': 'a 1Ã—', 'menos1s': 'menos 1 s', 'cada25': 'cada 25', 'S01â€“S04:80:04,8,688': 'S01â€“S04: 80:04,8, 688', 'y40': 'y 40', '30line/10scrum': '30 line/10 scrum', 'los688': 'los 688', 'de35s': 'de 35 s', 'Cada5': 'Cada 5', 'cada5': 'cada 5', 'final5': 'final 5', 'ejecutados:0': 'ejecutados: 0'}.items():
        document = document.replace(before, after)
    for path in (HERE / 'report.html', WEB / 'index.html'):
        path.write_text(document, encoding='utf-8')
    proof = dict(status='PASS', independent_maximum_matching_and_interval_masks=proofs, source_models_and_caches_unchanged=True, comparison_history_roundtrip=True, training_checkpoints_saved_before_evaluation=True, publisher_sha256=sha(Path(__file__)), vf25_baseline_all688_reproduced=True, vf25_independent_checks=vfproof, vf25_evaluator_sha256=sha(_resolve_model_artifact(HERE / 'evaluate_vf25.py')), report_url=BASEURL + '/static/reports/m1_recall_cost_20261027/index.html')
    save(HERE / 'verification.json', proof)
    save(WEB / 'verification.json', proof)
    print(json.dumps(dict(status='published', report_url=proof['report_url'], totals={v['title']: v['metrics']['total'] for v in summaries})), flush=True)
if __name__ == '__main__':
    main()
