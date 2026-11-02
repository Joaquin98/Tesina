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
import sys, json, copy, importlib.util, html, subprocess
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(ROOT / 'data/m1_recall_cost_20261027')), str(ROOT)]
spec = importlib.util.spec_from_file_location('prior_other_pub', _resolve_model_artifact(ROOT / 'models/common/studies/m1_cost2_other_professionals_20261027/publish.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
WEB = ROOT / 'src/web/static/reports/m2_professional_human_20261027'
CID = '20261027_argentina_allblacks_20250816_human_m1_cost2_m2'

def main():
    c = p.read(HERE / 'completion.json')
    assert c['status'] == 'complete' and c['windows'] == 6160 and c['sources_unchanged']
    for path, h in c['source_hashes'].items():
        assert p.sha(Path(path)) == h
    ref = p.read(ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json')
    truth = ref['reference_segments']
    old = p.get('/api/comparison-history/20261027_argentina_allblacks_20250816_m1_recall_costs')['comparison']['results']
    original = copy.deepcopy(old['Original · completo'])
    cost = copy.deepcopy(old['Costo ×2 · completo'])
    d = copy.deepcopy(original)
    events = p.read(HERE / 'events.json')
    rid = '20261027_argentina_allblacks_20250816_m2_original_human'
    d.update(result_id=rid, model_name='M2 original · highres quantile100', detections=events, primary_detections=copy.deepcopy(events), window_predictions=[json.loads(x) for x in _resolve_model_artifact(HERE / 'windows.jsonl').read_text(encoding='utf-8').splitlines()], human_evaluation=c['metrics']['metrics'])
    d.pop('omission_review', None)
    d['metadata'].update(source_configuration='M2-B-highres-quantile100', new_training=False, source_model_hashes=c['source_hashes'], decision_rule='raw argmax; same temporal policy as M1 comparison', saved_decisions_locked=True, experiment_report_url='/static/reports/m2_professional_human_20261027/index.html', resnet_executed=False, new_yolo_observations=c['new_yolo_observations'])
    d['metadata'].update(human_evaluation_report='/static/reports/m2_professional_human_20261027/index.html', protocol='M2-B highres quantile100; global YOLO cadence100/7Hz; cached full RGB', training_regime='Original saved B deployment; auxiliary fit4082 historical examples, not M1-H2-D2 retraining', sampling_note='RGB full25FPS; original M2 YOLO global cadence100/7Hz. Both classifiers use common argmax event policy.')
    by = {k: dict(count=sum((e['play_type'] == k for e in events)), total_duration=sum((e['end_time'] - e['start_time'] for e in events if e['play_type'] == k))) for k in ('line', 'scrum')}
    d['statistics'] = dict(by_type=by, avg_confidence=sum((e['confidence'] for e in events)) / len(events) if events else 0, avg_duration=sum((e['end_time'] - e['start_time'] for e in events)) / len(events) if events else 0)
    d['summary'] = dict(total_detections=len(events), by_type={k: v['count'] for k, v in by.items()})
    p.save(ROOT / f'data/results/result_{rid}.json', d)
    records = {'M1 original': original, 'M1 costo ×2': cost, 'M2 original': d}
    summaries = []
    proofs = {}
    matches = {}
    for name, r in records.items():
        v = p.evaluate_events(truth, r['detections'])
        proof = p.prior.independent_check(truth, r['detections'], v['metrics'])
        proofs[name] = proof
        for k in ('unique_flagged_s', 'flagged_outside_true_segments_s'):
            assert abs(proof[k] - v['review'][k]) < 1e-06
        matches[name] = {a['reference_id'] for a in v['assignments']}
        v.update(title=name, viewer_url='/viewer?result_id=' + r['result_id'])
        summaries.append(v)
    keys = {name: 'model_' + str(i) for i, name in enumerate(records)}
    models = [dict(key=keys[name], title=name, missed=v['metrics']['total']['fn'], prediction_signature=[[e['play_type'], e['start_time'], e['end_time']] for e in records[name]['detections']]) for name, v in zip(records, summaries)]
    omissions = [dict(**t, detected={keys[name]: t['id'] in matches[name] for name in records}) for t in truth if not all((t['id'] in matches[name] for name in records))]
    original['omission_review'] = dict(reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], models=models, segments=omissions, context_s=3, method='same-class one-to-one overlap>=1s')
    history = dict(id=CID, video_path=d['video_path'], video_name='Argentina–NZ16/08/2025 · M1 original / ×2 / M2 · etiquetas humanas', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={name: dict(total=len(r['detections']), by_type={k: sum((e['play_type'] == k for e in r['detections'])) for k in ('line', 'scrum')}, avg_confidence=r['statistics']['avg_confidence']) for name, r in records.items()})
    assert p.post('/api/comparison-history', history)['success']
    assert p.get('/api/comparison-history/' + CID)['comparison']['results'] == records
    result = dict(status='complete', variants=summaries, comparison_url='/compare?comparison=' + CID + '#omisiones', reference_id=ref['revision_id'], completion=c)
    p.save(HERE / 'results.json', result)
    p.save(WEB / 'results.json', result)
    rows = ''
    for v in summaries:
        m = v['metrics']
        t = m['total']
        rev = v['review']
        rows += f'''<tr><th>{v['title']}</th><td>{m['line']['tp']}/32</td><td>{m['scrum']['tp']}/15</td><td>{p.pct(t['recall'])}</td><td>{p.pct(t['precision'])}</td><td>{t['alerts']}</td><td>{t['fp']}</td><td>{t['fn']}</td><td>{p.tm(rev['unique_flagged_s'])}</td><td>{p.tm(rev['flagged_outside_true_segments_s'])}</td><td><a href="{v['viewer_url']}">Ver</a></td></tr>'''
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M2 · etiquetas humanas</title><style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:28px auto;padding:0 18px;color:#243447}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}thead,aside{background:#edf3f8}aside{padding:16px}a{color:#126b84}</style><h1>M2 original frente a M1 · Argentina–NZ16/08/2025</h1><aside>Referencia: las anotaciones manuales corregidas fd96032b00fa,32 line y15 scrum. Misma clase, solapamiento mínimo1s, una pareja por jugada. Modelos congelados y decisiones argmax; misma política temporal M1 para aislar la clasificación, no los umbrales/agrupación por defecto de M2. Tiempo de revisión a1×, sin pausas, uniendo solapamientos. Extra: tiempo marcado fuera de segmentos positivos, también incluye márgenes de alertas correctas. Estos resultados son descriptivos, sin certificación de independencia externa.</aside><p><a href="' + result['comparison_url'] + '">Abrir video, timelines y jugadas omitidas</a> · <a href="results.json">Datos completos</a></p><div class="scroll"><table><thead><tr><th>Modelo</th><th>Line</th><th>Scrum</th><th>Sensibilidad</th><th>Precisión</th><th>Alertas</th><th>FP</th><th>FN</th><th>Revisión única</th><th>Extra</th><th>Video</th></tr></thead><tbody>' + rows + '</tbody></table></div><p>M2 original guardado: ResNet50+BiGRU histórico, dinámica RGB704 y descriptores YOLO E04/E06 de982 valores, cuantiles y logística; combinación aritmética0,4/0,6. No se entrenó. Se reutilizó ResNet completo; se calcularon88.092 observaciones YOLO nuevas con el detector original y rejilla oficial de100/7Hz, ROI nativa640, sin cambiar pesos.</p></html>'
    conclusion = '<p><strong>Resultado práctico:</strong> M2 encuentra 47/47 segmentos, pero marca 88:46 (86,4% de la transmisión). M1 ×2 encuentra 46/47 y marca 23:20: M2 agrega 65:26 de revisión para recuperar un segmento adicional. El 100% corresponde a este partido y al criterio de al menos 1 segundo por segmento; no garantiza captura completa ni resultados iguales en otros videos.</p>'
    doc = doc.replace('<div class="scroll">', conclusion + '<div class="scroll">', 1)
    doc = doc.replace('</html>', '<p>M2 es el deployment original B, con rama auxiliar ajustada sobre 4.082 ejemplos históricos. M1 ×2 se entrenó sobre 8.796 ejemplos D2. Esta comparación utiliza el mismo criterio de evaluación y agrupación, pero no aísla exclusivamente la arquitectura ni reproduce la política de eventos predeterminada de M2.</p></html>')
    for a, b in {'NZ16': 'NZ 16', 'fd96032b00fa,32': 'fd96032b00fa, 32', 'line y15': 'line y 15', 'mínimo1s': 'mínimo 1 s', 'a1×': 'a 1×', 'RGB704': 'RGB 704', 'de982': 'de 982', 'aritmética0,4/0,6': 'aritmética 0,4/0,6', 'calcularon88.092': 'calcularon 88.092', 'de100/7Hz': 'de 100/7 Hz', 'nativa640': 'nativa 640'}.items():
        doc = doc.replace(a, b)
    (WEB / 'index.html').write_text(doc, encoding='utf-8')
    p.save(WEB / 'verification.json', dict(status='PASS', independent_checks=proofs, official_m2_control=p.read(HERE / 'baseline_validation.json'), source_hashes_unchanged=True, comparison_roundtrip=True))
    subprocess.run(['node', str(HERE / 'verify_web.cjs')], cwd=ROOT, check=True)
    print(json.dumps({'status': 'published_and_web_verified', 'metrics': c['metrics']['metrics']['total']}))
if __name__ == '__main__':
    main()
