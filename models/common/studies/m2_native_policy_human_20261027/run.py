"""Replay cached M2 probabilities with original class event policy; no inference."""
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
import sys, json, copy, time, importlib.util, html
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
spec = importlib.util.spec_from_file_location('other_pub', _resolve_model_artifact(ROOT / 'models/common/studies/m1_cost2_other_professionals_20261027/publish.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from models.M2.predictor import RGBYoloHighresQuantileFusionDetector as M2
from review_coverage import review_coverage
WEB = ROOT / 'src/web/static/reports/m2_native_policy_human_20261027'
CID = '20261027_argentina_allblacks_20250816_human_m1x2_m2_native_policy'

def main():
    tick = time.perf_counter()
    source = ROOT / 'data/m2_professional_human_20261027'
    refpath = ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json'
    ref = p.read(refpath)
    truth = ref['reference_segments']
    cached = source / 'windows.jsonl'
    oldid = '20261027_argentina_allblacks_20250816_m2_original_human'
    oldpath = ROOT / f'data/results/result_{oldid}.json'
    old = p.read(oldpath)
    paths = [cached, oldpath, refpath, _resolve_model_artifact(ROOT / 'models/M2/predictor.py'), _resolve_model_artifact(ROOT / 'models/M0/web_adapter.py'), Path(__file__)]
    hashes = {str(f): p.sha(f) for f in paths}
    for s in ref['provenance']:
        assert p.sha(Path(s['path'])) == s['sha256']
    rows = [json.loads(x) for x in _resolve_model_artifact(cached).read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 6160
    d = M2.__new__(M2)
    policy = json.loads(json.dumps({k: getattr(d, k) for k in dir(d) if k.startswith('EVENT_')}))
    assert policy['EVENT_MIN_TARGET_CONFIDENCE'] == 0.6 and policy['EVENT_MAX_DURATION_SECONDS'] == 15
    events = [e for e in d._post_process_event_detections(rows, 0.0, 6166.44, 25, {'line': 0.0}) if e['end_time'] > 188.2]
    assert all((e['duration'] <= 15.000001 for e in events))
    control = M2.__new__(M2)
    for k, v in old['event_post_processing'].items():
        setattr(control, k, v)
    replay = [e for e in control._post_process_event_detections(rows, 0.0, 6166.44, 25, {'line': 0.0}) if e['end_time'] > 188.2]
    assert replay == old['detections'], 'Cached prior grouping changed'
    rid = '20261027_argentina_allblacks_20250816_m2_native_policy_human'
    native = copy.deepcopy(old)
    native.update(result_id=rid, model_name='M2 · política original', detections=events, primary_detections=copy.deepcopy(events), event_post_processing=policy)
    native.pop('omission_review', None)
    native['metadata'].update(event_post_processing=policy, decision_rule='raw cached argmax plus original M2 EVENT_* defaults; admission floor0.6; zero additional UI thresholds', human_evaluation_report='/static/reports/m2_native_policy_human_20261027/index.html', experiment_report_url='/static/reports/m2_native_policy_human_20261027/index.html', saved_decisions_locked=True, source_result_id=oldid, new_yolo_observations=0, resnet_executed=False, new_training=False)
    by = {k: dict(count=sum((e['play_type'] == k for e in events)), total_duration=sum((e['duration'] for e in events if e['play_type'] == k))) for k in ('line', 'scrum')}
    native['summary'] = dict(total_detections=len(events), by_type={k: v['count'] for k, v in by.items()})
    native['statistics'] = dict(by_type=by, avg_confidence=sum((e['confidence'] for e in events)) / len(events) if events else 0, avg_duration=sum((e['duration'] for e in events)) / len(events) if events else 0)
    native['human_evaluation'] = p.evaluate_events(truth, events)['metrics']
    p.save(ROOT / f'data/results/result_{rid}.json', native)
    history = p.get('/api/comparison-history/20261027_argentina_allblacks_20250816_m1_recall_costs')['comparison']['results']
    records = {'M1 costo ×2': copy.deepcopy(history['Costo ×2 · completo']), 'M2 · política original': native, 'M2 · agrupación común anterior': copy.deepcopy(old)}
    variants = []
    proofs = {}
    matches = {}
    for name, r in records.items():
        v = p.evaluate_events(truth, r['detections'])
        proof = p.prior.independent_check(truth, r['detections'], v['metrics'])
        for k in ('unique_flagged_s', 'flagged_outside_true_segments_s'):
            assert abs(proof[k] - v['review'][k]) < 1e-06
        proofs[name] = proof
        matches[name] = {a['reference_id'] for a in v['assignments']}
        v.update(title=name, key='model_' + str(len(variants)), viewer_url='/viewer?result_id=' + r['result_id'], review_coverage=review_coverage(truth, r['detections']))
        variants.append(v)
    models = [dict(key=v['key'], title=v['title'], missed=v['metrics']['total']['fn'], prediction_signature=[[e['play_type'], e['start_time'], e['end_time']] for e in records[v['title']]['detections']]) for v in variants]
    omissions = [dict(**t, detected={v['key']: t['id'] in matches[v['title']] for v in variants}) for t in truth if not all((t['id'] in matches[n] for n in records))]
    first = next(iter(records.values()))
    first['omission_review'] = dict(reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], models=models, segments=omissions, context_s=3, method='same-class one-to-one overlap>=1s')
    h = dict(id=CID, video_path=native['video_path'], video_name='Argentina–NZ16/08/2025 · M1×2 vs M2 con política original', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={name: dict(total=len(r['detections']), by_type={k: sum((e['play_type'] == k for e in r['detections'])) for k in ('line', 'scrum')}, avg_confidence=r['statistics']['avg_confidence']) for name, r in records.items()})
    assert p.post('/api/comparison-history', h)['success']
    assert p.get('/api/comparison-history/' + CID)['comparison']['results'] == records
    assert hashes == {f: p.sha(Path(f)) for f in hashes}
    for s in ref['provenance']:
        assert p.sha(Path(s['path'])) == s['sha256']
    result = dict(status='complete', variants=variants, reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], policy=policy, comparison_url='/compare?comparison=' + CID + '#omisiones', cached_windows=6160, extractors_executed=0, new_training=False, cached_probabilities_unchanged=True, prior_events_reproduced=True, source_hashes=hashes, elapsed_s=time.perf_counter() - tick)
    p.save(HERE / 'results.json', result)
    p.save(WEB / 'results.json', result)
    p.save(HERE / 'events.json', events)
    table = ''
    classes = ''
    misses = ''
    for v in variants:
        m = v['metrics']
        t = m['total']
        rev = v['review']
        table += f"<tr><th>{v['title']}</th><td>{m['line']['tp']}/32</td><td>{m['scrum']['tp']}/15</td><td>{p.pct(t['recall'])}</td><td>{p.pct(t['precision'])}</td><td>{t['alerts']}</td><td>{t['fp']}</td><td>{t['fn']}</td><td>{p.tm(rev['unique_flagged_s'])}</td><td>{p.tm(rev['flagged_outside_true_segments_s'])}</td></tr>"
        for label in ('line', 'scrum'):
            x = m[label]
            classes += f"<tr><th>{v['title']}</th><td>{label}</td><td>{x['tp']}/{x['reference_segments']}</td><td>{p.pct(x['recall'])}</td><td>{p.pct(x['precision'])}</td><td>{p.pct(x['f1'])}</td></tr>"
        misses += f"<tr><th>{v['title']}</th><td>" + (', '.join((f"{t['play_type']} {p.tm(t['start_s'])}–{p.tm(t['end_s'])}" for t in v['missed_segments'])) or 'Ninguno') + '</td></tr>'
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M2 · política original</title><style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:28px auto;padding:0 18px;color:#243447}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}thead,aside{background:#edf3f8}aside{padding:16px}a{color:#126b84}</style><h1>M2 con su política original frente a M1 ×2</h1><p>Argentina–NZ · 16/08/2025 · las anotaciones manuales corregidas: 32 line y 15 scrum. Mismas 6.160 probabilidades M2 guardadas, sin ejecutar extractores, clasificadores ni entrenar.</p><aside>Política M2 original leída del adaptador: clase objetivo ganadora; confianza mínima 0,60; agrupación por inicios separados como máximo 1,5 s; al menos dos ventanas de soporte, o una ventana con confianza ≥0,85; duración máxima 15 s; sin promoción por vecinos. No se aplicaron umbrales de interfaz adicionales ni se buscaron parámetros mirando las etiquetas. La tercera fila conserva la prueba anterior con agrupación común M1 como contexto.</aside><p><a href="' + result['comparison_url'] + '">Abrir video, timelines y omisiones</a> · <a href="results.json">Datos completos y política</a></p><div class="scroll"><table data-table="total"><thead><tr><th>Modelo</th><th>Line</th><th>Scrum</th><th>Sensibilidad</th><th>Precisión</th><th>Alertas</th><th>FP</th><th>FN</th><th>Revisión única</th><th>Extra</th></tr></thead><tbody>' + table + '</tbody></table></div><p>Revisión en minutos:segundos, a velocidad 1×, sin pausas ni navegación, uniendo intervalos solapados. Extra: tiempo marcado fuera de cualquier segmento positivo humano, incluyendo márgenes de alertas correctas. Evaluación por clase, solapamiento ≥1 s y una pareja por segmento; no garantiza capturar toda su duración ni generalización. Los regímenes de entrenamiento M1 y M2 difieren.</p><h2>Por clase</h2><div class="scroll"><table><thead><tr><th>Modelo</th><th>Clase</th><th>Encontrados</th><th>Sensibilidad</th><th>Precisión</th><th>F1</th></tr></thead><tbody>' + classes + '</tbody></table></div><h2>Segmentos sin pareja correcta</h2><div class="scroll"><table><thead><tr><th>Modelo</th><th>Tiempos de transmisión</th></tr></thead><tbody>' + misses + '</tbody></table></div></html>'
    (WEB / 'index.html').write_text(doc, encoding='utf-8')
    proof = dict(status='PASS', independent_checks=proofs, source_hashes_unchanged=True, prior_grouping_exactly_reproduced=True, native_policy=policy)
    p.save(HERE / 'verification.json', proof)
    p.save(WEB / 'verification.json', proof)
    print(json.dumps(dict(status='published', policy=policy, variants=[dict(title=v['title'], metrics=v['metrics'], review=v['review']) for v in variants])))
if __name__ == '__main__':
    main()
