import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import sys, copy, json, time, importlib.util, subprocess
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
spec = importlib.util.spec_from_file_location('helpers', _resolve_model_artifact(ROOT / 'models/common/studies/m1_cost2_other_professionals_20261027/publish.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
WEB = ROOT / 'src/web/static/reports/m0_cost2_professional_human_20261027'
CID = '20261027_argentina_allblacks_20250816_human_m0x2_vs_m1x2_full_every5'

def main():
    c = p.read(HERE / 'completion.json')
    assert c['status'] == 'complete' and len(c['records']) == 2 and (c['extractors_executed'] == 0)
    for path, h in c['source_hashes'].items():
        assert p.sha(Path(path)) == h
    model = HERE / 'models/m0_cost2.keras'
    assert p.sha(model) == c['training']['model_sha256']
    ref = p.read(ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json')
    truth = ref['reference_segments']
    for s in ref['provenance']:
        assert p.sha(Path(s['path'])) == s['sha256']
    old = p.get('/api/comparison-history/20261027_argentina_allblacks_20250816_m1_recall_costs')['comparison']['results']
    records = {}
    for s in (1, 5):
        suffix = 'completo' if s == 1 else 'cada 5'
        source = copy.deepcopy(old['Costo Ã—2 Â· ' + suffix])
        source.pop('omission_review', None)
        records['M1 Ã—2 Â· ' + suffix] = source
        d = copy.deepcopy(source)
        rid = f'20261027_argentina_allblacks_20250816_m0_cost2_stride{s}'
        events = p.read(HERE / f'stride_{s}/events.json')
        v = p.read(HERE / f'stride_{s}/metrics.json')
        d.update(result_id=rid, model_name='M0 Ã—2 Â· ' + suffix, detections=events, primary_detections=copy.deepcopy(events), window_predictions=[json.loads(x) for x in _resolve_model_artifact(HERE / f'stride_{s}/windows.jsonl').read_text(encoding='utf-8').splitlines()], human_evaluation=v['metrics'])
        d['metadata'].update(source_configuration='M0-H2-D2-S42-COST2', training_models={str(model): c['training']['model_sha256']}, uses_yolo=False, new_training=True, extractors_executed=0, resnet_executed=False, experimental_sampling_stride=s, human_evaluation_report='/static/reports/m0_cost2_professional_human_20261027/index.html', saved_decisions_locked=True, sampling_note='RGB-only full25FPS or selected every5 with original global linear interpolation; zero pose/geometry')
        d['metadata'].update(protocol='M0-H2-D2 RGB200;175 valid frames at25FPS; no YOLO or forest', uses_geometry_forest=False, experimental_interpolation='none' if s == 1 else 'linear RGB on global native frame grid', source_initialization_checkpoint=str(ROOT / 'models/M0/checkpoints/best_model/best_model.keras'), decision_rule='raw argmax without posterior class factors; same M1 event grouping')
        d['metadata']['event_policy_reference_result_id'] = d['metadata'].pop('source_result_id', None)
        by = {k: dict(count=sum((e['play_type'] == k for e in events)), total_duration=sum((e['duration'] for e in events if e['play_type'] == k))) for k in ('line', 'scrum')}
        d['summary'] = dict(total_detections=len(events), by_type={k: x['count'] for k, x in by.items()})
        d['statistics'] = dict(by_type=by, avg_confidence=sum((e['confidence'] for e in events)) / len(events) if events else 0, avg_duration=sum((e['duration'] for e in events)) / len(events) if events else 0)
        p.save(ROOT / f'data/results/result_{rid}.json', d)
        records['M0 Ã—2 Â· ' + suffix] = d
    variants = []
    proofs = {}
    matches = {}
    for name, r in records.items():
        v = p.evaluate_events(truth, r['detections'])
        pr = p.prior.independent_check(truth, r['detections'], v['metrics'])
        proofs[name] = pr
        for k in ('unique_flagged_s', 'flagged_outside_true_segments_s'):
            assert abs(pr[k] - v['review'][k]) < 1e-06
        matches[name] = {a['reference_id'] for a in v['assignments']}
        v.update(title=name, key='model_' + str(len(variants)), viewer_url='/viewer?result_id=' + r['result_id'])
        variants.append(v)
    models = [dict(key=v['key'], title=v['title'], missed=v['metrics']['total']['fn'], prediction_signature=[[e['play_type'], e['start_time'], e['end_time']] for e in records[v['title']]['detections']]) for v in variants]
    omissions = [dict(**t, detected={v['key']: t['id'] in matches[v['title']] for v in variants}) for t in truth if not all((t['id'] in matches[n] for n in records))]
    first = next(iter(records.values()))
    first['omission_review'] = dict(reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], models=models, segments=omissions, context_s=3, method='same-class one-to-one overlap>=1s')
    history = dict(id=CID, video_path=first['video_path'], video_name='Argentinaâ€“NZ16/08/2025 Â· M0Ã—2 vs M1Ã—2 Â· completo/cada5', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={n: dict(total=len(r['detections']), by_type={k: sum((e['play_type'] == k for e in r['detections'])) for k in ('line', 'scrum')}, avg_confidence=r['statistics']['avg_confidence']) for n, r in records.items()})
    assert p.post('/api/comparison-history', history)['success']
    assert p.get('/api/comparison-history/' + CID)['comparison']['results'] == records
    result = dict(status='complete', variants=variants, reference_id=ref['revision_id'], comparison_url='/compare?comparison=' + CID + '#omisiones', completion=c)
    p.save(HERE / 'results.json', result)
    p.save(WEB / 'results.json', result)
    rows = ''
    cls = ''
    miss = ''
    for v in variants:
        m = v['metrics']
        t = m['total']
        r = v['review']
        rows += f"<tr><th>{v['title']}</th><td>{m['line']['tp']}/32</td><td>{m['scrum']['tp']}/15</td><td>{p.pct(t['recall'])}</td><td>{p.pct(t['precision'])}</td><td>{p.pct(t['f1'])}</td><td>{t['alerts']}</td><td>{t['fp']}</td><td>{t['fn']}</td><td>{p.tm(r['unique_flagged_s'])}</td><td>{p.tm(r['flagged_outside_true_segments_s'])}</td></tr>"
        for label in ('line', 'scrum'):
            z = m[label]
            cls += f"<tr><th>{v['title']}</th><td>{label}</td><td>{z['tp']}/{z['reference_segments']}</td><td>{z['fp']}</td><td>{z['fn']}</td><td>{p.pct(z['recall'])}</td><td>{p.pct(z['precision'])}</td><td>{p.pct(z['f1'])}</td></tr>"
        miss += f"<tr><th>{v['title']}</th><td>" + (', '.join((f"{t['play_type']} {p.tm(t['start_s'])}â€“{p.tm(t['end_s'])}" for t in v['missed_segments'])) or 'Ninguno') + '</td></tr>'
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M0 Ã—2 frente a M1 Ã—2</title><style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:28px auto;padding:0 18px;color:#243447}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}thead,aside{background:#edf3f8}aside{padding:16px}a{color:#126b84}</style><h1>M0 Ã—2 frente a M1 Ã—2 Â· completo y cada 5</h1><p>Argentinaâ€“NZ 16/08/2025. Mismas etiquetas corregidas:32 line/15 scrum. M0 es el RGB200 histÃ³rico reajustado H2-D2, no el central8posiciones de otro experimento. ResNet congelada; M0 no usa YOLO ni bosque.</p><aside>Misma receta que M1 Ã—2:8.796 ejemplos, semilla42,5Ã©pocas fijas, lote16, Adam1e-4, balanceo de clasesÃ—[2,1,2] normalizado a media1. No se entrenÃ³ ni seleccionÃ³ con etiquetas del partido. Mismas6160ventanas7s/1s,175RGB+25ceros, misma polÃ­tica argmax de eventos y coincidencia por clase/solapamientoâ‰¥1s/una pareja. Cada5 reconstruye entradas mediante interpolaciÃ³n global; el 20% de fotogramas no implica 20% del tiempo total. Resultados descriptivos, sin generalizaciÃ³n certificada.</aside><p><a href="' + result['comparison_url'] + '">Video, timelines y omisiones</a> Â· <a href="results.json">Datos completos y tiempos</a></p><div class="scroll"><table data-table="total"><thead><tr><th>Modelo</th><th>Line</th><th>Scrum</th><th>Sensibilidad</th><th>PrecisiÃ³n</th><th>F1</th><th>Alertas</th><th>FP</th><th>FN</th><th>RevisiÃ³n Ãºnica</th><th>Extra</th></tr></thead><tbody>' + rows + '</tbody></table></div><p>Tiempos a1Ã— sin pausas, uniendo solapamientos. Extra: tiempo marcado fuera de segmentos positivos, incluyendo mÃ¡rgenes de alertas correctas. Hallar un segmento no garantiza cubrir toda su duraciÃ³n.</p><h2>Por clase</h2><div class="scroll"><table><thead><tr><th>Modelo</th><th>Clase</th><th>Aciertos</th><th>FP</th><th>FN</th><th>Sensibilidad</th><th>PrecisiÃ³n</th><th>F1</th></tr></thead><tbody>' + cls + '</tbody></table></div><h2>Segmentos sin pareja</h2><div class="scroll"><table><thead><tr><th>Modelo</th><th>Tiempos de transmisiÃ³n</th></tr></thead><tbody>' + miss + '</tbody></table></div></html>'
    (WEB / 'index.html').write_text(doc, encoding='utf-8')
    p.save(HERE / 'verification.json', dict(status='PASS', independent_checks=proofs, source_hashes_unchanged=True, model_hash_unchanged=True, comparison_roundtrip=True))
    subprocess.run(['node', str(HERE / 'verify_web.cjs')], cwd=ROOT, check=True)
    print(json.dumps(dict(status='published', variants=[dict(title=v['title'], metrics=v['metrics']['total'], review=v['review']) for v in variants])))
if __name__ == '__main__':
    main()
