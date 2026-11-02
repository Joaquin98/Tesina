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
import sys, copy, json, importlib.util, subprocess
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
spec = importlib.util.spec_from_file_location('helpers', _resolve_model_artifact(ROOT / 'models/common/studies/m1_cost2_other_professionals_20261027/publish.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
WEB = ROOT / 'src/web/static/reports/m1_line2_scrum1_other_scrum2_20261027'
CID = '20261027_argentina_allblacks_20250816_human_m1x2_vs_line2_scrum1_other_scrum2'

def main():
    assert p.read(HERE / 'evaluation_complete.json')['status'] == 'complete'
    train = p.read(HERE / 'training_complete.json')
    protocol = p.read(HERE / 'protocol.json')
    assert all((p.sha(Path(path)) == h for path, h in train['source_hashes'].items()))
    trained = train['models'][0]
    assert all((p.sha(Path(path)) == h for path, h in trained['model_hashes'].items()))
    ref = p.read(ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json')
    truth = ref['reference_segments']
    for s in ref['provenance']:
        assert p.sha(Path(s['path'])) == s['sha256']
    base = p.get('/api/comparison-history/20261027_argentina_allblacks_20250816_human_m0x2_vs_m1x2_full_every5')['comparison']['results']
    records = {}
    for stride in (1, 5):
        suffix = 'completo' if stride == 1 else 'cada 5'
        source = copy.deepcopy(base['M1 Ã—2 Â· ' + suffix])
        source.pop('omission_review', None)
        records['M1 Ã—2 Â· original Â· ' + suffix] = source
        r = copy.deepcopy(source)
        out = HERE / 'evaluation' / f'cost_2_stride_{stride}'
        ev = p.read(out / 'events.json')
        m = p.read(out / 'metrics.json')
        rid = f'20261027_argentina_allblacks_20250816_m1_l2s1_other_scrum2_stride{stride}'
        name = 'M1 L2-S1 Â· otroâ†’scrum Ã—2 Â· ' + suffix
        r.update(result_id=rid, model_name=name, detections=ev, primary_detections=copy.deepcopy(ev), window_predictions=[json.loads(x) for x in _resolve_model_artifact(out / 'windows.jsonl').read_text().splitlines()], human_evaluation=m['metrics'])
        r['metadata'].update(source_configuration='M1-L2-S1-OTHER_SCRUM2', training_models=trained['model_hashes'], training_priorities=protocol['requested_priorities'], training_weighting=protocol['implementation'], training_weighting_limitation=protocol['limitation'], source_checkpoint=protocol['initialization'], saved_decisions_locked=True, coverage_note='Argentinaâ€“NZ 16/08/2025; referencia humana corregida; decisiones argmax guardadas.', human_evaluation_report='/static/reports/m1_line2_scrum1_other_scrum2_20261027/index.html', extractors_executed=0)
        by = {k: dict(count=sum((e['play_type'] == k for e in ev)), total_duration=sum((e['duration'] for e in ev if e['play_type'] == k))) for k in ('line', 'scrum')}
        r['summary'] = dict(total_detections=len(ev), by_type={k: v['count'] for k, v in by.items()})
        r['statistics'] = dict(by_type=by, avg_confidence=sum((e['confidence'] for e in ev)) / len(ev) if ev else 0, avg_duration=sum((e['duration'] for e in ev)) / len(ev) if ev else 0)
        p.save(ROOT / f'data/results/result_{rid}.json', r)
        records[name] = r
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
    first['omission_review'] = dict(reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], models=models, segments=omissions, context_s=3, method='sameclass one-to-one overlap>=1s')
    hist = dict(id=CID, video_path=first['video_path'], video_name='Argentinaâ€“NZ 16/08/2025 Â· M1 line2/scrum1/otroâ†’scrum2', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={n: dict(total=len(r['detections']), by_type={k: sum((e['play_type'] == k for e in r['detections'])) for k in ('line', 'scrum')}, avg_confidence=r['statistics']['avg_confidence']) for n, r in records.items()})
    assert p.post('/api/comparison-history', hist)['success']
    assert p.get('/api/comparison-history/' + CID)['comparison']['results'] == records
    result = dict(status='complete', variants=variants, reference_id=ref['revision_id'], comparison_url='/compare?comparison=' + CID + '#omisiones', training=train, protocol=protocol, evaluation=p.read(HERE / 'evaluation_complete.json'))
    p.save(HERE / 'results.json', result)
    p.save(WEB / 'results.json', result)
    rows = ''
    classes = ''
    miss = ''
    for v in variants:
        m = v['metrics']
        t = m['total']
        r = v['review']
        rows += f"<tr><th>{v['title']}</th><td>{m['line']['tp']}/32</td><td>{m['scrum']['tp']}/15</td><td>{p.pct(t['recall'])}</td><td>{p.pct(t['precision'])}</td><td>{p.pct(t['f1'])}</td><td>{t['alerts']}</td><td>{t['fp']}</td><td>{t['fn']}</td><td>{p.tm(r['unique_flagged_s'])}</td><td>{p.tm(r['flagged_outside_true_segments_s'])}</td></tr>"
        for label in ('line', 'scrum'):
            z = m[label]
            classes += f"<tr><th>{v['title']}</th><td>{label}</td><td>{z['tp']}/{z['reference_segments']}</td><td>{z['fp']}</td><td>{z['fn']}</td><td>{p.pct(z['precision'])}</td><td>{p.pct(z['recall'])}</td><td>{p.pct(z['f1'])}</td></tr>"
        miss += f"<tr><th>{v['title']}</th><td>" + (', '.join((f"{s['play_type']} {p.tm(s['start_s'])}â€“{p.tm(s['end_s'])}" for s in v['missed_segments'])) or 'Ninguno') + '</td></tr>'
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1: line Ã—2, scrum Ã—1, otroâ†’scrum Ã—2</title><style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:28px auto;padding:0 18px;color:#243447}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}thead,aside{background:#edf3f8}aside{padding:16px}a{color:#126b84}</style><h1>M1: line Ã—2, scrum Ã—1, otroâ†’scrum Ã—2</h1><p>Argentinaâ€“Nueva Zelanda, 16/08/2025. Anotaciones manuales corregidas: 32 segmentos line y 15 scrum.</p><aside>Configuración evaluada: line realâ†’cualquier error Ã—2; scrum realâ†’cualquier error Ã—1; otroâ†’scrum Ã—2; otroâ†’line Ã—1. ImplementaciÃ³n: pesos balanceados de clase multiplicados por [2,1,1] y peso doble en los 106 ejemplos de entrenamiento Â«otroÂ» que M1 Ã—2 confundÃ­a con scrum, normalizados a media 1. La lista se congela antes de entrenar: es ponderaciÃ³n de negativos difÃ­ciles, no una pÃ©rdida dinÃ¡mica por cada confusiÃ³n. Los ejemplos corregidos conservan su peso.</aside><p>Cinco Ã©pocas fijas desde el mismo M0/H2, mismo fingerprint inicial, semilla 42, lote 16, 8.796 ejemplos D2 y Adam 1e-4 del M1 original; RF de 500 Ã¡rboles con los mismos pesos efectivos. FusiÃ³n geomÃ©trica con Î±=0,475. Se reutilizÃ³ RGB, pose, geometrÃ­a y negativos cacheados; cero extracciÃ³n. Las etiquetas del partido sÃ³lo se usan para evaluar. Misma polÃ­tica argmax y coincidencia de clase con â‰¥1 s de solapamiento, asignaciÃ³n uno a uno. Cada 5 reconstruye la grilla mediante interpolaciÃ³n global.</p><p><a href="' + result['comparison_url'] + '">Video, timelines y omisiones</a> Â· <a href="results.json">Datos completos y procedencia</a></p><div class="scroll"><table data-table="total"><thead><tr><th>Modelo</th><th>Line</th><th>Scrum</th><th>Sensibilidad</th><th>PrecisiÃ³n</th><th>F1</th><th>Alertas</th><th>FP</th><th>FN</th><th>RevisiÃ³n</th><th>Extra</th></tr></thead><tbody>' + rows + '</tbody></table></div><p>RevisiÃ³n a 1Ã— sin pausas, uniendo solapamientos. Extra: tiempo marcado fuera de segmentos positivos, incluidos mÃ¡rgenes de alertas correctas. Detectar un segmento no garantiza cubrir toda su duraciÃ³n. Resultados descriptivos de un partido cuyas etiquetas ya se consultaron en pruebas previas.</p><h2>Por clase</h2><div class="scroll"><table><thead><tr><th>Modelo</th><th>Clase</th><th>Aciertos</th><th>FP</th><th>FN</th><th>PrecisiÃ³n</th><th>Sensibilidad</th><th>F1</th></tr></thead><tbody>' + classes + '</tbody></table></div><h2>Omisiones</h2><div class="scroll"><table><thead><tr><th>Modelo</th><th>Tiempos del video</th></tr></thead><tbody>' + miss + '</tbody></table></div><h2>Tiempos medidos</h2><p>Entrenamiento de red: ' + p.tm(trained['dual_s']) + '. Bosque: ' + f"{trained['forest_s']:.1f}" + ' s. EvaluaciÃ³n con cachÃ© completo/cada 5: ' + ' / '.join((f"{r['cached_evaluation_seconds']:.1f} s" for r in result['evaluation']['records'])) + '. No son tiempos desde video sin cachÃ©.</p></html>'
    (WEB / 'index.html').write_text(doc, encoding='utf-8')
    p.save(HERE / 'verification.json', dict(status='PASS', independent_checks=proofs, source_hashes_unchanged=True, comparison_roundtrip=True))
    subprocess.run(['node', str(HERE / 'verify_web.cjs')], cwd=ROOT, check=True)
    print('Published and verified original vs L2-S1-otherScrum2', flush=True)
if __name__ == '__main__':
    main()
