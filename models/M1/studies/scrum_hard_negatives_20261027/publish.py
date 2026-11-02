"""Publish all fixed experimental arms, including matched continuation controls."""
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
import sys, copy, json, importlib.util, subprocess, html
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(source_directory(ROOT / 'data/m1_recall_cost_20261027'))]
spec = importlib.util.spec_from_file_location('helpers', _resolve_model_artifact(ROOT / 'models/common/studies/m1_cost2_other_professionals_20261027/publish.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
WEB = ROOT / 'src/web/static/reports/scrum_hard_negatives_20261027'
BASECID = '20261027_argentina_allblacks_20250816_human_m0x2_vs_m1x2_full_every5'
FACTORS = (1, 1.5, 2)

def title(family, factor, stride):
    arm = 'original' if factor is None else 'control +1 Ã©poca' if factor == 1 else 'negativos Ã—' + str(factor).replace('.', ',')
    return family.upper() + ' Ã—2 Â· ' + arm + ' Â· ' + ('completo' if stride == 1 else 'cada 5')

def main():
    WEB.mkdir(parents=True, exist_ok=True)
    ref = p.read(ROOT / 'data/sampling_audit_20261026/human_revisions/labels_fd96032b00fa/metrics.json')
    truth = ref['reference_segments']
    for s in ref['provenance']:
        assert p.sha(Path(s['path'])) == s['sha256']
    base = p.get('/api/comparison-history/' + BASECID)['comparison']['results']
    all_variants = []
    comparisons = []
    proofs = {}
    training = {}
    for family in ('m0', 'm1'):
        folder = HERE / family
        assert p.read(folder / 'evaluation_complete.json')['status'] == 'complete'
        train = p.read(folder / 'training_complete.json')
        training[family] = train
        assert all((p.sha(Path(path)) == h for path, h in train['source_hashes'].items()))
        records = {}
        variants = []
        matches = {}
        for factor in (None,) + FACTORS:
            for stride in (1, 5):
                name = title(family, factor, stride)
                suffix = 'completo' if stride == 1 else 'cada 5'
                r = copy.deepcopy(base[family.upper() + ' Ã—2 Â· ' + suffix])
                r.pop('omission_review', None)
                if factor is not None:
                    out = folder / 'evaluation' / f'cost_{factor}_stride_{stride}'
                    ev = p.read(out / 'events.json')
                    metric = p.read(out / 'metrics.json')
                    rid = f"20261027_argentina_allblacks_20250816_{family}x2_hard{str(factor).replace('.', 'p')}_stride{stride}"
                    r.update(result_id=rid, model_name=name, detections=ev, primary_detections=copy.deepcopy(ev), window_predictions=[json.loads(x) for x in _resolve_model_artifact(out / 'windows.jsonl').read_text(encoding='utf-8').splitlines()], human_evaluation=metric['metrics'])
                    r['metadata'].update(hard_negative_factor=factor, additional_training_epochs=1, hard_negative_training_examples=train['mining']['hard_negatives'], training_models=metric['model_hashes'], source_configuration=f'{family.upper()}-COST2-HARDNEG-{factor}', saved_decisions_locked=True, source_checkpoint=p.read(folder / 'protocol.json')['source_checkpoint'], source_initial_weights_sha256=train['models'][0]['initial_weights_sha256'], human_evaluation_report='/static/reports/scrum_hard_negatives_20261027/index.html', coverage_note='Argentinaâ€“NZ, 16/08/2025; mismas etiquetas corregidas y polÃ­tica de eventos.', extractors_executed=0)
                    by = {k: dict(count=sum((e['play_type'] == k for e in ev)), total_duration=sum((e['duration'] for e in ev if e['play_type'] == k))) for k in ('line', 'scrum')}
                    r['summary'] = dict(total_detections=len(ev), by_type={k: v['count'] for k, v in by.items()})
                    r['statistics'] = dict(by_type=by, avg_confidence=sum((e['confidence'] for e in ev)) / len(ev) if ev else 0, avg_duration=sum((e['duration'] for e in ev)) / len(ev) if ev else 0)
                    p.save(ROOT / f'data/results/result_{rid}.json', r)
                records[name] = r
                v = p.evaluate_events(truth, r['detections'])
                pr = p.prior.independent_check(truth, r['detections'], v['metrics'])
                for k in ('unique_flagged_s', 'flagged_outside_true_segments_s'):
                    assert abs(pr[k] - v['review'][k]) < 1e-06
                proofs[name] = pr
                matches[name] = {a['reference_id'] for a in v['assignments']}
                v.update(title=name, family=family, factor=factor, stride=stride, key='model_' + str(len(variants)), viewer_url='/viewer?result_id=' + r['result_id'])
                if factor is not None:
                    v['cached_evaluation_s'] = metric.get('cached_evaluation_seconds')
                variants.append(v)
                all_variants.append(v)
        models = [dict(key=v['key'], title=v['title'], missed=v['metrics']['total']['fn'], prediction_signature=[[e['play_type'], e['start_time'], e['end_time']] for e in records[v['title']]['detections']]) for v in variants]
        omissions = [dict(**t, detected={v['key']: t['id'] in matches[v['title']] for v in variants}) for t in truth if not all((t['id'] in matches[n] for n in records))]
        first = next(iter(records.values()))
        first['omission_review'] = dict(reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], models=models, segments=omissions, context_s=3, method='sameclass one-to-one overlap>=1s')
        cid = f'20261027_argentina_allblacks_20250816_human_{family}x2_scrum_hardnegatives'
        hist = dict(id=cid, video_path=first['video_path'], video_name='Argentinaâ€“NZ 16/08/2025 Â· ' + family.upper() + ' Ã—2 Â· negativos scrum', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={n: dict(total=len(r['detections']), by_type={k: sum((e['play_type'] == k for e in r['detections'])) for k in ('line', 'scrum')}, avg_confidence=r['statistics']['avg_confidence']) for n, r in records.items()})
        assert p.post('/api/comparison-history', hist)['success']
        assert p.get('/api/comparison-history/' + cid)['comparison']['results'] == records
        comparisons.append(dict(family=family, url='/compare?comparison=' + cid + '#omisiones', variants=variants))
    result = dict(status='complete', reference_id=ref['revision_id'], variants=all_variants, comparisons=comparisons, training=training)
    p.save(HERE / 'results.json', result)
    p.save(WEB / 'results.json', result)
    tables = ''
    details = ''
    timing = ''
    for comp in comparisons:
        family = comp['family']
        mining = training[family]['mining']
        rows = ''
        for v in comp['variants']:
            m = v['metrics']
            t = m['total']
            r = v['review']
            s = m['scrum']
            rows += f"<tr><th>{v['title']}</th><td>{m['line']['tp']}/32</td><td>{s['tp']}/15</td><td>{p.pct(t['recall'])}</td><td>{p.pct(t['precision'])}</td><td>{p.pct(s['precision'])}</td><td>{s['fp']}</td><td>{t['fp']}</td><td>{t['fn']}</td><td>{p.tm(r['unique_flagged_s'])}</td><td>{p.tm(r['flagged_outside_true_segments_s'])}</td></tr>"
            detail_rows = ''
            for label in ('line', 'scrum', 'total'):
                z = m[label]
                detail_rows += f"<tr><th>{label}</th><td>{z['tp']}</td><td>{z['fp']}</td><td>{z['fn']}</td><td>{p.pct(z['precision'])}</td><td>{p.pct(z['recall'])}</td><td>{p.pct(z['f1'])}</td></tr>"
            miss = ', '.join((f"{t['play_type']} {p.tm(t['start_s'])}â€“{p.tm(t['end_s'])}" for t in v['missed_segments'])) or 'Ninguna'
            details += f"""<details><summary>{v['title']}</summary><div class="scroll"><table><thead><tr><th>Clase</th><th>TP</th><th>FP</th><th>FN</th><th>PrecisiÃ³n</th><th>Sensibilidad</th><th>F1</th></tr></thead><tbody>{detail_rows}</tbody></table></div><p>Omisiones: {miss}.</p><p>Suma de alertas sin unir: {p.tm(r['sum_alert_s'])}. EvaluaciÃ³n con cachÃ©: """ + (f"{v['cached_evaluation_s']:.2f} s" if v.get('cached_evaluation_s') is not None else 'ver informe original') + '.</p></details>'
        tables += f'''<h2>{family.upper()} Ã—2</h2><p>{mining['hard_negatives']} negativos difÃ­ciles entre {mining['training_other']} ejemplos Â«otroÂ» de entrenamiento. <a href="{comp['url']}">Ver video, ocho timelines y omisiones</a>.</p><div class="scroll"><table data-family="{family}"><thead><tr><th>Variante</th><th>Line</th><th>Scrum</th><th>Sensibilidad total</th><th>PrecisiÃ³n total</th><th>PrecisiÃ³n scrum</th><th>FP scrum</th><th>FP total</th><th>FN</th><th>RevisiÃ³n</th><th>Extra</th></tr></thead><tbody>{rows}</tbody></table></div>'''
        for r in training[family]['models']:
            timing += f"<tr><th>{family.upper()} Ã—2 Â· factor {r['factor']}</th><td>{r['training_s']:.1f} s</td><td>{r['forest_s']:.1f} s</td></tr>"
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reducir falsas alertas scrum Â· M0 y M1 Ã—2</title><style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:28px auto;padding:0 18px;color:#243447}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}thead,aside{background:#edf3f8}aside{padding:16px}a{color:#126b84}details{border:1px solid #ddd;margin:12px 0;padding:12px}summary{cursor:pointer}</style><h1>Reducir falsas alertas scrum Â· M0 y M1 Ã—2</h1><p>Argentinaâ€“Nueva Zelanda, 16/08/2025. Anotaciones manuales corregidas: 32 segmentos line y 15 scrum.</p><aside>Prueba exploratoria de una Ã©poca adicional desde cada checkpoint Ã—2. Se congelan los ejemplos D2 con etiqueta Â«otroÂ» que el modelo Ã—2 predice como scrum. Se prueban pesos Ã—1,5 y Ã—2 sÃ³lo en esos ejemplos; el control recibe la misma Ã©poca con peso Ã—1. Pesos de clase Ã—2 originales, normalizaciÃ³n a media 1, semilla 42, lote 16 y Adam 1e-4 reiniciado en todos los brazos. M1 pondera igual la red y el bosque. Las etiquetas del partido sÃ³lo se usan para evaluar. No es un reentrenamiento completo ni un ajuste de umbrales.</aside><p>Completo y cada 5 usan la misma polÃ­tica argmax, asignaciÃ³n uno a uno por clase con â‰¥1 s de solapamiento y 6.160 ventanas. Cada 5 reconstruye la grilla mediante interpolaciÃ³n global a partir de los vectores seleccionados. Cero extracciÃ³n nueva.</p>' + tables + '<p>RevisiÃ³n: tiempo marcado a 1Ã— sin pausas, uniendo solapamientos. Extra: tiempo marcado fuera de los segmentos manuales, incluidos mÃ¡rgenes de alertas acertadas. Detectar un segmento no garantiza cubrir toda su duraciÃ³n. Los resultados describen este partido; las etiquetas ya fueron consultadas en experimentos previos.</p><h2>EstadÃ­sticas por clase y omisiones</h2>' + details + '<h2>Tiempos de ajuste medidos</h2><p>Trabajadores simultÃ¡neos con cuatro hilos por familia. Estos tiempos tienen competencia de recursos; no representan una comparaciÃ³n de velocidad con ejecuciones anteriores. EvaluaciÃ³n desde cachÃ© no incluye extracciÃ³n desde el video.</p><div class="scroll"><table><thead><tr><th>Variante</th><th>Red, una Ã©poca</th><th>Bosque</th></tr></thead><tbody>' + timing + '</tbody></table></div><p><a href="results.json">Resultados completos y procedencia</a></p></html>'
    (WEB / 'index.html').write_text(doc, encoding='utf-8')
    p.save(HERE / 'verification.json', dict(status='PASS', independent_checks=proofs, original_hashes_unchanged=True, comparison_roundtrips=True))
    subprocess.run(['node', str(HERE / 'verify_web.cjs')], cwd=ROOT, check=True)
    print('Published and verified all 16 variants', flush=True)
if __name__ == '__main__':
    main()
