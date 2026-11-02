"""Publish native comparison and full metrics, preserving earlier models.common.studies."""
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
import sys, json, copy, html, importlib.util
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(HERE)), str(ROOT)]
spec = importlib.util.spec_from_file_location('prior_publish', _resolve_model_artifact(ROOT / 'models/common/studies/m1_cost2_other_professionals_20261027/publish.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from extract import read, save, sha
WEB = ROOT / 'src/web/static/reports/m1_cost2_rwc2023_full_20261028'

def main():
    complete = read(HERE / 'evaluation_complete.json')
    assert complete['status'] == 'complete'
    proof = read(HERE / 'cache_verification.json')
    assert proof['status'] == 'PASS' and proof['all_anchors_bitwise_identical']
    base = read(ROOT / 'data/results/result_20261025_argentina_allblacks_20231020_argmax_every25.json')
    truth = complete['reference_segments']
    new = []
    checks = {}
    models = {}
    for value in complete['records']:
        mode = value['mode']
        dest = HERE / mode
        events = read(dest / 'events.json')
        rows = [json.loads(x) for x in _resolve_model_artifact(dest / 'windows.jsonl').read_text(encoding='utf-8').splitlines()]
        metrics = p.evaluate_events(truth, events)
        assert metrics['metrics'] == value['metrics']['metrics']
        check = p.prior.independent_check(truth, events, metrics['metrics'], complete['duration_s'])
        for k in ('unique_flagged_s', 'flagged_outside_true_segments_s'):
            assert abs(check[k] - metrics['review'][k]) < 1e-06
        for path, h in value['model_hashes'].items():
            assert sha(path) == h
        assert read(HERE / f'{mode}_sparse_control.json')['status'] == 'PASS'
        checks[mode] = check
        title = 'M1 original · completo' if mode == 'original' else 'M1 ×2 · completo'
        rid = f'20261028_argentina_allblacks_20231020_m1_{mode}_stride1'
        payload = copy.deepcopy(base)
        for k in ('sampling_comparison', 'exploratory_review', 'processing_timing', 'manual_comparison', 'omission_review'):
            payload.pop(k, None)
        payload.update(result_id=rid, model_name=title, detections=events, primary_detections=copy.deepcopy(events), window_predictions=rows)
        by = {label: dict(count=sum((e['play_type'] == label for e in events)), total_duration=sum((e['end_time'] - e['start_time'] for e in events if e['play_type'] == label))) for label in ('line', 'scrum')}
        payload['statistics'] = dict(by_type=by, avg_confidence=sum((e['confidence'] for e in events)) / len(events) if events else 0.0, avg_duration=sum((e['end_time'] - e['start_time'] for e in events)) / len(events) if events else 0.0)
        payload['summary'] = dict(total_detections=len(events), by_type={k: v['count'] for k, v in by.items()})
        payload.setdefault('metadata', {}).update(source_configuration='ALL-DEVELOPMENT-M1-H2-D2-S42' + ('-COST2' if mode == 'cost2' else ''), source_result_id='20261025_argentina_allblacks_20231020_argmax_every25', protocol='training_faithful_highres_rint100_full', experimental_sampling_stride=1, experimental_interpolation=False, rgb_stride=1, yolo_stride=1, saved_decisions_locked=True, new_training=False, human_labels_available=False, training_models=value['model_hashes'], cached_evaluation_seconds=value['cached_inference_s'], reference_scope='partial automated diagnostic inventory; not certified full-match sensitivity or precision', quality_status='Exploratory temporal matching against partial automated diagnostic inventory', imported_from_completed_experiment=str(HERE), experiment_report_url=p.BASE + '/static/reports/m1_cost2_rwc2023_full_20261028/index.html')
        payload['metadata']['native_frames'] = 144422
        payload['metadata']['selected_frames'] = 144422
        if 'sampling' in payload:
            payload['sampling'] = dict(rgb_stride=1, yolo_stride=1, source_frames=144422, selected_frames=144422)
        save(ROOT / f'data/results/result_{rid}.json', payload)
        models[title] = payload
        new.append(dict(date='20231020', title=title, stride=1, mode=mode, metrics=metrics['metrics'], review=metrics['review'], missed_segments=metrics['missed_segments'], cached_inference_s=value['cached_inference_s'], viewer_url='/viewer?result_id=' + rid))
    prior_results = read(ROOT / 'data/m1_cost2_other_professionals_20261027/results.json')
    variants = []
    comparisons = {}
    for date in ('20250823', '20231020'):
        variants.extend((v for v in prior_results['variants'] if v['date'] == date and (v['stride'] == 1 if date == '20250823' else True)))
    variants.extend(new)
    for date in ('20250823', '20231020'):
        results = {}
        for v in variants:
            if v['date'] != date:
                continue
            title = ('M1 original' if v['mode'] == 'original' else 'M1 ×2') + (' · completo' if v['stride'] == 1 else ' · cada 25')
            rid = v['viewer_url'].split('result_id=')[1]
            results[title] = read(ROOT / f'data/results/result_{rid}.json')
        cid = f'20261028_argentina_allblacks_{date}_m1_original_vs_x2_full'
        first = next(iter(results.values()))
        history = dict(id=cid, video_path=first['video_path'], video_name=f'Argentina–NZ · {date} · M1 original vs ×2 · completo', model_names=list(results), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=results, comparison={name: dict(total=len(v['detections']), by_type={label: sum((e['play_type'] == label for e in v['detections'])) for label in ('line', 'scrum')}, avg_confidence=v['statistics']['avg_confidence']) for name, v in results.items()})
        assert p.post('/api/comparison-history', history)['success']
        assert p.get('/api/comparison-history/' + cid)['comparison']['results'] == results
        comparisons[date] = '/compare?comparison=' + cid
    sections = []
    for date in ('20250823', '20231020'):
        selected = [v for v in variants if v['date'] == date]
        rows = ''
        classes = ''
        timelines = ''
        reference = truth if date == '20231020' else read(ROOT / 'data/m1_cost2_other_professionals_20261027/20250823_stride1/metrics.json')['reference_segments']
        duration = 5776.88 if date == '20231020' else 5680.44
        timelines = p.timeline('Inventario diagnóstico automático parcial', reference, duration, True)
        for v in selected:
            m = v['metrics']['total']
            rv = v['review']
            events = read(ROOT / ('data/results/result_' + v['viewer_url'].split('result_id=')[1] + '.json'))['detections']
            title = ('M1 original' if v['mode'] == 'original' else 'M1 ×2') + (' completo' if v['stride'] == 1 else ' cada25')
            rows += f'''<tr><th>{title}</th><td>{v['metrics']['line']['tp']}/{v['metrics']['line']['reference_segments']}</td><td>{v['metrics']['scrum']['tp']}/{v['metrics']['scrum']['reference_segments']}</td><td>{p.pct(m['recall'])}</td><td>{p.pct(m['precision'])}</td><td>{m['alerts']}</td><td>{p.tm(rv['unique_flagged_s'])}</td><td>{p.tm(rv['flagged_outside_true_segments_s'])}</td><td><a href="{v['viewer_url']}">Video</a></td></tr>'''
            for label in ('line', 'scrum'):
                c = v['metrics'][label]
                classes += f"<tr><th>{title}</th><td>{label}</td><td>{c['tp']}</td><td>{c['fn']}</td><td>{c['fp']}</td><td>{p.pct(c['precision'])}</td><td>{p.pct(c['recall'])}</td><td>{p.pct(c['f1'])}</td></tr>"
            timelines += p.timeline(title, events, duration)
        date_name = '23/08/2025 · Argentina 29–23 NZ' if date == '20250823' else '20/10/2023 · Argentina 6–44 NZ'
        sections.append(f'<section><h2>{date_name}</h2><p><a href="{comparisons[date]}">Comparar con video en la herramienta</a></p><div class="scroll"><table data-game="{date}"><thead><tr><th>Variante</th><th>Line</th><th>Scrum</th><th>Sensibilidad sobre inventario</th><th>Precisión temporal exploratoria</th><th>Alertas</th><th>Revisión a 1×</th><th>Fuera del inventario</th><th>Ver</th></tr></thead><tbody>{rows}</tbody></table></div>{timelines}<div class="scroll"><table><thead><tr><th>Variante</th><th>Clase</th><th>Con pareja</th><th>Omitidos</th><th>Sin pareja</th><th>Precisión temporal</th><th>Sensibilidad</th><th>F1</th></tr></thead><tbody>{classes}</tbody></table></div></section>')
    document = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 ×2 completo · otros profesionales</title><style>body{font:16px/1.5 system-ui;max-width:1450px;margin:28px auto;padding:0 18px;color:#243447}aside{background:#fff0d6;padding:18px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{padding:10px;text-align:left;border-bottom:1px solid #ccd5dd}thead{background:#edf3f8}.timeline svg{display:block;width:100%;height:26px}.timeline{margin:10px 0}a{color:#126b84}section{margin:40px 0}</style><h1>M1 original y ×2 completos en los otros profesionales</h1><p>28/10/2026. Mismos modelos congelados, sin entrenamiento nuevo. Mundial: 144.422 observaciones nativas para RGB y YOLO; se copiaron exactamente 5.778 observaciones existentes y se calcularon 138.644 faltantes. Ventanas de 7 s cada 1 s, 175 posiciones válidas más 25 ceros; política argmax y agrupación originales.</p><aside>Ambos partidos usan inventarios diagnósticos automáticos parciales, no etiquetas humanas completas: 20 line + 7 scrum (23/08) y 29 line + 8 scrum (Mundial). Sensibilidad sobre esos inventarios; precisión temporal con parejas únicas de la misma clase y solapamiento mínimo de 1 s. Las alertas sin pareja no son falsos positivos confirmados; pueden ser repeticiones o jugadas fuera del inventario. El tiempo fuera del inventario tampoco es costo falso confirmado. Revisión a 1×, unión de intervalos, sin pausas.</aside><p><a href="results.json">Datos completos</a> · <a href="verification.json">Verificación independiente</a></p>' + ''.join(sections) + f'<h2>Tiempos del Mundial</h2><p>Inferencia desde el caché completado: ' + ', '.join((v['mode'] + ' ' + p.tm(v['cached_inference_s']) for v in new)) + '. La extracción nueva se mide por separado; los 5.778 anclajes ya existían, por lo que esta ejecución no es un ensayo en frío sin caché.</p></html>'
    result = dict(status='complete', variants=variants, comparisons=comparisons, cache_verification=proof, evaluation=complete, reference_scope='partial automated diagnostic inventory; not human full-match ground truth')
    verification = dict(status='PASS', independent_matching_and_review=checks, native_comparison_roundtrip=True, all_anchors_bitwise_identical=True, original_model_hashes_unchanged=True)
    for path in (HERE, WEB):
        save(path / 'results.json', result)
        save(path / 'verification.json', verification)
        (path / 'index.html').write_text(document, encoding='utf-8')
    print(json.dumps(dict(status='published', comparisons=comparisons)), flush=True)
if __name__ == '__main__':
    main()
    import subprocess
    subprocess.run(['node', str(HERE / 'verify_web.cjs')], cwd=str(ROOT), creationflags=subprocess.CREATE_NO_WINDOW, check=True)
