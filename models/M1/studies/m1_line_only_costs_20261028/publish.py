"""Publish the fixed cost sweep; keep human and partial references distinct."""
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
import sys, json, copy, html, importlib.util, subprocess, traceback
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(HERE)), str(ROOT)]
from models.M1.studies.m1_line_only_costs_20261028.run import read, save, sha
spec = importlib.util.spec_from_file_location('previous_publication', _resolve_model_artifact(ROOT / 'models/common/studies/m1_cost2_other_professionals_20261027/publish.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
from models.M1.studies.m1_line_only_costs_20261028.evaluate import JOBS
WEB = ROOT / 'src/web/static/reports/m1_line_only_costs_20261028'
BASE = 'http://127.0.0.1:5000'

def main():
    training = read(HERE / 'training_complete.json')
    evaluated = read(HERE / 'evaluation_complete.json')
    protocol = read(HERE / 'protocol.json')
    assert training['status'] == evaluated['status'] == 'complete' and len(evaluated['records']) == 9
    for path, h in protocol['source_hashes'].items():
        assert sha(path) == h
    for path, stats in protocol['cache_stats'].items():
        assert [Path(path).stat().st_size, Path(path).stat().st_mtime_ns] == stats
    summaries = []
    checks = {}
    comparisons = {}
    sections = []
    for job in JOBS:
        date = job['date']
        ref = read(job['reference'])
        truth = ref['reference_segments'] if job['reference_kind'] == 'human' else ref['events']
        human = job['reference_kind'] == 'human'
        pinned = None
        if human:
            for source in ref['provenance']:
                assert sha(source['path']) == source['sha256']
            pinned = dict(reference_session_pinned=True, composite_reference=True, reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], session_id=ref['session_ids'][0], source_session_ids=ref['session_ids'], labels=read(job['reference'].parent / 'labels.json')['labels'], summary=dict(unit='blocks', reviewed=881, total_candidates=881, pending=0, final_counts=ref['block_counts']))
        source = read(ROOT / f"data/results/result_{job['baseline_id']}.json")
        records = {}
        local = []
        matches = {}
        variants = [dict(key='original', title='Original Â· completo', rid=job['original_id']), dict(key='x2', title='M1 Ã—2 Â· line2 / scrum2', rid=job['baseline_id']), dict(key='old_line4', title='Anterior Â· line4 / scrum2', rid=f'20261028_argentina_allblacks_{date}_m1_line4_scrum2_stride1', line_cost=4), dict(key='old_line8', title='Anterior Â· line8 / scrum2', rid=f'20261028_argentina_allblacks_{date}_m1_line8_scrum2_stride1', line_cost=8)] + [dict(key=f'line{cost}', title=f'Nuevo Â· line{cost} / scrum1', cost=cost, rid=f'20261028_argentina_allblacks_{date}_m1_line{cost}_scrum1_stride1') for cost in (2, 4, 8)]
        for v in variants:
            cost = v.get('cost')
            rid = v['rid']
            title = v['title']
            if cost:
                out = HERE / f'evaluation/{date}/line{cost}_scrum1'
                value = read(out / 'metrics.json')
                events = read(out / 'events.json')
                rows = [json.loads(x) for x in _resolve_model_artifact(out / 'windows.jsonl').read_text(encoding='utf-8').splitlines()]
                assert value['reference_sha256'] == sha(job['reference']) and len(rows) == value['windows']
                payload = copy.deepcopy(source)
                for k in ('sampling_comparison', 'exploratory_review', 'processing_timing', 'manual_comparison', 'omission_review', 'human_evaluation'):
                    payload.pop(k, None)
                payload.update(result_id=rid, model_name=title, detections=events, primary_detections=copy.deepcopy(events), window_predictions=rows)
                by = {label: dict(count=sum((e['play_type'] == label for e in events)), total_duration=sum((e['end_time'] - e['start_time'] for e in events if e['play_type'] == label))) for label in ('line', 'scrum')}
                payload['statistics'] = dict(by_type=by, avg_confidence=sum((e['confidence'] for e in events)) / len(events) if events else 0.0, avg_duration=sum((e['end_time'] - e['start_time'] for e in events)) / len(events) if events else 0.0)
                payload['summary'] = dict(total_detections=len(events), by_type={k: v['count'] for k, v in by.items()})
                payload['metadata'].pop('training_class_cost', None)
                payload['metadata'].update(source_configuration=f'M1-H2-D2-S42-LINE{cost}-OTHER1-SCRUM1', training_class_costs=dict(line=cost, other=1, scrum=1), training_models=value['model_hashes'], source_result_id=job['baseline_id'], experimental_sampling_stride=1, saved_decisions_locked=True, extractors_executed=0, new_training=True, human_labels_available=human, reference_scope=job['reference_kind'], cached_evaluation_seconds=value['cached_inference_s'], experiment_report_url=BASE + '/static/reports/m1_line_only_costs_20261028/index.html')
                for path, h in value['model_hashes'].items():
                    assert sha(path) == h
            else:
                payload = read(ROOT / f'data/results/result_{rid}.json')
                payload.pop('omission_review', None)
                payload['result_id'] = rid
                payload['model_name'] = title
                events = payload['detections']
                value = None
            if human:
                payload['manual_comparison'] = copy.deepcopy(pinned)
            else:
                payload.pop('manual_comparison', None)
                payload.pop('human_evaluation', None)
            result = p.evaluate_events(truth, events)
            if value:
                assert result['metrics'] == value['metrics']
            if human:
                payload['human_evaluation'] = result['metrics']
            check = p.prior.independent_check(truth, events, result['metrics'], job['duration'])
            for k in ('unique_flagged_s', 'flagged_outside_true_segments_s'):
                assert abs(check[k] - result['review'][k]) < 1e-06
            checks[date + '_' + v['key']] = check
            matches[v['key']] = {a['reference_id'] for a in result['assignments']}
            if cost:
                save(ROOT / f'data/results/result_{rid}.json', payload)
            records[title] = payload
            item = dict(date=date, key=v['key'], title=title, line_cost=cost if cost else v.get('line_cost', 2 if v['key'] == 'x2' else None), scrum_cost=1 if cost else 2 if v['key'] != 'original' else None, reference_kind=job['reference_kind'], metrics=result['metrics'], review=result['review'], missed_segments=result['missed_segments'], viewer_url='/viewer?result_id=' + rid, cached_inference_s=value['cached_inference_s'] if value else None)
            summaries.append(item)
            local.append(item)
        first = next(iter(records.values()))
        if human:
            omissions = [dict(**s, detected={v['key']: s['id'] in matches[v['key']] for v in local}) for s in truth if not all((s['id'] in found for found in matches.values()))]
            first['omission_review'] = dict(reference_id=ref['revision_id'], reference_sha256=ref['labels_sha256'], models=[dict(key=v['key'], title=v['title'], missed=v['metrics']['total']['fn'], prediction_signature=[[e['play_type'], e['start_time'], e['end_time']] for e in records[v['title']]['detections']]) for v in local], segments=omissions, context_s=3, method='sameclass overlap>=1s, one-to-one')
        cid = f'20261028_argentina_allblacks_{date}_m1_line_only_costs_full'
        history = dict(id=cid, video_path=first['video_path'], video_name=f'Argentinaâ€“NZ Â·{date} Â· costos de line Â· completo', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={name: dict(total=len(r['detections']), by_type={k: sum((e['play_type'] == k for e in r['detections'])) for k in ('line', 'scrum')}, avg_confidence=r['statistics']['avg_confidence']) for name, r in records.items()})
        assert p.post('/api/comparison-history', history)['success']
        assert p.get('/api/comparison-history/' + cid)['comparison']['results'] == records
        comparisons[date] = '/compare?comparison=' + cid + ('#omisiones' if human else '')
        rows = ''
        perclass = ''
        misses = ''
        timeline = p.timeline('Anotaciones manuales manuales' if human else 'Inventario diagnÃ³stico automÃ¡tico parcial', truth, job['duration'], True)
        for v in local:
            m = v['metrics']
            rv = v['review']
            label = v['title']
            rows += f'''<tr><th>{label}</th><td>{m['line']['tp']}/{m['line']['reference_segments']}</td><td>{p.pct(m['line']['recall'])}</td><td>{m['scrum']['tp']}/{m['scrum']['reference_segments']}</td><td>{p.pct(m['scrum']['recall'])}</td><td>{p.pct(m['total']['precision'])}</td><td>{m['total']['alerts']}</td><td>{p.tm(rv['unique_flagged_s'])}</td><td>{p.tm(rv['flagged_outside_true_segments_s'])}</td><td><a href="{v['viewer_url']}">Video</a></td></tr>'''
            for k in ('line', 'scrum'):
                c = m[k]
                perclass += f"<tr><th>{label}</th><td>{k}</td><td>{c['tp']}</td><td>{c['fn']}</td><td>{c['fp']}</td><td>{p.pct(c['precision'])}</td><td>{p.pct(c['recall'])}</td><td>{p.pct(c['f1'])}</td></tr>"
            times = ', '.join((f"{s['play_type']} {p.tm(s['start_s'])}â€“{p.tm(s['end_s'])}" for s in v['missed_segments'])) or 'Ninguna'
            misses += f'<tr><th>{label}</th><td>{times}</td></tr>'
            timeline += p.timeline(label, records[label]['detections'], job['duration'])
        date_label = {'20250816': '16/08/2025 Â· Argentina24â€“41NZ', '20250823': '23/08/2025 Â· Argentina29â€“23NZ', '20231020': '20/10/2023 Â· Argentina6â€“44NZ'}[date]
        scope = 'Anotaciones manuales:32line+15scrum=47segmentos.' if human else f"Inventario diagnÃ³stico automÃ¡tico parcial:{sum((s['play_type'] == 'line' for s in truth))}line+{sum((s['play_type'] == 'scrum' for s in truth))}scrum; no verdad humana completa."
        sections.append(f'<section><h2>{date_label}</h2><p>{scope}</p><p><a href="{comparisons[date]}">Comparar detecciones y video</a></p><div class="scroll"><table data-game="{date}"><thead><tr><th>Variante</th><th>Line</th><th>Sensibilidad line</th><th>Scrum</th><th>Sensibilidad scrum</th><th>PrecisiÃ³n total</th><th>Alertas</th><th>RevisiÃ³n1Ã—</th><th>Fuera de referencia</th><th>Ver</th></tr></thead><tbody>{rows}</tbody></table></div>{timeline}<div class="scroll"><table><thead><tr><th>Variante</th><th>Clase</th><th>Aciertos</th><th>Omisiones</th><th>Sin pareja</th><th>PrecisiÃ³n</th><th>Sensibilidad</th><th>F1</th></tr></thead><tbody>{perclass}</tbody></table></div><h3>Jugadas omitidas</h3><div class="scroll"><table><tbody>{misses}</tbody></table></div></section>')
        baseline = next((v for v in local if v['key'] == 'x2'))
        line_ids = {s['id'] for s in truth if s['play_type'] == 'line'}
        scrum_ids = {s['id'] for s in truth if s['play_type'] == 'scrum'}
        changes = [dict(variant=v['key'], line_recovered=sorted(matches[v['key']] - matches['x2'] & line_ids), line_lost=sorted(matches['x2'] - matches[v['key']] & line_ids), scrum_lost=sorted(matches['x2'] - matches[v['key']] & scrum_ids), review_delta_s=v['review']['unique_flagged_s'] - baseline['review']['unique_flagged_s']) for v in local if v['key'].startswith('line')]
        for dest in (HERE, WEB):
            save(dest / f'changes_{date}.json', changes)
        change_rows = ''
        for c in changes:
            added = [s for s in truth if s['id'] in c['line_recovered']]
            times = ', '.join((p.tm(s['start_s']) + 'â€“' + p.tm(s['end_s']) for s in added)) or 'Ninguno'
            delta = c['review_delta_s']
            delta_text = ('+' if delta >= 0 else 'âˆ’') + p.tm(abs(delta))
            change_rows += f"<tr><th>{c['variant']}</th><td>{len(c['line_recovered'])}</td><td>{len(c['line_lost'])}</td><td>{len(c['scrum_lost'])}</td><td>{delta_text}</td><td>{times}</td></tr>"
        sections.append(f'<h3>Cambios frente a M1Ã—2 Â·{date_label}</h3><div class="scroll"><table data-changes="{date}"><thead><tr><th>Variante</th><th>Lines recuperados</th><th>Lines perdidos</th><th>Scrums perdidos</th><th>RevisiÃ³n adicional</th><th>Lines recuperados: tiempo del video</th></tr></thead><tbody>{change_rows}</tbody></table></div>')
        peer_changes = []
        for cost in (2, 4, 8):
            key = f'line{cost}'
            peer = 'x2' if cost == 2 else f'old_line{cost}'
            new = next((v for v in local if v['key'] == key))
            previous = next((v for v in local if v['key'] == peer))
            peer_changes.append(dict(variant=key, peer=peer, line_recovered=sorted(matches[key] - matches[peer] & line_ids), line_lost=sorted(matches[peer] - matches[key] & line_ids), scrum_lost=sorted(matches[peer] - matches[key] & scrum_ids), review_delta_s=new['review']['unique_flagged_s'] - previous['review']['unique_flagged_s']))
        for dest in (HERE, WEB):
            save(dest / f'peer_changes_{date}.json', peer_changes)
        peer_rows = ''
        for c in peer_changes:
            delta = c['review_delta_s']
            text = ('+' if delta >= 0 else 'âˆ’') + p.tm(abs(delta))
            peer_rows += f"<tr><th>{c['variant']}:scrum1 frente a scrum2</th><td>{len(c['line_recovered'])}</td><td>{len(c['line_lost'])}</td><td>{len(c['scrum_lost'])}</td><td>{text}</td></tr>"
        sections.append(f'<h3>Efecto de quitar el costo extra de scrum Â·{date_label}</h3><div class="scroll"><table data-peers="{date}"><thead><tr><th>ComparaciÃ³n emparejada</th><th>Lines recuperados</th><th>Lines perdidos</th><th>Scrums perdidos</th><th>Cambio de revisiÃ³n</th></tr></thead><tbody>{peer_rows}</tbody></table></div>')
    result = dict(status='complete', variants=summaries, comparisons=comparisons, protocol=protocol, training=training, evaluation_timing_amendment=read(HERE / 'evaluation_amendment.json') if _resolve_model_artifact(HERE / 'evaluation_amendment.json').exists() else None, peer_changes={j['date']: read(HERE / f"peer_changes_{j['date']}.json") for j in JOBS}, reference_kinds={j['date']: j['reference_kind'] for j in JOBS}, changes={j['date']: read(HERE / f"changes_{j['date']}.json") for j in JOBS})
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>M1 Â· mayor peso de line</title><style>body{font:16px/1.5 system-ui;max-width:1500px;margin:28px auto;padding:0 18px;color:#243447}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd5dd}thead{background:#edf3f8}aside{padding:16px;background:#fff0d6}.timeline svg{display:block;width:100%;height:26px}.timeline{margin:10px 0}section{margin:40px 0}a{color:#126b84}</style><h1>M1: peso adicional sÃ³lo en la fila de line</h1><p>ComparaciÃ³n completa: original y anteriores line2/4/8 con scrum2, frente a nuevos line2/4/8 con otro1 y scrum1. Se elimina sÃ³lo el multiplicador extra de scrum; conserva pÃ©rdida normal y balanceo de clases. Pesos balanceados por frecuencia y normalizados a media1, en red y RF. Mismos8796ejemplos amateur, inicializaciÃ³nM0/H2, semilla42,batch16,Adam1e-4 y5Ã©pocas fijas. Mayor costo sobre todos los errores de un line real; no se entrenÃ³ con los lines omitidos de los profesionales.</p><aside>16/08: las anotaciones manuales humanas.23/08 y Mundial: inventarios diagnÃ³sticos automÃ¡ticos parciales; precisiÃ³n y sensibilidad exploratorias sobre ese inventario, no certificadas para el partido entero. Alertas sin pareja y tiempo fuera del inventario no son falsos positivos confirmados. Parejas Ãºnicas de misma clase y solapamientoâ‰¥1s. RevisiÃ³n: uniÃ³n de intervalos a1Ã—, sin pausas. Todos los partidos se procesan con cachÃ© nativo completo, sin nueva extracciÃ³n, conservando recorte, ventanas7s/1s y agrupaciÃ³n argmax. Son experimentos sobre partidos ya observados, no una prueba independiente ciega.</aside><p><a href="results.json">Datos y entrenamiento</a> Â· <a href="verification.json">VerificaciÃ³n independiente</a></p>' + ''.join(sections) + '</html>'
    verification = dict(status='PASS', independent_matching_and_review=checks, training_source_hashes_unchanged=True, cache_stats_unchanged=True, native_comparison_roundtrip=True, configurations=21, extractors_executed=0)
    for dest in (HERE, WEB):
        save(dest / 'results.json', result)
        save(dest / 'verification.json', verification)
        (dest / 'index.html').write_text(doc, encoding='utf-8')
    print('Published21variants across3professional full matches', flush=True)
if __name__ == '__main__':
    try:
        main()
        subprocess.run(['node', str(HERE / 'verify_web.cjs')], cwd=str(ROOT), creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    except Exception:
        save(HERE / 'publication_error.json', dict(error=traceback.format_exc()))
        raise
