"""Evaluate cached every10 and publish all three frozen variants."""
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
from collections import Counter
import json, sys, runpy, hashlib, copy, shutil, urllib.request, urllib.error
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
OUT = HERE / 'game1_cached10'
WEB = ROOT / 'src/web/static/reports/professional_rugby_human_20261026'
RESULT_ID = '20261026_argentina_allblacks_20250816_argmax_human_every10'
COMPARISON_ID = '20261026_argentina_allblacks_20250816_human_original_vs_every10_vs_every25'
ORDER = [('original', 'Original'), ('every10', 'Cada 10'), ('every25', 'Cada 25')]

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def dump(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')

def pct(v):
    return f'{100 * v:.1f}%'.replace('.', ',')

def clock(v):
    return f'{int(v) // 60:02d}:{int(v) % 60:02d}'

def section(report):
    rows, changes, timing = ('', '', '')
    for key, title in ORDER:
        for label in ('line', 'scrum', 'total'):
            m = report['variants'][key]['metrics'][label]
            rows += f"<tr><td>{title}</td><td>{label}</td><td>{m['tp']}/{m['alerts']} ({pct(m['precision'])})</td><td>{m['tp']}/{m['reference_segments']} ({pct(m['recall'])})</td><td>{m['fp']}</td><td>{m['fn']}</td></tr>"
    for t in report['reference_segments']:
        hits = [t['id'] in report['variants'][key]['detected_ids'] for key, _ in ORDER]
        if len(set(hits)) > 1:
            changes += f"<tr><td>{t['play_type']}</td><td>{clock(t['start_s'])}–{clock(t['end_s'])}</td>" + ''.join(('<td>' + ('Sí' if v else 'No') + '</td>' for v in hits)) + '</tr>'
    for key, title in ORDER[1:]:
        c = report['cached_execution'][key]
        timing += f"<tr><td>{title}</td><td>{clock(c['wall_elapsed_s'])}</td><td>{c['selected_frames']:,}</td><td>{c['windows']}</td><td>{c['extractors_executed']}</td></tr>"
    totals = report['variants']['every10']['metrics']['total']
    return '<section id="cada10"><h2>Original, uno de cada 10 y uno de cada 25 · las anotaciones manuales</h2>\n<p>Las tres versiones usan los mismos pesos y se evalúan con tus 881 clips completos: 26 segmentos de line y 15 de scrum al unir bloques contiguos de la misma clase. Cada pareja debe coincidir en clase y tener al menos 1 segundo de solapamiento. Cada alerta y segmento se cuentan una sola vez. No se ajustaron umbrales con estas etiquetas.</p>\n<div class="scroll"><table data-table="threeway-human"><thead><tr><th>Versión</th><th>Clase</th><th>Precisión</th><th>Sensibilidad</th><th>Alertas sin pareja</th><th>Segmentos omitidos</th></tr></thead><tbody>' + rows + '</tbody></table></div>\n<p>Cada 10 encuentra ' + str(totals['tp']) + ' de 41 segmentos con ' + str(totals['alerts']) + ' alertas; el original encuentra 36/41 con 105 alertas y cada 25 encuentra 32/41 con 83 alertas. Las etiquetas describen segmentos de la transmisión y no distinguen jugadas en vivo, repeticiones o rearmados.</p>\n<h3>Segmentos cuya cobertura cambia entre versiones</h3><div class="scroll"><table data-table="threeway-changes"><thead><tr><th>Clase</th><th>Tiempo del video</th><th>Original</th><th>Cada 10</th><th>Cada 25</th></tr></thead><tbody>' + changes + '</tbody></table></div>\n<h3>Tiempo medido desde caché</h3><div class="scroll"><table data-table="threeway-timing"><thead><tr><th>Versión</th><th>Tiempo desde caché</th><th>Observaciones seleccionadas por extractor</th><th>Ventanas</th><th>Extractores ejecutados</th></tr></thead><tbody>' + timing + '</tbody></table></div>\n<p>Estos tiempos incluyen carga, controles, interpolación y clasificador. No representan procesamiento desde video ni permiten medir el ahorro de extracción. Se usaron sólo las características seleccionadas; las lecturas RGB descartadas están prohibidas y la geometría descartada se reemplaza por NaN. Se conserva la grilla temporal 7 s/1 s y el mismo filtro posterior al inicio del partido (188,2 s). No se repitió la inferencia del original ni de cada 25.</p>\n<p><a href="/compare?comparison=' + COMPARISON_ID + '">Ver original, cada 10, cada 25 y las anotaciones manuales en la herramienta</a> · <a href="/viewer?result_id=' + RESULT_ID + '">Abrir detecciones cada 10</a> · <a href="comparison_every10_every25.json">Métricas, coincidencias y trazabilidad</a></p></section>'

def append_threeway_comparison(document, human):
    path = OUT / 'comparison.json'
    if not _resolve_model_artifact(path).exists() or 'id="cada10"' in document:
        return document
    report = read(path)
    assert report['session_id'] == human['session_id']
    assert report['variants']['original']['metrics'] == human['metrics']
    return document.replace('<h2>Segmentos humanos sin detección', section(report) + '<h2>Segmentos humanos sin detección')

def main():
    sys.path.insert(0, str(source_directory(HERE)))
    baseline = runpy.run_path(str(_resolve_model_artifact(HERE / 'evaluate_human_labels.py')))
    base_report, truth, match = (baseline['report'], baseline['truth'], baseline['match'])
    previous_path = HERE / 'game1_cached25/comparison.json'
    previous_hash = hashlib.sha256(_resolve_model_artifact(previous_path).read_bytes()).hexdigest()
    previous = read(previous_path)
    assert previous['variants']['original']['metrics'] == base_report['metrics']
    assert previous['reference_segments'] == truth
    assert previous['labels_sha256'] == hashlib.sha256(_resolve_model_artifact(baseline['SESSION']).read_bytes()).hexdigest()
    completion = read(OUT / 'completion.json')
    assert completion['success'] and completion['windows'] == 6160 and (completion['extractors_executed'] == 0)
    config = read(OUT / 'config.json')
    assert config['stride_rgb'] == config['stride_pose'] == 10
    events = read(OUT / 'events_argmax.json')
    metrics, pairs, detected, missed = ({}, [], [], [])
    for label in ('line', 'scrum'):
        ts = [t for t in truth if t['play_type'] == label]
        ps = [dict(id=i + 1, play_type=p['play_type'], start_s=p['start_time'], end_s=p['end_time']) for i, p in enumerate(events) if p['play_type'] == label]
        matches = match(ts, ps, 1.0)
        tp, nt, npred = (len(matches), len(ts), len(ps))
        metrics[label] = dict(tp=tp, fp=npred - tp, fn=nt - tp, reference_segments=nt, alerts=npred, precision=tp / npred, recall=tp / nt, f1=2 * tp / (nt + npred))
        used = {i for i, _, _ in matches}
        detected += [t['id'] for i, t in enumerate(ts) if i in used]
        missed += [t for i, t in enumerate(ts) if i not in used]
        pairs += [dict(reference_id=ts[i]['id'], prediction_id=ps[j]['id'], overlap_s=seconds) for i, j, seconds in matches]
    tp = sum((m['tp'] for m in metrics.values()))
    nt = len(truth)
    npred = len(events)
    metrics['total'] = dict(tp=tp, fp=npred - tp, fn=nt - tp, reference_segments=nt, alerts=npred, precision=tp / npred, recall=tp / nt, f1=2 * tp / (nt + npred))
    assert len({p['prediction_id'] for p in pairs}) == tp
    variants = copy.deepcopy(previous['variants'])
    variants['every10'] = dict(metrics=metrics, matches=pairs, detected_ids=detected, missed_segments=missed, result_id=RESULT_ID, viewer_url=f'/viewer?result_id={RESULT_ID}')
    report = dict(status='complete', session_id=base_report['session_id'], match_date='2025-08-16', reference_segments=truth, method=base_report['method'], variants=variants, cached_execution=dict(every10=completion, every25=previous['cached_execution']), baseline_validation=read(OUT / 'baseline_validation.json'), sampling=config, labels_sha256=previous['labels_sha256'], source_original_sha256=previous['source_original_sha256'], provenance=previous['provenance'], previous_comparison_sha256=previous_hash, comparison_id=COMPARISON_ID, changes={key: dict(gained_ids=sorted(set(detected) - set(variants[key]['detected_ids'])), lost_ids=sorted(set(variants[key]['detected_ids']) - set(detected))) for key in ('original', 'every25')})
    dump(OUT / 'comparison.json', report)
    original_payload = read(ROOT / f"data/results/result_{previous['variants']['original']['result_id']}.json")
    sparse25_payload = read(ROOT / f"data/results/result_{previous['variants']['every25']['result_id']}.json")
    payload = copy.deepcopy(sparse25_payload)
    windows = [json.loads(s) for s in _resolve_model_artifact(OUT / 'windows.jsonl').read_text().splitlines()]
    counts = Counter((e['play_type'] for e in events))
    stats = dict(avg_confidence=sum((e['confidence'] for e in events)) / len(events), avg_duration=sum((e['duration'] for e in events)) / len(events), by_type={c: dict(count=v, total_duration=sum((e['duration'] for e in events if e['play_type'] == c))) for c, v in counts.items()})
    payload.update(result_id=RESULT_ID, model_name='Argentina 24–41 NZ · 16/08/2025 · YOLO y ResNet · cada 10 · alta resolución', detections=events, primary_detections=events, window_predictions=windows, statistics=stats, summary=dict(total_detections=len(events), total_windows=len(windows), counts=dict(counts)), human_evaluation=metrics)
    payload['metadata'].update(experimental_sampling_stride=10, source_result_id=RESULT_ID, human_evaluation_report='/static/reports/professional_rugby_human_20261026/index.html#cada10')
    target = ROOT / f'data/results/result_{RESULT_ID}.json'
    if _resolve_model_artifact(target).exists():
        existing = read(target)
        assert existing['detections'] == events and existing['window_predictions'] == windows
    else:
        with _resolve_model_artifact(target).open('x', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    records = {'Original · todos los fotogramas': original_payload, 'YOLO y ResNet · cada 10': payload, 'YOLO y ResNet · cada 25': sparse25_payload}
    history = dict(id=COMPARISON_ID, video_path=payload['video_path'], video_name='Argentina–NZ · 16/08/2025 · original vs cada 10 vs cada 25', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={name: dict(total=len(p['detections']), by_type=dict(Counter((e['play_type'] for e in p['detections']))), avg_confidence=p['statistics']['avg_confidence']) for name, p in records.items()})
    endpoint = 'http://127.0.0.1:5000/api/comparison-history'
    try:
        with urllib.request.urlopen(endpoint + '/' + COMPARISON_ID) as response:
            assert json.load(response)['comparison']['results'] == records
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        request = urllib.request.Request(endpoint, data=json.dumps(history, ensure_ascii=False).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request) as response:
            assert json.load(response)['success']
    doc_path = HERE / 'human_validation/index.html'
    document = append_threeway_comparison(_resolve_model_artifact(doc_path).read_text(encoding='utf-8'), base_report)
    doc_path.write_text(document, encoding='utf-8')
    (WEB / 'index.html').write_text(document, encoding='utf-8')
    shutil.copyfile(OUT / 'comparison.json', WEB / 'comparison_every10_every25.json')
    assert hashlib.sha256(_resolve_model_artifact(previous_path).read_bytes()).hexdigest() == previous_hash
    for p in report['provenance']:
        assert hashlib.sha256(_resolve_model_artifact(Path(p['path'])).read_bytes()).hexdigest() == p['sha256']
    print(json.dumps(dict(status=report['status'], every10=metrics, changes=report['changes'], cached_wall_s=completion['wall_elapsed_s'], comparison_url=f'http://127.0.0.1:5000/compare?comparison={COMPARISON_ID}'), ensure_ascii=True))
if __name__ == '__main__':
    main()
