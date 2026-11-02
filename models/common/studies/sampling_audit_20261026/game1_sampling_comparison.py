"""Publish original/every25 comparison using the unchanged human reference."""
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
import json, sys, runpy, hashlib, html, shutil, urllib.request, urllib.error
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
OUT = HERE / 'game1_cached25'
WEB = ROOT / 'src/web/static/reports/professional_rugby_human_20261026'
IDS = {'original': '20261026_argentina_allblacks_20250816_argmax_human_full', 'every25': '20261026_argentina_allblacks_20250816_argmax_human_every25'}
COMPARISON_ID = '20261026_argentina_allblacks_20250816_human_original_vs_every25'

def read(p):
    return json.loads(_resolve_model_artifact(p).read_text(encoding='utf-8-sig'))

def dump(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def pct(v):
    return f'{v * 100:.1f}%'.replace('.', ',')

def clock(v):
    return f'{int(v) // 60:02d}:{int(v) % 60:02d}'

def section(report):
    rows = ''
    for key, title in [('original', 'Original'), ('every25', 'Cada 25')]:
        for label in ('line', 'scrum', 'total'):
            m = report['variants'][key]['metrics'][label]
            rows += f"<tr><td>{title}</td><td>{label}</td><td>{m['tp']}/{m['alerts']} ({pct(m['precision'])})</td><td>{m['tp']}/{m['reference_segments']} ({pct(m['recall'])})</td><td>{m['fp']}</td><td>{m['fn']}</td></tr>"
    missed = ''
    for t in report['reference_segments']:
        a, b = (t['id'] in report['variants']['original']['detected_ids'], t['id'] in report['variants']['every25']['detected_ids'])
        if a != b:
            missed += f"<tr><td>{t['play_type']}</td><td>{clock(t['start_s'])}–{clock(t['end_s'])}</td><td>{('Sí' if a else 'No')}</td><td>{('Sí' if b else 'No')}</td></tr>"
    elapsed = report['cached_execution']['wall_elapsed_s']
    return '<section id="cada25"><h2>Original frente a uno de cada 25 · tus mismas etiquetas</h2>\n<p>Ambas versiones se evalúan con los mismos 26 segmentos de line y 15 de scrum. Misma clase, al menos 1 segundo de solapamiento y emparejamiento uno a uno. Se conserva el filtro posterior al inicio de partido (188,2 s) utilizado por el original, sin mover ventanas ni ajustar umbrales a las anotaciones manuales.</p>\n<div class="scroll"><table data-table="sampling-human"><thead><tr><th>Versión</th><th>Clase</th><th>Precisión</th><th>Sensibilidad</th><th>Alertas sin pareja</th><th>Segmentos omitidos</th></tr></thead><tbody>' + rows + '</tbody></table></div>\n<p>El original genera 105 alertas y cada 25 genera 83. Cada 25 mejora la precisión total de 34,3% a 38,6%, pero reduce la sensibilidad de 87,8% a 78,0%. La pérdida neta está en line: pasa de 22/26 a 18/26. Scrum conserva 14/15.</p>\n<h3>Segmentos cuya detección cambia</h3><div class="scroll"><table data-table="changed-segments"><thead><tr><th>Clase</th><th>Tiempo del video</th><th>Original</th><th>Cada 25</th></tr></thead><tbody>' + missed + '</tbody></table></div>\n<p>La prueba de cada 25 tardó <strong>' + clock(elapsed) + ' desde caché</strong>, incluidos carga y controles. Se seleccionaron 6.168 de 154.161 posiciones por extractor, interpolando vectores para las mismas 6.160 ventanas. No se ejecutaron YOLO ni ResNet ni se midió un procesamiento nuevo desde video. Se prohíben lecturas RGB descartadas y la geometría descartada se reemplaza por NaN antes de reconstruirla.</p>\n<p>Las etiquetas cubren toda la transmisión y no distinguen repeticiones ni rearmados. Los segmentos contiguos no son necesariamente jugadas únicas en vivo. El original se conserva congelado; la nueva versión usa exactamente los mismos pesos.</p>\n<p><a href="/compare?comparison=' + COMPARISON_ID + '">Abrir original, cada 25 y etiquetado manual en la herramienta</a> · <a href="/viewer?result_id=' + IDS['every25'] + '">Ver detecciones cada 25</a> · <a href="comparison_every25.json">Comparación completa y trazabilidad</a></p></section>'

def append_sampling_comparison(document, human):
    path = OUT / 'comparison.json'
    if not _resolve_model_artifact(path).exists() or 'id="cada25"' in document:
        return document
    report = read(path)
    assert report['session_id'] == human['session_id']
    assert report['variants']['original']['metrics'] == human['metrics']
    return document.replace('<h2>Segmentos humanos sin detección', section(report) + '<h2>Segmentos humanos sin detección')

def main():
    sys.path.insert(0, str(source_directory(HERE)))
    baseline = runpy.run_path(str(_resolve_model_artifact(HERE / 'evaluate_human_labels.py')))
    base_report = baseline['report']
    match, truth = (baseline['match'], baseline['truth'])
    original_path = ROOT / 'data/results/result_20261025_argentina_allblacks_20250816_argmax.json'
    original = read(original_path)
    sparse_events = read(OUT / 'events_argmax.json')
    completion = read(OUT / 'completion.json')
    assert completion['success'] and completion['windows'] == 6160 and (completion['extractors_executed'] == 0)
    variants = {}
    for key, events in [('original', original['detections']), ('every25', sparse_events)]:
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
        variants[key] = dict(metrics=metrics, matches=pairs, detected_ids=detected, missed_segments=missed, result_id=IDS[key], viewer_url=f'/viewer?result_id={IDS[key]}')
    assert variants['original']['metrics'] == base_report['metrics']
    report = dict(status='complete', session_id=base_report['session_id'], match_date='2025-08-16', reference_segments=truth, method=base_report['method'], variants=variants, cached_execution=completion, baseline_validation=read(OUT / 'baseline_validation.json'), sampling=read(OUT / 'config.json'), labels_sha256=hashlib.sha256(_resolve_model_artifact(baseline['SESSION']).read_bytes()).hexdigest(), source_original_sha256=hashlib.sha256(_resolve_model_artifact(original_path).read_bytes()).hexdigest(), provenance=base_report['provenance'], gained_ids=sorted(set(variants['every25']['detected_ids']) - set(variants['original']['detected_ids'])), lost_ids=sorted(set(variants['original']['detected_ids']) - set(variants['every25']['detected_ids'])), comparison_id=COMPARISON_ID)
    dump(OUT / 'comparison.json', report)
    alias = baseline['s']['video_path']
    records, comparison_counts = ({}, {})
    for key, title in [('original', 'Original · todos los fotogramas'), ('every25', 'YOLO y ResNet · cada 25')]:
        events = original['detections'] if key == 'original' else sparse_events
        windows = original['window_predictions'] if key == 'original' else [json.loads(x) for x in _resolve_model_artifact(OUT / 'windows.jsonl').read_text().splitlines()]
        counts = Counter((e['play_type'] for e in events))
        stats = dict(avg_confidence=sum((e['confidence'] for e in events)) / len(events), avg_duration=sum((e['duration'] for e in events)) / len(events), by_type={c: dict(count=v, total_duration=sum((e['duration'] for e in events if e['play_type'] == c))) for c, v in counts.items()})
        metadata = dict(original['metadata'], confidence_threshold=0.0, line_confidence_threshold=0.0, experimental_sampling_stride=1 if key == 'original' else 25, experimental_interpolation=key == 'every25', human_evaluation_report='/static/reports/professional_rugby_human_20261026/index.html#cada25', source_result_id='20261025_argentina_allblacks_20250816_argmax' if key == 'original' else IDS[key], sampling_note='Características dispersas interpoladas; sólo ejecución del clasificador desde caché.' if key == 'every25' else 'Copia de las predicciones originales congeladas; no se volvió a inferir el original.')
        payload = dict(status='success', result_id=IDS[key], video_path=alias, model_name=f'Argentina 24–41 NZ · 16/08/2025 · {title} · alta resolución', detections=events, primary_detections=events, window_predictions=windows, event_post_processing=original['event_post_processing'], statistics=stats, summary=dict(total_detections=len(events), total_windows=len(windows), counts=dict(counts)), metadata=metadata, human_evaluation=variants[key]['metrics'])
        target = ROOT / f'data/results/result_{IDS[key]}.json'
        if _resolve_model_artifact(target).exists():
            existing = read(target)
            assert existing['detections'] == events and existing['window_predictions'] == windows
        else:
            with _resolve_model_artifact(target).open('x', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        records[title] = payload
        comparison_counts[title] = dict(total=len(events), by_type=dict(counts), avg_confidence=stats['avg_confidence'])
    history = dict(id=COMPARISON_ID, video_path=alias, video_name='Argentina–NZ · 16/08/2025 · original vs cada 25', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison=comparison_counts)
    endpoint = 'http://127.0.0.1:5000/api/comparison-history'
    try:
        with urllib.request.urlopen(endpoint + '/' + COMPARISON_ID) as response:
            existing = json.load(response)['comparison']
            assert existing['results'] == records
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        req = urllib.request.Request(endpoint, data=json.dumps(history, ensure_ascii=False).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req) as response:
            assert json.load(response)['success']
    doc_path = HERE / 'human_validation/index.html'
    document = append_sampling_comparison(_resolve_model_artifact(doc_path).read_text(encoding='utf-8'), base_report)
    doc_path.write_text(document, encoding='utf-8')
    (WEB / 'index.html').write_text(document, encoding='utf-8')
    shutil.copyfile(OUT / 'comparison.json', WEB / 'comparison_every25.json')
    for path, expected in [(baseline['SESSION'], report['labels_sha256']), (original_path, report['source_original_sha256'])]:
        assert hashlib.sha256(_resolve_model_artifact(path).read_bytes()).hexdigest() == expected
    print(json.dumps(dict(status=report['status'], variants={k: v['metrics'] for k, v in variants.items()}, gained=report['gained_ids'], lost=report['lost_ids'], cached_wall_s=completion['wall_elapsed_s'], comparison_url=f'http://127.0.0.1:5000/compare?comparison={COMPARISON_ID}'), ensure_ascii=True))
if __name__ == '__main__':
    main()
