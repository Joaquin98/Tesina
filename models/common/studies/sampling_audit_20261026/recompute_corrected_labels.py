"""Versioned reevaluation: overlay manual corrections on complete labels."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
from collections import Counter
import copy, hashlib, json, re, shutil, urllib.request, urllib.error
from human_reference_comparison import same_protocol
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
REVISIONS = HERE / 'human_revisions'
WEB = ROOT / 'src/web/static/reports/professional_rugby_human_20261026'
BASE_ID = '20261026_212724_97d95f'
CORRECTION_ID = '20261026_230503_982e1e'
ORDER = [('original', 'Original'), ('every5', 'Cada 5'), ('every10', 'Cada 10'), ('every25', 'Cada 25')]

def read(path):
    return json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8-sig'))

def digest(path):
    return hashlib.sha256(_resolve_model_artifact(path).read_bytes()).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def pct(value):
    return f'{100 * value:.1f}%'.replace('.', ',')

def metric_table(report):
    rows = ''
    for key, title in ORDER:
        for label in ('line', 'scrum', 'total'):
            m = report['variants'][key]['metrics'][label]
            rows += f"<tr><td>{title}</td><td>{label}</td><td>{m['tp']}/{m['alerts']} ({pct(m['precision'])})</td><td>{m['tp']}/{m['reference_segments']} ({pct(m['recall'])})</td><td>{m['fp']}</td><td>{m['fn']}</td></tr>"
    return '<div class="scroll"><table data-table="corrected-human"><thead><tr><th>Versión</th><th>Clase</th><th>Precisión</th><th>Sensibilidad</th><th>Alertas sin pareja</th><th>Segmentos omitidos</th></tr></thead><tbody>' + rows + '</tbody></table></div>'

def append_corrected_reference_link(document, _original_report=None):
    pointer = REVISIONS / 'latest.json'
    if not _resolve_model_artifact(pointer).exists():
        return document
    latest = read(pointer)
    report = read(REVISIONS / latest['revision_id'] / 'metrics.json')
    section = '<section id="latest-corrected-labels"><h2>Estadísticas actualizadas con las anotaciones manuales corregidas</h2><p>Referencia consolidada: 881 bloques completos, ' + str(report['block_counts']['line']) + ' line, ' + str(report['block_counts']['scrum']) + ' scrum y ' + str(report['block_counts']['other']) + ' otro. Se aplicaron sólo las decisiones explícitas de la sesión de correcciones; sus bloques vacíos conservan la etiqueta previa.</p>' + metric_table(report) + '<p><a href="' + latest['report_url'] + '">Abrir informe actualizado y detalle de cambios</a> · <a href="/compare?comparison=' + report['comparison_id'] + '">Abrir comparación con la referencia corregida</a></p><aside>Las secciones que siguen conservan las evaluaciones anteriores como historial. Para las estadísticas actuales usá la tabla corregida de arriba.</aside></section>'
    document = re.sub('<section id="latest-corrected-labels">.*?</section>', '', document, flags=re.S)
    return document.replace('</h1>', '</h1>' + section, 1)

def main():
    import os
    session_dir = ROOT / 'data/active_learning_review/sessions'
    paths = [session_dir / f'review_session_{sid}.json' for sid in (BASE_ID, CORRECTION_ID)]
    hashes = {str(p): digest(p) for p in paths}
    base, correction = map(read, paths)
    assert os.path.samefile(base['video_path'], correction['video_path'])
    blocks = copy.deepcopy(base['candidates'])
    assert len(blocks) == len(correction['candidates']) == 881
    assert all((b['final_label'] in {'line', 'scrum', 'other'} for b in blocks))
    provenance_by_block = [BASE_ID] * len(blocks)
    changes = []
    for patch in correction['candidates']:
        label = patch.get('final_label')
        if label is None:
            continue
        assert label in {'line', 'scrum', 'other'}
        i = patch['candidate_index']
        b = blocks[i]
        assert b['candidate_index'] == i and b['start_time'] == patch['start_time'] and (b['end_time'] == patch['end_time'])
        if b['final_label'] != label:
            changes.append(dict(block_index=i, clip_number=i + 1, start_s=b['start_time'], end_s=b['end_time'], old_label=b['final_label'], new_label=label))
        b['final_label'] = label
        b['reviewed_at'] = patch.get('reviewed_at')
        provenance_by_block[i] = CORRECTION_ID
    labels_hash = hashlib.sha256(json.dumps([(b['candidate_index'], b['final_label']) for b in blocks], separators=(',', ':')).encode()).hexdigest()
    revision_id = 'labels_' + labels_hash[:12]
    dest = REVISIONS / revision_id
    dest.mkdir(parents=True, exist_ok=True)
    web = WEB / 'revisions' / revision_id
    web.mkdir(parents=True, exist_ok=True)
    truth = []
    for b in blocks:
        label = b['final_label']
        if label == 'other':
            continue
        if truth and truth[-1]['play_type'] == label and (truth[-1]['end_s'] == b['start_time']):
            truth[-1]['end_s'] = b['end_time']
            truth[-1]['blocks'].append(b['candidate_index'])
        else:
            truth.append(dict(id=f'corrected_{label}_{len(truth) + 1}', play_type=label, start_s=b['start_time'], end_s=b['end_time'], blocks=[b['candidate_index']]))
    previous = read(HERE / 'game1_cached5/comparison.json')
    records, variants = ({}, {})
    for key, title in ORDER:
        source = ROOT / f"data/results/result_{previous['variants'][key]['result_id']}.json"
        hashes[str(source)] = digest(source)
        payload = read(source)
        metrics = same_protocol(truth, payload['detections'])
        for m in metrics.values():
            m.update(fp=m['alerts'] - m['tp'], fn=m['reference_segments'] - m['tp'], f1=2 * m['tp'] / (m['alerts'] + m['reference_segments']))
        variants[key] = dict(metrics=metrics, result_id=payload['result_id'])
        records[title] = payload
    counts = dict(Counter((b['final_label'] for b in blocks)))
    report_url = f'/static/reports/professional_rugby_human_20261026/revisions/{revision_id}/index.html'
    comparison_id = '20261026_argentina_allblacks_20250816_human_corrected_' + labels_hash[:12]
    report = dict(status='complete', revision_id=revision_id, session_ids=[BASE_ID, CORRECTION_ID], comparison_id=comparison_id, block_counts=counts, reviewed_blocks=881, reference_segments=truth, variants=variants, explicit_corrections=changes, labels_sha256=labels_hash, previous_metrics=previous['variants'], source_video=base['video_path'], method='Base completa más decisiones no nulas de la sesión de correcciones; vacíos conservan la etiqueta anterior. Unión de bloques contiguos de misma clase; coincidencia uno a uno de misma clase con al menos1s de solapamiento. Mismas predicciones congeladas, sin inferencia ni ajuste de umbrales.', provenance=[dict(path=p, sha256=h) for p, h in hashes.items()])
    dump(dest / 'metrics.json', report)
    frozen_labels = [dict(block_index=b['candidate_index'], start_time=b['start_time'], end_time=b['end_time'], final_label=b['final_label'], source_session_id=provenance_by_block[i]) for i, b in enumerate(blocks)]
    dump(dest / 'labels.json', dict(revision_id=revision_id, session_ids=report['session_ids'], labels=frozen_labels, labels_sha256=labels_hash))
    pinned = dict(reference_session_pinned=True, composite_reference=True, reference_id=revision_id, session_id=BASE_ID, source_session_ids=report['session_ids'], reference_sha256=labels_hash, labels=frozen_labels, summary=dict(unit='blocks', reviewed=881, total_candidates=881, pending=0, final_counts=counts))
    for (key, _), (title, payload) in zip(ORDER, records.items()):
        payload['manual_comparison'] = copy.deepcopy(pinned)
        payload['human_evaluation'] = variants[key]['metrics']
        payload['metadata']['human_evaluation_report'] = report_url
    history = dict(id=comparison_id, video_path=base['video_path'], video_name='Argentina–NZ · 16/08/2025 · etiquetas corregidas', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={title: dict(total=len(p['detections']), by_type=dict(Counter((d['play_type'] for d in p['detections']))), avg_confidence=p['statistics']['avg_confidence']) for title, p in records.items()})
    endpoint = 'http://127.0.0.1:5000/api/comparison-history'
    try:
        with urllib.request.urlopen(endpoint + '/' + comparison_id) as response:
            assert json.load(response)['comparison']['results'] == records
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        request = urllib.request.Request(endpoint, data=json.dumps(history, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request) as response:
            assert json.load(response)['success']
    change_rows = ''.join((f"<tr><td>{c['clip_number']}</td><td>{int(c['start_s']) // 60:02d}:{int(c['start_s']) % 60:02d}–{int(c['end_s']) // 60:02d}:{int(c['end_s']) % 60:02d}</td><td>{c['old_label']} → {c['new_label']}</td></tr>" for c in changes))
    document = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Argentina–NZ · etiquetas corregidas</title><style>body{font:16px system-ui;line-height:1.6;max-width:1250px;margin:30px auto;padding:0 18px;color:#243447}table{width:100%;border-collapse:collapse;margin:22px 0}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow:auto}aside{background:#edf3f8;padding:18px}a{color:#126b84}</style><h1>Argentina–NZ · estadísticas con las anotaciones manuales corregidas</h1><p>Se integraron ' + str(len(changes)) + ' correcciones explícitas sobre tus881 etiquetas completas. Los872 bloques vacíos de la sesión de correcciones conservan la etiqueta anterior; no se interpretan como otro.</p><aside>Referencia actual:47 clips de line,43 de scrum y791 de otro. Al unir bloques contiguos:32 segmentos de line y15 de scrum,47 en total. Misma clase, al menos1s de solapamiento y emparejamiento uno a uno. Las predicciones de las cuatro versiones no cambian.</aside>' + metric_table(report) + '<p>Con esta referencia, el original encuentra42/47 y cada5 encuentra41/47. Cada5 ya no iguala toda la cobertura del original: omite un segmento adicional de line. Las etiquetas no distinguen vivo/replay/rearmados; los segmentos no equivalen necesariamente a jugadas únicas de las estadísticas oficiales.</p><p><a href="/compare?comparison=' + comparison_id + '">Ver cuatro modelos y las anotaciones manuales corregidas</a> · <a href="metrics.json">Métricas y trazabilidad</a> · <a href="labels.json">Referencia consolidada</a></p><h2>Correcciones aplicadas</h2><div class="scroll"><table data-table="label-corrections"><thead><tr><th>Clip</th><th>Tiempo del video</th><th>Cambio</th></tr></thead><tbody>' + change_rows + '</tbody></table></div><p><a href="/clip-generator?session_id=' + BASE_ID + '">Sesión original completa</a> · <a href="/clip-generator?session_id=' + CORRECTION_ID + '">Sesión donde hiciste las correcciones</a>. Ambas se conservan intactas.</p></html>'
    (dest / 'index.html').write_text(document, encoding='utf-8')
    for name in ('metrics.json', 'labels.json', 'index.html'):
        shutil.copyfile(dest / name, web / name)
    REVISIONS.mkdir(exist_ok=True)
    pointer = dict(revision_id=revision_id, report_url=report_url, comparison_id=comparison_id)
    dump(REVISIONS / 'latest.json', pointer)
    for path in [HERE / 'human_validation/index.html', WEB / 'index.html']:
        path.write_text(append_corrected_reference_link(_resolve_model_artifact(path).read_text(encoding='utf-8')), encoding='utf-8')
    for p, h in hashes.items():
        assert digest(Path(p)) == h, 'Source changed during evaluation'
    print(json.dumps(dict(status=report['status'], revision_id=revision_id, corrections=len(changes), block_counts=counts, segment_counts=dict(Counter((t['play_type'] for t in truth))), metrics={key: v['metrics'] for key, v in variants.items()}, comparison_url='http://127.0.0.1:5000/compare?comparison=' + comparison_id, report_url='http://127.0.0.1:5000' + report_url), ensure_ascii=True))
if __name__ == '__main__':
    main()
