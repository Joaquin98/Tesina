"""Verify and expose completed arms without claiming the whole sweep finished."""
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
import sys, json, html
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path[:0] = [str(source_directory(HERE)), str(ROOT)]
import models.M1.studies.m1_line_only_costs_20261028.publish as full
from models.M1.studies.m1_line_only_costs_20261028.run import read, save, sha
WEB = ROOT / 'src/web/static/reports/m1_line_only_costs_20261028'

def main():
    output = []
    sections = []
    for job in full.JOBS:
        date = job['date']
        ref = read(job['reference'])
        human = job['reference_kind'] == 'human'
        truth = ref['reference_segments'] if human else ref['events']
        if human:
            for source in ref['provenance']:
                assert sha(source['path']) == source['sha256']
        base = read(ROOT / f"data/results/result_{job['baseline_id']}.json")
        b = full.p.evaluate_events(truth, base['detections'])
        matched_base = {x['reference_id'] for x in b['assignments']}
        line_ids = {s['id'] for s in truth if s['play_type'] == 'line'}
        scrum_ids = {s['id'] for s in truth if s['play_type'] == 'scrum'}
        variants = []
        for cost in (2, 4, 8):
            folder = HERE / f'evaluation/{date}/line{cost}_scrum1'
            if not _resolve_model_artifact(folder / 'metrics.json').exists():
                continue
            saved = read(folder / 'metrics.json')
            events = read(folder / 'events.json')
            assert saved['reference_sha256'] == sha(job['reference'])
            for path, h in saved['model_hashes'].items():
                assert sha(path) == h
            v = full.p.evaluate_events(truth, events)
            assert v['metrics'] == saved['metrics']
            check = full.p.prior.independent_check(truth, events, v['metrics'], job['duration'])
            for k in ('unique_flagged_s', 'flagged_outside_true_segments_s'):
                assert abs(check[k] - v['review'][k]) < 1e-06
            matched = {x['reference_id'] for x in v['assignments']}
            item = dict(date=date, line_cost=cost, scrum_cost=1, reference_kind=job['reference_kind'], metrics=v['metrics'], review=v['review'], baseline_metrics=b['metrics'], baseline_review=b['review'], line_recovered=sorted(matched - matched_base & line_ids), line_lost=sorted(matched_base - matched & line_ids), scrum_lost=sorted(matched_base - matched & scrum_ids), review_delta_s=v['review']['unique_flagged_s'] - b['review']['unique_flagged_s'])
            variants.append(item)
            output.append(item)
        rows = ''
        t = full.p.tm
        pct = full.p.pct
        if variants:
            m = b['metrics']
            rows = f"<tr><th>M1Ã—2 Â· line2/scrum2</th><td>{m['line']['tp']}/{m['line']['reference_segments']}</td><td>â€”</td><td>â€”</td><td>{m['scrum']['tp']}/{m['scrum']['reference_segments']}</td><td>{pct(m['total']['precision'])}</td><td>{t(b['review']['unique_flagged_s'])}</td><td>â€”</td></tr>"
            for v in variants:
                m = v['metrics']
                delta = v['review_delta_s']
                change = ('+' if delta >= 0 else 'âˆ’') + t(abs(delta))
                rows += f"<tr><th>LineÃ—{v['line_cost']} / scrumÃ—1</th><td>{m['line']['tp']}/{m['line']['reference_segments']}</td><td>{len(v['line_recovered'])}</td><td>{len(v['line_lost'])}</td><td>{m['scrum']['tp']}/{m['scrum']['reference_segments']}</td><td>{pct(m['total']['precision'])}</td><td>{t(v['review']['unique_flagged_s'])}</td><td>{change}</td></tr>"
        scope = 'Anotaciones manuales humanas' if human else 'Inventario diagnÃ³stico automÃ¡tico parcial'
        date_label = f'{date[6:8]}/{date[4:6]}/{date[:4]}'
        sections.append(f'<section><h2>{date_label} Â· {scope}</h2>' + ('<div class="scroll"><table><thead><tr><th>Modelo</th><th>Lines</th><th>Recuperados</th><th>Perdidos</th><th>Scrums</th><th>PrecisiÃ³n total</th><th>RevisiÃ³n1Ã—</th><th>Cambio vsM1Ã—2</th></tr></thead><tbody>' + rows + '</tbody></table></div>' if rows else '<p>EvaluaciÃ³n pendiente.</p>') + '</section>')
    doc = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parciales Â·scrumÃ—1</title><style>body{font:16px/1.5 system-ui;max-width:1400px;margin:28px auto;padding:0 18px;color:#243447}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ccd5dd;text-align:left}thead{background:#edf3f8}aside{background:#fff0d6;padding:16px}a{color:#126b84}section{margin:35px 0}</style><h1>Resultados parciales Â·scrumÃ—1</h1><p>EvaluaciÃ³n anticipada de modelos ya guardados, mientras sigue el entrenamiento de los otros costos. Sus pesos, semilla, cinco Ã©pocas y datos siguen fijados; no se ajustan con estos resultados. Esta pÃ¡gina sÃ³lo incluye configuraciones con evaluaciÃ³n completa y comprobaciÃ³n independiente.</p><aside>16/08 usa las anotaciones manuales humanas;23/08 y Mundial usan inventarios diagnÃ³sticos automÃ¡ticos parciales. Parejas Ãºnicas de misma clase y solapamiento mÃ­nimo1s. RevisiÃ³n como uniÃ³n de intervalos a1Ã— sin pausas. Las alertas sin pareja en inventarios parciales no son todas falsas confirmadas. No hay extracciÃ³n nueva.</aside><p><a href="index.html">Avance del experimento completo</a> Â· <a href="partial_results.json">Datos verificados</a></p>' + ''.join(sections) + '</html>'
    result = dict(status='partial', variants=output, whole_experiment_complete=False, independent_verification='PASS', evaluation_amendment=read(HERE / 'evaluation_amendment.json'))
    for folder in (HERE, WEB):
        save(folder / 'partial_results.json', result)
        (folder / 'partial.html').write_text(doc, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
if __name__ == '__main__':
    main()
