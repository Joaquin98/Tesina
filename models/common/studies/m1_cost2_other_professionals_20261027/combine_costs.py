"""Combine immutable original/cost2/cost4/cost8 comparisons after verification."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import importlib.util, json, sys, html, urllib.request
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('other_publication', _resolve_model_artifact(HERE / 'publish.py'))
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)
WEB = ROOT / 'src/web/static/reports/m1_all_costs_other_professionals_20261027'

def main():
    variants = []
    for cost in (2, 4, 8):
        folder = ROOT / f'data/m1_cost{cost}_other_professionals_20261027'
        assert pub.read(folder / 'verification.json')['status'] == 'PASS'
        r = pub.read(folder / 'results.json')
        for v in r['variants']:
            if cost != 2 and v['mode'] == 'original':
                continue
            v['cost'] = 0 if v['mode'] == 'original' else cost
            variants.append(v)
    variants.sort(key=lambda v: (v['date'], v['stride'], v['cost']))
    assert len(variants) == 12
    comparisons = {}
    sections = []
    for date in ('20250823', '20231020'):
        vs = [v for v in variants if v['date'] == date]
        records = {}
        for v in vs:
            rid = v['viewer_url'].split('result_id=')[1]
            p = pub.read(ROOT / f'data/results/result_{rid}.json')
            p.setdefault('result_id', rid)
            records[v['title']] = p
        cid = f'20261027_argentina_allblacks_{date}_original_cost2_cost4_cost8'
        history = dict(id=cid, video_path=next(iter(records.values()))['video_path'], video_name=f'Argentina–NZ {date} · original y costos ×2/×4/×8', model_names=list(records), parameters=dict(confidence_threshold=0.0, line_confidence_threshold=0.0), results=records, comparison={k: dict(total=len(p['detections']), by_type={c: sum((e['play_type'] == c for e in p['detections'])) for c in ('line', 'scrum')}, avg_confidence=p['statistics']['avg_confidence']) for k, p in records.items()})
        assert pub.post('/api/comparison-history', history)['success']
        assert pub.get('/api/comparison-history/' + cid)['comparison']['results'] == records
        comparisons[date] = '/compare?comparison=' + cid
        rows = ''
        for v in vs:
            m = v['metrics']
            t = m['total']
            rev = v['review']
            rows += f'''<tr><th>{html.escape(v['title'])}</th><td>{m['line']['tp']}/{m['line']['reference_segments']}</td><td>{m['scrum']['tp']}/{m['scrum']['reference_segments']}</td><td>{pub.pct(t['recall'])}</td><td>{pub.pct(t['precision'])}</td><td>{t['alerts']}</td><td>{t['fp']}</td><td>{pub.tm(rev['unique_flagged_s'])}</td><td>{pub.tm(rev['flagged_outside_true_segments_s'])}</td><td><a href="{v['viewer_url']}">Ver</a></td></tr>'''
        label = '23/08/2025 · Argentina 29–23 NZ' if date == '20250823' else '20/10/2023 · Argentina 6–44 NZ'
        timeline = ''.join((pub.timeline(k, p['detections'], vs[0]['review']['unique_flagged_s'] * 0 + (5680.44 if date == '20250823' else 5776.88)) for k, p in records.items()))
        sections.append(f'<h2>{label}</h2><p><a href="{comparisons[date]}">Abrir video y comparar todos los modelos</a></p><div class="scroll"><table><thead><tr><th>Modelo</th><th>Line</th><th>Scrum</th><th>Sensibilidad*</th><th>Precisión temporal*</th><th>Alertas</th><th>Sin pareja*</th><th>Revisión única</th><th>Fuera del inventario*</th><th>Video</th></tr></thead><tbody>{rows}</tbody></table></div>{timeline}')
    report = '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Otros profesionales · todos los costos</title><style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:28px auto;padding:0 18px;color:#243447}aside{padding:16px;background:#fff0d6}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{padding:10px;border-bottom:1px solid #dbe2e9;text-align:left}thead{background:#edf3f8}a{color:#126b84}.timeline{margin:12px 0}.timeline svg{display:block;width:100%;height:26px}</style><h1>Original y costos ×2, ×4 y ×8 · otros profesionales</h1><p>Mismos checkpoints ya entrenados, argmax y cachés; sin entrenar, extraer ni ajustar umbrales nuevos. Verde: line; naranja: scrum.</p><aside>*Referencias diagnósticas automáticas parciales:20 line/7 scrum para23/08 y29 line/8 scrum para el Mundial. Faltan dos lines por localizar en cada inventario frente al PDF de competición. Estas métricas no certifican sensibilidad del partido completo. Coincidencia de la misma clase, al menos1s y una pareja por jugada. Alertas sin pareja y tiempo fuera del inventario no son todos falsos confirmados. No mezclar precisión temporal con la antigua precisión visual que admitía repeticiones/rearmados. Revisión única a1×, sin pausas, eliminando solapamientos. Mundial sólo cada25 porque no existe caché completo.</aside><p><a href="results.json">Todos los datos</a> · <a href="/static/reports/m1_recall_cost_20261027/index.html">16/08 con etiquetas humanas</a></p>' + ''.join(sections) + '</html>'
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / 'index.html').write_text(report, encoding='utf-8')
    pub.save(WEB / 'results.json', dict(status='complete', variants=variants, comparisons=comparisons, extractors_executed=0, new_training=False))
    pub.save(HERE / 'all_costs_results.json', pub.read(WEB / 'results.json'))
    print(json.dumps({'status': 'published', 'url': 'http://127.0.0.1:5000/static/reports/m1_all_costs_other_professionals_20261027/index.html'}))
if __name__ == '__main__':
    main()
