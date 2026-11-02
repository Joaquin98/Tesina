"""New report and viewer IDs only; preserve all earlier comparisons."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import sys, html
import pandas as pd
from common import *
from train import STRATEGIES
sys.path.insert(0, str(ROOT))
from models.common.comparison.metrics import events_from_blocks
NAMES = {'central': 'Central', 'residual': 'Corrección residual', 'late_mix': 'Mezcla de decisiones', 'protected_residual': 'Corrección con protección'}

def main():
    assert load(HERE / 'evaluated.json')['complete']
    out = ROOT / 'src/web/static/reports' / HERE.name
    out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(_resolve_model_artifact(HERE / 'results.csv'))
    means = results.groupby('strategy').mean(numeric_only=True)
    pilot = pd.read_csv(_resolve_model_artifact(HERE / 'pilot/results.csv')).set_index('strategy')
    modes = ['central'] + STRATEGIES
    keys = ['f1_macro', 'accuracy', 'line_precision', 'line_recall', 'line_f1', 'scrum_precision', 'scrum_recall', 'scrum_f1']
    body = '<h1>Tres estrategias que conservan el central · D2 / VF</h1><p>Argmax sin umbrales. Medias de tres semillas17/42/2026, no ensamble. D2:8796 ejemplos; VF:688 bloques humanos. Todas las cabezas guardadas antes de evaluar VF.</p><table><tr><th>Métrica</th>' + ''.join(('<th>' + NAMES[k] + '</th>' for k in modes)) + '</tr>'
    for k in keys:
        body += '<tr><td>' + k + '</td>' + ''.join((f'<td>{100 * means.loc[m, k]:.2f}%</td>' for m in modes)) + '</tr>'
    body += '</table><h2>Corregidos y rotos respecto al central</h2><p>Conteos medios por semilla sobre los mismos688bloques; cada comparación usa el central de su semilla.</p><table><tr><th>Estrategia</th><th>Errores corregidos</th><th>Aciertos rotos</th></tr>'
    for m in STRATEGIES:
        body += f"<tr><td>{NAMES[m]}</td><td>{means.loc[m, 'corrected']:.2f}</td><td>{means.loc[m, 'broken']:.2f}</td></tr>"
    body += '</table><h2>Cambios por clase real</h2><table><tr><th>Estrategia</th><th>Clase real</th><th>Corregidos medios</th><th>Rotos medios</th></tr>'
    for mode in STRATEGIES:
        for label in LABELS:
            values = [load(HERE / f'seed_{s}/{mode}/vf_metrics.json')['changes']['by_true_class'][label] for s in SEEDS]
            body += f"<tr><td>{NAMES[mode]}</td><td>{label}</td><td>{np.mean([v['corrected'] for v in values]):.2f}</td><td>{np.mean([v['broken'] for v in values]):.2f}</td></tr>"
    body += '</table><h2>Pesos medios por vista</h2><p>Promedio de688clips y3semillas. En mezcla son coeficientes de probabilidades; en residual sólo ponderan contextos: el central se suma por una vía fija y su cero en esta tabla no significa que se ignore. Pesos no causales.</p><table><tr><th>Estrategia</th>' + ''.join((f'<th>{i}</th>' for i in range(9))) + '</tr>'
    for mode in STRATEGIES:
        a = np.mean([_load_model_result_array(HERE / f'seed_{s}/{mode}/vf_attention.npy').mean(0) for s in SEEDS], axis=0)
        body += '<tr><td>' + NAMES[mode] + '</td>' + ''.join((f'<td>{100 * v:.1f}%</td>' for v in a)) + '</tr>'
    body += '</table><h2>Criterio fijado antes de evaluar</h2><p>MacroF1+3pp, F1 de line y scrum mayores, ninguna precisión/sensibilidad pierde más de2pp, mejora en2/3semillas e IC95 positivo tanto estratificado como temporal60s. Tres pruebas exploratorias; no corrección de multiplicidad ni test externo independiente. VF se observó previamente.</p>'
    comparisons = load(HERE / 'comparisons.json')['comparisons']
    for m, c in comparisons.items():
        b = c['stratified_bootstrap']
        t = c['temporal_bootstrap']
        body += f"<p><b>{NAMES[m]}</b>: delta macroF1 {100 * b['mean_difference']:+.2f}pp; IC95 estratificado [{100 * b['ci95'][0]:+.2f},{100 * b['ci95'][1]:+.2f}], temporal60s [{100 * t['ci95'][0]:+.2f},{100 * t['ci95'][1]:+.2f}]. Cumple criterio: {c['evidence_rule']['passed']}.</p>"
    body += '<h2>Diagnóstico interno V6</h2><p>Central piloto entrenado desde cero excluyendoV6, y cabezas piloto con la misma partición. Sin selección de épocas ni ajustes conV6. CH tiene procedencia incompleta: no afirmamos independencia absoluta.</p>' + pilot[keys].mul(100).round(2).to_html()
    links = []
    frames = {}
    for mode in modes:
        df = pd.read_csv(_resolve_model_artifact(HERE / f'seed_42/{mode}/vf_predictions.csv'))
        frames[mode] = df
        for section, f in df.groupby('section', sort=True):
            detections = []
            truth = []
            for label in ['line', 'scrum']:
                for column, target in [('y_final', detections), ('y_true', truth)]:
                    for e in events_from_blocks(f, label, prediction_column=column, probability_column='p_' + label):
                        target.append(dict(play_type=label, start_time=e.start, end_time=e.end, duration=e.end - e.start, anchor=e.anchor, confidence=float(f[f.block_index.isin(e.blocks)]['p_' + label].max()) if column == 'y_final' else 1.0))
            windows = [dict(play_type=r.y_final, start_time=r.start_s, end_time=r.end_s, duration=r.end_s - r.start_s, confidence=max((getattr(r, 'p_' + k) for k in LABELS)), probabilities={k: getattr(r, 'p_' + k) for k in LABELS}) for r in f.itertuples()]
            rid = f'20261026_central_preserving_{mode}_D2_VF_{section}_S42'
            payload = dict(status='success', video_path=f.source.iloc[0], model_name=NAMES[mode] + ' D2 VF ' + section + ' · argmax', detections=detections, primary_detections=detections, window_predictions=windows, human_reference_events=truth, summary=dict(total_detections=len(detections), total_windows=len(windows), counts={k: sum((e['play_type'] == k for e in detections)) for k in ['line', 'scrum']}), statistics=dict(avg_confidence=float(np.mean([e['confidence'] for e in detections])) if detections else 0, avg_duration=float(np.mean([e['duration'] for e in detections])) if detections else 0, by_type={k: dict(count=sum((e['play_type'] == k for e in detections)), total_duration=sum((e['duration'] for e in detections if e['play_type'] == k))) for k in ['line', 'scrum']}), metadata=dict(duration=float(f.end_s.max()), fps=25, confidence_threshold=0, line_confidence_threshold=0, analysis_mode='segments', decision_rule='argmax without thresholds', source_experiment=str(HERE), experiment_report_url=f'/static/reports/{HERE.name}/index.html', independent_test_claim_allowed=False))
            path = ROOT / 'data/results' / f'result_{rid}.json'
            if _resolve_model_artifact(path).exists():
                assert load(path)['window_predictions'] == windows
            path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding='utf8')
            links.append(f'<li><a href="/viewer?result_id={rid}">{NAMES[mode]} · {section}</a></li>')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(17, 18))
    for ax, section in zip(axes, ['S01', 'S02', 'S03', 'S04']):
        names = []
        n = 0
        for label, color in [('line', '#167a9c'), ('scrum', '#bd7330')]:
            for mode, col in [('human', 'y_true')] + [(m, 'y_final') for m in modes]:
                f = frames['central' if mode == 'human' else mode]
                f = f[f.section == section]
                for e in events_from_blocks(f, label, prediction_column=col, probability_column='p_' + label):
                    ax.broken_barh([(e.start / 60, (e.end - e.start) / 60)], (n - 0.3, 0.6), facecolors=color)
                names.append(('Humano' if mode == 'human' else NAMES[mode]) + ' · ' + label)
                n += 1
        ax.set_yticks(range(n), names)
        ax.set_title(section + ' · semilla42')
        ax.set_xlim(0, 20.1)
        ax.grid(axis='x', alpha=0.2)
        ax.set_xlabel('Minutos relativos')
    fig.tight_layout()
    fig.savefig(out / 'timelines.png', dpi=130)
    plt.close(fig)
    body += '<h2>Timelines y visores · semilla42</h2><img style="width:100%" src="timelines.png"><ul>' + ''.join(links) + '</ul><p>Eventos: consultar vf_metrics.json para métricas separadas con emparejamiento uno a uno35s; F1 ponderado penaliza localización del ancla. Las tablas principales son por bloques.</p><p>Encoder central congelado aplicado a contextos implica cambio de distribución; pesos no causales. Protección usa aciertos de entrenamiento, no garantiza conservar aciertos nuevos. Caché y centrales reutilizados: no benchmark fresco.</p>'
    timing = load(HERE / 'timing.json')
    body += '<h2>Tiempos reales</h2><pre>' + html.escape(json.dumps(timing, indent=2)) + '</pre>'
    (out / 'index.html').write_text('<!doctype html><html lang="es"><meta charset="utf-8"><title>Central y tres estrategias</title><style>body{max-width:1250px;margin:25px auto;font:17px system-ui;line-height:1.5}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}pre{white-space:pre-wrap}</style>' + body + '</html>', encoding='utf8')
    (HERE / 'RESULTS.md').write_text('# Central y tres estrategias\n\n' + means[keys + ['corrected', 'broken']].to_csv() + '\n\n' + json.dumps(comparisons, indent=2) + '\n\nExploratorio:VF reutilizado, no test externo. Media3semillas, no ensamble.\n', encoding='utf8')
    save(HERE / 'publication.json', dict(published=True, report=f'http://127.0.0.1:5000/static/reports/{HERE.name}/index.html', viewer_results=16))
if __name__ == '__main__':
    main()
