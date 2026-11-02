"""Final descriptive tables derived from already frozen, completed runs."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import csv, json
import numpy as np, pandas as pd
HERE = experiment_directory(__file__)
OUT = HERE / 'outputs'

def read(path):
    return json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8-sig'))

def extend_report(lines):
    f = pd.read_csv(_resolve_model_artifact(OUT / 'per_match_seed.csv'))
    validation = read(OUT / 'final_validation.json')
    assert validation['status'] == 'passed' and len(f) == 36
    assert all((len(f[f['size'] == s]) == 18 for s in (224, 320)))
    assert not f.duplicated(['size', 'seed', 'fold']).any()
    cols = ['f1_temporal_macro', 'line_weighted_f1', 'line_weighted_precision', 'line_weighted_recall', 'scrum_weighted_f1', 'scrum_weighted_precision', 'scrum_weighted_recall', 'line_far_false_positives_per_hour', 'scrum_far_false_positives_per_hour', 'line_all_unmatched_false_positives_per_hour', 'scrum_all_unmatched_false_positives_per_hour']
    means = f.groupby(['model', 'size'])[cols].mean()
    means.to_csv(OUT / 'comparison_means.csv')
    lines += ['', 'ValidaciÃ³n final', 'verify_results.py pasÃ³ las36 evaluaciones (18 por condiciÃ³n), comprobÃ³ checkpoints y probabilidades, recalculÃ³ mÃ©tricas y verificÃ³ mÃ¡scaras idÃ©nticas. Los1561 ejemplos iniciales320 coinciden exactamente con sus filas en la cachÃ© completa. La carpeta anterior mantiene su inventario sin cambios.', 'El F1 temporal macro promedia line y scrum. El F1 macro de la primera etapa promedia tres clases y no debe compararse directamente con este resultado.', '', 'Falsas alarmas por clase (promedios de18 evaluaciones)', 'Recorte | clase | FP lejanos/h | todos los FP no emparejados/h']
    for size in (224, 320):
        for label in ('line', 'scrum'):
            r = f[f['size'] == size]
            lines.append(f"{size} | {label} | {r[label + '_far_false_positives_per_hour'].mean():.4f} | {r[label + '_all_unmatched_false_positives_per_hour'].mean():.4f}")
    lines += ['FP lejanos/h conserva la definiciÃ³n original: eventos predichos no emparejados sin evento real de su clase, partido y secciÃ³n dentro de35s del ancla. Todos los no emparejados/h agrega duplicados cercanos. El denominador original suma el mÃ¡ximo end_s por partido y secciÃ³n, dividido por3600. Son medias de tasas por partido/semilla, no una tasa global agregada.', '', 'Resultados emparejados por partido y semilla', 'Semilla | pliegue | prueba | F1 macro224 | F1 macro320 | delta macro | delta line | delta scrum']
    base = f[f['size'] == 224]
    candidate = f[f['size'] == 320]
    paired = base.merge(candidate, on=['model', 'fold', 'seed', 'test_group'], suffixes=('_224', '_320')).sort_values(['seed', 'fold'])
    assert len(paired) == 18
    for row in paired.itertuples():
        d = row.f1_temporal_macro_320 - row.f1_temporal_macro_224
        dl = row.line_weighted_f1_320 - row.line_weighted_f1_224
        ds = row.scrum_weighted_f1_320 - row.scrum_weighted_f1_224
        lines.append(f'{row.seed} | {row.fold} | {row.test_group} | {row.f1_temporal_macro_224:.4f} | {row.f1_temporal_macro_320:.4f} | {d:+.4f} | {dl:+.4f} | {ds:+.4f}')
    lines += ['', 'Diferencias por partido (media de las tres semillas)', 'Partido | F1 macro224 | F1 macro320 | delta']
    for group, r in paired.groupby('test_group'):
        a = r.f1_temporal_macro_224.mean()
        b = r.f1_temporal_macro_320.mean()
        lines.append(f'{group} | {a:.4f} | {b:.4f} | {b - a:+.4f}')
    lines += ['', 'Diferencias por semilla (media de seis partidos)', 'Semilla | F1 macro224 | F1 macro320 | delta']
    for seed, r in paired.groupby('seed'):
        a = r.f1_temporal_macro_224.mean()
        b = r.f1_temporal_macro_320.mean()
        lines.append(f'{seed} | {a:.4f} | {b:.4f} | {b - a:+.4f}')
    thresholds = []
    costs = []
    ledger = [json.loads(v) for v in _resolve_model_artifact(OUT / 'worker_timings.jsonl').read_text().splitlines()]
    for size in (224, 320):
        initial = [read(p) for p in _model_artifact_glob(OUT / 'initial', f'R{size}_S*/complete.json', recursive=False)]
        paths = sorted(_model_artifact_glob(OUT / 'runs', f'M0_R{size}_*/complete.json', recursive=False))
        runs = [read(p) for p in paths]
        cache = read(OUT / f'cache/R{size}/state.json')
        stage = read(OUT / f'initial_stage/R{size}/state.json') if size == 320 else {}
        assert len(initial) == 3 and {r['seed'] for r in initial} == {17, 42, 2026}
        for path, meta in zip(paths, runs):
            t = read(path.parent / 'thresholds.json')
            assert t['source'] == 'validation_only'
            for key in ('line', 'scrum'):
                assert np.isfinite(t[key]) and min(abs(t[key] - np.arange(0.1, 1, 0.05))) < 1e-08
            thresholds.append(dict(size=size, seed=meta['seed'], fold=meta['fold'], test_group=meta['test_group'], line=t['line'], scrum=t['scrum'], source=t['source']))

        def measured_sum(records, key):
            return None if any((r.get(key) is None for r in records)) else sum((r[key] for r in records)) / 3600
        costs.append(dict(size=size, artifact_origin='historical_equivalent_reference' if size == 224 else 'new_fixed_halfscale320', extraction_h=(cache['active_s'] + stage.get('active_s', 0)) / 3600, cnn_h=(cache['inference_s'] + stage.get('inference_s', 0)) / 3600, decode_h=(cache['decode_resize_s'] + stage.get('decode_resize_s', 0)) / 3600, initial_elapsed_h=sum((r['elapsed_s'] for r in initial)) / 3600, adapt_elapsed_h=sum((r['elapsed_s'] for r in runs)) / 3600, initial_fit_h=measured_sum(initial, 'training_s'), adapt_fit_h=measured_sum(runs, 'training_s'), prediction_and_checkpoint_h=measured_sum(runs, 'prediction_s'), validation_threshold_search_and_metrics_h=measured_sum(runs, 'threshold_and_metrics_s'), worker_wall_new_h=sum((t['wall_s'] for t in ledger if '--size' in t['command'] and int(t['command'][t['command'].index('--size') + 1]) == size)) / 3600, max_rss_gib=max([cache['peak_rss_bytes'], stage.get('peak_rss_bytes', 0)] + [r['peak_rss_bytes'] for r in initial + runs]) / 2 ** 30, main_cache_gib=cache['features_bytes'] / 2 ** 30, aux_cache_gib=stage.get('features_bytes', 0) / 2 ** 30))
    pd.DataFrame(thresholds).to_csv(OUT / 'thresholds_per_match_seed.csv', index=False)
    pd.DataFrame(costs).to_csv(OUT / 'cost_breakdown.csv', index=False)

    def fmt(v):
        return 'no medido por separado' if v is None else f'{v:.4f}'
    lines += ['', 'Desglose de tiempos medidos (horas)', 'Recorte | fit inicial | fit adaptaciÃ³n | predicciÃ³n y guardado | bÃºsqueda en validaciÃ³n y mÃ©tricas | workers nuevos completos']
    for c in costs:
        lines.append(f"{c['size']} | {fmt(c['initial_fit_h'])} | {fmt(c['adapt_fit_h'])} | {fmt(c['prediction_and_checkpoint_h'])} | {fmt(c['validation_threshold_search_and_metrics_h'])} | {c['worker_wall_new_h']:.4f}")
    lines += ['El componente prediction_s incluye guardado del checkpoint y predicciones de validaciÃ³n y prueba. threshold_and_metrics_s incluye bÃºsqueda de umbrales en validaciÃ³n, aplicaciÃ³n en prueba, cÃ¡lculo y escritura de mÃ©tricas. Las categorÃ­as desglosan las etapas y no se suman otra vez a sus tiempos elapsed_s. La comprobaciÃ³n inicial aÃ±ade su worker al total nuevo de320.', 'La espera por otro experimento, la auditorÃ­a comÃºn y la verificaciÃ³n final del monitor no forman parte de los workers de320. Los intentos interrumpidos de copia anteriores no quedaron cronometrados Ã­ntegramente. No se presenta el total nuevo de224 como costo de entrenar224 desde cero.', '', 'Costo y recomendaciÃ³n final']
    c224, c320 = costs
    lines += [f"ExtracciÃ³n224 histÃ³rica:{c224['extraction_h']:.3f}h. ExtracciÃ³n320 nueva, incluidas las caracterÃ­sticas de la primera etapa reutilizadas:{c320['extraction_h']:.3f}h ({c320['extraction_h'] / c224['extraction_h']:.2f} veces). Workers nuevos completos de320:{c320['worker_wall_new_h']:.3f}h. RSS mÃ¡ximo:{c320['max_rss_gib']:.3f}GiB. CachÃ© principal:{c320['main_cache_gib']:.3f}GiB y auxiliar inicial:{c320['aux_cache_gib']:.3f}GiB.", 'Estos costos corresponden a sesiones distintas sobre el mismo Ryzen7 5700G,32GB RAM y TensorFlow CPU con8 hilos intraop y1 interop. Hay interferencia documentada y variaciÃ³n de rendimiento durante la extracciÃ³n. Las razones temporales describen lo observado, no un benchmark aislado del tamaÃ±o de imagen. RSS incluye pÃ¡ginas de cachÃ© mapeadas.', 'Recomiendo mantener224 para este modelo y corpus. El recorte320 no aporta una mejora consistente del F1 temporal y requiere aproximadamente el doble del tiempo de extracciÃ³n registrado. La reducciÃ³n pequeÃ±a de FP lejanos totales por hora no compensa la caÃ­da de F1 de scrum y macro observada para este objetivo.', 'La media de320 es menor, pero el intervalo descriptivo del delta cruza cero. No se demuestra superioridad universal de224 ni equivalencia estadÃ­stica. Los tres partidos con delta positivo y la semilla2026 positiva muestran heterogeneidad.', 'El bootstrap de20000 remuestreos utiliza sÃ³lo seis partidos reutilizados. No son18 partidos independientes y los entrenamientos de los pliegues se superponen. Las posibles fuentes CH/CV compartidas y los duplicados iniciales siguen limitando la interpretaciÃ³n. La recomendaciÃ³n queda condicionada a este corpus y protocolo.', '', 'Archivos finales', 'outputs/per_match_seed.csv: las36 evaluaciones y mÃ©tricas por clase.', 'outputs/paired_differences.csv y paired_summary.json: diferencias emparejadas e intervalo descriptivo.', 'outputs/thresholds_per_match_seed.csv: los36 pares de umbrales seleccionados sÃ³lo en validaciÃ³n.', 'outputs/cost_breakdown.csv: costos reales separados y procedencia histÃ³rica o nueva.', 'outputs/final_validation.json: integridad y recÃ¡lculo aprobados.', 'outputs/geometry/field_of_view_comparison.png: comprobaciÃ³n visual del encuadre.']
    return costs
