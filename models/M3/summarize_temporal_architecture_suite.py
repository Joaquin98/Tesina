"""Consolidate the multi-seed temporal architecture confirmation.

This script reads the out-of-fold predictions already produced by
``run_temporal_architecture_suite.py``.  It never reads or processes video and
does not run YOLO.  The probability average across seeds is explicitly marked
as exploratory because it is computed after the confirmatory runs.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from models.M3.run_temporal_attention_experiment import paired_test, probability_metrics

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with _resolve_model_artifact(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def pct(value: float) -> str:
    return f'{100.0 * value:.3f}'

def signed_pct(value: float) -> str:
    return f'{100.0 * value:+.3f}'

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=experiment_directory(__file__) / 'outputs' / 'temporal_architecture_suite_20260814')
    parser.add_argument('--architecture', default='tcn_bigru_attention')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 17, 2026])
    args = parser.parse_args()
    confirmation_root = args.root / 'confirmation' / args.architecture
    with _resolve_model_artifact(args.root / 'screening_seed_42' / 'ranking.csv').open(newline='', encoding='utf-8') as handle:
        screening_rows = list(csv.DictReader(handle))
    seed_rows: list[dict] = []
    fold_rows: list[dict] = []
    temporal_probabilities: list[np.ndarray] = []
    fused_probabilities: list[np.ndarray] = []
    canonical: dict[str, np.ndarray] | None = None
    for seed in args.seeds:
        seed_root = confirmation_root / f'seed_{seed}'
        summary = json.loads(_resolve_model_artifact(seed_root / 'summary.json').read_text(encoding='utf-8'))
        if not summary.get('complete'):
            raise RuntimeError(f'Incomplete confirmation: {seed_root}')
        with _load_model_result_array(seed_root / 'oof_predictions.npz', allow_pickle=True) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if canonical is None:
            canonical = arrays
        else:
            for key in ('names', 'labels', 'source', 'official'):
                if not np.array_equal(canonical[key], arrays[key]):
                    raise RuntimeError(f'Seed {seed} does not align on {key}')
        temporal_probabilities.append(arrays['temporal'].astype(np.float64))
        fused_probabilities.append(arrays['fused'].astype(np.float64))
        candidate = summary['metrics']['fixed_30_percent_fused']
        official = summary['metrics']['official']
        paired = summary['paired_against_official']
        seed_rows.append({'seed': seed, 'accuracy': candidate['accuracy'], 'delta_accuracy': candidate['accuracy'] - official['accuracy'], 'balanced_accuracy': candidate['balanced_accuracy'], 'delta_balanced_accuracy': candidate['balanced_accuracy'] - official['balanced_accuracy'], 'macro_f1': candidate['f1_macro'], 'delta_macro_f1': candidate['f1_macro'] - official['f1_macro'], 'negative_log_likelihood': candidate['negative_log_likelihood'], 'delta_negative_log_likelihood': candidate['negative_log_likelihood'] - official['negative_log_likelihood'], 'ece_15': candidate['ece_15'], 'delta_ece_15': candidate['ece_15'] - official['ece_15'], 'candidate_only_correct': paired['candidate_only_correct'], 'official_only_correct': paired['reference_only_correct'], 'mcnemar_exact_p': paired['mcnemar_exact_p'], 'bootstrap_ci95_low': paired['source_group_bootstrap_delta_accuracy_ci95'][0], 'bootstrap_ci95_high': paired['source_group_bootstrap_delta_accuracy_ci95'][1]})
        for fold in summary['folds']:
            candidate_fold = fold['fixed_30_percent_fused']
            official_fold = fold['official']
            fold_rows.append({'seed': seed, 'fold': fold['fold'], 'validation_fold': fold['validation_fold'], 'selected_epoch': fold['selected_epoch'], 'temperature': fold['temperature'], 'temporal_macro_f1': fold['temporal']['f1_macro'], 'delta_accuracy': candidate_fold['accuracy'] - official_fold['accuracy'], 'delta_balanced_accuracy': candidate_fold['balanced_accuracy'] - official_fold['balanced_accuracy'], 'delta_macro_f1': candidate_fold['f1_macro'] - official_fold['f1_macro']})
    assert canonical is not None
    labels = canonical['labels'].astype(np.int64)
    source = canonical['source']
    official_probabilities = canonical['official'].astype(np.float64)
    ensemble_temporal = np.mean(temporal_probabilities, axis=0)
    ensemble_fused = np.mean(fused_probabilities, axis=0)
    ensemble_metrics = {'temporal': probability_metrics(labels, ensemble_temporal), 'fixed_30_percent_fused': probability_metrics(labels, ensemble_fused), 'official': probability_metrics(labels, official_probabilities)}
    ensemble_paired = paired_test(labels, official_probabilities, ensemble_fused, source, seed=20260814)
    official_prediction = official_probabilities.argmax(axis=1)
    ensemble_prediction = ensemble_fused.argmax(axis=1)
    source_rows: list[dict] = []
    for source_name in sorted(set(source.tolist()), key=str):
        mask = source == source_name
        official_accuracy = float(np.mean(official_prediction[mask] == labels[mask]))
        ensemble_accuracy = float(np.mean(ensemble_prediction[mask] == labels[mask]))
        source_rows.append({'source': source_name, 'clips': int(mask.sum()), 'official_accuracy': official_accuracy, 'ensemble_accuracy': ensemble_accuracy, 'delta_accuracy': ensemble_accuracy - official_accuracy})
    source_wins = sum((row['delta_accuracy'] > 0 for row in source_rows))
    source_ties = sum((row['delta_accuracy'] == 0 for row in source_rows))
    source_losses = sum((row['delta_accuracy'] < 0 for row in source_rows))
    deltas = {metric: np.asarray([row[metric] for row in seed_rows], dtype=np.float64) for metric in ('delta_accuracy', 'delta_balanced_accuracy', 'delta_macro_f1', 'delta_negative_log_likelihood', 'delta_ece_15')}
    aggregate = {metric: {'mean': float(values.mean()), 'sample_std': float(values.std(ddof=1)), 'min': float(values.min()), 'max': float(values.max())} for metric, values in deltas.items()}
    payload = {'version': 1, 'status': 'exploratory_candidate_not_promoted', 'architecture': args.architecture, 'seeds': args.seeds, 'clips': int(len(labels)), 'sources': len(source_rows), 'screening_seed_42': screening_rows, 'confirmatory_seed_results': seed_rows, 'across_seed_delta_summary': aggregate, 'exploratory_probability_ensemble': {'metrics': ensemble_metrics, 'paired_against_official': ensemble_paired, 'source_wins': source_wins, 'source_ties': source_ties, 'source_losses': source_losses}, 'decision_basis': ['All three seeds improve accuracy and macro-F1 only slightly.', 'No individual seed excludes zero in the source-group bootstrap interval.', 'Calibration (NLL and ECE) is worse than the official model.', 'Fold 3 is consistently worse across all three seeds.']}
    write_csv(confirmation_root / 'multi_seed_results.csv', seed_rows)
    write_csv(confirmation_root / 'fold_seed_deltas.csv', fold_rows)
    write_csv(confirmation_root / 'ensemble_source_deltas.csv', source_rows)
    (confirmation_root / 'multi_seed_summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    official = ensemble_metrics['official']
    ensemble = ensemble_metrics['fixed_30_percent_fused']
    lines = ['# Comparación de arquitecturas temporales sobre secuencias YOLO cacheadas', '', '## Objetivo y protocolo', '', 'Se compararon TCN con atención, Transformer, TCN+BiGRU con atención y un MLP por frame con atención usando las mismas 4.082 secuencias de 100 pasos y 90 variables ya extraídas. No se reprocesaron videos ni se volvió a ejecutar YOLO.', '', 'El screening seleccionó la arquitectura exclusivamente por macro-F1 temporal en validación source-disjoint. El ganador fue TCN+BiGRU con atención (113.541 parámetros). Después se congeló y se confirmó con cinco folds externos y tres semillas. La fusión permaneció fija: 40% histórico + 60% × (70% auxiliar estático + 30% temporal).', '', '## Screening de arquitecturas', '', '| Arquitectura | Parámetros | Época | Macro-F1 temporal de validación | Accuracy temporal de validación |', '|---|---:|---:|---:|---:|']
    for row in screening_rows:
        lines.append(f"| {row['architecture']} | {int(row['parameters']):,} | {row['selected_epoch']} | {pct(float(row['validation_temporal_f1_macro']))}% | {pct(float(row['validation_temporal_accuracy']))}% |")
    lines += ['', 'La selección correspondió a TCN+BiGRU con atención. El fold reservado mostrado por el script de screening no se utilizó para cambiar esta decisión.', '', '## Resultado confirmatorio por semilla', '', '| Semilla | Accuracy | Δ accuracy (pp) | Balanced acc. | Δ bal. (pp) | Macro-F1 | Δ F1 (pp) | McNemar p |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in seed_rows:
        lines.append(f"| {row['seed']} | {pct(row['accuracy'])}% | {signed_pct(row['delta_accuracy'])} | {pct(row['balanced_accuracy'])}% | {signed_pct(row['delta_balanced_accuracy'])} | {pct(row['macro_f1'])}% | {signed_pct(row['delta_macro_f1'])} | {row['mcnemar_exact_p']:.4f} |")
    lines += ['', f"Promedio de Δ accuracy entre semillas: {pct(aggregate['delta_accuracy']['mean'])} pp (rango {pct(aggregate['delta_accuracy']['min'])} a {pct(aggregate['delta_accuracy']['max'])}). Promedio de Δ macro-F1: {pct(aggregate['delta_macro_f1']['mean'])} pp.", '', 'Ninguna semilla obtuvo un intervalo bootstrap agrupado por fuente que excluyera cero. El fold 3 empeoró en las tres semillas, evidenciando heterogeneidad entre dominios. NLL y ECE empeoraron en las tres repeticiones.', '', '## Ensemble exploratorio de semillas', '', f"El promedio de probabilidades obtuvo accuracy {pct(ensemble['accuracy'])}% frente a {pct(official['accuracy'])}% del oficial (Δ {pct(ensemble['accuracy'] - official['accuracy'])} pp), balanced accuracy {pct(ensemble['balanced_accuracy'])}% y macro-F1 {pct(ensemble['f1_macro'])}%. McNemar p={ensemble_paired['mcnemar_exact_p']:.4f}; IC95% bootstrap por fuente [{pct(ensemble_paired['source_group_bootstrap_delta_accuracy_ci95'][0])}, {pct(ensemble_paired['source_group_bootstrap_delta_accuracy_ci95'][1])}] pp. Por fuente: {source_wins} mejoran, {source_ties} empatan y {source_losses} empeoran.", '', 'Este ensemble es exploratorio porque se calculó después de observar las repeticiones; no reemplaza la confirmación predefinida.', '', '## Decisión', '', 'No promover todavía TCN+BiGRU como modelo por defecto de la web. La señal es positiva pero pequeña, no excluye cero al agrupar por fuente y degrada la calibración. Se conserva como candidato experimental. El siguiente experimento recomendado es una compuerta aprendida y calibrada que reduzca el peso temporal cuando detecta baja calidad o cambio de dominio, validada de forma nested y con partidos nuevos etiquetados.', '', '## Artefactos', '', '- `multi_seed_summary.json`: resumen completo y métricas.', '- `multi_seed_results.csv`: una fila por semilla.', '- `fold_seed_deltas.csv`: diferencias por fold y semilla.', '- `ensemble_source_deltas.csv`: diferencias exploratorias por fuente.', '- `seed_*/`: checkpoints, historias y predicciones OOF originales.']
    (confirmation_root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
