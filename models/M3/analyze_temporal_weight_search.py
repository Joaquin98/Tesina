"""Analyse architecture-specific temporal weights without video reprocessing.

Two complementary analyses are produced:

1. Screening: each architecture selects its temporal share on source-disjoint
   fold 4 and is evaluated once on fold 5.
2. Cross-fitted confirmation: for each existing TCN+BiGRU OOF seed, the share
   used on an outer fold is selected using OOF predictions from the other four
   folds only.  Thus no hold-out label chooses its own weight.
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
import csv
import json
from pathlib import Path
import numpy as np
from models.M3.run_temporal_architecture_suite import ARCHITECTURES, PAIRED, fuse_with_temporal_share, paired_test, probability_metrics, select_temporal_share
from models.M2.multimodal import audit_performance as audit
HERE = experiment_directory(__file__)
SCREEN_ROOT = HERE / 'outputs' / 'temporal_weight_search_20260815'
CONFIRM_ROOT = HERE / 'outputs' / 'temporal_architecture_suite_20260814' / 'confirmation' / 'tcn_bigru_attention'
SEEDS = (42, 17, 2026)

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with _resolve_model_artifact(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def compact_metrics(metrics: dict) -> dict:
    return {key: metrics[key] for key in ('accuracy', 'balanced_accuracy', 'f1_macro', 'negative_log_likelihood', 'brier_multiclass', 'ece_15')}

def main() -> None:
    with _load_model_result_array(PAIRED, allow_pickle=True) as data:
        names = data['names'].astype(str)
        labels = data['labels'].astype(np.int64)
        historical = data['historical'].astype(np.float64)
        static = data['highres_auxiliary'].astype(np.float64)
    folds, external_historical, external_labels, source = audit.load_external(names, labels)
    if not np.array_equal(labels, external_labels):
        raise AssertionError('external labels do not align')
    if not np.allclose(historical, external_historical, atol=2e-06):
        raise AssertionError('historical probabilities do not align')
    official = 0.4 * historical + 0.6 * static
    official_metrics = probability_metrics(labels, official)
    screening_rows: list[dict] = []
    for architecture in ARCHITECTURES:
        root = SCREEN_ROOT / 'screening_seed_42' / architecture
        summary = json.loads(_resolve_model_artifact(root / 'summary.json').read_text(encoding='utf-8'))
        with _load_model_result_array(root / 'predictions.npz', allow_pickle=True) as predictions:
            indices = predictions['indices'].astype(np.int64)
            selected_probabilities = predictions['selected_weight_fused'].astype(np.float64)
            fixed_probabilities = predictions['fused'].astype(np.float64)
            cached_official = predictions['official'].astype(np.float64)
        if not np.allclose(cached_official, official[indices], atol=2e-06):
            raise AssertionError(f'official probabilities do not align for {architecture}')
        selected_metrics = probability_metrics(labels[indices], selected_probabilities)
        fixed_metrics = probability_metrics(labels[indices], fixed_probabilities)
        official_metrics = probability_metrics(labels[indices], official[indices])
        selected_paired = paired_test(labels[indices], official[indices], selected_probabilities, source[indices], seed=20260815)
        screening_rows.append({'architecture': architecture, 'selected_share_within_auxiliary': summary['selected_temporal_share_within_auxiliary'], 'selected_effective_final_temporal_weight': summary['selected_effective_final_temporal_weight'], 'validation_selected_macro_f1': summary['validation']['selected_weight_fused']['f1_macro'], 'test_selected_accuracy': selected_metrics['accuracy'], 'test_selected_balanced_accuracy': selected_metrics['balanced_accuracy'], 'test_selected_macro_f1': selected_metrics['f1_macro'], 'test_fixed_18_accuracy': fixed_metrics['accuracy'], 'test_fixed_18_balanced_accuracy': fixed_metrics['balanced_accuracy'], 'test_fixed_18_macro_f1': fixed_metrics['f1_macro'], 'official_accuracy': official_metrics['accuracy'], 'official_balanced_accuracy': official_metrics['balanced_accuracy'], 'official_macro_f1': official_metrics['f1_macro'], 'selected_delta_accuracy': selected_metrics['accuracy'] - official_metrics['accuracy'], 'selected_delta_balanced_accuracy': selected_metrics['balanced_accuracy'] - official_metrics['balanced_accuracy'], 'selected_delta_macro_f1': selected_metrics['f1_macro'] - official_metrics['f1_macro'], 'mcnemar_exact_p': selected_paired['mcnemar_exact_p'], 'bootstrap_ci95_low': selected_paired['source_group_bootstrap_delta_accuracy_ci95'][0], 'bootstrap_ci95_high': selected_paired['source_group_bootstrap_delta_accuracy_ci95'][1]})
    crossfit_rows: list[dict] = []
    fold_share_rows: list[dict] = []
    fixed_curve_rows: list[dict] = []
    crossfit_payload = {}
    for seed in SEEDS:
        prediction_path = CONFIRM_ROOT / f'seed_{seed}' / 'oof_predictions.npz'
        with _load_model_result_array(prediction_path, allow_pickle=True) as predictions:
            if not np.array_equal(predictions['names'].astype(str), names):
                raise AssertionError(f'OOF names do not align for seed {seed}')
            temporal = predictions['temporal'].astype(np.float64)
        crossfit = np.zeros_like(temporal)
        selected_shares = []
        all_indices = np.arange(len(labels))
        for fold_number, fold in enumerate(folds, start=1):
            hold = np.asarray(fold['hold'], dtype=np.int64)
            train = np.setdiff1d(all_indices, hold, assume_unique=True)
            selected, _ = select_temporal_share(labels[train], historical[train], static[train], temporal[train])
            share = selected['temporal_share_within_auxiliary']
            crossfit[hold] = fuse_with_temporal_share(historical[hold], static[hold], temporal[hold], share)
            selected_shares.append(share)
            fold_share_rows.append({'seed': seed, 'fold': fold_number, 'selected_share_within_auxiliary': share, 'effective_final_temporal_weight': 0.6 * share, 'selection_macro_f1': selected['f1_macro']})
        crossfit_metrics = probability_metrics(labels, crossfit)
        official_metrics = probability_metrics(labels, official)
        crossfit_paired = paired_test(labels, official, crossfit, source, seed=20260815 + seed)
        crossfit_rows.append({'seed': seed, 'selected_shares': ';'.join((f'{value:.2f}' for value in selected_shares)), 'accuracy': crossfit_metrics['accuracy'], 'delta_accuracy': crossfit_metrics['accuracy'] - official_metrics['accuracy'], 'balanced_accuracy': crossfit_metrics['balanced_accuracy'], 'delta_balanced_accuracy': crossfit_metrics['balanced_accuracy'] - official_metrics['balanced_accuracy'], 'macro_f1': crossfit_metrics['f1_macro'], 'delta_macro_f1': crossfit_metrics['f1_macro'] - official_metrics['f1_macro'], 'negative_log_likelihood': crossfit_metrics['negative_log_likelihood'], 'ece_15': crossfit_metrics['ece_15'], 'candidate_only_correct': crossfit_paired['candidate_only_correct'], 'official_only_correct': crossfit_paired['reference_only_correct'], 'mcnemar_exact_p': crossfit_paired['mcnemar_exact_p'], 'bootstrap_ci95_low': crossfit_paired['source_group_bootstrap_delta_accuracy_ci95'][0], 'bootstrap_ci95_high': crossfit_paired['source_group_bootstrap_delta_accuracy_ci95'][1]})
        crossfit_payload[str(seed)] = {'selected_shares': selected_shares, 'metrics': compact_metrics(crossfit_metrics), 'paired_against_official': crossfit_paired}
        for share in np.round(np.arange(0.0, 1.0001, 0.05), 2):
            probabilities = fuse_with_temporal_share(historical, static, temporal, float(share))
            metrics = probability_metrics(labels, probabilities)
            fixed_curve_rows.append({'seed': seed, 'share_within_auxiliary': float(share), 'effective_final_temporal_weight': float(0.6 * share), 'accuracy': metrics['accuracy'], 'balanced_accuracy': metrics['balanced_accuracy'], 'macro_f1': metrics['f1_macro'], 'negative_log_likelihood': metrics['negative_log_likelihood'], 'ece_15': metrics['ece_15']})
    fixed_aggregate_rows: list[dict] = []
    for share in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        rows = [row for row in fixed_curve_rows if np.isclose(row['share_within_auxiliary'], share)]
        fixed_aggregate_rows.append({'share_within_auxiliary': float(share), 'effective_final_temporal_weight': float(0.6 * share), 'mean_accuracy': float(np.mean([row['accuracy'] for row in rows])), 'minimum_accuracy': float(np.min([row['accuracy'] for row in rows])), 'mean_balanced_accuracy': float(np.mean([row['balanced_accuracy'] for row in rows])), 'minimum_balanced_accuracy': float(np.min([row['balanced_accuracy'] for row in rows])), 'mean_macro_f1': float(np.mean([row['macro_f1'] for row in rows])), 'minimum_macro_f1': float(np.min([row['macro_f1'] for row in rows])), 'mean_negative_log_likelihood': float(np.mean([row['negative_log_likelihood'] for row in rows])), 'mean_ece_15': float(np.mean([row['ece_15'] for row in rows]))})
    robust_fixed = max(fixed_aggregate_rows, key=lambda row: (row['mean_macro_f1'], row['mean_balanced_accuracy'], row['mean_accuracy'], -row['share_within_auxiliary']))
    robust_share = robust_fixed['share_within_auxiliary']
    robust_fixed_seed_rows: list[dict] = []
    for seed in SEEDS:
        with _load_model_result_array(CONFIRM_ROOT / f'seed_{seed}' / 'oof_predictions.npz', allow_pickle=True) as predictions:
            temporal = predictions['temporal'].astype(np.float64)
        probabilities = fuse_with_temporal_share(historical, static, temporal, robust_share)
        metrics = probability_metrics(labels, probabilities)
        paired = paired_test(labels, official, probabilities, source, seed=20260815 + seed + 10000)
        robust_fixed_seed_rows.append({'seed': seed, 'share_within_auxiliary': robust_share, 'effective_final_temporal_weight': 0.6 * robust_share, 'accuracy': metrics['accuracy'], 'delta_accuracy': metrics['accuracy'] - official_metrics['accuracy'], 'balanced_accuracy': metrics['balanced_accuracy'], 'delta_balanced_accuracy': metrics['balanced_accuracy'] - official_metrics['balanced_accuracy'], 'macro_f1': metrics['f1_macro'], 'delta_macro_f1': metrics['f1_macro'] - official_metrics['f1_macro'], 'negative_log_likelihood': metrics['negative_log_likelihood'], 'ece_15': metrics['ece_15'], 'candidate_only_correct': paired['candidate_only_correct'], 'official_only_correct': paired['reference_only_correct'], 'mcnemar_exact_p': paired['mcnemar_exact_p'], 'bootstrap_ci95_low': paired['source_group_bootstrap_delta_accuracy_ci95'][0], 'bootstrap_ci95_high': paired['source_group_bootstrap_delta_accuracy_ci95'][1]})
    official_metrics = probability_metrics(labels, official)
    result = {'version': 1, 'video_decoded': False, 'yolo_executed': False, 'screening_protocol': 'share selected on source-disjoint fold 4; evaluated once on fold 5', 'screening': screening_rows, 'crossfit_protocol': 'for each outer hold, share selected from other four OOF folds only', 'official': compact_metrics(official_metrics), 'crossfit_tcn_bigru': crossfit_payload, 'posthoc_robust_fixed_weight': {'selection_rule': 'highest mean macro-F1 across the three OOF seeds', 'aggregate': robust_fixed, 'by_seed': robust_fixed_seed_rows, 'status': 'exploratory_posthoc'}, 'decision': 'weight tuning is experimental; promote only if cross-fitted/source-group evidence is stable and calibration does not materially degrade'}
    write_csv(SCREEN_ROOT / 'architecture_weight_screening.csv', screening_rows)
    write_csv(SCREEN_ROOT / 'tcn_bigru_crossfit_results.csv', crossfit_rows)
    write_csv(SCREEN_ROOT / 'tcn_bigru_crossfit_fold_shares.csv', fold_share_rows)
    write_csv(SCREEN_ROOT / 'tcn_bigru_fixed_weight_curves.csv', fixed_curve_rows)
    write_csv(SCREEN_ROOT / 'tcn_bigru_fixed_weight_aggregate.csv', fixed_aggregate_rows)
    write_csv(SCREEN_ROOT / 'tcn_bigru_robust_fixed_weight.csv', robust_fixed_seed_rows)
    (SCREEN_ROOT / 'weight_search_analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    report = ['# Búsqueda adicional de peso temporal', '', 'No se decodificaron videos ni se ejecutó YOLO. Se reutilizaron probabilidades y secuencias cacheadas.', '', '## Selección por arquitectura: validación → test reservado', '', '| Arquitectura | Temporal dentro del auxiliar | Peso temporal final | Accuracy test | Balanced acc. | Macro-F1 | Δ accuracy vs. oficial |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in screening_rows:
        report.append(f"| {row['architecture']} | {100 * row['selected_share_within_auxiliary']:.0f}% | {100 * row['selected_effective_final_temporal_weight']:.0f}% | {100 * row['test_selected_accuracy']:.2f}% | {100 * row['test_selected_balanced_accuracy']:.2f}% | {100 * row['test_selected_macro_f1']:.2f}% | {100 * row['selected_delta_accuracy']:+.2f} pp |")
    report += ['', 'El ajuste de peso sobre un solo fold fue inestable: TCN y Transformer eligieron pesos altos que empeoraron en test. TCN+BiGRU eligió 12% final y mejoró en ese test; esta observación se somete abajo a una prueba OOF más estricta.', '', '## TCN+BiGRU: selección cruzada del peso en 5 folds', '', '| Semilla | Pesos internos elegidos por fold | Accuracy | Δ accuracy | Balanced acc. | Macro-F1 | McNemar p | IC95% Δ accuracy |', '|---:|---|---:|---:|---:|---:|---:|---|']
    for row in crossfit_rows:
        report.append(f"| {row['seed']} | {row['selected_shares']} | {100 * row['accuracy']:.2f}% | {100 * row['delta_accuracy']:+.2f} pp | {100 * row['balanced_accuracy']:.2f}% | {100 * row['macro_f1']:.2f}% | {row['mcnemar_exact_p']:.4f} | [{100 * row['bootstrap_ci95_low']:+.2f}, {100 * row['bootstrap_ci95_high']:+.2f}] pp |")
    report += ['', 'Los pesos varían por fold y semilla; por lo tanto, la selección puntual sobre un único partido no debe interpretarse como un porcentaje universal. Los CSV incluyen la curva completa de 0% a 60% temporal efectivo.', '', '## Peso fijo robusto entre semillas (análisis post hoc)', '', f'El máximo macro-F1 medio apareció con {100 * robust_share:.0f}% temporal dentro del auxiliar, equivalente a {100 * 0.6 * robust_share:.0f}% del resultado final. Este valor se obtuvo después de observar las tres semillas y, por lo tanto, es exploratorio.', '', '| Semilla | Accuracy | Δ accuracy | Balanced acc. | Macro-F1 | McNemar p | IC95% Δ accuracy |', '|---:|---:|---:|---:|---:|---:|---|']
    for row in robust_fixed_seed_rows:
        report.append(f"| {row['seed']} | {100 * row['accuracy']:.2f}% | {100 * row['delta_accuracy']:+.2f} pp | {100 * row['balanced_accuracy']:.2f}% | {100 * row['macro_f1']:.2f}% | {row['mcnemar_exact_p']:.4f} | [{100 * row['bootstrap_ci95_low']:+.2f}, {100 * row['bootstrap_ci95_high']:+.2f}] pp |")
    (SCREEN_ROOT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
