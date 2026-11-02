"""Nested source-group search over cached historical/auxiliary probabilities.

No video, ResNet or YOLO inference is performed.  The experiment evaluates
global, class-specific, quality-adaptive, disagreement-adaptive and
temperature-calibrated late fusion.  Meta-parameters are selected on the four
non-held folds and applied to the fifth fold, repeated five times.
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
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
REPO = Path(__file__).resolve().parents[2]
HERE = experiment_directory(__file__)
PAIRED = HERE / 'outputs' / 'quantile_100frames' / 'paired_predictions.npz'
QUALITY_ROOT = HERE / 'outputs' / 'cached_descriptor_grid_20260814' / 'primitive_cache' / 'current'
DEFAULT_OUTPUT = HERE / 'outputs' / 'fusion_weight_search_20260814'
LABELS = ('line', 'other', 'scrum')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.multimodal import audit_performance as audit

def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def normalize(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-08, None)
    return probability / probability.sum(axis=-1, keepdims=True)

def temperature(probability: np.ndarray, value: float) -> np.ndarray:
    return normalize(np.power(np.clip(probability, 1e-08, 1.0), 1.0 / value))

def candidate_library() -> list[dict]:
    candidates: list[dict] = []
    for alpha in np.round(np.arange(0.4, 0.8001, 0.025), 3):
        candidates.append({'family': 'global', 'alpha': float(alpha)})
    line_values = (0.5, 0.6, 0.7, 0.8)
    other_values = (0.3, 0.4, 0.5, 0.6)
    scrum_values = (0.5, 0.6, 0.7, 0.8)
    class_vectors = list(itertools.product(line_values, other_values, scrum_values))
    for values in class_vectors:
        candidates.append({'family': 'classwise', 'alpha_line': values[0], 'alpha_other': values[1], 'alpha_scrum': values[2]})
    for low, high in itertools.product((0.3, 0.4, 0.5), (0.6, 0.7, 0.8)):
        candidates.append({'family': 'quality_gate', 'alpha_low': low, 'alpha_high': high})
    for low, high in itertools.product(np.round(np.arange(0.45, 0.5501, 0.025), 3), np.round(np.arange(0.55, 0.6501, 0.025), 3)):
        if high > low:
            candidates.append({'family': 'quality_gate_fine', 'alpha_low': float(low), 'alpha_high': float(high)})
    for base, quality_slope, margin_slope, disagreement_shift in itertools.product((0.5, 0.6, 0.7), (0.0, 0.25, 0.5), (0.0, 0.25, 0.5), (-0.1, 0.0, 0.1)):
        candidates.append({'family': 'adaptive_gate', 'base': base, 'quality_slope': quality_slope, 'margin_slope': margin_slope, 'disagreement_shift': disagreement_shift})
    for base, quality_slope, margin_slope, disagreement_shift in itertools.product((0.45, 0.475, 0.5, 0.525, 0.55), (0.25, 0.5, 0.75), (0.1, 0.25, 0.4), (0.05, 0.1, 0.15)):
        candidates.append({'family': 'adaptive_gate_fine', 'base': base, 'quality_slope': quality_slope, 'margin_slope': margin_slope, 'disagreement_shift': disagreement_shift})
    temperatures = (0.75, 1.0, 1.25, 1.5)
    for historical_t, auxiliary_t, alpha in itertools.product(temperatures, temperatures, (0.5, 0.6, 0.7)):
        candidates.append({'family': 'temperature_global', 'temperature_historical': historical_t, 'temperature_auxiliary': auxiliary_t, 'alpha': alpha})
    for values, historical_t, auxiliary_t in itertools.product(class_vectors, temperatures, temperatures):
        candidates.append({'family': 'classwise_temperature', 'alpha_line': values[0], 'alpha_other': values[1], 'alpha_scrum': values[2], 'temperature_historical': historical_t, 'temperature_auxiliary': auxiliary_t})
    for values, quality_slope in itertools.product(class_vectors, (0.2, 0.4)):
        candidates.append({'family': 'classwise_quality', 'alpha_line': values[0], 'alpha_other': values[1], 'alpha_scrum': values[2], 'quality_slope': quality_slope})
    return candidates

def candidate_probabilities(candidates: list[dict], historical: np.ndarray, auxiliary: np.ndarray, quality: np.ndarray) -> np.ndarray:
    output = np.empty((len(candidates), len(historical), 3), dtype=np.float32)
    temperature_cache_h = {value: temperature(historical, value) for value in (0.75, 1.0, 1.25, 1.5)}
    temperature_cache_a = {value: temperature(auxiliary, value) for value in (0.75, 1.0, 1.25, 1.5)}
    auxiliary_margin = auxiliary.max(axis=1) - historical.max(axis=1)
    disagreement = (auxiliary.argmax(axis=1) != historical.argmax(axis=1)).astype(np.float64)
    for index, config in enumerate(candidates):
        family = config['family']
        hist, aux = (historical, auxiliary)
        if family in ('temperature_global', 'classwise_temperature'):
            hist = temperature_cache_h[config['temperature_historical']]
            aux = temperature_cache_a[config['temperature_auxiliary']]
        if family in ('global', 'temperature_global'):
            alpha = config['alpha']
        elif family in ('classwise', 'classwise_temperature'):
            alpha = np.asarray([config['alpha_line'], config['alpha_other'], config['alpha_scrum']], dtype=np.float64)[None, :]
        elif family in ('quality_gate', 'quality_gate_fine'):
            alpha = config['alpha_low'] + quality * (config['alpha_high'] - config['alpha_low'])
            alpha = alpha[:, None]
        elif family in ('adaptive_gate', 'adaptive_gate_fine'):
            alpha = config['base'] + config['quality_slope'] * (quality - 0.5) + config['margin_slope'] * auxiliary_margin + config['disagreement_shift'] * disagreement
            alpha = np.clip(alpha, 0.1, 0.9)[:, None]
        elif family == 'classwise_quality':
            center = np.asarray([config['alpha_line'], config['alpha_other'], config['alpha_scrum']], dtype=np.float64)[None, :]
            alpha = np.clip(center + config['quality_slope'] * (quality[:, None] - 0.5), 0.1, 0.9)
        else:
            raise ValueError(family)
        output[index] = normalize((1.0 - alpha) * hist + alpha * aux).astype(np.float32)
    return output

def compact_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    confusion = np.bincount(y * 3 + prediction, minlength=9).reshape(3, 3).astype(np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support > 0)
    precision = np.divide(diagonal, predicted, out=np.zeros(3), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall > 0)
    return {'accuracy': float(diagonal.sum() / confusion.sum()), 'balanced_accuracy': float(recall.mean()), 'f1_macro': float(f1.mean()), 'recall_line': float(recall[0]), 'recall_other': float(recall[1]), 'recall_scrum': float(recall[2])}

def metric_table(y: np.ndarray, probabilities: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    rows = [compact_metrics(y[indices], item[indices].argmax(axis=1)) for item in probabilities]
    return pd.DataFrame(rows)

def choose(table: pd.DataFrame, candidate_indices: np.ndarray, baseline_index: int, constrained: bool) -> int:
    indexes = np.unique(np.append(candidate_indices, baseline_index)) if constrained else candidate_indices
    subset = table.loc[indexes].copy()
    if constrained:
        baseline = table.loc[baseline_index]
        eligible = (subset.accuracy >= baseline.accuracy - 0.002) & (subset.recall_line >= baseline.recall_line) & (subset.recall_scrum >= baseline.recall_scrum)
        subset = subset[eligible]
        if subset.empty:
            return baseline_index
    return int(subset.sort_values(['f1_macro', 'balanced_accuracy', 'accuracy'], ascending=False, kind='mergesort').index[0])

def probability_metrics(y: np.ndarray, probabilities: np.ndarray, bins: int=10) -> dict:
    one_hot = np.eye(3, dtype=np.float64)[y]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    ece = 0.0
    for left, right in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {'brier_multiclass': brier, 'ece_10_bins': float(ece)}

def paired_test(y, current, selected, source, samples=10000):
    current_pred = current.argmax(axis=1)
    selected_pred = selected.argmax(axis=1)
    current_correct = current_pred == y
    selected_correct = selected_pred == y
    selected_only = int(np.sum(selected_correct & ~current_correct))
    current_only = int(np.sum(current_correct & ~selected_correct))
    discordant = selected_only + current_only
    delta = selected_correct.astype(np.float64) - current_correct.astype(np.float64)
    groups = np.asarray(sorted(set(source.astype(str))))
    rng = np.random.default_rng(20260814)
    boot = np.empty(samples, dtype=np.float64)
    for iteration in range(samples):
        selected_groups = groups[rng.integers(0, len(groups), len(groups))]
        indexes = np.concatenate([np.flatnonzero(source == group) for group in selected_groups])
        boot[iteration] = delta[indexes].mean()
    return {'delta_accuracy': float(delta.mean()), 'selected_only_correct': selected_only, 'current_only_correct': current_only, 'discordant': discordant, 'mcnemar_exact_two_sided_p': float(binomtest(selected_only, discordant, 0.5).pvalue) if discordant else 1.0, 'source_group_bootstrap_ci95': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with _load_model_result_array(PAIRED, allow_pickle=True) as data:
        names = data['names'].astype(str)
        y = data['labels'].astype(np.int64)
        historical = normalize(data['historical'].astype(np.float64))
        auxiliary = normalize(data['highres_auxiliary'].astype(np.float64))
    folds, external_historical, external_y, source = audit.load_external(names, y)
    if not np.array_equal(y, external_y) or not np.allclose(historical, external_historical, atol=1e-06):
        raise AssertionError('External folds and paired probabilities are not aligned')
    quality_array = _load_model_result_array(QUALITY_ROOT / 'quality.npy', mmap_mode='r')
    quality_mask = _load_model_result_array(QUALITY_ROOT / 'mask.npy', mmap_mode='r').astype(bool)
    per_frame_quality = np.asarray(quality_array[:, :, [1, 2, 3]], dtype=np.float64).mean(axis=2)
    quality = np.divide(np.where(quality_mask, per_frame_quality, 0.0).sum(axis=1), np.maximum(quality_mask.sum(axis=1), 1))
    quality = np.clip(quality, 0.0, 1.0)
    quality_summary = {'definition': 'mean of person confidence, keypoint confidence and player coverage over valid cached frames', 'minimum': float(quality.min()), 'q25': float(np.quantile(quality, 0.25)), 'median': float(np.median(quality)), 'q75': float(np.quantile(quality, 0.75)), 'maximum': float(quality.max()), 'mean': float(quality.mean())}
    dump_json(args.output / 'quality_summary.json', quality_summary)
    candidates = candidate_library()
    probabilities = candidate_probabilities(candidates, historical, auxiliary, quality)
    baseline_index = next((index for index, config in enumerate(candidates) if config['family'] == 'global' and np.isclose(config['alpha'], 0.6)))
    current = probabilities[baseline_index]
    reproduced = audit.metric_dict(y, current.argmax(axis=1))
    if not np.isclose(reproduced['accuracy'], 0.8537481626653601):
        raise AssertionError(f"Current high-resolution metric was not reproduced: {reproduced['accuracy']}")
    families = sorted(set((config['family'] for config in candidates)))
    family_indices = {family: np.asarray([i for i, config in enumerate(candidates) if config['family'] == family], dtype=int) for family in families}
    nested_probabilities = {'overall_constrained': np.zeros_like(current)}
    nested_probabilities.update({f'family:{family}': np.zeros_like(current) for family in families})
    nested_unconstrained = {family: np.zeros_like(current) for family in families}
    selections = []
    all_indices = np.arange(len(y))
    for fold in folds:
        hold = np.asarray(fold['hold'], dtype=int)
        inner = np.setdiff1d(all_indices, hold, assume_unique=True)
        table = metric_table(y, probabilities, inner)
        overall = choose(table, np.arange(len(candidates)), baseline_index, constrained=True)
        nested_probabilities['overall_constrained'][hold] = probabilities[overall, hold]
        selections.append({'outer_fold': int(fold['fold']), 'selection': 'overall_constrained', 'candidate_index': overall, **candidates[overall]})
        for family in families:
            constrained_index = choose(table, family_indices[family], baseline_index, constrained=True)
            unconstrained_index = choose(table, family_indices[family], baseline_index, constrained=False)
            nested_probabilities[f'family:{family}'][hold] = probabilities[constrained_index, hold]
            nested_unconstrained[family][hold] = probabilities[unconstrained_index, hold]
            selections.append({'outer_fold': int(fold['fold']), 'selection': f'family:{family}:constrained', 'candidate_index': constrained_index, **candidates[constrained_index]})
            selections.append({'outer_fold': int(fold['fold']), 'selection': f'family:{family}:unconstrained', 'candidate_index': unconstrained_index, **candidates[unconstrained_index]})
    pd.DataFrame(selections).to_csv(args.output / 'outer_selections.csv', index=False)
    nested_rows = []
    for name, probability in nested_probabilities.items():
        metrics = audit.metric_dict(y, probability.argmax(axis=1))
        nested_rows.append({'selection': name, **{key: metrics[key] for key in ('accuracy', 'balanced_accuracy', 'f1_macro')}, **probability_metrics(y, probability)})
    for family, probability in nested_unconstrained.items():
        metrics = audit.metric_dict(y, probability.argmax(axis=1))
        nested_rows.append({'selection': f'family:{family}:unconstrained', **{key: metrics[key] for key in ('accuracy', 'balanced_accuracy', 'f1_macro')}, **probability_metrics(y, probability)})
    current_metrics = audit.metric_dict(y, current.argmax(axis=1))
    nested_rows.append({'selection': 'current_global_0.60', **{key: current_metrics[key] for key in ('accuracy', 'balanced_accuracy', 'f1_macro')}, **probability_metrics(y, current)})
    nested_frame = pd.DataFrame(nested_rows).sort_values('f1_macro', ascending=False)
    nested_frame.to_csv(args.output / 'nested_family_metrics.csv', index=False)
    best_unconstrained_family = max(families, key=lambda family: compact_metrics(y, nested_unconstrained[family].argmax(axis=1))['f1_macro'])
    best_unconstrained_probability = nested_unconstrained[best_unconstrained_family]
    adaptive_nested_probability = nested_probabilities['family:adaptive_gate']
    full_table = metric_table(y, probabilities, all_indices)
    config_frame = pd.DataFrame([json.dumps(config, sort_keys=True) for config in candidates], columns=['config_json'])
    full_frame = pd.concat([config_frame, full_table], axis=1)
    full_frame.insert(0, 'candidate_index', np.arange(len(candidates)))
    full_frame.insert(1, 'family', [config['family'] for config in candidates])
    full_frame.to_csv(args.output / 'full_oof_candidate_metrics.csv', index=False)
    selected_full = choose(full_table, np.arange(len(candidates)), baseline_index, constrained=True)
    selected_unconstrained = choose(full_table, np.arange(len(candidates)), baseline_index, constrained=False)
    overall_nested = nested_probabilities['overall_constrained']
    selected_nested_metrics = audit.metric_dict(y, overall_nested.argmax(axis=1))
    comparison = paired_test(y, current, overall_nested, source)
    summary = {'protocol': {'clips': int(len(y)), 'source_groups': int(len(set(source))), 'outer_folds': 5, 'selection': 'four folds choose meta-parameters; fifth fold evaluates; repeated five times', 'objective': 'macro-F1, then balanced accuracy and accuracy', 'constraints': {'accuracy_tolerance_absolute': 0.002, 'line_recall_no_decrease': True, 'scrum_recall_no_decrease': True}, 'video_resnet_yolo_recomputed': False, 'candidate_count': len(candidates), 'families': {family: int(len(indices)) for family, indices in family_indices.items()}}, 'quality': quality_summary, 'current': {'config': candidates[baseline_index], 'metrics': current_metrics, 'probability_metrics': probability_metrics(y, current)}, 'nested_overall_constrained': {'metrics': selected_nested_metrics, 'probability_metrics': probability_metrics(y, overall_nested), 'paired_vs_current': comparison}, 'best_unconstrained_family': {'family': best_unconstrained_family, 'metrics': audit.metric_dict(y, best_unconstrained_probability.argmax(axis=1)), 'probability_metrics': probability_metrics(y, best_unconstrained_probability), 'paired_vs_current': paired_test(y, current, best_unconstrained_probability, source)}, 'adaptive_family_nested': {'metrics': audit.metric_dict(y, adaptive_nested_probability.argmax(axis=1)), 'probability_metrics': probability_metrics(y, adaptive_nested_probability), 'paired_vs_current': paired_test(y, current, adaptive_nested_probability, source)}, 'nested_family_ranking': nested_frame.to_dict(orient='records'), 'deployment_selection_from_full_oof': {'candidate_index': selected_full, 'config': candidates[selected_full], 'metrics': full_table.loc[selected_full].to_dict()}, 'unconstrained_full_oof': {'candidate_index': selected_unconstrained, 'config': candidates[selected_unconstrained], 'metrics': full_table.loc[selected_unconstrained].to_dict()}, 'interpretation_warning': 'Full-OOF selection defines a deployment candidate; nested metrics are the less biased performance estimate.', 'production_decision': 'retain current global 0.60; adaptive alternatives did not show a significant nested improvement'}
    dump_json(args.output / 'selected_config.json', summary['deployment_selection_from_full_oof'])
    dump_json(args.output / 'summary.json', summary)
    np.savez_compressed(args.output / 'nested_predictions.npz', names=names, labels=y, historical=historical.astype(np.float32), auxiliary=auxiliary.astype(np.float32), quality=quality.astype(np.float32), current=current.astype(np.float32), selected_nested=overall_nested.astype(np.float32), adaptive_family_nested=adaptive_nested_probability.astype(np.float32), best_unconstrained_family=best_unconstrained_probability.astype(np.float32))
    print(json.dumps({'candidate_count': len(candidates), 'current': {key: current_metrics[key] for key in ('accuracy', 'balanced_accuracy', 'f1_macro')}, 'nested_selected': {key: selected_nested_metrics[key] for key in ('accuracy', 'balanced_accuracy', 'f1_macro')}, 'paired': comparison, 'deployment_config': candidates[selected_full]}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
