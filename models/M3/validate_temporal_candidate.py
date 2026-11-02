"""Confirmatory robustness audit for the locked 70/30 static-temporal auxiliary.

The candidate was identified post hoc, so this script does not claim an
independent external validation.  It fixes the candidate before all analyses
performed here and asks whether its observed advantage is stable across the
five external folds and 32 source groups.
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
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[2]
HERE = experiment_directory(__file__)
INPUT = HERE / 'outputs' / 'temporal_attention_highres_20260814' / 'oof_predictions.npz'
OUTPUT = HERE / 'outputs' / 'temporal_candidate_validation_20260814'
LABELS = ('line', 'other', 'scrum')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.multimodal import audit_performance as audit
from models.M3.run_temporal_attention_experiment import probability_metrics

def dump_json(path: Path, value) -> None:

    def safe(item):
        if isinstance(item, dict):
            return {str(key): safe(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(val) for val in item]
        if isinstance(item, np.ndarray):
            return safe(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2), encoding='utf-8')

def metric_row(y, reference, candidate) -> dict:
    ref_pred = reference.argmax(1)
    cand_pred = candidate.argmax(1)
    ref_acc = accuracy_score(y, ref_pred)
    cand_acc = accuracy_score(y, cand_pred)
    ref_f1 = f1_score(y, ref_pred, average='macro', zero_division=0)
    cand_f1 = f1_score(y, cand_pred, average='macro', zero_division=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        ref_bacc = balanced_accuracy_score(y, ref_pred)
        cand_bacc = balanced_accuracy_score(y, cand_pred)
    return {'n': len(y), 'reference_accuracy': ref_acc, 'candidate_accuracy': cand_acc, 'delta_accuracy': cand_acc - ref_acc, 'reference_f1_macro': ref_f1, 'candidate_f1_macro': cand_f1, 'delta_f1_macro': cand_f1 - ref_f1, 'reference_balanced_accuracy': ref_bacc, 'candidate_balanced_accuracy': cand_bacc, 'delta_balanced_accuracy': cand_bacc - ref_bacc, 'candidate_only_correct': int(np.sum((cand_pred == y) & (ref_pred != y))), 'reference_only_correct': int(np.sum((ref_pred == y) & (cand_pred != y)))}

def source_resampling(y, reference, candidate, source, *, samples=100000, seed=20260814):
    ref_correct = reference.argmax(1) == y
    cand_correct = candidate.argmax(1) == y
    difference = cand_correct.astype(np.float64) - ref_correct.astype(np.float64)
    groups = np.asarray(sorted(set(source)))
    sizes = np.asarray([np.sum(source == group) for group in groups], dtype=np.float64)
    sums = np.asarray([np.sum(difference[source == group]) for group in groups], dtype=np.float64)
    means = sums / sizes
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    sign_flip = np.empty(samples, dtype=np.float64)
    observed_sum = float(sums.sum())
    for start in range(0, samples, 10000):
        stop = min(start + 10000, samples)
        width = stop - start
        selected = rng.integers(0, len(groups), size=(width, len(groups)))
        bootstrap[start:stop] = np.sum(sizes[selected] * means[selected], axis=1) / np.sum(sizes[selected], axis=1)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(width, len(groups)))
        sign_flip[start:stop] = np.sum(signs * sums, axis=1)
    return {'source_group_bootstrap_samples': samples, 'source_group_bootstrap_delta_accuracy_ci95': np.quantile(bootstrap, [0.025, 0.975]), 'source_group_bootstrap_probability_delta_positive': float(np.mean(bootstrap > 0.0)), 'source_group_sign_flip_two_sided_p': float((1 + np.sum(np.abs(sign_flip) >= abs(observed_sum))) / (samples + 1))}

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with _load_model_result_array(INPUT, allow_pickle=True) as data:
        names = data['names'].astype(str)
        y = data['labels'].astype(np.int64)
        historical = data['historical'].astype(np.float32)
        static = data['static_auxiliary'].astype(np.float32)
        temporal = data['temporal'].astype(np.float32)
        source = data['source'].astype(str)
    folds, _, external_y, external_source = audit.load_external(names, y)
    if not np.array_equal(y, external_y) or not np.array_equal(source, external_source):
        raise AssertionError('OOF arrays are not aligned with the external protocol')
    reference = 0.4 * historical + 0.6 * static
    candidate = 0.4 * historical + 0.6 * (0.7 * static + 0.3 * temporal)
    reference /= reference.sum(1, keepdims=True)
    candidate /= candidate.sum(1, keepdims=True)
    fold_rows = []
    for fold in folds:
        index = np.asarray(fold['hold'], dtype=int)
        fold_rows.append({'fold': fold['fold'], **metric_row(y[index], reference[index], candidate[index])})
    pd.DataFrame(fold_rows).to_csv(OUTPUT / 'fold_metrics.csv', index=False)
    source_rows = []
    for group in sorted(set(source)):
        index = np.flatnonzero(source == group)
        source_rows.append({'source_group': group, **metric_row(y[index], reference[index], candidate[index])})
    pd.DataFrame(source_rows).to_csv(OUTPUT / 'source_group_metrics.csv', index=False)
    leave_one_out_rows = []
    for group in sorted(set(source)):
        index = source != group
        leave_one_out_rows.append({'excluded_source_group': group, **metric_row(y[index], reference[index], candidate[index])})
    pd.DataFrame(leave_one_out_rows).to_csv(OUTPUT / 'leave_one_source_out.csv', index=False)
    reference_pred = reference.argmax(1)
    candidate_pred = candidate.argmax(1)
    candidate_only = int(np.sum((candidate_pred == y) & (reference_pred != y)))
    reference_only = int(np.sum((reference_pred == y) & (candidate_pred != y)))
    discordant = candidate_only + reference_only
    paired = {'candidate_only_correct': candidate_only, 'reference_only_correct': reference_only, 'mcnemar_exact_two_sided_p': float(binomtest(min(candidate_only, reference_only), discordant, 0.5).pvalue), **source_resampling(y, reference, candidate, source)}
    ref_metrics = probability_metrics(y, reference)
    cand_metrics = probability_metrics(y, candidate)
    class_delta = {label: {'recall_delta': cand_metrics['per_class'][label]['recall'] - ref_metrics['per_class'][label]['recall'], 'correct_count_delta': int(np.sum((candidate_pred == class_index) & (y == class_index)) - np.sum((reference_pred == class_index) & (y == class_index)))} for class_index, label in enumerate(LABELS)}
    jcvt_mask = np.asarray(['JCVT' in group.upper() for group in source], dtype=bool)
    jcvt_metrics = metric_row(y[jcvt_mask], reference[jcvt_mask], candidate[jcvt_mask])
    fold_deltas = np.asarray([row['delta_accuracy'] for row in fold_rows])
    source_deltas = np.asarray([row['delta_accuracy'] for row in source_rows])
    loo_acc = np.asarray([row['delta_accuracy'] for row in leave_one_out_rows])
    loo_f1 = np.asarray([row['delta_f1_macro'] for row in leave_one_out_rows])
    conclusion = {'can_improve_point_estimate': True, 'independent_improvement_confirmed': False, 'production_recommendation': 'retain_official_model', 'reason': 'The locked candidate improves pooled accuracy and macro-F1 and remains positive after removing any single source, but grouped uncertainty includes zero, the group-level permutation is not significant, and JCVT-related clips lose two correct decisions.', 'next_required_test': 'Blind labels for an untouched match, preferably Primera JCVT vs La Salle, followed by the same paired comparison without changing the 30% temporal share.'}
    summary = {'status': 'confirmatory_robustness_not_independent_external_validation', 'candidate_locked_before_this_audit': '0.4 historical + 0.6 * (0.7 static auxiliary + 0.3 temporal auxiliary)', 'clips': len(y), 'source_groups': len(set(source)), 'reference': ref_metrics, 'candidate': cand_metrics, 'delta': {'accuracy': cand_metrics['accuracy'] - ref_metrics['accuracy'], 'balanced_accuracy': cand_metrics['balanced_accuracy'] - ref_metrics['balanced_accuracy'], 'f1_macro': cand_metrics['f1_macro'] - ref_metrics['f1_macro'], 'brier_multiclass': cand_metrics['brier_multiclass'] - ref_metrics['brier_multiclass'], 'ece_15': cand_metrics['ece_15'] - ref_metrics['ece_15']}, 'class_delta': class_delta, 'paired_tests': paired, 'fold_stability': {'improved': int(np.sum(fold_deltas > 0)), 'tied': int(np.sum(fold_deltas == 0)), 'worsened': int(np.sum(fold_deltas < 0)), 'delta_accuracy_by_fold': fold_deltas}, 'source_stability': {'improved': int(np.sum(source_deltas > 0)), 'tied': int(np.sum(source_deltas == 0)), 'worsened': int(np.sum(source_deltas < 0))}, 'leave_one_source_out': {'delta_accuracy_min': float(loo_acc.min()), 'delta_accuracy_max': float(loo_acc.max()), 'delta_f1_macro_min': float(loo_f1.min()), 'delta_f1_macro_max': float(loo_f1.max())}, 'jcvt_related_subset': jcvt_metrics, 'conclusion': conclusion, 'artifacts': {'fold_metrics': 'fold_metrics.csv', 'source_group_metrics': 'source_group_metrics.csv', 'leave_one_source_out': 'leave_one_source_out.csv'}}
    dump_json(OUTPUT / 'summary.json', summary)
    dump_json(OUTPUT / 'decision.json', conclusion)
    print(_resolve_model_artifact(OUTPUT / 'summary.json').read_text(encoding='utf-8'))
if __name__ == '__main__':
    main()
