"""Broad, reproducible audit of the frozen RGB+YOLO fusion.

This script never uses an external holdout to select hyperparameters.  It
reconstructs the five independently generated evaluation folds, fits every
classifier only on the other clips, and reports ablations, sensitivity,
paired bootstrap intervals, McNemar's test, per-class metrics, and a fair
natural train/validation/test check.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
from pathlib import Path
import json
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import QuantileTransformer
try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None
HERE = experiment_directory(__file__)
REPO = HERE.parents[1]
CLASS_NAMES = ['line', 'other', 'scrum']
CLASS_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}
MODERATE = {0: 1.2, 1: 0.65, 2: 1.8}

def metric_dict(y, pred):
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)), 'per_class': {CLASS_NAMES[i]: {'precision': float(p[i]), 'recall': float(r[i]), 'f1': float(f[i]), 'support': int(support[i])} for i in range(3)}, 'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}

def load_features():
    d = _load_model_result_array(HERE / 'rgb_dynamics_seed42/fusion_features.npz', allow_pickle=True)
    dyn = d['features'][:, :704].astype(np.float32)
    e04 = _load_model_result_array(HERE / 'multi_yolo_seed42/E04_tracking_smoothing.npy').astype(np.float32)
    e06 = _load_model_result_array(HERE / 'invariant_yolo_seed42/E06_rugby_descriptors.npy').astype(np.float32)
    X = np.concatenate([dyn, e04, e06], axis=1)
    names = d['names'].astype(str)
    y = d['labels'].astype(int)
    historical = np.exp(d['features'][:, -3:].astype(np.float32))
    if not np.allclose(historical.sum(1), 1.0, atol=0.0002):
        raise AssertionError('historical probabilities do not sum to one')
    return (X, names, y, historical)

def load_external(names, y):
    root = REPO / 'data/alternative_split_robustness'
    paths = sorted(_model_artifact_glob(root, 'final_fold_*_stream/outer_*/evaluation_predictions.csv', recursive=False))
    if len(paths) != 5:
        raise AssertionError(f'expected 5 external folds, got {len(paths)}')
    idx = {n: i for i, n in enumerate(names)}
    manifest = pd.read_csv(_resolve_model_artifact(study_csv_path(REPO / 'data/independent_evaluation/outputs/nested_source_manifest.csv')))
    source = manifest.set_index('video_name').loc[names, 'source_group'].astype(str).to_numpy()
    hist = np.zeros((len(names), 3), dtype=np.float32)
    external_y = np.full(len(names), -1, dtype=int)
    folds = []
    seen = set()
    for fold_no, path in enumerate(paths, 1):
        frame = pd.read_csv(_resolve_model_artifact(study_csv_path(path)))
        hold = np.asarray([idx[v] for v in frame.video_name], dtype=int)
        if seen.intersection(frame.video_name):
            raise AssertionError(f'external folds overlap in {path}')
        seen.update(frame.video_name)
        expected = frame.true_label.map(CLASS_TO_INT).to_numpy(dtype=int)
        if not np.array_equal(expected, y[hold]):
            raise AssertionError(f'label mismatch in external fold {fold_no}')
        hist[hold] = frame[['historical_prob_line', 'historical_prob_other', 'historical_prob_scrum']].to_numpy(dtype=np.float32)
        external_y[hold] = expected
        folds.append({'fold': fold_no, 'hold': hold, 'n': int(len(hold)), 'source_groups': sorted(set(source[hold]))})
    if len(seen) != len(names) or np.any(external_y < 0):
        raise AssertionError('external folds do not cover all feature rows exactly once')
    return (folds, hist, external_y, source)

def fit_quantile(train_x, train_y, hold_x, *, C=0.003, class_weight=MODERATE, n_quantiles=256):
    scaler = QuantileTransformer(n_quantiles=min(n_quantiles, len(train_x)), output_distribution='normal', random_state=42)
    z_train = scaler.fit_transform(train_x)
    z_hold = scaler.transform(hold_x)
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=700, random_state=42)
    clf.fit(z_train, train_y)
    return clf.predict_proba(z_hold)

def fit_on_transformed(train_z, train_y, hold_z, *, C=0.003, class_weight=MODERATE):
    """Fit a classifier after the fold's transformer has already been cached."""
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=700, random_state=42)
    clf.fit(train_z, train_y)
    return clf.predict_proba(hold_z)

def evaluate_external(X, y, names, folds, hist, source):
    feature_sets = {'rgb_dynamics': np.arange(0, 704), 'e04_e06': np.arange(704, 1686), 'rgb_e04': np.arange(0, 1411), 'rgb_e06': np.concatenate([np.arange(0, 704), np.arange(1411, 1686)]), 'rgb_e04_e06': np.arange(0, 1686)}
    fold_cache = []
    for fold in folds:
        hold = fold['hold']
        fit_mask = np.ones(len(y), dtype=bool)
        fit_mask[hold] = False
        scaler = QuantileTransformer(n_quantiles=min(256, int(fit_mask.sum())), output_distribution='normal', random_state=42)
        z_all = scaler.fit(X[fit_mask]).transform(X)
        fold_cache.append((hold, fit_mask, z_all))
    ablation_rows = []
    final_probs = np.zeros_like(hist)
    control_probs = {'none': np.zeros_like(hist), 'balanced': np.zeros_like(hist)}
    for set_name, columns in feature_sets.items():
        probs = np.zeros_like(hist)
        fold_metrics = []
        for hold, fit_mask, z_all in fold_cache:
            probs[hold] = fit_on_transformed(z_all[fit_mask][:, columns], y[fit_mask], z_all[hold][:, columns])
            fold_metrics.append(metric_dict(y[hold], probs[hold].argmax(1)))
        pooled = metric_dict(y, probs.argmax(1))
        ablation_rows.append({'model': set_name, 'pooled': pooled, 'mean_fold_accuracy': float(np.mean([m['accuracy'] for m in fold_metrics])), 'mean_fold_f1_macro': float(np.mean([m['f1_macro'] for m in fold_metrics])), 'folds': fold_metrics})
        if set_name == 'rgb_e04_e06':
            final_probs = probs
    for weight_name, weights in [('none', None), ('balanced', 'balanced')]:
        probs = np.zeros_like(hist)
        for hold, fit_mask, z_all in fold_cache:
            probs[hold] = fit_on_transformed(z_all[fit_mask], y[fit_mask], z_all[hold], class_weight=weights)
        control_probs[weight_name] = probs
    alphas = []
    for alpha in np.arange(0.0, 1.01, 0.1):
        fused = (1.0 - alpha) * hist + alpha * final_probs
        fold_acc = [accuracy_score(y[f['hold']], fused[f['hold']].argmax(1)) for f in folds]
        alphas.append({'alpha_yolo': round(float(alpha), 1), 'pooled': metric_dict(y, fused.argmax(1)), 'mean_fold_accuracy': float(np.mean(fold_acc)), 'min_fold_accuracy': float(np.min(fold_acc)), 'max_fold_accuracy': float(np.max(fold_acc))})
    weights_sensitivity = []
    for name, probs in [('none', control_probs['none']), ('moderate', final_probs), ('balanced', control_probs['balanced'])]:
        fused = 0.4 * hist + 0.6 * probs
        fold_acc = [accuracy_score(y[f['hold']], fused[f['hold']].argmax(1)) for f in folds]
        weights_sensitivity.append({'class_weight': name, 'pooled': metric_dict(y, fused.argmax(1)), 'mean_fold_accuracy': float(np.mean(fold_acc)), 'fold_accuracies': [float(v) for v in fold_acc]})
    quantile_sensitivity = []
    for nq in [64, 128, 256, 512]:
        probs = np.zeros_like(hist)
        for fold in folds:
            hold = fold['hold']
            fit_mask = np.ones(len(y), dtype=bool)
            fit_mask[hold] = False
            probs[hold] = fit_quantile(X[fit_mask], y[fit_mask], X[hold], n_quantiles=nq)
        fused = 0.4 * hist + 0.6 * probs
        fold_acc = [accuracy_score(y[f['hold']], fused[f['hold']].argmax(1)) for f in folds]
        quantile_sensitivity.append({'n_quantiles': nq, 'pooled': metric_dict(y, fused.argmax(1)), 'mean_fold_accuracy': float(np.mean(fold_acc))})
    hist_metrics = metric_dict(y, hist.argmax(1))
    final_fused = 0.4 * hist + 0.6 * final_probs
    final_metrics = metric_dict(y, final_fused.argmax(1))
    fold_rows = []
    for fold in folds:
        hold = fold['hold']
        h = metric_dict(y[hold], hist[hold].argmax(1))
        f = metric_dict(y[hold], final_fused[hold].argmax(1))
        fold_rows.append({'fold': fold['fold'], 'n': int(len(hold)), 'source_groups': fold['source_groups'], 'historical': h, 'fused': f, 'delta_accuracy': f['accuracy'] - h['accuracy'], 'delta_f1_macro': f['f1_macro'] - h['f1_macro']})
    yhat_h = hist.argmax(1)
    yhat_f = final_fused.argmax(1)
    delta = (yhat_f == y).astype(float) - (yhat_h == y).astype(float)
    rng = np.random.default_rng(20260813)
    B = 10000
    clip_boot = np.empty(B, dtype=np.float64)
    for b in range(B):
        sample = rng.integers(0, len(y), len(y))
        clip_boot[b] = delta[sample].mean()
    group_values = []
    for group in sorted(set(source)):
        mask = source == group
        group_values.append((int(mask.sum()), float(delta[mask].mean())))
    group_boot = np.empty(B, dtype=np.float64)
    sizes = np.asarray([v[0] for v in group_values], dtype=float)
    vals = np.asarray([v[1] for v in group_values], dtype=float)
    for b in range(B):
        sample = rng.integers(0, len(vals), len(vals))
        group_boot[b] = float(np.sum(sizes[sample] * vals[sample]) / np.sum(sizes[sample]))
    only_fused = int(np.sum((yhat_f != y) == False))
    fused_only = int(np.sum((yhat_f == y) & (yhat_h != y)))
    hist_only = int(np.sum((yhat_f != y) & (yhat_h == y)))
    p_value = None
    if binomtest is not None and fused_only + hist_only:
        p_value = float(binomtest(fused_only, fused_only + hist_only, 0.5, alternative='greater').pvalue)
    bootstrap = {'clip_delta_accuracy': float(delta.mean()), 'clip_ci95': [float(np.quantile(clip_boot, 0.025)), float(np.quantile(clip_boot, 0.975))], 'source_group_delta_accuracy': float(delta.mean()), 'source_group_ci95': [float(np.quantile(group_boot, 0.025)), float(np.quantile(group_boot, 0.975))], 'n_source_groups': len(group_values), 'fused_only_correct': fused_only, 'historical_only_correct': hist_only, 'mcnemar_one_sided_p': p_value}
    return {'historical': hist_metrics, 'fused': final_metrics, 'delta_accuracy': final_metrics['accuracy'] - hist_metrics['accuracy'], 'delta_f1_macro': final_metrics['f1_macro'] - hist_metrics['f1_macro'], 'folds': fold_rows, 'ablation': ablation_rows, 'alpha_sensitivity': alphas, 'class_weight_sensitivity': weights_sensitivity, 'quantile_resolution_sensitivity': quantile_sensitivity, 'bootstrap_and_paired_test': bootstrap, 'predictions': {'historical': hist, 'fused': final_fused, 'y': y}}

def natural_check(X, names, y, historical):
    root = CURRENT_DATASET
    idx = {n: i for i, n in enumerate(names)}
    splits = {}
    for key, filename in [('train', 'train_natural.csv'), ('validation', 'validation.csv'), ('test', 'test_natural.csv')]:
        frame = pd.read_csv(_resolve_model_artifact(study_csv_path(root / filename)))
        indices = np.asarray([idx[n] for n in frame.video_name], dtype=int)
        labels = frame.tag.map(CLASS_TO_INT).to_numpy(dtype=int)
        if not np.array_equal(labels, y[indices]):
            raise AssertionError(f'natural label mismatch: {key}')
        splits[key] = indices
    if set(splits['train']).intersection(splits['validation']) or set(splits['train']).intersection(splits['test']) or set(splits['validation']).intersection(splits['test']):
        raise AssertionError('natural splits overlap')
    if sum((len(v) for v in splits.values())) != len(names):
        raise AssertionError('natural splits do not cover all rows')
    train, val, test = (splits['train'], splits['validation'], splits['test'])
    q = QuantileTransformer(n_quantiles=min(256, len(train)), output_distribution='normal', random_state=42)
    q.fit(X[train])
    clf = LogisticRegression(C=0.003, class_weight=MODERATE, max_iter=700, random_state=42)
    clf.fit(q.transform(X[train]), y[train])
    learned = clf.predict_proba(q.transform(X))
    rows = []
    for name, ix in [('validation', val), ('test', test)]:
        hist_m = metric_dict(y[ix], historical[ix].argmax(1))
        fused = 0.4 * historical[ix] + 0.6 * learned[ix]
        fused_m = metric_dict(y[ix], fused.argmax(1))
        rows.append({'split': name, 'n': int(len(ix)), 'historical': hist_m, 'fused': fused_m, 'delta_accuracy': fused_m['accuracy'] - hist_m['accuracy']})
    return {'splits': {k: int(len(v)) for k, v in splits.items()}, 'results': rows, 'note': 'model fit only on natural train; validation/test are untouched'}

def main():
    X, names, y, historical = load_features()
    folds, external_hist, external_y, source = load_external(names, y)
    external = evaluate_external(X, external_y, names, folds, external_hist, source)
    natural = natural_check(X, names, y, historical)
    pred = external.pop('predictions')
    leakage = {'feature_rows': int(len(names)), 'unique_feature_names': int(len(set(names))), 'external_folds': len(folds), 'external_fold_rows': [int(f['n']) for f in folds], 'external_unique_rows': int(sum((f['n'] for f in folds))), 'external_name_overlap': 0, 'natural_split_rows': natural['splits'], 'natural_split_overlap': 0, 'external_label_match': True, 'fit_policy': 'each external classifier fit on all non-holdout clips; natural classifier fit only on train split'}
    out = {'protocol': {'external': '5 disjoint pre-generated source/temporal folds, 4,082 clips', 'natural': 'train 2,612 / validation 653 / test 817, name-aligned manifests', 'production_config': 'QuantileTransformer(256, normal) + LogisticRegression(C=.003, moderate weights) + 40% historical / 60% YOLO', 'random_seed': 20260813}, 'leakage_checks': leakage, 'external_audit': external, 'natural_train_only_audit': natural}
    out_path = HERE / 'audit_performance_summary.json'
    np.savez_compressed(HERE / 'audit_performance_predictions.npz', **pred)
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
