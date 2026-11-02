"""Independent grouped-split check for the fixed quantile YOLO+RGB fusion.

The five evaluation files come from a separately generated source/temporal
split.  No hyperparameter is selected here: the configuration is frozen to
the one reported by group_cv_fixed_quantile.py (QuantileTransformer,
LogisticRegression C=.003, moderate class weights, YOLO alpha=.6).
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)

def metrics(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    data = _load_model_result_array(HERE / 'rgb_dynamics_seed42/fusion_features.npz', allow_pickle=True)
    dyn = data['features'][:, :704].astype(np.float32)
    labels = data['labels'].astype(int)
    names = data['names'].astype(str)
    e04 = _load_model_result_array(HERE / 'multi_yolo_seed42/E04_tracking_smoothing.npy').astype(np.float32)
    e06 = _load_model_result_array(HERE / 'invariant_yolo_seed42/E06_rugby_descriptors.npy').astype(np.float32)
    X = np.concatenate([dyn, e04, e06], axis=1)
    index = {name: i for i, name in enumerate(names)}
    label_index = {'line': 0, 'other': 1, 'scrum': 2}
    fold_files = sorted(_model_artifact_glob(REPO / 'data/alternative_split_robustness', 'final_fold_*_stream/outer_*/evaluation_predictions.csv', recursive=False))
    if len(fold_files) != 5:
        raise RuntimeError(f'expected five independent folds, found {len(fold_files)}')
    all_rows = []
    folds = []
    for fold_file in fold_files:
        frame = pd.read_csv(_resolve_model_artifact(fold_file))
        folds.append(frame)
        all_rows.append(frame)
    all_eval = pd.concat(all_rows, ignore_index=True)
    if all_eval.video_name.nunique() != len(all_eval):
        raise RuntimeError('independent evaluation folds overlap')
    missing = sorted(set(names) - set(all_eval.video_name))
    extra = sorted(set(all_eval.video_name) - set(names))
    if missing or extra:
        raise RuntimeError(f'feature/evaluation name mismatch: missing={len(missing)} extra={len(extra)}')
    hist_prob = np.zeros((len(names), 3), dtype=np.float32)
    for row in all_eval.itertuples(index=False):
        i = index[row.video_name]
        hist_prob[i] = [row.historical_prob_line, row.historical_prob_other, row.historical_prob_scrum]
    y_eval = np.asarray([label_index[row.true_label] for row in all_eval.itertuples(index=False)], dtype=int)
    y_all = labels
    configs = [('fixed_quantile_moderate_a06', 0.003, {0: 1.2, 1: 0.65, 2: 1.8}, 0.6)]
    rows = []
    for config_name, C, class_weight, alpha in configs:
        pred = np.zeros((len(names), 3), dtype=np.float32)
        fold_results = []
        cursor = 0
        for fold_no, frame in enumerate(folds, start=1):
            hold = np.asarray([index[v] for v in frame.video_name], dtype=int)
            fit_mask = np.ones(len(names), dtype=bool)
            fit_mask[hold] = False
            scaler = QuantileTransformer(n_quantiles=min(256, int(fit_mask.sum())), output_distribution='normal', random_state=42)
            train_x = scaler.fit_transform(X[fit_mask])
            hold_x = scaler.transform(X[hold])
            if 'power' in config_name:
                from sklearn.preprocessing import PowerTransformer
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
                train_x = scaler.fit_transform(X[fit_mask])
                hold_x = scaler.transform(X[hold])
            clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=700, n_jobs=-1, random_state=42).fit(train_x, y_all[fit_mask])
            pred[hold] = clf.predict_proba(hold_x)
            hold_y = y_all[hold]
            fused = (1.0 - alpha) * hist_prob[hold] + alpha * pred[hold]
            fold_results.append({'fold': fold_no, 'n': int(len(hold)), 'historical': metrics(hold_y, hist_prob[hold].argmax(1)), 'yolo_fusion': metrics(hold_y, fused.argmax(1)), 'delta_accuracy': float(metrics(hold_y, fused.argmax(1))['accuracy'] - metrics(hold_y, hist_prob[hold].argmax(1))['accuracy']), 'delta_f1_macro': float(metrics(hold_y, fused.argmax(1))['f1_macro'] - metrics(hold_y, hist_prob[hold].argmax(1))['f1_macro'])})
        hold_all = np.asarray([index[v] for v in all_eval.video_name], dtype=int)
        fused_all = (1.0 - alpha) * hist_prob[hold_all] + alpha * pred[hold_all]
        hist_metrics = metrics(y_eval, hist_prob[hold_all].argmax(1))
        fused_metrics = metrics(y_eval, fused_all.argmax(1))
        rows.append({'config': config_name, 'C': C, 'class_weight': class_weight, 'alpha': alpha, 'pooled_historical': hist_metrics, 'pooled_yolo_fusion': fused_metrics, 'pooled_delta_accuracy': fused_metrics['accuracy'] - hist_metrics['accuracy'], 'pooled_delta_f1_macro': fused_metrics['f1_macro'] - hist_metrics['f1_macro'], 'mean_fold_delta_accuracy': float(np.mean([r['delta_accuracy'] for r in fold_results])), 'mean_fold_delta_f1_macro': float(np.mean([r['delta_f1_macro'] for r in fold_results])), 'folds': fold_results})
    out = {'protocol': {'description': 'Independent source/temporal split; each evaluation fold is disjoint from training groups', 'n_clips': int(len(all_eval)), 'n_folds': 5, 'features': 'RGB dynamics + E04 tracking smoothing + E06 rugby descriptors', 'historical_predictions': 'precomputed alternative-split outer checkpoints', 'selection': 'frozen; no metric-driven tuning in this script', 'selection_provenance': 'configuration came from the earlier current-split exploratory screen; nested current-split result remains authoritative for unbiased selection'}, 'configs': rows}
    out_path = HERE / 'cross_split_fixed_quantile_summary.json'
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
