"""Evaluación agrupada: clips de un mismo origen nunca cruzan train/test."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M0.studies.natural_with_human_review import run_experiment as common
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M2.studies.historical_locked_yolo_residual_20260811.ensemble_seeds import probabilities_for_seed
FEATURES = HERE / 'rgb_dynamics_seed42' / 'fusion_features.npz'
MANIFEST = REPO / 'data/independent_evaluation/outputs/nested_source_manifest.csv'

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    d = _load_model_result_array(FEATURES, allow_pickle=True)
    full_X = d['features'].astype(np.float32)
    X = full_X[:, :-3]
    y = d['labels'].astype(int)
    names = d['names'].astype(str)
    manifest = pd.read_csv(_resolve_model_artifact(MANIFEST)).set_index('video_name').loc[names]
    groups = manifest.source_group.astype(str).to_numpy()
    tr, va, te = common.load_base_splits(False, 'natural')
    base_tr = R.historical_train_probabilities(tr, REPO / 'data/historical_locked_yolo_residual_20260811/results_seed42')
    base_v = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', va)
    base_t = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', te)
    base = np.concatenate([base_tr, base_v, base_t])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260712)
    rows = []
    for C in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        for cwname, cw in [('none', None), ('balanced', 'balanced'), ('moderate', {0: 1.2, 1: 0.65, 2: 1.8})]:
            pred = np.zeros(len(y), int)
            fold_metrics = []
            for fold_id, (fit, hold) in enumerate(cv.split(X, y, groups)):
                sc = StandardScaler().fit(X[fit])
                m = LogisticRegression(C=C, class_weight=cw, max_iter=3000, random_state=42).fit(sc.transform(X[fit]), y[fit])
                pred[hold] = m.predict(sc.transform(X[hold]))
                fold_metrics.append({'fold': fold_id, 'n': int(len(hold)), **met(y[hold], pred[hold])})
            rows.append({'model': f'logit_C{C}_{cwname}', 'pooled': met(y, pred), 'folds': fold_metrics})
    baseline = {'historical_in_sample_not_group_valid': met(y, base.argmax(1))}
    rows.sort(key=lambda r: r['pooled']['f1_macro'], reverse=True)
    out = {'baseline': baseline, 'ranking': rows, 'protocol': {'folds': 5, 'group_column': 'source_group', 'group_count': int(pd.Series(groups).nunique()), 'no_group_overlap': True}}
    (HERE / 'group_cv_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'baseline': baseline, 'top': rows[:10]}, indent=2), flush=True)
if __name__ == '__main__':
    main()
