"""Grouped CV for RGB dynamics plus the two cached YOLO views (E02/E04)."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    d = _load_model_result_array(HERE / 'rgb_dynamics_seed42' / 'fusion_features.npz', allow_pickle=True)
    dyn = d['features'][:, :704].astype(np.float32)
    y = d['labels'].astype(int)
    names = d['names'].astype(str)
    reps = [_load_model_result_array(HERE / 'multi_yolo_seed42' / f'{n}.npy').astype(np.float32) for n in ('E02_quality_mask', 'E04_tracking_smoothing')]
    X = np.concatenate([dyn, *reps], axis=1)
    manifest = pd.read_csv(_resolve_model_artifact(REPO / 'data/independent_evaluation/outputs/nested_source_manifest.csv')).set_index('video_name').loc[names]
    groups = manifest.source_group.astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260712)
    rows = []
    for C in (0.001, 0.003, 0.01):
        for cwname, cw in (('balanced', 'balanced'), ('moderate', {0: 1.2, 1: 0.65, 2: 1.8})):
            pred = np.zeros(len(y), dtype=int)
            folds = []
            for fold, (fit, hold) in enumerate(cv.split(X, y, groups)):
                sc = StandardScaler().fit(X[fit])
                m = OneVsRestClassifier(LogisticRegression(C=C, class_weight=cw, solver='liblinear', max_iter=500, random_state=42), n_jobs=1)
                m.fit(sc.transform(X[fit]), y[fit])
                pred[hold] = m.predict(sc.transform(X[hold]))
                folds.append({'fold': fold, 'n': int(len(hold)), **met(y[hold], pred[hold])})
            rows.append({'model': f'logit_C{C}_{cwname}', 'pooled': met(y, pred), 'folds': folds})
    rows.sort(key=lambda r: (r['pooled']['f1_macro'], r['pooled']['accuracy']), reverse=True)
    out = {'ranking': rows, 'protocol': {'folds': 5, 'group_column': 'source_group', 'group_count': int(pd.Series(groups).nunique()), 'features': ['rgb_dynamics', 'E02_quality_mask', 'E04_tracking_smoothing'], 'historical_logits_excluded': True}}
    (HERE / 'group_cv_multi_yolo_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'top': rows[:8], 'feature_dim': int(X.shape[1])}, indent=2), flush=True)
if __name__ == '__main__':
    main()
