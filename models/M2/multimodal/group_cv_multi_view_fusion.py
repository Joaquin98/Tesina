"""Grouped evaluation of RGB dynamics + E02 quality + E04 tracking views."""
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    d = _load_model_result_array(HERE / 'rgb_dynamics_seed42/fusion_features.npz', allow_pickle=True)
    dyn = d['features'][:, :704].astype(np.float32)
    e04 = _load_model_result_array(HERE / 'multi_yolo_seed42/E04_tracking_smoothing.npy').astype(np.float32)
    e02 = _load_model_result_array(HERE / 'multi_yolo_seed42/E02_quality_mask.npy').astype(np.float32)
    X = np.concatenate([dyn, e04, e02], 1)
    y = d['labels'].astype(int)
    names = d['names'].astype(str)
    groups = pd.read_csv(_resolve_model_artifact(REPO / 'data/independent_evaluation/outputs/nested_source_manifest.csv')).set_index('video_name').loc[names].source_group.astype(str).to_numpy()
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=20260712)
    base = np.zeros((len(y), 3), np.float32)
    for fold in range(1, 6):
        p = pd.read_csv(_resolve_model_artifact(REPO / f'data/independent_evaluation/outputs/nested_evaluation/outer_{fold:02d}/outer_predictions.csv'))
        ix = {str(n): i for i, n in enumerate(names)}
        for r in p.itertuples():
            base[ix[str(r.video_name)]] = [r.probability_line, r.probability_other, r.probability_scrum]
    rows = []
    for C in (0.003, 0.01, 0.03):
        pyo = np.zeros_like(base)
        for fit, hold in outer.split(X, y, groups):
            sc = StandardScaler().fit(X[fit])
            m = LogisticRegression(C=C, class_weight='balanced', solver='saga', max_iter=600, tol=0.001, n_jobs=1, random_state=42).fit(sc.transform(X[fit]), y[fit])
            pyo[hold] = m.predict_proba(sc.transform(X[hold]))
        for alpha in (0.25, 0.4, 0.5, 0.6):
            p = (1 - alpha) * base + alpha * pyo
            rows.append({'C': C, 'alpha': alpha, 'pooled': met(y, p.argmax(1)), 'mean_fold_accuracy': float(np.mean([met(y[h], p[h].argmax(1))['accuracy'] for _, h in outer.split(X, y, groups)]))})
    rows.sort(key=lambda r: (r['mean_fold_accuracy'], r['pooled']['f1_macro']), reverse=True)
    out = {'ranking': rows, 'features': int(X.shape[1]), 'protocol': {'groups': 5, 'historical_predictions': 'nested outer checkpoints'}}
    (HERE / 'group_cv_multi_view_fusion_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
