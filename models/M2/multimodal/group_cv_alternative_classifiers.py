"""Grouped audit of lower-variance classifiers on RGB+YOLO features."""
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    d = _load_model_result_array(HERE / 'rgb_dynamics_seed42' / 'fusion_features.npz', allow_pickle=True)
    X = d['features'][:, :-3].astype(np.float32)
    y = d['labels'].astype(int)
    names = d['names'].astype(str)
    manifest = pd.read_csv(_resolve_model_artifact(REPO / 'data/independent_evaluation/outputs/nested_source_manifest.csv')).set_index('video_name').loc[names]
    groups = manifest.source_group.astype(str).to_numpy()
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=20260712)
    specs = [('logit', lambda: LogisticRegression(C=0.003, class_weight='balanced', max_iter=1000)), ('linear_svc', lambda: LinearSVC(C=0.01, class_weight='balanced', dual='auto', max_iter=2000)), ('rbf_svc_1', lambda: SVC(C=1, class_weight='balanced', gamma='scale')), ('rbf_svc_3', lambda: SVC(C=3, class_weight='balanced', gamma='scale'))]
    rows = []
    for name, make in specs:
        pred = np.zeros(len(y), dtype=int)
        folds = []
        for fold, (fit, hold) in enumerate(cv.split(X, y, groups)):
            sc = StandardScaler().fit(X[fit])
            xx = sc.transform(X[fit])
            xt = sc.transform(X[hold])
            if name.startswith('rbf'):
                pca = PCA(n_components=64, random_state=42, svd_solver='randomized').fit(xx)
                xx = pca.transform(xx)
                xt = pca.transform(xt)
            m = make()
            m.fit(xx, y[fit])
            pred[hold] = m.predict(xt)
            folds.append({'fold': fold, 'n': int(len(hold)), **met(y[hold], pred[hold])})
        rows.append({'model': name, 'pooled': met(y, pred), 'folds': folds})
    out = {'ranking': sorted(rows, key=lambda r: (r['pooled']['f1_macro'], r['pooled']['accuracy']), reverse=True), 'protocol': {'features': 'rgb_dynamics + E04 YOLO; historical logits excluded', 'group_column': 'source_group', 'folds': 5}}
    (HERE / 'group_cv_alternative_classifiers_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
