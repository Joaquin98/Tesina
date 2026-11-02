"""Train the fixed, leakage-safe YOLO fusion used for deployment.

The hyperparameters are fixed before fitting the final natural training set:
RGB dynamics + E04 tracking/smoothing + E06 rugby descriptors, logistic
regression C=.01 without class reweighting, and a 40% YOLO probability blend.
This is an operational model, not a new test-selection step.
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
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
HERE = experiment_directory(__file__)

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    d = _load_model_result_array(HERE / 'rgb_dynamics_seed42/fusion_features.npz', allow_pickle=True)
    dyn = d['features'][:, :704].astype(np.float32)
    hist = np.exp(d['features'][:, -3:].astype(np.float32))
    e04 = _load_model_result_array(HERE / 'multi_yolo_seed42/E04_tracking_smoothing.npy').astype(np.float32)
    e06 = _load_model_result_array(HERE / 'invariant_yolo_seed42/E06_rugby_descriptors.npy').astype(np.float32)
    X = np.concatenate([dyn, e04, e06], 1)
    y = d['labels'].astype(int)
    ntr, nv = (int(d['train_count']), int(d['validation_count']))
    scaler = StandardScaler().fit(X[:ntr])
    model = LogisticRegression(C=0.01, class_weight=None, solver='lbfgs', max_iter=1200, random_state=42)
    model.fit(scaler.transform(X[:ntr]), y[:ntr])
    outdir = HERE / 'deployment_fusion_e04_e06_seed42'
    outdir.mkdir(exist_ok=True)
    joblib.dump(scaler, outdir / 'scaler.joblib')
    joblib.dump(model, outdir / 'logistic.joblib')
    py = model.predict_proba(scaler.transform(X))
    for name, sl in [('validation', slice(ntr, ntr + nv)), ('test_natural', slice(-817, None))]:
        p = 0.6 * hist[sl] + 0.4 * py[sl]
        print(name, met(y[sl], p.argmax(1)))
    summary = {'features': '704 RGB dynamics + 707 E04 + 275 E06', 'feature_dim': int(X.shape[1]), 'train_count': ntr, 'classifier': {'type': 'LogisticRegression', 'C': 0.01, 'class_weight': None}, 'blend': {'historical': 0.6, 'yolo': 0.4}, 'artifacts': ['scaler.joblib', 'logistic.joblib'], 'note': 'Deployment artifact; nested grouped metrics are in nested_group_invariant_e04e06_summary.json'}
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
