"""FusiÃ³n YOLO con clasificadores binarios one-vs-rest por clase."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data/player_geometry_fusion/pose_cache/frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
ROOT = REPO / 'data/historical_locked_yolo_residual_20260811'
CACHE = ROOT / 'results_seed42' / 'feature_cache'
OUT = HERE / 'pairwise_classes_seed42'

def norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-08, None)
    return p / p.sum(1, keepdims=True)

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p.argmax(1))), 'f1_macro': float(f1_score(y, p.argmax(1), average='macro', zero_division=0))}

def feat(split):
    with _load_model_result_array(CACHE / 'E04_tracking_smoothing' / f'{split}.npz', allow_pickle=True) as z:
        return z['features'].astype(np.float32)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = R.common.load_base_splits(False, 'natural')
    ytr, yv, yt = map(R.labels, (tr, va, te))
    btr = R.historical_train_probabilities(tr, CACHE)
    bv = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', va)
    bt = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', te)
    xtr = feat('train')
    xv = feat('validation')
    xt = feat('test')
    sc = StandardScaler().fit(xtr)
    a = sc.transform(xtr)
    v = sc.transform(xv)
    t = sc.transform(xt)
    p = PCA(n_components=64, random_state=42).fit(a)
    a = p.transform(a)
    v = p.transform(v)
    t = p.transform(t)
    rows = []
    for C in (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 1.0):
        models = []
        for cls in (0, 2):
            target = (ytr == cls).astype(int)
            m = LogisticRegression(C=C, class_weight='balanced', max_iter=3000, random_state=42).fit(np.c_[a, np.log(norm(btr))], target)
            models.append(m)

        def predict(x, b):
            hist = np.log(norm(b))
            scores = np.c_[hist[:, 0], hist[:, 1], hist[:, 2]]
            for m, cls in zip(models, (0, 2)):
                q = m.predict_proba(np.c_[x, hist])[:, 1]
                scores[:, cls] += np.log(np.clip(q, 1e-06, 1))
                scores[:, 1] += np.log(np.clip(1 - q, 1e-06, 1))
            return np.exp(scores - scores.max(1, keepdims=True))
        pv = predict(v, bv)
        pt = predict(t, bt)
        rows.append({'C': C, 'validation': met(yv, pv), 'test': met(yt, pt)})
    out = {'historical_test': met(yt, bt), 'candidates': rows, 'selected': max(rows, key=lambda r: (r['validation']['f1_macro'], r['validation']['accuracy'])), 'protocol': {'selection': 'natural validation only', 'test_locked': True}}
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)
if __name__ == '__main__':
    main()
