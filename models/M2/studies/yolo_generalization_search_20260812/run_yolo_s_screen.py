"""Screening de YOLO11s-320 sobre el cache completo, cuando termina."""
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
os.environ['PLAYER_GEOMETRY_LENGTH'] = '50'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data/player_geometry_fusion/pose_cache/frames_50_yolo11s_320')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M2.studies.yolo_resnet_improvement_suite_20260811 import run_suite as suite

def _load_s_clip(name: str) -> dict[str, np.ndarray]:
    safe = 'yolo11s-pose_pt'
    path = REPO / 'data/player_geometry_fusion/pose_cache/frames_50_yolo11s_320' / safe / f'{Path(name).stem}.npz'
    with _load_model_result_array(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}
suite.load_clip = _load_s_clip
suite.SEQUENCE_LENGTH = 50
feature_matrix = suite.feature_matrix
OUT = HERE / 'yolo11s_screen_seed42'
ROOT = REPO / 'data/historical_locked_yolo_residual_20260811'
CACHE = ROOT / 'results_seed42'

def norm(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-08, None)
    return x / x.sum(1, keepdims=True)

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p.argmax(1))), 'f1_macro': float(f1_score(y, p.argmax(1), average='macro', zero_division=0))}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = R.common.load_base_splits(False, 'natural')
    yt, yv, ys = map(R.labels, (tr, va, te))
    bases = {'train': R.historical_train_probabilities(tr, CACHE), 'validation': R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', va), 'test': R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', te)}
    rows = []
    for rep in ('E02_quality_mask', 'E04_tracking_smoothing', 'E06_rugby_descriptors', 'E09_player_graph'):
        xs = {}
        for name, frame in [('train', tr), ('validation', va), ('test', te)]:
            p = OUT / f'{rep}_{name}.npy'
            if _resolve_model_artifact(p).exists():
                x = _load_model_result_array(p)
            else:
                x, _, _ = feature_matrix(frame, rep)
                np.save(p, x)
            xs[name] = x
        sc = StandardScaler().fit(xs['train'])
        a = sc.transform(xs['train'])
        b = sc.transform(xs['validation'])
        c = sc.transform(xs['test'])
        pca = PCA(n_components=min(64, a.shape[1], a.shape[0] - 1), random_state=42).fit(a)
        a = pca.transform(a)
        b = pca.transform(b)
        c = pca.transform(c)
        for mode in ('logit', 'prob'):
            ba = np.log(norm(bases['train'])) if mode == 'logit' else bases['train']
            bv = np.log(norm(bases['validation'])) if mode == 'logit' else bases['validation']
            bt = np.log(norm(bases['test'])) if mode == 'logit' else bases['test']
            for C in (0.01, 0.05, 0.25, 1.0):
                xtr = np.concatenate([ba, a], 1)
                xv = np.concatenate([bv, b], 1)
                xt = np.concatenate([bt, c], 1)
                model = LogisticRegression(C=C, class_weight='balanced', max_iter=3000)
                model.fit(xtr, yt)
                pv = model.predict_proba(xv)
                pt = model.predict_proba(xt)
                rows.append({'representation': rep, 'mode': mode, 'C': C, 'validation': met(yv, pv), 'test': met(ys, pt)})
    out = {'historical_test': met(ys, bases['test']), 'candidates': rows}
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)
if __name__ == '__main__':
    main()
