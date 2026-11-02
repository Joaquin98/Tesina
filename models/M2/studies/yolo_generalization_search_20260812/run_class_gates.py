"""Busca compuertas YOLO independientes para line y scrum."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json, os, sys
from pathlib import Path
import joblib, numpy as np
from sklearn.metrics import accuracy_score, f1_score
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data/player_geometry_fusion/pose_cache/frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M2.studies.yolo_resnet_improvement_suite_20260811 import run_suite
OUT = HERE / 'class_gates_seed42'
WEIGHTS = np.array([0.2, 0.2, 0.6])

def norm(p):
    p = np.clip(np.asarray(p, float), 1e-08, None)
    return p / p.sum(1, keepdims=True)

def qual(frame):
    vals = []
    for name in frame.video_name.astype(str):
        d = run_suite.load_clip(name)
        v = d['valid'].astype(bool)
        pc = d['person_conf'].astype(float)
        kc = d['keypoint_conf'].astype(float)
        cnt = v.sum(1)
        pm = np.divide(np.where(v, pc, 0).sum(1), np.maximum(cnt, 1))
        km = np.divide(np.where(v[..., None], kc, 0).sum((1, 2)), np.maximum(cnt * kc.shape[-1], 1))
        vals.append([pm.mean(), km.mean(), np.clip(cnt / 30, 0, 1).mean(), (cnt > 0).mean()])
    return np.asarray(vals)

def main():
    tr, va, te = R.common.load_base_splits(False, 'natural')
    yv, yt = (R.labels(va), R.labels(te))
    bv = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', va)
    bt = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', te)
    qv, qt = (qual(va), qual(te))
    gv = []
    gt = []
    for seed in (17, 42, 2026):
        root = REPO / 'data/historical_locked_yolo_residual_20260811' / f'results_seed{seed}'
        res = json.loads(_resolve_model_artifact(root / 'selected_result.json').read_text())
        art = joblib.load(_resolve_model_artifact(root / 'selected_model.joblib'))
        cfg = res['selected_configuration']
        rep = cfg['representation']
        xv = R.cached_features(va, 'validation', rep, OUT / 'cache')
        xt = R.cached_features(te, 'test', rep, OUT / 'cache')
        xv2 = art['pca'].transform(art['scaler'].transform(xv))
        xt2 = art['pca'].transform(art['scaler'].transform(xt))
        gv.append(norm(art['model'].predict_proba(xv2)))
        gt.append(norm(art['model'].predict_proba(xt2)))
    geomv = norm(np.average(gv, axis=0, weights=WEIGHTS))
    geomt = norm(np.average(gt, axis=0, weights=WEIGHTS))
    rows = []
    for conf in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        for qthr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for la in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
                for sa in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
                    av = (bv.max(1) < conf) & (qv[:, 0] >= qthr)
                    at = (bt.max(1) < conf) & (qt[:, 0] >= qthr)
                    v = bv.copy()
                    t = bt.copy()
                    v[av, 0] = (1 - la) * bv[av, 0] + la * geomv[av, 0]
                    v[av, 2] = (1 - sa) * bv[av, 2] + sa * geomv[av, 2]
                    v = norm(v)
                    t[at, 0] = (1 - la) * bt[at, 0] + la * geomt[at, 0]
                    t[at, 2] = (1 - sa) * bt[at, 2] + sa * geomt[at, 2]
                    t = norm(t)
                    rows.append({'val_f1': float(f1_score(yv, v.argmax(1), average='macro', zero_division=0)), 'val_acc': float(accuracy_score(yv, v.argmax(1))), 'test_f1': float(f1_score(yt, t.argmax(1), average='macro', zero_division=0)), 'test_acc': float(accuracy_score(yt, t.argmax(1))), 'conf': conf, 'qthr': qthr, 'line_alpha': la, 'scrum_alpha': sa, 'changed': int(av.sum())})
    rows = sorted(rows, key=lambda z: (z['val_f1'], z['val_acc']), reverse=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'ranking.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows[:20], indent=2), flush=True)
if __name__ == '__main__':
    main()
