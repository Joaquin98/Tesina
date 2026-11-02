"""Blends RGB-dynamics classifier with the frozen three-seed YOLO ensemble."""
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
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data/player_geometry_fusion/pose_cache/frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M2.studies.historical_locked_yolo_residual_20260811.ensemble_seeds import probabilities_for_seed

def metric(y, p):
    q = p.argmax(1)
    return {'accuracy': float(accuracy_score(y, q)), 'f1_macro': float(f1_score(y, q, average='macro', zero_division=0))}

def main():
    tr, va, te = R.common.load_base_splits(False, 'natural')
    yv, yt = (R.labels(va), R.labels(te))
    bv = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', va)
    bt = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', te)
    gv = []
    gt = []
    for seed in (17, 42, 2026):
        pv, _ = probabilities_for_seed(seed, 'validation', bv)
        pt, _ = probabilities_for_seed(seed, 'test', bt)
        gv.append(pv)
        gt.append(pt)
    ensv = np.average(gv, axis=0, weights=[0.2, 0.2, 0.6])
    enst = np.average(gt, axis=0, weights=[0.2, 0.2, 0.6])
    rows = []
    for path in sorted(_model_artifact_glob(HERE / 'rgb_dynamics_seed42', '*_probabilities.npz', recursive=False)):
        name = path.stem.replace('_probabilities', '')
        d = _load_model_result_array(path)
        pv = d['validation']
        pt = d['test']
        for a in np.linspace(0, 1, 101):
            qv = (1 - a) * ensv + a * pv
            qt = (1 - a) * enst + a * pt
            rows.append({'model': name, 'alpha': float(a), 'validation': metric(yv, qv), 'test': metric(yt, qt)})
    rows = sorted(rows, key=lambda r: (r['validation']['f1_macro'], r['validation']['accuracy']), reverse=True)
    result = {'ensemble_validation': metric(yv, ensv), 'ensemble_test': metric(yt, enst), 'ranking': rows}
    (HERE / 'rgb_dynamics_blend_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'ensemble_test': result['ensemble_test'], 'top': rows[:15]}, indent=2), flush=True)
if __name__ == '__main__':
    main()
