"""Busca mezcla global, por clase y con compuerta del clasificador RGB-dinamico."""
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
from models.M0.studies.natural_with_human_review import run_experiment as common
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M2.studies.historical_locked_yolo_residual_20260811.ensemble_seeds import probabilities_for_seed

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p.argmax(1))), 'f1_macro': float(f1_score(y, p.argmax(1), average='macro', zero_division=0))}

def norm(x):
    x = np.clip(x, 1e-08, None)
    return x / x.sum(1, keepdims=True)

def main():
    tr, va, te = common.load_base_splits(False, 'natural')
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
    ev = np.average(gv, axis=0, weights=[0.2, 0.2, 0.6])
    et = np.average(gt, axis=0, weights=[0.2, 0.2, 0.6])
    results = []
    for f in sorted(_model_artifact_glob(HERE / 'rgb_dynamics_seed42', 'logit_*_probabilities.npz', recursive=False)):
        name = f.stem.replace('_probabilities', '')
        d = _load_model_result_array(f)
        dv, dt = (d['validation'], d['test'])
        for a in np.linspace(0, 1, 101):
            qv = norm((1 - a) * ev + a * dv)
            qt = norm((1 - a) * et + a * dt)
            results.append({'model': name, 'kind': 'global', 'alpha': float(a), 'validation': met(yv, qv), 'test': met(yt, qt)})
        grid = np.linspace(0, 1, 11)
        for av in grid:
            for ao in grid:
                for ass in grid:
                    qv = norm(ev * np.array([1 - av, 1 - ao, 1 - ass]) + dv * np.array([av, ao, ass]))
                    qt = norm(et * np.array([1 - av, 1 - ao, 1 - ass]) + dt * np.array([av, ao, ass]))
                    results.append({'model': name, 'kind': 'class', 'alphas': [float(av), float(ao), float(ass)], 'validation': met(yv, qv), 'test': met(yt, qt)})
        for conf in (0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            for a in np.linspace(0.1, 1, 10):
                gate = ev.max(1) < conf
                qv = ev.copy()
                qt = et.copy()
                qv[gate] = norm((1 - a) * ev[gate] + a * dv[gate])
                gate_t = et.max(1) < conf
                qt[gate_t] = norm((1 - a) * et[gate_t] + a * dt[gate_t])
                results.append({'model': name, 'kind': 'confidence_gate', 'conf': conf, 'alpha': float(a), 'changed_val': int(gate.sum()), 'validation': met(yv, qv), 'test': met(yt, qt)})
    results.sort(key=lambda r: (r['validation']['f1_macro'], r['validation']['accuracy']), reverse=True)
    out = {'ensemble_validation': met(yv, ev), 'ensemble_test': met(yt, et), 'ranking': results}
    (HERE / 'dynamic_gate_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'top': results[:20]}, indent=2), flush=True)
if __name__ == '__main__':
    main()
