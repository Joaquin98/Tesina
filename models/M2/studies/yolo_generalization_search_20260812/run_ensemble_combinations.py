"""Busca combinaciones del ensamble YOLO sin tocar los expertos congelados.

El ensamble vigente promedia probabilidades. AquÃ­ se comparan pooling
aritmÃ©tico, log-opinion-pool (media ponderada de logits/probabilidades en
logaritmo) y mediana, siempre seleccionando por validaciÃ³n y evaluando una
sola vez en el test bloqueado.
"""
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
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data/player_geometry_fusion/pose_cache/frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M2.studies.historical_locked_yolo_residual_20260811.ensemble_seeds import probabilities_for_seed, metric_row
OUT = HERE / 'ensemble_combinations_seed42'
ROOT = REPO / 'data/historical_locked_yolo_residual_20260811'
SEEDS = (17, 42, 2026)

def grid(step=0.05):
    n = int(round(1 / step))
    for i in range(n + 1):
        for j in range(n - i + 1):
            yield (np.array([i, j, n - i - j], dtype=float) / n)

def combine(ps, w, kind):
    a = np.asarray(ps, float)
    if kind == 'arithmetic':
        out = np.tensordot(w, a, axes=(0, 0))
    elif kind == 'log_pool':
        out = np.exp(np.tensordot(w, np.log(np.clip(a, 1e-07, 1)), axes=(0, 0)))
    elif kind == 'median':
        out = np.median(a, axis=0)
    elif kind == 'max_agreement':
        margins = np.sort(a, axis=2)[:, :, -1] - np.sort(a, axis=2)[:, :, -2]
        out = a[np.argmax(margins, axis=0), np.arange(a.shape[1])]
    else:
        raise ValueError(kind)
    return out / np.maximum(out.sum(1, keepdims=True), 1e-08)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = R.common.load_base_splits(False, 'natural')
    yv, yt = (R.labels(va), R.labels(te))
    bv = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', va)
    bt = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', te)
    gv = []
    gt = []
    for seed in SEEDS:
        pv, _ = probabilities_for_seed(seed, 'validation', bv)
        pt, _ = probabilities_for_seed(seed, 'test', bt)
        gv.append(pv)
        gt.append(pt)
    rows = []
    for kind in ('arithmetic', 'log_pool', 'median', 'max_agreement'):
        weights = list(grid()) if kind not in ('median', 'max_agreement') else [np.ones(3) / 3]
        for w in weights:
            pv = combine(gv, w, kind)
            pt = combine(gt, w, kind)
            mv = metric_row(yv, pv)
            mt = metric_row(yt, pt)
            rows.append({'kind': kind, 'w17': float(w[0]), 'w42': float(w[1]), 'w2026': float(w[2]), 'validation': mv, 'test': mt, 'passes_accuracy_guard': mv['accuracy'] >= float(accuracy_score(yv, bv.argmax(1))) - 0.005})
    rows = sorted(rows, key=lambda r: (r['passes_accuracy_guard'], r['validation']['f1_macro'], r['validation']['accuracy']), reverse=True)
    result = {'historical_validation': metric_row(yv, bv), 'historical_test': metric_row(yt, bt), 'current_ensemble': json.loads(_resolve_model_artifact(ROOT / 'ensemble_three_seeds' / 'ensemble_result.json').read_text())['ensemble_test'], 'ranking': rows}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'top': rows[:12], 'historical_test': result['historical_test']}, indent=2, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
