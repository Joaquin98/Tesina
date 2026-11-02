"""Refit de los expertos YOLO con train+validaciÃ³n natural, configuraciÃ³n fija."""
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
import joblib, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data/player_geometry_fusion/pose_cache/frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
OUT = HERE / 'refit_natural_all_seed42'
ROOT = REPO / 'data/historical_locked_yolo_residual_20260811'
CACHE = ROOT / 'results_seed42' / 'feature_cache'
SEEDS = (17, 42, 2026)
WEIGHTS = np.array([0.2, 0.2, 0.6])

def norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-08, None)
    return p / p.sum(1, keepdims=True)

def metric(y, p):
    return {'accuracy': float(accuracy_score(y, p.argmax(1))), 'f1_macro': float(f1_score(y, p.argmax(1), average='macro', zero_division=0))}

def loadf(rep, split):
    with _load_model_result_array(CACHE / rep / f'{split}.npz', allow_pickle=True) as z:
        return z['features'].astype(np.float32)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = R.common.load_base_splits(False, 'natural')
    all_train = pd.concat([tr, va], ignore_index=True)
    yall = R.labels(all_train)
    yt = R.labels(te)
    base_all = np.concatenate([R.historical_train_probabilities(tr, ROOT / 'results_seed42'), R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', va)])
    base_test = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', te)
    vals = []
    tests = []
    for seed in SEEDS:
        res = json.loads(_resolve_model_artifact(ROOT / f'results_seed{seed}' / 'selected_result.json').read_text())
        cfg = res['selected_configuration']
        rep = cfg['representation']
        xall = np.concatenate([loadf(rep, 'train'), loadf(rep, 'validation')])
        xt = loadf(rep, 'test')
        scaler = StandardScaler().fit(xall)
        xs = scaler.transform(xall)
        xts = scaler.transform(xt)
        pca = PCA(n_components=min(64, xs.shape[1], xs.shape[0] - 1), random_state=42).fit(xs)
        model = LogisticRegression(C=float(cfg['C']), class_weight='balanced', max_iter=3000, random_state=seed)
        model.fit(pca.transform(xs), yall)
        geom = norm(model.predict_proba(pca.transform(xts)))
        final = norm(R.apply_gate(base_test, geom, float(cfg['alpha']), str(cfg['gate']), float(cfg['threshold'])))
        tests.append(final)
        joblib.dump({'scaler': scaler, 'pca': pca, 'model': model, 'configuration': cfg, 'fit_rows': len(all_train)}, OUT / f'seed{seed}.joblib', compress=3)
        vals.append({'seed': seed, 'representation': rep, 'C': cfg['C'], 'fit_rows': len(all_train), 'test': metric(yt, final)})
    ensemble = norm(sum((w * p for w, p in zip(WEIGHTS, tests))))
    result = {'historical_test': metric(yt, base_test), 'refit_seeds': vals, 'ensemble_test': metric(yt, ensemble), 'delta_accuracy_vs_historical': metric(yt, ensemble)['accuracy'] - metric(yt, base_test)['accuracy'], 'delta_f1_vs_historical': metric(yt, ensemble)['f1_macro'] - metric(yt, base_test)['f1_macro'], 'weights': WEIGHTS.tolist(), 'selection': 'configurations fixed from prior natural validation; fit train+validation; test locked'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
