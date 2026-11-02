"""Selector pairwise: historial vs. ensamble YOLO sÃ³lo ante desacuerdo.

El selector se entrena Ãºnicamente con train natural. ValidaciÃ³n natural elige
la cabeza y el umbral; el test natural permanece bloqueado hasta el final.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json
import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data' / 'player_geometry_fusion' / 'pose_cache' / 'frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
OUT = HERE / 'pairwise_selector_seed42'
SEEDS = (17, 42, 2026)
WEIGHTS = np.asarray([0.2, 0.2, 0.6], dtype=np.float64)

def norm(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 1e-08, None)
    return x / x.sum(axis=1, keepdims=True)

def entropy(p):
    p = norm(p)
    return -(p * np.log(p)).sum(axis=1, keepdims=True)

def metric(y, p):
    pred = np.asarray(p).argmax(axis=1)
    return {'accuracy': float(accuracy_score(y, pred)), 'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0))}

def stored(path, frame):
    return R.stored_probabilities(path, frame)

def expert(seed, split, base):
    root = REPO / 'data' / 'historical_locked_yolo_residual_20260811' / f'results_seed{seed}'
    result = json.loads(_resolve_model_artifact(root / 'selected_result.json').read_text(encoding='utf-8'))
    cfg = result['selected_configuration']
    art = joblib.load(_resolve_model_artifact(root / 'selected_model.joblib'))
    with _load_model_result_array(REPO / 'data' / 'historical_locked_yolo_residual_20260811' / 'results_seed42' / 'feature_cache' / f"{cfg['representation']}" / f'{split}.npz', allow_pickle=True) as z:
        x = z['features'].astype(np.float32)
    reduced = art['pca'].transform(art['scaler'].transform(x)).astype(np.float32)
    geom = norm(art['model'].predict_proba(reduced))
    return norm(R.apply_gate(base, geom, float(cfg['alpha']), str(cfg['gate']), float(cfg['threshold'])))

def pair_features(hist, yolo):
    hist, yolo = (norm(hist), norm(yolo))
    diff = yolo - hist
    disagree = (hist.argmax(1) != yolo.argmax(1)).astype(np.float64)[:, None]
    return np.concatenate([hist, yolo, diff, hist.max(1, keepdims=True), yolo.max(1, keepdims=True), entropy(hist), entropy(yolo), disagree], axis=1)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = R.common.load_base_splits(False, 'natural')
    y_train, y_val, y_test = map(R.labels, (train, val, test))
    base_train = R.historical_train_probabilities(train, REPO / 'data' / 'historical_locked_yolo_residual_20260811' / 'results_seed42' / 'feature_cache')
    base_val = stored(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', val)
    base_test = stored(R.REPRODUCTION / 'checkpoint_test_predictions.csv', test)
    yolo = {}
    for split, base in (('train', base_train), ('validation', base_val), ('test', base_test)):
        outs = [expert(seed, split, base) for seed in SEEDS]
        yolo[split] = norm(sum((w * p for w, p in zip(WEIGHTS, outs))))
    hist = {'train': base_train, 'validation': base_val, 'test': base_test}
    X = {s: pair_features(hist[s], yolo[s]) for s in ('train', 'validation', 'test')}
    pred_h = {s: hist[s].argmax(1) for s in X}
    pred_y = {s: yolo[s].argmax(1) for s in X}
    diff = pred_h['train'] != pred_y['train']
    target = (pred_y['train'][diff] == y_train[diff]).astype(np.int32)
    heads = {'logistic_c01': make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000, random_state=42)), 'logistic_c1': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=42)), 'extra_trees': ExtraTreesClassifier(n_estimators=400, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1), 'random_forest': RandomForestClassifier(n_estimators=400, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1), 'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)}
    rows = []
    models = {}
    for name, model in heads.items():
        model.fit(X['train'][diff], target)
        models[name] = model
        probs = model.predict_proba(X['validation'])[:, 1]
        for threshold in np.arange(0.3, 0.701, 0.025):
            pred = pred_h['validation'].copy()
            choose = (pred_h['validation'] != pred_y['validation']) & (probs >= threshold)
            pred[choose] = pred_y['validation'][choose]
            rows.append({'head': name, 'threshold': float(threshold), **metric(y_val, np.eye(3)[pred]), 'changed': int(choose.sum())})
    ranking = pd.DataFrame(rows).sort_values(['f1_macro', 'accuracy'], ascending=False)
    ranking.to_csv(OUT / 'validation_ranking.csv', index=False)
    baseline_val, baseline_test = (metric(y_val, hist['validation']), metric(y_test, hist['test']))
    residual_val, residual_test = (metric(y_val, yolo['validation']), metric(y_test, yolo['test']))
    selected = ranking.iloc[0].to_dict()
    model = models[selected['head']]
    val_switch = model.predict_proba(X['validation'])[:, 1] >= float(selected['threshold'])
    test_switch = model.predict_proba(X['test'])[:, 1] >= float(selected['threshold'])
    val_pred = pred_h['validation'].copy()
    val_pred[(pred_h['validation'] != pred_y['validation']) & val_switch] = pred_y['validation'][(pred_h['validation'] != pred_y['validation']) & val_switch]
    test_pred = pred_h['test'].copy()
    test_pred[(pred_h['test'] != pred_y['test']) & test_switch] = pred_y['test'][(pred_h['test'] != pred_y['test']) & test_switch]
    selected_val, selected_test = (metric(y_val, np.eye(3)[val_pred]), metric(y_test, np.eye(3)[test_pred]))
    result = {'n_train': int(len(y_train)), 'n_disagreements_train': int(diff.sum()), 'baseline_validation': baseline_val, 'baseline_test': baseline_test, 'frozen_yolo_validation': residual_val, 'frozen_yolo_test': residual_test, 'selected': selected, 'selected_validation': selected_val, 'selected_test': selected_test, 'delta_test_accuracy_vs_historical': selected_test['accuracy'] - baseline_test['accuracy'], 'delta_test_accuracy_vs_frozen_yolo': selected_test['accuracy'] - residual_test['accuracy'], 'selection': 'validation macro-F1, no test selection'}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    joblib.dump({'model': model, 'threshold': float(selected['threshold']), 'weights': WEIGHTS.tolist()}, OUT / 'selected_model.joblib', compress=3)
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
