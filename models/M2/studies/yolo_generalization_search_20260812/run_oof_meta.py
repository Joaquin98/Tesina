"""Meta-fusiÃ³n histÃ³rica+YOLO con predicciones out-of-fold."""
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
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
OUT = HERE / 'oof_meta_seed42'
SEEDS = (17, 42, 2026)
WEIGHTS = np.array([0.2, 0.2, 0.6])
LABELS = R.LABELS
CONFIGS = []
for s in SEEDS:
    cfg = json.loads(_resolve_model_artifact(ROOT / f'results_seed{s}' / 'selected_result.json').read_text())['selected_configuration']
    CONFIGS.append((s, cfg))

def norm(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-08, None)
    return p / p.sum(1, keepdims=True)

def logits(p):
    p = norm(p)
    q = np.log(p)
    return q - q.mean(1, keepdims=True)

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p.argmax(1))), 'f1_macro': float(f1_score(y, p.argmax(1), average='macro', zero_division=0))}

def feat(rep, split):
    with _load_model_result_array(CACHE / rep / f'{split}.npz', allow_pickle=True) as z:
        return z['features'].astype(np.float32)

def expert_fit_predict(xtr, ytr, xout, cfg, seed):
    sc = StandardScaler().fit(xtr)
    a = sc.transform(xtr)
    b = sc.transform(xout)
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(64, a.shape[1], a.shape[0] - 1), random_state=seed).fit(a)
    m = LogisticRegression(C=float(cfg['C']), class_weight='balanced', max_iter=3000, random_state=seed).fit(p.transform(a), ytr)
    return norm(m.predict_proba(p.transform(b)))

def apply_gate(base, geom, cfg):
    return norm(R.apply_gate(base, geom, float(cfg['alpha']), str(cfg['gate']), float(cfg['threshold'])))

def meta_features(hist, experts):
    h = norm(hist)
    es = [norm(x) for x in experts]
    stack = np.concatenate([logits(h)] + [logits(x) for x in es] + es, 1)
    disagreement = np.column_stack([(h.argmax(1) != x.argmax(1)).astype(float) for x in es])
    maxes = np.column_stack([h.max(1)] + [x.max(1) for x in es])
    ent = lambda x: -(x * np.log(x)).sum(1, keepdims=True)
    entropy = np.concatenate([ent(h)] + [ent(x) for x in es], 1)
    return np.concatenate([stack, disagreement, maxes, entropy], 1)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = R.common.load_base_splits(False, 'natural')
    ytr, yv, yt = map(R.labels, (tr, va, te))
    base_tr = R.historical_train_probabilities(tr, ROOT / 'results_seed42')
    base_va = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', va)
    base_te = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', te)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = [np.zeros((len(tr), 3)) for _ in SEEDS]
    all_x = []
    for fold, (fit_idx, hold_idx) in enumerate(skf.split(np.zeros(len(ytr)), ytr)):
        print('fold', fold, flush=True)
        for j, (seed, cfg) in enumerate(CONFIGS):
            x = feat(cfg['representation'], 'train')
            g = expert_fit_predict(x[fit_idx], ytr[fit_idx], x[hold_idx], cfg, seed)
            oof[j][hold_idx] = apply_gate(base_tr[hold_idx], g, cfg)
    hist_oof = base_tr
    X_oof = meta_features(hist_oof, oof)
    experts_val = []
    experts_test = []
    for seed, cfg in CONFIGS:
        rep = cfg['representation']
        geomv = expert_fit_predict(feat(rep, 'train'), ytr, feat(rep, 'validation'), cfg, seed)
        geomt = expert_fit_predict(feat(rep, 'train'), ytr, feat(rep, 'test'), cfg, seed)
        experts_val.append(apply_gate(base_va, geomv, cfg))
        experts_test.append(apply_gate(base_te, geomt, cfg))
    Xv = meta_features(base_va, experts_val)
    Xt = meta_features(base_te, experts_test)
    rows = []
    models = {'logistic_001': make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight='balanced', max_iter=3000, random_state=42)), 'logistic_01': make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000, random_state=42)), 'logistic_1': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=42)), 'extra_trees': ExtraTreesClassifier(n_estimators=500, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1), 'random_forest': RandomForestClassifier(n_estimators=500, min_samples_leaf=8, class_weight='balanced', random_state=42, n_jobs=-1), 'hist_gradient': HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)}
    for name, m in models.items():
        m.fit(X_oof, ytr)
        pv = norm(m.predict_proba(Xv))
        pt = norm(m.predict_proba(Xt))
        rows.append({'head': name, 'validation': met(yv, pv), 'test': met(yt, pt)})
    base_ens = norm(sum((w * x for w, x in zip(WEIGHTS, experts_test))))
    selected = max(rows, key=lambda r: (r['validation']['f1_macro'], r['validation']['accuracy']))
    out = {'historical_test': met(yt, base_te), 'frozen_ensemble_test': met(yt, base_ens), 'candidates': rows, 'selected': selected, 'delta_selected_vs_frozen': selected['test']['accuracy'] - met(yt, base_ens)['accuracy'], 'delta_selected_vs_historical': selected['test']['accuracy'] - met(yt, base_te)['accuracy'], 'protocol': {'oof_folds': 5, 'meta_train': 'out-of-fold natural train only', 'selection': 'natural validation only', 'test_locked': True}}
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)
if __name__ == '__main__':
    main()
