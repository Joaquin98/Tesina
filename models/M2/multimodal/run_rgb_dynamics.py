"""Fusion tabular de dinamica RGB + descriptores YOLO + experto historico."""
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
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data/player_geometry_fusion/pose_cache/frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M0.studies.natural_with_human_review import run_experiment as common
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite import feature_matrix
MEM = REPO / 'data/validation_overlap_comparison/feature_memmap'
CACHE = REPO / 'data/historical_locked_yolo_residual_20260811/results_seed42'
OUT = HERE / 'rgb_dynamics_seed42'

def met(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)), 'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0))}

def dynamics(ids, rgb):
    n = len(ids)
    seg = np.zeros((n, 10, 2048), np.float32)
    delta = np.zeros_like(seg)
    mean = np.zeros((n, 2048), np.float32)
    std = np.zeros_like(mean)
    for start in range(0, n, 128):
        ix = ids[start:start + 128]
        a = np.asarray(rgb[ix], np.float32).reshape(len(ix), 10, 20, 2048)
        seg[start:start + len(ix)] = a.mean(2)
        delta[start:start + len(ix)] = np.abs(np.diff(a, axis=2)).mean(2)
        flat = a.reshape(len(ix), 200, 2048)
        mean[start:start + len(ix)] = flat.mean(1)
        std[start:start + len(ix)] = flat.std(1)
    return (seg, delta, mean, std)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = common.load_base_splits(False, 'natural')
    all_df = pd.concat([train, val, test], ignore_index=True)
    names = all_df.video_name.astype(str).tolist()
    index = json.loads(_resolve_model_artifact(MEM / 'resnet50_index.json').read_text())['video_names']
    lookup = {str(name): i for i, name in enumerate(index)}
    ids = np.asarray([lookup[name] for name in names])
    rgb = _load_model_result_array(MEM / 'resnet50_features.npy', mmap_mode='r')
    seg, delta, mean, std = dynamics(ids, rgb)
    n_train = len(train)
    p_seg = PCA(32, random_state=42).fit(seg[:n_train].reshape(-1, 2048))
    p_delta = PCA(32, random_state=42).fit(delta[:n_train].reshape(-1, 2048))
    p_mean = PCA(32, random_state=42).fit(mean[:n_train])
    p_std = PCA(32, random_state=42).fit(std[:n_train])
    joblib.dump({'segment': p_seg, 'delta': p_delta, 'mean': p_mean, 'std': p_std}, OUT / 'dynamic_pca.joblib')
    dyn = np.concatenate([p_seg.transform(seg.reshape(-1, 2048)).reshape(len(all_df), -1), p_delta.transform(delta.reshape(-1, 2048)).reshape(len(all_df), -1), p_mean.transform(mean), p_std.transform(std)], axis=1).astype(np.float32)
    print('dynamic_features', dyn.shape, flush=True)
    y = all_df.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy()
    base_train = R.historical_train_probabilities(train, CACHE)
    base_val = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', val)
    base_test = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', test)
    base = np.concatenate([base_train, base_val, base_test])
    yolo_parts = []
    for start in range(0, len(all_df), 256):
        x, _, _ = feature_matrix(all_df.iloc[start:start + 256], 'E04_tracking_smoothing')
        yolo_parts.append(x)
    yolo = np.concatenate(yolo_parts).astype(np.float32)
    features = np.concatenate([dyn, yolo, np.log(np.clip(base, 1e-07, 1.0))], axis=1)
    np.savez_compressed(OUT / 'fusion_features.npz', features=features, labels=y, names=np.asarray(names), train_count=np.asarray(n_train), validation_count=np.asarray(len(val)))
    scaler = StandardScaler().fit(features[:n_train])
    x_train = scaler.transform(features[:n_train])
    x_val = scaler.transform(features[n_train:n_train + len(val)])
    x_test = scaler.transform(features[-len(test):])
    models = [(f'logit_bal_{c}', LogisticRegression(C=c, class_weight='balanced', max_iter=3000, random_state=42)) for c in (0.003, 0.01, 0.03, 0.1, 0.3)]
    models += [(f'logit_unweighted_{c}', LogisticRegression(C=c, class_weight=None, max_iter=3000, random_state=42)) for c in (0.001, 0.003, 0.01, 0.03)]
    models += [(f'logit_moderate_{c}', LogisticRegression(C=c, class_weight={0: 1.2, 1: 0.65, 2: 1.8}, max_iter=3000, random_state=42)) for c in (0.003, 0.01, 0.03)]
    models += [('extra', ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.5, class_weight='balanced', n_jobs=-1, random_state=42)), ('rf', RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.5, class_weight='balanced', n_jobs=-1, random_state=42))]
    rows = []
    for name, model in models:
        model.fit(x_train, y[:n_train])
        pv = model.predict(x_val)
        pt = model.predict(x_test)
        if hasattr(model, 'predict_proba'):
            np.savez_compressed(OUT / f'{name}_probabilities.npz', validation=model.predict_proba(x_val), test=model.predict_proba(x_test))
        rows.append({'model': name, 'validation': met(y[n_train:n_train + len(val)], pv), 'test': met(y[-len(test):], pt)})
        joblib.dump(model, OUT / f'{name}.joblib')
    rows.sort(key=lambda item: (item['validation']['f1_macro'], item['validation']['accuracy']), reverse=True)
    result = {'historical_test': met(y[-len(test):], base_test.argmax(1)), 'candidates': rows}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
