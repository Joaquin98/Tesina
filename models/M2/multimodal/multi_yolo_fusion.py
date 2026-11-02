"""Combine several complementary YOLO summaries with RGB dynamics."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
import json, os, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
BASE = HERE / 'rgb_dynamics_seed42'
OUT = HERE / 'multi_yolo_seed42'

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = _load_model_result_array(BASE / 'fusion_features.npz', allow_pickle=True)
    dyn = d['features'][:, :704].astype(np.float32)
    n = len(dyn)
    ntr = int(d['train_count'])
    nv = int(d['validation_count'])
    split_dir = CURRENT_DATASET
    df = pd.concat([pd.read_csv(_resolve_model_artifact(study_csv_path(split_dir / 'train_natural.csv'))), pd.read_csv(_resolve_model_artifact(study_csv_path(split_dir / 'validation.csv'))), pd.read_csv(_resolve_model_artifact(study_csv_path(split_dir / 'test_natural.csv')))], ignore_index=True)
    parts = []
    for rep in ('E02_quality_mask', 'E04_tracking_smoothing'):
        p = OUT / f'{rep}.npy'
        if not _resolve_model_artifact(study_csv_path(p)).exists():
            raise FileNotFoundError(f'Cached representation missing: {p}')
        x = _load_model_result_array(p)
        parts.append(x.astype(np.float32))
        print(rep, parts[-1].shape, flush=True)
    yolo = np.concatenate(parts, 1)
    base = d['features'][:, -3:].astype(np.float32)
    X = np.concatenate([dyn, yolo, base], 1)
    y = df.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy()
    sc = StandardScaler().fit(X[:ntr])
    a = sc.transform(X[:ntr])
    b = sc.transform(X[ntr:ntr + nv])
    c = sc.transform(X[-817:])
    rows = []
    for C, cwname, cw in ((0.0003, 'none', None), (0.001, 'balanced', 'balanced'), (0.003, 'balanced', 'balanced'), (0.01, 'moderate', {0: 1.2, 1: 0.65, 2: 1.8}), (0.03, 'none', None), (0.1, 'balanced', 'balanced')):
        m = LogisticRegression(C=C, class_weight=cw, max_iter=1000, random_state=42)
        m.fit(a, y[:ntr])
        pv = m.predict(b)
        pt = m.predict(c)
        rows.append({'C': C, 'class_weight': cwname, 'validation': met(y[ntr:ntr + nv], pv), 'test': met(y[-817:], pt)})
    rows.sort(key=lambda r: (r['validation']['f1_macro'], r['validation']['accuracy']), reverse=True)
    res = {'historical_ensemble_test': {'accuracy': 0.8776009791921665, 'f1_macro': 0.8479262047057438}, 'candidates': rows}
    (OUT / 'summary.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps(res, indent=2), flush=True)
if __name__ == '__main__':
    main()
