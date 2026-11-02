"""Candidate evaluation on the immutable 5-fold grouped nested partitions."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
FEATURES = HERE / 'rgb_dynamics_seed42' / 'fusion_features.npz'
SPLITS = REPO / 'data/independent_evaluation/outputs/splits'

def metric(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    d = _load_model_result_array(FEATURES, allow_pickle=True)
    all_x = d['features'].astype(np.float32)[:, :-3]
    lookup = {str(n): i for i, n in enumerate(d['names'].astype(str))}
    rows = []
    for fold_dir in sorted(_model_artifact_glob(SPLITS, 'outer_*', recursive=False)):
        if not _resolve_model_artifact(fold_dir / 'train.csv').exists():
            continue
        tr = pd.read_csv(_resolve_model_artifact(fold_dir / 'train.csv'))
        va = pd.read_csv(_resolve_model_artifact(fold_dir / 'validation.csv'))
        ev = pd.read_csv(_resolve_model_artifact(fold_dir / 'evaluation.csv'))
        get = lambda f: all_x[[lookup[str(n)] for n in f.video_name]]
        y = lambda f: f.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy()
        xtr, xv, xe = (get(tr), get(va), get(ev))
        yt, yv, ye = (y(tr), y(va), y(ev))
        candidates = []
        for C in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
            for cwname, cw in [('none', None), ('balanced', 'balanced'), ('moderate', {0: 1.2, 1: 0.65, 2: 1.8})]:
                sc = StandardScaler().fit(xtr)
                m = LogisticRegression(C=C, class_weight=cw, max_iter=3000, random_state=42).fit(sc.transform(xtr), yt)
                pv = m.predict(sc.transform(xv))
                candidates.append((metric(yv, pv)['f1_macro'], metric(yv, pv)['accuracy'], C, cwname, cw))
        best = max(candidates, key=lambda z: (z[0], z[1]))
        C, cwname, cw = best[2:]
        sc = StandardScaler().fit(np.concatenate([xtr, xv]))
        m = LogisticRegression(C=C, class_weight=cw, max_iter=3000, random_state=42).fit(sc.transform(np.concatenate([xtr, xv])), np.concatenate([yt, yv]))
        pe = m.predict(sc.transform(xe))
        rows.append({'outer_fold': fold_dir.name, 'selected_C': C, 'selected_weights': cwname, 'validation': {'f1_macro': best[0], 'accuracy': best[1]}, 'evaluation': metric(ye, pe), 'train_rows': len(tr), 'validation_rows': len(va), 'evaluation_rows': len(ev)})
    pooled_y = []
    pooled_p = []
    for row in rows:
        pass
    result = {'candidate': 'RGB dynamics + YOLO descriptors (no historical logits)', 'folds': rows, 'mean_accuracy': float(np.mean([r['evaluation']['accuracy'] for r in rows])), 'mean_f1_macro': float(np.mean([r['evaluation']['f1_macro'] for r in rows])), 'pooled_note': 'fold-wise metrics; no evaluation labels used for selection'}
    (HERE / 'nested_group_candidate_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
