"""Train the compact historical+YOLO meta-model with all reviewed labels.

Natural RGB/YOLO features stay frozen; this isolates the effect of adding the
998 corrected labels and evaluates on the untouched natural validation/test.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
NAT = HERE / 'rgb_dynamics_seed42' / 'fusion_features.npz'
REV = REPO / 'data/historical_locked_yolo_residual_20260811/human_review_evaluation_20260811/predictions.csv'

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    d = _load_model_result_array(NAT, allow_pickle=True)
    nat = np.exp(d['features'][:, -3:].astype(np.float32))
    y_nat = d['labels'].astype(int)
    rev = pd.read_csv(_resolve_model_artifact(REV))
    y_rev = rev.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy(dtype=int)
    base_rev = rev[['historical_prob_line', 'historical_prob_other', 'historical_prob_scrum']].to_numpy(dtype=np.float32)
    yolo_rev = rev[['yolo_residual_prob_line', 'yolo_residual_prob_other', 'yolo_residual_prob_scrum']].to_numpy(dtype=np.float32)
    q = (rev['yolo_residual_prediction'].astype(str) != rev['historical_prediction'].astype(str)).to_numpy(dtype=np.float32)[:, None]
    rev_x = np.concatenate([np.log(np.clip(base_rev, 1e-07, 1)), np.log(np.clip(yolo_rev, 1e-07, 1)), q], axis=1)
    nat_x = np.concatenate([np.log(np.clip(nat, 1e-07, 1)), np.log(np.clip(nat, 1e-07, 1)), np.zeros((len(nat), 1), np.float32)], axis=1)
    ntr = int(d['train_count'])
    nv = int(d['validation_count'])
    Xtr = np.concatenate([nat_x[:ntr], rev_x], axis=0)
    ytr = np.concatenate([y_nat[:ntr], y_rev])
    sc = StandardScaler().fit(Xtr)
    rows = []
    for C, cwname, cw in ((0.003, 'balanced', 'balanced'), (0.01, 'balanced', 'balanced'), (0.03, 'none', None), (0.01, 'moderate', {0: 1.2, 1: 0.65, 2: 1.8})):
        m = LogisticRegression(C=C, class_weight=cw, max_iter=2000, random_state=42).fit(sc.transform(Xtr), ytr)
        pv = m.predict(sc.transform(nat_x[ntr:ntr + nv]))
        pt = m.predict(sc.transform(nat_x[-817:]))
        pr = m.predict(sc.transform(rev_x))
        rows.append({'C': C, 'class_weight': cwname, 'natural_validation': met(y_nat[ntr:ntr + nv], pv), 'natural_test': met(y_nat[-817:], pt), 'human_review_train_apparent': met(y_rev, pr)})
    out = {'rows': rows, 'counts': {'natural_train': ntr, 'human_reviews': len(rev)}, 'note': 'Human review labels are used for training; natural validation/test remain untouched.'}
    (HERE / 'all_human_reviews_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
