"""Locked natural test of the group-selected C=.01, alpha=.5 fusion."""
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
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    d = _load_model_result_array(HERE / 'rgb_dynamics_seed42/fusion_features.npz', allow_pickle=True)
    X = d['features'][:, :-3].astype(np.float32)
    y = d['labels'].astype(int)
    ntr = int(d['train_count'])
    nv = int(d['validation_count'])
    sc = StandardScaler().fit(X[:ntr])
    m = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1500, random_state=42).fit(sc.transform(X[:ntr]), y[:ntr])
    py = m.predict_proba(sc.transform(X[ntr:ntr + nv]))
    pt = m.predict_proba(sc.transform(X[-817:]))
    ep = pd.read_csv(_resolve_model_artifact(study_csv_path(REPO / 'data/historical_locked_yolo_residual_20260811/ensemble_three_seeds/test_predictions.csv')))
    bp = ep[['ensemble_prob_line', 'ensemble_prob_other', 'ensemble_prob_scrum']].to_numpy(np.float32)
    vp = pd.read_csv(_resolve_model_artifact(study_csv_path(REPO / 'data/historical_locked_yolo_residual_20260811/ensemble_three_seeds/validation_weight_search.csv')))
    seed = REPO / 'data/historical_locked_yolo_residual_20260811'
    vm = pd.read_csv(_resolve_model_artifact(study_csv_path(seed / 'results_seed42/validation_candidates.csv'))) if _resolve_model_artifact(study_csv_path(seed / 'results_seed42/validation_candidates.csv')).exists() else None
    nat_train = pd.read_csv(_resolve_model_artifact(study_csv_path(CURRENT_DATASET / 'train.csv')))
    va = pd.read_csv(_resolve_model_artifact(study_csv_path(CURRENT_DATASET / 'validation.csv')))
    te = pd.read_csv(_resolve_model_artifact(study_csv_path(CURRENT_DATASET / 'test.csv')))
    z = 0.5 * bp + 0.5 * pt
    out = {'candidate_validation': met(y[ntr:ntr + nv], py.argmax(1)), 'candidate_test': met(y[-817:], pt.argmax(1)), 'locked_blend_test': met(y[-817:], z.argmax(1)), 'historical_ensemble_test': met(y[-817:], bp.argmax(1)), 'protocol': {'C': 0.01, 'alpha': 0.5, 'features': 'RGB dynamics + E04 YOLO', 'test_locked': True}}
    (HERE / 'natural_group_fusion_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
