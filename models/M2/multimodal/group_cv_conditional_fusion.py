"""Grouped evaluation of conditional YOLO correction.

Each outer fold trains a RGB+YOLO logistic expert on the other source groups.
The frozen historical probabilities are the predictions produced by the
corresponding nested outer-fold checkpoint.  Alpha/threshold are reported as
an audit grid; the pre-registered decision rule uses alpha=.25 and confidence
threshold=.85, selected before reading the outer metrics.
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
FEAT = HERE / 'rgb_dynamics_seed42/fusion_features.npz'
MAN = REPO / 'data/independent_evaluation/outputs/nested_source_manifest.csv'
OUTER = REPO / 'data/independent_evaluation/outputs/nested_evaluation'

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    d = _load_model_result_array(FEAT, allow_pickle=True)
    X = d['features'][:, :-3].astype(np.float32)
    y = d['labels'].astype(int)
    names = d['names'].astype(str)
    manifest = pd.read_csv(_resolve_model_artifact(MAN)).set_index('video_name').loc[names]
    groups = manifest.source_group.astype(str).to_numpy()
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=20260712)
    base = np.zeros((len(y), 3), np.float32)
    for fold in range(1, 6):
        p = pd.read_csv(_resolve_model_artifact(OUTER / f'outer_{fold:02d}/outer_predictions.csv'))
        lookup = {str(n): i for i, n in enumerate(names)}
        for row in p.itertuples():
            base[lookup[str(row.video_name)]] = [row.probability_line, row.probability_other, row.probability_scrum]
    rows = []
    for C in (0.001, 0.003, 0.01):
        pred_exp = np.zeros(len(y), int)
        prob_exp = np.zeros_like(base)
        for fit, hold in cv.split(X, y, groups):
            sc = StandardScaler().fit(X[fit])
            m = LogisticRegression(C=C, class_weight='balanced', max_iter=1500, random_state=42).fit(sc.transform(X[fit]), y[fit])
            prob_exp[hold] = m.predict_proba(sc.transform(X[hold]))
            pred_exp[hold] = prob_exp[hold].argmax(1)
        for alpha in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5):
            for threshold in (0.75, 0.8, 0.85, 0.9, 1.01):
                p = base.copy()
                conf = base.max(1)
                use = conf < threshold
                p[use] = (1 - alpha) * base[use] + alpha * prob_exp[use]
                pred = p.argmax(1)
                folds = []
                for j, (_, hold) in enumerate(cv.split(X, y, groups)):
                    folds.append({'fold': j + 1, 'n': int(len(hold)), **met(y[hold], pred[hold]), 'historical': met(y[hold], base[hold].argmax(1))})
                rows.append({'C': C, 'alpha': alpha, 'threshold': threshold, 'pooled': met(y, pred), 'folds': folds, 'used_fraction': float(use.mean())})
    rows.sort(key=lambda r: (r['pooled']['f1_macro'], r['pooled']['accuracy']), reverse=True)
    out = {'pre_registered': next((r for r in rows if r['C'] == 0.003 and r['alpha'] == 0.25 and (r['threshold'] == 0.85))), 'ranking': rows, 'protocol': {'groups': 5, 'historical_predictions': 'nested outer checkpoints', 'selection_warning': 'ranking is descriptive; only pre_registered row is promotion candidate'}}
    (HERE / 'group_cv_conditional_fusion_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'pre_registered': out['pre_registered'], 'top': rows[:5]}, indent=2))
if __name__ == '__main__':
    main()
