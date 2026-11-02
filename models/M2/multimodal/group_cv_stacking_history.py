"""Grouped screen for learned stacking of historical probabilities + YOLO/RGB.

Unlike a fixed alpha blend, this head learns class-specific corrections from
the historical expert probabilities and the cached YOLO views. The screen is
descriptive; the best result must be rechecked with nested inner selection.
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
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
MAN = REPO / 'data/independent_evaluation/outputs/nested_source_manifest.csv'
OUTER = REPO / 'data/independent_evaluation/outputs/nested_evaluation'

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def main():
    d = _load_model_result_array(HERE / 'rgb_dynamics_seed42/fusion_features.npz', allow_pickle=True)
    dyn = d['features'][:, :704].astype(np.float32)
    y = d['labels'].astype(int)
    names = d['names'].astype(str)
    e04 = _load_model_result_array(HERE / 'multi_yolo_seed42/E04_tracking_smoothing.npy').astype(np.float32)
    e06 = _load_model_result_array(HERE / 'invariant_yolo_seed42/E06_rugby_descriptors.npy').astype(np.float32)
    groups = pd.read_csv(_resolve_model_artifact(MAN)).set_index('video_name').loc[names].source_group.astype(str).to_numpy()
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=20260712).split(dyn, y, groups))
    hist = np.zeros((len(y), 3), np.float32)
    ix = {str(v): i for i, v in enumerate(names)}
    for fold in range(1, 6):
        p = pd.read_csv(_resolve_model_artifact(OUTER / f'outer_{fold:02d}/outer_predictions.csv'))
        for r in p.itertuples():
            hist[ix[str(r.video_name)]] = [r.probability_line, r.probability_other, r.probability_scrum]
    X = np.concatenate([dyn, e04, e06, np.log(np.clip(hist, 1e-07, 1))], 1)
    rows = []
    for C in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        for cname, cw in [('none', None), ('balanced', 'balanced'), ('moderate', {0: 1.2, 1: 0.65, 2: 1.8})]:
            pred = np.zeros(len(y), int)
            for fit, hold in outer:
                sc = StandardScaler().fit(X[fit])
                m = LogisticRegression(C=C, class_weight=cw, solver='lbfgs', max_iter=1000, random_state=42).fit(sc.transform(X[fit]), y[fit])
                pred[hold] = m.predict(sc.transform(X[hold]))
            rows.append({'C': C, 'class_weight': cname, 'pooled': met(y, pred), 'mean_fold_accuracy': float(np.mean([met(y[h], pred[h])['accuracy'] for _, h in outer]))})
    rows.sort(key=lambda r: (r['mean_fold_accuracy'], r['pooled']['f1_macro']), reverse=True)
    out = {'top': rows[:30], 'protocol': {'groups': 5, 'features': 'RGB dynamics + E04 + E06 + log historical probabilities', 'historical_predictions': 'nested outer checkpoints', 'selection_warning': 'exploratory grouped screen; nested validation required'}}
    (HERE / 'group_cv_stacking_history_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
