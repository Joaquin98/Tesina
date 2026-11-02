"""Confirmatory fixed configuration: quantile scaler + moderate weights + alpha .6."""
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
from sklearn.preprocessing import QuantileTransformer
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
    X = np.concatenate([dyn, e04, e06], 1)
    groups = pd.read_csv(_resolve_model_artifact(MAN)).set_index('video_name').loc[names].source_group.astype(str).to_numpy()
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=20260712).split(X, y, groups))
    base = np.zeros((len(y), 3), np.float32)
    ix = {str(v): i for i, v in enumerate(names)}
    for fold in range(1, 6):
        p = pd.read_csv(_resolve_model_artifact(OUTER / f'outer_{fold:02d}/outer_predictions.csv'))
        for r in p.itertuples():
            base[ix[str(r.video_name)]] = [r.probability_line, r.probability_other, r.probability_scrum]
    configs = [('quantile_moderate_a06', 0.003, {0: 1.2, 1: 0.65, 2: 1.8}, 0.6), ('quantile_none_a05', 0.003, None, 0.5), ('quantile_moderate_a05', 0.003, {0: 1.2, 1: 0.65, 2: 1.8}, 0.5), ('power_none_a05', 0.01, None, 0.5)]
    rows = []
    for name, C, cw, a in configs:
        pyo = np.zeros_like(base)
        for fit, hold in outer:
            sc = QuantileTransformer(n_quantiles=256, output_distribution='normal', random_state=42)
            u = sc.fit_transform(X[fit])
            v = sc.transform(X[hold])
            m = LogisticRegression(C=C, class_weight=cw, max_iter=1200, random_state=42).fit(u, y[fit])
            pyo[hold] = m.predict_proba(v)
        p = (1 - a) * base + a * pyo
        rows.append({'config': name, 'C': C, 'class_weight': str(cw), 'alpha': a, 'pooled': met(y, p.argmax(1)), 'mean_fold_accuracy': float(np.mean([met(y[h], p[h].argmax(1))['accuracy'] for _, h in outer])), 'folds': [met(y[h], p[h].argmax(1)) for _, h in outer]})
    out = {'configs': rows, 'historical_mean_accuracy': float(np.mean([met(y[h], base[h].argmax(1))['accuracy'] for _, h in outer])), 'historical_mean_f1': float(np.mean([met(y[h], base[h].argmax(1))['f1_macro'] for _, h in outer])), 'protocol': {'fixed_before_metric': True, 'features': 'RGB dynamics + E04 + E06', 'historical_predictions': 'nested outer checkpoints', 'selection_warning': 'configuration originated in exploratory screen; nested result remains authoritative'}}
    (HERE / 'group_cv_fixed_quantile_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
