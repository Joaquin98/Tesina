"""Bias/threshold screen for the hierarchical rugby classifier."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import itertools, json
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

def bias(p, b):
    z = np.log(np.clip(p, 1e-07, 1)) + np.asarray(b, np.float32)
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)

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
    rows = []
    for C1, C2 in itertools.product((0.001, 0.003, 0.01, 0.03), repeat=2):
        for cw1, cw2 in [(None, None), ('balanced', None), (None, 'balanced')]:
            q = np.zeros_like(base)
            for fit, hold in outer:
                sc = StandardScaler().fit(X[fit])
                a = sc.transform(X[fit])
                b = sc.transform(X[hold])
                m1 = LogisticRegression(C=C1, class_weight=cw1, max_iter=1000, random_state=42).fit(a, (y[fit] == 2).astype(int))
                keep = y[fit] != 2
                m2 = LogisticRegression(C=C2, class_weight=cw2, max_iter=1000, random_state=42).fit(a[keep], y[fit][keep])
                ps = m1.predict_proba(b)[:, 1]
                po = m2.predict_proba(b)
                q[hold, 2] = ps
                q[hold, :2] = (1 - ps)[:, None] * po
            for alpha in (0.4, 0.5, 0.6, 0.7):
                for bb in itertools.product((-0.2, 0, 0.2), repeat=3):
                    z = bias(q, bb)
                    p = (1 - alpha) * base + alpha * z
                    rows.append({'C_scrum': C1, 'C_line_other': C2, 'cw_scrum': cw1, 'cw_line_other': cw2, 'alpha': alpha, 'bias': bb, 'pooled': met(y, p.argmax(1)), 'mean_fold_accuracy': float(np.mean([met(y[h], p[h].argmax(1))['accuracy'] for _, h in outer]))})
    rows.sort(key=lambda r: (r['mean_fold_accuracy'], r['pooled']['f1_macro']), reverse=True)
    out = {'top': rows[:50], 'protocol': {'groups': 5, 'features': 'RGB dynamics + E04 + E06', 'hierarchy': 'scrum vs non-scrum then line vs other', 'selection_warning': 'exploratory; nested validation required'}}
    (HERE / 'group_cv_hierarchical_bias_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
