"""Nested validation of a compact hierarchical E04+E06 fusion."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import itertools, json, gc, sys
import numpy as np, pandas as pd, keras
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
MAN = REPO / 'data/independent_evaluation/outputs/nested_source_manifest.csv'
MEM = REPO / 'data/validation_overlap_comparison/feature_memmap'
EVAL = REPO / 'data/independent_evaluation'
OUTER = REPO / 'data/independent_evaluation/outputs/nested_evaluation'
sys.path.insert(0, str(source_directory(EVAL)))
from run_nested_evaluation import build_model
SPECS = [(0.003, 0.01, None, None, 0.4, (-0.0, 0.0, 0.0)), (0.003, 0.01, None, None, 0.5, (0, -0.2, 0.2)), (0.003, 0.01, None, None, 0.6, (0, -0.2, 0.2)), (0.01, 0.01, 'balanced', None, 0.5, (0.2, 0, -0.2)), (0.01, 0.01, 'balanced', None, 0.6, (0.2, 0, -0.2)), (0.001, 0.01, None, None, 0.6, (0, -0.2, 0.2)), (0.01, 0.03, 'balanced', None, 0.5, (0, 0.2, -0.2)), (0.03, 0.03, 'balanced', None, 0.5, (0, 0.2, -0.2))]

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def bias(p, b):
    z = np.log(np.clip(p, 1e-07, 1)) + np.asarray(b, np.float32)
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)

def hist_probs(sel, names):
    model = build_model(sel)
    model.load_weights(_resolve_model_artifact(sel['checkpoint']))
    idx = json.loads(_resolve_model_artifact(MEM / 'resnet50_index.json').read_text())['video_names']
    lookup = {str(n): i for i, n in enumerate(idx)}
    ids = np.asarray([lookup[n] for n in names])
    feat = _load_model_result_array(MEM / 'resnet50_features.npy', mmap_mode='r')
    mask = _load_model_result_array(MEM / 'resnet50_masks.npy', mmap_mode='r')
    out = np.zeros((len(names), 3), np.float32)
    for s in range(0, len(names), 16):
        out[s:s + len(ids[s:s + 16])] = model.predict_on_batch((np.asarray(feat[ids[s:s + 16]], np.float32), np.asarray(mask[ids[s:s + 16]], bool)))
    del model, feat, mask
    keras.backend.clear_session()
    gc.collect()
    return out

def hierarchical(X, y, fit, hold, spec):
    c1, c2, cw1, cw2, alpha, bb = spec
    sc = StandardScaler().fit(X[fit])
    a = sc.transform(X[fit])
    b = sc.transform(X[hold])
    m1 = LogisticRegression(C=c1, class_weight=cw1, max_iter=800, random_state=42).fit(a, (y[fit] == 2).astype(int))
    keep = y[fit] != 2
    m2 = LogisticRegression(C=c2, class_weight=cw2, max_iter=800, random_state=42).fit(a[keep], y[fit][keep])
    ps = m1.predict_proba(b)[:, 1]
    po = m2.predict_proba(b)
    q = np.zeros((len(hold), 3), np.float32)
    q[:, 2] = ps
    q[:, :2] = (1 - ps)[:, None] * po
    return (q, alpha, bb)

def main():
    d = _load_model_result_array(HERE / 'rgb_dynamics_seed42/fusion_features.npz', allow_pickle=True)
    dyn = d['features'][:, :704].astype(np.float32)
    y = d['labels'].astype(int)
    names = d['names'].astype(str)
    e04 = _load_model_result_array(HERE / 'multi_yolo_seed42/E04_tracking_smoothing.npy').astype(np.float32)
    e06 = _load_model_result_array(HERE / 'invariant_yolo_seed42/E06_rugby_descriptors.npy').astype(np.float32)
    X = np.concatenate([dyn, e04, e06], 1)
    groups = pd.read_csv(_resolve_model_artifact(MAN)).set_index('video_name').loc[names].source_group.astype(str).to_numpy()
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=20260712)
    rows = []
    for fold, (fit, hold) in enumerate(outer.split(X, y, groups), 1):
        selected = json.loads(_resolve_model_artifact(OUTER / f'outer_{fold:02d}/selected_candidate.json').read_text())
        hp = hist_probs(selected, names[fit])
        hhold = hist_probs(selected, names[hold])
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=900 + fold)
        scores = []
        for spec in SPECS:
            vals = []
            for ifit, ival in inner.split(X[fit], y[fit], groups[fit]):
                q, a, b = hierarchical(X[fit], y[fit], ifit, ival, spec)
                p = (1 - a) * hp[ival] + a * bias(q, b)
                vals.append(met(y[fit][ival], p.argmax(1)))
            scores.append({'spec': spec, 'inner_accuracy': float(np.mean([v['accuracy'] for v in vals])), 'inner_f1': float(np.mean([v['f1_macro'] for v in vals]))})
        sel = max(scores, key=lambda r: (r['inner_accuracy'], r['inner_f1']))
        q, a, b = hierarchical(X, y, fit, hold, sel['spec'])
        p = (1 - a) * hhold + a * bias(q, b)
        row = {'fold': fold, 'selected': sel, 'historical': met(y[hold], hhold.argmax(1)), 'fused': met(y[hold], p.argmax(1)), 'n': int(len(hold))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = {'features': 'RGB dynamics + E04 + E06 + hierarchical scrum/non-scrum head', 'folds': rows, 'historical_mean_accuracy': float(np.mean([r['historical']['accuracy'] for r in rows])), 'fused_mean_accuracy': float(np.mean([r['fused']['accuracy'] for r in rows])), 'historical_mean_f1': float(np.mean([r['historical']['f1_macro'] for r in rows])), 'fused_mean_f1': float(np.mean([r['fused']['f1_macro'] for r in rows]))}
    out['mean_delta_accuracy'] = out['fused_mean_accuracy'] - out['historical_mean_accuracy']
    out['mean_delta_f1'] = out['fused_mean_f1'] - out['historical_mean_f1']
    (HERE / 'nested_group_hierarchical_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
