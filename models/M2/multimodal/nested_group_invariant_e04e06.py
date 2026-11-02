"""Leakage-safe nested evaluation for the best invariant YOLO screen: E04+E06."""
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
import json, gc, sys
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

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def hist_probs(selected, names):
    model = build_model(selected)
    model.load_weights(_resolve_model_artifact(selected['checkpoint']))
    idx = json.loads(_resolve_model_artifact(MEM / 'resnet50_index.json').read_text())['video_names']
    lookup = {str(n): i for i, n in enumerate(idx)}
    ids = np.asarray([lookup[n] for n in names])
    feat = _load_model_result_array(MEM / 'resnet50_features.npy', mmap_mode='r')
    mask = _load_model_result_array(MEM / 'resnet50_masks.npy', mmap_mode='r')
    out = np.zeros((len(names), 3), np.float32)
    for s in range(0, len(names), 16):
        xb = np.asarray(feat[ids[s:s + 16]], np.float32)
        mb = np.asarray(mask[ids[s:s + 16]], bool)
        out[s:s + len(xb)] = model.predict_on_batch((xb, mb))
    del model, feat, mask
    keras.backend.clear_session()
    gc.collect()
    return out

def main():
    d = _load_model_result_array(HERE / 'rgb_dynamics_seed42/fusion_features.npz', allow_pickle=True)
    dyn = d['features'][:, :704].astype(np.float32)
    y = d['labels'].astype(int)
    names = d['names'].astype(str)
    groups = pd.read_csv(_resolve_model_artifact(MAN)).set_index('video_name').loc[names].source_group.astype(str).to_numpy()
    e04 = _load_model_result_array(HERE / 'multi_yolo_seed42/E04_tracking_smoothing.npy').astype(np.float32)
    e06 = _load_model_result_array(HERE / 'invariant_yolo_seed42/E06_rugby_descriptors.npy').astype(np.float32)
    X = np.concatenate([dyn, e04, e06], axis=1)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=20260712)
    rows = []
    cw_specs = [('none', None), ('balanced', 'balanced'), ('moderate', {0: 1.2, 1: 0.65, 2: 1.8})]
    for fold, (fit, hold) in enumerate(outer.split(X, y, groups), 1):
        selected = json.loads(_resolve_model_artifact(OUTER / f'outer_{fold:02d}/selected_candidate.json').read_text())
        bp = hist_probs(selected, names[fit])
        hp = hist_probs(selected, names[hold])
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=900 + fold)
        candidates = []
        for cname, cw in cw_specs:
            for C in (0.003, 0.01):
                for alpha in (0.4, 0.5, 0.6):
                    vals = []
                    for ifit, ival in inner.split(X[fit], y[fit], groups[fit]):
                        sc = StandardScaler().fit(X[fit][ifit])
                        m = LogisticRegression(C=C, class_weight=cw, solver='lbfgs', max_iter=700, random_state=42)
                        m.fit(sc.transform(X[fit][ifit]), y[fit][ifit])
                        py = m.predict_proba(sc.transform(X[fit][ival]))
                        p = (1 - alpha) * bp[ival] + alpha * py
                        vals.append(met(y[fit][ival], p.argmax(1)))
                    candidates.append({'class_weight': cname, 'C': C, 'alpha': alpha, 'inner_accuracy': float(np.mean([v['accuracy'] for v in vals])), 'inner_f1': float(np.mean([v['f1_macro'] for v in vals]))})
        sel = max(candidates, key=lambda r: (r['inner_accuracy'], r['inner_f1']))
        sc = StandardScaler().fit(X[fit])
        cw = dict(cw_specs)[sel['class_weight']]
        m = LogisticRegression(C=sel['C'], class_weight=cw, solver='lbfgs', max_iter=900, random_state=42)
        m.fit(sc.transform(X[fit]), y[fit])
        py = m.predict_proba(sc.transform(X[hold]))
        p = (1 - sel['alpha']) * hp + sel['alpha'] * py
        rows.append({'fold': fold, 'selected': sel, 'historical': met(y[hold], hp.argmax(1)), 'fused': met(y[hold], p.argmax(1)), 'n': int(len(hold))})
        print(json.dumps(rows[-1]), flush=True)
        del m, sc, bp, hp, py
        gc.collect()
    out = {'features': 'RGB dynamics + E04 tracking/smoothing + E06 rugby descriptors', 'folds': rows, 'historical_mean_accuracy': float(np.mean([r['historical']['accuracy'] for r in rows])), 'fused_mean_accuracy': float(np.mean([r['fused']['accuracy'] for r in rows])), 'historical_mean_f1': float(np.mean([r['historical']['f1_macro'] for r in rows])), 'fused_mean_f1': float(np.mean([r['fused']['f1_macro'] for r in rows]))}
    out['mean_delta_accuracy'] = out['fused_mean_accuracy'] - out['historical_mean_accuracy']
    out['mean_delta_f1'] = out['fused_mean_f1'] - out['historical_mean_f1']
    (HERE / 'nested_group_invariant_e04e06_summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
