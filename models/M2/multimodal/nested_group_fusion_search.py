"""Nested grouped selection of the RGB+YOLO/historical blend."""
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
import json
import numpy as np
import pandas as pd
import keras
import sys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
FEAT = HERE / 'rgb_dynamics_seed42/fusion_features.npz'
MAN = REPO / 'data/independent_evaluation/outputs/nested_source_manifest.csv'
OUTER = REPO / 'data/independent_evaluation/outputs/nested_evaluation'
sys.path.insert(0, str(source_directory(REPO / 'data/independent_evaluation')))
from run_nested_evaluation import build_model

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p)), 'f1_macro': float(f1_score(y, p, average='macro', zero_division=0))}

def fit_prob(X, y, hold, C):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, class_weight='balanced', max_iter=1500, random_state=42).fit(sc.transform(X), y)
    return m.predict_proba(sc.transform(hold))

def main():
    d = _load_model_result_array(FEAT, allow_pickle=True)
    X = d['features'][:, :-3].astype(np.float32)
    y = d['labels'].astype(int)
    names = d['names'].astype(str)
    groups = pd.read_csv(_resolve_model_artifact(MAN)).set_index('video_name').loc[names].source_group.astype(str).to_numpy()
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=20260712)
    Cs = (0.001, 0.003, 0.01)
    alphas = (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    rows = []
    for ofold, (fit, hold) in enumerate(outer.split(X, y, groups), 1):
        selected_hist = json.loads(_resolve_model_artifact(OUTER / f'outer_{ofold:02d}/selected_candidate.json').read_text())
        model = build_model(selected_hist)
        model.load_weights(_resolve_model_artifact(selected_hist['checkpoint']))
        idx = json.loads(_resolve_model_artifact(REPO / 'data/validation_overlap_comparison/feature_memmap/resnet50_index.json').read_text())['video_names']
        lookup = {str(n): i for i, n in enumerate(idx)}
        mem = _load_model_result_array(REPO / 'data/validation_overlap_comparison/feature_memmap/resnet50_features.npy', mmap_mode='r')
        masks = _load_model_result_array(REPO / 'data/validation_overlap_comparison/feature_memmap/resnet50_masks.npy', mmap_mode='r')
        ids = np.asarray([lookup[n] for n in names])
        base = np.zeros((len(names), 3), np.float32)
        for s in range(0, len(names), 128):
            base[s:s + 128] = model.predict([np.asarray(mem[ids[s:s + 128]], np.float32), np.asarray(masks[ids[s:s + 128]], bool)], verbose=0, batch_size=16)
        del model
        keras.backend.clear_session()
        base_hold = base[hold]
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=100 + ofold)
        candidates = []
        for C in Cs:
            for alpha in alphas:
                vals = []
                for ifit, ival in inner.split(X[fit], y[fit], groups[fit]):
                    pyo = fit_prob(X[fit][ifit], y[fit][ifit], X[fit][ival], C)
                    b = base[fit][ival]
                    p = (1 - alpha) * b + alpha * pyo
                    vals.append(met(y[fit][ival], p.argmax(1)))
                candidates.append({'C': C, 'alpha': alpha, 'inner_accuracy': float(np.mean([v['accuracy'] for v in vals])), 'inner_f1': float(np.mean([v['f1_macro'] for v in vals]))})
        selected = max(candidates, key=lambda r: (r['inner_accuracy'], r['inner_f1']))
        pyo = fit_prob(X[fit], y[fit], X[hold], selected['C'])
        p = (1 - selected['alpha']) * base_hold + selected['alpha'] * pyo
        rows.append({'outer_fold': ofold, 'selected': selected, 'historical': met(y[hold], base_hold.argmax(1)), 'fused': met(y[hold], p.argmax(1)), 'n': int(len(hold))})
    summary = {'folds': rows, 'historical_mean_accuracy': float(np.mean([r['historical']['accuracy'] for r in rows])), 'fused_mean_accuracy': float(np.mean([r['fused']['accuracy'] for r in rows])), 'historical_mean_f1': float(np.mean([r['historical']['f1_macro'] for r in rows])), 'fused_mean_f1': float(np.mean([r['fused']['f1_macro'] for r in rows]))}
    summary['mean_delta_accuracy'] = summary['fused_mean_accuracy'] - summary['historical_mean_accuracy']
    summary['mean_delta_f1'] = summary['fused_mean_f1'] - summary['historical_mean_f1']
    (HERE / 'nested_group_fusion_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
