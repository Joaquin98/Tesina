"""Fit the frozen winning YOLO correction on all 4,082 natural clips.

This creates the operational artifact after evaluation.  Human-review clips
are intentionally not concatenated to this 1,414-D feature matrix: their
cached RGB/E04/E06 vectors are not available, and the separate 998-review
meta-head experiment is reported as a negative control.
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
import joblib
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
HERE = experiment_directory(__file__)

def main():
    data = _load_model_result_array(HERE / 'rgb_dynamics_seed42/fusion_features.npz', allow_pickle=True)
    dyn = data['features'][:, :704].astype(np.float32)
    e04 = _load_model_result_array(HERE / 'multi_yolo_seed42/E04_tracking_smoothing.npy').astype(np.float32)
    e06 = _load_model_result_array(HERE / 'invariant_yolo_seed42/E06_rugby_descriptors.npy').astype(np.float32)
    X = np.concatenate([dyn, e04, e06], axis=1)
    y = data['labels'].astype(int)
    scaler = QuantileTransformer(n_quantiles=min(256, len(X)), output_distribution='normal', random_state=42).fit(X)
    model = LogisticRegression(C=0.003, class_weight={0: 1.2, 1: 0.65, 2: 1.8}, max_iter=700, random_state=42).fit(scaler.transform(X), y)
    outdir = HERE / 'deployment_fusion_quantile_all_seed42'
    outdir.mkdir(exist_ok=True)
    joblib.dump(scaler, outdir / 'quantile_scaler.joblib')
    joblib.dump(model, outdir / 'logistic.joblib')
    summary = {'features': '704 RGB dynamics + 707 E04 tracking/smoothing + 275 E06 rugby descriptors', 'feature_dim': int(X.shape[1]), 'fit_count_natural': int(len(X)), 'classifier': {'type': 'LogisticRegression', 'C': 0.003, 'class_weight': {'line': 1.2, 'other': 0.65, 'scrum': 1.8}}, 'preprocessing': {'type': 'QuantileTransformer', 'n_quantiles': 256, 'output_distribution': 'normal'}, 'blend': {'historical': 0.4, 'yolo': 0.6}, 'artifacts': ['quantile_scaler.joblib', 'logistic.joblib'], 'human_review_policy': '998 reviewed clips were not appended because their RGB/E04/E06 vectors are not cached; the separate review meta-head was evaluated and was negative.', 'validation': 'cross_split_fixed_quantile_summary.json'}
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
