"""Fit the deployable RGB+YOLO head for the 100-frame YOLO budget."""
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
HERE = experiment_directory(__file__)
OUT = HERE / 'deployment_fusion_quantile_100frames_seed42'
FEATURES = HERE / 'rgb_dynamics_seed42/fusion_features.npz'
YOLO = HERE / 'yolo_frame_budget_20260813/e04_e06_100frames.npy'

def main() -> None:
    data = _load_model_result_array(FEATURES, allow_pickle=True)
    dynamic = data['features'][:, :704].astype(np.float32)
    labels = data['labels'].astype(np.int32)
    yolo = _load_model_result_array(YOLO, mmap_mode='r').astype(np.float32)
    x = np.concatenate([dynamic, yolo], axis=1)
    if x.shape != (len(labels), 1686):
        raise ValueError(f'Unexpected deployment matrix shape: {x.shape}')
    scaler = QuantileTransformer(n_quantiles=min(256, len(x)), output_distribution='normal', random_state=42)
    transformed = scaler.fit_transform(x)
    classifier = LogisticRegression(C=0.003, class_weight={0: 1.2, 1: 0.65, 2: 1.8}, max_iter=1200, random_state=42)
    classifier.fit(transformed, labels)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, OUT / 'quantile_scaler.joblib')
    joblib.dump(classifier, OUT / 'logistic.joblib')
    summary = {'features': '704 RGB dynamics + 707 E04 tracking/smoothing + 275 E06 rugby descriptors', 'feature_dim': 1686, 'fit_count_natural': int(len(labels)), 'yolo_frame_budget': 100, 'yolo_temporal_budget': '100 uniform YOLO frames per seven-second window', 'classifier': {'type': 'LogisticRegression', 'C': 0.003, 'class_weight': {'line': 1.2, 'other': 0.65, 'scrum': 1.8}}, 'preprocessing': {'type': 'QuantileTransformer', 'n_quantiles': 256, 'output_distribution': 'normal'}, 'blend': {'historical': 0.4, 'yolo': 0.6}, 'source_experiment': 'yolo_frame_budget_20260813', 'external_budget_metrics': {'accuracy': 0.8392944634982852, 'balanced_accuracy': 0.8281630157475696, 'f1_macro': 0.8123015138995129}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'shape': list(x.shape), 'summary': summary}, indent=2))
if __name__ == '__main__':
    main()
