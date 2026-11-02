"""Dispatch comparative training to the implementation of each model family."""
from __future__ import annotations
from models.training_common import *
from models.M0.comparison_training import _configure_m0_trainable, fit_m0
from models.M1.comparison_training import build_dual_model, fit_m1_dual, M1ExactSequence, _m1_rgb_model, _m1_pose_model, _m1_geometry_matrix, _predict_m1_exact, fit_predict_m1
from models.M2.comparison_training import _predict_static_b, _static_head, fit_predict_m2
from models.M3.comparison_training import _temporal_arrays, _softmax, _predict_temporal_arrays, _predict_temporal_checkpoint, _predict_temporal_ensemble, _fit_temporal, fit_predict_m3_or_m4

def fit_predict(model_name, train, development, test, *, level, seed, run_dir, epochs=40, patience=7, batch_size=16) -> PredictionBundle:
    seed_everything(seed)
    if model_name == 'M0':
        model_path = run_dir / 'model.keras'
        model = fit_m0(train, development, level=level, seed=seed, output=model_path, epochs=epochs, patience=patience, batch_size=batch_size)
        return PredictionBundle(probabilities=_predict_keras(model, test, batch_size=batch_size), development_probabilities=_predict_keras(model, development, batch_size=batch_size), training_probabilities=None if level == 'B' else _predict_keras(model, train, batch_size=batch_size), artifacts=[M0_PATH if level == 'B' else model_path])
    if model_name == 'M1':
        return fit_predict_m1(train, development, test, level=level, seed=seed, run_dir=run_dir, epochs=epochs, patience=patience, batch_size=batch_size)
    if model_name == 'M2':
        return fit_predict_m2(train, development, test, level=level, seed=seed, run_dir=run_dir, epochs=epochs, patience=patience, batch_size=batch_size)
    if model_name in {'M3', 'M4'}:
        return fit_predict_m3_or_m4(model_name, train, development, test, level=level, seed=seed, run_dir=run_dir, epochs=epochs, patience=patience, batch_size=batch_size)
    raise ValueError(f'Unknown model: {model_name}')
