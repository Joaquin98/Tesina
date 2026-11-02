"""Rama YOLO residual con histÃ³rico real congelado.

La inicializaciÃ³n es exactamente el checkpoint histÃ³rico: la proyecciÃ³n YOLO
comienza en cero y sÃ³lo esa rama se entrena. Es una prueba de si la geometrÃ­a
aporta correcciones sin permitir que el RGB pierda su representaciÃ³n.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, keras
from sklearn.metrics import accuracy_score, f1_score
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(experiment_directory(__file__).parents[1] / 'data/player_geometry_fusion/pose_cache/frames_200')
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M1 import same_rnn as S
OUT = HERE / 'frozen_historical_residual_seed42'
HIST = REPO / 'data/line_recall_training/full/training/line_weight_1_25/best_accuracy.keras'
VARIANT = 'all_residual'

def metric(y, p):
    return {'accuracy': float(accuracy_score(y, p.argmax(1))), 'f1_macro': float(f1_score(y, p.argmax(1), average='macro', zero_division=0))}

def make_model(pose_dim):
    base = keras.models.load_model(_resolve_model_artifact(HIST), compile=False)
    base.trainable = False
    rgb = keras.Input((200, 2048), name='rgb')
    pose = keras.Input((200, pose_dim), name='pose')
    mask = keras.Input((200,), dtype='bool', name='sequence_mask')
    z = keras.layers.LayerNormalization(name='yolo_norm')(pose)
    z = keras.layers.Dense(128, activation='gelu', name='yolo_mlp')(z)
    z = keras.layers.Dropout(0.1, name='yolo_dropout')(z)
    z = keras.layers.Dense(2048, activation='tanh', kernel_initializer='zeros', bias_initializer='zeros', name='yolo_zero_projection')(z)
    z = keras.layers.Rescaling(0.05, name='yolo_residual_scale')(z)
    fused = keras.layers.Add(name='historical_plus_yolo')([rgb, z])
    out = base([fused, mask], training=False)
    model = keras.Model([rgb, pose, mask], out, name='frozen_historical_yolo_residual')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class Select(keras.callbacks.Callback):

    def __init__(self, x, y, path):
        super().__init__()
        self.x = x
        self.y = y
        self.path = path
        self.best = (-1, -1)
        self.rows = []

    def on_epoch_end(self, e, logs=None):
        p = self.model.predict(self.x, verbose=0)
        m = metric(self.y, p)
        self.rows.append({'epoch': e + 1, **m})
        key = (m['f1_macro'], m['accuracy'])
        if key > self.best:
            self.best = key
            self.model.save_weights(self.path)

def inp(a):
    return [a['rgb'], a['pose'], a['rgb_mask']]

def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = R.common.load_base_splits(False, 'natural')
    print('loading train/validation', flush=True)
    a, ytr, _ = S.load_arrays(train, 'yolo11n-pose.pt', VARIANT)
    b, yv, _ = S.load_arrays(val, 'yolo11n-pose.pt', VARIANT)
    model = make_model(a['pose'].shape[-1])
    path = OUT / 'best.weights.h5'
    cb = Select(inp(b), yv, path)
    base_val = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', val)
    base_test = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', test)
    print('initial', metric(yv, base_val), flush=True)
    model.fit(inp(a), ytr, validation_data=(inp(b), yv), epochs=8, batch_size=64, callbacks=[cb], verbose=1)
    model.load_weights(_resolve_model_artifact(path))
    print('loading test', flush=True)
    c, yt, _ = S.load_arrays(test, 'yolo11n-pose.pt', VARIANT)
    p = model.predict(inp(c), verbose=0)
    result = {'historical_validation': metric(yv, base_val), 'historical_test': metric(yt, base_test), 'selected_validation': cb.best, 'selected_test': metric(yt, p), 'delta_test_accuracy': metric(yt, p)['accuracy'] - metric(yt, base_test)['accuracy'], 'delta_test_f1_macro': metric(yt, p)['f1_macro'] - metric(yt, base_test)['f1_macro'], 'protocol': {'base_checkpoint': str(HIST), 'base_frozen': True, 'yolo_projection_zero_initialized': True, 'train_rows': len(train), 'validation_rows': len(val), 'test_rows': len(test), 'selection': 'natural validation macro-F1 then accuracy', 'test_locked': True}, 'seconds': time.perf_counter() - started}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(cb.rows).to_csv(OUT / 'history.csv', index=False)
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
