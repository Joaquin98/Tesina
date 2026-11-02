"""Reentrena la misma RNN histÃ³rica con YOLO sobre la particiÃ³n natural."""
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
from sklearn.utils.class_weight import compute_class_weight
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(experiment_directory(__file__).parents[1] / 'data/player_geometry_fusion/pose_cache/frames_200')
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M1 import same_rnn as S
OUT = HERE / 'natural_same_rnn_seed42'
M0 = REPO / 'data/player_geometry_fusion/results/m0/seed_42/best.weights.h5'
SEED = 42

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p.argmax(1))), 'f1_macro': float(f1_score(y, p.argmax(1), average='macro', zero_division=0))}

class Callback(keras.callbacks.Callback):

    def __init__(self, x, y, path):
        super().__init__()
        self.x = x
        self.y = y
        self.path = path
        self.best = (-1, -1)
        self.rows = []

    def on_epoch_end(self, e, logs=None):
        p = self.model.predict(self.x, verbose=0)
        m = met(self.y, p)
        self.rows.append({'epoch': e + 1, **m})
        key = (m['f1_macro'], m['accuracy'])
        if key > self.best:
            self.best = key
            self.model.save_weights(self.path)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = R.common.load_base_splits(False, 'natural')
    rows = []
    for variant in ('all_residual', 'all_projected'):
        print('loading', variant, flush=True)
        a, ytr, _ = S.load_arrays(train, 'yolo11n-pose.pt', variant)
        b, yv, _ = S.load_arrays(val, 'yolo11n-pose.pt', variant)
        inputs = lambda q: {'rgb': q['rgb'], 'pose': q['pose'], 'sequence_mask': q['rgb_mask']}
        model = S.branch_model(variant, a['pose'].shape[-1])
        S.initialize_from_m0(model, SEED)
        path = OUT / f'{variant}.weights.h5'
        cb = Callback(inputs(b), yv, path)
        w = compute_class_weight(class_weight='balanced', classes=np.arange(3), y=ytr)
        started = time.perf_counter()
        model.fit(inputs(a), ytr, validation_data=(inputs(b), yv), epochs=30, batch_size=16, class_weight={i: float(x) for i, x in enumerate(w)}, callbacks=[cb], verbose=1)
        model.load_weights(_resolve_model_artifact(path))
        c, yt, _ = S.load_arrays(test, 'yolo11n-pose.pt', variant)
        p = model.predict(inputs(c), verbose=0)
        m = met(yt, p)
        row = {'variant': variant, 'test': m, 'validation_best': cb.best, 'epochs': len(cb.rows), 'seconds': time.perf_counter() - started}
        rows.append(row)
        pd.DataFrame(cb.rows).to_csv(OUT / f'{variant}_history.csv', index=False)
        print(json.dumps(row), flush=True)
        del a, b, c, model
        keras.backend.clear_session()
    out = {'historical_test': {'accuracy': 0.8690330477356181, 'f1_macro': 0.8393456400337739}, 'candidates': rows, 'protocol': {'train': len(train), 'validation': len(val), 'test': len(test), 'selection': 'natural validation only', 'warm_start': 'M0 seed42', 'test_locked': True}}
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)
if __name__ == '__main__':
    main()
