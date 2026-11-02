"""Aprende sólo una corrección temporal YOLO sobre logits históricos congelados."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import gc, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, keras, tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M0.studies.natural_with_human_review import run_experiment as common
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M1.config import LABEL_TO_ID, LABELS
from models.M2.multimodal.train_joint import load_yolo, SEQ, POSE_ROOT
CACHE = REPO / 'data/historical_locked_yolo_residual_20260811/results_seed42'

def met(y, p):
    q = p.argmax(1)
    return {'accuracy': float(accuracy_score(y, q)), 'f1_macro': float(f1_score(y, q, average='macro', zero_division=0))}

class B(keras.utils.PyDataset):

    def __init__(self, base, yolo, mask, y, shuffle=False):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
        self.base = base
        self.yolo = yolo
        self.mask = mask
        self.y = y
        self.shuffle = shuffle
        self.e = 0
        self.o = np.arange(len(y))
        self.reshuffle()

    def reshuffle(self):
        self.o = np.arange(len(self.y))
        self.o = np.random.default_rng(42 + self.e).permutation(self.o) if self.shuffle else self.o

    def __len__(self):
        return int(np.ceil(len(self.y) / 32))

    def __getitem__(self, i):
        j = self.o[i * 32:(i + 1) * 32]
        return ((self.base[j], self.yolo[j], self.mask[j]), self.y[j])

    def on_epoch_end(self):
        self.e += 1
        self.reshuffle()

def build(scale):
    b = keras.Input((3,), name='base_logits')
    y = keras.Input((SEQ, 64), name='yolo')
    m = keras.Input((SEQ,), dtype='bool', name='mask')
    z = keras.layers.Dense(64, activation='gelu')(keras.layers.LayerNormalization()(y))
    z = keras.layers.Bidirectional(keras.layers.GRU(48, return_sequences=True, dropout=0.15))(z, mask=m)
    z = keras.layers.Bidirectional(keras.layers.GRU(24, dropout=0.15))(z)
    z = keras.layers.Concatenate()([z, b])
    z = keras.layers.Dense(48, activation='gelu')(z)
    c = keras.layers.Dense(3, kernel_initializer='zeros', bias_initializer='zeros', name='correction')(z)
    o = keras.layers.Activation('softmax')(keras.layers.Add()([b, keras.layers.Rescaling(float(scale))(c)]))
    model = keras.Model([b, y, m], o)
    model.compile(optimizer=keras.optimizers.Adam(0.0002), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class Save(keras.callbacks.Callback):

    def __init__(self, v, y, out):
        super().__init__()
        self.v = v
        self.y = y
        self.out = out
        self.best = (-1, -1)
        self.rows = []

    def on_epoch_end(self, e, logs=None):
        p = self.model.predict(self.v, verbose=0)
        q = met(self.y, p)
        self.rows.append({'epoch': e + 1, **q})
        pd.DataFrame(self.rows).to_csv(self.out / 'validation.csv', index=False)
        if (q['f1_macro'], q['accuracy']) > self.best:
            self.best = (q['f1_macro'], q['accuracy'])
            self.model.save(self.out / 'best.keras', overwrite=True)
        print(f"epoch {e + 1}: val_acc={q['accuracy']:.4f} val_f1={q['f1_macro']:.4f}", flush=True)

def main():
    tr, va, te = common.load_base_splits(False, 'natural')
    names = tr.video_name.astype(str).tolist() + va.video_name.astype(str).tolist() + te.video_name.astype(str).tolist()
    y = np.concatenate([R.labels(tr), R.labels(va), R.labels(te)])
    base_tr = R.historical_train_probabilities(tr, CACHE)
    base_v = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', va)
    base_t = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', te)
    base = np.concatenate([base_tr, base_v, base_t])
    base = np.log(np.clip(base, 1e-07, 1.0))
    yy, mask = load_yolo(names)
    mean = yy[:len(tr)].reshape(-1, 64).mean(0)
    std = yy[:len(tr)].reshape(-1, 64).std(0) + 1e-05
    yy = (yy - mean) / std
    for scale in (0.15, 0.3, 0.6, 1.0):
        out = HERE / f"residual_temporal_scale_{str(scale).replace('.', '_')}_seed42"
        out.mkdir(parents=True, exist_ok=True)
        train = B(base[:len(tr)], yy[:len(tr)], mask[:len(tr)], y[:len(tr)], True)
        val = B(base[len(tr):len(tr) + len(va)], yy[len(tr):len(tr) + len(va)], mask[len(tr):len(tr) + len(va)], y[len(tr):len(tr) + len(va)])
        test = B(base[-len(te):], yy[-len(te):], mask[-len(te):], y[-len(te):])
        model = build(scale)
        cb = Save(val, y[len(tr):len(tr) + len(va)], out)
        h = model.fit(train, validation_data=val, epochs=35, verbose=2, callbacks=[cb, keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=False), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-06)], shuffle=False)
        sm = keras.models.load_model(_resolve_model_artifact(out / 'best.keras'), compile=False)
        pv = sm.predict(val, verbose=0)
        pt = sm.predict(test, verbose=0)
        res = {'scale': scale, 'epochs': len(h.history['loss']), 'historical_validation': met(y[len(tr):len(tr) + len(va)], base_v), 'historical_test': met(y[-len(te):], base_t), 'selected_validation': met(y[len(tr):len(tr) + len(va)], pv), 'selected_test': met(y[-len(te):], pt)}
        (out / 'summary.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
        print(json.dumps(res, indent=2), flush=True)
        del model, sm
        keras.backend.clear_session()
        gc.collect()
if __name__ == '__main__':
    main()
