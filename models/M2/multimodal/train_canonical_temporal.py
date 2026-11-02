"""Train temporal models on canonical left-to-right player sequences."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse, gc, json, os
from pathlib import Path
import numpy as np
import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
FEAT = HERE / 'canonical_yolo_seed42/features.npy'
LAB = HERE / 'canonical_yolo_seed42/labels.npy'
BASE = HERE / 'rgb_dynamics_seed42/fusion_features.npz'
OOF = HERE / 'oof_historical_fusion_seed42/historical_oof_probabilities.npz'
SEQ = 50
DIM = 163

def metrics(y, p):
    return {'accuracy': float(accuracy_score(y, p.argmax(1))), 'f1_macro': float(f1_score(y, p.argmax(1), average='macro', zero_division=0))}

class Batch(keras.utils.PyDataset):

    def __init__(self, feat, mask, base, y, indices, shuffle=False, weights=None):
        super().__init__()
        self.feat = feat
        self.mask = mask
        self.base = base
        self.y = y
        self.ids = np.asarray(indices)
        self.shuffle = shuffle
        self.weights = weights
        self.rng = np.random.default_rng(42)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ids) / 32))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.ids)

    def __getitem__(self, i):
        ix = self.ids[i * 32:(i + 1) * 32]
        batch = ((np.asarray(self.feat[ix], np.float32), np.asarray(self.mask[ix]), np.asarray(self.base[ix], np.float32)), self.y[ix])
        return (batch[0], batch[1], self.weights[self.y[ix]]) if self.weights is not None else batch

def build(kind):
    x = keras.Input((SEQ, DIM), name='canonical')
    m = keras.Input((SEQ,), dtype='bool', name='mask')
    b = keras.Input((3,), name='historical_logits')
    z = keras.layers.LayerNormalization()(x)
    z = keras.layers.Dense(128, activation='gelu')(z)
    z = keras.layers.Dropout(0.1)(z)
    mf = keras.layers.Reshape((SEQ, 1))(keras.ops.cast(m, 'float32'))
    z = keras.layers.Multiply()([z, mf])
    if kind == 'bigru':
        z = keras.layers.Bidirectional(keras.layers.GRU(80, return_sequences=True, dropout=0.15))(z, mask=m)
        z = keras.layers.Bidirectional(keras.layers.GRU(40, return_sequences=True, dropout=0.15))(z, mask=m)
    elif kind == 'temporal_conv':
        for dilation in (1, 2, 4, 8):
            h = keras.layers.Conv1D(128, 5, padding='same', dilation_rate=dilation, activation='gelu')(z)
            h = keras.layers.Dropout(0.1)(h)
            z = keras.layers.Add()([z, h])
    elif kind == 'attention':
        for _ in range(2):
            att = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(z, z, attention_mask=keras.ops.cast(m[:, None, :], 'int32'))
            z = keras.layers.LayerNormalization()(z + att)
            ff = keras.layers.Dense(256, activation='gelu')(z)
            ff = keras.layers.Dense(128)(ff)
            z = keras.layers.LayerNormalization()(z + ff)
    else:
        raise ValueError(kind)
    pooled = keras.layers.GlobalAveragePooling1D()(z)
    pooled = keras.layers.Concatenate()([pooled, b])
    pooled = keras.layers.Dense(96, activation='gelu')(pooled)
    pooled = keras.layers.Dropout(0.3)(pooled)
    out = keras.layers.Dense(3, activation='softmax')(pooled)
    model = keras.Model([x, m, b], out, name='canonical_' + kind)
    model.compile(keras.optimizers.Adam(0.0002), 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class ValCallback(keras.callbacks.Callback):

    def __init__(self, val, y, out):
        super().__init__()
        self.val = val
        self.y = y
        self.out = out
        self.best = -1

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.val, verbose=0)
        q = metrics(self.y, p)
        print(f"epoch={epoch + 1} val_acc={q['accuracy']:.4f} val_f1={q['f1_macro']:.4f}", flush=True)
        if q['f1_macro'] > self.best:
            self.best = q['f1_macro']
            self.model.save(self.out / 'best_macro.keras', overwrite=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['bigru', 'temporal_conv', 'attention'], default='bigru')
    ap.add_argument('--epochs', type=int, default=30)
    a = ap.parse_args()
    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    feat = _load_model_result_array(FEAT, mmap_mode='r')
    labels = _load_model_result_array(LAB)
    mask = np.any(np.asarray(feat) != 0, axis=2)
    mask[:, 0] = True
    d = _load_model_result_array(BASE, allow_pickle=True)
    base = np.exp(d['features'][:, -3:].astype(np.float32))
    od = _load_model_result_array(OOF, allow_pickle=True)
    oof = {str(n): p for n, p in zip(od['names'], od['probabilities'])}
    names = d['names'].astype(str)
    for i, n in enumerate(names[:2612]):
        if n in oof:
            base[i] = oof[n]
    base = np.log(np.clip(base, 1e-07, 1)).astype(np.float32)
    tr = np.arange(0, 2612)
    va = np.arange(2612, 3265)
    te = np.arange(3265, 4082)
    out = HERE / f'canonical_temporal_{a.kind}_seed42'
    out.mkdir(exist_ok=True)
    weights = np.asarray([1.2, 0.55, 2.0], np.float32)
    train = Batch(feat, mask, base, labels, tr, True, weights)
    val = Batch(feat, mask, base, labels, va)
    test = Batch(feat, mask, base, labels, te)
    model = build(a.kind)
    v = ValCallback(val, labels[va], out)
    hist = model.fit(train, validation_data=val, epochs=a.epochs, verbose=2, callbacks=[v, keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=False), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-06)], shuffle=False)
    best = keras.models.load_model(_resolve_model_artifact(out / 'best_macro.keras'), compile=False, safe_mode=False)
    pv = best.predict(val, verbose=0)
    pt = best.predict(test, verbose=0)
    res = {'kind': a.kind, 'epochs': len(hist.history['loss']), 'validation': metrics(labels[va], pv), 'test': metrics(labels[te], pt), 'historical_ensemble_test': {'accuracy': 0.8776009791921665, 'f1_macro': 0.8479262047057438}, 'protocol': {'sequence': 50, 'canonical_players': 12, 'features_per_player': 12, 'historical_train': 'OOF'}}
    (out / 'summary.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps(res, indent=2))
    del model, best
    keras.backend.clear_session()
    gc.collect()
if __name__ == '__main__':
    main()
