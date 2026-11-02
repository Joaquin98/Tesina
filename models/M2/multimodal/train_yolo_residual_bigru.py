"""Temporal YOLO residual correction on top of frozen historical probabilities."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
import gc, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score, f1_score
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
SPLIT = CURRENT_DATASET
POSE = REPO / 'data/player_geometry_fusion/pose_cache/frames_200/yolo11n-pose_pt'
BASE = HERE / 'rgb_dynamics_seed42/fusion_features.npz'
OOF = HERE / 'oof_historical_fusion_seed42/historical_oof_probabilities.npz'
LABEL = {'line': 0, 'other': 1, 'scrum': 2}
SEQ = 200

def met(y, p):
    return {'accuracy': float(accuracy_score(y, p.argmax(1))), 'f1_macro': float(f1_score(y, p.argmax(1), average='macro', zero_division=0))}

def load_yolo(names):
    out = np.zeros((len(names), SEQ, 64), np.float32)
    masks = np.zeros((len(names), SEQ), bool)
    for i, name in enumerate(names):
        with _load_model_result_array(POSE / (Path(str(name)).stem + '.npz')) as d:
            pos = d['position_features'].astype(np.float32)
            pose = d['pose_features'].astype(np.float32)
            valid = d['valid'].astype(bool)
            pc = d['person_conf'].astype(np.float32)
            kc = d['keypoint_conf'].astype(np.float32)
            count = valid.sum(1).astype(np.float32)
            pm = np.divide(np.where(valid, pc, 0).sum(1), np.maximum(count, 1))
            km = np.divide(np.where(valid[..., None], kc, 0).sum((1, 2)), np.maximum(count * kc.shape[1], 1))
            q = np.column_stack([count / 30.0, pm, km]).astype(np.float32)
            out[i] = np.nan_to_num(np.concatenate([pos, pose, q], 1))
            masks[i] = d['frame_mask'].astype(bool)
    return (out, masks)

def model():
    y = keras.Input((SEQ, 64), name='yolo')
    m = keras.Input((SEQ,), dtype='bool', name='mask')
    b = keras.Input((3,), name='base_logits')
    z = keras.layers.LayerNormalization()(y)
    z = keras.layers.Dense(96, activation='gelu')(z)
    z = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True, dropout=0.15))(z, mask=m)
    z = keras.layers.Bidirectional(keras.layers.GRU(32, dropout=0.15))(z)
    z = keras.layers.Dense(48, activation='gelu')(z)
    z = keras.layers.Dropout(0.25)(z)
    z = keras.layers.Concatenate()([z, b])
    logits = keras.layers.Dense(3)(z)
    out = keras.layers.Activation('softmax')(logits)
    mm = keras.Model([y, m, b], out, name='historical_yolo_residual_bigru')
    mm.compile(keras.optimizers.Adam(0.0002), 'sparse_categorical_crossentropy', metrics=['accuracy'])
    return mm

def main():
    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    tr = pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'train_natural.csv')))
    va = pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'validation.csv')))
    te = pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'test_natural.csv')))
    all_df = pd.concat([tr, va, te], ignore_index=True)
    names = all_df.video_name.astype(str).tolist()
    y = all_df.tag.map(LABEL).to_numpy(np.int32)
    ntr = len(tr)
    nv = len(va)
    yolo, mask = load_yolo(names)
    mu = yolo[:ntr].reshape(-1, 64).mean(0)
    sd = yolo[:ntr].reshape(-1, 64).std(0) + 1e-05
    yolo = (yolo - mu) / sd
    d = _load_model_result_array(BASE, allow_pickle=True)
    base = np.exp(d['features'][:, -3:].astype(np.float32))
    od = _load_model_result_array(OOF, allow_pickle=True)
    om = {str(n): p for n, p in zip(od['names'], od['probabilities'])}
    for i, name in enumerate(names[:ntr]):
        if name in om:
            base[i] = om[name]
    base = np.log(np.clip(base, 1e-07, 1.0)).astype(np.float32)
    out = HERE / 'yolo_residual_bigru_seed42'
    out.mkdir(exist_ok=True)
    trw = np.asarray([1.2, 0.55, 2.0], np.float32)
    m = model()
    cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=False), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-06, verbose=1), keras.callbacks.ModelCheckpoint(out / 'best.keras', monitor='val_loss', save_best_only=True)]
    hist = m.fit([yolo[:ntr], mask[:ntr], base[:ntr]], y[:ntr], sample_weight=trw[y[:ntr]], validation_data=([yolo[ntr:ntr + nv], mask[ntr:ntr + nv], base[ntr:ntr + nv]], y[ntr:ntr + nv]), epochs=45, batch_size=32, verbose=2, callbacks=cb, shuffle=True)
    best = keras.models.load_model(_resolve_model_artifact(study_csv_path(out / 'best.keras')), compile=False)
    pv = best.predict([yolo[ntr:ntr + nv], mask[ntr:ntr + nv], base[ntr:ntr + nv]], verbose=0)
    pt = best.predict([yolo[-len(te):], mask[-len(te):], base[-len(te):]], verbose=0)
    res = {'epochs': len(hist.history['loss']), 'validation': met(y[ntr:ntr + nv], pv), 'test': met(y[-len(te):], pt), 'historical_ensemble_test': {'accuracy': 0.8776009791921665, 'f1_macro': 0.8479262047057438}, 'protocol': {'train_historical_probabilities': '5-fold OOF', 'validation_test_historical_probabilities': 'frozen checkpoint', 'yolo_sequence_length': 200}}
    (out / 'summary.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print(json.dumps(res, indent=2))
    del m, best
    keras.backend.clear_session()
    gc.collect()
if __name__ == '__main__':
    main()
