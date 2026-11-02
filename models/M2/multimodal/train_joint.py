"""Entrenamiento conjunto ResNet temporal + YOLO temporal.

Genera y reutiliza las características de los clips actuales. La selección de checkpoint se hace por
validación natural fija; el test natural queda bloqueado hasta el final.
"""
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
from models.common.study_data import study_csv_path, RGBFeatureStore
from models.common.legacy_training import cached_pose_arrays
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M0.studies.natural_with_human_review import run_experiment as common
from models.M1.config import LABEL_TO_ID, LABELS
SEQ = 200
RGB_DIM = 2048
YOLO_DIM = 64
SEED = 42
BATCH = 16
SPLITS = CURRENT_DATASET

def load_yolo(names):
    out = np.zeros((len(names), SEQ, YOLO_DIM), dtype=np.float32)
    masks = np.zeros((len(names), SEQ), dtype=bool)
    for i, name in enumerate(names):
        d = cached_pose_arrays(str(name), 'yolo11n-pose.pt', SEQ)
        if d:
            pos = np.asarray(d['position_features'], np.float32)
            pose = np.asarray(d['pose_features'], np.float32)
            valid = np.asarray(d['valid'], bool)
            pc = np.asarray(d['person_conf'], np.float32)
            kc = np.asarray(d['keypoint_conf'], np.float32)
            count = valid.sum(1).astype(np.float32)
            pm = np.divide(np.where(valid, pc, 0).sum(1), np.maximum(count, 1))
            km = np.divide(np.where(valid[..., None], kc, 0).sum((1, 2)), np.maximum(count * kc.shape[1], 1))
            q = np.column_stack([count / 30.0, pm, km]).astype(np.float32)
            out[i] = np.nan_to_num(np.concatenate([pos, pose, q], 1))
            masks[i] = np.asarray(d['frame_mask'], bool)
    return (out, masks)

class Batch(keras.utils.PyDataset):

    def __init__(self, ids, y, rgb, yolo, mask, weights=None, shuffle=False):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
        self.ids = np.asarray(ids)
        self.y = np.asarray(y, np.int32)
        self.rgb = rgb
        self.yolo = yolo
        self.mask = mask
        self.weights = weights
        self.shuffle = shuffle
        self.epoch = 0
        self.order = np.arange(len(self.ids))
        self._reshuffle()

    def _reshuffle(self):
        self.order = np.arange(len(self.ids))
        if self.shuffle:
            np.random.default_rng(SEED + self.epoch).shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.ids) / BATCH))

    def __getitem__(self, i):
        ix = self.order[i * BATCH:(i + 1) * BATCH]
        ids = self.ids[ix]
        x = (np.asarray(self.rgb[ids], np.float32), np.asarray(self.yolo[ids], np.float32), np.asarray(self.mask[ids], bool))
        y = self.y[ix]
        if self.weights is None:
            return (x, y)
        return (x, y, self.weights[y])

    def on_epoch_end(self):
        self.epoch += 1
        self._reshuffle()

def build_model(variant='joint_gru'):
    r = keras.Input((SEQ, RGB_DIM), name='rgb')
    y = keras.Input((SEQ, YOLO_DIM), name='yolo')
    m = keras.Input((SEQ,), dtype='bool', name='mask')
    rp = keras.layers.Dense(128, activation='gelu', name='rgb_projection')(keras.layers.LayerNormalization()(r))
    yp = keras.layers.Dense(64, activation='gelu', name='yolo_projection')(keras.layers.LayerNormalization()(y))
    if variant == 'joint_gru':
        z = keras.layers.Concatenate()([rp, yp])
        z = keras.layers.Bidirectional(keras.layers.GRU(96, return_sequences=True, dropout=0.15), name='joint_bigru')(z, mask=m)
        z = keras.layers.Bidirectional(keras.layers.GRU(48, dropout=0.15), name='joint_summary')(z)
    elif variant == 'dual_attention':
        rr = keras.layers.Bidirectional(keras.layers.GRU(80, return_sequences=True, dropout=0.15))(rp, mask=m)
        yy = keras.layers.Bidirectional(keras.layers.GRU(48, return_sequences=True, dropout=0.15))(yp, mask=m)
        z = keras.layers.Concatenate()([rr, yy])
        z = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(z, z, attention_mask=keras.ops.cast(m[:, None, :], 'int32'))
        z = keras.layers.GlobalAveragePooling1D()(z)
    elif variant == 'temporal_conv':
        z = keras.layers.Concatenate()([rp, yp])
        for dilation in (1, 2, 4, 8):
            h = keras.layers.Conv1D(192, 5, padding='same', dilation_rate=dilation, activation='gelu')(z)
            h = keras.layers.Dropout(0.15)(h)
            z = keras.layers.Add()([z, h]) if z.shape[-1] == h.shape[-1] else h
        z = keras.layers.Bidirectional(keras.layers.GRU(64, dropout=0.15))(z, mask=m)
    else:
        raise ValueError(variant)
    z = keras.layers.Dropout(0.3)(z)
    z = keras.layers.Dense(96, activation='gelu')(z)
    out = keras.layers.Dense(3, activation='softmax')(z)
    model = keras.Model([r, y, m], out, name='joint_resnet_yolo_' + variant)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def metrics(y, p):
    pred = np.argmax(p, 1)
    pr, re, f1, sup = precision_recall_fscore_support(y, pred, labels=np.arange(3), zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)), 'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(y, pred, average='weighted', zero_division=0)), 'per_class': {l: {'precision': float(pr[i]), 'recall': float(re[i]), 'f1': float(f1[i]), 'support': int(sup[i])} for i, l in enumerate(LABELS)}}

class ValSave(keras.callbacks.Callback):

    def __init__(self, val, y, out):
        super().__init__()
        self.val = val
        self.y = y
        self.out = out
        self.best_acc = -1
        self.best_f1 = -1
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.val, verbose=0)
        q = metrics(self.y, p)
        self.rows.append({'epoch': epoch + 1, **q, 'loss': float(logs.get('loss', np.nan)), 'val_loss': float(logs.get('val_loss', np.nan))})
        pd.DataFrame(self.rows).to_json(self.out / 'validation_by_epoch.json', orient='records', indent=2)
        if q['accuracy'] > self.best_acc:
            self.best_acc = q['accuracy']
            self.model.save(self.out / 'best_accuracy.keras', overwrite=True)
        if q['f1_macro'] > self.best_f1:
            self.best_f1 = q['f1_macro']
            self.model.save(self.out / 'best_macro.keras', overwrite=True)
        print(f"epoch={epoch + 1} val_acc={q['accuracy']:.4f} val_f1={q['f1_macro']:.4f}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['joint_gru', 'dual_attention', 'temporal_conv'], default='joint_gru')
    ap.add_argument('--weights', choices=['none', 'balanced', 'moderate'], default='moderate')
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--out', type=Path, default=HERE / 'runs')
    args = ap.parse_args()
    keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    tr, va, te = common.load_base_splits(False, 'natural')
    all_df = pd.concat([tr, va, te], ignore_index=True)
    names = all_df.video_name.astype(str).tolist()
    bank = RGBFeatureStore()
    ids = np.arange(len(names), dtype=np.int64)
    # Every modality follows the same concatenated partition order.
    if bank.names != names:
        raise ValueError("Las modalidades deben seguir el mismo orden de clips")
    rgb = bank.features
    yolo, masks = load_yolo(names)
    ntr = len(tr)
    ylo_mean = yolo[:ntr].reshape(-1, YOLO_DIM).mean(0)
    ylo_std = yolo[:ntr].reshape(-1, YOLO_DIM).std(0) + 1e-05
    yolo = (yolo - ylo_mean) / ylo_std
    ys = all_df.tag.map(LABEL_TO_ID).to_numpy(np.int32)
    weights = np.ones(3, np.float32) if args.weights == 'none' else np.asarray([1.5, 0.5, 3.0], np.float32) if args.weights == 'balanced' else np.asarray([1.2, 0.55, 2.0], np.float32)
    out = (args.out / ('%s_%s_seed42' % (args.variant, args.weights))).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'normalization.json').write_text(json.dumps({'yolo_mean': ylo_mean.tolist(), 'yolo_std': ylo_std.tolist()}), encoding='utf-8')
    train = Batch(ids[:ntr], ys[:ntr], rgb, yolo, masks, weights if args.weights != 'none' else None, True)
    val = Batch(ids[ntr:ntr + len(va)], ys[ntr:ntr + len(va)], rgb, yolo, masks, None, False)
    test = Batch(ids[ntr + len(va):], ys[ntr + len(va):], rgb, yolo, masks, None, False)
    model = build_model(args.variant)
    cb = ValSave(val, ys[ntr:ntr + len(va)], out)
    callbacks = [cb, keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=False), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-06, verbose=1)]
    start = time.perf_counter()
    hist = model.fit(train, validation_data=val, epochs=args.epochs, verbose=2, callbacks=callbacks, shuffle=False)
    seconds = time.perf_counter() - start
    rows = []
    for sel in ('best_accuracy', 'best_macro'):
        pth = out / (sel + '.keras')
        if not _resolve_model_artifact(study_csv_path(pth)).exists():
            continue
        sm = keras.models.load_model(_resolve_model_artifact(study_csv_path(pth)), compile=False)
        pv = sm.predict(val, verbose=0)
        pt = sm.predict(test, verbose=0)
        rows.append({'selection': sel, 'validation': metrics(ys[ntr:ntr + len(va)], pv), 'test': metrics(ys[ntr + len(va):], pt)})
        pd.DataFrame({'video_name': te.video_name, 'true_label': te.tag, 'predicted_label': [LABELS[i] for i in np.argmax(pt, 1)], **{f'prob_{l}': pt[:, i] for i, l in enumerate(LABELS)}}).to_csv(out / f'{sel}_test_predictions.csv', index=False)
        del sm
    result = {'variant': args.variant, 'weights': args.weights, 'epochs_ran': len(hist.history.get('loss', [])), 'training_seconds': seconds, 'results': rows, 'split_counts': {'train': len(tr), 'validation': len(va), 'test': len(te)}}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    del model
    keras.backend.clear_session()
    gc.collect()
if __name__ == '__main__':
    main()
