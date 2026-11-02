"""OOF-corrected fusion: trains historical folds, then learns RGB+YOLO correction.

The train-side historical probabilities are genuinely out-of-fold, unlike the
earlier screening artifacts. Validation and test use the frozen historical
checkpoint probabilities and remain untouched for selection.
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
import gc, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, keras, tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M0.studies.natural_with_human_review import run_experiment as common
from models.M0.studies.natural_with_human_review.run_experiment import build_model
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
MEM = REPO / 'data/validation_overlap_comparison/feature_memmap'
ROOT = REPO / 'data/historical_locked_yolo_residual_20260811'
FEATURE = HERE / 'rgb_dynamics_seed42/fusion_features.npz'
OUT = HERE / 'oof_historical_fusion_seed42'
SEED = 42

class Batch(keras.utils.PyDataset):

    def __init__(self, x, m, y, shuffle=False):
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
        self.x = x
        self.m = m
        self.y = y
        self.shuffle = shuffle
        self.e = 0
        self.o = np.arange(len(y))
        self.reshuffle()

    def reshuffle(self):
        self.o = np.arange(len(self.y))
        self.o = np.random.default_rng(SEED + self.e).permutation(self.o) if self.shuffle else self.o

    def __len__(self):
        return int(np.ceil(len(self.y) / 32))

    def __getitem__(self, i):
        j = self.o[i * 32:(i + 1) * 32]
        return ((self.x[j], self.m[j]), self.y[j])

    def on_epoch_end(self):
        self.e += 1
        self.reshuffle()

def met(y, p):
    q = p.argmax(1) if np.asarray(p).ndim == 2 else np.asarray(p)
    return {'accuracy': float(accuracy_score(y, q)), 'f1_macro': float(f1_score(y, q, average='macro', zero_division=0))}

def train_oof(rgb, mask, y):
    oof = np.zeros((len(y), 3), np.float32)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = []
    for fid, (fit, hold) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        print(f'OOF fold {fid}/5 fit={len(fit)} hold={len(hold)}', flush=True)
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED + fid)
        model = build_model()
        train = Batch(rgb[fit], mask[fit], y[fit], True)
        val = Batch(rgb[hold], mask[hold], y[hold], False)
        best = HERE / f'_oof_tmp_{fid}.keras'
        callbacks = [keras.callbacks.ModelCheckpoint(best, monitor='val_accuracy', mode='max', save_best_only=True, verbose=0), keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=False, verbose=0), keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-06, verbose=0)]
        h = model.fit(train, validation_data=val, epochs=10, verbose=2, callbacks=callbacks, shuffle=False)
        sm = keras.models.load_model(_resolve_model_artifact(best), compile=False)
        p = sm.predict(val, verbose=0)
        oof[hold] = p
        folds.append({'fold': fid, 'epochs': len(h.history.get('loss', [])), **met(y[hold], p)})
        del model, sm
        keras.backend.clear_session()
        gc.collect()
    return (oof, folds)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = common.load_base_splits(False, 'natural')
    n = len(tr)
    all_names = tr.video_name.astype(str).tolist() + va.video_name.astype(str).tolist() + te.video_name.astype(str).tolist()
    idx = json.loads(_resolve_model_artifact(MEM / 'resnet50_index.json').read_text())['video_names']
    lookup = {str(v): i for i, v in enumerate(idx)}
    ids = np.asarray([lookup[x] for x in all_names])
    rgb = _load_model_result_array(MEM / 'resnet50_features.npy', mmap_mode='r')
    mask = _load_model_result_array(MEM / 'resnet50_masks.npy', mmap_mode='r')
    y = tr.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy(np.int32)
    oof_path = OUT / 'historical_oof_probabilities.npz'
    if _resolve_model_artifact(oof_path).exists():
        oof = _load_model_result_array(oof_path)['probabilities']
        folds = []
    else:
        oof, folds = train_oof(np.asarray(rgb[ids[:n]], np.float32), np.asarray(mask[ids[:n]], bool), y)
        np.savez_compressed(oof_path, probabilities=oof, names=np.asarray(tr.video_name.astype(str)))
    base_v = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', va)
    base_t = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', te)
    d = _load_model_result_array(FEATURE, allow_pickle=True)
    features = d['features'].astype(np.float32).copy()
    features[:n, -3:] = np.log(np.clip(oof, 1e-07, 1.0))
    features[n:, -3:] = np.log(np.clip(np.concatenate([base_v, base_t]), 1e-07, 1.0))
    labels = np.concatenate([y, va.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy(np.int32), te.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy(np.int32)])
    sc = StandardScaler().fit(features[:n])
    x = sc.transform(features)
    rows = []
    for C in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        for cwname, cw in [('none', None), ('balanced', 'balanced'), ('moderate', {0: 1.2, 1: 0.65, 2: 1.8})]:
            m = LogisticRegression(C=C, class_weight=cw, max_iter=3000, random_state=SEED).fit(x[:n], labels[:n])
            pv = m.predict(x[n:n + len(va)])
            pt = m.predict(x[-len(te):])
            rows.append({'model': f'C{C}_{cwname}', 'validation': met(labels[n:n + len(va)], pv), 'test': met(labels[-len(te):], pt)})
    rows.sort(key=lambda r: (r['validation']['f1_macro'], r['validation']['accuracy']), reverse=True)
    result = {'historical_oof_folds': folds, 'historical_oof_metrics': met(y, oof), 'historical_validation': met(labels[n:n + len(va)], base_v), 'historical_test': met(labels[-len(te):], base_t), 'candidates': rows, 'protocol': {'oof_folds': 5, 'oof_epochs_cap': 10, 'train_probabilities': 'out_of_fold', 'validation_test': 'frozen historical checkpoint', 'test_locked': True}}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
