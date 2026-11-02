"""EvalÃºa descriptores YOLO con tracking temporal por proximidad."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json
import os
import sys
import time
from pathlib import Path
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(experiment_directory(__file__).parents[1] / 'data' / 'player_geometry_fusion' / 'pose_cache' / 'frames_200')
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M2.studies.yolo_resnet_improvement_suite_20260811 import run_suite
OUTPUT = HERE / 'tracking_search_seed42'

def write_json(path, value):

    def safe(x):
        if isinstance(x, dict):
            return {str(k): safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [safe(v) for v in x]
        if isinstance(x, np.ndarray):
            return [safe(v) for v in x.tolist()]
        if isinstance(x, np.generic):
            return x.item()
        return x
    Path(path).write_text(json.dumps(safe(value), indent=2, ensure_ascii=False), encoding='utf-8')

def qchannels(data):
    valid = data['valid'].astype(bool)
    pc = data['person_conf'].astype(np.float32)
    kc = data['keypoint_conf'].astype(np.float32)
    count = valid.sum(1).astype(np.float32)
    pm = np.divide(np.where(valid, pc, 0).sum(1), np.maximum(count, 1))
    km = np.divide(np.where(valid[..., None], kc, 0).sum((1, 2)), np.maximum(count * kc.shape[-1], 1))
    cov = np.clip(count / 30, 0, 1)
    return (np.column_stack([count / 30, pm, km, cov]).astype(np.float32), data['frame_mask'].astype(bool))

def track_centers(data, max_distance=0.18):
    boxes = data['boxes'].astype(np.float32)
    valid = data['valid'].astype(bool)
    conf = data['person_conf'].astype(np.float32)
    centers = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    tracks = np.full((len(boxes), 30, 2), np.nan, np.float32)
    track_conf = np.zeros((len(boxes), 30), np.float32)
    active = np.zeros((len(boxes), 30), bool)
    previous = []
    for t in range(len(boxes)):
        current = [i for i in range(30) if valid[t, i]]
        used = set()
        assignments = {}
        for slot, center in previous:
            if not current:
                break
            distances = [(float(np.linalg.norm(centers[t, i] - center)), i) for i in current if i not in used]
            if distances:
                distance, i = min(distances)
                if distance <= max_distance:
                    assignments[slot] = i
                    used.add(i)
        next_slots = []
        free = [slot for slot in range(30) if slot not in assignments]
        for i in current:
            if i in used:
                continue
            if free:
                slot = free.pop(0)
                assignments[slot] = i
                used.add(i)
        for slot, i in assignments.items():
            tracks[t, slot] = centers[t, i]
            track_conf[t, slot] = conf[t, i]
            active[t, slot] = True
            next_slots.append((slot, centers[t, i]))
        previous = next_slots
    return (tracks, track_conf, active)

def summary(values, mask):
    values = np.nan_to_num(values, nan=0, posinf=0, neginf=0).astype(np.float32)
    mask = np.asarray(mask, bool)
    valid = values[mask] if mask.any() else values
    if len(valid) == 0:
        valid = np.zeros((1, values.shape[-1]), np.float32)
    delta = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, valid.shape[1]), np.float32)
    x = np.linspace(-1, 1, len(valid), dtype=np.float32)
    x = x - x.mean()
    sl = (x[:, None] * (valid - valid.mean(0))).sum(0) / (float((x * x).sum()) + 1e-06)
    return np.concatenate([valid.mean(0), valid.std(0), valid.min(0), valid.max(0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), delta.mean(0), delta.std(0), np.abs(delta).mean(0), sl, valid[-1] - valid[0]]).astype(np.float32)

def tracked_features(data):
    q, mask = qchannels(data)
    tracks, tc, active = track_centers(data)
    count = active.sum(1).astype(np.float32)
    coverage = np.column_stack([count / 30, np.divide(np.where(active, tc, 0).sum(1), np.maximum(count, 1)), active.mean(1)]).astype(np.float32)
    velocity = np.diff(tracks, axis=0, prepend=tracks[:1])
    speed = np.linalg.norm(np.nan_to_num(velocity), axis=2)
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])
    accnorm = np.linalg.norm(np.nan_to_num(acceleration), axis=2)
    motion = np.column_stack([np.nanmean(speed, axis=1), np.nanstd(speed, axis=1), np.nanmax(speed, axis=1), np.nanmean(accnorm, axis=1), np.nanstd(accnorm, axis=1), count]).astype(np.float32)
    formation = np.asarray([run_suite.formation_frame(data['boxes'][i], data['person_conf'][i], data['valid'][i]) for i in range(len(mask))], np.float32)
    pieces = []
    for left, right in zip(np.linspace(0, len(mask), 8, endpoint=False, dtype=int), np.linspace(0, len(mask), 9, dtype=int)[1:]):
        sm = mask[left:right]
        pieces.extend([summary(motion[left:right], sm), summary(formation[left:right], sm), np.asarray([float(sm.mean())], np.float32)])
    return np.concatenate(pieces + [summary(q, mask)]).astype(np.float32)

def features(frame, split):
    root = OUTPUT / 'cache'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{split}.npy'
    names_path = root / f'{split}_names.npy'
    names = frame.video_name.astype(str).to_numpy()
    if _resolve_model_artifact(path).exists() and _resolve_model_artifact(names_path).exists() and np.array_equal(_load_model_result_array(names_path, allow_pickle=True).astype(str), names):
        return _load_model_result_array(path, mmap_mode='r')
    values = []
    start = time.perf_counter()
    for i, name in enumerate(names, 1):
        values.append(tracked_features(run_suite.load_clip(name)))
        if i % 250 == 0 or i == len(names):
            print(f'features {split} {i}/{len(names)} elapsed={time.perf_counter() - start:.1f}s', flush=True)
    values = np.nan_to_num(np.asarray(values, np.float32), nan=0, posinf=0, neginf=0)
    np.save(path, values)
    np.save(names_path, names)
    return values

def metric(y, p):
    return {'accuracy': float(accuracy_score(y, p.argmax(1))), 'f1_macro': float(f1_score(y, p.argmax(1), average='macro', zero_division=0)), 'f1_weighted': float(f1_score(y, p.argmax(1), average='weighted', zero_division=0))}

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, val, test = R.common.load_base_splits(False, 'natural')
    yt, yv, ytest = map(R.labels, (train, val, test))
    cache = OUTPUT / 'base_cache'
    cache.mkdir(parents=True, exist_ok=True)
    bt = R.historical_train_probabilities(train, cache)
    bv = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_validation_predictions.csv', val)
    btest = R.stored_probabilities(R.REPRODUCTION / 'checkpoint_test_predictions.csv', test)
    xt = features(train, 'train')
    xv = features(val, 'validation')
    xs = features(test, 'test')
    scaler = StandardScaler().fit(xt)
    pca = PCA(n_components=min(128, xt.shape[1], xt.shape[0] - 1), random_state=42).fit(scaler.transform(xt))
    tr = pca.transform(scaler.transform(xt))
    va = pca.transform(scaler.transform(xv))
    te = pca.transform(scaler.transform(xs))
    rows = []
    fitted = []
    for base_mode in ('prob', 'logit'):
        base_train = bt if base_mode == 'prob' else R.logits(bt)
        base_val = bv if base_mode == 'prob' else R.logits(bv)
        base_test = btest if base_mode == 'prob' else R.logits(btest)
        model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=3000, random_state=42)
        model.fit(np.concatenate([base_train, tr], 1), yt)
        pv = R.normalize(model.predict_proba(np.concatenate([base_val, va], 1)))
        pt = R.normalize(model.predict_proba(np.concatenate([base_test, te], 1)))
        rows.append({'base_mode': base_mode, **metric(yv, pv), 'test_accuracy': metric(ytest, pt)['accuracy'], 'test_f1_macro': metric(ytest, pt)['f1_macro']})
        fitted.append((base_mode, model))
    selected = max(rows, key=lambda r: (r['f1_macro'], r['accuracy']))
    write_json(OUTPUT / 'summary.json', {'historical_test': metric(ytest, btest), 'candidates': rows, 'selected': selected, 'protocol': {'tracking': 'greedy center association', 'selection': 'natural validation only', 'test_locked': True}})
    pd.DataFrame(rows).to_csv(OUTPUT / 'summary.csv', index=False)
    print(json.dumps({'rows': rows, 'selected': selected}, indent=2), flush=True)
if __name__ == '__main__':
    main()
