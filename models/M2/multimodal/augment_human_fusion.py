"""Augment the RGB+YOLO fusion with reviewed clips that have both caches."""
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
import json, os, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data/player_geometry_fusion/pose_cache/frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
BASE = HERE / 'rgb_dynamics_seed42'
REVIEW = REPO / 'data/aligned_human_review_weight_balanced_distribution/full/manifests/all_reviews.csv'
RGB_H = REPO / 'models/M0/feature_cache_human_review_retrain_860/resnet50'
POSE_H = REPO / 'data/historical_locked_yolo_residual_20260811/human_review_evaluation_20260811/pose_cache'
OUT = HERE / 'human_augmented_seed42'
LABEL = {'line': 0, 'other': 1, 'scrum': 2}

def met(y, p):
    p = np.asarray(p)
    pred = p.argmax(1) if p.ndim == 2 else p
    return {'accuracy': float(accuracy_score(y, pred)), 'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0))}

def dynamic_one(a, pcs):
    a = np.asarray(a, np.float32)
    seg = a.reshape(10, 20, 2048).mean(1)
    delta = np.abs(np.diff(a.reshape(10, 20, 2048), axis=1)).mean(1)
    mean = a.mean(0)
    std = a.std(0)
    return np.concatenate([pcs['segment'].transform(seg).reshape(-1), pcs['delta'].transform(delta).reshape(-1), pcs['mean'].transform(mean.reshape(1, -1)).reshape(-1), pcs['std'].transform(std.reshape(1, -1)).reshape(-1)]).astype(np.float32)

def pose_loader(name):
    with _load_model_result_array(POSE_H / f'{Path(str(name)).stem}.npz', allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}

def summarize(seq, mask, temporal=False):
    valid = np.asarray(seq, np.float32)[np.asarray(mask, bool)]
    valid = valid if len(valid) else np.asarray(seq, np.float32)
    valid = np.nan_to_num(valid)
    delta = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, valid.shape[-1]), np.float32)
    out = [valid.mean(0), valid.std(0), valid.min(0), valid.max(0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), delta.mean(0), delta.std(0), np.abs(delta).mean(0)]
    if temporal:
        t = np.linspace(-1, 1, len(valid), dtype=np.float32)
        t = t - t.mean()
        den = float((t * t).sum()) + 1e-06
        slope = (t[:, None] * (valid - valid.mean(0))).sum(0) / den
        out.extend([slope, valid[-1] - valid[0]])
    return np.concatenate(out).astype(np.float32)

def human_e04(name):
    d = pose_loader(name)
    pos = d['position_features'].astype(np.float32)
    pose = d['pose_features'].astype(np.float32)
    mask = d['frame_mask'].astype(bool)
    pc = d['person_conf'].astype(np.float32)
    kc = d['keypoint_conf'].astype(np.float32)
    valid = d['valid'].astype(bool)
    cnt = valid.sum(1).astype(np.float32)
    pm = np.divide(np.where(valid, pc, 0).sum(1), np.maximum(cnt, 1))
    km = np.divide(np.where(valid[..., None], kc, 0).sum((1, 2)), np.maximum(cnt * kc.shape[-1], 1))
    quality = np.column_stack([cnt / 30.0, pm, km, np.clip(cnt / 30.0, 0, 1)]).astype(np.float32)
    cur = np.concatenate([pos, pose], 1)
    sm = np.median(np.stack([np.pad(cur, ((1, 1), (0, 0)), mode='edge')[:-2], np.pad(cur, ((1, 1), (0, 0)), mode='edge')[1:-1], np.pad(cur, ((1, 1), (0, 0)), mode='edge')[2:]], 0), 0)
    sm[~mask] = cur[~mask]
    return np.concatenate([summarize(sm, mask, True), summarize(quality, mask)])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = _load_model_result_array(BASE / 'fusion_features.npz', allow_pickle=True)
    natural = d['features'].astype(np.float32)
    ntr = int(d['train_count'])
    nv = int(d['validation_count'])
    names = d['names'].astype(str)
    yn = np.concatenate([pd.read_csv(_resolve_model_artifact(study_csv_path(CURRENT_DATASET / 'train.csv'))).tag.map(LABEL).to_numpy(), pd.read_csv(_resolve_model_artifact(study_csv_path(CURRENT_DATASET / 'validation.csv'))).tag.map(LABEL).to_numpy(), pd.read_csv(_resolve_model_artifact(study_csv_path(CURRENT_DATASET / 'test.csv'))).tag.map(LABEL).to_numpy()])
    review = pd.read_csv(_resolve_model_artifact(study_csv_path(REVIEW)))
    review = review[review.dataset_id.map(lambda x: _resolve_model_artifact(study_csv_path(RGB_H / f'{x}.npz')).exists() and _resolve_model_artifact(study_csv_path(POSE_H / f'{x}.npz')).exists())].copy()
    review['tag'] = review.correct_label.astype(str).str.lower()
    review = review.drop_duplicates('dataset_id').reset_index(drop=True)
    print('review_with_both_caches', len(review), review.tag.value_counts().to_dict(), flush=True)
    pcs = joblib.load(_resolve_model_artifact(study_csv_path(BASE / 'dynamic_pca.joblib')))
    print('building_yolo_features', flush=True)
    yolo = np.stack([human_e04(x) for x in review.dataset_id.astype(str)])
    print('yolo_done', yolo.shape, flush=True)
    dyn = []
    for x in review.dataset_id.astype(str):
        with _load_model_result_array(RGB_H / f'{x}.npz', allow_pickle=False) as q:
            dyn.append(dynamic_one(q['frame_features'], pcs))
    print('building_dynamic_features', flush=True)
    human = np.concatenate([np.asarray(dyn, np.float32), yolo.astype(np.float32), np.zeros((len(review), 3), np.float32)], 1)
    print('human_features', human.shape, flush=True)
    yh = review.tag.map(LABEL).to_numpy(np.int32)
    train_x = natural[:ntr]
    train_y = yn[:ntr]
    val_x = natural[ntr:ntr + nv]
    val_y = yn[ntr:ntr + nv]
    test_x = natural[-817:]
    test_y = yn[-817:]
    rows = []
    print('starting_models', flush=True)
    for rw, C, cwname, cw in ((0.25, 0.003, 'balanced', 'balanced'), (0.5, 0.01, 'moderate', {0: 1.2, 1: 0.65, 2: 1.8}), (1.0, 0.003, 'none', None)):
        print('model', rw, C, cwname, flush=True)
        x = np.concatenate([train_x, human])
        y = np.concatenate([train_y, yh])
        weights = np.concatenate([np.ones(len(train_y)), np.full(len(yh), rw)])
        sc = StandardScaler().fit(train_x)
        m = LogisticRegression(C=C, class_weight=cw, max_iter=300, solver='lbfgs', random_state=42)
        m.fit(sc.transform(x), y, sample_weight=weights)
        pv = m.predict(sc.transform(val_x))
        pt = m.predict(sc.transform(test_x))
        rows.append({'review_weight': rw, 'C': C, 'class_weight': cwname, 'validation': met(val_y, pv), 'test': met(test_y, pt)})
    rows.sort(key=lambda r: (r['validation']['f1_macro'], r['validation']['accuracy']), reverse=True)
    result = {'natural_baseline_accuracy': 0.8776009791921665, 'natural_baseline_f1_macro': 0.8479262047057438, 'review_rows_with_both_caches': len(review), 'review_counts': review.tag.value_counts().to_dict(), 'candidates': rows}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'top': rows[:15]}, indent=2), flush=True)
if __name__ == '__main__':
    main()
