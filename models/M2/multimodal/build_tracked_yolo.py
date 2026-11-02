"""Build YOLO pose sequences with Hungarian temporal identity assignment.

The standard canonical view sorts players left-to-right independently in each
frame. This view keeps a player slot attached to the nearest previous center,
which removes artificial identity swaps from the temporal statistics.
"""
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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
SPLIT = CURRENT_DATASET
POSE = REPO / 'data/player_geometry_fusion/pose_cache/frames_200/yolo11n-pose_pt'
OUT = HERE / 'tracked_yolo_seed42'
K, GLOBAL, PERSON, SEQ, SOURCE_STEP = (12, 19, 12, 50, 4)

def person_vector(box, kp, kc, conf):
    x1, y1, x2, y2 = box
    w, h = (max(float(x2 - x1), 0.001), max(float(y2 - y1), 0.001))
    c = (box[:2] + box[2:]) * 0.5
    pts = np.asarray(kp, np.float32)
    rel = np.clip((pts - np.asarray([x1, y1], np.float32)) / np.asarray([w, h], np.float32), 0, 1)
    valid = np.asarray(kc) >= 0.2
    rr = rel[valid] if np.any(valid) else rel
    return np.asarray([c[0], c[1], w, h, conf, rr[:, 0].mean(), rr[:, 1].mean(), rr[:, 0].std(), rr[:, 1].std(), float(np.mean(kc)), float(np.mean(valid)), float(np.nanmax(rel[:, 1]) - np.nanmin(rel[:, 1]))], np.float32)

def global_vector(boxes, conf, valid):
    keep = np.asarray(valid, bool) & (np.asarray(conf) >= 0.25)
    b = np.asarray(boxes, np.float32)[keep]
    c = np.asarray(conf, np.float32)[keep]
    if len(b) == 0:
        return np.zeros(GLOBAL, np.float32)
    centers = (b[:, :2] + b[:, 2:]) * 0.5
    ext = np.ptp(centers, axis=0) if len(b) > 1 else np.zeros(2)
    std = centers.std(0)
    delta = centers[:, None] - centers[None, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    pair = dist[np.triu_indices(len(b), 1)] if len(b) > 1 else np.zeros(1)
    near = np.sort(dist + np.eye(len(b)) * 999, axis=1)[:, 0] if len(b) > 1 else np.zeros(1)
    return np.asarray([len(b) / 30.0, centers[:, 0].mean(), centers[:, 1].mean(), std[0], std[1], ext[0], ext[1], pair.mean(), pair.std(), np.quantile(pair, 0.25), np.quantile(pair, 0.5), np.quantile(pair, 0.75), near.mean(), near.std(), np.quantile(near, 0.5), float(c.mean()), float(c.std()), float(np.mean(centers[:, 0] < np.median(centers[:, 0]))), float(ext[0] * ext[1])], np.float32)

def make(name):
    with _load_model_result_array(POSE / (Path(str(name)).stem + '.npz')) as d:
        boxes = d['boxes'].astype(np.float32)
        conf = d['person_conf'].astype(np.float32)
        kp = d['keypoints'].astype(np.float32)
        kc = d['keypoint_conf'].astype(np.float32)
        valid = d['valid'].astype(bool)
    out = np.zeros((SEQ, K * PERSON + GLOBAL), np.float16)
    prev = np.full((K, 2), np.nan, np.float32)
    for t in range(SEQ):
        src = min(t * SOURCE_STEP, len(valid) - 1)
        keep = np.flatnonzero(valid[src] & (conf[src] >= 0.2))
        centers = (boxes[src, keep, :2] + boxes[src, keep, 2:]) * 0.5 if len(keep) else np.zeros((0, 2), np.float32)
        assigned = {}
        active = np.flatnonzero(np.isfinite(prev).all(1))
        if len(active) and len(keep):
            cost = np.linalg.norm(prev[active, None] - centers[None, :], axis=2)
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                if cost[r, c] <= 180:
                    assigned[int(active[r])] = int(keep[c])
        used = set(assigned.values())
        empty = [i for i in range(K) if i not in assigned]
        remaining = [i for i in range(len(keep)) if int(keep[i]) not in used]
        for slot, j in zip(empty, sorted(remaining, key=lambda j: centers[j, 0])):
            assigned[slot] = int(keep[j])
        newprev = np.full_like(prev, np.nan)
        for slot, ix in assigned.items():
            out[t, slot * PERSON:(slot + 1) * PERSON] = person_vector(boxes[src, ix], kp[src, ix], kc[src, ix], conf[src, ix])
            newprev[slot] = (boxes[src, ix, :2] + boxes[src, ix, 2:]) * 0.5
        prev = newprev
        out[t, K * PERSON:] = global_vector(boxes[src], conf[src], valid[src])
    return out

def summarize(seq):
    x = seq.astype(np.float32)
    d = np.diff(x, axis=0)
    slope = ((np.arange(len(x), dtype=np.float32) - len(x) / 2)[:, None] * (x - x.mean(0))).sum(0) / (np.sum((np.arange(len(x), dtype=np.float32) - len(x) / 2) ** 2) + 1e-06)
    return np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0), np.quantile(x, 0.25, axis=0), np.quantile(x, 0.75, axis=0), d.mean(0), d.std(0), np.abs(d).mean(0), slope, x[-1] - x[0]]).astype(np.float32)

def main():
    df = pd.concat([pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'train_natural.csv'))), pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'validation.csv'))), pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'test_natural.csv')))], ignore_index=True)
    names = df.video_name.astype(str).tolist()
    OUT.mkdir(exist_ok=True)
    seq = np.lib.format.open_memmap(OUT / 'features.npy', mode='w+', dtype='float16', shape=(len(names), SEQ, K * PERSON + GLOBAL))
    summary = np.zeros((len(names), (K * PERSON + GLOBAL) * 11), np.float32)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, x in enumerate(pool.map(make, names)):
            seq[i] = x
            summary[i] = summarize(x)
            if (i + 1) % 500 == 0:
                print(i + 1, flush=True)
    seq.flush()
    np.save(OUT / 'summary.npy', summary)
    np.save(OUT / 'labels.npy', df.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy(np.int8))
    df.to_csv(OUT / 'manifest.csv', index=False)
    (OUT / 'meta.json').write_text(json.dumps({'shape': list(seq.shape), 'summary_shape': list(summary.shape), 'assignment': 'Hungarian center distance with 180px gate'}), encoding='utf-8')
    print({'shape': seq.shape, 'summary_shape': summary.shape})
if __name__ == '__main__':
    main()
