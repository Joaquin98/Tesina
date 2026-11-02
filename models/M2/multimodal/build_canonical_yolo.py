"""Build a temporally ordered, per-player YOLO representation.

The original cache ranks detections by confidence independently in each frame,
which is appropriate for clip statistics but unstable for an RNN.  Here the
top K detections are re-ranked left-to-right in every frame and represented by
compact box/posture descriptors plus global formation statistics.
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
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
SPLIT = CURRENT_DATASET
POSE = REPO / 'data/player_geometry_fusion/pose_cache/frames_200/yolo11n-pose_pt'
K = 12
GLOBAL = 19
PERSON = 12
DIM = K * PERSON + GLOBAL
SEQ = 50
SOURCE_STEP = 4

def person_vector(box, kp, kc, conf):
    x1, y1, x2, y2 = box
    w = max(float(x2 - x1), 0.001)
    h = max(float(y2 - y1), 0.001)
    c = (box[:2] + box[2:]) * 0.5
    valid = np.asarray(kc) >= 0.2
    pts = np.asarray(kp, dtype=np.float32)
    rel = np.clip((pts - np.asarray([x1, y1], np.float32)) / np.asarray([w, h], np.float32), 0, 1)
    rr = rel[valid] if np.any(valid) else rel
    mean = rr.mean(0)
    std = rr.std(0)
    body = float(np.nanmax(rel[:, 1]) - np.nanmin(rel[:, 1])) if np.any(np.isfinite(rel)) else 0.0
    return np.asarray([c[0], c[1], w, h, conf, mean[0], mean[1], std[0], std[1], float(np.mean(kc)), float(np.mean(valid)), body], np.float32)

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
    nearest = np.sort(dist + np.eye(len(b)) * 999, axis=1)[:, 0] if len(b) > 1 else np.zeros(1)
    return np.asarray([len(b) / 30.0, centers[:, 0].mean(), centers[:, 1].mean(), std[0], std[1], ext[0], ext[1], pair.mean(), pair.std(), np.quantile(pair, 0.25), np.quantile(pair, 0.5), np.quantile(pair, 0.75), nearest.mean(), nearest.std(), np.quantile(nearest, 0.5), float(c.mean()), float(c.std()), float(np.mean(centers[:, 0] < np.median(centers[:, 0]))), float(ext[0] * ext[1])], np.float32)

def make(name):
    with _load_model_result_array(POSE / (Path(str(name)).stem + '.npz')) as d:
        boxes = d['boxes'].astype(np.float32)
        conf = d['person_conf'].astype(np.float32)
        kp = d['keypoints'].astype(np.float32)
        kc = d['keypoint_conf'].astype(np.float32)
        valid = d['valid'].astype(bool)
        mask = d['frame_mask'].astype(bool)
    out = np.zeros((SEQ, DIM), np.float16)
    for t in range(SEQ):
        src = min(t * SOURCE_STEP, len(valid) - 1)
        keep = np.flatnonzero(valid[src] & (conf[src] >= 0.2))
        if len(keep):
            centers = (boxes[src, keep, :2] + boxes[src, keep, 2:]) * 0.5
            keep = keep[np.argsort(centers[:, 0])[:K]]
            for j, ix in enumerate(keep):
                out[t, j * PERSON:(j + 1) * PERSON] = person_vector(boxes[src, ix], kp[src, ix], kc[src, ix], conf[src, ix])
        out[t, K * PERSON:] = global_vector(boxes[src], conf[src], valid[src])
    return out

def main():
    df = pd.concat([pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'train_natural.csv'))), pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'validation.csv'))), pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'test_natural.csv')))], ignore_index=True)
    OUT = HERE / 'canonical_yolo_seed42'
    OUT.mkdir(exist_ok=True)
    path = OUT / 'features.npy'
    arr = np.lib.format.open_memmap(path, mode='w+', dtype='float16', shape=(len(df), SEQ, DIM))
    names = df.video_name.astype(str).tolist()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, (i, features) in enumerate(pool.map(lambda pair: (pair[0], make(pair[1])), enumerate(names)), 1):
            arr[i] = features
            if done % 100 == 0:
                print(done, flush=True)
    arr.flush()
    np.save(OUT / 'labels.npy', df.tag.map({'line': 0, 'other': 1, 'scrum': 2}).to_numpy(np.int8))
    df.to_csv(OUT / 'manifest.csv', index=False)
    (OUT / 'meta.json').write_text(json.dumps({'dim': DIM, 'person_dim': PERSON, 'global_dim': GLOBAL, 'players': K, 'sequence': SEQ, 'source_step': SOURCE_STEP}), encoding='utf-8')
    print({'shape': arr.shape, 'path': str(path)})
if __name__ == '__main__':
    main()
