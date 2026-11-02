"""Build the E04 tracking/smoothing vector from the cached YOLO11s detector."""
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
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
SPLIT = CURRENT_DATASET
ROOT = REPO / 'data/player_geometry_fusion/pose_cache/frames_50_yolo11s_320/yolo11s-pose_pt'
OUT = HERE / 'yolo11s_e04_seed42'
SEQ = 50

def smooth(x, mask):
    p = np.pad(x, ((1, 1), (0, 0)), mode='edge')
    s = np.median(np.stack([p[:-2], p[1:-1], p[2:]], 0), 0)
    s[~mask] = x[~mask]
    return s

def summ(x, mask):
    v = x[mask] if mask.any() else x
    v = np.nan_to_num(v)
    dd = np.diff(v, axis=0) if len(v) > 1 else np.zeros((1, x.shape[-1]), np.float32)
    return np.concatenate([v.mean(0), v.std(0), v.min(0), v.max(0), np.quantile(v, 0.25, axis=0), np.quantile(v, 0.75, axis=0), dd.mean(0), dd.std(0), np.abs(dd).mean(0), ((np.arange(len(v), dtype=np.float32) - max(len(v) - 1, 1) / 2)[:, None] * (v - v.mean(0))).sum(0) / (np.sum((np.arange(len(v), dtype=np.float32) - max(len(v) - 1, 1) / 2) ** 2) + 1e-06), v[-1] - v[0]]).astype(np.float32)

def one(name):
    with _load_model_result_array(ROOT / (Path(str(name)).stem + '.npz')) as d:
        pos = d['position_features'].astype(np.float32)
        pose = d['pose_features'].astype(np.float32)
        mask = d['frame_mask'].astype(bool)
        valid = d['valid'].astype(bool)
        pc = d['person_conf'].astype(np.float32)
        kc = d['keypoint_conf'].astype(np.float32)
        cnt = valid.sum(1).astype(np.float32)
        pm = np.divide(np.where(valid, pc, 0).sum(1), np.maximum(cnt, 1))
        km = np.divide(np.where(valid[..., None], kc, 0).sum((1, 2)), np.maximum(cnt * kc.shape[-1], 1))
        q = np.column_stack([cnt / 30.0, pm, km, np.clip(cnt / 30.0, 0, 1)]).astype(np.float32)
        cur = np.concatenate([pos, pose], 1)
        return np.concatenate([summ(smooth(cur, mask), mask), summ(q, mask)[:36]])

def main():
    df = pd.concat([pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'train_natural.csv'))), pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'validation.csv'))), pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'test_natural.csv')))], ignore_index=True)
    names = df.video_name.astype(str).tolist()
    OUT.mkdir(exist_ok=True)
    arr = np.zeros((len(names), 707), np.float32)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, x in enumerate(pool.map(one, names)):
            arr[i] = x
            if (i + 1) % 500 == 0:
                print(i + 1, flush=True)
    np.save(OUT / 'features.npy', arr)
    print({'shape': arr.shape})
if __name__ == '__main__':
    main()
