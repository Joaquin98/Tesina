"""Cache camera-invariant YOLO summaries for every natural clip."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.common.legacy_training import DATASET as CURRENT_DATASET, current_frames
from models.common.study_data import study_csv_path
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os, sys
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data/player_geometry_fusion/pose_cache/frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite import load_clip, make_variant
SPLIT = CURRENT_DATASET
OUT = HERE / 'invariant_yolo_seed42'

def one(pair):
    i, name = pair
    d = load_clip(name)
    return (i, {rep: make_variant(d, rep) for rep in ('E05_camera_normalized', 'E06_rugby_descriptors', 'E09_player_graph')})

def main():
    df = pd.concat([pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'train_natural.csv'))), pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'validation.csv'))), pd.read_csv(_resolve_model_artifact(study_csv_path(SPLIT / 'test_natural.csv')))], ignore_index=True)
    names = df.video_name.astype(str).tolist()
    OUT.mkdir(exist_ok=True)
    allx = {rep: [] for rep in ('E05_camera_normalized', 'E06_rugby_descriptors', 'E09_player_graph')}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, (i, x) in enumerate(pool.map(one, enumerate(names)), 1):
            for rep in allx:
                allx[rep].append(x[rep])
            if done % 200 == 0:
                print(done, flush=True)
    for rep, x in allx.items():
        np.save(OUT / (rep + '.npy'), np.asarray(x, np.float32))
        print(rep, np.asarray(x).shape, flush=True)
    df.to_csv(OUT / 'manifest.csv', index=False)
if __name__ == '__main__':
    main()
