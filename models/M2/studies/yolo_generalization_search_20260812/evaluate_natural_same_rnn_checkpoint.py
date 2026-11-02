from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import json, os, sys
from pathlib import Path
import numpy as np, keras
from sklearn.metrics import accuracy_score, f1_score
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(experiment_directory(__file__).parents[1] / 'data/player_geometry_fusion/pose_cache/frames_200')
REPO = experiment_directory(__file__).parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
from models.M1 import same_rnn as S
OUT = experiment_directory(__file__) / 'natural_same_rnn_seed42'
variant = 'all_residual'
train, val, test = R.common.load_base_splits(False, 'natural')
a, ya, _ = S.load_arrays(test, 'yolo11n-pose.pt', variant)
m = S.branch_model(variant, a['pose'].shape[-1])
S.initialize_from_m0(m, 42)
m.load_weights(_resolve_model_artifact(OUT / f'{variant}.weights.h5'))
inp = {'rgb': a['rgb'], 'pose': a['pose'], 'sequence_mask': a['rgb_mask']}
p = m.predict(inp, verbose=0)
out = {'variant': variant, 'test_accuracy': float(accuracy_score(ya, p.argmax(1))), 'test_f1_macro': float(f1_score(ya, p.argmax(1), average='macro', zero_division=0)), 'test_rows': len(test), 'checkpoint': str(OUT / f'{variant}.weights.h5'), 'protocol': 'natural test locked'}
(OUT / 'evaluated_checkpoint.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
