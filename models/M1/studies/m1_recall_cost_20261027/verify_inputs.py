import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import os, sys, json, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from models.common.comparison import models, data
from models.M1.studies.m1_recall_cost_20261027.cached_sequence import NativeRGBCache, make_sequence
frame = pd.read_csv(_resolve_model_artifact(ROOT / 'data/m1_ensemble_vs_all_development_20261012/inputs/all_training_examples.csv')).iloc[[0, 10, 4081, 4082, 5000, 8795]]
original = models.ResNetCacheSequence(frame, batch_size=6, shuffle=False, seed=42, dual=True)
cache = NativeRGBCache()
sequence, _ = make_sequence(models, data, cache)
actual = sequence(frame, batch_size=6, shuffle=False, seed=42, dual=True)
a, y = original[0]
b, z = actual[0]
np.testing.assert_array_equal(y, z)
hashes = {}
for key in a:
    np.testing.assert_array_equal(a[key], b[key])
    hashes[key] = hashlib.sha256(a[key].tobytes()).hexdigest()
result = dict(status='PASS', blocks=frame.block_id.tolist(), inputs_bitwise_identical=True, labels_identical=True, sha256=hashes, reason='CH kept float32 and CV kept native float16; final model batches explicitly cast float32, exactly as original reader')
(HERE / 'input_equivalence.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
