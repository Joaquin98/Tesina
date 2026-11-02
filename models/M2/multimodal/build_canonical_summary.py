import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import numpy as np
HERE = experiment_directory(__file__)
F = _load_model_result_array(HERE / 'canonical_yolo_seed42/features.npy', mmap_mode='r')
OUT = HERE / 'canonical_yolo_seed42/canonical_summary.npy'
arr = np.lib.format.open_memmap(OUT, mode='w+', dtype='float32', shape=(len(F), 1630))
for s in range(0, len(F), 128):
    x = np.asarray(F[s:s + 128], np.float32)
    d = np.diff(x, axis=1)
    t = np.arange(x.shape[1], dtype=np.float32)
    tc = t - t.mean()
    slope = (tc[None, :, None] * (x - x.mean(1, keepdims=True))).sum(1) / (np.sum(tc * tc) + 1e-06)
    arr[s:s + len(x)] = np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1), np.quantile(x, 0.25, axis=1), np.quantile(x, 0.75, axis=1), d.mean(1), d.std(1), np.abs(d).mean(1), slope], 1)
arr.flush()
print({'shape': arr.shape, 'path': str(OUT)})
