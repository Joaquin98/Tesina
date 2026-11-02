"""Bounded native-dtype RGB cache and exact prepared-pose cache; no quantization."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from collections import OrderedDict
from functools import lru_cache
from types import SimpleNamespace
import numpy as np

class NativeRGBCache:

    def __init__(self, max_bytes=8 * 2 ** 30):
        self.max_bytes = max_bytes
        self.values = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def __call__(self, path, block_id):
        key = (path, block_id)
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]
        self.misses += 1
        with _load_model_result_array(path, allow_pickle=False) as z:
            if 'features' in z.files:
                rgb = z['features']
                mask = z['mask'].astype(bool)
                length = int(z['length'])
            else:
                rgb = z['frame_features']
                mask = z['frame_masks'].astype(bool)
                length = int(mask.sum())
        assert rgb.shape == (200, 2048) and mask.shape == (200,) and (rgb.dtype in (np.float16, np.float32))
        rgb.setflags(write=False)
        mask.setflags(write=False)
        value = (rgb, mask, length)
        size = rgb.nbytes + mask.nbytes
        while self.values and self.bytes + size > self.max_bytes:
            _, old = self.values.popitem(last=False)
            self.bytes -= old[0].nbytes + old[1].nbytes
        self.values[key] = value
        self.bytes += size
        return value

    def cache_clear(self):
        self.values.clear()
        self.bytes = 0

def make_sequence(models, data, rgb_cache):

    @lru_cache(maxsize=8796)
    def pose_for(path):
        value = data.geometry_sequence(data.load_yolo(SimpleNamespace(yolo_path=path)))
        value.setflags(write=False)
        return value

    class CachedSequence(models.ResNetCacheSequence):

        def __getitem__(self, batch_index):
            selected = self.indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
            rgb = []
            mask = []
            pose = []
            labels = []
            for index in selected:
                row = self.frame.iloc[int(index)]
                r, m, _ = rgb_cache(str(row.resnet_path), str(row.block_id))
                rgb.append(r)
                mask.append(m)
                labels.append(int(data.labels_for(self.frame.iloc[[int(index)]])[0]))
                if self.dual:
                    pose.append(pose_for(str(row.yolo_path)))
            inputs = {'rgb': np.asarray(rgb, np.float32), 'pose': np.asarray(pose, np.float32), 'sequence_mask': np.asarray(mask, bool)}
            return (inputs, np.asarray(labels, np.int64))
    return (CachedSequence, pose_for)
