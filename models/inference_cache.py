"""Numeric feature caches for offline video inference; never cache decisions."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / '.runtime/cache/features'
SCHEMA = 1
RAW_FIELDS = ('boxes', 'person_conf', 'keypoints', 'keypoint_conf', 'valid', 'position_features', 'pose_features', 'frame_mask')

def file_digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def load_with_revision(path, loader):
    """Identify the frozen weights actually loaded, including file updates."""
    before = file_digest(path)
    model = loader(path)
    after = file_digest(path)
    return model, before if before == after else None

def _constant_record(value):
    if isinstance(value, types.CodeType):
        return _code_record(value)
    if isinstance(value, frozenset):
        return ['frozenset', sorted((_constant_record(item) for item in value), key=lambda item: json.dumps(item))]
    if isinstance(value, tuple):
        return ['tuple', [_constant_record(item) for item in value]]
    return [type(value).__name__, repr(value)]

def _code_record(code):
    """Stable across processes and Python's runtime string interning."""
    constants = [_constant_record(value) for value in code.co_consts]
    return [code.co_code.hex(), code.co_exceptiontable.hex(), constants, code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount, code.co_flags]

def _function_record(function):
    return [_code_record(function.__code__), _constant_record(function.__defaults__), {key: _constant_record(value) for key, value in sorted((function.__kwdefaults__ or {}).items())}]

def pipeline_revision(model):
    digest = hashlib.sha256()
    modules = {'models.M0.config', 'models.M0.data_utils', 'models.common.spatial', __name__}
    for cls in type(model).__mro__:
        for name, value in sorted(vars(cls).items()):
            function = getattr(value, '__func__', value)
            if hasattr(function, '__code__'):
                digest.update((cls.__qualname__ + '.' + name).encode())
                digest.update(json.dumps(_function_record(function)).encode())
    for name in ('_detect_frame', '_build_geometry_features'):
        function = getattr(model, name, None)
        if function is not None:
            modules.add(function.__module__)
    for name in sorted(modules):
        module = sys.modules.get(name)
        if module is None:
            continue
        for key, value in sorted(vars(module).items()):
            if inspect.isfunction(value) and value.__module__ == name:
                digest.update((name + '.' + key).encode())
                digest.update(json.dumps(_function_record(value)).encode())
            elif isinstance(value, (str, int, float, bool)) and not key.startswith('__'):
                digest.update(json.dumps([name, key, value], sort_keys=True).encode())
    return digest.hexdigest()

def cache_key(model, video, parameters):
    rgb = getattr(model, '_feature_cache_rgb_revision', None)
    pose = getattr(model, '_feature_cache_pose_revision', None)
    if not rgb or (hasattr(model, 'pose_detector') and pose is None):
        return None
    config = {}
    for name in ('extractor_name', 'preprocessing_filter', 'pose_input_mode', 'MAX_PEOPLE', 'POSE_PERSON_CONFIDENCE', 'POSE_KEYPOINT_CONFIDENCE', 'YOLO_FRAME_BUDGET', 'YOLO_WINDOW_DURATION', 'RGB_LENGTH', 'GEOMETRY_FRAMES', 'POSE_FRAMES_PER_WINDOW'):
        value = getattr(model, name, None)
        config[name] = value.item() if isinstance(value, np.generic) else value
    record = {'schema': SCHEMA, 'video_sha256': file_digest(video), 'rgb_weights': rgb, 'pose_weights': pose, 'pipeline': pipeline_revision(model), 'family': type(model).__module__ + '.' + type(model).__qualname__, 'parameters': parameters, 'config': config, 'libraries': {'numpy': np.__version__, 'opencv': cv2.__version__, **{name: getattr(sys.modules.get(name), '__version__', None) for name in ('tensorflow', 'keras', 'torch', 'ultralytics')}}}
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def auxiliary_state(model):
    if getattr(model, '_feature_cache_pose_revision', None) is None:
        return {}
    if hasattr(model, '_yolo_source_indices'):
        values = {'yolo_indices': np.asarray(model._yolo_source_indices)}
        if model._raw_geometry is not None:
            values.update({'raw.' + key: np.asarray(model._raw_geometry[key]) for key in RAW_FIELDS})
        return values
    return {key: np.asarray(getattr(model, attr)) for key, attr in [('positions', '_geometry_positions'), ('poses', '_geometry_poses'), ('mask', '_geometry_mask')]}

def restore_auxiliary(model, values, count):
    if getattr(model, '_feature_cache_pose_revision', None) is None:
        return
    if hasattr(model, '_yolo_source_indices'):
        indices = values['yolo_indices']
        expected = model._yolo_source_indices
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or not np.array_equal(indices, expected) or (len(indices) and (indices.min() < 0 or indices.max() >= count)):
            raise ValueError('Invalid cached pose sampling')
        raw = {key: values['raw.' + key] for key in RAW_FIELDS} if len(indices) else None
        if raw is not None:
            people = int(model.MAX_PEOPLE)
            shapes = {'boxes': (len(indices), people, 4), 'person_conf': (len(indices), people), 'keypoints': (len(indices), people, 17, 2), 'keypoint_conf': (len(indices), people, 17), 'valid': (len(indices), people), 'position_features': (len(indices), 31), 'pose_features': (len(indices), 30), 'frame_mask': (len(indices),)}
            if any(raw[key].shape != shape for key, shape in shapes.items()):
                raise ValueError('Invalid cached pose dimensions')
            if any(raw[key].dtype != np.bool_ for key in ('valid', 'frame_mask')):
                raise ValueError('Invalid cached pose masks')
        model._raw_geometry = raw
        model._yolo_source_indices = indices
        model._yolo_index_set = set(indices.tolist())
        model._sampled_frame_index = count
    else:
        if values['positions'].shape != (count, 31) or values['poses'].shape != (count, 30) or values['mask'].shape != (count,) or values['mask'].dtype != np.bool_:
            raise ValueError('Invalid cached geometry dimensions')
        model._geometry_positions = values['positions']
        model._geometry_poses = values['poses']
        model._geometry_mask = values['mask']

def read_features(path, key, count, feature_dim):
    try:
        with np.load(path, allow_pickle=False) as stored:
            if int(stored['__schema']) != SCHEMA or str(stored['__key']) != key:
                return None
            values = {name: stored[name] for name in stored.files if not name.startswith('__')}
        features = values['features']
        if features.ndim != 2 or features.shape[0] != count or features.shape[1] <= 0 or (feature_dim is not None and features.shape[1] != feature_dim) or features.dtype.kind != 'f':
            return None
        if any(value.dtype.kind not in 'fiub' or not np.isfinite(value).all() for value in values.values()):
            return None
        return values
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None

def write_features(path, key, values):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.partial', delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, __schema=np.asarray(SCHEMA), __key=np.asarray(key), **values)
        os.replace(temporary, path)
        return True
    except OSError:
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

def cached_features(model, video, parameters, count, compute, *, use_cache=True, cache_dir=None, auxiliary=True):
    info = {'enabled': bool(use_cache), 'hit': False, 'path': None, 'status': 'disabled'}
    if not use_cache:
        return compute(), info
    try:
        key = cache_key(model, video, parameters)
    except OSError:
        key = None
    if key is None:
        info['status'] = 'unavailable'
        return compute(), info
    path = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE
    path = path / key[:2] / (key + '.npz')
    info['path'] = str(path)
    try:
        feature_dim = int(model.feature_extractor.output_shape[-1])
    except (AttributeError, TypeError, ValueError):
        feature_dim = None
    values = read_features(path, key, count, feature_dim)
    if values is not None:
        try:
            if auxiliary:
                restore_auxiliary(model, values, count)
            info.update(hit=True, status='hit')
            return values['features'], info
        except (KeyError, ValueError, TypeError):
            pass
    features = compute()
    values = {'features': features}
    if auxiliary:
        values.update(auxiliary_state(model))
    complete = features.ndim == 2 and features.shape[0] == count and all(np.isfinite(value).all() for value in values.values())
    info['status'] = 'created' if complete and write_features(path, key, values) else 'not_saved'
    return features, info
