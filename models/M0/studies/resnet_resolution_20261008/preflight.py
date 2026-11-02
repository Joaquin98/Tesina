"""Read-only source audit and paired small extraction before resolution training.
Writes exclusively below this experiment directory. Existing datasets, models,
caches and thesis files are never modified. No test performance is inspected.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse, csv, hashlib, json, os, platform, sys, time, threading
from pathlib import Path
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
CONFIG = json.loads(_resolve_model_artifact(HERE / 'config.json').read_text(encoding='utf-8-sig'))
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', str(CONFIG['tensorflow_threads']))
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
import cv2
import numpy as np
import pandas as pd
import psutil
from models.common.spatial import training_normalization_geometry, extract_training_equivalent_source_roi
from models.common.studies.final_tests_feature_cache_20260826.build_cache import sampled_block_indices

def digest(path):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(path)).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

def probe(path):
    cap = cv2.VideoCapture(str(path))
    r = dict(opened=cap.isOpened(), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=float(cap.get(cv2.CAP_PROP_FPS)), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return r

def roi(frame, domain):
    if domain == 'CV':
        return extract_training_equivalent_source_roi(frame, target_size=224)
    h, w = frame.shape[:2]
    sw, sh = (int(round(w * 0.5)), int(round(h * 0.5)))
    x, y = ((sw - 224) // 2, (sh - 224) // 2)
    bounds = (int(round(x * w / sw)), int(round(y * h / sh)), int(round((x + 224) * w / sw)), int(round((y + 224) * h / sh)))
    x0, y0, x1, y1 = bounds
    return frame[y0:y1, x0:x1]

def sample_indices(info, domain, row=None):
    if domain == 'CV':
        return sampled_block_indices(info['fps'], info['frames'], float(row.start_time), float(row.end_time))[0][:200]
    n = max(1, int(np.floor(info['frames'] / info['fps'] * 25.0)))
    ts = np.arange(n, dtype=np.float32) / 25.0
    return np.clip(np.rint(ts * float(info['fps'])).astype(np.int32), 0, info['frames'] - 1)[:200]

def native_frames(path, indices, domain):
    cap = cv2.VideoCapture(str(path))
    wanted = set(map(int, indices))
    first, last = (min(wanted), max(wanted))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    decoded = {}
    for idx in range(first, last + 1):
        ok, f = cap.read()
        if not ok:
            break
        if idx in wanted:
            decoded[idx] = roi(f, domain).copy()
    cap.release()
    if wanted - set(decoded):
        raise RuntimeError(f'Missing decoded frames from {path}: {wanted - set(decoded)}')
    return [decoded[int(i)] for i in indices]

def sources():
    h = pd.read_csv(_resolve_model_artifact(ROOT / 'data/independent_evaluation/outputs/nested_source_manifest.csv'))
    f = pd.read_csv(_resolve_model_artifact(ROOT / 'data/final_tests_feature_cache_20260826/outputs/final_test_feature_manifest.csv'))
    return (h, f)

def audit():
    out = HERE / 'outputs'
    out.mkdir(exist_ok=True)
    h, f = sources()
    records = []
    failures = []
    t = time.perf_counter()
    for i, row in enumerate(h.itertuples(index=False)):
        path = ROOT / 'data/7seg unique' / row.video_name
        ref = Path(row.clip_path)
        cache = ROOT / 'models/M0/feature_cache/resnet50' / f'{Path(row.video_name).stem}.npz'
        status = []
        p = probe(path)
        r = probe(ref)
        if not p['opened']:
            status.append('source_unreadable')
        elif p['fps'] <= 0 or p['frames'] <= 0:
            status.append('source_invalid_time')
        if not r['opened']:
            status.append('reference_unreadable')
        if (r['width'], r['height']) != (224, 224):
            status.append('reference_not224')
        if p['frames'] != r['frames'] or abs(p['fps'] - r['fps']) > 0.001:
            status.append('source_reference_time_mismatch')
        if min(p['width'], p['height']) < 448:
            status.append('native_roi_below448')
        indices = sample_indices(p, 'CH') if p['fps'] > 0 and p['frames'] > 0 else np.array([])
        try:
            with _load_model_result_array(cache, allow_pickle=False) as c:
                mask = c['frame_masks'].astype(bool)
                if c['frame_features'].shape != (200, 2048) or mask.shape != (200,):
                    status.append('cache_shape_mismatch')
                expected = np.arange(200) < len(indices)
                if not np.array_equal(mask, expected):
                    status.append('cache_mask_time_mismatch')
                cache_length = int(mask.sum())
        except Exception as e:
            status.append('cache_unreadable:' + str(e))
            cache_length = -1
        rec = dict(domain='CH', id=row.video_name, source=str(path), reference=str(ref), status=';'.join(status) or 'ok', samples=len(indices), cache_length=cache_length, **p)
        records.append(rec)
        if status:
            failures.append(rec)
        if i % 250 == 0:
            print(f'AUDIT_CH {i}/{len(h)} failures={len(failures)}', flush=True)
    for path, part in f.groupby('source_video', sort=False):
        p = probe(path)
        for row in part.itertuples(index=False):
            status = []
            if not p['opened'] or p['fps'] <= 0 or p['frames'] <= 0:
                status.append('source_unreadable_or_invalid')
            indices = sample_indices(p, 'CV', row) if not status else []
            if p['opened']:
                g = training_normalization_geometry(p['width'], p['height'])
                if min(g['source_roi_width'], g['source_roi_height']) < 448:
                    status.append('native_roi_below448')
            try:
                with _load_model_result_array(row.resnet_cache_path, allow_pickle=False) as c:
                    mask = c['mask'].astype(bool)
                    if c['features'].shape != (200, 2048):
                        status.append('cache_shape_mismatch')
                    if not np.array_equal(mask, np.arange(200) < len(indices)):
                        status.append('cache_mask_time_mismatch')
                    cache_length = int(mask.sum())
            except Exception as e:
                status.append('cache_unreadable:' + str(e))
                cache_length = -1
            rec = dict(domain='CV', id=row.dataset_id, source=path, reference='', status=';'.join(status) or 'ok', samples=len(indices), cache_length=cache_length, **p)
            records.append(rec)
            if status:
                failures.append(rec)
        print(f'AUDIT_CV {Path(path).name} failures={len(failures)}', flush=True)
    pd.DataFrame(records).to_csv(out / 'source_audit.csv', index=False)
    info = dict(status='ready' if not failures else 'blocked', historical=len(h), final=len(f), source_audit_seconds=time.perf_counter() - t, failures=len(failures), failure_examples=failures[:15], failure_counts=pd.DataFrame(failures).status.value_counts().to_dict() if failures else {}, source_samples=int(sum((r['samples'] for r in records))))
    write_json(out / 'source_audit.json', info)
    print(json.dumps(info, indent=2), flush=True)
    return 0 if not failures else 2

class Peak:

    def __init__(self):
        self.process = psutil.Process()
        self.value = 0
        self.done = threading.Event()

    def __enter__(self):

        def poll():
            while not self.done.is_set():
                self.value = max(self.value, self.process.memory_info().rss)
                self.done.wait(0.02)
        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
        self.value = max(self.value, self.process.memory_info().rss)

def smoke():
    import tensorflow as tf, keras
    out = HERE / 'outputs/smoke'
    out.mkdir(parents=True, exist_ok=True)
    weights = Path.home() / '.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if not _resolve_model_artifact(weights).is_file():
        raise FileNotFoundError('Local ImageNet weights unavailable')
    h, f = sources()
    examples = [('CH', ROOT / 'data/7seg unique' / h.iloc[0].video_name, None), ('CV', Path(f.iloc[0].source_video), f.iloc[0])]
    rois = []
    meta = []
    for domain, path, row in examples:
        p = probe(path)
        indices = sample_indices(p, domain, row)[:8]
        frames = native_frames(path, indices, domain)
        if any((min(x.shape[:2]) < 448 for x in frames)):
            raise RuntimeError('Native source ROI is too small')
        rois += frames
        meta.append(dict(domain=domain, source=str(path), indices=indices.tolist(), native_roi_shape=list(frames[0].shape), video=p))
        cv2.imwrite(str(out / f'{domain}_native_roi.png'), frames[0])
        tiles = [cv2.resize(cv2.resize(frames[0], (s, s), interpolation=cv2.INTER_LINEAR), (448, 448)) for s in CONFIG['resolutions']]
        cv2.imwrite(str(out / f'{domain}_resolutions_preview.png'), np.concatenate(tiles, axis=1))
    rows = []
    for size in CONFIG['resolutions']:
        keras.backend.clear_session()
        t = time.perf_counter()
        with Peak() as peak:
            base = keras.applications.ResNet50(include_top=False, pooling='avg', input_shape=(size, size, 3), weights=None)
            base.load_weights(_resolve_model_artifact(weights))
            base.trainable = False
            x = np.asarray([cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1] for frame in rois], dtype=np.float32)
            x = keras.applications.resnet50.preprocess_input(x)
            base(x[:1], training=False).numpy()
            ticks = []
            features = []
            for start in range(0, len(x), CONFIG['feature_batch_size']):
                tick = time.perf_counter()
                v = base(x[start:start + CONFIG['feature_batch_size']], training=False).numpy()
                ticks.append(time.perf_counter() - tick)
                features.append(v)
            v = np.concatenate(features)
            if v.shape != (len(rois), 2048) or not np.isfinite(v).all():
                raise RuntimeError('Invalid frozen features')
            np.savez_compressed(out / f'features_{size}.npz', features=v, condition=size)
        rows.append(dict(resolution=size, n_frames=len(rois), inference_s=sum(ticks), s_per_frame=sum(ticks) / len(rois), elapsed_with_model_init_s=time.perf_counter() - t, peak_rss_bytes=peak.value, feature_shape=list(v.shape)))
        print('SMOKE_RESULT', json.dumps(rows[-1]), flush=True)
        del base, x, v
    pd.DataFrame(rows).to_csv(out / 'benchmark.csv', index=False)
    write_json(out / 'smoke.json', dict(status='passed', examples=meta, results=rows, weights=str(weights), weights_sha256=digest(weights), python=sys.version, tensorflow=tf.__version__, keras=keras.__version__, opencv=cv2.__version__, physical_devices=[str(x) for x in tf.config.list_physical_devices()], hardware=dict(cpu_count=psutil.cpu_count(), ram_bytes=psutil.virtual_memory().total, platform=platform.platform()), note='Small engineering smoke only. No head training or performance comparison. Peak RSS includes interpreter and model initialization.'))
    return 0
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['audit', 'smoke'])
    a = p.parse_args()
    try:
        sys.exit(audit() if a.command == 'audit' else smoke())
    except Exception as e:
        write_json(HERE / 'outputs' / f'{a.command}_failure.json', dict(status='failed', error=repr(e)))
        raise
