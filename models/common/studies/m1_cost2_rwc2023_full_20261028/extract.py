"""Complete native features, copying all existing stride25 anchors unchanged."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import os, sys, json, time, ast, hashlib
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for k, v in dict(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1', PYTHONDONTWRITEBYTECODE='1', YOLO_CONFIG_DIR=str(HERE / 'runtime/ultralytics')).items():
    os.environ[k] = v
OLD = ROOT / 'data/professional_rugby_rwc2023_every25_20261025'
N = 144422
BLOCK = 2048

def read(p):
    return json.loads(_resolve_model_artifact(Path(p)).read_text(encoding='utf-8-sig'))

def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(v, indent=2), encoding='utf-8')
    t.replace(p)

def sha(p):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(p)).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def anchors():
    import numpy as np
    chunks = []
    for p in sorted(_model_artifact_glob(OLD / 'results/features', '*.npz', recursive=False)):
        with _load_model_result_array(p) as d:
            chunks.append({k: d[k].copy() for k in ('indices', 'rgb', 'positions', 'poses', 'mask')})
    s = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    assert np.array_equal(s['indices'], np.unique(np.r_[np.arange(0, N, 25), N - 1]))
    return s

def worker(wid, workers):
    import numpy as np, cv2, tensorflow as tf, torch, psutil
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from ultralytics import YOLO
    from models.common.spatial import normalize_training_frame, normalize_yolo_source_roi
    from models.M1.geometry_features import build_geometry_features
    tree = ast.parse(_resolve_model_artifact(_resolve_model_artifact(ROOT / 'models/M1/pose_cache.py')).read_text(encoding='utf-8-sig'))
    node = next((x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'detect_frame'))
    ns = dict(np=np)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen detect_frame>', 'exec'), ns)
    detect = ns['detect_frame']
    config = read(OLD / 'inputs/config.json')
    source = Path(config['video'])
    s = anchors()
    refs = [ROOT / 'models/M0/checkpoints/feature_extractor_resnet50.keras', ROOT / 'yolo11n-pose.pt']
    for p in refs:
        assert sha(p) == config['model_hashes'][str(p)]
    extractor = tf.keras.models.load_model(_resolve_model_artifact(refs[0]), compile=False)
    graph = tf.function(lambda x: extractor(x, training=False), input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.uint8)], autograph=False)
    yolo = YOLO(_resolve_model_artifact(str(refs[1])))
    cap = cv2.VideoCapture(str(source))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N

    def compute(frames):
        t = time.perf_counter()
        rgb = np.concatenate([graph(np.asarray([normalize_training_frame(f)[:, :, [2, 1, 0]] for f in frames[j:j + 8]], np.uint8)).numpy() for j in range(0, len(frames), 8)])
        rgb_s = time.perf_counter() - t
        t = time.perf_counter()
        obs = [detect(yolo, normalize_yolo_source_roi(f, yolo_size=640), 640, 0.25, 30) for f in frames]
        yolo_s = time.perf_counter() - t
        t = time.perf_counter()
        p, q, m = build_geometry_features(*[np.asarray([o[j] for o in obs]) for j in range(5)], person_threshold=0.25, keypoint_threshold=0.2)
        return ((rgb, p, q, m), dict(rgb_s=rgb_s, yolo_s=yolo_s, geometry_s=time.perf_counter() - t))
    control_indices = [750, 1600]
    frames = []
    for i in control_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        assert ok
        frames.append(f)
    actual, _ = compute(frames)
    where = np.searchsorted(s['indices'], control_indices)
    errors = {}
    for k, v in zip(('rgb', 'positions', 'poses', 'mask'), actual):
        want = s[k][where]
        errors[k] = float(np.max(np.abs(v.astype(float) - want.astype(float))))
        assert np.allclose(v, want, atol=0.01 if k == 'rgb' else 0.0001, rtol=1e-05), (k, errors[k])
    save(HERE / f'control_worker{wid}.json', dict(status='PASS', indices=control_indices, max_errors=errors))
    started = time.perf_counter()
    completed = 0
    last = 0
    for block, a in enumerate(range(0, N, BLOCK)):
        if block % workers != wid:
            continue
        z = min(a + BLOCK, N)
        dest = HERE / f'features/{a:08d}_{z:08d}.npz'
        if _resolve_model_artifact(dest).exists():
            with _load_model_result_array(dest) as d:
                assert int(d['a']) == a and int(d['z']) == z and (d['rgb'].shape == (z - a, 2048))
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        assert abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - a) < 0.5
        vals = dict(rgb=np.empty((z - a, 2048), np.float32), positions=np.empty((z - a, 31), np.float32), poses=np.empty((z - a, 30), np.float32), mask=np.empty(z - a, bool))
        tick = time.perf_counter()
        stages = dict(rgb_s=0.0, yolo_s=0.0, geometry_s=0.0, decode_s=0.0)
        pending = []
        native = []
        reused = 0

        def flush():
            nonlocal completed
            if not pending:
                return
            data, timing = compute(pending)
            for k, v in zip(vals, data):
                vals[k][np.asarray(native) - a] = v
            for k, v in timing.items():
                stages[k] += v
            completed += len(pending)
            pending.clear()
            native.clear()
        for i in range(a, z):
            t = time.perf_counter()
            ok, f = cap.read()
            assert ok, (i, a, z)
            stages['decode_s'] += time.perf_counter() - t
            at = np.searchsorted(s['indices'], i)
            if at < len(s['indices']) and int(s['indices'][at]) == i:
                for k in vals:
                    vals[k][i - a] = s[k][at]
                reused += 1
            else:
                pending.append(f)
                native.append(i)
            if len(pending) == 16 or i == z - 1:
                flush()
            if time.perf_counter() - last > 25:
                save(HERE / f'progress_worker{wid}.json', dict(stage='extracting', worker=wid, pid=os.getpid(), frame=i, block_start=a, new_observations=completed, elapsed_s=time.perf_counter() - started, updated_unix=time.time(), available_ram_gb=psutil.virtual_memory().available / 1000000000.0))
                last = time.perf_counter()
        assert all((np.isfinite(v).all() for v in vals.values()))
        timing = dict(a=a, z=z, elapsed_s=time.perf_counter() - tick, reused_anchors=reused, new_observations=z - a - reused, **stages)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with _resolve_model_artifact(dest.with_suffix('.tmp')).open('wb') as f:
            np.savez(f, a=a, z=z, **vals)
        dest.with_suffix('.tmp').replace(dest)
        save(dest.with_suffix('.json'), timing)
        print(json.dumps(dict(worker=wid, checkpoint=z, **timing)), flush=True)
    cap.release()
    save(HERE / f'progress_worker{wid}.json', dict(stage='complete', worker=wid, new_observations=completed, elapsed_s=time.perf_counter() - started, updated_unix=time.time()))
if __name__ == '__main__':
    try:
        worker(int(sys.argv[1]), int(sys.argv[2]))
    except Exception:
        import traceback
        save(HERE / f'error_worker{sys.argv[1]}.json', dict(error=traceback.format_exc()))
        raise
