"""Replay only argmax FP anchor windows of the all-development M1 on VF.

Compute branch outputs with the configured video preprocessing,
25 FPS sampling, 100 pose observations, extractor and frozen heads. Validate
the fused result against each original stored probability before attribution.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import os, sys, json, ast, hashlib, time, argparse
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--anchors', type=Path)
parser.add_argument('--output', type=Path, default=HERE)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
for k, v in {'TF_CPP_MIN_LOG_LEVEL': '2', 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '8', 'TF_NUM_INTRAOP_THREADS': '8', 'TF_NUM_INTEROP_THREADS': '1', 'YOLO_CONFIG_DIR': str(HERE / 'runtime/ultralytics')}.items():
    os.environ.setdefault(k, v)
(HERE / 'runtime/ultralytics').mkdir(parents=True, exist_ok=True)
CACHE = HERE / 'anchor_replay'
CACHE.mkdir(exist_ok=True)
import numpy as np, pandas as pd, cv2, joblib
if args.anchors:
    anchors = pd.read_csv(_resolve_model_artifact(args.anchors)).sort_values(['section', 'block_index'])
else:
    fp = pd.read_csv(_resolve_model_artifact(HERE / 'fp_inventory.csv'))
    wanted = fp[fp.policy.eq('argmax')].anchor_block_id
    anchors = pd.read_csv(_resolve_model_artifact(HERE / 'anchors.csv'))
    anchors = anchors[anchors.block_id.isin(wanted)].sort_values(['section', 'block_index'])
    assert len(anchors) == 11
assert anchors.block_id.is_unique
expected = len(anchors)
print(f'Loading frozen inference models for {expected} diagnostic anchors', flush=True)
import tensorflow as tf, keras, torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)
from ultralytics import YOLO
from models.M1.geometry_features import build_geometry_features
from models.common.spatial import normalize_training_frame

def load_functions(path, names, env):
    tree = ast.parse(_resolve_model_artifact(path).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env
env = load_functions(_resolve_model_artifact(ROOT / 'models/M1/pose_cache.py'), ['detect_frame'], dict(np=np))
detect_frame = env['detect_frame']
env = load_functions(_resolve_model_artifact(ROOT / 'models/M0/data_utils.py'), ['_sample_frame_indices'], dict(np=np))
sample_indices = env['_sample_frame_indices']
modeldir = ROOT / 'data/m1_ensemble_vs_all_development_20261012/models'
paths = [modeldir / 'dual.keras', modeldir / 'geometry_rf.joblib', ROOT / 'models/M0/checkpoints/feature_extractor_resnet50.keras', ROOT / 'yolo11n-pose.pt']
sha = lambda p: hashlib.sha256(_resolve_model_artifact(p).read_bytes()).hexdigest()
hashes = {str(p): sha(p) for p in paths}
training = json.loads(_resolve_model_artifact(modeldir.parent / 'results/training.json').read_text())
for p in paths[:2]:
    assert hashes[str(p)] == training['new_weight_hashes'][str(p)]
dual = keras.models.load_model(_resolve_model_artifact(paths[0]), compile=False)
forest = joblib.load(_resolve_model_artifact(paths[1]))
forest.n_jobs = 2
extractor = keras.models.load_model(_resolve_model_artifact(paths[2]), compile=False)
pose = YOLO(_resolve_model_artifact(str(paths[3])))
extract = tf.function(lambda x: extractor(x, training=False), input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.uint8)], autograph=False)
predict = tf.function(lambda x: dual(x, training=False), input_signature=[{'rgb': tf.TensorSpec([None, 200, 2048], tf.float32), 'pose': tf.TensorSpec([None, 200, 62], tf.float32), 'sequence_mask': tf.TensorSpec([None, 200], tf.bool)}], autograph=False)
alpha = 0.4750000000000001

def align(x):
    old = np.linspace(0, 1, len(x), dtype=np.float32)
    new = np.linspace(0, 1, 200, dtype=np.float32)
    return np.stack([np.interp(new, old, x[:, j]) for j in range(x.shape[1])], axis=1).astype(np.float32)

def normalize(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-07, None)
    return p / p.sum()
records = []
for n, row in enumerate(anchors.itertuples(), 1):
    cache = CACHE / (row.block_id + '.json')
    if _resolve_model_artifact(cache).exists():
        records.append(json.loads(_resolve_model_artifact(cache).read_text()))
        continue
    began = time.perf_counter()
    videos = list(_model_artifact_glob(ROOT / 'data/videos/partidos_evaluacion', f'VF_{row.section}_*.mp4', recursive=False))
    assert len(videos) == 1
    cap = cv2.VideoCapture(str(videos[0]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = np.unique(sample_indices(total, fps, 25))
    first = int(round(row.start_s * 25))
    last = min(len(samples), int(round(row.end_s * 25)))
    indices = samples[first:last]
    selected = np.linspace(0, len(indices) - 1, 100, dtype=np.int32) if len(indices) > 100 else np.arange(len(indices))
    selected_set = set(map(int, selected))
    obs = []
    rgb = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
    native = int(indices[0])
    targets = {int(v): j for j, v in enumerate(indices)}
    while native <= int(indices[-1]):
        ok, raw = cap.read()
        assert ok
        if native in targets:
            crop = normalize_training_frame(raw, target_size=224)
            h, w = raw.shape[:2]
            half = cv2.resize(raw, (round(w * 0.5), round(h * 0.5)), interpolation=cv2.INTER_LINEAR)
            yy = (half.shape[0] - 224) // 2
            xx = (half.shape[1] - 224) // 2
            assert np.array_equal(crop, half[yy:yy + 224, xx:xx + 224])
            rgb.append(crop[:, :, [2, 1, 0]])
            if targets[native] in selected_set:
                obs.append(detect_frame(pose, cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR), 640, 0.25, 30))
        native += 1
    cap.release()
    assert len(rgb) == len(indices) and len(obs) == len(selected)
    values = [np.asarray([o[j] for o in obs]) for j in range(5)]
    positions, poses, mask = build_geometry_features(*values, person_threshold=0.25, keypoint_threshold=0.2)
    posevalues = np.concatenate([positions, poses], axis=1)
    poseinput = np.concatenate([align(posevalues), align(mask.astype(np.float32)[:, None])], axis=1)
    valid = posevalues[mask] if mask.any() else posevalues
    if len(valid) == 0:
        valid = np.zeros((1, 61), np.float32)
    delta = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, 61), np.float32)
    vector = np.concatenate([valid.mean(0), valid.std(0), valid.min(0), valid.max(0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), delta.mean(0), delta.std(0), np.abs(delta).mean(0)]).astype(np.float32).reshape(1, -1)
    features = np.concatenate([extract(np.asarray(rgb[j:j + 8], dtype=np.uint8)).numpy() for j in range(0, len(rgb), 8)])
    padded = np.zeros((200, 2048), np.float32)
    padded[:len(features)] = features
    rgbmask = np.arange(200) < len(features)
    dp = predict({'rgb': padded[None], 'pose': poseinput[None], 'sequence_mask': rgbmask[None]}).numpy()[0].astype(np.float64)
    gp = np.zeros(3)
    gp[np.asarray(forest.classes_).astype(int)] = forest.predict_proba(vector)[0]
    fused = normalize(normalize(np.exp((1 - alpha) * np.log(np.clip(dp, 1e-07, 1)) + alpha * np.log(np.clip(gp, 1e-07, 1))))).astype(np.float32)
    old = np.array([row.p_line, row.p_other, row.p_scrum])
    err = float(np.abs(fused - old).max())
    result = dict(block_id=row.block_id, section=row.section, block_index=row.block_index, dual=dp.tolist(), geometry=gp.tolist(), fused_replayed=fused.tolist(), fused_original=old.tolist(), max_absolute_error=err, seconds=time.perf_counter() - began, video=str(videos[0]), alpha=alpha)
    cache.write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(CACHE / (row.block_id + '.npz'), rgb=padded, pose=poseinput, mask=rgbmask, geometry=vector)
    print(f"{n}/{expected} {row.section} block {row.block_index}: error={err:.8g}, {result['seconds']:.1f}s", flush=True)
    assert err < 2e-05, ('Replay does not reproduce saved run', result)
    records.append(result)
assert len(records) == expected and all((r['max_absolute_error'] < 2e-05 for r in records))
assert all((sha(Path(p)) == h for p, h in hashes.items()))
(args.output / 'branch_probabilities.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
(args.output / 'replay_verification.json').write_text(json.dumps(dict(passed=True, anchors=expected, max_error=max((r['max_absolute_error'] for r in records)), model_hashes=hashes, alpha=alpha, policy='both argmax', no_retraining=True, scope='VF S01-S04; diagnostic anchor blocks only', anchor_input=str(args.anchors) if args.anchors else 'original 11 FP anchors'), indent=2), encoding='utf-8')
