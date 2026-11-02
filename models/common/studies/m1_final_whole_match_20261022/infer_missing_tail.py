"""Infer only the missing VF S05 tail with the unchanged all-development M1 single model."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import os, sys, json, hashlib, time
HERE = experiment_directory(__file__)
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for name in ['inputs', 'results', 'logs', 'runtime/tmp', 'runtime/keras', 'runtime/ultralytics', 'runtime/cache', 'runtime/matplotlib']:
    (HERE / name).mkdir(parents=True, exist_ok=True)
for key, value in {'CUDA_VISIBLE_DEVICES': '-1', 'TF_CPP_MIN_LOG_LEVEL': '2', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'NUMEXPR_NUM_THREADS': '4', 'PYTHONDONTWRITEBYTECODE': '1', 'YOLO_CONFIG_DIR': str(HERE / 'runtime/ultralytics'), 'KERAS_HOME': str(HERE / 'runtime/keras'), 'TORCH_HOME': str(HERE / 'runtime/torch'), 'MPLCONFIGDIR': str(HERE / 'runtime/matplotlib'), 'XDG_CACHE_HOME': str(HERE / 'runtime/cache'), 'PLAYER_GEOMETRY_POSE_CACHE_ROOT': str(HERE / 'runtime/pose_cache'), 'TEMP': str(HERE / 'runtime/tmp'), 'TMP': str(HERE / 'runtime/tmp')}.items():
    os.environ[key] = value
os.chdir(HERE)

def audit(event, args):
    targets = []
    if event == 'open':
        p, mode, flags = args
        if isinstance(p, (str, bytes, os.PathLike)) and (mode and any((c in mode for c in 'wax+')) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            targets = [p]
    elif event in ['os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime']:
        targets = [args[0]]
    elif event in ['os.rename', 'os.link', 'os.symlink']:
        targets = [args[0], args[1]]
    for p in targets:
        if not Path(os.fsdecode(p)).resolve().is_relative_to(HERE):
            raise PermissionError(f'Write outside isolated experiment blocked: {event} {p}')
sys.addaudithook(audit)

def save(path, value):
    (HERE / path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with _resolve_model_artifact(Path(path)).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
if _resolve_model_artifact(HERE / 'results/S05_probabilities.csv').exists():
    raise FileExistsError('Tail probabilities already exist; refusing overwrite')
source = ROOT / 'data/m1_ensemble_vs_all_development_20261012'
manifestpath = source / 'inputs/M1-single_manifest.json'
manifest = json.loads(_resolve_model_artifact(manifestpath).read_text(encoding='utf-8'))
video = ROOT / 'data/videos/partidos_evaluacion/VF_S05_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4'
prior = json.loads(_resolve_model_artifact(ROOT / 'data/m1_ensemble_vs_single_vf_s04_20261013/results/M1-single_startup.json').read_text(encoding='utf-8'))
references = [manifestpath, video, _resolve_model_artifact(ROOT / 'src/web/keras_inference_engine.py'), _resolve_model_artifact(ROOT / 'models/M0/web_adapter.py'), _resolve_model_artifact(ROOT / 'models/common/spatial.py'), _resolve_model_artifact(ROOT / 'models/M1/geometry_features.py'), _resolve_model_artifact(ROOT / 'models/M1/pose_cache.py'), ROOT / 'models/M0/checkpoints/feature_extractor_resnet50.keras', ROOT / 'yolo11n-pose.pt']
references += [ROOT / manifest['folds'][0][k] for k in ['dual_model', 'geometry_rf']]
before = {str(p): sha(p) for p in references}
for p, h in before.items():
    if p in prior['hashes_before']:
        assert h == prior['hashes_before'][p], p
save('inputs/source_hashes_before.json', before)
save('inputs/request.json', dict(scope='Whole-match inference: compute S05 and reuse the existing S01-S04 probabilities.', model=manifest['source_configuration'], video=str(video), threads=4, training=False, existing_sections_recomputed=False, manual_labels_available=False, tail_evaluation='Predictions only; ground-truth labels and event metrics are unavailable for this section.'))
sys.path.insert(0, str(ROOT))
started = time.perf_counter()
print('Loading unchanged single M1 and frozen extractors', flush=True)
import cv2, numpy as np, pandas as pd, tensorflow as tf, torch
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
cv2.setNumThreads(1)
from src.web import keras_inference_engine as em
isolated = _resolve_model_artifact(ROOT / 'models/common/studies/m1_ensemble_vs_single_vf_s04_20261013/inputs/M1H2D2CrossFoldDetector_isolated.py')
classcode = _resolve_model_artifact(isolated).read_text(encoding='utf-8')
(_resolve_model_artifact(HERE / 'inputs/isolated_model_class.py')).write_text(classcode, encoding='utf-8')
namespace = dict(vars(em))
namespace.update(SELECTED_MANIFEST_PATH=manifestpath, EXPECTED_MEMBERS=1)
exec(compile(classcode, str(_resolve_model_artifact(HERE / 'inputs/isolated_model_class.py')), 'exec'), namespace)
detector = namespace['M1H2D2CrossFoldDetector'](ROOT)
assert len(detector.fold_models) == 1
for item in detector.fold_models:
    item['geometry'].n_jobs = 1
    model = item['dual']

    class GraphProxy:

        def __init__(self, model):
            self.graph = tf.function(lambda values: model(values, training=False), input_signature=[{'rgb': tf.TensorSpec([None, 200, 2048], tf.float32), 'pose': tf.TensorSpec([None, 200, 62], tf.float32), 'sequence_mask': tf.TensorSpec([None, 200], tf.bool)}], autograph=False)

        def __call__(self, inputs, training=False):
            return self.graph({k: tf.convert_to_tensor(v) for k, v in inputs.items()})
    item['dual'] = GraphProxy(model)
captured = []
predict_features = detector._predict_feature_batch

def capture(frames, initial_batch_size):
    value = predict_features(frames, initial_batch_size)
    captured.append(value)
    return value
detector._predict_feature_batch = capture
engine = em.KerasInferenceEngine()
engine.current_model = detector
engine.current_model_name = 'M1-H2-D2 single · missing VF S05'
engine.current_model_type = 'm1_h2_d2_cross_fold'
last = [0.0]

def progress(percent, stage, message):
    now = time.perf_counter()
    if now - last[0] >= 15:
        row = dict(stage=stage, percent=percent, message=message, elapsed_seconds=now - started)
        save('progress.json', row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        last[0] = now
result = engine.analyze_video(video, confidence_threshold=0.45, line_confidence_threshold=0.5, use_segments=True, segment_duration=7, temporal_fps_mode='training_match_25', feature_batch_size=128, feature_predict_batch_size=8, progress_callback=progress)
assert result['status'] == 'success', result
save('results/S05_streaming_reference.json', result)
features = np.concatenate(captured, axis=0)
duration = float(result['metadata']['duration'])
assert abs(duration - 169.24) < 0.1 and len(features) == len(detector._geometry_positions)
np.savez_compressed(HERE / 'results/S05_frozen_features.npz', rgb=features, positions=detector._geometry_positions, poses=detector._geometry_poses, pose_mask=detector._geometry_mask)
rows = []
for i, begin in enumerate(np.arange(0, duration, 7)):
    end = min(duration, float(begin) + 7)
    a = round(float(begin) * 25)
    z = min(len(features), round(end * 25))
    n = z - a
    assert 0 < n <= 200
    padded = np.zeros((200, 2048), np.float32)
    padded[:n] = features[a:z]
    mask = np.arange(200) < n
    p = detector._predict_sequence_window(padded, mask, a, z)[0]
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1, atol=1e-06)
    rows.append(dict(block_id=f'VF_S05_frozen_single_{i:04d}', group_id='VF', section='S05', block_index=i, start_s=float(begin), end_s=end, duration=end - float(begin), y_true=None, p_line=float(p[0]), p_other=float(p[1]), p_scrum=float(p[2]), y_argmax=['line', 'other', 'scrum'][int(p.argmax())], label_status='unlabeled'))
frame = pd.DataFrame(rows)
assert len(frame) == 25 and np.isclose(frame.duration.sum(), duration)
frame.to_csv(HERE / 'results/S05_probabilities.csv', index=False)
frame[['block_id', 'block_index', 'start_s', 'end_s', 'duration', 'y_true']].to_csv(HERE / 'results/S05_manual_review_template.csv', index=False)
after = {p: sha(p) for p in before}
assert before == after
save('results/completion.json', dict(success=True, source_hashes_unchanged=True, hashes_after=after, blocks=25, duration_seconds=duration, sampled_frames=len(features), elapsed_seconds=time.perf_counter() - started, same_weights_and_implementation_as_S01_S04=True, labels='not available; predictions must remain unverified'))
print('COMPLETE ' + json.dumps(dict(seconds=duration, blocks=len(frame), elapsed=time.perf_counter() - started)), flush=True)
