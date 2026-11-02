import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAMILIES = [('models.M0.web_adapter', 'RugbyPlayDetector'), ('models.M1.inference', 'M1H2D2CrossFoldDetector'), ('models.M2.predictor', 'RGBYoloHighresQuantileFusionDetector'), ('models.M3.predictor', 'M3CrossFoldDetector')]

@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'clip.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (224, 224))
    assert writer.isOpened()
    for n in range(10):
        writer.write(np.full((224, 224, 3), 20 + n, np.uint8))
    writer.release()
    return path

def detector(family, stats, revision='rgb1'):
    cls = getattr(importlib.import_module(family[0]), family[1])
    model = object.__new__(cls)
    model.preprocessing_filter = 'none'
    model.extractor_name = 'resnet50'
    model.feature_extractor = SimpleNamespace(output_shape=(None, 2))
    model._feature_cache_rgb_revision = hashlib.sha256(revision.encode()).hexdigest()
    model._feature_cache_pose_revision = hashlib.sha256(b'pose1').hexdigest() if family != FAMILIES[0] else None
    model.label_vocab = np.asarray(['line', 'other', 'scrum'])
    model.pose_input_mode = 'source_equivalent_roi'
    model._geometry_positions = np.zeros((0, 31), np.float32)
    model._geometry_poses = np.zeros((0, 30), np.float32)
    model._geometry_mask = np.zeros(0, bool)
    if family != FAMILIES[0]:
        model.pose_detector = object()
    def features(frames, initial_batch_size):
        stats['rgb'] += len(frames)
        values = np.asarray(frames, dtype=np.float32).mean(axis=(1, 2, 3))
        return np.stack([values, values + 1], axis=1)
    def pose(*args, **kwargs):
        stats['pose'] += 1
        return (np.zeros((30, 4), np.float32), np.ones(30, np.float32), np.zeros((30, 17, 2), np.float32), np.ones((30, 17), np.float32), np.ones(30, bool))
    def geometry(boxes, *args, **kwargs):
        n = len(boxes)
        return (np.tile(np.arange(31, dtype=np.float32), (n, 1)), np.tile(np.arange(30, dtype=np.float32), (n, 1)), np.ones(n, bool))
    def predict(features, mask, start, end):
        stats['heads'] += 1
        if family == FAMILIES[1]:
            assert len(model._geometry_positions) == 10
            assert model._geometry_mask.all()
        if family in FAMILIES[2:]:
            assert model._raw_geometry is not None
            assert len(model._raw_geometry['boxes']) == len(model._yolo_source_indices)
        return np.asarray([[.2, .5, .3]], np.float32)
    model._predict_feature_batch = features
    model._detect_frame = pose
    model._build_geometry_features = geometry
    model._predict_sequence_window = predict
    model._post_process_event_detections = lambda *args, **kwargs: []
    return model

def analyze(model, video, cache_dir, **kwargs):
    from models.M0.web_adapter import RugbyPlayDetector
    return RugbyPlayDetector.analyze_video_segments(model, str(video), cache_dir=cache_dir, feature_batch_size=4, feature_predict_batch_size=2, **kwargs)

@pytest.mark.parametrize('family', FAMILIES)
def test_cold_and_warm_runs_reuse_rgb_and_pose_across_instances(family, video, tmp_path):
    stats = {'rgb': 0, 'pose': 0, 'heads': 0}
    first = analyze(detector(family, stats), video, tmp_path / 'cache')
    assert first['status'] == 'success' and not first['feature_cache']['hit']
    extracted = dict(stats)
    second = analyze(detector(family, stats), video, tmp_path / 'cache')
    assert second['feature_cache']['hit']
    assert second['window_predictions'] == first['window_predictions']
    assert stats['rgb'] == extracted['rgb'] and stats['pose'] == extracted['pose']
    assert stats['heads'] == extracted['heads'] + 1

def test_changed_weights_invalidate_features(video, tmp_path):
    stats = {'rgb': 0, 'pose': 0, 'heads': 0}
    analyze(detector(FAMILIES[0], stats), video, tmp_path / 'cache')
    before = stats['rgb']
    changed = analyze(detector(FAMILIES[0], stats, revision='rgb2'), video, tmp_path / 'cache')
    assert not changed['feature_cache']['hit'] and stats['rgb'] > before

def test_corrupt_cache_is_recomputed(video, tmp_path):
    stats = {'rgb': 0, 'pose': 0, 'heads': 0}
    first = analyze(detector(FAMILIES[0], stats), video, tmp_path / 'cache')
    Path(first['feature_cache']['path']).write_bytes(b'truncated')
    before = stats['rgb']
    second = analyze(detector(FAMILIES[0], stats), video, tmp_path / 'cache')
    assert not second['feature_cache']['hit'] and stats['rgb'] > before
    assert analyze(detector(FAMILIES[0], stats), video, tmp_path / 'cache')['feature_cache']['hit']

def test_thresholds_are_reapplied_and_cache_can_be_disabled(video, tmp_path):
    stats = {'rgb': 0, 'pose': 0, 'heads': 0}
    analyze(detector(FAMILIES[0], stats), video, tmp_path / 'cache', confidence_threshold=.1)
    assert analyze(detector(FAMILIES[0], stats), video, tmp_path / 'cache', confidence_threshold=.9)['feature_cache']['hit']
    before = stats['rgb']
    fresh = analyze(detector(FAMILIES[0], stats), video, tmp_path / 'cache', use_cache=False)
    assert not fresh['feature_cache']['enabled'] and stats['rgb'] > before

@pytest.mark.parametrize('change', ['video', 'filter', 'sampling', 'pose_weights'])
def test_input_and_configuration_changes_invalidate_cache(change, video, tmp_path):
    stats = {'rgb': 0, 'pose': 0, 'heads': 0}
    family = FAMILIES[1]
    analyze(detector(family, stats), video, tmp_path / 'cache')
    model = detector(family, stats)
    kwargs = {'feature_batch_size': 4}
    if change == 'video':
        # Same pathname and timestamp, different content.
        import os
        timestamp = video.stat().st_mtime_ns
        with video.open('ab') as handle:
            handle.write(b'\x00\x00\x00\x08free')
        os.utime(video, ns=(timestamp, timestamp))
    elif change == 'filter':
        model.preprocessing_filter = 'clahe'
    elif change == 'sampling':
        kwargs['feature_batch_size'] = 5
    else:
        model._feature_cache_pose_revision = 'different_pose_weights'
    from models.M0.web_adapter import RugbyPlayDetector
    result = RugbyPlayDetector.analyze_video_segments(model, str(video), cache_dir=tmp_path / 'cache', feature_predict_batch_size=2, **kwargs)
    assert not result['feature_cache']['hit'] and stats['rgb'] > 10

@pytest.mark.parametrize('damage', ['missing_pose', 'nan', 'wrong_shape'])
def test_invalid_numeric_or_pose_payload_is_recomputed(damage, video, tmp_path):
    stats = {'rgb': 0, 'pose': 0, 'heads': 0}
    family = FAMILIES[2]
    first = analyze(detector(family, stats), video, tmp_path / 'cache')
    path = Path(first['feature_cache']['path'])
    with np.load(path, allow_pickle=False) as stored:
        values = {name: stored[name] for name in stored.files}
    if damage == 'missing_pose':
        del values['raw.boxes']
    elif damage == 'nan':
        values['features'][0, 0] = np.nan
    else:
        values['features'] = values['features'][:, :1]
    np.savez_compressed(path, **values)
    assert not analyze(detector(family, stats), video, tmp_path / 'cache')['feature_cache']['hit']
    assert analyze(detector(family, stats), video, tmp_path / 'cache')['feature_cache']['hit']

def test_unwritable_cache_does_not_break_inference(video, tmp_path):
    cache = tmp_path / 'not_a_directory'
    cache.write_text('occupied')
    stats = {'rgb': 0, 'pose': 0, 'heads': 0}
    result = analyze(detector(FAMILIES[0], stats), video, cache)
    assert result['status'] == 'success'
    assert result['feature_cache']['status'] == 'not_saved'

def test_unknown_loaded_weights_disable_reuse(video, tmp_path):
    stats = {'rgb': 0, 'pose': 0, 'heads': 0}
    model = detector(FAMILIES[1], stats)
    model._feature_cache_pose_revision = None
    result = analyze(model, video, tmp_path / 'cache')
    assert result['status'] == 'success'
    assert result['feature_cache']['status'] == 'unavailable'
    assert not (tmp_path / 'cache').exists()

def test_concurrent_writers_leave_a_complete_cache(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from models.inference_cache import write_features, read_features
    path = tmp_path / 'cache.npz'
    values = {'features': np.ones((10, 2), np.float32)}
    with ThreadPoolExecutor(max_workers=4) as workers:
        # Windows may reject replacement while another process opens the target;
        # the optional write can fail, but must preserve the complete file.
        outcomes = list(workers.map(lambda _: write_features(path, 'key', values), range(12)))
        assert any(outcomes)
    assert np.array_equal(read_features(path, 'key', 10, 2)['features'], values['features'])
    assert not list(tmp_path.glob('*.partial'))

def test_single_clip_api_reuses_rgb_features(video, tmp_path):
    from models.M0.web_adapter import RugbyPlayDetector
    stats = {'rgb': 0, 'pose': 0, 'heads': 0}
    model = detector(FAMILIES[0], stats)
    def extract(frames, **kwargs):
        stats['rgb'] += len(frames)
        return np.ones((len(frames), 2), np.float32)
    model.feature_extractor.predict = extract
    model.sequence_model = SimpleNamespace(predict=lambda *args, **kwargs: np.asarray([[.2, .5, .3]], np.float32))
    first = RugbyPlayDetector.analyze_video(model, str(video), cache_dir=tmp_path / 'cache')
    before = stats['rgb']
    second = RugbyPlayDetector.analyze_video(model, str(video), cache_dir=tmp_path / 'cache')
    assert second['feature_cache']['hit'] and stats['rgb'] == before
    assert second['all_probabilities'] == first['all_probabilities']

def test_default_cache_location_is_excluded_from_git():
    import subprocess
    from models.inference_cache import DEFAULT_CACHE
    assert DEFAULT_CACHE == ROOT / '.runtime/cache/features'
    result = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '--no-index', str(DEFAULT_CACHE / 'test.npz')], capture_output=True)
    assert result.returncode == 0, result.stderr

def test_code_fingerprint_is_stable_between_python_processes():
    import os
    import subprocess
    code = "from models.inference_cache import _code_record; import json; print(json.dumps(_code_record(compile(\"x in {'none', 'clahe', 'rgb', 'pose'}\", '<pipeline>', 'eval'))))"
    results = [subprocess.check_output([sys.executable, '-B', '-c', code], cwd=ROOT, env={**os.environ, 'PYTHONHASHSEED': seed}) for seed in ('1', '2')]
    assert results[0] == results[1]
