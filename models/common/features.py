"""One frozen RGB/pose feature path for preparation, fitting and prediction."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

import cv2
import numpy as np

from models.inference_cache import cached_features, auxiliary_state, restore_auxiliary, load_with_revision, read_features, file_digest
from models.model_catalog import resolve_artifact, temporary_settings

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('line', 'other', 'scrum')
RGB_WEIGHTS = ROOT / 'models/M0/checkpoints/feature_extractor_resnet50.keras'
HISTORICAL_WEIGHTS = ROOT / 'models/M0/checkpoints/best_model/best_model.keras'
PCA_WEIGHTS = ROOT / 'data/joint_multimodal_20260813/rgb_dynamics_seed42/dynamic_pca.joblib'
POSE_WEIGHTS = ROOT / 'yolo11n-pose.pt'


def feature_source(family, existing=None, references=None):
    if family not in ('M0', 'M1', 'M2', 'M3'):
        raise ValueError(f'Familia desconocida: {family}')
    from models.M0.web_adapter import RugbyPlayDetector
    from models.M1.inference import M1H2D2CrossFoldDetector
    from models.M2.predictor import RGBYoloHighresQuantileFusionDetector
    cls = RugbyPlayDetector if family == 'M0' else M1H2D2CrossFoldDetector if family == 'M1' else RGBYoloHighresQuantileFusionDetector
    source = object.__new__(cls)
    source.preprocessing_filter = 'none'
    source.extractor_name = 'resnet50'
    if existing is None:
        import keras
        rgb_path = ROOT / references['rgb']['path'] if references else resolve_artifact(RGB_WEIGHTS)
        source.feature_extractor, source._feature_cache_rgb_revision = load_with_revision(rgb_path, lambda path: keras.models.load_model(path, compile=False))
    else:
        source.feature_extractor = existing.feature_extractor
        source._feature_cache_rgb_revision = existing._feature_cache_rgb_revision
    if family != 'M0':
        from models.M1.pose_cache import detect_frame
        from models.M1.geometry_features import build_geometry_features
        source._detect_frame = detect_frame
        source._build_geometry_features = build_geometry_features
        if existing is None:
            os.environ['YOLO_CONFIG_DIR'] = str(temporary_settings())
            os.environ['YOLO_VERBOSE'] = 'False'
            from ultralytics import YOLO
            pose_path = ROOT / references['pose']['path'] if references else resolve_artifact(POSE_WEIGHTS)
            source.pose_detector, source._feature_cache_pose_revision = load_with_revision(pose_path, YOLO)
        else:
            source.pose_detector = existing.pose_detector
            source._feature_cache_pose_revision = existing._feature_cache_pose_revision
        if family in ('M2', 'M3'):
            from models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite import make_variant
            source._make_variant = make_variant
            source.pose_input_mode = 'source_equivalent_roi'
    return source


def video_parameters(source, path, feature_batch_size=8, feature_predict_batch_size=2):
    from models.M0.data_utils import _sample_frame_indices
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0 or count <= 0:
        raise ValueError(f'Video sin fotogramas decodificables: {path}')
    effective = source._resolve_temporal_sampling_fps(fps, 7, temporal_fps_mode='training_match_25')
    indices = _sample_frame_indices(count, fps, effective)
    return dict(mode='segments', source_fps=fps, total_frames=count, effective_fps=float(effective), sample_indices=[int(i) for i in indices], feature_batch_size=int(feature_batch_size), feature_predict_batch_size=int(feature_predict_batch_size))


def prepare(source, video, *, cache_dir=None, use_cache=True, parameters=None):
    parameters = parameters or video_parameters(source, video)
    indices = set(parameters['sample_indices'])
    if not indices:
        raise ValueError(f'Video sin muestras: {video}')
    source._web_effective_fps = parameters['effective_fps']
    source._web_window_duration = 7.0
    source._begin_auxiliary_sequence(len(indices))
    compute = lambda: source._extract_sampled_video_features(str(video), indices, parameters['total_frames'], parameters['feature_batch_size'], parameters['feature_predict_batch_size'], None)
    features, info = cached_features(source, video, parameters, len(indices), compute, use_cache=use_cache, cache_dir=cache_dir)
    if len(features) != len(indices):
        raise ValueError(f'El video no contiene todos los fotogramas declarados: {video}')
    return features, auxiliary_state(source), info, parameters


class FeatureProvider:
    def __init__(self, family):
        self.family = family
        self.source = None

    def __call__(self, detector, video, parameters, *, use_cache, cache_dir):
        if self.source is None:
            self.source = feature_source(self.family, existing=detector)
        features, state, info, _ = prepare(self.source, video, parameters=parameters, cache_dir=cache_dir, use_cache=use_cache)
        restore_auxiliary(detector, state, len(features))
        return features, info


@dataclass
class Record:
    video: Path
    tag: str
    path: Path
    key: str
    count: int
    stat: tuple
    sha256: str


def make_record(source, row, cache_dir=None):
    if file_digest(row['path']) != row['sha256']:
        raise ValueError(f"El video cambió desde la validación del dataset: {row['path']}")
    parameters = video_parameters(source, row['path'])
    if parameters['total_frames'] / parameters['source_fps'] > 7.1:
        raise ValueError('El entrenamiento por clips requiere videos de hasta 7 segundos; segmentar los partidos antes de entrenar')
    features, _state, info, parameters = prepare(source, row['path'], cache_dir=cache_dir, parameters=parameters)
    if info['status'] not in ('created', 'hit'):
        raise OSError(f"No se pudo guardar la caché: {info['status']} ({info['path']})")
    path = Path(info['path'])
    stat = row['path'].stat()
    return Record(row['path'], row['tag'], path, path.stem, len(features), (stat.st_size, stat.st_mtime_ns), row['sha256']), info


def load_record(record):
    stat = record.video.stat()
    if (stat.st_size, stat.st_mtime_ns) != record.stat:
        raise ValueError(f'El video cambió durante el entrenamiento: {record.video}')
    values = read_features(record.path, record.key, record.count, 2048)
    if values is None:
        raise ValueError(f'Caché ausente o dañada; ejecutar cache o train de nuevo: {record.path}')
    return values


def rgb_inputs(values):
    source = values['features']
    length = min(len(source), 200)
    features = np.zeros((200, 2048), np.float32)
    features[:length] = source[:length]
    return features, np.arange(200) < length


def m1_inputs(values):
    from models.M1.inference import PlayerGeometryFusionDetector
    from models.common.comparison.data import geometry_statistics
    positions, poses, mask = values['positions'], values['poses'], values['mask']
    if len(positions) > 100:
        indices = np.linspace(0, len(positions)-1, 100, dtype=np.int32)
        positions, poses, mask = positions[indices], poses[indices], mask[indices]
    raw = dict(position_features=positions, pose_features=poses, frame_mask=mask)
    combined = np.concatenate([positions, poses], axis=1)
    pose = np.concatenate([PlayerGeometryFusionDetector._align(combined, 200), PlayerGeometryFusionDetector._align(mask.astype(np.float32)[:, None], 200)], axis=1)
    return pose.astype(np.float32), geometry_statistics(raw)


def raw_pose(values):
    return {key[4:]: value for key, value in values.items() if key.startswith('raw.')}


def static_vector(values, pca):
    from models.M2.predictor import RGBYoloHighresQuantileFusionDetector
    from models.M2.studies.yolo_resnet_improvement_suite_20260811.run_suite import make_variant
    model = object.__new__(RGBYoloHighresQuantileFusionDetector)
    model.dynamic_pca = pca
    model._make_variant = make_variant
    model._raw_geometry = raw_pose(values)
    model._yolo_source_indices = values['yolo_indices']
    features, mask = rgb_inputs(values)
    return np.concatenate([model._rgb_dynamic_vector(features, int(mask.sum())), model._yolo_vector(0, len(values['features']))]).astype(np.float32)


def temporal_inputs(values):
    from models.common.comparison.data import temporal_from_raw
    from models.M2.predictor import RGBYoloTemporalEventVerifierDetector
    return temporal_from_raw(RGBYoloTemporalEventVerifierDetector._resample_raw_window(raw_pose(values)))
