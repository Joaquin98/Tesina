"""Compact M1 inference: central RGB, pose and fusion; no persisted features."""
from pathlib import Path
import json
import numpy as np
from .decode import decode_result
from .geometry_features import build_geometry_features

ROOT = Path(__file__).resolve().parents[2]


def weights():
    published = ROOT/'models/M1/checkpoints'
    if (published/'selection.json').is_file():
        return published
    for name in ['m1_h2_d2_corrected_argmax_20261024', 'm1_h2_d2_corrected_all_vf_20261024']:
        path = ROOT/'data'/name
        record = path/'verification.json'
        if record.exists() and json.loads(record.read_text(encoding='utf-8')).get('passed'):
            return path
    raise FileNotFoundError('No hay un checkpoint M1 verificado')


def align(values):
    a = np.linspace(0, 1, len(values), dtype=np.float32)
    b = np.linspace(0, 1, 200, dtype=np.float32)
    return np.stack([np.interp(b, a, values[:, j]) for j in range(values.shape[1])], axis=1).astype(np.float32)


def inputs(raw):
    sequence = np.concatenate([raw['position_features'], raw['pose_features']], axis=1).astype(np.float32)
    mask = raw['frame_mask'].astype(bool)
    valid = sequence[mask] if mask.any() else sequence
    delta = np.diff(valid, axis=0) if len(valid)>1 else np.zeros((1, 61), np.float32)
    geometry = np.concatenate([valid.mean(0), valid.std(0), valid.min(0), valid.max(0),
                               np.quantile(valid, .25, axis=0), np.quantile(valid, .75, axis=0),
                               delta.mean(0), delta.std(0), np.abs(delta).mean(0)]).astype(np.float32)
    pose = np.concatenate([align(sequence), align(mask.astype(np.float32)[:, None])], axis=1)
    return pose, geometry


class Head:
    def __init__(self, load_pose=True):
        import keras
        import joblib
        base = weights()
        self.base = base
        self.dual = keras.models.load_model(base/'dual.keras', compile=False)
        self.forest = joblib.load(base/'geometry_rf.joblib')
        self.forest.n_jobs = 1
        assert np.array_equal(self.forest.classes_, np.arange(3))
        self.selection = json.loads((base/'selection.json').read_text(encoding='utf-8'))
        self.alpha = float(self.selection['geometry_alpha'])
        self.detector = None
        if load_pose:
            checkpoint = ROOT/'yolo11n-pose.pt'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            import contextlib
            import os
            import sys
            import tempfile
            self._temporary_settings = tempfile.TemporaryDirectory(prefix='tesina-yolo-')
            os.environ['YOLO_CONFIG_DIR'] = self._temporary_settings.name
            os.environ['YOLO_VERBOSE'] = 'False'
            # Library settings are transient and never go into the repository.
            with contextlib.redirect_stdout(sys.stderr):
                from ultralytics import YOLO
                self.detector = YOLO(str(checkpoint))

    def predict_raw(self, features, mask, raw):
        pose, geometry = inputs(raw)
        q = np.asarray(self.dual({'rgb':features[None], 'pose':pose[None], 'sequence_mask':mask[None]}, training=False))[0].astype(np.float64)
        r = self.forest.predict_proba(geometry[None])[0]
        p = np.exp((1-self.alpha)*np.log(np.clip(q, 1e-7, 1))+self.alpha*np.log(np.clip(r, 1e-7, 1)))
        return p/p.sum()

    def predict(self, features, mask, frames):
        import cv2
        selected = np.linspace(0, len(frames)-1, min(100, len(frames)), dtype=np.int32)
        # This follows the existing web inference ROI, not the unavailable
        # high-resolution pose extraction used in older training models.common.studies.
        images = [cv2.resize(np.ascontiguousarray(frames[i][:,:,::-1]), (640,640), interpolation=cv2.INTER_LINEAR) for i in selected]
        decoded = []
        for start in range(0, len(images), 4):
            batch = images[start:start+4]
            results = self.detector.predict(batch, imgsz=640, conf=.25, device='cpu', verbose=False, batch=4)
            decoded.extend(decode_result(result, image, 30) for result, image in zip(results, batch))
        boxes, confidence, points, point_confidence, valid = [np.stack([row[i] for row in decoded]) for i in range(5)]
        position, pose, valid_frames = build_geometry_features(boxes, confidence, points, point_confidence, valid,
                                                              person_threshold=.25, keypoint_threshold=.20)
        return self.predict_raw(features, mask, dict(position_features=position, pose_features=pose, frame_mask=valid_frames))
