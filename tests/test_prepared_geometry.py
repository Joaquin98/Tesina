from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.common.spatial import (
    extract_training_equivalent_source_roi,
    normalize_training_frame,
    normalize_yolo_source_roi,
    training_normalization_geometry,
)

def prepared_frame():
    return np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)

def test_prepared_geometry_uses_the_entire_frame():
    geometry = training_normalization_geometry(224, 224)
    assert geometry['scale'] == 1
    assert (geometry['crop_x'], geometry['crop_y']) == (0, 0)
    assert (geometry['source_roi_width'], geometry['source_roi_height']) == (224, 224)

def test_prepared_rgb_pixels_are_unchanged():
    frame = prepared_frame()
    np.testing.assert_array_equal(normalize_training_frame(frame), frame)

def test_prepared_pose_uses_all_pixels_before_yolo_resize():
    frame = prepared_frame()
    np.testing.assert_array_equal(extract_training_equivalent_source_roi(frame), frame)
    np.testing.assert_array_equal(normalize_yolo_source_roi(frame), cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR))
