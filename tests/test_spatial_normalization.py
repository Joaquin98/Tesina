import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[1]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
import cv2
import numpy as np
from models.common.spatial import SPATIAL_PREPROCESSING_VERSION, extract_training_equivalent_source_roi, normalize_training_frame, normalize_yolo_source_roi, training_normalization_geometry

class SpatialNormalizationTests(unittest.TestCase):

    def test_training_resolution_reproduces_historical_geometry(self):
        geometry = training_normalization_geometry(1280, 720)
        self.assertEqual(geometry['version'], SPATIAL_PREPROCESSING_VERSION)
        self.assertEqual((geometry['resized_width'], geometry['resized_height']), (640, 360))
        self.assertEqual((geometry['crop_x'], geometry['crop_y']), (208, 68))
        self.assertAlmostEqual(geometry['source_roi_width'], 448.0)
        self.assertAlmostEqual(geometry['source_roi_height'], 448.0)

    def test_1080p_uses_same_relative_field_of_view_as_720p(self):
        geometry_720 = training_normalization_geometry(1280, 720)
        geometry_1080 = training_normalization_geometry(1920, 1080)
        self.assertEqual((geometry_1080['resized_width'], geometry_1080['resized_height']), (640, 360))
        for key, dimension_720, dimension_1080 in (('source_roi_x', 1280, 1920), ('source_roi_y', 720, 1080), ('source_roi_width', 1280, 1920), ('source_roi_height', 720, 1080)):
            self.assertAlmostEqual(geometry_720[key] / dimension_720, geometry_1080[key] / dimension_1080)
        self.assertAlmostEqual(geometry_1080['source_roi_width'], 672.0)

    def test_common_resolutions_preserve_aspect_ratio_and_allow_crop(self):
        for width, height in ((640, 360), (640, 480), (720, 480), (854, 480), (1440, 1080), (3840, 2160), (320, 180), (1080, 1920), (100, 100)):
            with self.subTest(width=width, height=height):
                geometry = training_normalization_geometry(width, height)
                self.assertGreaterEqual(geometry['resized_width'], 224)
                self.assertGreaterEqual(geometry['resized_height'], 224)
                self.assertAlmostEqual(geometry['resized_width'] / geometry['resized_height'], width / height, delta=0.01)

    def test_frame_output_is_always_model_size(self):
        for width, height in ((1280, 720), (1920, 1080), (640, 360), (320, 180), (224, 224), (160, 90), (1080, 1920)):
            with self.subTest(width=width, height=height):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                output = normalize_training_frame(frame)
                self.assertEqual(output.shape, (224, 224, 3))

    def test_small_source_is_upscaled_for_yolo(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        output = normalize_yolo_source_roi(frame)
        self.assertEqual(output.shape, (640, 640, 3))

    def test_yolo_roi_preserves_source_detail_for_720p(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[136:584, 416:864] = 255
        roi = extract_training_equivalent_source_roi(frame)
        self.assertEqual(roi.shape, (448, 448, 3))
        self.assertTrue(np.all(roi == 255))
        yolo = normalize_yolo_source_roi(frame)
        self.assertEqual(yolo.shape, (640, 640, 3))
        self.assertTrue(np.all(yolo == 255))

    def test_source_roi_reduced_to_224_matches_historical_720p_crop(self):
        rng = np.random.default_rng(20260813)
        frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
        historical = normalize_training_frame(frame)
        source_roi = extract_training_equivalent_source_roi(frame)
        reconstructed = cv2.resize(source_roi, (224, 224), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(reconstructed, historical)

    def test_yolo_roi_preserves_equivalent_field_of_view_for_1080p(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        roi = extract_training_equivalent_source_roi(frame)
        self.assertEqual(roi.shape, (672, 672, 3))
        self.assertEqual(normalize_yolo_source_roi(frame).shape, (640, 640, 3))

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            training_normalization_geometry(0, 720)
        with self.assertRaises(ValueError):
            training_normalization_geometry(1280, 0)
if __name__ == '__main__':
    unittest.main()
