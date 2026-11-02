"""Synthetic tests for the high-resolution equivalent field-of-view mapping."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
import numpy as np
from models.M2.extract_cache import equivalent_highres_roi, equivalent_roi_mapping

class EquivalentRoiMappingTest(unittest.TestCase):

    def test_1280x720_maps_to_expected_448_square(self) -> None:
        mapping = equivalent_roi_mapping(1280, 720, pre_crop_scale=0.5, crop_size=224)
        self.assertEqual((mapping.scaled_width, mapping.scaled_height), (640, 360))
        self.assertEqual((mapping.x0, mapping.y0, mapping.x1, mapping.y1), (416, 136, 864, 584))
        self.assertEqual((mapping.roi_width, mapping.roi_height), (448, 448))

    def test_roi_uses_the_same_bounds(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[136:584, 416:864] = 255
        roi, mapping = equivalent_highres_roi(frame)
        self.assertEqual(roi.shape, (448, 448, 3))
        self.assertTrue(np.all(roi == 255))
        self.assertEqual((mapping.x0, mapping.y0), (416, 136))

    def test_supports_small_sources_by_preserving_the_same_normalized_fov(self) -> None:
        mapping = equivalent_roi_mapping(320, 320, pre_crop_scale=0.5, crop_size=224)
        self.assertGreater(mapping.roi_width, 0)
        self.assertGreater(mapping.roi_height, 0)

    def test_rejects_invalid_source_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            equivalent_roi_mapping(0, 320, pre_crop_scale=0.5, crop_size=224)
if __name__ == '__main__':
    unittest.main()
