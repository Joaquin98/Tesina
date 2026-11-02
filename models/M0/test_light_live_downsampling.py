"""Tests for the light-live sparse temporal view."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
import numpy as np
from models.M0.config import MAX_SEQ_LENGTH, NUM_FEATURES
from models.M0.train_light_live import build_sparse_temporal_view

class LightLiveDownsamplingTest(unittest.TestCase):

    def test_sparse_view_selects_evenly_spaced_valid_frames_and_pads_after(self):
        features = np.zeros((MAX_SEQ_LENGTH, NUM_FEATURES), dtype=np.float32)
        mask = np.zeros((MAX_SEQ_LENGTH,), dtype=bool)
        mask[:175] = True
        for index in range(MAX_SEQ_LENGTH):
            features[index, 0] = index
        sparse_features, sparse_mask, selected_indices = build_sparse_temporal_view(features, mask, sampled_frame_count=50)
        self.assertEqual(int(sparse_mask.sum()), 50)
        self.assertTrue(np.all(sparse_mask[:50]))
        self.assertFalse(np.any(sparse_mask[50:]))
        self.assertEqual(sparse_features[0, 0], 0)
        self.assertEqual(sparse_features[49, 0], 174)
        self.assertTrue(np.all(np.diff(selected_indices) >= 3))
        self.assertTrue(np.all(sparse_features[50:] == 0))
if __name__ == '__main__':
    unittest.main()
