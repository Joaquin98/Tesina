"""Tests for sparse-frame evaluation sampling policies."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
import numpy as np
from models.M0.evaluate_best_model_sparse_frames import select_indices

class SparseFrameEvaluationTest(unittest.TestCase):

    def test_uniform_uses_centered_bins_not_edges(self):
        valid = np.arange(10)
        np.testing.assert_array_equal(select_indices(valid, 1, 'uniform'), np.array([5]))
        np.testing.assert_array_equal(select_indices(valid, 2, 'uniform'), np.array([2, 7]))
        np.testing.assert_array_equal(select_indices(valid, 4, 'uniform'), np.array([1, 3, 6, 8]))

    def test_uniform_edges_preserves_previous_endpoint_behavior(self):
        valid = np.arange(10)
        np.testing.assert_array_equal(select_indices(valid, 1, 'uniform_edges'), np.array([0]))
        np.testing.assert_array_equal(select_indices(valid, 2, 'uniform_edges'), np.array([0, 9]))
if __name__ == '__main__':
    unittest.main()
