import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
import numpy as np
from models.M1.studies.scrum_hard_negatives_20261027.weights import hard_negative_weights

class WeightTests(unittest.TestCase):

    def setUp(self):
        self.y = np.array([0, 1, 1, 1, 2, 2])
        self.p = np.array([2, 2, 0, 1, 1, 2])
        self.c = np.array([3.0, 0.5, 4.0])

    def test_exact_selection_and_effective_ratios(self):
        weights, hard = hard_negative_weights(self.y, self.p, self.c, 2.0)
        np.testing.assert_array_equal(hard, [False, True, False, False, False, False])
        self.assertAlmostEqual(weights.mean(), 1.0)
        self.assertAlmostEqual(weights[1] / weights[2], 2.0)
        self.assertAlmostEqual(weights[0] / weights[2], 6.0)
        self.assertTrue((weights > 0).all())

    def test_control_is_original_class_weighting(self):
        weights, hard = hard_negative_weights(self.y, self.p, self.c, 1.0)
        np.testing.assert_allclose(weights, self.c[self.y] / self.c[self.y].mean())

    def test_no_negatives_keeps_original_ratios(self):
        a, _ = hard_negative_weights(self.y, np.ones(6, int), self.c, 2.0)
        b, _ = hard_negative_weights(self.y, self.p, self.c, 1.0)
        np.testing.assert_array_equal(a, b)

    def test_bad_inputs_rejected(self):
        for y, p, c, f in [(self.y, self.p[:-1], self.c, 2.0), (self.y, self.p, self.c, 0.9), (self.y, self.p, self.c, float('nan')), (self.y, self.p, [1, 0, 1], 2.0), (self.y, np.full(6, 9), self.c, 2.0)]:
            with self.assertRaises(ValueError):
                hard_negative_weights(y, p, c, f)
if __name__ == '__main__':
    unittest.main()
