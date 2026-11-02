import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
import numpy as np
from models.M1.studies.m1_line2_scrum1_other_scrum2_20261027.weights import effective_weights

class Tests(unittest.TestCase):

    def test_priorities_and_selected_other(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        w = effective_weights(y, np.array([2]))
        self.assertAlmostEqual(w.mean(), 1.0)
        self.assertAlmostEqual(w[0] / w[4], 2.0)
        self.assertAlmostEqual(w[2] / w[3], 2.0)
        self.assertAlmostEqual(w[3] / w[4], 1.0)
        self.assertAlmostEqual(w[0] / w[2], 1.0)

    def test_balancing_uses_original_counts(self):
        y = np.array([0, 1, 1, 2, 2, 2])
        w = effective_weights(y, np.array([], int))
        self.assertAlmostEqual(w[y == 0].sum() / w[y == 2].sum(), 2.0)
        self.assertAlmostEqual(w[y == 1].sum() / w[y == 2].sum(), 1.0)

    def test_invalid_inputs(self):
        for y, idx in [(np.array([0, 1, 2]), np.array([0])), (np.array([0, 1, 2]), np.array([3])), (np.array([0, 1, 2]), np.array([1, 1])), (np.array([0, 1]), np.array([1]))]:
            with self.assertRaises(ValueError):
                effective_weights(y, idx)
if __name__ == '__main__':
    unittest.main()
