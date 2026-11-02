import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest, numpy as np
from hard import sample_weights

class TestHard(unittest.TestCase):

    def test_only_line_mistakes_are_weighted(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 2, 0, 1, 0, 2])
        w, hard = sample_weights(y, pred)
        np.testing.assert_array_equal(hard, [False, True, False, False, False, False])
        self.assertAlmostEqual(w[1] / w[0], 3)
        self.assertAlmostEqual(w[2] / w[3], 1)
        self.assertAlmostEqual(w[4] / w[5], 1)
        self.assertAlmostEqual(w.mean(), 1)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            sample_weights(np.array([0, 1, 2]), np.array([0, 1]))
if __name__ == '__main__':
    unittest.main()
