import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
import numpy as np
from fusion import fold_ids, combine, weights

class FusionTest(unittest.TestCase):

    def test_folds_no_leak(self):
        y = np.tile([0, 1, 2], 12)
        groups = np.repeat(np.arange(12), 3)
        fold = fold_ids(y, groups)
        for g in set(groups):
            self.assertEqual(len(set(fold[groups == g])), 1)
        self.assertEqual(set(fold), {0, 1, 2})

    def test_residual_initial_is_central(self):
        p = np.full((2, 9, 3), 1 / 3)
        p[:, 0] = [0.1, 0.7, 0.2]
        logits = combine(p, np.zeros((27, 3)), np.zeros(3))
        np.testing.assert_allclose(np.exp(logits) / np.exp(logits).sum(1, keepdims=True), p[:, 0], atol=1e-07)

    def test_cost4_normalized(self):
        y = np.array([0] * 2 + [1] * 8 + [2] * 4)
        w = weights(y)
        self.assertAlmostEqual(np.mean(w[y]), 1)
        self.assertAlmostEqual(w[0] / w[1] / 4, 8 / 2)
if __name__ == '__main__':
    unittest.main()
