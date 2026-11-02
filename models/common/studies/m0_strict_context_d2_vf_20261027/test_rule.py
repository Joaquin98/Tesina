import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest, numpy as np
from rule import decide, select

def p0():
    return np.tile([0.1, 0.8, 0.1], (1, 9, 1))

class Tests(unittest.TestCase):

    def test_fallback_one_context(self):
        p = p0()
        p[0, 1] = [0.91, 0.04, 0.05]
        self.assertEqual(decide(p, [0.9, 0.9])[0], 1)

    def test_same_class_two_contexts(self):
        p = p0()
        p[0, 1] = [0.91, 0.04, 0.05]
        p[0, 2] = [0.92, 0.04, 0.04]
        self.assertEqual(decide(p, [0.9, 0.9])[0], 0)

    def test_two_different_classes(self):
        p = p0()
        p[0, 1] = [0.91, 0.04, 0.05]
        p[0, 2] = [0.03, 0.03, 0.94]
        self.assertEqual(decide(p, [0.9, 0.9])[0], 1)

    def test_below_confidence(self):
        p = p0()
        p[0, 1] = [0.89, 0.06, 0.05]
        p[0, 2] = [0.89, 0.06, 0.05]
        self.assertEqual(decide(p, [0.9, 0.9])[0], 1)

    def test_no_sufficient_examples_disables(self):
        p = np.tile(p0(), (10, 1, 1))
        y = np.zeros(10, dtype=int)
        self.assertEqual(select(p, y)[0], [None, None])
if __name__ == '__main__':
    unittest.main()
