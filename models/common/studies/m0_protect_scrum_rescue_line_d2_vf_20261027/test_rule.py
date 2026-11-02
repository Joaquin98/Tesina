import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest, numpy as np
from rule import decide

def make(base, contexts):
    a = np.full((1, 9, 3), 0.05)
    a[0, 0, base] = 0.9
    for j, c in enumerate(contexts, 1):
        a[0, j, c] = 0.9
    for j in range(len(contexts) + 1, 9):
        a[0, j, 1] = 0.9
    return a

class Tests(unittest.TestCase):

    def test_other_rescued(self):
        self.assertEqual(decide(make(1, [0, 0]), 0.8)[0], 0)

    def test_one_not_enough(self):
        self.assertEqual(decide(make(1, [0]), 0.8)[0], 1)

    def test_central_scrum_protected(self):
        self.assertEqual(decide(make(2, [0, 0, 0]), 0.8)[0], 2)

    def test_central_line_preserved(self):
        self.assertEqual(decide(make(0, [2, 2]), 0.8)[0], 0)

    def test_confidence_required(self):
        self.assertEqual(decide(make(1, [0, 0]), 0.95)[0], 1)
if __name__ == '__main__':
    unittest.main()
