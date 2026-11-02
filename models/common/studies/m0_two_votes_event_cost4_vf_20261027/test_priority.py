import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest, numpy as np
from priority import decide

def mk(a):
    p = np.full((1, 9, 3), 0.05)
    for i, c in enumerate(a):
        p[0, i, c] = 0.9
    return p

class Tests(unittest.TestCase):

    def test_one_each_is_other(self):
        self.assertEqual(decide(mk([0, 2] + [1] * 7))[0][0], 1)

    def test_one_line_is_other(self):
        self.assertEqual(decide(mk([0] + [1] * 8))[0][0], 1)

    def test_two_line_beats_one_scrum(self):
        self.assertEqual(decide(mk([0, 0, 2] + [1] * 6))[0][0], 0)

    def test_two_scrum_beats_one_line(self):
        self.assertEqual(decide(mk([2, 2, 0] + [1] * 6))[0][0], 2)

    def test_tie_scrum(self):
        self.assertEqual(decide(mk([2, 2, 0, 0] + [1] * 5))[0][0], 2)

    def test_more_line(self):
        self.assertEqual(decide(mk([2, 2, 0, 0, 0] + [1] * 4))[0][0], 0)
if __name__ == '__main__':
    unittest.main()
