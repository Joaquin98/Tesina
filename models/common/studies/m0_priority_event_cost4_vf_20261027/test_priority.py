import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
import numpy as np
from priority import decide

class TestPriority(unittest.TestCase):

    def test_other_all(self):
        p = np.tile([0.1, 0.8, 0.1], (1, 9, 1))
        y, v = decide(p)
        self.assertEqual((y[0], v[0]), (1, -1))

    def test_event_beats_other_and_highest_event_wins(self):
        p = np.tile([0.01, 0.98, 0.01], (1, 9, 1))
        p[0, 2] = [0.7, 0.2, 0.1]
        p[0, 6] = [0.1, 0.1, 0.8]
        y, v = decide(p)
        self.assertEqual((y[0], v[0]), (2, 6))

    def test_only_argmax_views_eligible(self):
        p = np.tile([0.1, 0.8, 0.1], (1, 9, 1))
        p[0, 3] = [0.45, 0.5, 0.05]
        p[0, 7] = [0.4, 0.35, 0.25]
        y, v = decide(p)
        self.assertEqual((y[0], v[0]), (0, 7))

    def test_ties_first_view(self):
        p = np.tile([0.1, 0.8, 0.1], (1, 9, 1))
        p[0, 1] = [0.7, 0.2, 0.1]
        p[0, 4] = [0.1, 0.2, 0.7]
        y, v = decide(p)
        self.assertEqual((y[0], v[0]), (0, 1))
if __name__ == '__main__':
    unittest.main()
