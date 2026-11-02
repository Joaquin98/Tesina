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

def votes(classes):
    p = np.full((1, 9, 3), 0.05)
    for i, c in enumerate(classes):
        p[0, i, c] = 0.9
    return p

class TestRule(unittest.TestCase):

    def test_other_only(self):
        self.assertEqual(decide(votes([1] * 9))[0][0], 1)

    def test_single_event_beats_eight_other(self):
        self.assertEqual(decide(votes([0] + [1] * 8))[0][0], 0)

    def test_scrum_more(self):
        self.assertEqual(decide(votes([2] * 3 + [0] * 2 + [1] * 4))[0][0], 2)

    def test_line_more(self):
        self.assertEqual(decide(votes([2] + [0] * 3 + [1] * 5))[0][0], 0)

    def test_tie_scrum(self):
        self.assertEqual(decide(votes([2] * 2 + [0] * 2 + [1] * 5))[0][0], 2)

    def test_confidence_does_not_override_votes(self):
        p = votes([2] + [0] * 2 + [1] * 6)
        p[0, 0] = [0.0005, 0.0005, 0.999]
        self.assertEqual(decide(p)[0][0], 0)
if __name__ == '__main__':
    unittest.main()
