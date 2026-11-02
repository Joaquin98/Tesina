import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
import numpy as np
from analysis_math import split_indices, metrics, changes, success_rule, temporal_bootstrap

class AnalysisTests(unittest.TestCase):

    def test_split_never_leaks_vf_or_v6(self):
        rows = [dict(split='train', domain='CH', group_id='CH'), dict(split='train', domain='CV', group_id='V_6'), dict(split='train', domain='CV', group_id='V_1'), dict(split='test', domain='VF', group_id='VF')]
        train, val, test = split_indices(rows)
        self.assertEqual(train.tolist(), [0, 2])
        self.assertEqual(val.tolist(), [1])
        self.assertEqual(test.tolist(), [3])

    def test_corrections_and_breakages(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        a = np.array([0, 1, 1, 2, 2, 0])
        b = np.array([1, 0, 1, 1, 2, 2])
        c = changes(y, a, b)
        self.assertEqual(c['corrected'], 3)
        self.assertEqual(c['broken'], 1)
        self.assertEqual(c['by_true_class']['line']['broken'], 1)

    def test_precision_recall_and_evidence_rule(self):
        m = metrics(np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 1, 2, 0]))
        self.assertAlmostEqual(m['line_precision'], 0.5)
        self.assertAlmostEqual(m['scrum_recall'], 0.5)
        base = {k: 0.7 for k in m}
        better = {k: 0.75 for k in m}
        self.assertTrue(success_rule(base, better, [0.01, 0.09], [True, True, False])['passed'])
        bad = better.copy()
        bad['scrum_precision'] = 0.67
        self.assertFalse(success_rule(base, bad, [0.01, 0.09], [True, True, True])['passed'])
        self.assertFalse(success_rule(base, better, [-0.01, 0.09], [True, True, True])['passed'])

    def test_temporal_bootstrap_identity(self):
        y = np.tile([0, 1, 2], 30)
        pred = [y.copy() for _ in range(3)]
        section = np.repeat(['S01', 'S02', 'S03'], 30)
        start = np.tile(np.arange(30) * 7, 3)
        b = temporal_bootstrap(y, pred, pred, section, start, replicates=20)
        np.testing.assert_allclose(b['ci95'], [0, 0])
        self.assertEqual(b['mean_difference'], 0)
if __name__ == '__main__':
    unittest.main()
