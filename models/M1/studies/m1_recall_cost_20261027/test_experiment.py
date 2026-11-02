import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(source_directory(experiment_directory(__file__))))
from models.M1.studies.m1_recall_cost_20261027.core import training_weights, SelectedRGB, reconstruct_geometry, evaluate_events

class ExperimentTests(unittest.TestCase):

    def test_costs_preserve_mean_and_increase_positive_to_other_ratio(self):
        y = np.array([0, 0] + [1] * 7 + [2] * 3)
        base = training_weights(y, 1)
        for cost in (2, 4, 8):
            w = training_weights(y, cost)
            self.assertAlmostEqual(float(w[y].mean()), 1)
            for cls in (0, 2):
                self.assertAlmostEqual(w[cls] / w[1] / (base[cls] / base[1]), cost)
        np.testing.assert_allclose(base, [2, 12 / 21, 12 / 9])

    def test_omitted_rgb_cannot_be_read(self):
        keep = np.array([True, False, True, False, True])
        reader = SelectedRGB(np.arange(10).reshape(5, 2), keep)
        np.testing.assert_array_equal(reader[np.array([0, 2])], [[0, 1], [4, 5]])
        with self.assertRaises(AssertionError):
            reader[np.array([1])]

    def test_sparse_geometry_does_not_use_omitted_values(self):
        values = np.array([[0.0], [999.0], [2.0], [999.0], [4.0]], np.float32)
        mask = np.array([True, False, False, False, True])
        a, b, m, idx, li, ri, w = reconstruct_geometry(values, values, mask, 2)
        np.testing.assert_array_equal(a[:, 0], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(b, a)
        np.testing.assert_array_equal(m, [True, True, False, False, True])
        np.testing.assert_array_equal(idx, [0, 2, 4])

    def test_one_prediction_cannot_match_two_events_and_wrong_class_is_not_a_hit(self):
        truth = [dict(id='a', play_type='line', start_s=0, end_s=3), dict(id='b', play_type='line', start_s=4, end_s=7), dict(id='c', play_type='scrum', start_s=10, end_s=13)]
        pred = [dict(play_type='line', start_time=1, end_time=6), dict(play_type='line', start_time=10, end_time=13)]
        result = evaluate_events(truth, pred)
        self.assertEqual(result['metrics']['total']['tp'], 1)
        self.assertEqual(result['metrics']['total']['fp'], 1)
        self.assertEqual(result['metrics']['total']['fn'], 2)

    def test_short_overlap_does_not_count(self):
        result = evaluate_events([dict(id='a', play_type='line', start_s=0, end_s=2)], [dict(play_type='line', start_time=1.5, end_time=3)])
        self.assertEqual(result['metrics']['total']['tp'], 0)
if __name__ == '__main__':
    unittest.main()
