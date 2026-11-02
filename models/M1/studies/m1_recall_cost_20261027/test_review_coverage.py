import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
from models.M1.studies.m1_recall_cost_20261027.review_coverage import review_coverage

class ReviewCoverageTests(unittest.TestCase):

    def test_one_long_alert_includes_two_events_but_does_not_change_matching(self):
        real = [dict(id='a', play_type='line', start_s=0, end_s=3), dict(id='b', play_type='line', start_s=7, end_s=10)]
        value = review_coverage(real, [dict(play_type='line', start_time=0, end_time=10)])
        self.assertEqual(value['any_alert_segments'], 2)
        self.assertEqual(value['same_class_segments'], 2)

    def test_wrong_class_is_reviewed_but_not_correctly_classified(self):
        value = review_coverage([dict(id='a', play_type='line', start_s=0, end_s=3)], [dict(play_type='scrum', start_time=1, end_time=3)])
        self.assertEqual(value['any_alert_segments'], 1)
        self.assertEqual(value['same_class_segments'], 0)

    def test_less_than_one_second_does_not_count(self):
        value = review_coverage([dict(id='a', play_type='line', start_s=0, end_s=3)], [dict(play_type='line', start_time=2.5, end_time=4)])
        self.assertEqual(value['any_alert_segments'], 0)
if __name__ == '__main__':
    unittest.main()
