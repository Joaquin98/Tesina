import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models import model_catalog as catalog


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.asset = self.root/'models/familias/pesos/unique.keras'
        self.asset.parent.mkdir(parents=True)
        self.asset.write_bytes(b'trained-weights')
        self.index = {'source_root': 'D:/Tesina', 'aliases': {
            'data/a/model.keras': 'models/familias/pesos/unique.keras',
            'data/b/copy.keras': 'models/familias/pesos/unique.keras'}}
        self.root_patch = patch.object(catalog, 'ROOT', self.root)
        self.index_patch = patch.object(catalog, 'index', return_value=self.index)
        self.root_patch.start(); self.index_patch.start()

    def tearDown(self):
        self.index_patch.stop(); self.root_patch.stop(); self.temp.cleanup()

    def test_aliases_share_one_existing_file(self):
        self.assertEqual(catalog.resolve_artifact(self.root/'data/a/model.keras'), self.asset)
        self.assertEqual(catalog.resolve_artifact('D:/Tesina/data/b/copy.keras'), self.asset)

    def test_unknown_original_path_never_reads_original(self):
        with self.assertRaises(FileNotFoundError):
            catalog.resolve_artifact('D:/Tesina/data/unknown/cache.npy')

    def test_alias_cannot_escape_delivery(self):
        self.index['aliases']['data/a/model.keras'] = '../outside.keras'
        with self.assertRaises(ValueError):
            catalog.resolve_artifact(self.root/'data/a/model.keras')

    def test_glob_discovers_deduplicated_historical_names(self):
        found = list(catalog.artifact_glob(self.root/'data/a', '*.keras'))
        self.assertEqual(found, [self.root/'data/a/model.keras'])
        self.assertEqual(catalog.resolve_artifact(found[0]), self.asset)

    def test_glob_exposes_virtual_recipe_directories(self):
        self.index['aliases']['data/a/attempt_001/model.keras'] = 'models/familias/pesos/unique.keras'
        self.assertIn(self.root/'data/a/attempt_001', list(catalog.artifact_glob(self.root/'data/a', 'attempt_*')))

    def test_glob_matches_components_without_crossing_directories(self):
        self.index['aliases']['data/a/attempt_001/model.keras'] = 'models/familias/pesos/unique.keras'
        self.index['aliases']['data/a/nested/attempt_002/model.keras'] = 'models/familias/pesos/unique.keras'
        directory = self.root/'data/a'
        self.assertEqual(list(catalog.artifact_glob(directory, 'attempt_*/*.keras')), [directory/'attempt_001/model.keras'])
        self.assertEqual(len(list(catalog.artifact_glob(directory, '**/attempt_*/*.keras'))), 2)

    def test_historical_original_video_does_not_override_explicit_current_video(self):
        current=self.root/'data/clip.mp4';current.parent.mkdir();current.write_bytes(b'corrected')
        self.index['historical_media_aliases']={'data/clip.mp4':'models/familias/pesos/unique.keras'}
        self.assertEqual(catalog.resolve_artifact('D:/Tesina/data/clip.mp4'),self.asset)
        self.assertEqual(catalog.resolve_artifact(current),current)

    def test_declared_archived_directory_keeps_glob_available(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory);(archive/'files').mkdir()
            target=archive/'files/report.csv';target.write_text('value\n1\n',encoding='utf-8')
            self.index['aliases']['data/archive']='historic_files'
            self.index['archived_reports']={'historic_files':'files'}
            self.index['source_root']=str(self.root.parent)
            with patch.object(catalog,'ARCHIVE_ROOT',archive):
                self.assertEqual(list(catalog.artifact_glob(self.root/'data/archive','*.csv')),[target])
                self.assertEqual(catalog.resolve_artifact(target),target)

    def test_result_array_can_read_declared_external_archive(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)
            (archive/'values.csv').write_text('row,value_0,value_1\n0,0.125,0.875\n',encoding='utf-8')
            self.index['result_arrays']={'results/probabilities.npy':{'path':'results.csv','dtype':'float32','shape':[1,2]}}
            self.index['archived_reports']={'results.csv':'values.csv'}
            with patch.object(catalog,'ARCHIVE_ROOT',archive):
                values=catalog.load_array('results/probabilities.npy',allow_pickle=False)
            np.testing.assert_array_equal(values,np.asarray([[.125,.875]],dtype='float32'))

    def test_removed_optional_report_remains_an_absent_path(self):
        self.index['aliases']['data/a/report.csv'] = 'historic/report.csv'
        self.index['archived_reports'] = {'historic/report.csv': 'reports/report.csv'}
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            with patch.object(catalog, 'ARCHIVE_ROOT', archive):
                target = catalog.resolve_artifact(self.root / 'data/a/report.csv')
                self.assertEqual(target, archive / 'reports/report.csv')
                self.assertFalse(target.exists())
                with self.assertRaises(FileNotFoundError):
                    target.read_bytes()

    def test_result_array_logical_path_cannot_escape_delivery(self):
        self.index['result_arrays']={'results/probabilities.npy':{'path':'../outside.csv','dtype':'float32','shape':[1,2]}}
        with self.assertRaises(ValueError):catalog.load_array('results/probabilities.npy',allow_pickle=False)

    def test_converted_prediction_array_keeps_shape_and_dtype(self):
        import numpy as np
        values=self.root/'predictions.csv';values.write_text('row,value_0,value_1\n0,0.125,0.875\n',encoding='utf-8')
        self.index['result_arrays']={'data/a/probabilities.npy':{'path':'predictions.csv','dtype':'float32','shape':[1,2]}}
        array=catalog.load_array('D:/Tesina/data/a/probabilities.npy',allow_pickle=False)
        self.assertEqual(array.dtype,np.dtype('float32'))
        np.testing.assert_array_equal(array,np.asarray([[.125,.875]],dtype='float32'))


if __name__ == '__main__':
    unittest.main()
