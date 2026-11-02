import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(source_directory(experiment_directory(__file__))))
from models.M1.studies.m1_recall_cost_20261027.cached_sequence import NativeRGBCache

class CacheTests(unittest.TestCase):

    def test_keeps_native_dtype_and_exact_values_and_evicts_to_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'a.npz'
            q = Path(directory) / 'b.npz'
            values = np.zeros((200, 2048), np.float32)
            values[0, 0] = 1.1234567
            mask = np.arange(200) < 175
            np.savez(p, features=values, mask=mask, length=175)
            np.savez(q, features=values.astype(np.float16), mask=mask, length=175)
            cache = NativeRGBCache(max_bytes=values.nbytes + mask.nbytes)
            a = cache(str(p), 'a')
            self.assertEqual(a[0].dtype, np.float32)
            self.assertEqual(a[0][0, 0], values[0, 0])
            self.assertFalse(a[0].flags.writeable)
            self.assertEqual(a[2], 175)
            self.assertIs(cache(str(p), 'a'), a)
            b = cache(str(q), 'b')
            self.assertEqual(b[0].dtype, np.float16)
            np.testing.assert_array_equal(b[0].astype(np.float32), values.astype(np.float16).astype(np.float32))
            self.assertLessEqual(cache.bytes, cache.max_bytes)
            self.assertEqual(len(cache.values), 1)
if __name__ == '__main__':
    unittest.main()
