import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import unittest, tempfile, json
from unittest.mock import patch
import numpy as np, pandas as pd
import models.M0.studies.resnet_fixed_halfscale_crop_20261029.experiment as e
class StagedCacheTest(unittest.TestCase):

    def test_subset_reused_without_cnn_recomputation_and_preserves_global_order(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'cache').mkdir()
            m = pd.DataFrame([dict(cache_index=i, video_name=f'v{i}', domain='CH', source_video=__file__, source_bytes=Path(__file__).stat().st_size, source_mtime_ns=Path(__file__).stat().st_mtime_ns, block_id=f'b{i}', start_s=0, end_s=1) for i in range(4)])
            pd.DataFrame(dict(video_name=['v3', 'v1', 'v3'], initial_partition=['train', 'train', 'validation'])).to_csv(out / 'initial_split.csv', index=False)
            calls = []

            class Tensor:

                def __init__(self, v):
                    self.v = v

                def numpy(self):
                    return self.v

            def infer(x):
                calls.append(len(x))
                return Tensor(np.repeat(x[:, 0, 0, :1], 2048, axis=1))

            def frames(*args):
                return [np.full((1, 1, 3), j, dtype=np.uint8) for j in (1, 2)]
            with patch.multiple(e, OUT=out, CACHE=out / 'cache'), patch.object(e, 'guard'), patch.object(e, 'manifest', return_value=m), patch.object(e, 'config_hash', return_value='fixed'), patch.object(e, 'probe', return_value={}), patch.object(e, 'sample_indices', return_value=[0, 1]), patch.object(e, 'native_frames', side_effect=frames), patch.object(e, 'cv2_resize_rgb', side_effect=lambda f, s: f), patch.object(e, 'extractor', return_value=(None, infer)):
                e.extract(320, initial_only=True)
                self.assertEqual(sum(calls), 4)
                x, mask, mapped = e.cache_for_rows(320, np.array([3, 1, 3]))
                self.assertEqual(mapped.tolist(), [1, 0, 1])
                self.assertEqual(x.shape, (2, 200, 2048))
                with self.assertRaises(RuntimeError):
                    e.cache_for_rows(320, np.array([0]))
                calls.clear()
                e.extract(320)
                self.assertEqual(sum(calls), 4)
                full, fullmask = e.open_cache(320)
                np.testing.assert_array_equal(full[[3, 1, 3]], x[mapped])
                np.testing.assert_array_equal(fullmask[[3, 1, 3]], mask[mapped])
                self.assertTrue((full[:, 2:] == 0).all())
                state = json.loads(_resolve_model_artifact(out / 'cache/R320/state.json').read_text())
                self.assertEqual(state['reused_initial_stage_examples'], 2)
                calls.clear()
                e.extract(320)
                self.assertFalse(calls)
                for a in (x, mask, full, fullmask):
                    a._mmap.close()
if __name__ == '__main__':
    unittest.main()
