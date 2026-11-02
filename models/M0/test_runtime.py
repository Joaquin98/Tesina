import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
from models.M0.runtime import encode_frames, split_unique, segment_frames, LABELS


class NoCacheRuntimeTest(unittest.TestCase):
    def test_prepared_clip_keeps_its_full_field_of_view(self):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(224, dtype=np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 25 if prop == cv2.CAP_PROP_FPS else 1
        capture.read.return_value = (True, frame)
        with patch('cv2.VideoCapture', return_value=capture):
            frames, duration = segment_frames(Path('ignored.mp4'))
        np.testing.assert_array_equal(frames[0], frame[:, :, ::-1])
        self.assertEqual(duration, .04)
        capture.release.assert_called_once()

    def test_feature_chunks_preserve_order_and_padding_without_disk(self):
        calls = []
        def encoder(frames, training=False):
            calls.append(len(frames))
            return np.repeat(frames[:, 0, 0, :1], 2048, axis=1)
        frames = np.repeat(np.arange(11, dtype=np.uint8)[:, None, None, None], 3, axis=3)
        with tempfile.TemporaryDirectory() as folder:
            before = list(Path(folder).iterdir())
            features, mask = encode_frames(frames, encoder, chunk_size=4)
            self.assertEqual(before, list(Path(folder).iterdir()))
        np.testing.assert_array_equal(features[:11, 0], np.arange(11))
        self.assertEqual(calls, [4, 4, 3])
        self.assertEqual(features.shape, (200, 2048))
        self.assertEqual(int(mask.sum()), 11)
        self.assertTrue((features[11:] == 0).all())

    def test_empty_clip_fails_instead_of_training_on_padding(self):
        with self.assertRaises(ValueError):
            encode_frames(np.empty((0, 224, 224, 3)), lambda *a, **k: None)

    def test_split_is_disjoint_and_does_not_balance_validation(self):
        rows = [dict(video_name=f'{label}_{i}.mp4', tag=label) for label in LABELS for i in range(10)]
        train, validation = split_unique(rows, seed=42, validation_fraction=.2)
        self.assertEqual(len(train), 24)
        self.assertEqual(len(validation), 6)
        self.assertFalse({r['video_name'] for r in train} & {r['video_name'] for r in validation})
        with self.assertRaises(ValueError):
            split_unique(rows+[rows[0]], seed=42, validation_fraction=.2)


if __name__ == '__main__':
    unittest.main()
