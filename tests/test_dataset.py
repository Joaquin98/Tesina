"""Regressions: balance after splitting; never manufacture missing video time."""
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DatasetTests(unittest.TestCase):
    def fixture(self):
        return [dict(clip_id=f'{tag}_{n}', video_name=f'{tag}_{n}.mp4', tag=tag,
                     sha256=f'{tag}_{n}', split='test' if n >= 10 else 'train')
                for tag in ['line', 'other', 'scrum'] for n in range(12)]

    def test_oversampling_cannot_cross_partition_boundaries(self):
        from tools.dataset import build_partitions
        parts = build_partitions(self.fixture(), target=20)
        self.assertEqual(len(parts['balanced_train']), 42)
        self.assertEqual(len(parts['balanced_validation']), 18)
        self.assertEqual(len(parts['test']), 6)
        train = {r['video_name'] for r in parts['balanced_train']}
        val = {r['video_name'] for r in parts['balanced_validation']}
        test = {r['video_name'] for r in parts['test']}
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(parts, build_partitions(self.fixture(), target=20))

    def test_duplicate_content_under_another_name_is_rejected(self):
        from tools.dataset import build_partitions
        rows = self.fixture()
        rows[-1]['sha256'] = rows[0]['sha256']
        with self.assertRaises(ValueError):
            build_partitions(rows)

    def test_known_fifteen_second_source_uses_actual_center(self):
        from tools.dataset import seven_second_interval
        self.assertEqual(seven_second_interval(15), (4., 11.))

    def test_reviewed_interval_is_extended_around_its_center(self):
        from tools.dataset import seven_second_interval
        self.assertEqual(seven_second_interval(100, 20, 26), (19.5, 26.5))
        self.assertEqual(seven_second_interval(10, 0, 6), (0., 7.))
        with self.assertRaises(ValueError):
            seven_second_interval(6.96)
        with self.assertRaises(ValueError):
            seven_second_interval(100, 95, 102)

    def test_real_extraction_preserves_source_time_without_stretching(self):
        import cv2
        import numpy as np
        from tools.dataset import extract_real_segment, probe_video
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root/'source.avi', root/'clip.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25, (320, 240))
            self.assertTrue(writer.isOpened())
            for n in range(225):
                writer.write(np.full((240, 320, 3), n, np.uint8))
            writer.release()
            extract_real_segment(source, output, 1, 8)
            info = probe_video(output, decode=True)
            self.assertEqual((info['frames'], info['fps'], info['width'], info['height']), (175, 25., 224, 224))
            cap = cv2.VideoCapture(str(output))
            ok, first = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 174)
            ok2, last = cap.read()
            cap.release()
            self.assertTrue(ok and ok2)
            self.assertAlmostEqual(float(first.mean()), 25, delta=5)
            self.assertAlmostEqual(float(last.mean()), 199, delta=5)
            with self.assertRaises(FileExistsError):
                extract_real_segment(source, output, 1, 8)

    def test_fractional_seek_and_fps_preserve_175_frames(self):
        import cv2
        import numpy as np
        from tools.dataset import extract_real_segment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 29.97, (320, 240))
            for n in range(300):
                writer.write(np.full((240, 320, 3), n%200, np.uint8))
            writer.release()
            info = extract_real_segment(source, root/'clip.mp4', 1.073, 8.073)
            self.assertEqual(info['frames'], 175)

    def test_export_preserves_half_scale_field_of_view(self):
        import cv2
        import numpy as np
        from tools.dataset import extract_real_segment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 25, (854, 480))
            # At half scale, the model crop includes x~203..651 of the source.
            frame = np.zeros((480, 854, 3), np.uint8)
            frame[:, 220:300] = 255
            for _ in range(200):
                writer.write(frame)
            writer.release()
            extract_real_segment(source, root/'clip.mp4', 0, 7)
            cap = cv2.VideoCapture(str(root/'clip.mp4'))
            ok, result = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertGreater(float(result[:, 12:35].mean()), 220)

    def test_training_manifest_rejects_external_and_shared_inputs(self):
        from tools.dataset import validate_partition_rows
        row = dict(video_name='clip.mp4', tag='line', sha256='x', path='../original.mp4')
        with self.assertRaises(ValueError):
            validate_partition_rows({'train': [row], 'validation': [], 'test': []}, ROOT)

    def test_training_consumes_explicit_balanced_partitions_without_resplitting(self):
        from tools.dataset import build_partitions, write_csv, digest
        from models.M0.runtime import training_partitions
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            root = Path(directory)
            rows = self.fixture()
            for n, row in enumerate(rows):
                path = root/row['video_name']
                path.write_bytes(str(n).encode())
                row.update(path=path.relative_to(ROOT).as_posix(), sha256=digest(path))
            parts = build_partitions(rows, target=20)
            write_csv(root/'manifest.csv', rows)
            for name, values in parts.items():
                write_csv(root/f'{name}.csv', values)
            train, validation, test = training_partitions(root)
            self.assertEqual((len(train), len(validation), len(test)), (42, 18, 6))
            self.assertEqual({x['video_name'] for x in validation}, {x['video_name'] for x in parts['validation']})

    def test_preparation_cli_writes_separate_validation_and_refuses_overwrite(self):
        import subprocess
        from tools.dataset import write_csv, read_csv
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            root = Path(directory)
            write_csv(root/'manifest.csv', self.fixture())
            command = [sys.executable, '-B', str(ROOT/'scripts/data_prep/create_balanced_splits.py'),
                       '--manifest', str(root/'manifest.csv'), '--out-dir', str(root/'out'), '--target', '20']
            result = subprocess.run(command, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            self.assertEqual(len(read_csv(root/'out/balanced_validation.csv')), 18)
            before = (root/'out/balanced_train.csv').read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((root/'out/balanced_train.csv').read_bytes(), before)

    def test_verifier_rejects_duplicate_test_rows_and_empty_validation(self):
        from tools.dataset import build_partitions, write_csv, verify, digest
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            root = Path(directory)
            rows = self.fixture()
            for n, row in enumerate(rows):
                path = root/row['video_name']
                path.write_bytes(str(n).encode())
                row.update(path=path.relative_to(ROOT).as_posix(), sha256=digest(path))
            parts = build_partitions(rows, target=20)
            write_csv(root/'manifest.csv', rows)
            for name, values in parts.items():
                write_csv(root/f'{name}.csv', values)
            write_csv(root/'test.csv', parts['test']+[parts['test'][0]])
            with self.assertRaises(ValueError):
                verify(root)
            write_csv(root/'test.csv', parts['test'])
            write_csv(root/'balanced_validation.csv', [])
            with self.assertRaises(ValueError):
                verify(root)

    def test_partition_exports_preserve_optional_source_metadata_consistently(self):
        from tools.dataset import build_partitions, write_partition_files, read_csv
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            root = Path(directory)
            rows = self.fixture()
            rows[-1]['source_start_s'] = '9.5'
            parts = build_partitions(rows, target=20)
            write_partition_files(root, rows, parts)
            manifest = {r['video_name']: r for r in read_csv(root/'manifest.csv')}
            for name in parts:
                for row in read_csv(root/f'{name}.csv'):
                    self.assertEqual(row, manifest[row['video_name']])


if __name__ == '__main__':
    unittest.main()
