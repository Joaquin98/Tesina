import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest
import numpy as np
import subprocess, sys
from costs import class_weights, partitions, classification_counts, decide

class CostsTest(unittest.TestCase):

    def test_declared_costs(self):
        from common import COSTS
        self.assertEqual(COSTS, [8, 16])

    def test_cost_one_is_original(self):
        y = np.array([0] * 2 + [1] * 7 + [2] * 3)
        np.testing.assert_allclose(class_weights(y, 1), len(y) / (3 * np.bincount(y)))

    def test_ratios_and_mean(self):
        y = np.array([0] * 2 + [1] * 7 + [2] * 3)
        base = class_weights(y, 1)
        for cost in [8, 16]:
            w = class_weights(y, cost)
            self.assertAlmostEqual(np.mean(w[y]), 1.0)
            self.assertAlmostEqual(w[0] / w[1] / (base[0] / base[1]), cost)
            self.assertAlmostEqual(w[2] / w[1] / (base[2] / base[1]), cost)

    def test_invalid(self):
        for y, c in [([0, 1], 2), ([0, 1, 2], 0), ([0, 1, 3], 2), ([0, 1, 2], float('nan'))]:
            with self.assertRaises(ValueError):
                class_weights(np.array(y), c)

    def test_partitions(self):
        rows = [{'domain': 'CH', 'split': 'train'}, {'domain': 'CV', 'split': 'train'}, {'domain': 'VF', 'split': 'test'}]
        a, b = partitions(rows)
        self.assertEqual(a.tolist(), [0, 1])
        self.assertEqual(b.tolist(), [2])
        rows[2]['split'] = 'train'
        with self.assertRaises(ValueError):
            partitions(rows)

    def test_counts_and_raw_argmax(self):
        p = np.array([[0.36, 0.35, 0.29], [0.2, 0.39, 0.41], [0.45, 0.5, 0.05], [0.3, 0.4, 0.3]])
        pred = decide(p)
        np.testing.assert_array_equal(pred, [0, 2, 1, 1])
        result = classification_counts(np.array([0, 2, 0, 1]), pred)
        self.assertEqual(result['line'], {'tp': 1, 'fn': 1, 'fp': 0, 'support': 2})
        self.assertEqual(result['scrum'], {'tp': 1, 'fn': 0, 'fp': 0, 'support': 1})

    def test_evaluation_and_publication_take_locks(self):
        from pathlib import Path
        root = experiment_directory(__file__)
        for file, name in [('evaluate.py', 'evaluation.lock'), ('publish_verify.py', 'publication.lock')]:
            self.assertIn("with lock('" + name + "')", _resolve_model_artifact(root / file).read_text())

    def test_lock_rejects_second_process(self):
        from common import lock, HERE
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            filename = str(Path(directory)/'writer.lock')
            code = f"from common import lock\ntry:\n with lock({filename!r}):pass\nexcept RuntimeError:\n raise SystemExit(23)"
            with lock(filename):
                result = subprocess.run([sys.executable, '-B', '-c', code], cwd=HERE, capture_output=True)
            self.assertEqual(result.returncode, 23, result.stderr)
            result = subprocess.run([sys.executable, '-B', '-c', code], cwd=HERE, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
if __name__ == '__main__':
    unittest.main()
