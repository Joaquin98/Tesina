import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import unittest, tempfile
import numpy as np
from model import build, protection_loss, keras, tf

class PreservingTests(unittest.TestCase):

    def setUp(self):
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)
        self.e = np.random.default_rng(1).normal(size=(7, 9, 64)).astype('float32')
        self.z = np.random.default_rng(2).normal(size=(7, 9, 3)).astype('float32')

    def test_residuals_start_exactly_central_and_zero_correction(self):
        for mode in ['residual', 'protected_residual']:
            net, probe = build(mode)
            p = net([self.e, self.z], training=False)
            np.testing.assert_allclose(p, tf.nn.softmax(self.z[:, 0]), atol=1e-07)
            a = probe([self.e, self.z]).numpy()
            np.testing.assert_allclose(a[:, 0], 0)
            np.testing.assert_allclose(a.sum(1), 1, atol=1e-06)

    def test_mixture_center_floor_and_convex_hull(self):
        net, probe = build('late_mix')
        a = probe([self.e, self.z]).numpy()
        p = net([self.e, self.z]).numpy()
        pv = tf.nn.softmax(self.z).numpy()
        self.assertTrue((a[:, 0] >= 0.5).all())
        np.testing.assert_allclose(a.sum(1), 1, atol=1e-06)
        np.testing.assert_allclose(p, np.einsum('bv,bvc->bc', a, pv), atol=1e-06)
        self.assertTrue((p >= pv.min(1) - 1e-06).all())
        self.assertTrue((p <= pv.max(1) + 1e-06).all())

    def test_protection_only_correct_base_examples(self):
        y = tf.constant([0, 2])
        p0 = tf.constant([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
        p = tf.constant([[0.2, 0.4, 0.4], [0.1, 0.1, 0.8]])
        actual = protection_loss(y, p0, p).numpy()
        expected = np.sum(np.array([0.8, 0.1, 0.1]) * np.log(np.array([0.8, 0.1, 0.1]) / np.array([0.2, 0.4, 0.4]))) / 2
        self.assertAlmostEqual(float(actual), float(expected), places=6)
        self.assertAlmostEqual(float(protection_loss(y, p0, p0)), 0, places=6)

    def test_context_permutation_and_roundtrip(self):
        perm = [0, 4, 2, 1, 3, 8, 6, 7, 5]
        for mode in ['residual', 'late_mix', 'protected_residual']:
            net, probe = build(mode)
            net([self.e, self.z])
            net.get_layer('attention_score').kernel.assign(tf.ones_like(net.get_layer('attention_score').kernel) * 0.03)
            if mode != 'late_mix':
                net.get_layer('correction').kernel.assign(tf.ones_like(net.get_layer('correction').kernel) * tf.constant([0.01, -0.02, 0.03]))
            p = net([self.e, self.z], training=False).numpy()
            self.assertGreater(float(np.max(abs(p - tf.nn.softmax(self.z[:, 0]).numpy()))), 0.0001)
            changed = self.e.copy()
            changed[:, 1] += 3.0
            self.assertGreater(float(np.max(abs(p - net([changed, self.z], training=False).numpy()))), 1e-06)
            np.testing.assert_allclose(p, net([self.e[:, perm], self.z[:, perm]], training=False), atol=1e-06)
            with tempfile.TemporaryDirectory() as d:
                path = d + '/head.weights.h5'
                net.save_weights(path)
                other, _ = build(mode)
                other.load_weights(_resolve_model_artifact(path))
                np.testing.assert_allclose(p, other([self.e, self.z], training=False), atol=1e-06)

    def test_no_temporal_weights_and_training_gradients(self):
        for mode in ['residual', 'late_mix', 'protected_residual']:
            net, _ = build(mode)
            with tf.GradientTape() as tape:
                p = net([self.e, self.z], training=True)
                loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(np.arange(7) % 3, p))
            grads = tape.gradient(loss, net.trainable_variables)
            self.assertTrue(all((g is not None for g in grads)))
            self.assertTrue(any((float(tf.reduce_sum(abs(g))) > 1e-07 for g in grads)))
            self.assertLess(net.count_params(), 100000)

    def test_bounded_residual(self):
        net, _ = build('residual')
        net.get_layer('correction').bias.assign([100.0, -100.0, 0.0])
        p = net([self.e, self.z]).numpy()
        expected = tf.nn.softmax(self.z[:, 0] + np.array([2.0, -2.0, 0.0], dtype='float32')).numpy()
        np.testing.assert_allclose(p, expected, atol=1e-06)
if __name__ == '__main__':
    unittest.main()
