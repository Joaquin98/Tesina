"""Heads only: frozen central embeddings/logits are inputs, never trainable."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import keras
import tensorflow as tf

class ContextInputs(keras.layers.Layer):

    def __init__(self, all_views=False, **kwargs):
        super().__init__(**kwargs)
        self.all_views = all_views

    def call(self, inputs):
        e, z = inputs
        v = e if self.all_views else e[:, 1:]
        c = tf.repeat(e[:, :1], tf.shape(v)[1], axis=1)
        pieces = [c, v, v - c]
        if self.all_views:
            p = tf.nn.softmax(z)
            pieces.append(p - p[:, :1])
        return tf.concat(pieces, axis=-1)

class ContextPool(keras.layers.Layer):

    def call(self, inputs):
        e, z, s = inputs
        w = tf.nn.softmax(s[:, :, 0], axis=1)
        pool = tf.einsum('bv,bvd->bd', w, e[:, 1:])
        features = tf.concat([e[:, 0], pool, pool - e[:, 0], z[:, 0]], axis=1)
        return (features, tf.concat([tf.zeros_like(w[:, :1]), w], axis=1))

class ResidualOutput(keras.layers.Layer):

    def call(self, inputs):
        z, delta = inputs
        return tf.nn.softmax(z[:, 0] + 2.0 * tf.tanh(delta), axis=-1)

class MixtureOutput(keras.layers.Layer):

    def call(self, inputs):
        z, s = inputs
        prior = tf.constant([2.0794415416798357] + [0.0] * 8, dtype=s.dtype)
        w = 0.5 * tf.nn.softmax(s[:, :, 0] + prior, axis=1) + tf.constant([0.5] + [0.0] * 8, dtype=s.dtype)
        return (tf.einsum('bv,bvc->bc', w, tf.nn.softmax(z, axis=-1)), w)

def build(strategy):
    if strategy not in ['residual', 'late_mix', 'protected_residual']:
        raise ValueError(strategy)
    e = keras.Input((9, 64), name='frozen_temporal_embeddings')
    z = keras.Input((9, 3), name='frozen_view_logits')
    context = ContextInputs(all_views=strategy == 'late_mix')([e, z])
    s = keras.layers.Dense(32, activation='tanh', name='attention_hidden')(context)
    s = keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros', name='attention_score')(s)
    if strategy == 'late_mix':
        p, w = MixtureOutput()([z, s])
    else:
        h, w = ContextPool()([e, z, s])
        h = keras.layers.Dense(64, activation='relu', name='correction_hidden')(h)
        delta = keras.layers.Dense(3, kernel_initializer='zeros', bias_initializer='zeros', name='correction')(h)
        p = ResidualOutput()([z, delta])
    return (keras.Model([e, z], p, name=strategy), keras.Model([e, z], w, name=strategy + '_weights'))

def protection_loss(y, p0, p):
    correct = tf.cast(tf.equal(tf.argmax(p0, axis=1, output_type=tf.int32), tf.cast(y, tf.int32)), p.dtype)
    p0 = tf.stop_gradient(p0)
    kl = tf.reduce_sum(p0 * (tf.math.log(tf.maximum(p0, 1e-07)) - tf.math.log(tf.maximum(p, 1e-07))), axis=1)
    return tf.reduce_mean(correct * kl)
