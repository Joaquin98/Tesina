"""Shared per-view temporal encoder and three class-specific attention pools."""
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
import hashlib
import keras
import tensorflow as tf
import numpy as np

class PerViewTemporal(keras.layers.Layer):
    """Apply ONE shared temporal encoder independently to each entire view."""

    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder

    def call(self, inputs, training=None):
        features, mask = inputs
        batch = tf.shape(features)[0]
        by_view = tf.reshape(tf.transpose(features, [0, 2, 1, 3]), [-1, 8, 2048])
        repeated_mask = tf.repeat(mask, repeats=9, axis=0)
        encoded = self.encoder([by_view, repeated_mask], training=training)
        return tf.reshape(encoded, [batch, 9, 64])

class SequenceAttention(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden = keras.layers.Dense(64, activation='tanh')
        self.score = keras.layers.Dense(3, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, embeddings):
        scores = tf.transpose(self.score(self.hidden(embeddings)), [0, 2, 1])
        weights = tf.nn.softmax(scores, axis=-1)
        return (tf.einsum('bcv,bvd->bcd', weights, embeddings), weights)

class ClassLogits(keras.layers.Layer):
    """One logit per class from that class's representation, before any softmax."""

    def __init__(self, classifier, **kwargs):
        super().__init__(**kwargs)
        self.classifier = classifier

    def call(self, hidden):
        return tf.einsum('bcd,dc->bc', hidden, self.classifier.kernel) + self.classifier.bias

def core_hash(core):
    h = hashlib.sha256()
    for a in core.get_weights():
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def build(mode):
    f = keras.Input((8, 2048), name='core_features')
    m = keras.Input((8,), dtype='bool', name='core_mask')
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True), name='bigru1')(f, mask=m)
    x = keras.layers.Bidirectional(keras.layers.GRU(32), name='bigru2')(x, mask=m)
    x = keras.layers.Dropout(0.3, name='dropout')(x)
    x = keras.layers.Dense(64, activation='relu', name='dense')(x)
    y = keras.layers.Dense(3, activation='softmax', name='classes')(x)
    core = keras.Model([f, m], y, name='temporal_core')
    shape = (8, 2048) if mode == 'fixed' else (8, 9, 2048)
    features = keras.Input(shape, name='features')
    mask = keras.Input((8,), dtype='bool', name='valid_frames')
    probe = None
    if mode == 'multi':
        encoder = keras.Model(core.inputs, core.get_layer('bigru2').output, name='shared_temporal_encoder')
        embeddings = PerViewTemporal(encoder, name='per_view_temporal')([features, mask])
        pooled, weights = SequenceAttention(name='sequence_attention')(embeddings)
        hidden = core.get_layer('dense')(core.get_layer('dropout')(pooled))
        logits = ClassLogits(core.get_layer('classes'), name='class_logits')(hidden)
        output = keras.layers.Softmax(name='class_probabilities')(logits)
        probe = keras.Model([features, mask], weights)
    elif mode == 'fixed':
        output = core([features, mask])
    else:
        raise ValueError('Unknown mode')
    model = keras.Model([features, mask], output)
    model.compile(optimizer=keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return (model, core, probe)
