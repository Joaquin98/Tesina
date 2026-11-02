"""Model architectures for video classification using pretrained feature extractors and sequence models."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import keras
import numpy as np
from models.M0.config import IMG_SIZE, MAX_SEQ_LENGTH, NUM_FEATURES

def build_feature_extractor_inceptionv3():
    """Build a feature extractor using InceptionV3 pretrained on ImageNet."""
    feature_extractor = keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name='feature_extractor_inceptionv3')

def build_feature_extractor_resnet50():
    """Build a feature extractor using ResNet50 pretrained on ImageNet."""
    feature_extractor = keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocess_input = keras.applications.resnet50.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name='feature_extractor_resnet50')

def build_feature_extractor_efficientnetb0():
    """Build a feature extractor using EfficientNetB0 pretrained on ImageNet."""
    base = keras.applications.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    preprocess_input = keras.applications.efficientnet.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = preprocess_input(inputs)
    x = base(x)
    if x.shape[-1] != NUM_FEATURES:
        x = keras.layers.Dense(NUM_FEATURES, activation=None, name='feature_projection')(x)
    return keras.Model(inputs, x, name='feature_extractor_efficientnetb0')

def build_optimizer(name: str='adam', lr: float=0.0001, weight_decay: float=1e-05):
    """Return a Keras optimizer instance based on provided name and learning rate."""
    name = (name or 'adam').lower()
    try:
        if name == 'adam':
            return keras.optimizers.Adam(learning_rate=lr)
        if name == 'adamw':
            try:
                return keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
            except Exception:
                print('AdamW optimizer not available; falling back to Adam')
                return keras.optimizers.Adam(learning_rate=lr)
        if name == 'sgd':
            return keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        return keras.optimizers.get(name)
    except Exception as e:
        print(f"Failed to build optimizer '{name}': {e} — falling back to Adam(lr={lr})")
        return keras.optimizers.Adam(learning_rate=lr)

def get_sequence_model_gru_small(num_classes: int, optimizer):
    """Small GRU model for sequence classification."""
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool')
    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation='relu')(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model([frame_features_input, mask_input], output, name='gru_small')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def get_sequence_model_gru_big(num_classes: int, optimizer):
    """Larger bidirectional GRU model for sequence classification."""
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool')
    x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True))(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(32)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model([frame_features_input, mask_input], output, name='gru_big')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def get_sequence_model_lstm(num_classes: int, optimizer):
    """Bidirectional LSTM model for sequence classification."""
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool')
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(frame_features_input, mask=mask_input)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model([frame_features_input, mask_input], output, name='lstm')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def get_sequence_model_conv1d(num_classes: int, optimizer):
    """Conv1D model for sequence classification (faster, captures local patterns)."""
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool')
    x = keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(frame_features_input)
    x = keras.layers.MaxPooling1D(pool_size=2)(x)
    x = keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model([frame_features_input, mask_input], output, name='conv1d')
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
FEATURE_EXTRACTORS = {'inceptionv3': build_feature_extractor_inceptionv3, 'resnet50': build_feature_extractor_resnet50, 'efficientnetb0': build_feature_extractor_efficientnetb0}
SEQUENCE_MODELS = {'gru_small': get_sequence_model_gru_small, 'gru_big': get_sequence_model_gru_big, 'lstm': get_sequence_model_lstm, 'conv1d': get_sequence_model_conv1d}
