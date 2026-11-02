"""Shared temporal classifier with optional learned nine-view pooling."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
import hashlib
import keras
import tensorflow as tf
import numpy as np

@keras.saving.register_keras_serializable(package='multicrop')
class RegionAttention(keras.layers.Layer):
    def __init__(self,units=64,**kwargs):
        super().__init__(**kwargs); self.units=units
        self.hidden=keras.layers.Dense(units,activation='tanh',name='attention_hidden')
        self.score=keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer='zeros',name='attention_score')
    def call(self,inputs):
        features,mask=inputs
        weights=tf.nn.softmax(tf.squeeze(self.score(self.hidden(features)),axis=-1),axis=-1)
        weights=weights*tf.cast(mask[:,:,None],weights.dtype)
        pooled=tf.reduce_sum(features*weights[:,:,:,None],axis=2)
        return pooled,weights
    def get_config(self): return {**super().get_config(),'units':self.units}

def core_hash(core):
    h=hashlib.sha256()
    for a in core.get_weights(): h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def build(mode):
    # Instantiate and initialize shared weights BEFORE the attention layers.
    f=keras.Input((200,2048),name='core_features')
    m=keras.Input((200,),dtype='bool',name='core_mask')
    x=keras.layers.Bidirectional(keras.layers.GRU(64,return_sequences=True),name='bigru1')(f,mask=m)
    x=keras.layers.Bidirectional(keras.layers.GRU(32),name='bigru2')(x,mask=m)
    x=keras.layers.Dropout(.3,name='dropout')(x)
    x=keras.layers.Dense(64,activation='relu',name='dense')(x)
    y=keras.layers.Dense(3,activation='softmax',name='classes')(x)
    core=keras.Model([f,m],y,name='temporal_core')
    shape=(200,2048) if mode=='fixed' else (200,9,2048)
    features=keras.Input(shape,name='features'); mask=keras.Input((200,),dtype='bool',name='valid_frames')
    probe=None
    if mode=='multi':
        pooled,weights=RegionAttention(name='region_attention')([features,mask])
        probe=keras.Model([features,mask],weights)
    elif mode=='fixed': pooled=features
    else: raise ValueError('Unknown mode')
    model=keras.Model([features,mask],core([pooled,mask]))
    model.compile(optimizer=keras.optimizers.Adam(1e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model,core,probe
