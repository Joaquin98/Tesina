"""Extract the hidden 64-D embedding of the frozen historical checkpoint."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json
import numpy as np
import keras
REPO = Path(__file__).resolve().parents[3]
HERE = experiment_directory(__file__)
MEM = REPO / 'data/validation_overlap_comparison/feature_memmap'
CK = REPO / 'data/line_recall_training/full/training/line_weight_1_25/best_accuracy.keras'
OUT = HERE / 'historical_embeddings_seed42'

def main():
    idx = json.loads(_resolve_model_artifact(MEM / 'resnet50_index.json').read_text())['video_names']
    feat = _load_model_result_array(MEM / 'resnet50_features.npy', mmap_mode='r')
    mask = _load_model_result_array(MEM / 'resnet50_masks.npy', mmap_mode='r')
    model = keras.models.load_model(_resolve_model_artifact(CK), compile=False)
    emb = keras.Model(model.inputs, model.get_layer('dense').output)
    arr = np.zeros((len(idx), 64), np.float32)
    for s in range(0, len(idx), 8):
        xb = np.asarray(feat[s:s + 8], np.float32)
        mb = np.asarray(mask[s:s + 8], bool)
        arr[s:s + len(xb)] = emb((xb, mb), training=False).numpy()
    OUT.mkdir(exist_ok=True)
    np.save(OUT / 'embeddings.npy', arr)
    (OUT / 'names.json').write_text(json.dumps(idx), encoding='utf-8')
    print({'shape': arr.shape, 'path': str(OUT / 'embeddings.npy')})
if __name__ == '__main__':
    main()
