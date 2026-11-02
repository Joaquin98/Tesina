import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[3]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
from pathlib import Path
import json
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import accuracy_score, f1_score
from models.M2.multimodal.train_yolo_residual_bigru import load_yolo, met, SPLIT, BASE, LABEL, HERE

def main():
    tr = pd.read_csv(_resolve_model_artifact(SPLIT / 'train.csv'))
    va = pd.read_csv(_resolve_model_artifact(SPLIT / 'validation.csv'))
    te = pd.read_csv(_resolve_model_artifact(SPLIT / 'test.csv'))
    df = pd.concat([tr, va, te], ignore_index=True)
    names = df.video_name.astype(str).tolist()
    y = df.tag.map(LABEL).to_numpy(np.int32)
    ntr = len(tr)
    nv = len(va)
    yy, mask = load_yolo(names)
    mu = yy[:ntr].reshape(-1, 64).mean(0)
    sd = yy[:ntr].reshape(-1, 64).std(0) + 1e-05
    yy = (yy - mu) / sd
    d = _load_model_result_array(BASE, allow_pickle=True)
    base = np.log(np.clip(np.exp(d['features'][:, -3:].astype(np.float32)), 1e-07, 1)).astype(np.float32)
    m = keras.models.load_model(_resolve_model_artifact(HERE / 'yolo_residual_bigru_seed42/best.keras'), compile=False)
    pv = m.predict([yy[ntr:ntr + nv], mask[ntr:ntr + nv], base[ntr:ntr + nv]], verbose=0)
    pt = m.predict([yy[-len(te):], mask[-len(te):], base[-len(te):]], verbose=0)
    out = {'validation': met(y[ntr:ntr + nv], pv), 'test': met(y[-len(te):], pt), 'historical_ensemble_test': {'accuracy': 0.8776009791921665, 'f1_macro': 0.8479262047057438}, 'checkpoint': 'best.keras'}
    (HERE / 'yolo_residual_bigru_seed42/summary.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
