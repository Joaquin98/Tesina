"""Entrena una fusiÃ³n temporal ResNet-secuencia + YOLO-secuencia.

El histÃ³rico usado hasta ahora sÃ³lo aporta sus probabilidades a la fusiÃ³n.
Este experimento conserva los vectores ResNet por frame y aprende, junto con
los 65 descriptores YOLO, una cabeza temporal residual. El test natural queda
bloqueado: arquitectura, Ã©poca y candidato se eligen en validaciÃ³n.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import json
import os
import random
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
OUT = HERE / 'dual_sequence_seed42'
RESNET_ROOT = REPO / 'data' / 'validation_overlap_comparison' / 'feature_memmap'
YOLO_ROOT = HERE / 'temporal_search_seed42' / 'sequence_cache'
PCA_PATH = OUT / 'resnet_pca.joblib'
SEQ_LENGTH = 50
RESNET_COMPONENTS = 64
MAX_EPOCHS = 30
BATCH = 128
DEVICE = torch.device('cpu')

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)

def metric(y, logits):
    pred = np.asarray(logits).argmax(1)
    return {'accuracy': float(accuracy_score(y, pred)), 'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0))}

def norm(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-08, None)
    return p / p.sum(1, keepdims=True)

def load_yolo(split):
    seq = _load_model_result_array(YOLO_ROOT / f'{split}.npy', allow_pickle=True).astype(np.float32)
    mask = _load_model_result_array(YOLO_ROOT / f'{split}_mask.npy', allow_pickle=True).astype(bool)
    names = _load_model_result_array(YOLO_ROOT / f'{split}_names.npy', allow_pickle=True).astype(str)
    return (seq, mask, names)

def load_resnet_index():
    values = json.loads(_resolve_model_artifact(RESNET_ROOT / 'resnet50_index.json').read_text(encoding='utf-8'))
    return {name: i for i, name in enumerate(values['video_names'])}

def frame_sample_indices(names, index, total, per_clip=20):
    ids = []
    for name in names:
        start = index[name]
        ids.extend(start * 200 + np.linspace(0, 199, per_clip, dtype=np.int64))
    return np.asarray(ids, dtype=np.int64)

def fit_pca(train_names, index):
    if _resolve_model_artifact(PCA_PATH).exists():
        return joblib.load(_resolve_model_artifact(PCA_PATH))
    features = _load_model_result_array(RESNET_ROOT / 'resnet50_features.npy', mmap_mode='r')
    features_flat = features.reshape(-1, features.shape[-1])
    sample_ids = frame_sample_indices(train_names, index, len(features), per_clip=20)
    pca = IncrementalPCA(n_components=RESNET_COMPONENTS, batch_size=2048)
    for begin in range(0, len(sample_ids), 4096):
        batch = np.asarray(features_flat[sample_ids[begin:begin + 4096]], dtype=np.float32)
        pca.partial_fit(batch)
    PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, PCA_PATH, compress=3)
    return pca

def transform_resnet(names, index, pca):
    features = _load_model_result_array(RESNET_ROOT / 'resnet50_features.npy', mmap_mode='r')
    masks = _load_model_result_array(RESNET_ROOT / 'resnet50_masks.npy', mmap_mode='r')
    output = np.empty((len(names), 200, RESNET_COMPONENTS), dtype=np.float32)
    valid = np.empty((len(names), 200), dtype=bool)
    for begin in range(0, len(names), 32):
        part = names[begin:begin + 32]
        ids = np.asarray([index[name] for name in part], dtype=np.int64)
        flat = np.asarray(features[ids], dtype=np.float32).reshape(-1, features.shape[-1])
        output[begin:begin + len(part)] = pca.transform(flat).reshape(len(part), 200, RESNET_COMPONENTS).astype(np.float32)
        valid[begin:begin + len(part)] = np.asarray(masks[ids], dtype=bool)
    return (output, valid)

def pool50(x, mask):
    n, t, d = x.shape
    width = t // SEQ_LENGTH
    x = x[:, :SEQ_LENGTH * width].reshape(n, SEQ_LENGTH, width, d)
    m = mask[:, :SEQ_LENGTH * width].reshape(n, SEQ_LENGTH, width)
    weights = m.astype(np.float32)[..., None]
    return ((x * weights).sum(2) / np.maximum(weights.sum(2), 1.0), m.mean(2))

class DualTemporal(nn.Module):

    def __init__(self, mode: str, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.mode = mode
        self.res = nn.Sequential(nn.LayerNorm(RESNET_COMPONENTS), nn.Linear(RESNET_COMPONENTS, 48), nn.GELU())
        self.yolo = nn.Sequential(nn.LayerNorm(65), nn.Linear(65, 32), nn.GELU())
        self.mix = nn.Sequential(nn.LayerNorm(80), nn.Linear(80, 80), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(80, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.LayerNorm(128 + 3), nn.Linear(131, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 3))
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, res, yolo, valid_ratio, base_logits):
        x = torch.cat([self.res(res), self.yolo(yolo)], dim=-1)
        x = self.mix(x)
        h, _ = self.gru(x)
        score = self.attn(h).squeeze(-1)
        score = score.masked_fill(valid_ratio <= 0, -10000.0)
        weights = torch.softmax(score, dim=1).unsqueeze(-1)
        pooled = (h * weights).sum(1)
        delta = self.head(torch.cat([pooled, base_logits], dim=-1))
        return delta if self.mode == 'direct' else base_logits + torch.tanh(self.delta_scale) * delta

def base_probs(split, frame):
    if split == 'train':
        return R.historical_train_probabilities(frame, REPO / 'data' / 'historical_locked_yolo_residual_20260811' / 'results_seed42')
    path = R.REPRODUCTION / ('checkpoint_validation_predictions.csv' if split == 'validation' else 'checkpoint_test_predictions.csv')
    return R.stored_probabilities(path, frame)

def train_one(kind, seed, data, y_train, y_val, base_train, base_val):
    seed_all(seed)
    res_tr, yo_tr, vr_tr = data['train']
    res_va, yo_va, vr_va = data['validation']
    base_tr = np.log(norm(base_train)).astype(np.float32)
    base_va = np.log(norm(base_val)).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(res_tr), torch.from_numpy(yo_tr), torch.from_numpy(vr_tr), torch.from_numpy(base_tr), torch.from_numpy(y_train.astype(np.int64)))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    model = DualTemporal(kind, seed).to(DEVICE)
    weights = torch.tensor([1.1, 1.0, 1.05], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0002)
    best = None
    best_state = None
    stale = 0
    va = (torch.from_numpy(res_va), torch.from_numpy(yo_va), torch.from_numpy(vr_va), torch.from_numpy(base_va))
    for epoch in range(MAX_EPOCHS):
        model.train()
        for r, q, m, b, y in loader:
            opt.zero_grad(set_to_none=True)
            out = model(r, q, m, b)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(*va).cpu().numpy()
        val_metric = metric(y_val, val_logits)
        key = (val_metric['f1_macro'], val_metric['accuracy'])
        if best is None or key > best[0]:
            best = (key, epoch + 1, val_metric)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(*va).cpu().numpy()
    return (model, best, val_logits)

def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = R.common.load_base_splits(False, 'natural')
    y_train, y_val, y_test = map(R.labels, (train, val, test))
    yolo = {s: load_yolo(s) for s in ('train', 'validation', 'test')}
    index = load_resnet_index()
    pca = fit_pca(yolo['train'][2], index)
    data = {}
    projected_cache = OUT / 'projected_sequences.npz'
    if _resolve_model_artifact(projected_cache).exists():
        with _load_model_result_array(projected_cache) as z:
            for s in ('train', 'validation', 'test'):
                data[s] = (z[f'res_{s}'].astype(np.float32), z[f'yolo_{s}'].astype(np.float32), z[f'valid_{s}'].astype(np.float32))
    else:
        for s in ('train', 'validation', 'test'):
            rseq, rmask = transform_resnet(yolo[s][2], index, pca)
            rp, rv = pool50(rseq, rmask)
            yp, yv = pool50(yolo[s][0], yolo[s][1])
            yp = np.concatenate([yp, yv[..., None]], axis=-1)
            data[s] = (rp.astype(np.float32), yp[..., :65].astype(np.float32), rv.astype(np.float32))
        np.savez_compressed(projected_cache, **{f'{name}_{s}': value for s, triplet in data.items() for name, value in zip(('res', 'yolo', 'valid'), triplet)})
    base = {s: base_probs(s, frame) for s, frame in (('train', train), ('validation', val), ('test', test))}
    baseline = metric(y_test, np.log(norm(base['test'])))
    rows = []
    trained = {}
    for kind in ('direct', 'residual'):
        for seed in (17, 42):
            model, best, _ = train_one(kind, seed, data, y_train, y_val, base['train'], base['validation'])
            model.eval()
            with torch.no_grad():
                test_logits = model(torch.from_numpy(data['test'][0]), torch.from_numpy(data['test'][1]), torch.from_numpy(data['test'][2]), torch.from_numpy(np.log(norm(base['test'])).astype(np.float32))).cpu().numpy()
            row = {'kind': kind, 'seed': seed, 'best_epoch': best[1], 'validation_accuracy': best[2]['accuracy'], 'validation_f1_macro': best[2]['f1_macro'], **{'test_' + k: v for k, v in metric(y_test, test_logits).items()}}
            rows.append(row)
            trained[kind, seed] = model.state_dict()
            print(json.dumps(row), flush=True)
    ranking = pd.DataFrame(rows).sort_values(['validation_f1_macro', 'validation_accuracy'], ascending=False)
    ranking.to_csv(OUT / 'ranking.csv', index=False)
    selected = ranking.iloc[0].to_dict()
    selected_test = rows[int(ranking.index[0])]
    result = {'baseline_historical_test': baseline, 'candidates': rows, 'selected': selected_test, 'delta_selected_accuracy': float(selected_test['test_accuracy'] - baseline['accuracy']), 'protocol': {'resnet_components': RESNET_COMPONENTS, 'yolo_components': 65, 'pooled_steps': SEQ_LENGTH, 'selection': 'natural validation only', 'test_locked': True, 'pca_fit': 'train natural frames only'}, 'seconds': time.perf_counter() - started}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
