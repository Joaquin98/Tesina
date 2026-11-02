"""DeepSets temporal sobre las detecciones crudas de YOLO.

Cada frame conserva hasta 30 personas (caja, confianza, keypoints y
confianzas). Un encoder compartido por persona produce una representaciÃ³n
permutation-invariant (media + mÃ¡ximo) y una GRU resume los 40 instantes. La
cabeza es residual sobre los logits histÃ³ricos.
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
import json, os, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as R
OUT = HERE / 'raw_set_temporal_seed42'
CACHE = OUT / 'raw_cache'
POSE = REPO / 'data/player_geometry_fusion/pose_cache/frames_200/yolo11n-pose_pt'
T = 40
P = 30
D = 56
BATCH = 96
EPOCHS = 35
DEVICE = torch.device('cpu')

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)

def norm(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-08, None)
    return p / p.sum(1, keepdims=True)

def metric(y, z):
    pred = np.asarray(z).argmax(1)
    return {'accuracy': float(accuracy_score(y, pred)), 'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0))}

def load_raw(names, split):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{split}.npy'
    if _resolve_model_artifact(path).exists():
        return _load_model_result_array(path, mmap_mode='r')
    out = np.zeros((len(names), T, P, D), dtype=np.float32)
    step = np.linspace(0, 199, T, dtype=np.int64)
    for i, name in enumerate(names):
        z = _load_model_result_array(POSE / f'{Path(name).stem}.npz', allow_pickle=True)
        boxes = z['boxes'][step]
        pc = z['person_conf'][step][..., None]
        kp = z['keypoints'][step]
        kc = z['keypoint_conf'][step]
        valid = z['valid'][step]
        out[i] = np.concatenate([boxes, pc, kp.reshape(T, P, 34), kc], axis=-1).astype(np.float32)
        out[i, ~valid] = 0.0
    np.save(path, out)
    return out

def base(split, frame):
    if split == 'train':
        return R.historical_train_probabilities(frame, REPO / 'data/historical_locked_yolo_residual_20260811/results_seed42')
    return R.stored_probabilities(R.REPRODUCTION / ('checkpoint_validation_predictions.csv' if split == 'validation' else 'checkpoint_test_predictions.csv'), frame)

class SetGRU(nn.Module):

    def __init__(self, mode, seed_value):
        super().__init__()
        torch.manual_seed(seed_value)
        self.mode = mode
        self.person = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 48), nn.GELU(), nn.Linear(48, 48), nn.GELU())
        self.frame = nn.Sequential(nn.LayerNorm(96 + 2), nn.Linear(98, 80), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(80, 64, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(128, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.LayerNorm(128 + 3), nn.Linear(131, 64), nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 3))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, valid, base_logits):
        h = self.person(x)
        m = valid.float().unsqueeze(-1)
        mean = (h * m).sum(2) / m.sum(2).clamp_min(1)
        mx = h.masked_fill(~valid.unsqueeze(-1), -10000.0).max(2).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        frac = valid.float().mean(2).unsqueeze(-1)
        conf = x[..., 4].mean(2).unsqueeze(-1)
        f = self.frame(torch.cat([mean, mx, frac, conf], -1))
        q, _ = self.gru(f)
        score = self.attn(q).squeeze(-1)
        score = score.masked_fill(~valid.any(2), -10000.0)
        a = torch.softmax(score, 1).unsqueeze(-1)
        pooled = (q * a).sum(1)
        delta = self.head(torch.cat([pooled, base_logits], -1))
        return delta if self.mode == 'direct' else base_logits + torch.tanh(self.scale) * delta

def train_model(kind, seed_value, data, ytr, yv, btr, bv):
    seed(seed_value)
    btr = np.log(norm(btr)).astype(np.float32)
    bv = np.log(norm(bv)).astype(np.float32)
    ds = TensorDataset(torch.from_numpy(np.asarray(data['train'])), torch.from_numpy(btr), torch.from_numpy(ytr.astype(np.int64)))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    model = SetGRU(kind, seed_value)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0002)
    lossfn = nn.CrossEntropyLoss(weight=torch.tensor([1.1, 1.0, 1.05]))
    va_x = torch.from_numpy(np.asarray(data['validation']))
    va_b = torch.from_numpy(bv)
    best = None
    state = None
    stale = 0
    for epoch in range(EPOCHS):
        model.train()
        for x, b, y in loader:
            opt.zero_grad(set_to_none=True)
            out = model(x, x[..., 4] > 0, b)
            loss = lossfn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vz = model(va_x, va_x[..., 4] > 0, va_b).numpy()
        mm = metric(yv, vz)
        key = (mm['f1_macro'], mm['accuracy'])
        if best is None or key > best[0]:
            best = (key, epoch + 1, mm)
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(state)
    model.eval()
    return (model, best)

def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    train, val, test = R.common.load_base_splits(False, 'natural')
    ytr, yv, yt = map(R.labels, (train, val, test))
    names = {s: frame.video_name.astype(str).to_numpy() for s, frame in [('train', train), ('validation', val), ('test', test)]}
    data = {s: load_raw(names[s], s) for s in names}
    bp = {s: base(s, frame) for s, frame in [('train', train), ('validation', val), ('test', test)]}
    rows = []
    for kind in ('direct', 'residual'):
        for seed_value in (17, 42):
            model, best = train_model(kind, seed_value, data, ytr, yv, bp['train'], bp['validation'])
            with_dummy = torch.from_numpy(np.asarray(data['test']))
            with_base = torch.from_numpy(np.log(norm(bp['test'])).astype(np.float32))
            with torch.no_grad():
                z = model(with_dummy, with_dummy[..., 4] > 0, with_base).numpy()
            mm = metric(yt, z)
            row = {'kind': kind, 'seed': seed_value, 'best_epoch': best[1], 'validation_accuracy': best[2]['accuracy'], 'validation_f1_macro': best[2]['f1_macro'], 'test_accuracy': mm['accuracy'], 'test_f1_macro': mm['f1_macro']}
            rows.append(row)
            print(json.dumps(row), flush=True)
    ranking = pd.DataFrame(rows).sort_values(['validation_f1_macro', 'validation_accuracy'], ascending=False)
    ranking.to_csv(OUT / 'ranking.csv', index=False)
    selected = ranking.iloc[0].to_dict()
    result = {'historical_test': metric(yt, np.log(norm(bp['test']))), 'candidates': rows, 'selected': selected, 'delta_selected_accuracy': float(selected['test_accuracy'] - metric(yt, np.log(norm(bp['test'])))['accuracy']), 'protocol': {'raw_person_dims': D, 'people': P, 'steps': T, 'selection': 'natural validation only', 'test_locked': True}, 'seconds': time.perf_counter() - started}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
if __name__ == '__main__':
    main()
