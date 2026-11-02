"""Busca cabezas temporales residuales sobre la secuencia YOLO de 200 pasos."""
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
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(experiment_directory(__file__).parents[1] / 'data' / 'player_geometry_fusion' / 'pose_cache' / 'frames_200')
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as residual
from models.M2.studies.yolo_resnet_improvement_suite_20260811 import run_suite
OUTPUT = HERE / 'temporal_search_seed42'
SEQ_DIM = 65
LABELS = residual.LABELS

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def write_json(path: Path, value) -> None:

    def safe(item):
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(v) for v in item]
        if isinstance(item, np.ndarray):
            return [safe(v) for v in item.tolist()]
        if isinstance(item, np.generic):
            return item.item()
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, ensure_ascii=False), encoding='utf-8')

def quality(data: dict[str, np.ndarray]) -> np.ndarray:
    valid = data['valid'].astype(bool)
    person = data['person_conf'].astype(np.float32)
    keypoint = data['keypoint_conf'].astype(np.float32)
    count = valid.sum(1).astype(np.float32)
    person_mean = np.divide(np.where(valid, person, 0).sum(1), np.maximum(count, 1))
    keypoint_mean = np.divide(np.where(valid[..., None], keypoint, 0).sum((1, 2)), np.maximum(count * keypoint.shape[-1], 1))
    coverage = np.clip(count / 30.0, 0, 1)
    return np.column_stack([count / 30.0, person_mean, keypoint_mean, coverage]).astype(np.float32)

def sequence_for(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = run_suite.load_clip(name)
    sequence = np.concatenate([data['position_features'], data['pose_features'], quality(data)], axis=1).astype(np.float32)
    return (np.nan_to_num(sequence, nan=0, posinf=0, neginf=0), data['frame_mask'].astype(bool))

def build_sequences(frame: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    root = OUTPUT / 'sequence_cache'
    root.mkdir(parents=True, exist_ok=True)
    seq_path = root / f'{split}.npy'
    mask_path = root / f'{split}_mask.npy'
    names_path = root / f'{split}_names.npy'
    names = frame.video_name.astype(str).to_numpy()
    if _resolve_model_artifact(seq_path).exists() and _resolve_model_artifact(mask_path).exists() and _resolve_model_artifact(names_path).exists() and np.array_equal(_load_model_result_array(names_path, allow_pickle=True).astype(str), names):
        return (_load_model_result_array(seq_path, mmap_mode='r'), _load_model_result_array(mask_path, mmap_mode='r'))
    sequences, masks = ([], [])
    started = time.perf_counter()
    for index, name in enumerate(names, start=1):
        seq, mask = sequence_for(name)
        sequences.append(seq)
        masks.append(mask)
        if index % 250 == 0 or index == len(names):
            print(f'sequences {split}: {index}/{len(names)} elapsed={time.perf_counter() - started:.1f}s', flush=True)
    sequences = np.asarray(sequences, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    np.save(seq_path, sequences)
    np.save(mask_path, masks)
    np.save(names_path, names)
    return (sequences, masks)

class TemporalResidual(nn.Module):

    def __init__(self, kind: str, input_dim: int, hidden: int=96):
        super().__init__()
        self.kind = kind
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(0.15))
        if kind.startswith('gru'):
            self.temporal = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=0.15, bidirectional=True)
            out_dim = hidden * 2
        elif kind.startswith('tcn'):
            self.temporal = nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2, dilation=1), nn.GELU(), nn.Dropout(0.15), nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.GELU(), nn.Dropout(0.15), nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.GELU(), nn.Dropout(0.15))
            out_dim = hidden
        else:
            raise ValueError(kind)
        self.attention = nn.Sequential(nn.Linear(out_dim, max(16, out_dim // 2)), nn.Tanh(), nn.Linear(max(16, out_dim // 2), 1))
        self.head = nn.Sequential(nn.LayerNorm(out_dim + 3), nn.Linear(out_dim + 3, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 3))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor, historical_logits: torch.Tensor) -> torch.Tensor:
        x = self.input(sequence)
        if self.kind.startswith('gru'):
            x, _ = self.temporal(x)
        else:
            x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        valid = mask.bool()
        scores = self.attention(x).squeeze(-1).masked_fill(~valid, -10000.0)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (x * weights).sum(1)
        return self.head(torch.cat([pooled, historical_logits], dim=1))

def metrics(y: np.ndarray, logits: np.ndarray) -> dict:
    pred = logits.argmax(1)
    return {'accuracy': float(accuracy_score(y, pred)), 'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(y, pred, average='weighted', zero_division=0))}

def train_model(kind: str, train_seq, train_mask, train_base, y_train, val_seq, val_mask, val_base, y_val, seed: int):
    seed_all(seed)
    device = torch.device('cpu')
    model = TemporalResidual(kind, SEQ_DIM).to(device)
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = torch.tensor((counts.sum() / (3.0 * np.maximum(counts, 1))).astype(np.float32), device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.0002)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(np.asarray(train_seq, dtype=np.float32)), torch.from_numpy(np.asarray(train_mask, dtype=bool)), torch.from_numpy(residual.logits(train_base).astype(np.float32)), torch.from_numpy(y_train.astype(np.int64))), batch_size=64, shuffle=True)
    val_seq_t = torch.from_numpy(np.asarray(val_seq, dtype=np.float32))
    val_mask_t = torch.from_numpy(np.asarray(val_mask, dtype=bool))
    val_base_t = torch.from_numpy(residual.logits(val_base).astype(np.float32))
    best = None
    patience = 0
    for epoch in range(1, 61):
        model.train()
        for seq, mask, base, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(seq.to(device), mask.to(device), base.to(device))
            combined_logits = output + base.to(device) if 'residual' in kind else output
            loss = criterion(combined_logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_output = model(val_seq_t, val_mask_t, val_base_t)
            val_logits = val_output + val_base_t if 'residual' in kind else val_output
        score = metrics(y_val, val_logits.numpy())
        key = (score['f1_macro'], score['accuracy'])
        if best is None or key > best[0]:
            best = (key, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
        if patience >= 10:
            break
    model.load_state_dict(best[2])
    model.eval()
    return (model, {'best_epoch': best[1], **{'validation_' + k: v for k, v in metrics(y_val, (model(val_seq_t, val_mask_t, val_base_t) + val_base_t if 'residual' in kind else model(val_seq_t, val_mask_t, val_base_t)).detach().numpy()).items()}})

def predict(model, kind, seq, mask, base):
    with torch.no_grad():
        seq_t = torch.from_numpy(np.asarray(seq, dtype=np.float32))
        mask_t = torch.from_numpy(np.asarray(mask, dtype=bool))
        base_t = torch.from_numpy(residual.logits(base).astype(np.float32))
        output = model(seq_t, mask_t, base_t)
        return (output + base_t if 'residual' in kind else output).numpy()

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, validation, test = residual.common.load_base_splits(False, 'natural')
    y_train, y_val, y_test = map(residual.labels, (train, validation, test))
    cache_root = OUTPUT / 'base_cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    base_train = residual.historical_train_probabilities(train, cache_root)
    base_val = residual.stored_probabilities(residual.REPRODUCTION / 'checkpoint_validation_predictions.csv', validation)
    base_test = residual.stored_probabilities(residual.REPRODUCTION / 'checkpoint_test_predictions.csv', test)
    train_seq, train_mask = build_sequences(train, 'train')
    val_seq, val_mask = build_sequences(validation, 'validation')
    test_seq, test_mask = build_sequences(test, 'test')
    candidates = []
    for kind in ('gru_direct', 'gru_residual', 'tcn_direct', 'tcn_residual'):
        model, details = train_model(kind, train_seq, train_mask, base_train, y_train, val_seq, val_mask, base_val, y_val, 42)
        val_logits = predict(model, kind, val_seq, val_mask, base_val)
        test_logits = predict(model, kind, test_seq, test_mask, base_test)
        row = {'kind': kind, **details, 'test_accuracy': metrics(y_test, test_logits)['accuracy'], 'test_f1_macro': metrics(y_test, test_logits)['f1_macro'], 'test_f1_weighted': metrics(y_test, test_logits)['f1_weighted']}
        candidates.append(row)
        torch.save(model.state_dict(), OUTPUT / f'{kind}.pt')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    baseline = metrics(y_test, residual.logits(base_test))
    selected = max(candidates, key=lambda row: (row['validation_f1_macro'], row['validation_accuracy']))
    result = {'historical_test': baseline, 'candidates': candidates, 'selected_by_validation': selected, 'delta_selected_accuracy': selected['test_accuracy'] - baseline['accuracy'], 'delta_selected_f1_macro': selected['test_f1_macro'] - baseline['f1_macro'], 'protocol': {'sequence_length': 200, 'input_dim': SEQ_DIM, 'selection': 'natural validation only', 'test_locked': True, 'software': {'torch': torch.__version__, 'numpy': np.__version__}}}
    write_json(OUTPUT / 'summary.json', result)
    pd.DataFrame(candidates).to_csv(OUTPUT / 'summary.csv', index=False)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
