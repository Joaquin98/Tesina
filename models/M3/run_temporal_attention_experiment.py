"""Leakage-safe temporal YOLO-pose attention experiment on high-resolution caches.

This experiment does not decode video and does not run YOLO.  It consumes the
100-step high-resolution primitive cache, trains one source-disjoint model per
external fold, and compares:

* the standalone temporal YOLO branch;
* the frozen 40/60 historical + temporal fusion;
* a pre-registered hybrid that averages static and temporal YOLO auxiliaries;
* the current frozen 40/60 historical + static high-resolution auxiliary.

For each outer fold, the next fold is used as an inner validation set to select
the epoch and a temperature using only temporal-branch NLL.  The model is then
reinitialized and fitted on all four non-test folds for the selected number of
epochs.  No outer-test prediction participates in selection.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
REPO = Path(__file__).resolve().parents[2]
HERE = experiment_directory(__file__)
PRIMITIVES = HERE / 'outputs' / 'cached_descriptor_grid_20260814' / 'primitive_cache' / 'current'
PAIRED = HERE / 'outputs' / 'quantile_100frames' / 'paired_predictions.npz'
DEFAULT_OUTPUT = HERE / 'outputs' / 'temporal_attention_highres_20260814'
SHARED_FEATURE_CACHE = HERE / 'outputs' / 'temporal_attention_sequence_cache_20260814' / 'features_e04_e06_quality.npy'
CLASS_NAMES = ('line', 'other', 'scrum')
CLASS_WEIGHTS = (1.2, 0.65, 1.8)
TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 2.0)
VERSION = 1
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.multimodal import audit_performance as audit

def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2), encoding='utf-8')

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def smooth_three(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Three-step median used by the promoted E04 descriptor."""
    padded = np.pad(values, ((0, 0), (1, 1), (0, 0)), mode='edge')
    smoothed = np.median(np.stack((padded[:, 0:-2], padded[:, 1:-1], padded[:, 2:]), axis=0), axis=0).astype(np.float32)
    smoothed[~mask] = values[~mask]
    return smoothed

def build_feature_cache(*, force: bool=False) -> Path:
    cache = SHARED_FEATURE_CACHE
    metadata_path = cache.with_suffix('.json')
    if _resolve_model_artifact(cache).exists() and _resolve_model_artifact(metadata_path).exists() and (not force):
        metadata = json.loads(_resolve_model_artifact(metadata_path).read_text(encoding='utf-8'))
        if metadata.get('complete') and metadata.get('shape') == [4082, 100, 90]:
            return cache
    current = _load_model_result_array(PRIMITIVES / 'current.npy', mmap_mode='r')
    rugby = _load_model_result_array(PRIMITIVES / 'rugby.npy', mmap_mode='r')
    quality = _load_model_result_array(PRIMITIVES / 'quality.npy', mmap_mode='r')
    mask = _load_model_result_array(PRIMITIVES / 'mask.npy', mmap_mode='r')
    expected = (len(current), current.shape[1], current.shape[2] + quality.shape[2] + rugby.shape[2])
    cache.parent.mkdir(parents=True, exist_ok=True)
    store = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float32, shape=expected)
    started = time.perf_counter()
    for start in range(0, len(current), 256):
        stop = min(start + 256, len(current))
        batch_mask = np.asarray(mask[start:stop], dtype=bool)
        left = smooth_three(np.asarray(current[start:stop], dtype=np.float32), batch_mask)
        right = smooth_three(np.asarray(rugby[start:stop], dtype=np.float32), batch_mask)
        batch = np.concatenate((left, np.asarray(quality[start:stop], dtype=np.float32), right), axis=2)
        store[start:stop] = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
        print(json.dumps({'stage': 'feature_cache', 'clips': stop}), flush=True)
    store.flush()
    write_json(metadata_path, {'complete': True, 'shape': list(expected), 'dtype': 'float32', 'components': ['E04_smoothed_current_61', 'quality_4', 'E06_smoothed_rugby_25'], 'source': str(PRIMITIVES.relative_to(REPO)), 'video_decoded': False, 'yolo_executed': False, 'seconds': time.perf_counter() - started})
    return cache

def quality_score(quality: np.ndarray) -> np.ndarray:
    """Continuous attention prior: detection count, person and pose confidence."""
    count_factor = np.clip(quality[:, :, 0] / 0.25, 0.0, 1.0)
    confidence = np.sqrt(np.clip(quality[:, :, 1] * quality[:, :, 2], 0.0, 1.0))
    return (count_factor * confidence).astype(np.float32)

def usable_mask(frame_mask: np.ndarray, score: np.ndarray) -> np.ndarray:
    usable = np.asarray(frame_mask, dtype=bool) & (score > 0.0)
    fallback = ~usable.any(axis=1)
    if fallback.any():
        usable[fallback] = np.asarray(frame_mask[fallback], dtype=bool)
    still_empty = ~usable.any(axis=1)
    if still_empty.any():
        usable[still_empty, 0] = True
    return usable

def fit_normalizer(features: np.ndarray, indices: np.ndarray, usable: np.ndarray):
    total = np.zeros(features.shape[2], dtype=np.float64)
    square = np.zeros(features.shape[2], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), 256):
        ix = indices[start:start + 256]
        batch = np.asarray(features[ix], dtype=np.float32)
        valid = np.asarray(usable[ix], dtype=bool)
        rows = batch[valid]
        total += rows.sum(axis=0, dtype=np.float64)
        square += np.square(rows, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += len(rows)
    mean = total / max(count, 1)
    variance = np.maximum(square / max(count, 1) - mean * mean, 1e-06)
    std = np.sqrt(variance)
    return (mean.astype(np.float32), std.astype(np.float32))

def normalize_all(features: np.ndarray, mean: np.ndarray, std: np.ndarray, usable: np.ndarray):
    normalized = np.empty(features.shape, dtype=np.float32)
    for start in range(0, len(features), 256):
        stop = min(start + 256, len(features))
        batch = (np.asarray(features[start:stop], dtype=np.float32) - mean) / std
        batch = np.clip(batch, -8.0, 8.0)
        batch[~usable[start:stop]] = 0.0
        normalized[start:stop] = batch
    return normalized

class IndexedSequenceDataset(Dataset):

    def __init__(self, x, mask, quality, labels):
        self.x = torch.from_numpy(x)
        self.mask = torch.from_numpy(mask)
        self.quality = torch.from_numpy(quality)
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.x[index], self.mask[index], self.quality[index], self.labels[index], index)

class QualityMaskedTemporalAttention(nn.Module):

    def __init__(self, input_dim: int=90, projection: int=96, hidden: int=64):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dim, projection), nn.GELU(), nn.Dropout(0.15))
        self.temporal = nn.GRU(projection, hidden, num_layers=1, batch_first=True, bidirectional=True)
        width = hidden * 2
        self.attention = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 1))
        self.quality_strength_raw = nn.Parameter(torch.tensor(0.5413))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 3))

    def forward(self, sequence, mask, quality, *, return_attention: bool=False):
        embedded = self.projection(sequence)
        embedded = embedded * mask.unsqueeze(-1).to(embedded.dtype)
        encoded, _ = self.temporal(embedded)
        scores = self.attention(encoded).squeeze(-1)
        strength = torch.nn.functional.softplus(self.quality_strength_raw)
        scores = scores + strength * torch.log(torch.clamp(quality, min=0.0001))
        scores = scores.masked_fill(~mask, -10000.0)
        attention = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        logits = self.head(pooled)
        if return_attention:
            return (logits, attention)
        return logits

def make_loader(dataset, indices, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(Subset(dataset, np.asarray(indices, dtype=int).tolist()), batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)

def predict(model, dataset, indices, batch_size):
    loader = make_loader(dataset, indices, batch_size, False, 0)
    logits = np.empty((len(indices), 3), dtype=np.float32)
    attention = np.empty((len(indices), 100), dtype=np.float32)
    returned_indices = np.empty(len(indices), dtype=np.int64)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for sequence, mask, quality, _, original_index in loader:
            output, weights = model(sequence, mask, quality, return_attention=True)
            size = len(sequence)
            logits[cursor:cursor + size] = output.numpy()
            attention[cursor:cursor + size] = weights.numpy()
            returned_indices[cursor:cursor + size] = original_index.numpy()
            cursor += size
    return (logits, attention, returned_indices)

def probability_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    probabilities = np.clip(probabilities.astype(np.float64), 1e-07, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=[0, 1, 2], zero_division=0)
    one_hot = np.eye(3, dtype=np.float64)[y]
    confidence = probabilities.max(axis=1)
    correct = prediction == y
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, 16)[:-1]:
        upper = lower + 1.0 / 15.0
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return {'accuracy': float(accuracy_score(y, prediction)), 'balanced_accuracy': float(balanced_accuracy_score(y, prediction)), 'f1_macro': float(f1_score(y, prediction, average='macro', zero_division=0)), 'negative_log_likelihood': float(log_loss(y, probabilities, labels=[0, 1, 2])), 'brier_multiclass': float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))), 'ece_15': float(ece), 'per_class': {CLASS_NAMES[i]: {'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'support': int(support[i])} for i in range(3)}, 'confusion_matrix': confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist()}

def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)

def temperature_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / float(temperature)).astype(np.float32)

def choose_temperature(y: np.ndarray, logits: np.ndarray):
    rows = []
    for temperature in TEMPERATURES:
        probabilities = temperature_probabilities(logits, temperature)
        rows.append({'temperature': temperature, 'negative_log_likelihood': float(log_loss(y, probabilities, labels=[0, 1, 2])), 'f1_macro': float(f1_score(y, probabilities.argmax(1), average='macro', zero_division=0))})
    selected = min(rows, key=lambda row: (row['negative_log_likelihood'], abs(row['temperature'] - 1.0)))
    return (selected['temperature'], rows)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for sequence, mask, quality, labels, _ in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(sequence, mask, quality)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))

def fit_with_validation(dataset, train_indices, validation_indices, labels, *, seed, batch_size, max_epochs, patience):
    seed_everything(seed)
    model = QualityMaskedTemporalAttention()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0002)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32))
    loader = make_loader(dataset, train_indices, batch_size, True, seed)
    best = None
    history = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion)
        logits, _, returned = predict(model, dataset, validation_indices, batch_size)
        if not np.array_equal(returned, validation_indices):
            raise AssertionError('validation prediction order changed')
        probabilities = softmax(logits)
        score = probability_metrics(labels[validation_indices], probabilities)
        history.append({'epoch': epoch, 'train_loss': loss, **score})
        key = (score['f1_macro'], score['balanced_accuracy'], score['accuracy'])
        if best is None or key > best['key']:
            best = {'key': key, 'epoch': epoch, 'state': copy.deepcopy(model.state_dict()), 'logits': logits.copy()}
            stale = 0
        else:
            stale += 1
        print(json.dumps({'stage': 'inner_training', 'epoch': epoch, 'loss': loss, 'validation_f1_macro': score['f1_macro'], 'best_epoch': best['epoch']}), flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best['state'])
    return (model, best, history)

def fit_fixed_epochs(dataset, train_indices, *, seed, batch_size, epochs):
    seed_everything(seed)
    model = QualityMaskedTemporalAttention()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0002)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32))
    loader = make_loader(dataset, train_indices, batch_size, True, seed)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion)
        history.append({'epoch': epoch, 'train_loss': loss})
        print(json.dumps({'stage': 'outer_refit', 'epoch': epoch, 'epochs': epochs, 'loss': loss}), flush=True)
    return (model, history)

def attention_diagnostics(attention, quality, usable):
    entropies = []
    weighted_quality = []
    correlations = []
    for weights, scores, valid in zip(attention, quality, usable):
        selected_weights = weights[valid]
        selected_scores = scores[valid]
        if len(selected_weights) == 0:
            continue
        selected_weights = selected_weights / max(selected_weights.sum(), 1e-08)
        entropy = -np.sum(selected_weights * np.log(np.clip(selected_weights, 1e-08, 1.0)))
        entropies.append(float(entropy / max(math.log(max(len(selected_weights), 2)), 1e-08)))
        weighted_quality.append(float(np.sum(selected_weights * selected_scores)))
        if len(selected_weights) > 2:
            centered_weights = selected_weights - selected_weights.mean()
            centered_scores = selected_scores - selected_scores.mean()
            denominator = float(np.sqrt(np.sum(centered_weights ** 2) * np.sum(centered_scores ** 2)))
            if denominator > 1e-10:
                correlations.append(float(np.sum(centered_weights * centered_scores) / denominator))
    return {'normalized_attention_entropy_mean': float(np.mean(entropies)), 'quality_at_attended_frames_mean': float(np.mean(weighted_quality)), 'attention_quality_correlation_mean': float(np.mean(correlations)) if correlations else None}

def paired_test(y, reference_probabilities, candidate_probabilities, source, *, seed=20260814):
    reference = reference_probabilities.argmax(1)
    candidate = candidate_probabilities.argmax(1)
    reference_correct = reference == y
    candidate_correct = candidate == y
    candidate_only = int(np.sum(candidate_correct & ~reference_correct))
    reference_only = int(np.sum(reference_correct & ~candidate_correct))
    total = candidate_only + reference_only
    p_value = float(binomtest(min(candidate_only, reference_only), total, 0.5).pvalue) if total else 1.0
    delta = candidate_correct.astype(np.float64) - reference_correct.astype(np.float64)
    groups = sorted(set(source))
    group_sizes = np.asarray([np.sum(source == group) for group in groups], dtype=np.float64)
    group_delta = np.asarray([np.mean(delta[source == group]) for group in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(10000, dtype=np.float64)
    for index in range(len(bootstrap)):
        sample = rng.integers(0, len(groups), len(groups))
        bootstrap[index] = np.sum(group_sizes[sample] * group_delta[sample]) / np.sum(group_sizes[sample])
    return {'candidate_only_correct': candidate_only, 'reference_only_correct': reference_only, 'mcnemar_exact_p': p_value, 'delta_accuracy': float(delta.mean()), 'source_group_bootstrap_delta_accuracy_ci95': np.quantile(bootstrap, [0.025, 0.975]).tolist()}

def run_fold(fold_no, folds, features, frame_mask, quality, q_score, labels, names, historical, static_auxiliary, output, *, batch_size, max_epochs, patience, resume):
    fold_dir = output / f'fold_{fold_no}'
    result_path = fold_dir / 'predictions.npz'
    summary_path = fold_dir / 'summary.json'
    hold = np.asarray(folds[fold_no - 1]['hold'], dtype=np.int64)
    if resume and _resolve_model_artifact(result_path).exists() and _resolve_model_artifact(summary_path).exists():
        cached = _load_model_result_array(result_path, allow_pickle=True)
        summary = json.loads(_resolve_model_artifact(summary_path).read_text(encoding='utf-8'))
        if summary.get('version') == VERSION and np.array_equal(cached['indices'], hold):
            repaired_usable = usable_mask(frame_mask, q_score)
            summary['attention'] = attention_diagnostics(cached['attention'], q_score[hold], repaired_usable[hold])
            write_json(summary_path, summary)
            print(json.dumps({'stage': 'resume_fold', 'fold': fold_no}), flush=True)
            return (summary, {key: cached[key] for key in cached.files})
    validation_fold = fold_no % len(folds) + 1
    validation = np.asarray(folds[validation_fold - 1]['hold'], dtype=np.int64)
    forbidden = np.zeros(len(labels), dtype=bool)
    forbidden[hold] = True
    forbidden[validation] = True
    inner_train = np.flatnonzero(~forbidden)
    outer_train_mask = np.ones(len(labels), dtype=bool)
    outer_train_mask[hold] = False
    outer_train = np.flatnonzero(outer_train_mask)
    if set(hold).intersection(inner_train) or set(hold).intersection(validation):
        raise AssertionError('outer fold leaked into inner split')
    usable = usable_mask(frame_mask, q_score)
    inner_mean, inner_std = fit_normalizer(features, inner_train, usable)
    inner_x = normalize_all(features, inner_mean, inner_std, usable)
    inner_dataset = IndexedSequenceDataset(inner_x, usable, q_score, labels)
    seed = 42000 + fold_no
    print(json.dumps({'stage': 'fold_start', 'fold': fold_no, 'inner_train': len(inner_train), 'inner_validation': len(validation), 'outer_test': len(hold)}), flush=True)
    _, best, inner_history = fit_with_validation(inner_dataset, inner_train, validation, labels, seed=seed, batch_size=batch_size, max_epochs=max_epochs, patience=patience)
    temperature, temperature_rows = choose_temperature(labels[validation], best['logits'])
    del inner_dataset, inner_x
    outer_mean, outer_std = fit_normalizer(features, outer_train, usable)
    outer_x = normalize_all(features, outer_mean, outer_std, usable)
    outer_dataset = IndexedSequenceDataset(outer_x, usable, q_score, labels)
    model, outer_history = fit_fixed_epochs(outer_dataset, outer_train, seed=seed + 1000, batch_size=batch_size, epochs=best['epoch'])
    logits, attention, returned = predict(model, outer_dataset, hold, batch_size)
    if not np.array_equal(returned, hold):
        raise AssertionError('outer prediction order changed')
    temporal = temperature_probabilities(logits, temperature)
    historical_temporal = 0.4 * historical[hold] + 0.6 * temporal
    hybrid_auxiliary = 0.5 * static_auxiliary[hold] + 0.5 * temporal
    historical_hybrid = 0.4 * historical[hold] + 0.6 * hybrid_auxiliary
    official = 0.4 * historical[hold] + 0.6 * static_auxiliary[hold]
    fold_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(result_path, indices=hold, names=names[hold], labels=labels[hold], logits=logits, temporal=temporal, historical_temporal=historical_temporal, historical_hybrid=historical_hybrid, official=official, attention=attention)
    torch.save({'version': VERSION, 'fold': fold_no, 'state_dict': model.state_dict(), 'normalization_mean': outer_mean, 'normalization_std': outer_std, 'temperature': temperature, 'architecture': {'input_dim': 90, 'projection': 96, 'bigru_hidden_per_direction': 64, 'attention_hidden': 64, 'head_hidden': 64}}, fold_dir / 'model.pt')
    pd.DataFrame(inner_history).to_csv(fold_dir / 'inner_history.csv', index=False)
    pd.DataFrame(outer_history).to_csv(fold_dir / 'outer_refit_history.csv', index=False)
    diagnostics = attention_diagnostics(attention, q_score[hold], usable[hold])
    summary = {'version': VERSION, 'fold': fold_no, 'inner_validation_fold': validation_fold, 'sizes': {'inner_train': len(inner_train), 'inner_validation': len(validation), 'outer_refit': len(outer_train), 'outer_test': len(hold)}, 'selected_epoch': best['epoch'], 'temperature': temperature, 'temperature_search': temperature_rows, 'learned_quality_strength': float(torch.nn.functional.softplus(model.quality_strength_raw).detach()), 'metrics': {'temporal_yolo': probability_metrics(labels[hold], temporal), 'historical_temporal_40_60': probability_metrics(labels[hold], historical_temporal), 'historical_hybrid_40_60': probability_metrics(labels[hold], historical_hybrid), 'official_static_40_60': probability_metrics(labels[hold], official)}, 'attention': diagnostics}
    write_json(summary_path, summary)
    del outer_dataset, outer_x, model
    return (summary, {'indices': hold, 'names': names[hold], 'labels': labels[hold], 'logits': logits, 'temporal': temporal, 'historical_temporal': historical_temporal, 'historical_hybrid': historical_hybrid, 'official': official, 'attention': attention})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-epochs', type=int, default=35)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--max-folds', type=int, default=5)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--rebuild-feature-cache', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 2)
        args.max_folds = 1
        if args.output == DEFAULT_OUTPUT:
            args.output = HERE / 'outputs' / 'temporal_attention_highres_smoke'
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
    cache_path = build_feature_cache(force=args.rebuild_feature_cache)
    features = _load_model_result_array(cache_path, mmap_mode='r')
    frame_mask = _load_model_result_array(PRIMITIVES / 'mask.npy', mmap_mode='r')
    quality = _load_model_result_array(PRIMITIVES / 'quality.npy', mmap_mode='r')
    q_score = quality_score(np.asarray(quality))
    with _load_model_result_array(PAIRED, allow_pickle=True) as paired:
        labels = paired['labels'].astype(np.int64)
        names = paired['names'].astype(str)
        historical = paired['historical'].astype(np.float32)
        static_auxiliary = paired['highres_auxiliary'].astype(np.float32)
    folds, external_historical, external_labels, source = audit.load_external(names, labels)
    if not np.array_equal(labels, external_labels):
        raise AssertionError('external labels are not aligned')
    if not np.allclose(historical, external_historical, atol=2e-06):
        raise AssertionError('historical OOF probabilities are not aligned')
    if features.shape != (len(labels), 100, 90):
        raise AssertionError(f'unexpected temporal feature shape: {features.shape}')
    config = {'version': VERSION, 'experiment': 'high-resolution temporal YOLO-pose branch with quality-masked attention', 'architecture': 'Dense(96)-BiGRU(64x2)-quality-masked additive attention-Dense(64)-softmax(3)', 'inputs': ['E04 smoothed position+pose', 'four quality channels', 'E06 rugby descriptors'], 'sequence_length': 100, 'input_dim': 90, 'optimizer': {'name': 'AdamW', 'learning_rate': 0.0005, 'weight_decay': 0.0002}, 'class_weights': dict(zip(CLASS_NAMES, CLASS_WEIGHTS)), 'outer_protocol': 'five source-disjoint external folds', 'inner_protocol': 'next external fold validates epoch and temporal-only NLL temperature', 'fusion': {'historical_temporal': '0.4 historical + 0.6 temporal', 'hybrid': '0.4 historical + 0.6 * (0.5 static auxiliary + 0.5 temporal)', 'weights_tuned': False}, 'max_epochs': args.max_epochs, 'patience': args.patience, 'batch_size': args.batch_size, 'video_decoded': False, 'yolo_executed': False, 'software': {'torch': torch.__version__, 'numpy': np.__version__}}
    write_json(args.output / 'config.json', config)
    fold_summaries = []
    fold_results = []
    started = time.perf_counter()
    for fold_no in range(1, min(args.max_folds, 5) + 1):
        summary, result = run_fold(fold_no, folds, features, frame_mask, quality, q_score, labels, names, historical, static_auxiliary, args.output, batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience, resume=not args.no_resume)
        fold_summaries.append(summary)
        fold_results.append(result)
    if len(fold_results) < 5:
        write_json(args.output / 'partial_summary.json', {'complete': False, 'folds': fold_summaries, 'seconds': time.perf_counter() - started})
        print(json.dumps({'complete': False, 'folds': len(fold_results)}, indent=2), flush=True)
        return
    temporal = np.zeros((len(labels), 3), dtype=np.float32)
    historical_temporal = np.zeros_like(temporal)
    historical_hybrid = np.zeros_like(temporal)
    official = np.zeros_like(temporal)
    attention = np.zeros((len(labels), 100), dtype=np.float32)
    for result in fold_results:
        ix = result['indices']
        temporal[ix] = result['temporal']
        historical_temporal[ix] = result['historical_temporal']
        historical_hybrid[ix] = result['historical_hybrid']
        official[ix] = result['official']
        attention[ix] = result['attention']
    models = {'historical_only': historical, 'static_auxiliary_only': static_auxiliary, 'temporal_yolo_only': temporal, 'historical_temporal_40_60': historical_temporal, 'historical_hybrid_40_60': historical_hybrid, 'official_static_40_60': official}
    metrics = {name: probability_metrics(labels, probabilities) for name, probabilities in models.items()}
    comparisons = {name: paired_test(labels, official, probabilities, source, seed=20260814 + index) for index, (name, probabilities) in enumerate(models.items()) if name != 'official_static_40_60'}
    temporal_share_rows = []
    for temporal_share in np.arange(0.0, 1.0001, 0.05):
        auxiliary = (1.0 - temporal_share) * static_auxiliary + temporal_share * temporal
        probabilities = 0.4 * historical + 0.6 * auxiliary
        row_metrics = probability_metrics(labels, probabilities)
        temporal_share_rows.append({'temporal_share_within_auxiliary': round(float(temporal_share), 2), 'accuracy': row_metrics['accuracy'], 'balanced_accuracy': row_metrics['balanced_accuracy'], 'f1_macro': row_metrics['f1_macro'], 'line_recall': row_metrics['per_class']['line']['recall'], 'other_recall': row_metrics['per_class']['other']['recall'], 'scrum_recall': row_metrics['per_class']['scrum']['recall'], 'brier_multiclass': row_metrics['brier_multiclass'], 'ece_15': row_metrics['ece_15']})
    pd.DataFrame(temporal_share_rows).to_csv(args.output / 'posthoc_temporal_share_sweep.csv', index=False)
    exploratory_best = max(temporal_share_rows, key=lambda row: (row['f1_macro'], row['balanced_accuracy'], row['accuracy']))
    exploratory_share = exploratory_best['temporal_share_within_auxiliary']
    exploratory_probabilities = 0.4 * historical + 0.6 * ((1.0 - exploratory_share) * static_auxiliary + exploratory_share * temporal)
    peak = attention.argmax(axis=1)
    prediction_frame = pd.DataFrame({'video_name': names, 'true_label': [CLASS_NAMES[value] for value in labels], 'fold': np.concatenate([np.full(len(fold['hold']), fold['fold'], dtype=int) for fold in folds])[np.argsort(np.concatenate([fold['hold'] for fold in folds]))], 'temporal_prediction': [CLASS_NAMES[value] for value in temporal.argmax(1)], 'official_prediction': [CLASS_NAMES[value] for value in official.argmax(1)], 'hybrid_prediction': [CLASS_NAMES[value] for value in historical_hybrid.argmax(1)], 'attention_peak_step': peak, 'attention_peak_fraction': peak / 99.0, 'attention_peak_quality': q_score[np.arange(len(labels)), peak]})
    for model_name, probabilities in models.items():
        for class_index, class_name in enumerate(CLASS_NAMES):
            prediction_frame[f'{model_name}_prob_{class_name}'] = probabilities[:, class_index]
    prediction_frame.to_csv(args.output / 'oof_predictions.csv', index=False, quoting=csv.QUOTE_MINIMAL)
    np.savez_compressed(args.output / 'oof_predictions.npz', names=names, labels=labels, historical=historical, static_auxiliary=static_auxiliary, temporal=temporal, historical_temporal=historical_temporal, historical_hybrid=historical_hybrid, official=official, attention=attention, source=source)
    ranking = sorted(({'model': name, 'accuracy': value['accuracy'], 'balanced_accuracy': value['balanced_accuracy'], 'f1_macro': value['f1_macro'], 'brier_multiclass': value['brier_multiclass'], 'ece_15': value['ece_15']} for name, value in metrics.items()), key=lambda row: (row['f1_macro'], row['balanced_accuracy'], row['accuracy']), reverse=True)
    pd.DataFrame(ranking).to_csv(args.output / 'ranking.csv', index=False)
    final_summary = {'complete': True, 'version': VERSION, 'aggregation_invocation_seconds': time.perf_counter() - started, 'clips': len(labels), 'source_groups': len(set(source)), 'folds': fold_summaries, 'metrics': metrics, 'ranking': ranking, 'paired_against_official': comparisons, 'posthoc_temporal_share_sweep': {'status': 'exploratory_only_not_independently_selected', 'best_by_full_oof_macro_f1': exploratory_best, 'paired_against_official': paired_test(labels, official, exploratory_probabilities, source, seed=20260830), 'csv': 'posthoc_temporal_share_sweep.csv'}, 'claim_policy': 'Promote only if source-group bootstrap excludes zero and class behavior remains acceptable; otherwise retain the official model and treat this as exploratory evidence.'}
    partial_path = args.output / 'partial_summary.json'
    if _resolve_model_artifact(partial_path).exists():
        partial_path.unlink()
    write_json(args.output / 'summary.json', final_summary)
    print(json.dumps(json_safe(final_summary), ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
