"""Compare temporal neural architectures on the cached high-resolution YOLO data.

No video is decoded and YOLO is not executed.  Screening trains on external
folds 1--3, selects epoch/temperature on fold 4 using temporal-only metrics,
refits on folds 1--4, and reports fold 5.  Confirmation runs the selected
architecture through all five source-disjoint outer folds.
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
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
REPO = Path(__file__).resolve().parents[2]
HERE = experiment_directory(__file__)
DEFAULT_OUTPUT = HERE / 'outputs' / 'temporal_architecture_suite_20260814'
PAIRED = HERE / 'outputs' / 'quantile_100frames' / 'paired_predictions.npz'
PRIMITIVES = HERE / 'outputs' / 'cached_descriptor_grid_20260814' / 'primitive_cache' / 'current'
ARCHITECTURES = ('tcn_attention', 'transformer', 'tcn_bigru_attention', 'attention_mlp')
CLASS_WEIGHTS = (1.2, 0.65, 1.8)
VERSION = 1
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.multimodal import audit_performance as audit
from models.M3.run_temporal_attention_experiment import IndexedSequenceDataset, build_feature_cache, choose_temperature, fit_normalizer, make_loader, normalize_all, paired_test, probability_metrics, quality_score, softmax, temperature_probabilities, usable_mask

def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value

def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2), encoding='utf-8')

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

class QualityAttentionClassifier(nn.Module):
    """Shared quality-prior attention and classification head."""

    def __init__(self, width: int):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 1))
        self.quality_strength_raw = nn.Parameter(torch.tensor(0.5413))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 3))

    def classify(self, encoded, mask, quality, return_attention):
        scores = self.attention(encoded).squeeze(-1)
        strength = torch.nn.functional.softplus(self.quality_strength_raw)
        scores = scores + strength * torch.log(torch.clamp(quality, min=0.0001))
        scores = scores.masked_fill(~mask, -10000.0)
        attention = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * attention.unsqueeze(-1), dim=1)
        logits = self.head(pooled)
        return (logits, attention) if return_attention else logits

class ResidualTCNBlock(nn.Module):

    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation)
        self.dropout = nn.Dropout(0.15)
        self.norm = nn.LayerNorm(width)

    def forward(self, sequence):
        residual = sequence
        hidden = self.conv(sequence.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(torch.nn.functional.gelu(hidden))
        return self.norm(residual + hidden)

class TCNAttention(QualityAttentionClassifier):

    def __init__(self):
        super().__init__(96)
        self.projection = nn.Sequential(nn.Linear(90, 96), nn.GELU(), nn.Dropout(0.15))
        self.blocks = nn.ModuleList([ResidualTCNBlock(96, dilation) for dilation in (1, 2, 4, 8)])

    def forward(self, sequence, mask, quality, *, return_attention=False):
        hidden = self.projection(sequence) * mask.unsqueeze(-1).to(sequence.dtype)
        for block in self.blocks:
            hidden = block(hidden)
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return self.classify(hidden, mask, quality, return_attention)

class TemporalTransformer(QualityAttentionClassifier):

    def __init__(self):
        super().__init__(96)
        self.projection = nn.Sequential(nn.Linear(90, 96), nn.GELU(), nn.Dropout(0.15))
        self.position = nn.Parameter(torch.zeros(1, 100, 96))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=96, nhead=4, dim_feedforward=192, dropout=0.15, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(96)

    def forward(self, sequence, mask, quality, *, return_attention=False):
        hidden = self.projection(sequence) + self.position[:, :sequence.shape[1]]
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.norm(hidden) * mask.unsqueeze(-1).to(hidden.dtype)
        return self.classify(hidden, mask, quality, return_attention)

class TCNBiGRUAttention(QualityAttentionClassifier):

    def __init__(self):
        super().__init__(96)
        self.projection = nn.Sequential(nn.Linear(90, 64), nn.GELU(), nn.Dropout(0.15))
        self.blocks = nn.ModuleList([ResidualTCNBlock(64, dilation) for dilation in (1, 2, 4)])
        self.bigru = nn.GRU(64, 48, batch_first=True, bidirectional=True)

    def forward(self, sequence, mask, quality, *, return_attention=False):
        hidden = self.projection(sequence) * mask.unsqueeze(-1).to(sequence.dtype)
        for block in self.blocks:
            hidden = block(hidden)
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        hidden, _ = self.bigru(hidden)
        return self.classify(hidden, mask, quality, return_attention)

class AttentionMLP(QualityAttentionClassifier):

    def __init__(self):
        super().__init__(96)
        self.frame_encoder = nn.Sequential(nn.Linear(90, 96), nn.GELU(), nn.Dropout(0.15), nn.LayerNorm(96), nn.Linear(96, 96), nn.GELU())

    def forward(self, sequence, mask, quality, *, return_attention=False):
        hidden = self.frame_encoder(sequence) * mask.unsqueeze(-1).to(sequence.dtype)
        return self.classify(hidden, mask, quality, return_attention)

def build_model(architecture: str) -> nn.Module:
    models = {'tcn_attention': TCNAttention, 'transformer': TemporalTransformer, 'tcn_bigru_attention': TCNBiGRUAttention, 'attention_mlp': AttentionMLP}
    return models[architecture]()

def predict(model, dataset, indices, batch_size):
    loader = make_loader(dataset, indices, batch_size, False, 0)
    logits = np.empty((len(indices), 3), dtype=np.float32)
    attention = np.empty((len(indices), 100), dtype=np.float32)
    returned = np.empty(len(indices), dtype=np.int64)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for sequence, mask, quality, _, original_index in loader:
            output, weights = model(sequence, mask, quality, return_attention=True)
            size = len(sequence)
            logits[cursor:cursor + size] = output.numpy()
            attention[cursor:cursor + size] = weights.numpy()
            returned[cursor:cursor + size] = original_index.numpy()
            cursor += size
    return (logits, attention, returned)

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

def train_select(architecture, dataset, train_indices, validation_indices, labels, *, seed, batch_size, max_epochs, patience, initial_state=None):
    seed_all(seed)
    model = build_model(architecture)
    if initial_state is not None:
        model.load_state_dict(initial_state)
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
            raise AssertionError('validation order changed')
        metrics = probability_metrics(labels[validation_indices], softmax(logits))
        row = {'epoch': epoch, 'train_loss': loss, **metrics}
        history.append(row)
        key = (metrics['f1_macro'], metrics['balanced_accuracy'], metrics['accuracy'])
        if best is None or key > best['key']:
            best = {'key': key, 'epoch': epoch, 'state': copy.deepcopy(model.state_dict()), 'logits': logits.copy()}
            stale = 0
        else:
            stale += 1
        print(json.dumps({'architecture': architecture, 'stage': 'select', 'epoch': epoch, 'loss': loss, 'validation_f1': metrics['f1_macro'], 'best_epoch': best['epoch']}), flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best['state'])
    return (model, best, history)

def train_fixed(architecture, dataset, train_indices, *, seed, batch_size, epochs):
    seed_all(seed)
    model = build_model(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0002)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32))
    loader = make_loader(dataset, train_indices, batch_size, True, seed)
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion)
        history.append({'epoch': epoch, 'train_loss': loss})
        print(json.dumps({'architecture': architecture, 'stage': 'refit', 'epoch': epoch, 'epochs': epochs, 'loss': loss}), flush=True)
    return (model, history)

def load_data():
    feature_path = build_feature_cache()
    features = _load_model_result_array(feature_path, mmap_mode='r')
    frame_mask = _load_model_result_array(PRIMITIVES / 'mask.npy', mmap_mode='r')
    quality = _load_model_result_array(PRIMITIVES / 'quality.npy', mmap_mode='r')
    q_score = quality_score(np.asarray(quality))
    usable = usable_mask(frame_mask, q_score)
    with _load_model_result_array(PAIRED, allow_pickle=True) as paired:
        names = paired['names'].astype(str)
        labels = paired['labels'].astype(np.int64)
        historical = paired['historical'].astype(np.float32)
        static = paired['highres_auxiliary'].astype(np.float32)
    folds, external_historical, external_y, source = audit.load_external(names, labels)
    if not np.array_equal(labels, external_y) or not np.allclose(historical, external_historical, atol=2e-06):
        raise AssertionError('external protocol is not aligned')
    return (features, usable, q_score, names, labels, historical, static, folds, source)

def probabilities_for(logits, temperature, historical, static):
    temporal = temperature_probabilities(logits, temperature)
    fused = 0.4 * historical + 0.6 * (0.7 * static + 0.3 * temporal)
    official = 0.4 * historical + 0.6 * static
    return (temporal, fused, official)

def fuse_with_temporal_share(historical, static, temporal, temporal_share):
    """Blend a temporal branch inside the fixed 60% auxiliary budget."""
    return 0.4 * historical + 0.6 * ((1.0 - temporal_share) * static + temporal_share * temporal)

def select_temporal_share(labels, historical, static, temporal):
    """Select share on validation only, prioritising macro-F1.

    Ties are resolved by balanced accuracy, accuracy, and finally the smaller
    temporal share.  The grid covers 0--100% of the auxiliary budget in 5%
    increments, corresponding to 0--60% effective final temporal weight.
    """
    rows = []
    for share in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        probabilities = fuse_with_temporal_share(historical, static, temporal, float(share))
        metrics = probability_metrics(labels, probabilities)
        rows.append({'temporal_share_within_auxiliary': float(share), 'effective_final_temporal_weight': float(0.6 * share), 'accuracy': metrics['accuracy'], 'balanced_accuracy': metrics['balanced_accuracy'], 'f1_macro': metrics['f1_macro'], 'negative_log_likelihood': metrics['negative_log_likelihood'], 'brier_multiclass': metrics['brier_multiclass'], 'ece_15': metrics['ece_15']})
    selected = max(rows, key=lambda row: (row['f1_macro'], row['balanced_accuracy'], row['accuracy'], -row['temporal_share_within_auxiliary']))
    return (selected, rows)

def run_screen(args, data):
    features, usable, q_score, names, labels, historical, static, folds, source = data
    train_indices = np.concatenate([np.asarray(folds[index]['hold'], dtype=int) for index in (0, 1, 2)])
    validation_indices = np.asarray(folds[3]['hold'], dtype=int)
    test_indices = np.asarray(folds[4]['hold'], dtype=int)
    refit_indices = np.concatenate((train_indices, validation_indices))
    root = args.output / f'screening_seed_{args.seed}'
    rows = []
    summaries = {}
    for architecture in ARCHITECTURES:
        out = root / architecture
        summary_path = out / 'summary.json'
        predictions_path = out / 'predictions.npz'
        if args.resume and _resolve_model_artifact(summary_path).exists() and _resolve_model_artifact(predictions_path).exists():
            summary = json.loads(_resolve_model_artifact(summary_path).read_text(encoding='utf-8'))
            if summary.get('version') == VERSION:
                print(json.dumps({'stage': 'resume_screen', 'architecture': architecture}), flush=True)
                summaries[architecture] = summary
                rows.append(summary['ranking_row'])
                continue
        mean, std = fit_normalizer(features, train_indices, usable)
        normalized = normalize_all(features, mean, std, usable)
        dataset = IndexedSequenceDataset(normalized, usable, q_score, labels)
        _, best, selection_history = train_select(architecture, dataset, train_indices, validation_indices, labels, seed=args.seed, batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience)
        temperature, temperature_rows = choose_temperature(labels[validation_indices], best['logits'])
        val_temporal, val_fused, val_official = probabilities_for(best['logits'], temperature, historical[validation_indices], static[validation_indices])
        selected_weight, temporal_share_search = select_temporal_share(labels[validation_indices], historical[validation_indices], static[validation_indices], val_temporal)
        selected_share = selected_weight['temporal_share_within_auxiliary']
        val_selected_fused = fuse_with_temporal_share(historical[validation_indices], static[validation_indices], val_temporal, selected_share)
        del dataset, normalized
        refit_mean, refit_std = fit_normalizer(features, refit_indices, usable)
        normalized = normalize_all(features, refit_mean, refit_std, usable)
        dataset = IndexedSequenceDataset(normalized, usable, q_score, labels)
        model, refit_history = train_fixed(architecture, dataset, refit_indices, seed=args.seed + 1000, batch_size=args.batch_size, epochs=best['epoch'])
        test_logits, attention, returned = predict(model, dataset, test_indices, args.batch_size)
        if not np.array_equal(returned, test_indices):
            raise AssertionError('test order changed')
        test_temporal, test_fused, test_official = probabilities_for(test_logits, temperature, historical[test_indices], static[test_indices])
        test_selected_fused = fuse_with_temporal_share(historical[test_indices], static[test_indices], test_temporal, selected_share)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(predictions_path, indices=test_indices, names=names[test_indices], labels=labels[test_indices], temporal=test_temporal, fused=test_fused, selected_weight_fused=test_selected_fused, official=test_official, attention=attention)
        torch.save({'version': VERSION, 'architecture': architecture, 'state_dict': model.state_dict(), 'mean': refit_mean, 'std': refit_std, 'temperature': temperature, 'selected_epoch': best['epoch'], 'selected_temporal_share_within_auxiliary': selected_share}, out / 'model.pt')
        pd.DataFrame(selection_history).to_csv(out / 'selection_history.csv', index=False)
        pd.DataFrame(refit_history).to_csv(out / 'refit_history.csv', index=False)
        val_temporal_metrics = probability_metrics(labels[validation_indices], val_temporal)
        test_temporal_metrics = probability_metrics(labels[test_indices], test_temporal)
        test_fused_metrics = probability_metrics(labels[test_indices], test_fused)
        val_selected_metrics = probability_metrics(labels[validation_indices], val_selected_fused)
        test_selected_metrics = probability_metrics(labels[test_indices], test_selected_fused)
        ranking_row = {'architecture': architecture, 'parameters': sum((parameter.numel() for parameter in model.parameters())), 'selected_epoch': best['epoch'], 'temperature': temperature, 'validation_temporal_f1_macro': val_temporal_metrics['f1_macro'], 'validation_temporal_accuracy': val_temporal_metrics['accuracy'], 'test_temporal_f1_macro': test_temporal_metrics['f1_macro'], 'test_temporal_accuracy': test_temporal_metrics['accuracy'], 'test_fused_f1_macro': test_fused_metrics['f1_macro'], 'test_fused_accuracy': test_fused_metrics['accuracy'], 'selected_temporal_share_within_auxiliary': selected_share, 'selected_effective_final_temporal_weight': 0.6 * selected_share, 'validation_selected_fused_f1_macro': val_selected_metrics['f1_macro'], 'validation_selected_fused_balanced_accuracy': val_selected_metrics['balanced_accuracy'], 'validation_selected_fused_accuracy': val_selected_metrics['accuracy'], 'test_selected_fused_f1_macro': test_selected_metrics['f1_macro'], 'test_selected_fused_balanced_accuracy': test_selected_metrics['balanced_accuracy'], 'test_selected_fused_accuracy': test_selected_metrics['accuracy'], 'test_official_f1_macro': probability_metrics(labels[test_indices], test_official)['f1_macro'], 'test_official_accuracy': probability_metrics(labels[test_indices], test_official)['accuracy']}
        summary = {'version': VERSION, 'architecture': architecture, 'selection_rule': 'temporal-only validation macro-F1 on source-disjoint fold 4', 'selected_epoch': best['epoch'], 'temperature': temperature, 'temperature_search': temperature_rows, 'temporal_share_selection_rule': 'validation macro-F1, then balanced accuracy, accuracy, smaller share', 'temporal_share_search': temporal_share_search, 'selected_temporal_share_within_auxiliary': selected_share, 'selected_effective_final_temporal_weight': 0.6 * selected_share, 'validation': {'temporal': val_temporal_metrics, 'fixed_30_percent_fused': probability_metrics(labels[validation_indices], val_fused), 'selected_weight_fused': val_selected_metrics, 'official': probability_metrics(labels[validation_indices], val_official)}, 'test': {'temporal': test_temporal_metrics, 'fixed_30_percent_fused': test_fused_metrics, 'selected_weight_fused': test_selected_metrics, 'official': probability_metrics(labels[test_indices], test_official)}, 'ranking_row': ranking_row}
        dump_json(summary_path, summary)
        summaries[architecture] = summary
        rows.append(ranking_row)
        del dataset, normalized, model
    ranking = sorted(rows, key=lambda row: (row['validation_selected_fused_f1_macro'], row['validation_selected_fused_balanced_accuracy'], row['validation_selected_fused_accuracy']), reverse=True)
    pd.DataFrame(ranking).to_csv(root / 'ranking.csv', index=False)
    result = {'complete': True, 'selection_uses_test': False, 'selection_target': 'validation selected-weight fused macro-F1', 'selected_by_validation': ranking[0]['architecture'], 'ranking': ranking, 'summaries': summaries}
    dump_json(root / 'summary.json', result)
    print(json.dumps(safe(result), ensure_ascii=False, indent=2))

def run_confirm(args, data):
    if args.architecture not in ARCHITECTURES:
        raise ValueError('--architecture is required for confirmation')
    features, usable, q_score, names, labels, historical, static, folds, source = data
    root = args.output / 'confirmation' / args.architecture / f'seed_{args.seed}'
    root.mkdir(parents=True, exist_ok=True)
    temporal_oof = np.zeros((len(labels), 3), dtype=np.float32)
    fused_oof = np.zeros_like(temporal_oof)
    official_oof = 0.4 * historical + 0.6 * static
    fold_rows = []
    for fold_no in range(1, min(args.max_folds, 5) + 1):
        out = root / f'fold_{fold_no}'
        prediction_path = out / 'predictions.npz'
        summary_path = out / 'summary.json'
        hold = np.asarray(folds[fold_no - 1]['hold'], dtype=int)
        if args.resume and _resolve_model_artifact(prediction_path).exists() and _resolve_model_artifact(summary_path).exists():
            cached = _load_model_result_array(prediction_path, allow_pickle=True)
            if np.array_equal(cached['indices'], hold):
                temporal_oof[hold] = cached['temporal']
                fused_oof[hold] = cached['fused']
                fold_rows.append(json.loads(_resolve_model_artifact(summary_path).read_text(encoding='utf-8')))
                print(json.dumps({'stage': 'resume_confirm', 'fold': fold_no}), flush=True)
                continue
        validation_fold = fold_no % 5 + 1
        validation = np.asarray(folds[validation_fold - 1]['hold'], dtype=int)
        excluded = np.zeros(len(labels), dtype=bool)
        excluded[hold] = True
        excluded[validation] = True
        inner_train = np.flatnonzero(~excluded)
        outer_train = np.flatnonzero(~np.isin(np.arange(len(labels)), hold))
        mean, std = fit_normalizer(features, inner_train, usable)
        normalized = normalize_all(features, mean, std, usable)
        dataset = IndexedSequenceDataset(normalized, usable, q_score, labels)
        seed = args.seed * 100 + fold_no
        _, best, selection_history = train_select(args.architecture, dataset, inner_train, validation, labels, seed=seed, batch_size=args.batch_size, max_epochs=args.max_epochs, patience=args.patience)
        temperature, temperature_rows = choose_temperature(labels[validation], best['logits'])
        del dataset, normalized
        mean, std = fit_normalizer(features, outer_train, usable)
        normalized = normalize_all(features, mean, std, usable)
        dataset = IndexedSequenceDataset(normalized, usable, q_score, labels)
        model, refit_history = train_fixed(args.architecture, dataset, outer_train, seed=seed + 1000, batch_size=args.batch_size, epochs=best['epoch'])
        logits, attention, returned = predict(model, dataset, hold, args.batch_size)
        if not np.array_equal(returned, hold):
            raise AssertionError('outer hold order changed')
        temporal, fused, official = probabilities_for(logits, temperature, historical[hold], static[hold])
        temporal_oof[hold] = temporal
        fused_oof[hold] = fused
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, indices=hold, names=names[hold], labels=labels[hold], temporal=temporal, fused=fused, official=official, attention=attention)
        torch.save({'version': VERSION, 'architecture': args.architecture, 'state_dict': model.state_dict(), 'mean': mean, 'std': std, 'temperature': temperature, 'selected_epoch': best['epoch']}, out / 'model.pt')
        pd.DataFrame(selection_history).to_csv(out / 'selection_history.csv', index=False)
        pd.DataFrame(refit_history).to_csv(out / 'refit_history.csv', index=False)
        fold_summary = {'version': VERSION, 'fold': fold_no, 'validation_fold': validation_fold, 'selected_epoch': best['epoch'], 'temperature': temperature, 'temperature_search': temperature_rows, 'temporal': probability_metrics(labels[hold], temporal), 'fixed_30_percent_fused': probability_metrics(labels[hold], fused), 'official': probability_metrics(labels[hold], official)}
        dump_json(summary_path, fold_summary)
        fold_rows.append(fold_summary)
        del dataset, normalized, model
    if args.max_folds < 5:
        dump_json(root / 'partial_summary.json', {'complete': False, 'folds': fold_rows})
        return
    metrics = {'temporal': probability_metrics(labels, temporal_oof), 'fixed_30_percent_fused': probability_metrics(labels, fused_oof), 'official': probability_metrics(labels, official_oof)}
    result = {'complete': True, 'architecture': args.architecture, 'seed': args.seed, 'metrics': metrics, 'paired_against_official': paired_test(labels, official_oof, fused_oof, source, seed=args.seed + 20260814), 'folds': fold_rows}
    np.savez_compressed(root / 'oof_predictions.npz', names=names, labels=labels, source=source, temporal=temporal_oof, fused=fused_oof, official=official_oof)
    dump_json(root / 'summary.json', result)
    partial = root / 'partial_summary.json'
    if _resolve_model_artifact(partial).exists():
        partial.unlink()
    print(json.dumps(safe(result), ensure_ascii=False, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('screen', 'confirm'), default='screen')
    parser.add_argument('--architecture', choices=ARCHITECTURES)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max-folds', type=int, default=5)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
    data = load_data()
    if args.mode == 'screen':
        run_screen(args, data)
    else:
        run_confirm(args, data)
if __name__ == '__main__':
    main()
