"""M3 training and adaptation for the comparative protocol."""
from __future__ import annotations
from models.training_common import *
from models.M0.comparison_training import fit_m0
from models.M2.comparison_training import _static_head, fit_predict_m2

def _temporal_arrays(frame: pd.DataFrame):
    values = [compact_features(row, ('temporal', 'temporal_mask', 'quality')) for row in frame.itertuples(index=False)]
    return (np.asarray([item['temporal'] for item in values], dtype=np.float32), np.asarray([item['temporal_mask'] for item in values], dtype=bool), np.asarray([item['quality'] for item in values], dtype=np.float32))

def _softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)

def _predict_temporal_arrays(x, mask, quality, *, seed=42, batch_size=64):
    import torch
    from models.M3.run_temporal_architecture_suite import build_model
    checkpoint = torch.load(_resolve_model_artifact(M3_TEMPORAL / f'seed_{seed}.pt'), map_location='cpu', weights_only=False)
    mean = np.asarray(checkpoint['mean'], dtype=np.float32)
    std = np.asarray(checkpoint['std'], dtype=np.float32)
    normalized = np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)
    normalized[~mask] = 0.0
    model = build_model(str(checkpoint['architecture']))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(len(x), start + batch_size)
            logits = model(torch.from_numpy(normalized[start:stop]), torch.from_numpy(mask[start:stop]), torch.from_numpy(quality[start:stop])).numpy()
            output.append(_softmax(logits / float(checkpoint['temperature'])))
    return (normalize_probabilities(np.concatenate(output)), M3_TEMPORAL / f'seed_{seed}.pt')

def _predict_temporal_checkpoint(frame, *, seed=42, batch_size=64):
    return _predict_temporal_arrays(*_temporal_arrays(frame), seed=seed, batch_size=batch_size)

def _predict_temporal_ensemble(frame, *, batch_size=64):
    keys = frame['block_id'].astype(str).tolist()
    if len(set(keys)) != len(keys):
        raise ValueError('Temporal ensemble frame contains duplicate block_id values')
    missing_positions = [index for index, key in enumerate(keys) if key not in _TEMPORAL_ENSEMBLE_CACHE]
    paths = [M3_TEMPORAL / f'seed_{seed}.pt' for seed in (17, 42, 2026)]
    if missing_positions:
        missing = frame.iloc[missing_positions]
        arrays = _temporal_arrays(missing)
        predictions = []
        for seed in (17, 42, 2026):
            probability, _path = _predict_temporal_arrays(*arrays, seed=seed, batch_size=batch_size)
            predictions.append(probability)
        probabilities = normalize_probabilities(np.mean(predictions, axis=0))
        for position, probability in zip(missing_positions, probabilities, strict=True):
            _TEMPORAL_ENSEMBLE_CACHE[keys[position]] = probability.copy()
    return (np.asarray([_TEMPORAL_ENSEMBLE_CACHE[key] for key in keys], dtype=np.float64), paths)

def _fit_temporal(train, development, test, *, seed, run_dir, epochs, patience, batch_size):
    import torch
    from models.M3.run_temporal_attention_experiment import IndexedSequenceDataset
    from models.M3.run_temporal_architecture_suite import predict, train_select
    train_x, train_mask, train_quality = _temporal_arrays(train)
    dev_x, dev_mask, dev_quality = _temporal_arrays(development)
    test_x, test_mask, test_quality = _temporal_arrays(test)
    valid_frames = train_x[train_mask]
    mean = valid_frames.mean(axis=0).astype(np.float32)
    std = np.maximum(valid_frames.std(axis=0), 1e-05).astype(np.float32)
    all_x = np.concatenate([train_x, dev_x, test_x])
    all_mask = np.concatenate([train_mask, dev_mask, test_mask])
    all_quality = np.concatenate([train_quality, dev_quality, test_quality])
    all_y = np.concatenate([labels_for(train), labels_for(development), labels_for(test)])
    all_x = np.clip((all_x - mean) / std, -8.0, 8.0).astype(np.float32)
    all_x[~all_mask] = 0.0
    dataset = IndexedSequenceDataset(all_x, all_mask, all_quality, all_y)
    train_index = np.arange(len(train))
    dev_index = np.arange(len(train), len(train) + len(development))
    test_index = np.arange(len(train) + len(development), len(all_y))
    model, best, history = train_select('tcn_bigru_attention', dataset, train_index, dev_index, all_y, seed=seed, batch_size=batch_size, max_epochs=epochs, patience=patience)
    probabilities = []
    for expected in (train_index, dev_index, test_index):
        logits, _attention, returned = predict(model, dataset, expected, batch_size)
        if not np.array_equal(returned, expected):
            raise AssertionError('Temporal prediction order changed')
        probabilities.append(normalize_probabilities(_softmax(logits)))
    path = run_dir / 'temporal.pt'
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'architecture': 'tcn_bigru_attention', 'state_dict': model.state_dict(), 'mean': mean, 'std': std, 'temperature': 1.0, 'seed': seed, 'best_epoch': int(best['epoch']), 'history': history}, path)
    return (*probabilities, path)

def fit_predict_m3_or_m4(model_name, train, development, test, *, level, seed, run_dir, epochs, patience, batch_size) -> PredictionBundle:
    base_path = M4_PATH if model_name == 'M4' else M0_PATH
    if model_name == 'M4' and level == 'H1':
        static_train, static_dev, static_test, artifacts = _static_head(train, development, test, level='B', seed=seed, run_dir=run_dir / 'static')
        historical_path = run_dir / 'historical_h1.keras'
        historical_model = fit_m0(train, development, level='H1', seed=seed, output=historical_path, epochs=epochs, patience=patience, batch_size=batch_size, base_path=base_path)
        historical_train = _predict_keras(historical_model, train, batch_size=batch_size)
        historical_dev = _predict_keras(historical_model, development, batch_size=batch_size)
        historical_test = _predict_keras(historical_model, test, batch_size=batch_size)
        alpha = 0.55
        static = PredictionBundle(probabilities=normalize_probabilities(alpha * historical_test + (1.0 - alpha) * static_test), development_probabilities=normalize_probabilities(alpha * historical_dev + (1.0 - alpha) * static_dev), training_probabilities=normalize_probabilities(alpha * historical_train + (1.0 - alpha) * static_train), branches={'historical': historical_test, 'static': static_test}, development_branches={'historical': historical_dev, 'static': static_dev}, training_branches={'historical': historical_train, 'static': static_train}, artifacts=[*artifacts, historical_path], metadata={'static_alpha': 1.0 - alpha, 'historical_alpha': alpha})
    else:
        static_level = 'B' if level in {'B', 'H0', 'H1'} else 'H2'
        static = fit_predict_m2(train, development, test, level=static_level, seed=seed, run_dir=run_dir / 'static', epochs=epochs, patience=patience, batch_size=batch_size, base_path=base_path)
    if level in {'B', 'H0'}:
        temporal_test, temporal_paths = _predict_temporal_ensemble(test, batch_size=batch_size)
        temporal_dev, _ = _predict_temporal_ensemble(development, batch_size=batch_size)
        temporal_train = None if level == 'B' else _predict_temporal_ensemble(train, batch_size=batch_size)[0]
    else:
        temporal_train, temporal_dev, temporal_test, temporal_path = _fit_temporal(train, development, test, seed=seed, run_dir=run_dir, epochs=epochs, patience=patience, batch_size=batch_size)
        temporal_paths = [temporal_path]
    official = static.probabilities
    official_dev = static.development_probabilities
    official_train = static.training_probabilities
    return PredictionBundle(probabilities=official, development_probabilities=official_dev, training_probabilities=official_train, branches={**static.branches, 'official': official, 'temporal': temporal_test}, development_branches={**static.development_branches, 'official': official_dev, 'temporal': temporal_dev}, training_branches={} if level == 'B' else {**static.training_branches, 'official': official_train, 'temporal': temporal_train}, artifacts=[*static.artifacts, *temporal_paths, M4_VERIFIER if model_name == 'M4' else M3_VERIFIER], metadata={**static.metadata, 'temporal_role': 'event_verifier_only', 'verifier_path': str(M4_VERIFIER if model_name == 'M4' else M3_VERIFIER), 'verifier_threshold': 0.4101333272622148 if model_name == 'M4' else 0.11, 'verifier_historical_weight': 0.55 if model_name == 'M4' else 0.4})
