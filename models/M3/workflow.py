"""Fit static fusion, temporal pose attention and a single-run event verifier."""
from pathlib import Path
import json
import joblib
import keras
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from models.common.features import ROOT, RGB_WEIGHTS, POSE_WEIGHTS, HISTORICAL_WEIGHTS, PCA_WEIGHTS, load_record, temporal_inputs
from models.common.training import labels, metrics, reference, get_reference, read_run, run_detector, keras_probabilities
from models.M2.workflow import fit_static
from models.common.event_verifier import event_vector, window_meta_features, LineEventVerifier
from models.model_catalog import resolve_artifact


def event_vectors(historical, static, temporal):
    official = .4*historical + .6*static
    return np.asarray([event_vector(window_meta_features(h[None], s[None], t[None]), h[None], s[None], o[None], t[None], 0, 1) for h, s, o, t in zip(historical, static, official, temporal)], np.float32)


def train(train_rows, validation, output, **options):
    from models.M3.run_temporal_attention_experiment import IndexedSequenceDataset
    from models.M3.run_temporal_architecture_suite import train_select, predict
    from models.M3.predictor import MANIFEST_RELATIVE, M3_H2_D2_MODEL_NAME
    scaler, logistic, _pca, train_x, validation_x = fit_static(train_rows, validation, output, **options)
    arrays = [temporal_inputs(load_record(row)) for row in train_rows + validation]
    x = np.asarray([item[0] for item in arrays], np.float32)
    mask = np.asarray([item[1] for item in arrays], bool)
    quality = np.asarray([item[2] for item in arrays], np.float32)
    n = len(train_rows)
    valid_train = x[:n][mask[:n]]
    mean, std = valid_train.mean(0), np.maximum(valid_train.std(0), 1e-5)
    normalized = np.clip((x-mean)/std, -8, 8).astype(np.float32)
    normalized[~mask] = 0
    y = labels(train_rows + validation)
    dataset = IndexedSequenceDataset(normalized, mask, quality, y)
    initial = None
    if options.get('initial_run') or options['init'] == 'published':
        if options.get('initial_run'):
            initial_path = Path(options['initial_run']) / 'temporal.pt'
        else:
            published = json.loads(resolve_artifact(ROOT / MANIFEST_RELATIVE[M3_H2_D2_MODEL_NAME]).read_text(encoding='utf-8'))
            initial_path = resolve_artifact(ROOT / published['folds'][0]['temporal_model'])
        checkpoint = torch.load(initial_path, map_location='cpu', weights_only=False)
        if checkpoint['architecture'] != 'tcn_bigru_attention':
            raise ValueError('La arquitectura temporal publicada no coincide con el entrenamiento seleccionado')
        initial = checkpoint['state_dict']
    model, best, history = train_select('tcn_bigru_attention', dataset, np.arange(n), np.arange(n, len(y)), y, seed=options['seed'], batch_size=options['batch_size'], max_epochs=options['epochs'], patience=options['patience'], initial_state=initial)
    logits, _attention, indices = predict(model, dataset, np.arange(len(y)), options['batch_size'])
    if not np.array_equal(indices, np.arange(len(y))):
        raise ValueError('Cambió el orden de las predicciones temporales')
    temporal = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    torch.save({'architecture': 'tcn_bigru_attention', 'state_dict': model.state_dict(), 'mean': mean.astype(np.float32), 'std': std.astype(np.float32), 'temperature': 1., 'best_epoch': best['epoch']}, output / 'temporal.pt')
    historical_model = keras.models.load_model(resolve_artifact(HISTORICAL_WEIGHTS), compile=False)
    historical = keras_probabilities(historical_model, train_rows + validation, batch_size=options['batch_size'])
    static = np.concatenate([logistic.predict_proba(scaler.transform(train_x)), logistic.predict_proba(scaler.transform(validation_x))])
    vectors = event_vectors(historical, static, temporal)
    verifier = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight='balanced', max_iter=4000, random_state=options['seed']))
    verifier.fit(vectors[:n], (y[:n] == 0).astype(int))
    joblib.dump(verifier, output / 'verifier.joblib')
    scores = verifier.predict_proba(vectors[n:])[:, list(verifier.classes_).index(1)]
    official = .4*historical[n:] + .6*static[n:]
    thresholds = [.5, *np.linspace(.05, .95, 19)]
    threshold = max(thresholds, key=lambda t: f1_score(y[n:] == 0, (official.argmax(1) == 0) & (scores >= t), zero_division=0))
    return {'artifacts': {'quantile': 'quantile.joblib', 'logistic': 'logistic.joblib', 'temporal': 'temporal.pt', 'verifier': 'verifier.joblib'}, 'references': {'rgb': reference(RGB_WEIGHTS), 'pose': reference(POSE_WEIGHTS), 'historical': reference(HISTORICAL_WEIGHTS), 'pca': reference(PCA_WEIGHTS)}, 'validation': metrics(y[n:], official), 'verifier_threshold': float(threshold), 'verifier_scope': 'individual labeled clips; threshold selected on validation only', 'static_alpha': .6, 'history': history, 'fold_count': 1}


def load(path):
    from models.M3.predictor import M3CrossFoldDetector
    from models.M3.run_temporal_architecture_suite import build_model
    from models.M2.grid_search_cached_descriptors import primitive_base
    from models.M3.run_temporal_attention_experiment import quality_score, smooth_three, usable_mask
    manifest = read_run(path, 'M3')
    model = run_detector(M3CrossFoldDetector, 'M3', manifest)
    model.manifest = {'validation_status': 'new_single_run_validation_only'}
    model.line_min_votes = 1
    model._torch, model._primitive_base = torch, primitive_base
    model._quality_score, model._smooth_three, model._usable_mask = quality_score, smooth_three, usable_mask
    checkpoint = torch.load(Path(path) / manifest['artifacts']['temporal']['path'], map_location='cpu', weights_only=False)
    temporal = build_model(checkpoint['architecture'])
    temporal.load_state_dict(checkpoint['state_dict'])
    temporal.eval()
    model.sequence_model = keras.models.load_model(get_reference(manifest['references']['historical']), compile=False)
    model.dynamic_pca = joblib.load(get_reference(manifest['references']['pca']))
    scaler = joblib.load(Path(path) / manifest['artifacts']['quantile']['path'])
    logistic = joblib.load(Path(path) / manifest['artifacts']['logistic']['path'])
    verifier = joblib.load(Path(path) / manifest['artifacts']['verifier']['path'])
    threshold = manifest['verifier_threshold']
    model.fold_models = [{'historical': model.sequence_model, 'dynamic_pca': model.dynamic_pca, 'scaler': scaler, 'logistic': logistic, 'temporal': temporal, 'temporal_mean': checkpoint['mean'], 'temporal_std': checkpoint['std'], 'temporal_temperature': 1., 'verifier': verifier, 'verifier_threshold': threshold, 'line_event_verifier': LineEventVerifier(verifier, threshold=threshold), 'static_alpha': .6}]
    return model
