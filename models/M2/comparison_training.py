"""M2 training and adaptation for the comparative protocol."""
from __future__ import annotations
from models.training_common import *
from models.M0.comparison_training import fit_m0

def _predict_static_b(frame: pd.DataFrame) -> np.ndarray:
    """Predict the immutable published M2 head once per unique block."""
    global _STATIC_B_ARTIFACTS
    keys = frame['block_id'].astype(str).tolist()
    if len(set(keys)) != len(keys):
        raise ValueError('Static B prediction frame contains duplicate block_id values')
    missing_positions = [index for index, key in enumerate(keys) if key not in _STATIC_B_PROBABILITY_CACHE]
    if missing_positions:
        if _STATIC_B_ARTIFACTS is None:
            _STATIC_B_ARTIFACTS = (joblib.load(_resolve_model_artifact(M2_DEPLOYMENT / 'quantile_scaler.joblib')), joblib.load(_resolve_model_artifact(M2_DEPLOYMENT / 'logistic.joblib')))
        scaler, classifier = _STATIC_B_ARTIFACTS
        matrix = compact_matrix(frame.iloc[missing_positions], ('dynamic', 'descriptor'))
        probabilities = normalize_probabilities(classifier.predict_proba(scaler.transform(matrix)))
        for position, probability in zip(missing_positions, probabilities, strict=True):
            _STATIC_B_PROBABILITY_CACHE[keys[position]] = probability.copy()
    return np.asarray([_STATIC_B_PROBABILITY_CACHE[key] for key in keys], dtype=np.float64)

def _static_head(train, development, test, *, level, seed, run_dir):
    if level == 'B':
        artifacts = [M2_DEPLOYMENT / 'quantile_scaler.joblib', M2_DEPLOYMENT / 'logistic.joblib']
        train_probability = None if not len(train) else _predict_static_b(train)
        development_probability = _predict_static_b(development)
        test_probability = _predict_static_b(test)
    else:
        descriptor_train = compact_matrix(train, ('descriptor',))
        descriptor_dev = compact_matrix(development, ('descriptor',))
        descriptor_test = compact_matrix(test, ('descriptor',))
        if level == 'H0':
            train_x = np.concatenate([dynamic_matrix(train), descriptor_train], axis=1)
            dev_x = np.concatenate([dynamic_matrix(development), descriptor_dev], axis=1)
            test_x = np.concatenate([dynamic_matrix(test), descriptor_test], axis=1)
            scaler = joblib.load(_resolve_model_artifact(M2_DEPLOYMENT / 'quantile_scaler.joblib'))
            pca_artifact = None
        else:
            pca_path = run_dir / 'dynamic_pca.joblib'
            pca = fit_dynamic_pca(train, output=pca_path)
            train_x = np.concatenate([dynamic_matrix(train, pca), descriptor_train], axis=1)
            dev_x = np.concatenate([dynamic_matrix(development, pca), descriptor_dev], axis=1)
            test_x = np.concatenate([dynamic_matrix(test, pca), descriptor_test], axis=1)
            scaler = QuantileTransformer(n_quantiles=min(256, len(train_x)), output_distribution='normal', subsample=None, random_state=seed).fit(train_x)
            pca_artifact = pca_path
        classifier = LogisticRegression(C=0.003, class_weight={0: 1.2, 1: 0.65, 2: 1.8}, max_iter=4000, random_state=seed).fit(scaler.transform(train_x), labels_for(train))
        run_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = run_dir / 'quantile_scaler.joblib'
        logistic_path = run_dir / 'logistic.joblib'
        joblib.dump(scaler, scaler_path)
        joblib.dump(classifier, logistic_path)
        artifacts = [scaler_path, logistic_path]
        if pca_artifact is not None:
            artifacts.insert(0, pca_artifact)
        train_probability = normalize_probabilities(classifier.predict_proba(scaler.transform(train_x)))
        development_probability = normalize_probabilities(classifier.predict_proba(scaler.transform(dev_x)))
        test_probability = normalize_probabilities(classifier.predict_proba(scaler.transform(test_x)))
    return (train_probability, development_probability, test_probability, artifacts)

def fit_predict_m2(train, development, test, *, level, seed, run_dir, epochs, patience, batch_size, base_path=M0_PATH) -> PredictionBundle:
    static_train, static_dev, static_test, artifacts = _static_head(train, development, test, level=level, seed=seed, run_dir=run_dir)
    if level == 'H2':
        base_model_path = run_dir / 'historical.keras'
        historical_model = fit_m0(train, development, level='H2', seed=seed, output=base_model_path, epochs=epochs, patience=patience, batch_size=batch_size, base_path=base_path)
        artifacts.append(base_model_path)
    else:
        artifacts.append(base_path)
    if level == 'H2':
        historical_dev = _predict_keras(historical_model, development, batch_size=batch_size)
        historical_test = _predict_keras(historical_model, test, batch_size=batch_size)
        historical_train = _predict_keras(historical_model, train, batch_size=batch_size) if len(train) else None
    else:
        historical_dev = _predict_frozen_keras(base_path, development, batch_size=batch_size)
        historical_test = _predict_frozen_keras(base_path, test, batch_size=batch_size)
        historical_train = _predict_frozen_keras(base_path, train, batch_size=batch_size) if len(train) else None
    alpha = 0.6 if level == 'B' else _choose_alpha(labels_for(development), historical_dev, static_dev)
    development_probabilities = normalize_probabilities((1.0 - alpha) * historical_dev + alpha * static_dev)
    probabilities = normalize_probabilities((1.0 - alpha) * historical_test + alpha * static_test)
    training_probabilities = None
    if historical_train is not None and static_train is not None:
        training_probabilities = normalize_probabilities((1.0 - alpha) * historical_train + alpha * static_train)
    return PredictionBundle(probabilities=probabilities, development_probabilities=development_probabilities, training_probabilities=training_probabilities, branches={'historical': historical_test, 'static': static_test}, development_branches={'historical': historical_dev, 'static': static_dev}, training_branches={} if training_probabilities is None else {'historical': historical_train, 'static': static_train}, artifacts=artifacts, metadata={'static_alpha': alpha})
