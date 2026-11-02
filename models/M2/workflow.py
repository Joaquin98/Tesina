"""Fit and load quantile/logistic RGB-dynamics and YOLO late fusion."""
from pathlib import Path
import joblib
import keras
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from models.common.features import RGB_WEIGHTS, POSE_WEIGHTS, HISTORICAL_WEIGHTS, PCA_WEIGHTS, load_record, static_vector
from models.common.training import labels, metrics, reference, get_reference, read_run, run_detector, keras_probabilities
from models.model_catalog import resolve_artifact


def fit_static(train_rows, validation, output, **options):
    pca = joblib.load(resolve_artifact(PCA_WEIGHTS))
    train_x = np.asarray([static_vector(load_record(row), pca) for row in train_rows])
    validation_x = np.asarray([static_vector(load_record(row), pca) for row in validation])
    scaler = QuantileTransformer(n_quantiles=min(256, len(train_x)), output_distribution='normal', subsample=None, random_state=options['seed'])
    transformed = scaler.fit_transform(train_x)
    if options.get('initial_run'):
        logistic = joblib.load(Path(options['initial_run']) / 'logistic.joblib')
        logistic.set_params(warm_start=True, random_state=options['seed'])
    else:
        logistic = LogisticRegression(C=.003, class_weight={0: 1.2, 1: .65, 2: 1.8}, max_iter=4000, random_state=options['seed'])
    logistic.fit(transformed, labels(train_rows))
    joblib.dump(scaler, output / 'quantile.joblib')
    joblib.dump(logistic, output / 'logistic.joblib')
    return scaler, logistic, pca, train_x, validation_x


def train(train_rows, validation, output, **options):
    scaler, logistic, _pca, _train_x, validation_x = fit_static(train_rows, validation, output, **options)
    historical = keras.models.load_model(resolve_artifact(HISTORICAL_WEIGHTS), compile=False)
    probabilities = .4*keras_probabilities(historical, validation, batch_size=options['batch_size']) + .6*logistic.predict_proba(scaler.transform(validation_x))
    return {'artifacts': {'quantile': 'quantile.joblib', 'logistic': 'logistic.joblib'}, 'references': {'rgb': reference(RGB_WEIGHTS), 'pose': reference(POSE_WEIGHTS), 'historical': reference(HISTORICAL_WEIGHTS), 'pca': reference(PCA_WEIGHTS)}, 'validation': metrics(labels(validation), probabilities), 'static_alpha': .6}


def load(path):
    from models.M2.predictor import RGBYoloHighresQuantileFusionDetector
    manifest = read_run(path, 'M2')
    model = run_detector(RGBYoloHighresQuantileFusionDetector, 'M2', manifest)
    model.dynamic_pca = joblib.load(get_reference(manifest['references']['pca']))
    model.sequence_model = keras.models.load_model(get_reference(manifest['references']['historical']), compile=False)
    model.quantile_scaler = joblib.load(Path(path) / manifest['artifacts']['quantile']['path'])
    model.auxiliary_model = joblib.load(Path(path) / manifest['artifacts']['logistic']['path'])
    model.historical_weight, model.auxiliary_weight = .4, .6
    return model
