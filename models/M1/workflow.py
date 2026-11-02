"""Fit and load the dual RGB/pose BiGRU and geometry forest."""
from pathlib import Path
import joblib
import keras
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from models.common.features import ROOT, RGB_WEIGHTS, POSE_WEIGHTS, load_record, m1_inputs
from models.common.training import keras_fit, keras_probabilities, labels, metrics, reference, read_run, run_detector
from models.model_catalog import resolve_artifact


def train(train_rows, validation, output, **options):
    initial = Path(options['initial_run']) / 'dual.keras' if options.get('initial_run') else resolve_artifact(ROOT / 'models/M1/checkpoints/dual.keras')
    model = keras.models.load_model(initial, compile=False)
    if options['init'] == 'random' and not options.get('initial_run'):
        keras.utils.set_random_seed(options['seed'])
        model = keras.models.clone_model(model)
    history = keras_fit(model, train_rows, validation, output / 'dual.keras', dual=True, **{key: options[key] for key in ('epochs', 'batch_size', 'patience', 'seed')})
    forest = RandomForestClassifier(n_estimators=500, max_features='sqrt', class_weight='balanced', random_state=options['seed'], n_jobs=options['threads'])
    forest.fit(np.asarray([m1_inputs(load_record(row))[1] for row in train_rows]), labels(train_rows))
    joblib.dump(forest, output / 'geometry.joblib')
    dual = keras_probabilities(model, validation, batch_size=options['batch_size'], dual=True)
    geometry = forest.predict_proba(np.asarray([m1_inputs(load_record(row))[1] for row in validation]))
    best = None
    for alpha in np.linspace(0, 1, 21):
        probability = np.exp((1-alpha)*np.log(np.clip(dual, 1e-7, 1)) + alpha*np.log(np.clip(geometry, 1e-7, 1)))
        probability /= probability.sum(1, keepdims=True)
        result = metrics(labels(validation), probability)
        if best is None or result['f1_macro'] > best['validation']['f1_macro']:
            best = {'geometry_alpha': float(alpha), 'validation': result}
    return {'artifacts': {'dual': 'dual.keras', 'geometry': 'geometry.joblib'}, 'references': {'rgb': reference(RGB_WEIGHTS), 'pose': reference(POSE_WEIGHTS)}, 'history': history, **best}


def load(path):
    from models.M1.inference import M1H2D2CrossFoldDetector
    manifest = read_run(path, 'M1')
    model = run_detector(M1H2D2CrossFoldDetector, 'M1', manifest)
    model.fold_models = [{'dual': keras.models.load_model(Path(path) / manifest['artifacts']['dual']['path'], compile=False), 'geometry': joblib.load(Path(path) / manifest['artifacts']['geometry']['path']), 'geometry_alpha': manifest['geometry_alpha']}]
    return model
