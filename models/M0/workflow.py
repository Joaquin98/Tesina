"""Fit and load the ResNet-feature BiGRU classifier."""
from pathlib import Path
import keras
from models.common.features import RGB_WEIGHTS, HISTORICAL_WEIGHTS
from models.common.training import keras_fit, keras_probabilities, labels, metrics, reference, read_run, run_detector
from models.model_catalog import resolve_artifact


def train(train_rows, validation, output, **options):
    initial = Path(options['initial_run']) / 'model.keras' if options.get('initial_run') else resolve_artifact(HISTORICAL_WEIGHTS)
    model = keras.models.load_model(initial, compile=False)
    if options['init'] == 'random' and not options.get('initial_run'):
        keras.utils.set_random_seed(options['seed'])
        model = keras.models.clone_model(model)
    history = keras_fit(model, train_rows, validation, output / 'model.keras', **{key: options[key] for key in ('epochs', 'batch_size', 'patience', 'seed')})
    return {'artifacts': {'model': 'model.keras'}, 'references': {'rgb': reference(RGB_WEIGHTS)}, 'validation': metrics(labels(validation), keras_probabilities(model, validation, batch_size=options['batch_size'])), 'history': history}


def load(path):
    from models.M0.web_adapter import RugbyPlayDetector
    manifest = read_run(path, 'M0')
    model = run_detector(RugbyPlayDetector, 'M0', manifest)
    model.sequence_model = keras.models.load_model(Path(path) / manifest['artifacts']['model']['path'], compile=False)
    return model
