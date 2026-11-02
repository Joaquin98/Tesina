"""Read training records present in this repository without loading model weights."""
import json
import logging
from pathlib import Path

from models.model_catalog import index, resolve_artifact

logger = logging.getLogger(__name__)


def read_training_history(root):
    root = Path(root).resolve()
    local_runs, historical_runs = [], []
    for family in ('M0', 'M1', 'M2', 'M3'):
        for path in sorted((root/'models'/family/'runs').glob('*/run.json')):
            if path.parent.name.startswith('.') or not path.resolve().is_relative_to(root):
                continue
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                if record.get('status') != 'complete' or record.get('family') != family:
                    continue
                validation = record.get('validation', {})
                dataset = record.get('dataset', {})
                local_runs.append({'family': family, 'run': path.parent.name,
                                   'validation_accuracy': validation.get('accuracy'),
                                   'validation_f1_macro': validation.get('f1_macro'),
                                   'training_examples': dataset.get('train', {}).get('selected_examples'),
                                   'validation_examples': dataset.get('validation', {}).get('selected_examples'),
                                   'epochs_requested': record.get('training', {}).get('epochs')})
            except (OSError, ValueError, TypeError) as exc:
                logger.warning('Could not read training record %s: %s', path, exc)
    for run in index()['runs']:
        path = resolve_artifact(root/run['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            historical_runs.append({'family': run['model'], 'run': run['run_id'],
                                    'fold': record.get('fold'), 'training_examples': record.get('n_train'),
                                    'validation_examples': record.get('n_dev'), 'test_examples': record.get('n_test'),
                                    'elapsed_seconds': record.get('elapsed_seconds')})
        except (OSError, ValueError, TypeError) as exc:
            logger.warning('Could not read historical training record %s: %s', path, exc)
    return {'local_runs': local_runs, 'historical_runs': historical_runs}
