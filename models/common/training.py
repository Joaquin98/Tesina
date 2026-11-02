"""Bounded minibatch fitting, validation and immutable run artifact records."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from models.common.features import ROOT, LABELS, load_record, rgb_inputs, m1_inputs
from models.inference_cache import file_digest
from models.model_catalog import resolve_artifact


def labels(records):
    return np.asarray([LABELS.index(row.tag) for row in records], np.int64)


def batch_inputs(records, dual=False):
    values = [load_record(row) for row in records]
    rgb = [rgb_inputs(row) for row in values]
    features = np.asarray([row[0] for row in rgb], np.float32)
    mask = np.asarray([row[1] for row in rgb], bool)
    if dual:
        return {'rgb': features, 'pose': np.asarray([m1_inputs(row)[0] for row in values], np.float32), 'sequence_mask': mask}
    return [features, mask]


def keras_fit(model, train, validation, output, *, epochs, batch_size, patience, seed, dual=False):
    import keras
    keras.utils.set_random_seed(seed)
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    y = labels(train)
    weights = len(y) / (3 * np.bincount(y, minlength=3))
    rng = np.random.default_rng(seed)
    best_loss, best_weights, stale = float('inf'), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train))
        model.reset_metrics()
        for start in range(0, len(train), batch_size):
            indices = order[start:start+batch_size]
            model.train_on_batch(batch_inputs([train[i] for i in indices], dual), y[indices], sample_weight=weights[y[indices]].astype(np.float32))
        model.reset_metrics()
        result = None
        for start in range(0, len(validation), batch_size):
            rows = validation[start:start+batch_size]
            result = model.test_on_batch(batch_inputs(rows, dual), labels(rows), return_dict=True)
        loss = float(result['loss'])
        if not np.isfinite(loss):
            raise ValueError('El entrenamiento produjo una pérdida no finita')
        history.append({'epoch': epoch, 'validation_loss': loss, 'validation_accuracy': float(result['accuracy'])})
        print(json.dumps(history[-1]), flush=True)
        if loss < best_loss:
            best_loss, best_weights, stale = loss, model.get_weights(), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.set_weights(best_weights)
    model.save(output)
    return history


def keras_probabilities(model, records, *, batch_size=8, dual=False):
    result = []
    for start in range(0, len(records), batch_size):
        result.extend(np.asarray(model(batch_inputs(records[start:start+batch_size], dual), training=False)))
    return np.asarray(result, np.float64)


def metrics(truth, probabilities):
    from sklearn.metrics import accuracy_score, f1_score
    p = np.asarray(probabilities)
    if p.shape != (len(truth), 3) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Probabilidades inválidas')
    prediction = p.argmax(1)
    return {'examples': len(truth), 'accuracy': float(accuracy_score(truth, prediction)), 'f1_macro': float(f1_score(truth, prediction, labels=[0, 1, 2], average='macro', zero_division=0))}


def reference(path):
    path = resolve_artifact(Path(path)).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Los pesos de referencia deben pertenecer a este repositorio')
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': file_digest(path)}


def get_reference(record):
    path = (ROOT / record['path']).resolve()
    if not path.is_relative_to(ROOT) or file_digest(path) != record['sha256']:
        raise ValueError('Los pesos de referencia cambiaron o están fuera del repositorio')
    return path


def read_run(path, family=None):
    path = Path(path).resolve()
    manifest = json.loads((path / 'run.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 1 or manifest.get('status') != 'complete' or manifest.get('labels') != list(LABELS):
        raise ValueError('Corrida incompleta o incompatible')
    if family is not None and manifest['family'] != family:
        raise ValueError('La corrida no pertenece al modelo seleccionado')
    for item in manifest['artifacts'].values():
        artifact = (path / item['path']).resolve()
        if not artifact.is_relative_to(path) or file_digest(artifact) != item['sha256']:
            raise ValueError('Peso de la corrida modificado o fuera de su carpeta')
    for item in manifest['references'].values():
        get_reference(item)
    return manifest


def run_detector(cls, family, manifest):
    from models.common.features import feature_source
    source = feature_source(family, references=manifest['references'])
    model = object.__new__(cls)
    model.__dict__.update(source.__dict__)
    model.project_root = ROOT
    model.label_vocab = np.asarray(LABELS, dtype=object)
    model._model_name = family + '_run'
    model._display_name = family + ' · corrida entrenada'
    model.fusion_metadata = {'family': family, 'deployment_policy': 'single_trained_run', 'validation_metrics': manifest['validation']}
    from models.common.features import FeatureProvider
    model.feature_provider = FeatureProvider(family)
    return model
