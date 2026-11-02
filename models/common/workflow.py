"""Unified local cache, training, evaluation and saved-model inference workflow."""
from __future__ import annotations
from collections import Counter
import csv
from datetime import datetime
import hashlib
from importlib import import_module
import json
from pathlib import Path
import tempfile

from models.common.features import ROOT, LABELS
from models.inference_cache import file_digest

DEFAULT_DATASET = ROOT / 'data/clips_dataset'
FAMILIES = ('M0', 'M1', 'M2', 'M3')
_THREADS = 2


def local_path(path):
    path = Path(path).resolve()
    original = ROOT.parents[1]
    if path.is_relative_to(original) and not path.is_relative_to(ROOT):
        raise ValueError('La operación debe usar el repositorio nuevo, no el original')
    return path


def read_splits(dataset=DEFAULT_DATASET):
    dataset = local_path(dataset)
    splits, owners, digests = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        rows = []
        with (dataset / (split + '.csv')).open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                tag = str(row.get('tag', '')).strip()
                if tag not in LABELS:
                    raise ValueError(f'Etiqueta desconocida: {tag}')
                value = row.get('path')
                if value:
                    path = Path(value)
                    if not path.is_absolute():
                        path = ROOT / path if (ROOT / path).is_file() else dataset / path
                else:
                    path = ROOT / 'data/videos/clips_224' / row['video_name']
                path = local_path(path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                if path not in digests:
                    digests[path] = file_digest(path)
                digest = digests[path]
                if row.get('sha256') and row['sha256'] != digest:
                    raise ValueError(f'El hash del video no coincide con el CSV: {path.name}')
                identity = row.get('clip_id') or row.get('video_name') or path.name
                for key in ('id:' + identity, 'path:' + str(path), 'sha256:' + digest):
                    if key in owners and owners[key] != (split, tag):
                        raise ValueError(f'Clips compartidos o etiquetas incompatibles (overlap): {identity}')
                    owners[key] = (split, tag)
                rows.append({**row, 'path': path, 'tag': tag, 'sha256': digest, 'split': split})
        if not rows:
            raise ValueError(f'La partición {split} está vacía')
        splits[split] = rows
    if {row['tag'] for row in splits['train']} != set(LABELS):
        raise ValueError('Entrenamiento requiere las tres clases')
    return splits


def selection(rows, limit_per_class=None):
    if limit_per_class is None:
        return rows
    if limit_per_class <= 0:
        raise ValueError('limit_per_class debe ser positivo')
    counts, selected = Counter(), []
    for row in rows:
        if counts[row['tag']] < limit_per_class:
            selected.append(row)
            counts[row['tag']] += 1
    return selected


def configure(threads=2):
    global _THREADS
    if threads < 1:
        raise ValueError('threads debe ser positivo')
    import torch
    torch.set_num_threads(threads)
    _THREADS = threads


def load_detector(family, run=None):
    if family not in FAMILIES:
        raise ValueError(family)
    if run is not None:
        model = import_module(f'models.{family}.workflow').load(local_path(run))
    else:
        from models.model_catalog import temporary_settings
        import os
        os.environ['YOLO_CONFIG_DIR'] = str(temporary_settings())
        os.environ['YOLO_VERBOSE'] = 'False'
        model = import_module('models.' + family).load_model(ROOT)
    for member in [*getattr(model, 'fold_models', []), *getattr(model, 'seed_models', [])]:
        geometry = member.get('geometry')
        if geometry is not None and hasattr(geometry, 'n_jobs'):
            geometry.n_jobs = _THREADS
    return model


def prepare_cache(family, *, dataset=DEFAULT_DATASET, split='all', video=None, cache_dir=None, limit_per_class=None):
    from models.common.features import feature_source, make_record, prepare
    cache_dir = local_path(cache_dir) if cache_dir is not None else None
    source = feature_source(family)
    if video is not None:
        _features, _state, info, _parameters = prepare(source, local_path(video), cache_dir=cache_dir)
        if info['status'] not in ('created', 'hit'):
            raise OSError('No se pudo guardar la caché local')
        return {'status': 'ok', 'family': family, 'cache': info}
    splits = read_splits(dataset)
    names = splits if split == 'all' else [split]
    counts = Counter()
    for name in names:
        for row in selection(splits[name], limit_per_class):
            _record, info = make_record(source, row, cache_dir)
            counts[info['status']] += 1
    return {'status': 'ok', 'family': family, 'cache': dict(counts)}


def train_run(family, *, dataset=DEFAULT_DATASET, output=None, cache_dir=None, epochs=40, batch_size=8, patience=7, seed=42, threads=2, init='published', initial_run=None, limit_per_class=None):
    from models.common.features import feature_source, make_record
    from models.common.training import read_run
    if family not in FAMILIES or init not in ('published', 'random'):
        raise ValueError('Familia o inicialización inválida')
    if min(epochs, batch_size, patience, threads) < 1:
        raise ValueError('Épocas, lote, paciencia y threads deben ser positivos')
    configure(threads)
    import keras
    keras.utils.set_random_seed(seed)
    dataset = local_path(dataset)
    initial_run = local_path(initial_run) if initial_run is not None else None
    if initial_run is not None:
        from models.common.features import RGB_WEIGHTS, POSE_WEIGHTS, HISTORICAL_WEIGHTS, PCA_WEIGHTS
        from models.model_catalog import resolve_artifact
        previous = read_run(initial_run, family)
        expected = {'rgb': RGB_WEIGHTS}
        if family != 'M0': expected['pose'] = POSE_WEIGHTS
        if family in ('M2', 'M3'): expected.update(historical=HISTORICAL_WEIGHTS, pca=PCA_WEIGHTS)
        if any(previous['references'][key]['sha256'] != file_digest(resolve_artifact(path)) for key, path in expected.items()):
            raise ValueError('La corrida inicial utiliza otros extractores o referencias congeladas')
    output = local_path(output or ROOT / 'models' / family / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists():
        raise FileExistsError('La carpeta de salida ya existe; las corridas no se sobrescriben')
    if output.is_relative_to(ROOT / 'models/familias'):
        raise ValueError('Los pesos publicados no son una salida de entrenamiento')
    splits = read_splits(dataset)
    source = feature_source(family)
    cache_dir = local_path(cache_dir) if cache_dir is not None else None
    prepared, cache_counts = {}, Counter()
    for split in ('train', 'validation'):
        prepared[split] = []
        for row in selection(splits[split], limit_per_class):
            record, info = make_record(source, row, cache_dir)
            prepared[split].append(record)
            cache_counts[info['status']] += 1
    del source
    output.parent.mkdir(parents=True, exist_ok=True)
    options = dict(epochs=epochs, batch_size=batch_size, patience=patience, seed=seed, threads=threads, init='run' if initial_run is not None else init, initial_run=str(initial_run) if initial_run is not None else None, initial_run_sha256=file_digest(initial_run / 'run.json') if initial_run is not None else None)
    with tempfile.TemporaryDirectory(prefix='.partial_run_', dir=output.parent) as temporary:
        stage = Path(temporary)
        result = import_module(f'models.{family}.workflow').train(prepared['train'], prepared['validation'], stage, **options)
        result['artifacts'] = {key: {'path': name, 'sha256': file_digest(stage / name)} for key, name in result['artifacts'].items()}
        manifest = {'schema': 1, 'status': 'complete', 'family': family, 'labels': list(LABELS), 'training': options, 'selection': {'limit_per_class': limit_per_class}, 'dataset': {name: {'csv_sha256': file_digest(dataset / (name + '.csv')), 'selected_examples': len(prepared[name]), 'selected_videos_sha256': hashlib.sha256(json.dumps([(str(row.video), row.tag, row.sha256) for row in prepared[name]]).encode()).hexdigest()} for name in prepared}, **result}
        (stage / 'run.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        read_run(stage, family)
        stage.rename(output)
    return {'status': 'ok', 'family': family, 'run': str(output), 'cache': dict(cache_counts), 'validation': result['validation']}


def predict(family, video, *, run=None, cache_dir=None, use_cache=True, detector=None, provider=None, **parameters):
    from models.common.features import FeatureProvider
    detector = detector or load_detector(family, run)
    parameters.setdefault('feature_batch_size', 8)
    parameters.setdefault('feature_predict_batch_size', 2)
    result = detector.analyze_video_segments(str(local_path(video)), cache_dir=local_path(cache_dir) if cache_dir is not None else None, use_cache=use_cache, feature_provider=provider or FeatureProvider(family), **parameters)
    result['family'] = family
    result['weights_source'] = 'published' if run is None else str(Path(run).resolve())
    return result


def evaluate(family, *, dataset=DEFAULT_DATASET, split='test', run=None, cache_dir=None, use_cache=True, limit_per_class=None):
    from models.common.features import FeatureProvider
    from models.common.training import metrics
    import numpy as np
    rows = selection(read_splits(dataset)[split], limit_per_class)
    detector, provider = load_detector(family, run), FeatureProvider(family)
    probabilities = []
    for row in rows:
        result = predict(family, row['path'], run=run, cache_dir=cache_dir, use_cache=use_cache, detector=detector, provider=provider)
        if result['status'] != 'success':
            raise ValueError(result.get('error'))
        probabilities.append(np.mean([[window['probabilities'][label] for label in LABELS] for window in result['window_predictions']], axis=0))
    return {'status': 'ok', 'family': family, 'split': split, 'scope': 'clip_classification', 'metrics': metrics([LABELS.index(row['tag']) for row in rows], probabilities)}
