"""Compare Keras Applications as rugby video frame feature extractors.

This script is intentionally self-contained and writes results only inside this
experiment folder by default. It does not import or edit the current production
CNN-RNN training modules.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import numpy as np
EXPERIMENT_DIR = experiment_directory(__file__)
CANDIDATES_CSV = EXPERIMENT_DIR / 'keras_applications_candidates.csv'
RESULTS_DIR = EXPERIMENT_DIR / 'results'
IMG_SIZE = 224
PROJECT_FEATURE_DIM = 2048

@dataclass(frozen=True)
class ExtractorSpec:
    name: str
    constructor_path: str
    preprocess_path: str | None
    native_feature_dim: int | None
    notes: str = ''
EXTRACTOR_SPECS: dict[str, ExtractorSpec] = {'inceptionv3': ExtractorSpec('inceptionv3', 'keras.applications.InceptionV3', 'keras.applications.inception_v3.preprocess_input', 2048), 'resnet50': ExtractorSpec('resnet50', 'keras.applications.ResNet50', 'keras.applications.resnet50.preprocess_input', 2048), 'resnet50v2': ExtractorSpec('resnet50v2', 'keras.applications.ResNet50V2', 'keras.applications.resnet_v2.preprocess_input', 2048), 'efficientnetb0': ExtractorSpec('efficientnetb0', 'keras.applications.EfficientNetB0', 'keras.applications.efficientnet.preprocess_input', 1280, 'Keras EfficientNet preprocessing is included in the model, but the call is harmless.'), 'efficientnetv2b0': ExtractorSpec('efficientnetv2b0', 'keras.applications.EfficientNetV2B0', 'keras.applications.efficientnet_v2.preprocess_input', 1280), 'efficientnetv2b1': ExtractorSpec('efficientnetv2b1', 'keras.applications.EfficientNetV2B1', 'keras.applications.efficientnet_v2.preprocess_input', 1280), 'efficientnetv2b2': ExtractorSpec('efficientnetv2b2', 'keras.applications.EfficientNetV2B2', 'keras.applications.efficientnet_v2.preprocess_input', 1408), 'nasnetmobile': ExtractorSpec('nasnetmobile', 'keras.applications.NASNetMobile', 'keras.applications.nasnet.preprocess_input', 1056), 'mobilenetv3large': ExtractorSpec('mobilenetv3large', 'keras.applications.MobileNetV3Large', 'keras.applications.mobilenet_v3.preprocess_input', None, 'Native dimension is detected at runtime.'), 'mobilenetv3small': ExtractorSpec('mobilenetv3small', 'keras.applications.MobileNetV3Small', 'keras.applications.mobilenet_v3.preprocess_input', None, 'Native dimension is detected at runtime.'), 'convnexttiny': ExtractorSpec('convnexttiny', 'keras.applications.ConvNeXtTiny', 'keras.applications.convnext.preprocess_input', 768), 'xception': ExtractorSpec('xception', 'keras.applications.Xception', 'keras.applications.xception.preprocess_input', 2048), 'densenet121': ExtractorSpec('densenet121', 'keras.applications.DenseNet121', 'keras.applications.densenet.preprocess_input', 1024)}
DEFAULT_MODELS = ['resnet50', 'resnet50v2', 'inceptionv3', 'efficientnetb0', 'efficientnetv2b0', 'efficientnetv2b1', 'nasnetmobile', 'mobilenetv3large', 'convnexttiny']

def load_candidate_rows() -> list[dict[str, str]]:
    with _resolve_model_artifact(CANDIDATES_CSV).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def _float_or_none(value: str | None) -> float | None:
    if value is None or value == '':
        return None
    return float(value)

def score_candidate(row: dict[str, str]) -> float:
    """Heuristic score for this project, not a general ImageNet ranking."""
    top1 = _float_or_none(row.get('keras_top1')) or 0.0
    cpu_ms = _float_or_none(row.get('cpu_ms'))
    size_mb = _float_or_none(row.get('size_mb'))
    params_m = _float_or_none(row.get('params_m'))
    priority = _float_or_none(row.get('priority')) or 5.0
    score = top1
    if cpu_ms is not None:
        score -= 0.08 * cpu_ms
    if size_mb is not None:
        score -= 0.015 * size_mb
    if params_m is not None:
        score -= 0.04 * params_m
    score -= 2.0 * max(priority - 1.0, 0.0)
    if row.get('project_role') == 'current_baseline':
        score += 2.0
    if row.get('project_role') == 'live_candidate':
        score += 1.5
    return round(score, 3)

def print_metadata_ranking() -> None:
    rows = load_candidate_rows()
    ranked = sorted(rows, key=score_candidate, reverse=True)
    print('Project-oriented Keras Applications ranking')
    print('(heuristic: ImageNet quality balanced against CPU/size/project priority)')
    print()
    for index, row in enumerate(ranked, start=1):
        print(f"{index:2d}. {row['model']:<20} score={score_candidate(row):>7} top1={row['keras_top1'] or 'n/a':>5} cpu_ms={row['cpu_ms'] or 'n/a':>6} size_mb={row['size_mb'] or 'n/a':>7} role={row['project_role']}")

def resolve_symbol(dotted_path: str):
    import keras
    value = {'keras': keras}
    current = value[dotted_path.split('.')[0]]
    for part in dotted_path.split('.')[1:]:
        current = getattr(current, part)
    return current

def build_feature_extractor(model_name: str, weights: str | None, project_to_2048: bool=False):
    import keras
    if model_name not in EXTRACTOR_SPECS:
        raise KeyError(f'Unknown model {model_name!r}. Choices: {sorted(EXTRACTOR_SPECS)}')
    spec = EXTRACTOR_SPECS[model_name]
    constructor = resolve_symbol(spec.constructor_path)
    preprocess: Callable | None = resolve_symbol(spec.preprocess_path) if spec.preprocess_path else None
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name='rgb_frame')
    x = inputs
    if preprocess is not None:
        x = preprocess(x)
    kwargs = {'weights': weights, 'include_top': False, 'pooling': 'avg', 'input_shape': (IMG_SIZE, IMG_SIZE, 3)}
    base = constructor(**kwargs)
    base.trainable = False
    outputs = base(x)
    if project_to_2048 and outputs.shape[-1] != PROJECT_FEATURE_DIM:
        outputs = keras.layers.Dense(PROJECT_FEATURE_DIM, activation=None, name='feature_projection_2048')(outputs)
    return keras.Model(inputs, outputs, name=f'feature_extractor_{model_name}')

def benchmark_models(model_names: list[str], weights: str | None, batch_size: int, steps: int, warmup: int, project_to_2048: bool) -> list[dict[str, object]]:
    import keras
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = np.random.randint(0, 256, size=(batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8).astype('float32')
    results: list[dict[str, object]] = []
    for model_name in model_names:
        started = time.perf_counter()
        try:
            model = build_feature_extractor(model_name, weights=weights, project_to_2048=project_to_2048)
            build_seconds = time.perf_counter() - started
            for _ in range(warmup):
                _ = model.predict(frames, batch_size=batch_size, verbose=0)
            step_times = []
            output_shape = None
            for _ in range(steps):
                step_started = time.perf_counter()
                output = model.predict(frames, batch_size=batch_size, verbose=0)
                step_times.append(time.perf_counter() - step_started)
                output_shape = list(output.shape)
            avg_step_seconds = float(np.mean(step_times))
            avg_frame_ms = avg_step_seconds * 1000.0 / batch_size
            params = int(model.count_params())
            result = {'model': model_name, 'status': 'ok', 'weights': weights or 'none', 'project_to_2048': project_to_2048, 'batch_size': batch_size, 'steps': steps, 'build_seconds': round(build_seconds, 4), 'avg_step_seconds': round(avg_step_seconds, 4), 'avg_frame_ms': round(avg_frame_ms, 3), 'output_shape': output_shape, 'params': params, 'keras_backend': keras.backend.backend()}
        except Exception as exc:
            result = {'model': model_name, 'status': 'error', 'weights': weights or 'none', 'error': repr(exc)}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    output_path = RESULTS_DIR / 'benchmark_results.json'
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    write_results_csv(results, RESULTS_DIR / 'benchmark_results.csv')
    return results

def write_results_csv(results: list[dict[str, object]], output_path: Path) -> None:
    if not results:
        return
    keys = sorted({key for row in results for key in row})
    with _resolve_model_artifact(output_path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', action='store_true', help='Print candidate metadata ranking.')
    parser.add_argument('--smoke', action='store_true', help='Build selected models and run one predict step.')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark selected models on random frames.')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='Model names to compare.')
    parser.add_argument('--weights', choices=['none', 'imagenet'], default='none', help='Use none to avoid downloads; use imagenet for real pretrained extractors.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--project-to-2048', action='store_true', help='Add a Dense projection to match the current CNN-RNN 2048-d feature contract.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    weights = None if args.weights == 'none' else args.weights
    if args.metadata or not (args.smoke or args.benchmark):
        print_metadata_ranking()
    if args.smoke:
        benchmark_models(model_names=args.models, weights=weights, batch_size=2, steps=1, warmup=0, project_to_2048=args.project_to_2048)
    if args.benchmark:
        benchmark_models(model_names=args.models, weights=weights, batch_size=args.batch_size, steps=args.steps, warmup=args.warmup, project_to_2048=args.project_to_2048)
if __name__ == '__main__':
    main()
