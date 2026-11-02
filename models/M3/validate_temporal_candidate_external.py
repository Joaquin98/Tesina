"""External validation of the fixed 9% TCN+BiGRU temporal candidate.

The evaluation set is a chronological, non-overlapping subset of the manually
reviewed ``Loga ORC - Primera`` session.  Its labels never select epochs,
temperature, architecture, or fusion weight.  Existing RGB feature caches are
reused; only the missing high-resolution YOLO cache is extracted.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import json
import os
import time
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from models.M2.evaluate_quantile_fusion import make_variant
from models.M2.extract_cache import extract_highres_clip
from models.M2.grid_search_cached_descriptors import primitive_base
from models.M3.run_temporal_architecture_suite import build_model, fit_normalizer, load_data, normalize_all, paired_test, probability_metrics, train_fixed
from models.M3.run_temporal_attention_experiment import IndexedSequenceDataset, quality_score, smooth_three, temperature_probabilities, usable_mask
REPO = Path(__file__).resolve().parents[2]
HERE = experiment_directory(__file__)
OUTPUT = HERE / 'outputs' / 'temporal_external_primera_20260815'
MANIFEST_SOURCE = REPO / 'data' / 'three_class_human_session_adaptation_20260721' / 'full' / 'human_test_manifest.csv'
REVIEW_CACHE = REPO / 'data' / 'line_false_positive_review_finetune_1800' / 'full' / 'cache'
STATIC_DEPLOYMENT = HERE / 'outputs' / 'deployment_quantile_100frames'
RGB_DEPLOYMENT = REPO / 'data' / 'joint_multimodal_20260813' / 'rgb_dynamics_seed42'
HISTORICAL_MODEL = REPO / 'models' / 'M0' / 'checkpoints' / 'best_model' / 'best_model.keras'
CONFIRMATION = HERE / 'outputs' / 'temporal_architecture_suite_20260814' / 'confirmation' / 'tcn_bigru_attention'
OPENVINO_MODEL = REPO / 'yolo11n-pose_openvino_model'
LABELS = np.asarray(['line', 'other', 'scrum'], dtype=object)
SEEDS = (42, 17, 2026)
TEMPORAL_SHARE_WITHIN_AUXILIARY = 0.15
EFFECTIVE_TEMPORAL_WEIGHT = 0.09

def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-07, None)
    return values / values.sum(axis=1, keepdims=True)

def prepare_manifest() -> pd.DataFrame:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / 'external_nonoverlap_manifest.csv'
    if _resolve_model_artifact(output_path).exists():
        return pd.read_csv(_resolve_model_artifact(output_path))
    frame = pd.read_csv(_resolve_model_artifact(MANIFEST_SOURCE)).sort_values(['source_video', 'start_time', 'end_time', 'dataset_id'])
    selected = []
    for _, group in frame.groupby('source_video', sort=False):
        last_end = -np.inf
        for row in group.to_dict('records'):
            if float(row['start_time']) >= last_end:
                selected.append(row)
                last_end = float(row['end_time'])
    manifest = pd.DataFrame(selected).reset_index(drop=True)
    index = json.loads(_resolve_model_artifact(REVIEW_CACHE / 'review_index.json').read_text(encoding='utf-8'))
    feature_lookup = {dataset_id: position for position, dataset_id in enumerate(index['dataset_ids'])}
    manifest['feature_index'] = manifest['dataset_id'].map(feature_lookup)
    manifest['temporal_block'] = (manifest['start_time'].astype(float) // 120.0).astype(int)
    if manifest['feature_index'].isna().any():
        raise RuntimeError('Some external clips are absent from the frozen RGB cache')
    missing = [path for path in manifest['clip_path'] if not _resolve_model_artifact(Path(path)).exists()]
    if missing:
        raise FileNotFoundError(f'Missing external clips: {len(missing)}; first={missing[0]}')
    manifest.to_csv(output_path, index=False)
    dump_json(OUTPUT / 'manifest_audit.json', {'source_rows': int(len(frame)), 'selected_nonoverlap_rows': int(len(manifest)), 'class_counts': manifest['correct_label'].value_counts().to_dict(), 'source_videos': int(manifest['source_video'].nunique()), 'temporal_blocks_120s': int(manifest['temporal_block'].nunique()), 'selection_uses_labels': False, 'selection_rule': 'chronological greedy non-overlap: next start >= previous end'})
    return manifest

def training_config(seed: int) -> tuple[int, float]:
    summary = json.loads(_resolve_model_artifact(CONFIRMATION / f'seed_{seed}' / 'summary.json').read_text(encoding='utf-8'))
    epochs = int(round(np.median([fold['selected_epoch'] for fold in summary['folds']])))
    temperature = float(np.median([fold['temperature'] for fold in summary['folds']]))
    return (epochs, temperature)

def train_deployment_models(batch_size: int) -> None:
    root = OUTPUT / 'temporal_models'
    root.mkdir(parents=True, exist_ok=True)
    features, usable, q_score, _, labels, _, _, _, _ = load_data()
    indices = np.arange(len(labels), dtype=np.int64)
    mean, std = fit_normalizer(features, indices, usable)
    normalized = normalize_all(features, mean, std, usable)
    dataset = IndexedSequenceDataset(normalized, usable, q_score, labels)
    for seed in SEEDS:
        path = root / f'seed_{seed}.pt'
        if _resolve_model_artifact(path).exists():
            print(json.dumps({'stage': 'train_resume', 'seed': seed}), flush=True)
            continue
        epochs, temperature = training_config(seed)
        model, history = train_fixed('tcn_bigru_attention', dataset, indices, seed=seed, batch_size=batch_size, epochs=epochs)
        torch.save({'architecture': 'tcn_bigru_attention', 'state_dict': model.state_dict(), 'mean': mean, 'std': std, 'temperature': temperature, 'epochs': epochs, 'seed': seed, 'trained_clips': int(len(labels)), 'external_labels_used': False}, path)
        pd.DataFrame(history).to_csv(root / f'seed_{seed}_history.csv', index=False)
    dump_json(root / 'training_protocol.json', {'architecture': 'tcn_bigru_attention', 'training_clips': int(len(labels)), 'epochs_and_temperature': {str(seed): {'epochs': training_config(seed)[0], 'temperature': training_config(seed)[1]} for seed in SEEDS}, 'epoch_rule': 'rounded median of five nested-confirmation selected epochs', 'temperature_rule': 'median of five nested-confirmation temperatures', 'external_labels_used': False})

def extract_yolo(manifest: pd.DataFrame, batch_size: int) -> None:
    cache = OUTPUT / 'yolo_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(OUTPUT / '.ultralytics'))
    from ultralytics import YOLO
    model = YOLO(_resolve_model_artifact(str(OPENVINO_MODEL)))
    rows = []
    started = time.perf_counter()
    for position, row in enumerate(manifest.to_dict('records'), start=1):
        path = cache / f"{row['dataset_id']}.npz"
        if _resolve_model_artifact(path).exists():
            rows.append({'dataset_id': row['dataset_id'], 'status': 'cached'})
            continue
        clip_started = time.perf_counter()
        try:
            arrays, metadata = extract_highres_clip(model, Path(row['clip_path']), frame_count=100, image_size=640, person_threshold=0.25, keypoint_threshold=0.2, max_people=30, pre_crop_scale=0.5, crop_size=224, batch_size=batch_size, device='cpu')
            np.savez_compressed(path, **arrays)
            rows.append({'dataset_id': row['dataset_id'], 'status': 'ok', 'seconds': time.perf_counter() - clip_started, 'frames_with_persons': metadata['frames_with_persons'], 'mean_people': metadata['mean_people']})
        except Exception as exc:
            rows.append({'dataset_id': row['dataset_id'], 'status': 'error', 'seconds': time.perf_counter() - clip_started, 'error': repr(exc)})
        pd.DataFrame(rows).to_csv(OUTPUT / 'yolo_extraction_log.csv', index=False)
        print(json.dumps({'stage': 'external_yolo', 'clip': position, 'total': len(manifest), 'status': rows[-1]['status']}), flush=True)
    errors = [row for row in rows if row['status'] == 'error']
    dump_json(OUTPUT / 'yolo_extraction_summary.json', {'clips': len(manifest), 'errors': len(errors), 'seconds': time.perf_counter() - started, 'model': str(OPENVINO_MODEL.relative_to(REPO)), 'frames_per_clip': 100, 'batch_size': batch_size})
    if errors:
        raise RuntimeError(f'YOLO extraction failed for {len(errors)} clips')

def align(sequence: np.ndarray, length: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape[0] == length:
        return sequence
    old = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
    new = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.stack([np.interp(new, old, sequence[:, index]) for index in range(sequence.shape[1])], axis=1).astype(np.float32)

def rgb_dynamic_vector(features: np.ndarray, mask: np.ndarray, pca: dict) -> np.ndarray:
    length = max(1, int(np.sum(mask)))
    rgb = align(np.asarray(features[:length], dtype=np.float32), 200)
    segments = rgb.reshape(10, 20, 2048)
    segment_mean = segments.mean(axis=1)
    segment_delta = np.abs(np.diff(segments, axis=1)).mean(axis=1)
    global_mean = rgb.mean(axis=0)
    global_std = rgb.std(axis=0)
    return np.concatenate([pca['segment'].transform(segment_mean).reshape(-1), pca['delta'].transform(segment_delta).reshape(-1), pca['mean'].transform(global_mean.reshape(1, -1)).reshape(-1), pca['std'].transform(global_std.reshape(1, -1)).reshape(-1)]).astype(np.float32)

def build_external_features(manifest: pd.DataFrame):
    feature_path = OUTPUT / 'external_features.npz'
    if _resolve_model_artifact(feature_path).exists():
        with _load_model_result_array(feature_path) as cached:
            return {key: cached[key] for key in cached.files}
    review_features = _load_model_result_array(REVIEW_CACHE / 'review_features.npy', mmap_mode='r')
    review_masks = _load_model_result_array(REVIEW_CACHE / 'review_masks.npy', mmap_mode='r')
    pca = joblib.load(_resolve_model_artifact(RGB_DEPLOYMENT / 'dynamic_pca.joblib'))
    dynamic_rows = []
    yolo_rows = []
    temporal_rows = []
    quality_rows = []
    mask_rows = []
    for position, row in enumerate(manifest.to_dict('records'), start=1):
        feature_index = int(row['feature_index'])
        rgb_features = np.asarray(review_features[feature_index], dtype=np.float32)
        rgb_mask = np.asarray(review_masks[feature_index], dtype=bool)
        dynamic_rows.append(rgb_dynamic_vector(rgb_features, rgb_mask, pca))
        with _load_model_result_array(OUTPUT / 'yolo_cache' / f"{row['dataset_id']}.npz") as archive:
            raw = {key: np.asarray(archive[key]) for key in archive.files}
        yolo_rows.append(np.concatenate([make_variant(raw, 'E04_tracking_smoothing'), make_variant(raw, 'E06_rugby_descriptors')]).astype(np.float32))
        primitive = primitive_base(raw)
        frame_mask = primitive['mask'].astype(bool)
        quality = primitive['quality'].astype(np.float32)
        current = smooth_three(primitive['current'][None], frame_mask[None])[0]
        rugby = smooth_three(primitive['rugby'][None], frame_mask[None])[0]
        temporal_rows.append(np.nan_to_num(np.concatenate([current, quality, rugby], axis=1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        score = quality_score(quality[None])[0]
        quality_rows.append(score)
        mask_rows.append(usable_mask(frame_mask[None], score[None])[0])
        if position % 25 == 0:
            print(json.dumps({'stage': 'external_features', 'clips': position}), flush=True)
    result = {'rgb_features': np.asarray(review_features[manifest['feature_index'].astype(int).to_numpy()], dtype=np.float32), 'rgb_masks': np.asarray(review_masks[manifest['feature_index'].astype(int).to_numpy()], dtype=bool), 'dynamic': np.asarray(dynamic_rows, dtype=np.float32), 'yolo_static': np.asarray(yolo_rows, dtype=np.float32), 'temporal': np.asarray(temporal_rows, dtype=np.float32), 'quality': np.asarray(quality_rows, dtype=np.float32), 'mask': np.asarray(mask_rows, dtype=bool)}
    np.savez_compressed(feature_path, **result)
    return result

def temporal_predict(features: dict, seed: int, batch_size: int) -> np.ndarray:
    checkpoint = torch.load(_resolve_model_artifact(OUTPUT / 'temporal_models' / f'seed_{seed}.pt'), map_location='cpu', weights_only=False)
    x = (features['temporal'] - checkpoint['mean']) / checkpoint['std']
    x = np.clip(x, -8.0, 8.0).astype(np.float32)
    x[~features['mask']] = 0.0
    model = build_model('tcn_bigru_attention')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            batch_logits = model(torch.from_numpy(x[start:stop]), torch.from_numpy(features['mask'][start:stop]), torch.from_numpy(features['quality'][start:stop]))
            logits.append(batch_logits.numpy())
    return temperature_probabilities(np.concatenate(logits, axis=0), checkpoint['temperature'])

def evaluate(manifest: pd.DataFrame, batch_size: int) -> None:
    features = build_external_features(manifest)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from tensorflow import keras
    historical_model = keras.models.load_model(_resolve_model_artifact(HISTORICAL_MODEL))
    historical = historical_model.predict([features['rgb_features'], features['rgb_masks']], batch_size=batch_size, verbose=0)
    historical = normalize_probabilities(historical)
    scaler = joblib.load(_resolve_model_artifact(STATIC_DEPLOYMENT / 'quantile_scaler.joblib'))
    classifier = joblib.load(_resolve_model_artifact(STATIC_DEPLOYMENT / 'logistic.joblib'))
    auxiliary_features = np.concatenate([features['dynamic'], features['yolo_static']], axis=1)
    static = normalize_probabilities(classifier.predict_proba(scaler.transform(auxiliary_features)))
    official = 0.4 * historical + 0.6 * static
    labels = np.asarray([int(np.flatnonzero(LABELS == label)[0]) for label in manifest['correct_label']], dtype=np.int64)
    groups = manifest['temporal_block'].astype(str).to_numpy()
    temporal_by_seed = {}
    candidate_by_seed = {}
    evaluations = {}
    for seed in SEEDS:
        temporal = temporal_predict(features, seed, batch_size)
        candidate = 0.4 * historical + 0.51 * static + EFFECTIVE_TEMPORAL_WEIGHT * temporal
        temporal_by_seed[seed] = temporal
        candidate_by_seed[seed] = candidate
        evaluations[str(seed)] = {'temporal': probability_metrics(labels, temporal), 'candidate': probability_metrics(labels, candidate), 'paired_against_official': paired_test(labels, official, candidate, groups, seed=20260815 + seed)}
    ensemble_temporal = np.mean(list(temporal_by_seed.values()), axis=0)
    ensemble_candidate = 0.4 * historical + 0.51 * static + 0.09 * ensemble_temporal
    evaluations['ensemble'] = {'temporal': probability_metrics(labels, ensemble_temporal), 'candidate': probability_metrics(labels, ensemble_candidate), 'paired_against_official': paired_test(labels, official, ensemble_candidate, groups, seed=20260815)}
    sensitivity = {}
    for temporal_weight in np.arange(0.0, 0.3001, 0.03):
        temporal_weight = float(round(temporal_weight, 2))
        weighted_candidate = 0.4 * historical + (0.6 - temporal_weight) * static + temporal_weight * ensemble_temporal
        sensitivity[f'{temporal_weight:.2f}'] = {'metrics': probability_metrics(labels, weighted_candidate), 'paired_against_official': paired_test(labels, official, weighted_candidate, groups, seed=20260815 + int(100 * temporal_weight))}
    official_metrics = probability_metrics(labels, official)
    ensemble_metrics = evaluations['ensemble']['candidate']
    ensemble_paired = evaluations['ensemble']['paired_against_official']
    promotion = {'decision': 'do_not_promote', 'reason': 'The preregistered three-seed ensemble does not improve external accuracy and reduces balanced accuracy and macro-F1; its block-bootstrap interval includes zero. Keep the temporal branch experimental.', 'accuracy_delta': float(ensemble_metrics['accuracy'] - official_metrics['accuracy']), 'balanced_accuracy_delta': float(ensemble_metrics['balanced_accuracy'] - official_metrics['balanced_accuracy']), 'macro_f1_delta': float(ensemble_metrics['f1_macro'] - official_metrics['f1_macro']), 'nll_delta': float(ensemble_metrics['negative_log_likelihood'] - official_metrics['negative_log_likelihood']), 'ece_delta': float(ensemble_metrics['ece_15'] - official_metrics['ece_15']), 'mcnemar_exact_p': float(ensemble_paired['mcnemar_exact_p']), 'block_bootstrap_accuracy_delta_ci95': ensemble_paired['source_group_bootstrap_delta_accuracy_ci95']}
    result = {'version': 2, 'dataset': 'Loga ORC - Primera, manually reviewed non-overlapping clips', 'clips': int(len(labels)), 'class_counts': manifest['correct_label'].value_counts().to_dict(), 'temporal_blocks_120s': int(len(set(groups))), 'external_labels_used_for_selection': False, 'formula': '0.40 historical + 0.51 RGB/YOLO static + 0.09 TCN+BiGRU temporal', 'official': official_metrics, 'historical': probability_metrics(labels, historical), 'static_auxiliary': probability_metrics(labels, static), 'evaluations': evaluations, 'posthoc_weight_sensitivity': {'uses_external_labels': True, 'must_not_be_used_for_model_selection': True, 'weights_are_final_temporal_share': True, 'results': sensitivity}, 'promotion_recommendation': promotion}
    dump_json(OUTPUT / 'external_validation.json', result)
    np.savez_compressed(OUTPUT / 'external_component_probabilities.npz', labels=labels, groups=groups, historical=historical, static=static, temporal_seed_42=temporal_by_seed[42], temporal_seed_17=temporal_by_seed[17], temporal_seed_2026=temporal_by_seed[2026], temporal_ensemble=ensemble_temporal)
    predictions = manifest[['dataset_id', 'start_time', 'end_time', 'correct_label', 'temporal_block']].copy()
    predictions['official_prediction'] = LABELS[official.argmax(axis=1)]
    for class_index, class_name in enumerate(LABELS):
        predictions[f'official_{class_name}'] = official[:, class_index]
    for seed, candidate in candidate_by_seed.items():
        predictions[f'candidate_seed_{seed}_prediction'] = LABELS[candidate.argmax(axis=1)]
        for class_index, class_name in enumerate(LABELS):
            predictions[f'candidate_seed_{seed}_{class_name}'] = candidate[:, class_index]
    predictions['candidate_ensemble_prediction'] = LABELS[ensemble_candidate.argmax(axis=1)]
    predictions.to_csv(OUTPUT / 'external_predictions.csv', index=False)
    report = ['# Validación externa del candidato temporal de 9%', '', 'Conjunto: 189 ventanas no superpuestas de `Loga ORC - Primera`, etiquetadas manualmente. Las etiquetas externas no participaron en la elección de arquitectura, época, temperatura ni peso.', '', 'Fusión evaluada: **40% histórico + 51% RGB/YOLO + 9% TCN+BiGRU**.', '', '| Modelo | Accuracy | Balanced accuracy | Macro-F1 | Δ accuracy | McNemar p | IC95% por bloques de 120 s |', '|---|---:|---:|---:|---:|---:|---|', f"| Oficial | {100 * official_metrics['accuracy']:.2f}% | {100 * official_metrics['balanced_accuracy']:.2f}% | {100 * official_metrics['f1_macro']:.2f}% | — | — | — |"]
    for key in [str(seed) for seed in SEEDS] + ['ensemble']:
        evaluation = evaluations[key]
        metrics = evaluation['candidate']
        paired = evaluation['paired_against_official']
        report.append(f"| Candidato {key} | {100 * metrics['accuracy']:.2f}% | {100 * metrics['balanced_accuracy']:.2f}% | {100 * metrics['f1_macro']:.2f}% | {100 * paired['delta_accuracy']:+.2f} pp | {paired['mcnemar_exact_p']:.4f} | [{100 * paired['source_group_bootstrap_delta_accuracy_ci95'][0]:+.2f}, {100 * paired['source_group_bootstrap_delta_accuracy_ci95'][1]:+.2f}] pp |")
    report.extend(['', '## Sensibilidad post hoc del peso temporal', '', 'Esta curva usa las etiquetas externas y, por lo tanto, **no puede utilizarse para elegir un nuevo peso**. Solo muestra cuánto cambia el resultado alrededor del candidato de 9% fijado antes de observar este partido.', '', '| Peso temporal final | Accuracy | Balanced accuracy | Macro-F1 | Δ accuracy |', '|---:|---:|---:|---:|---:|'])
    for weight, entry in sensitivity.items():
        metrics = entry['metrics']
        paired = entry['paired_against_official']
        report.append(f"| {100 * float(weight):.0f}% | {100 * metrics['accuracy']:.2f}% | {100 * metrics['balanced_accuracy']:.2f}% | {100 * metrics['f1_macro']:.2f}% | {100 * paired['delta_accuracy']:+.2f} pp |")
    report.extend(['', '## Decisión', '', '**No promover por ahora la rama temporal a la web.** El ensamble fijado en 9% conserva la misma accuracy que el modelo oficial, pero reduce balanced accuracy y macro-F1; además, el intervalo bootstrap por bloques incluye cero. Sí mejora NLL y ECE, por lo que la rama puede conservarse como experimento de calibración y retomarse con más partidos independientes.', '', 'Limitación: las 189 ventanas provienen de un único partido y de una sesión de revisión de detecciones. Aunque no se superponen y el bootstrap respeta bloques de 120 s, no equivalen a una muestra uniforme de varios partidos.'])
    (OUTPUT / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', 'train', 'extract', 'evaluate', 'all'), default='all')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
    manifest = prepare_manifest()
    if args.stage in ('train', 'all'):
        train_deployment_models(args.batch_size)
    if args.stage in ('extract', 'all'):
        extract_yolo(manifest, args.batch_size)
    if args.stage in ('evaluate', 'all'):
        evaluate(manifest, args.batch_size)
    if args.stage == 'prepare':
        print(_resolve_model_artifact(OUTPUT / 'manifest_audit.json').read_text(encoding='utf-8'))
if __name__ == '__main__':
    main()
