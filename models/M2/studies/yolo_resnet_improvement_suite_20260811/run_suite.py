"""Screening reproducible de diez mejoras para fusionar YOLO con ResNet-50.

La fase utiliza la cache YOLO de 200 pasos y el checkpoint historico congelado.
Cada experimento cambia exclusivamente la representacion geometrica, entrena
un clasificador de fusion sobre las probabilidades ResNet + el vector YOLO y
selecciona por macro-F1 de validacion. El test se informa, pero no participa
en la seleccion. Las variantes E7--E9 son los proxies de screening de las
arquitecturas gated, dual-temporal y graph-temporal; los modelos RNN completos
se reservan para los finalistas de esta fase.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import source_directory
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
import keras
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
if __name__ == '__main__':
    os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
    os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(REPO / 'data' / 'player_geometry_fusion' / 'pose_cache' / 'frames_200')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PLAYER_DIR = REPO / 'data' / 'player_geometry_fusion'
if str(PLAYER_DIR) not in sys.path:
    sys.path.insert(0, str(source_directory(PLAYER_DIR)))
from models.M0.studies.natural_with_human_review import run_experiment as common
from models.M0.studies.line_recall_training import run_experiment as recall_model
from models.M1.config import LABELS, LABEL_TO_ID, RESULTS_DIR
from models.M1.protocol import build_partition, pose_cache_path
from models.M1.targeted_geometry_experiments import POSTURE_CLEAN_INDICES, POSITION_CLEAN_INDICES, formation_frame, summarize
SOURCE_CHECKPOINT = REPO / 'data' / 'line_recall_training' / 'full' / 'training' / 'line_weight_1_25' / 'best_accuracy.keras'
POSE_MODEL = 'yolo11n-pose.pt'
SEQUENCE_LENGTH = 200
SCREENING_SEED = 42
EXPERIMENTS = {'E01_confidence_weighted': 'pooling ponderado por confianza de persona y articulaciones', 'E02_quality_mask': 'enmascarado de frames con baja cobertura o baja confianza', 'E03_roi_topk': 'ROI central y top-k jugadores por confianza', 'E04_tracking_smoothing': 'suavizado temporal robusto de detecciones', 'E05_camera_normalized': 'normalizacion relativa de camara y escala', 'E06_rugby_descriptors': 'descriptores especificos de scrum y line', 'E07_gated_residual': 'residuo YOLO con compuerta de calidad (screening)', 'E08_dual_temporal': 'ramas temporales separadas RGB/YOLO (screening)', 'E09_player_graph': 'descriptores de grafo de jugadores y dinamica', 'E10_calibrated_late_fusion': 'calibracion y mezcla tardia de expertos'}

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value

def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False), encoding='utf-8')

def load_clip(name: str) -> dict[str, np.ndarray]:
    path = pose_cache_path(name, POSE_MODEL)
    with _load_model_result_array(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}

def quality_channels(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = data['valid'].astype(bool)
    person_conf = data['person_conf'].astype(np.float32)
    keypoint_conf = data['keypoint_conf'].astype(np.float32)
    count = valid.sum(axis=1).astype(np.float32)
    person_mean = np.divide(np.where(valid, person_conf, 0.0).sum(axis=1), np.maximum(count, 1.0))
    keypoint_mean = np.divide(np.where(valid[..., None], keypoint_conf, 0.0).sum(axis=(1, 2)), np.maximum(count * keypoint_conf.shape[-1], 1.0))
    coverage = np.clip(count / 30.0, 0.0, 1.0)
    quality = np.column_stack([count / 30.0, person_mean, keypoint_mean, coverage]).astype(np.float32)
    frame_ok = (person_mean >= 0.25) & (keypoint_mean >= 0.2) & (count >= 1)
    frame_ok &= data['frame_mask'].astype(bool)
    return (quality, frame_ok)

def robust_smooth(sequence: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median smoothing that never changes the number of temporal steps."""
    values = np.asarray(sequence, dtype=np.float32).copy()
    if values.shape[0] < 3:
        return values
    padded = np.pad(values, ((1, 1), (0, 0)), mode='edge')
    smooth = np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]], axis=0), axis=0)
    smooth[~np.asarray(valid, dtype=bool)] = values[~np.asarray(valid, dtype=bool)]
    return smooth.astype(np.float32)

def relative_positions(data: dict[str, np.ndarray]) -> np.ndarray:
    """Centers and scales the position descriptors per frame."""
    boxes = data['boxes'].astype(np.float32)
    valid = data['valid'].astype(bool)
    out = np.zeros_like(data['position_features'], dtype=np.float32)
    for index in range(len(boxes)):
        keep = valid[index]
        if not keep.any():
            continue
        centers = (boxes[index, :, :2] + boxes[index, :, 2:]) * 0.5
        centers = centers[keep]
        center = centers.mean(axis=0)
        scale = float(np.linalg.norm(np.ptp(centers, axis=0))) + 0.0001
        out[index] = data['position_features'][index]
        out[index, :2] = (out[index, :2] - center[0]) / scale
        out[index, 2:4] = (out[index, 2:4] - center[1]) / scale
        out[index, 4:6] /= scale
        out[index, 6:8] /= scale
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

def graph_frame(data: dict[str, np.ndarray], index: int) -> np.ndarray:
    boxes = data['boxes'][index].astype(np.float32)
    keep = data['valid'][index].astype(bool)
    conf = data['person_conf'][index].astype(np.float32)
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    centers = centers[keep]
    conf = conf[keep]
    if len(centers) < 2:
        return np.zeros(16, dtype=np.float32)
    delta = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1))
    upper = distances[np.triu_indices(len(centers), k=1)]
    nearest = distances.copy()
    np.fill_diagonal(nearest, np.nan)
    nearest = np.nanmin(nearest, axis=1)
    weights = conf[:, None] * conf[None, :]
    weighted_edges = weights[np.triu_indices(len(centers), k=1)]
    covariance = np.atleast_2d(np.cov(centers - centers.mean(axis=0), rowvar=False))
    eigen = np.linalg.eigvalsh(covariance)
    eigen = np.sort(np.maximum(eigen, 0.0))[::-1]
    return np.asarray([len(centers) / 30.0, upper.mean(), upper.std(), np.quantile(upper, 0.25), np.quantile(upper, 0.75), nearest.mean(), nearest.std(), np.quantile(nearest, 0.5), weighted_edges.mean(), weighted_edges.std(), eigen[0], eigen[-1], eigen[-1] / (eigen[0] + 1e-06), float(np.linalg.norm(centers.std(axis=0))), float(np.ptp(centers[:, 0])), float(np.ptp(centers[:, 1]))], dtype=np.float32)

def rugby_frame(data: dict[str, np.ndarray], index: int) -> np.ndarray:
    formation = formation_frame(data['boxes'][index], data['person_conf'][index], data['valid'][index])
    pose = data['pose_features'][index].astype(np.float32)
    keypoint_conf = data['keypoint_conf'][index].astype(np.float32)
    valid = data['valid'][index].astype(bool)
    if valid.any():
        posture = np.atleast_1d(pose[valid].mean(axis=0)).astype(np.float32)
        kp = np.asarray([keypoint_conf[valid].mean()], dtype=np.float32)
    else:
        posture = np.zeros(1, dtype=np.float32)
        kp = np.zeros(1, dtype=np.float32)
    boxes = data['boxes'][index][valid]
    if len(boxes):
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        spread = centers.std(axis=0)
        extent = np.ptp(centers, axis=0) if len(centers) > 1 else np.zeros(2)
    else:
        spread = extent = np.zeros(2)
    return np.concatenate([formation, posture, kp, spread, extent]).astype(np.float32)

def make_variant(data: dict[str, np.ndarray], experiment: str) -> np.ndarray:
    positions = data['position_features'].astype(np.float32)
    poses = data['pose_features'].astype(np.float32)
    mask = data['frame_mask'].astype(bool)
    quality, quality_mask = quality_channels(data)
    current = np.concatenate([positions, poses], axis=1)
    posture = poses[:, POSTURE_CLEAN_INDICES]
    if experiment == 'E01_confidence_weighted':
        weights = np.clip(quality[:, 1] * quality[:, 2] * quality[:, 3], 0.0, 1.0)
        weighted_mask = mask & (weights > 0)
        return np.concatenate([summarize(current, weighted_mask), summarize(quality, mask)])
    if experiment == 'E02_quality_mask':
        return np.concatenate([summarize(current, mask & quality_mask), summarize(quality, mask)])
    if experiment == 'E03_roi_topk':
        selected_frames = []
        for box, conf, valid in zip(data['boxes'], data['person_conf'], data['valid']):
            keep = np.asarray(valid, dtype=bool).copy()
            if keep.any():
                centers = (box[:, :2] + box[:, 2:]) * 0.5
                centrality = 1.0 - np.linalg.norm(centers - 0.5, axis=1) / 0.71
                score = np.asarray(conf, dtype=np.float32) * np.clip(centrality, 0.25, 1.0)
                indexes = np.flatnonzero(keep)
                top = indexes[np.argsort(score[indexes])[::-1][:12]]
                keep[:] = False
                keep[top] = True
            selected_frames.append(formation_frame(box, conf, keep))
        formation = np.stack(selected_frames)
        return np.concatenate([summarize(formation, mask, True), summarize(posture, mask)])
    if experiment == 'E04_tracking_smoothing':
        smoothed = robust_smooth(np.concatenate([positions, poses], axis=1), mask)
        return np.concatenate([summarize(smoothed, mask, True), summarize(quality, mask)])
    if experiment == 'E05_camera_normalized':
        normalized = np.concatenate([relative_positions(data), posture], axis=1)
        return np.concatenate([summarize(normalized, mask, True), summarize(quality, mask)])
    if experiment == 'E06_rugby_descriptors':
        rugby = np.stack([rugby_frame(data, i) for i in range(len(mask))])
        return summarize(rugby, mask, True)
    if experiment == 'E07_gated_residual':
        gate = 1.0 / (1.0 + np.exp(-8.0 * (quality[:, 1] - 0.35))) * quality[:, 3]
        gated = current * gate[:, None]
        return np.concatenate([summarize(current, mask), summarize(gated, mask), summarize(quality, mask)])
    if experiment == 'E08_dual_temporal':
        formation = np.stack([formation_frame(box, conf, valid) for box, conf, valid in zip(data['boxes'], data['person_conf'], data['valid'])])
        return np.concatenate([summarize(current, mask, True), summarize(formation, mask, True), summarize(posture, mask, True)])
    if experiment == 'E09_player_graph':
        graph = np.stack([graph_frame(data, i) for i in range(len(mask))])
        return np.concatenate([summarize(graph, mask, True), summarize(quality, mask, True)])
    raise ValueError(f'Experimento no implementado en screening: {experiment}')

def feature_matrix(frame: pd.DataFrame, experiment: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values, labels, names = ([], [], [])
    for row in frame.to_dict('records'):
        data = load_clip(str(row['video_name']))
        values.append(make_variant(data, experiment))
        labels.append(LABEL_TO_ID[str(row['tag'])])
        names.append(str(row['video_name']))
    matrix = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return (matrix, np.asarray(labels, dtype=np.int32), names)

def historical_probabilities(model: keras.Model, rows: pd.DataFrame, store: common.FeatureStore) -> np.ndarray:
    frame_rows = recall_model.frame_rows(rows, store)
    data = recall_model.FeatureBatches(frame_rows, store, False)
    return recall_model.predict_probabilities(model, data)

def metric_row(y_true: np.ndarray, probabilities: np.ndarray, split: str, experiment: str, seconds: float) -> dict:
    pred = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, pred, labels=np.arange(len(LABELS)), zero_division=0)
    return {'experiment': experiment, 'split': split, 'seed': SCREENING_SEED, 'accuracy': float(accuracy_score(y_true, pred)), 'f1_macro': float(f1_score(y_true, pred, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(y_true, pred, average='weighted', zero_division=0)), 'precision_line': float(precision[0]), 'recall_line': float(recall[0]), 'f1_line': float(f1[0]), 'precision_other': float(precision[1]), 'recall_other': float(recall[1]), 'f1_other': float(f1[1]), 'precision_scrum': float(precision[2]), 'recall_scrum': float(recall[2]), 'f1_scrum': float(f1[2]), 'support_line': int(support[0]), 'support_other': int(support[1]), 'support_scrum': int(support[2]), 'seconds': float(seconds)}

def fit_fusion(train_x, train_y, val_x, val_y, test_x, test_y, seed: int):
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)
    model = LogisticRegression(C=0.25, max_iter=1000, class_weight='balanced', random_state=seed)
    model.fit(train_x, train_y)
    return (model.predict_proba(val_x), model.predict_proba(test_x), scaler, model)

def save_predictions(path: Path, names: list[str], labels: np.ndarray, probabilities: np.ndarray) -> None:
    pred = probabilities.argmax(axis=1)
    frame = pd.DataFrame({'video_name': names, 'true_label': [LABELS[i] for i in labels], 'predicted_label': [LABELS[i] for i in pred]})
    for i, label in enumerate(LABELS):
        frame[f'prob_{label}'] = probabilities[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def late_fusion_grid(base_val, y_val, geom_val, base_test, y_test, geom_test):
    best = None
    for alpha in np.linspace(0.0, 1.0, 101):
        probabilities_val = (1.0 - alpha) * base_val + alpha * geom_val
        score = float(f1_score(y_val, probabilities_val.argmax(axis=1), average='macro', zero_division=0))
        if best is None or score > best['validation_f1_macro']:
            best = {'alpha': float(alpha), 'validation_f1_macro': score}
    alpha = best['alpha']
    best['test'] = (1.0 - alpha) * base_test + alpha * geom_test
    best['validation'] = (1.0 - alpha) * base_val + alpha * geom_val
    return best

def run(seed: int, output_root: Path) -> None:
    global SCREENING_SEED
    SCREENING_SEED = int(seed)
    train_df, validation_df, test_df, partition = build_partition(seed, require_geometry=True, pose_model=POSE_MODEL)
    store = common.load_base_store()
    source_hash = hashlib.sha256(_resolve_model_artifact(SOURCE_CHECKPOINT).read_bytes()).hexdigest()
    model = keras.models.load_model(_resolve_model_artifact(SOURCE_CHECKPOINT), compile=False)
    base_train = historical_probabilities(model, train_df, store)
    base_val = historical_probabilities(model, validation_df, store)
    base_test = historical_probabilities(model, test_df, store)
    y_train = train_df['tag'].map(LABEL_TO_ID).to_numpy(dtype=np.int32)
    y_val = validation_df['tag'].map(LABEL_TO_ID).to_numpy(dtype=np.int32)
    y_test = test_df['tag'].map(LABEL_TO_ID).to_numpy(dtype=np.int32)
    names = {'train': train_df.video_name.astype(str).tolist(), 'validation': validation_df.video_name.astype(str).tolist(), 'test': test_df.video_name.astype(str).tolist()}
    all_rows = []
    cached = {}
    for experiment in list(EXPERIMENTS)[:9]:
        started = time.perf_counter()
        train_features, _, _ = feature_matrix(train_df, experiment)
        val_features, _, _ = feature_matrix(validation_df, experiment)
        test_features, _, _ = feature_matrix(test_df, experiment)
        cached[experiment] = (train_features, val_features, test_features)
        train_x = np.concatenate([train_features, base_train], axis=1)
        val_x = np.concatenate([val_features, base_val], axis=1)
        test_x = np.concatenate([test_features, base_test], axis=1)
        val_prob, test_prob, scaler, classifier = fit_fusion(train_x, y_train, val_x, y_val, test_x, y_test, seed)
        seconds = time.perf_counter() - started
        root = output_root / experiment / f'seed_{seed}'
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / 'feature_shapes.npz', train=np.asarray(train_features.shape), validation=np.asarray(val_features.shape), test=np.asarray(test_features.shape))
        save_predictions(root / 'validation_predictions.csv', names['validation'], y_val, val_prob)
        save_predictions(root / 'test_predictions.csv', names['test'], y_test, test_prob)
        val_row = metric_row(y_val, val_prob, 'validation', experiment, seconds)
        test_row = metric_row(y_test, test_prob, 'test', experiment, seconds)
        all_rows.extend([val_row, test_row])
        write_json(root / 'metrics.json', {'experiment': experiment, 'description': EXPERIMENTS[experiment], 'seed': seed, 'feature_dim': int(train_features.shape[1]), 'classifier': 'StandardScaler + LogisticRegression on YOLO vector + frozen historical probabilities', 'validation': val_row, 'test': test_row})
        print(json.dumps({'experiment': experiment, 'validation_f1_macro': val_row['f1_macro'], 'test_f1_macro': test_row['f1_macro'], 'test_accuracy': test_row['accuracy']}, ensure_ascii=False), flush=True)
    val_rows = [row for row in all_rows if row['split'] == 'validation']
    best_name = max(val_rows, key=lambda row: row['f1_macro'])['experiment']
    best_train, best_val, best_test = cached[best_name]
    yolo = LogisticRegression(C=0.25, max_iter=1000, class_weight='balanced', random_state=seed)
    scaler = StandardScaler().fit(best_train)
    yolo.fit(scaler.transform(best_train), y_train)
    yolo_val = yolo.predict_proba(scaler.transform(best_val))
    yolo_test = yolo.predict_proba(scaler.transform(best_test))
    calibrated = late_fusion_grid(base_val, y_val, yolo_val, base_test, y_test, yolo_test)
    e10 = 'E10_calibrated_late_fusion'
    root = output_root / e10 / f'seed_{seed}'
    root.mkdir(parents=True, exist_ok=True)
    save_predictions(root / 'validation_predictions.csv', names['validation'], y_val, calibrated['validation'])
    save_predictions(root / 'test_predictions.csv', names['test'], y_test, calibrated['test'])
    val_row = metric_row(y_val, calibrated['validation'], 'validation', e10, 0.0)
    test_row = metric_row(y_test, calibrated['test'], 'test', e10, 0.0)
    all_rows.extend([val_row, test_row])
    write_json(root / 'metrics.json', {'experiment': e10, 'description': EXPERIMENTS[e10], 'selected_geometry_expert_by': 'validation_f1_macro', 'selected_geometry_expert': best_name, 'alpha': calibrated['alpha'], 'validation': val_row, 'test': test_row})
    print(json.dumps({'experiment': e10, 'selected_geometry_expert': best_name, 'alpha': calibrated['alpha'], 'validation_f1_macro': val_row['f1_macro'], 'test_f1_macro': test_row['f1_macro']}, ensure_ascii=False), flush=True)
    summary = pd.DataFrame(all_rows)
    summary.to_csv(output_root / 'summary.csv', index=False)
    validation = summary[summary.split == 'validation'].sort_values('f1_macro', ascending=False)
    validation.to_csv(output_root / 'validation_ranking.csv', index=False)
    test = summary[summary.split == 'test']
    test.to_csv(output_root / 'test_metrics_after_selection.csv', index=False)
    write_json(output_root / 'protocol.json', {'suite': 'yolo_resnet_improvement_suite_20260811', 'seed': seed, 'pose_model': POSE_MODEL, 'geometry_length': SEQUENCE_LENGTH, 'historical_checkpoint': str(SOURCE_CHECKPOINT), 'historical_checkpoint_sha256': source_hash, 'selection': 'validation macro F1 only; test locked until after ranking', 'partition': partition, 'data': EXPERIMENTS, 'screening_note': 'E01-E09 use a common regularized fusion classifier; E07-E09 are architecture proxies. Full RNN finalists are a second stage.', 'software': {'python': platform.python_version(), 'keras': keras.__version__, 'numpy': np.__version__}})
    del model, store
    keras.backend.clear_session()
    gc.collect()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', default=str(HERE / 'screening_seed42'))
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    run(args.seed, root)
    print(f'Resultados: {root}', flush=True)
if __name__ == '__main__':
    main()
