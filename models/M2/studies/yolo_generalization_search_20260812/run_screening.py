"""BÃºsqueda amplia de representaciones y cabezas YOLO+ResNet sobre cache 200.

El histÃ³rico ResNet-50 + BiGRU queda congelado. Se comparan representaciones
YOLO globales, multiescala, con filtros de calidad, trayectorias y segmentos
temporales. Las cabezas reciben tambiÃ©n los logits del histÃ³rico. La selecciÃ³n
usa sÃ³lo la validaciÃ³n natural y el test se informa despuÃ©s.
"""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
os.environ['PLAYER_GEOMETRY_LENGTH'] = '200'
os.environ['PLAYER_GEOMETRY_POSE_CACHE_ROOT'] = str(experiment_directory(__file__).parents[1] / 'data' / 'player_geometry_fusion' / 'pose_cache' / 'frames_200')
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
REPO = experiment_directory(__file__).parents[1]
HERE = experiment_directory(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from models.M2.studies.historical_locked_yolo_residual_20260811 import run_experiment as residual
from models.M2.studies.yolo_resnet_improvement_suite_20260811 import run_suite
OUTPUT = HERE / 'screening_seed42'
REPRESENTATIONS = ('e04_global', 'quality_multiscale', 'trajectory_multiscale', 'segment_quality', 'formation_temporal')
LABELS = residual.LABELS

def write_json(path: Path, value) -> None:

    def safe(item):
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(v) for v in item]
        if isinstance(item, np.ndarray):
            return [safe(v) for v in item.tolist()]
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, ensure_ascii=False), encoding='utf-8')

def quality(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = data['valid'].astype(bool)
    person = data['person_conf'].astype(np.float32)
    keypoint = data['keypoint_conf'].astype(np.float32)
    count = valid.sum(axis=1).astype(np.float32)
    person_mean = np.divide(np.where(valid, person, 0).sum(axis=1), np.maximum(count, 1))
    keypoint_mean = np.divide(np.where(valid[..., None], keypoint, 0).sum(axis=(1, 2)), np.maximum(count * keypoint.shape[-1], 1))
    coverage = np.clip(count / 30.0, 0, 1)
    q = np.column_stack([count / 30.0, person_mean, keypoint_mean, coverage]).astype(np.float32)
    mask = data['frame_mask'].astype(bool) & (person_mean >= 0.25) & (keypoint_mean >= 0.2) & (count >= 1)
    return (q, mask)

def slope(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros(values.shape[1], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, len(values), dtype=np.float32)
    x = x - x.mean()
    return (x[:, None] * (values - values.mean(axis=0))).sum(axis=0) / (float((x * x).sum()) + 1e-06)

def stats(values: np.ndarray, mask: np.ndarray, extra: bool=True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    valid = values[mask] if mask.any() else values
    if len(valid) == 0:
        valid = np.zeros((1, values.shape[-1]), dtype=np.float32)
    valid = np.nan_to_num(valid, nan=0, posinf=0, neginf=0)
    delta = np.diff(valid, axis=0) if len(valid) > 1 else np.zeros((1, valid.shape[1]), dtype=np.float32)
    out = [valid.mean(0), valid.std(0), valid.min(0), valid.max(0), np.quantile(valid, 0.25, axis=0), np.quantile(valid, 0.75, axis=0), delta.mean(0), delta.std(0), np.abs(delta).mean(0)]
    if extra:
        out.extend([slope(valid), valid[-1] - valid[0]])
    return np.concatenate(out).astype(np.float32)

def segment_stats(values: np.ndarray, mask: np.ndarray, segments: int=8) -> np.ndarray:
    chunks = []
    for left, right in zip(np.linspace(0, len(values), segments, endpoint=False, dtype=int), np.linspace(0, len(values), segments + 1, dtype=int)[1:]):
        chunk = values[left:right]
        chunk_mask = mask[left:right]
        chunks.append(stats(chunk, chunk_mask, extra=True))
        chunks.append(np.asarray([float(chunk_mask.mean())], dtype=np.float32))
    return np.concatenate(chunks).astype(np.float32)

def raw_sequence(data: dict[str, np.ndarray]) -> np.ndarray:
    q, _ = quality(data)
    sequence = np.concatenate([data['position_features'], data['pose_features'], q], axis=1).astype(np.float32)
    return np.nan_to_num(sequence, nan=0, posinf=0, neginf=0)

def make_representation(data: dict[str, np.ndarray], name: str) -> np.ndarray:
    sequence = raw_sequence(data)
    mask = data['frame_mask'].astype(bool)
    q, qmask = quality(data)
    if name == 'e04_global':
        return run_suite.make_variant(data, 'E04_tracking_smoothing')
    if name == 'quality_multiscale':
        return np.concatenate([stats(sequence, mask), stats(sequence, mask & qmask), segment_stats(sequence, mask, 8), stats(q, mask, True)]).astype(np.float32)
    if name == 'trajectory_multiscale':
        delta = np.diff(sequence, axis=0, prepend=sequence[:1])
        acceleration = np.diff(delta, axis=0, prepend=delta[:1])
        return np.concatenate([stats(sequence, mask), segment_stats(delta, mask, 8), stats(acceleration, mask, True), stats(q, mask, True)]).astype(np.float32)
    if name == 'segment_quality':
        return np.concatenate([segment_stats(sequence, mask & qmask, 16), segment_stats(q, mask, 16)]).astype(np.float32)
    if name == 'formation_temporal':
        formation = []
        for index in range(len(mask)):
            formation.append(run_suite.formation_frame(data['boxes'][index], data['person_conf'][index], data['valid'][index]))
        formation = np.asarray(formation, dtype=np.float32)
        return np.concatenate([segment_stats(formation, mask, 16), stats(sequence, mask, True), stats(q, mask, True)]).astype(np.float32)
    raise ValueError(name)

def build_features(frame: pd.DataFrame, name: str, split: str) -> np.ndarray:
    cache = OUTPUT / 'feature_cache' / name / f'{split}.npy'
    names = np.asarray(frame.video_name.astype(str).tolist(), dtype=str)
    if _resolve_model_artifact(cache).exists() and _resolve_model_artifact(cache.with_suffix('.names.npy')).exists():
        old = _load_model_result_array(cache.with_suffix('.names.npy'), allow_pickle=False).astype(str)
        if np.array_equal(old, names):
            return _load_model_result_array(cache, mmap_mode='r')
    values = []
    started = time.perf_counter()
    for index, video_name in enumerate(names, start=1):
        values.append(make_representation(run_suite.load_clip(video_name), name))
        if index % 250 == 0 or index == len(names):
            print(f'features {name}/{split}: {index}/{len(names)} elapsed={time.perf_counter() - started:.1f}s', flush=True)
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0, posinf=0, neginf=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, values)
    np.save(cache.with_suffix('.names.npy'), names)
    return values

def metric(y: np.ndarray, prob: np.ndarray) -> dict:
    pred = prob.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(3), zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)), 'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(y, pred, average='weighted', zero_division=0)), 'recall_line': float(recall[0]), 'recall_other': float(recall[1]), 'recall_scrum': float(recall[2]), 'f1_line': float(f1[0]), 'f1_other': float(f1[1]), 'f1_scrum': float(f1[2]), 'support_line': int(support[0]), 'support_other': int(support[1]), 'support_scrum': int(support[2])}

def make_head(name: str, seed: int):
    if name.startswith('logistic'):
        balanced = 'balanced' in name
        c = float(name.split('_')[-1])
        return LogisticRegression(C=c, max_iter=3000, class_weight='balanced' if balanced else None, solver='lbfgs', random_state=seed)
    if name == 'svc_rbf':
        return SVC(C=2.0, gamma='scale', probability=True, class_weight='balanced', random_state=seed)
    if name == 'extra_trees':
        return ExtraTreesClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=seed, n_jobs=-1)
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=500, max_features='sqrt', min_samples_leaf=2, class_weight='balanced', random_state=seed, n_jobs=-1)
    if name == 'hist_gradient':
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed)
    if name == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.001, learning_rate_init=0.0003, early_stopping=True, validation_fraction=0.15, max_iter=250, random_state=seed)
    raise ValueError(name)

def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, validation, test = residual.common.load_base_splits(False, 'natural')
    y_train, y_validation, y_test = map(residual.labels, (train, validation, test))
    cache_root = OUTPUT / 'base_cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    base_train = residual.historical_train_probabilities(train, cache_root)
    base_validation = residual.stored_probabilities(residual.REPRODUCTION / 'checkpoint_validation_predictions.csv', validation)
    base_test = residual.stored_probabilities(residual.REPRODUCTION / 'checkpoint_test_predictions.csv', test)
    baseline_validation = metric(y_validation, base_validation)
    baseline_test = metric(y_test, base_test)
    heads = ('logistic_balanced_0.01', 'logistic_balanced_0.05', 'logistic_balanced_0.25', 'logistic_0.05', 'svc_rbf', 'extra_trees', 'random_forest', 'hist_gradient', 'mlp')
    candidates = []
    fitted = {}
    for representation in REPRESENTATIONS:
        train_geom = build_features(train, representation, 'train')
        validation_geom = build_features(validation, representation, 'validation')
        test_geom = build_features(test, representation, 'test')
        scaler = StandardScaler().fit(train_geom)
        train_scaled = scaler.transform(train_geom).astype(np.float32)
        val_scaled = scaler.transform(validation_geom).astype(np.float32)
        test_scaled = scaler.transform(test_geom).astype(np.float32)
        components = min(128, train_scaled.shape[1], train_scaled.shape[0] - 1)
        pca = PCA(n_components=components, random_state=42).fit(train_scaled)
        train_reduced = pca.transform(train_scaled).astype(np.float32)
        val_reduced = pca.transform(val_scaled).astype(np.float32)
        test_reduced = pca.transform(test_scaled).astype(np.float32)
        for base_mode in ('prob', 'logit'):
            train_base = base_train if base_mode == 'prob' else residual.logits(base_train)
            val_base = base_validation if base_mode == 'prob' else residual.logits(base_validation)
            test_base = base_test if base_mode == 'prob' else residual.logits(base_test)
            train_x = np.concatenate([train_base, train_reduced], axis=1)
            val_x = np.concatenate([val_base, val_reduced], axis=1)
            test_x = np.concatenate([test_base, test_reduced], axis=1)
            for head_name in heads:
                model = make_head(head_name, 42)
                if head_name == 'hist_gradient':
                    weights = np.asarray([1.25 if label == 0 else 1.0 for label in y_train], dtype=np.float32)
                    model.fit(train_x, y_train, sample_weight=weights)
                else:
                    model.fit(train_x, y_train)
                val_prob = residual.normalize(model.predict_proba(val_x))
                row = {'representation': representation, 'base_mode': base_mode, 'head': head_name, **metric(y_validation, val_prob)}
                row['delta_accuracy_vs_historical'] = row['accuracy'] - baseline_validation['accuracy']
                row['delta_f1_macro_vs_historical'] = row['f1_macro'] - baseline_validation['f1_macro']
                row['passes_accuracy_guard'] = row['accuracy'] >= baseline_validation['accuracy'] - 0.005
                candidates.append(row)
                fitted[representation, base_mode, head_name] = (scaler, pca, model, test_x, val_x)
        del train_geom, validation_geom, test_geom
        gc.collect()
    ranking = pd.DataFrame(candidates).sort_values(['passes_accuracy_guard', 'f1_macro', 'accuracy'], ascending=[False, False, False])
    ranking.to_csv(OUTPUT / 'validation_ranking.csv', index=False)
    eligible = ranking[ranking.passes_accuracy_guard]
    selected = (eligible.iloc[0] if len(eligible) else ranking.iloc[0]).to_dict()
    scaler, pca, model, test_x, val_x = fitted[str(selected['representation']), str(selected['base_mode']), str(selected['head'])]
    test_prob = residual.normalize(model.predict_proba(test_x))
    result = {'baseline_validation': baseline_validation, 'selected_validation': {k: selected[k] for k in metric(y_validation, residual.normalize(model.predict_proba(val_x)))}, 'historical_test': baseline_test, 'selected_test': metric(y_test, test_prob), 'delta_test_accuracy': float(metric(y_test, test_prob)['accuracy'] - baseline_test['accuracy']), 'delta_test_f1_macro': float(metric(y_test, test_prob)['f1_macro'] - baseline_test['f1_macro']), 'selected_configuration': selected, 'protocol': {'selection': 'natural validation only', 'pose_frames': 200, 'pose_model': 'yolo11n-pose.pt', 'representations': list(REPRESENTATIONS), 'heads': list(heads), 'software': {'python': platform.python_version(), 'numpy': np.__version__}}}
    joblib.dump({'scaler': scaler, 'pca': pca, 'model': model, 'configuration': selected}, OUTPUT / 'selected_model.joblib', compress=3)
    write_json(OUTPUT / 'summary.json', result)
    pd.DataFrame({'video_name': test.video_name.astype(str), 'true_label': test.tag.astype(str), 'historical_prediction': [LABELS[i] for i in base_test.argmax(1)], 'selected_prediction': [LABELS[i] for i in test_prob.argmax(1)]}).to_csv(OUTPUT / 'selected_test_predictions.csv', index=False)
    write_json(OUTPUT / 'runtime.json', {'seconds': time.perf_counter() - started})
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
