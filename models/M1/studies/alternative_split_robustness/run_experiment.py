"""ComparaciÃ³n fuera del split histÃ³rico entre ResNet-50+BiGRU y YOLO residual.

El experimento reutiliza Ãºnicamente las caracterÃ­sticas ya cacheadas. Cada pliegue
externo tiene entrenamiento, validaciÃ³n y evaluaciÃ³n separados por grupo de fuente
y prefijo temporal. La evaluaciÃ³n externa se consulta despuÃ©s de seleccionar Ã©poca,
alpha de fusiÃ³n y pesos de clase.
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
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.common.legacy_training import current_frames
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import keras
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
HERE = experiment_directory(__file__)
REPO = HERE.parents[1]
sys.path.insert(0, str(source_directory(REPO / 'data' / 'player_geometry_fusion')))
from config import LABELS
from models.M1.fusion_methods import fuse
from models.M1.late_fusion import geometry_statistics
from models.M1.same_rnn import MacroF1Checkpoint as ResidualMacroF1Checkpoint, branch_model as residual_model, inputs as residual_inputs, load_rgb_features, load_arrays as load_residual_arrays, pose_sequence
from train import MacroF1Checkpoint, build_model as historical_model, inputs_for_arrays, load_arrays as load_historical_arrays
POSE_MODEL = 'yolo11n-pose.pt'
FIXED_LINE_WEIGHT = 0.8
FIXED_SCRUM_WEIGHT = 0.9

def stable_seed(*items: object) -> int:
    raw = '|'.join(map(str, items)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'little')

def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def assert_disjoint(parts: dict[str, pd.DataFrame]) -> None:
    names = list(parts)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            left = parts[left_name]
            right = parts[right_name]
            clip_overlap = set(left['video_name']) & set(right['video_name'])
            group_overlap = set(left.get('source_group', pd.Series(dtype=str))) & set(right.get('source_group', pd.Series(dtype=str)))
            if clip_overlap or group_overlap:
                raise AssertionError(f'Solapamiento {left_name}/{right_name}: clips={len(clip_overlap)} grupos={len(group_overlap)}')

def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(',') if item.strip()]

def set_seed(seed: int) -> None:
    keras.utils.set_random_seed(int(seed))
    np.random.seed(int(seed) & 4294967295)

def labels_from_frame(frame: pd.DataFrame) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(LABELS)}
    return frame['tag'].map(mapping).to_numpy(dtype=np.int32)

def class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.arange(len(LABELS), dtype=np.int32)
    values = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return {int(index): float(value) for index, value in enumerate(values)}

def apply_class_calibration(probabilities: np.ndarray, line_weight: float, scrum_weight: float) -> np.ndarray:
    values = probabilities * np.asarray([line_weight, 1.0, scrum_weight], dtype=np.float64)[None, :]
    return values / np.clip(values.sum(axis=1, keepdims=True), 1e-12, None)

def metric_row(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    report = classification_report(labels, predicted, labels=np.arange(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(labels, predicted)), 'f1_macro': float(f1_score(labels, predicted, average='macro', zero_division=0)), 'f1_weighted': float(f1_score(labels, predicted, average='weighted', zero_division=0)), 'recall_line': float(report['line']['recall']), 'recall_other': float(report['other']['recall']), 'recall_scrum': float(report['scrum']['recall'])}

class ResidualSequence(keras.utils.Sequence):
    """Lee RGB+YOLO por lote para no materializar 200x2048 de todo el pliegue."""

    def __init__(self, frame: pd.DataFrame, batch_size: int, shuffle: bool, seed: int):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.frame), dtype=np.int64)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.frame) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, index: int):
        selected = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        rgb_values, rgb_masks, pose_values, labels = ([], [], [], [])
        for row_index in selected:
            row = self.frame.iloc[int(row_index)]
            name = str(row['video_name'])
            rgb, rgb_mask = load_rgb_features(name)
            pose, _ = pose_sequence(name, POSE_MODEL, 'all_residual')
            rgb_values.append(rgb)
            rgb_masks.append(rgb_mask)
            pose_values.append(pose)
            labels.append(LABELS.index(str(row['tag'])))
        arrays = {'rgb': np.asarray(rgb_values, dtype=np.float32), 'rgb_mask': np.asarray(rgb_masks, dtype=bool), 'pose': np.asarray(pose_values, dtype=np.float32)}
        return ({'rgb': arrays['rgb'], 'pose': arrays['pose'], 'sequence_mask': arrays['rgb_mask']}, np.asarray(labels, dtype=np.int32))

class HistoricalSequence(keras.utils.Sequence):
    """Secuencia RGB para evaluar sin reservar la cachÃ© completa del pliegue."""

    def __init__(self, frame: pd.DataFrame, batch_size: int=16):
        super().__init__()
        self.frame = frame.reset_index(drop=True)
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return int(np.ceil(len(self.frame) / self.batch_size))

    def __getitem__(self, index: int):
        selected = range(index * self.batch_size, min((index + 1) * self.batch_size, len(self.frame)))
        rgb_values, rgb_masks, labels = ([], [], [])
        for row_index in selected:
            row = self.frame.iloc[int(row_index)]
            rgb, rgb_mask = load_rgb_features(str(row['video_name']))
            rgb_values.append(rgb)
            rgb_masks.append(rgb_mask)
            labels.append(LABELS.index(str(row['tag'])))
        return ({'rgb': np.asarray(rgb_values, dtype=np.float32), 'rgb_mask': np.asarray(rgb_masks, dtype=bool)}, np.asarray(labels, dtype=np.int32))

def train_historical(train_df: pd.DataFrame, validation_df: pd.DataFrame, seed: int, output_dir: Path, epochs: int, patience: int) -> tuple[keras.Model, np.ndarray, np.ndarray, dict[str, object]]:
    """Entrena M0 desde cero en el entrenamiento del pliegue."""
    set_seed(seed)
    train_labels = labels_from_frame(train_df)
    validation_labels = labels_from_frame(validation_df)
    train_sequence = HistoricalSequence(train_df, batch_size=16)
    validation_sequence = HistoricalSequence(validation_df, batch_size=16)
    model, _ = historical_model('m0', 31, 30)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'best.weights.h5'
    callback = MacroF1Checkpoint(validation_sequence, validation_labels, checkpoint, patience=patience)
    started = time.perf_counter()
    history = model.fit(train_sequence, validation_data=validation_sequence, epochs=epochs, batch_size=16, shuffle=True, class_weight=class_weights(train_labels), callbacks=[callback], verbose=2)
    if _resolve_model_artifact(checkpoint).exists():
        model.load_weights(_resolve_model_artifact(checkpoint))
    probabilities = model.predict(validation_sequence, verbose=0)
    result = {'architecture': 'ResNet-50 cached features -> BiGRU(64) -> BiGRU(32) -> Dense(64)', 'seed': int(seed), 'epochs_requested': int(epochs), 'epochs_completed': int(len(history.history.get('loss', []))), 'best_validation_f1_macro': float(f1_score(validation_labels, probabilities.argmax(axis=1), average='macro', zero_division=0)), 'training_seconds': float(time.perf_counter() - started), 'checkpoint': str(checkpoint)}
    pd.DataFrame(callback.records).to_csv(output_dir / 'history.csv', index=False)
    write_json(output_dir / 'training.json', result)
    del train_sequence, validation_sequence
    return (model, probabilities, validation_labels, result)

def initialize_residual_from_historical(target: keras.Model, source: keras.Model) -> None:
    """Copia M0 en la parte RGB del residual; la proyecciÃ³n YOLO queda en cero."""
    source_bigru = [layer for layer in source.layers if isinstance(layer, keras.layers.Bidirectional)]
    target_bigru = [layer for layer in target.layers if isinstance(layer, keras.layers.Bidirectional)]
    for source_layer, target_layer in zip(source_bigru, target_bigru):
        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()
        copied = []
        for source_weight, target_weight in zip(source_weights, target_weights):
            if source_weight.shape == target_weight.shape:
                copied.append(source_weight)
            elif source_weight.ndim == 2 and source_weight.shape[0] < target_weight.shape[0]:
                expanded = np.zeros_like(target_weight)
                expanded[:source_weight.shape[0], :source_weight.shape[1]] = source_weight
                copied.append(expanded)
            else:
                copied.append(target_weight)
        target_layer.set_weights(copied)
    target.get_layer('same_fusion_dense').set_weights(source.get_layer('fusion_dense').get_weights())
    target.get_layer('same_class_output').set_weights(source.get_layer('class_output').get_weights())

def train_residual(train_df: pd.DataFrame, validation_df: pd.DataFrame, source_checkpoint: Path, seed: int, output_dir: Path, epochs: int, patience: int) -> tuple[keras.Model, np.ndarray, np.ndarray, dict[str, object]]:
    """Entrena all_residual con posiciones+posturas alineadas a 200 frames."""
    set_seed(seed)
    train_labels = labels_from_frame(train_df)
    validation_labels = labels_from_frame(validation_df)
    pose_sample, _ = pose_sequence(str(train_df.iloc[0]['video_name']), POSE_MODEL, 'all_residual')
    train_sequence = ResidualSequence(train_df, batch_size=64, shuffle=True, seed=seed)
    validation_sequence = ResidualSequence(validation_df, batch_size=64, shuffle=False, seed=seed)
    model = residual_model('all_residual', int(pose_sample.shape[-1]))
    source = historical_model('m0', 31, 30)[0]
    source.load_weights(_resolve_model_artifact(source_checkpoint))
    initialize_residual_from_historical(model, source)
    del source
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / 'best.weights.h5'
    callback = ResidualMacroF1Checkpoint(validation_sequence, validation_labels, checkpoint, patience=patience)
    started = time.perf_counter()
    history = model.fit(train_sequence, validation_data=validation_sequence, epochs=epochs, batch_size=64, shuffle=True, class_weight=class_weights(train_labels), callbacks=[callback], verbose=2)
    if _resolve_model_artifact(checkpoint).exists():
        model.load_weights(_resolve_model_artifact(checkpoint))
    probabilities = model.predict(validation_sequence, verbose=0)
    result = {'architecture': 'M0 warm-start + YOLO positions(31)+poses(30) residual projection', 'seed': int(seed), 'pose_dim': int(pose_sample.shape[-1]), 'epochs_requested': int(epochs), 'epochs_completed': int(len(history.history.get('loss', []))), 'best_validation_f1_macro': float(f1_score(validation_labels, probabilities.argmax(axis=1), average='macro', zero_division=0)), 'training_seconds': float(time.perf_counter() - started), 'checkpoint': str(checkpoint)}
    pd.DataFrame(callback.records).to_csv(output_dir / 'history.csv', index=False)
    write_json(output_dir / 'training.json', result)
    del train_sequence, validation_sequence, pose_sample
    gc.collect()
    return (model, probabilities, validation_labels, result)

def prediction_frame(names: list[str], labels: np.ndarray, probabilities: np.ndarray, prefix: str) -> pd.DataFrame:
    predicted = probabilities.argmax(axis=1)
    frame = pd.DataFrame({'video_name': names, 'true_label': [LABELS[i] for i in labels], f'{prefix}_label': [LABELS[i] for i in predicted]})
    for index, label in enumerate(LABELS):
        frame[f'{prefix}_prob_{label}'] = probabilities[:, index]
    return frame

def select_alpha(validation_labels: np.ndarray, rgb: np.ndarray, geometry: np.ndarray, step: float=0.05) -> tuple[float, float]:
    best = (-np.inf, 0.0)
    for alpha in np.arange(0.0, 1.0 + step * 0.5, step):
        candidate = fuse(rgb, geometry, 'geometric_mean', float(alpha))
        score = float(f1_score(validation_labels, candidate.argmax(axis=1), average='macro', zero_division=0))
        if score > best[0] + 1e-12:
            best = (score, float(alpha))
    return (best[1], best[0])

def select_class_calibration(validation_labels: np.ndarray, probabilities: np.ndarray, step: float=0.05) -> tuple[float, float, float]:
    """Selecciona pesos de line/scrum solo sobre la validaciÃ³n interna."""
    best = (-np.inf, 1.0, 1.0)
    values = np.arange(0.7, 1.51 + step * 0.5, step)
    for line_weight in values:
        for scrum_weight in values:
            candidate = apply_class_calibration(probabilities, float(line_weight), float(scrum_weight))
            score = float(f1_score(validation_labels, candidate.argmax(axis=1), average='macro', zero_division=0))
            if score > best[0] + 1e-12:
                best = (score, float(line_weight), float(scrum_weight))
    return (best[1], best[2], best[0])

def load_split(fold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = current_frames()
    # These repetitions use fixed clip partitions, without claiming match-group folds.
    return frames

def run_fold(fold: int, args: argparse.Namespace, output_root: Path) -> dict[str, object]:
    train_df, validation_df, evaluation_df = load_split(fold)
    fold_dir = output_root / f'outer_{fold:02d}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / 'partition_policy.json').write_text(json.dumps({'policy': 'current_fixed_csv', 'repetition': fold, 'match_group_disjointness': 'not_established'}), encoding='utf8')
    m0_seed = stable_seed(20260812, fold, 'm0')
    residual_seed = stable_seed(20260812, fold, 'all_residual')
    print(f'\n=== fixed_partition_repeat_{fold:02d}: train={len(train_df)} val={len(validation_df)} eval={len(evaluation_df)} ===', flush=True)
    if args.reuse_checkpoints and _resolve_model_artifact(fold_dir / 'historical' / 'best.weights.h5').exists():
        historical_checkpoint = fold_dir / 'historical' / 'best.weights.h5'
        m0_info = json.loads(_resolve_model_artifact(fold_dir / 'historical' / 'training.json').read_text(encoding='utf-8'))
        validation_labels = labels_from_frame(validation_df)
        historical = historical_model('m0', 31, 30)[0]
        historical.load_weights(_resolve_model_artifact(historical_checkpoint))
        historical_validation = historical.predict(HistoricalSequence(validation_df), verbose=0)
    else:
        historical, historical_validation, validation_labels, m0_info = train_historical(train_df, validation_df, m0_seed, fold_dir / 'historical', args.epochs, args.patience)
        historical_checkpoint = Path(str(m0_info['checkpoint']))
    print('  M0 validaciÃ³n lista', flush=True)
    del historical
    keras.backend.clear_session()
    gc.collect()
    if args.reuse_checkpoints and _resolve_model_artifact(fold_dir / 'all_residual' / 'best.weights.h5').exists():
        info_path = fold_dir / 'all_residual' / 'training.json'
        residual_info = json.loads(_resolve_model_artifact(info_path).read_text(encoding='utf-8')) if _resolve_model_artifact(info_path).exists() else {'checkpoint': str(fold_dir / 'all_residual' / 'best.weights.h5'), 'best_validation_f1_macro': None}
        residual = residual_model('all_residual', 62)
        residual.load_weights(_resolve_model_artifact(fold_dir / 'all_residual' / 'best.weights.h5'))
        residual_validation = residual.predict(ResidualSequence(validation_df, batch_size=64, shuffle=False, seed=residual_seed), verbose=0)
        residual_validation_labels = labels_from_frame(validation_df)
    else:
        residual, residual_validation, residual_validation_labels, residual_info = train_residual(train_df, validation_df, historical_checkpoint, residual_seed, fold_dir / 'all_residual', args.epochs, args.patience)
    print('  YOLO validaciÃ³n lista', flush=True)
    if not np.array_equal(validation_labels, residual_validation_labels):
        raise AssertionError('El orden de validaciÃ³n M0/Yolo no coincide')
    geometry_train, geometry_train_y, _ = geometry_statistics(train_df, POSE_MODEL, 'both')
    geometry_validation, geometry_validation_y, validation_names = geometry_statistics(validation_df, POSE_MODEL, 'both')
    if not np.array_equal(validation_labels, geometry_validation_y):
        raise AssertionError('El orden de validaciÃ³n M0/geometrÃ­a no coincide')
    forest = RandomForestClassifier(n_estimators=args.trees, max_features='sqrt', class_weight='balanced', random_state=stable_seed(20260812, fold, 'rf'), n_jobs=-1)
    forest.fit(geometry_train, geometry_train_y)
    print('  RF geomÃ©trico entrenado', flush=True)
    geometry_validation_prob = forest.predict_proba(geometry_validation)
    alpha_geometry, geometry_validation_f1 = select_alpha(validation_labels, historical_validation, geometry_validation_prob)
    geometry_validation_fused = fuse(historical_validation, geometry_validation_prob, 'geometric_mean', alpha_geometry)
    alpha_residual, ensemble_validation_f1 = select_alpha(validation_labels, residual_validation, geometry_validation_fused)
    ensemble_validation = (1.0 - alpha_residual) * residual_validation + alpha_residual * geometry_validation_fused
    calibrated_validation = apply_class_calibration(ensemble_validation, FIXED_LINE_WEIGHT, FIXED_SCRUM_WEIGHT)
    selected_line_weight, selected_scrum_weight, selected_calibration_f1 = select_class_calibration(validation_labels, ensemble_validation)
    eval_m0_labels = labels_from_frame(evaluation_df)
    eval_names = evaluation_df['video_name'].astype(str).tolist()
    eval_m0_rows = evaluation_df
    eval_residual_labels = labels_from_frame(evaluation_df)
    residual_checkpoint = Path(str(residual_info['checkpoint']))
    del residual
    keras.backend.clear_session()
    gc.collect()
    residual_eval_model = residual_model('all_residual', 62)
    residual_eval_model.load_weights(_resolve_model_artifact(residual_checkpoint))
    eval_residual_sequence = ResidualSequence(evaluation_df, batch_size=64, shuffle=False, seed=residual_seed)
    residual_eval = residual_eval_model.predict(eval_residual_sequence, verbose=0)
    print('  YOLO evaluaciÃ³n lista', flush=True)
    if eval_names != evaluation_df['video_name'].astype(str).tolist() or not np.array_equal(eval_m0_labels, eval_residual_labels):
        raise AssertionError('El orden de evaluaciÃ³n M0/Yolo no coincide')
    del eval_residual_sequence
    del residual_eval_model
    keras.backend.clear_session()
    gc.collect()
    historical_eval_model = historical_model('m0', 31, 30)[0]
    historical_eval_model.load_weights(_resolve_model_artifact(historical_checkpoint))
    historical_eval = historical_eval_model.predict(HistoricalSequence(evaluation_df), verbose=0)
    print('  M0 evaluaciÃ³n lista', flush=True)
    del historical_eval_model
    keras.backend.clear_session()
    gc.collect()
    geometry_eval, geometry_eval_y, eval_geometry_names = geometry_statistics(evaluation_df, POSE_MODEL, 'both')
    print('  GeometrÃ­a evaluaciÃ³n lista', flush=True)
    if eval_names != eval_geometry_names or not np.array_equal(eval_m0_labels, geometry_eval_y):
        raise AssertionError('El orden de evaluaciÃ³n M0/geometrÃ­a no coincide')
    geometry_eval_prob = forest.predict_proba(geometry_eval)
    geometry_eval_fused = fuse(historical_eval, geometry_eval_prob, 'geometric_mean', alpha_geometry)
    ensemble_eval = (1.0 - alpha_residual) * residual_eval + alpha_residual * geometry_eval_fused
    calibrated_eval = apply_class_calibration(ensemble_eval, FIXED_LINE_WEIGHT, FIXED_SCRUM_WEIGHT)
    selected_calibrated_eval = apply_class_calibration(ensemble_eval, selected_line_weight, selected_scrum_weight)
    table = prediction_frame(eval_names, eval_m0_labels, historical_eval, 'historical')
    table = table.join(prediction_frame(eval_names, eval_m0_labels, calibrated_eval, 'yolo_ensemble').drop(columns=['video_name', 'true_label']))
    table = table.join(prediction_frame(eval_names, eval_m0_labels, selected_calibrated_eval, 'yolo_ensemble_selected').drop(columns=['video_name', 'true_label']))
    table.to_csv(fold_dir / 'evaluation_predictions.csv', index=False)
    metrics = {'outer_fold': fold, 'train_rows': int(len(train_df)), 'validation_rows': int(len(validation_df)), 'evaluation_rows': int(len(evaluation_df)), 'historical': metric_row(eval_m0_labels, historical_eval), 'all_residual_uncalibrated': metric_row(eval_m0_labels, ensemble_eval), 'all_residual_calibrated': metric_row(eval_m0_labels, calibrated_eval), 'all_residual_recalibrated_internal': metric_row(eval_m0_labels, selected_calibrated_eval), 'alpha_geometry': float(alpha_geometry), 'alpha_residual': float(alpha_residual), 'validation_historical_f1_macro': float(m0_info['best_validation_f1_macro']), 'validation_all_residual_f1_macro': None if residual_info.get('best_validation_f1_macro') is None else float(residual_info['best_validation_f1_macro']), 'validation_geometry_fused_f1_macro': float(geometry_validation_f1), 'validation_ensemble_f1_macro': float(ensemble_validation_f1), 'validation_selected_calibration_f1_macro': float(selected_calibration_f1), 'selected_calibration': {'line': selected_line_weight, 'scrum': selected_scrum_weight}, 'fixed_calibration': {'line': FIXED_LINE_WEIGHT, 'scrum': FIXED_SCRUM_WEIGHT}, 'm0_seed': int(m0_seed), 'residual_seed': int(residual_seed)}
    write_json(fold_dir / 'metrics.json', metrics)
    del forest, geometry_train, geometry_validation, geometry_eval
    gc.collect()
    print(json.dumps({'fold': fold, 'historical': metrics['historical'], 'yolo': metrics['all_residual_calibrated'], 'alpha': [alpha_geometry, alpha_residual]}, ensure_ascii=False), flush=True)
    return metrics

def bootstrap_paired(predictions: pd.DataFrame, draws: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    indexes = np.arange(len(predictions))
    values = {'accuracy_delta': [], 'f1_macro_delta': [], 'f1_weighted_delta': []}
    historical_labels = predictions['historical_label']
    yolo_labels = predictions['yolo_ensemble_label']
    true_labels = predictions['true_label']
    for _ in range(draws):
        sample = rng.choice(indexes, size=len(indexes), replace=True)
        truth = true_labels.iloc[sample]
        hist = historical_labels.iloc[sample]
        yolo = yolo_labels.iloc[sample]
        values['accuracy_delta'].append(float(np.mean(yolo.to_numpy() == truth.to_numpy()) - np.mean(hist.to_numpy() == truth.to_numpy())))
        values['f1_macro_delta'].append(float(f1_score(truth, yolo, average='macro', zero_division=0) - f1_score(truth, hist, average='macro', zero_division=0)))
        values['f1_weighted_delta'].append(float(f1_score(truth, yolo, average='weighted', zero_division=0) - f1_score(truth, hist, average='weighted', zero_division=0)))
    return {key: {'mean': float(np.mean(series)), 'low': float(np.quantile(series, 0.025)), 'high': float(np.quantile(series, 0.975))} for key, series in values.items()}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', default='1,2,3,4,5')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--trees', type=int, default=500)
    parser.add_argument('--bootstrap-draws', type=int, default=2000)
    parser.add_argument('--output-root', default=str(HERE / 'results'))
    parser.add_argument('--reuse-checkpoints', action='store_true')
    args = parser.parse_args()
    folds = parse_int_list(args.folds)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = []
    all_predictions = []
    for fold in folds:
        metrics = run_fold(fold, args, output_root)
        all_results.append(metrics)
        all_predictions.append(pd.read_csv(_resolve_model_artifact(output_root / f'outer_{fold:02d}' / 'evaluation_predictions.csv')))
    result_frame = pd.DataFrame([{'outer_fold': row['outer_fold'], 'model': model, **values} for row in all_results for model, values in (('historical', row['historical']), ('all_residual_uncalibrated', row['all_residual_uncalibrated']), ('all_residual_calibrated', row['all_residual_calibrated']), ('all_residual_recalibrated_internal', row['all_residual_recalibrated_internal']))])
    result_frame.to_csv(output_root / 'fold_metrics.csv', index=False)
    aggregate_rows = []
    for model, group in result_frame.groupby('model'):
        aggregate_rows.append({'model': model, 'folds': int(len(group)), **{f'{metric}_mean': float(group[metric].mean()) for metric in ('accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum')}, **{f'{metric}_std': float(group[metric].std(ddof=0)) for metric in ('accuracy', 'f1_macro', 'f1_weighted', 'recall_line', 'recall_other', 'recall_scrum')}})
    pd.DataFrame(aggregate_rows).sort_values('model').to_csv(output_root / 'aggregate_metrics.csv', index=False)
    concatenated = pd.concat(all_predictions, ignore_index=True)
    concatenated.to_csv(output_root / 'all_evaluation_predictions.csv', index=False)
    write_json(output_root / 'paired_bootstrap.json', bootstrap_paired(concatenated, args.bootstrap_draws, 20260812))
    write_json(output_root / 'run_config.json', {'folds': folds, 'epochs': args.epochs, 'patience': args.patience, 'trees': args.trees, 'pose_model': POSE_MODEL, 'fixed_calibration': {'line': FIXED_LINE_WEIGHT, 'scrum': FIXED_SCRUM_WEIGHT}, 'evaluation_policy': 'outer grouped evaluation read only after model selection', 'calibration_policy': 'fixed 0.80/0.90 is the locked deployed configuration; selected weights use only each fold validation'})
    print(f'Resultados guardados en {output_root}')
if __name__ == '__main__':
    main()
