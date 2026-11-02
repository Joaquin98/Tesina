"""Compare CNN-RNN baseline against detector-assisted fusion variants."""
from __future__ import annotations
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import csv
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
REPO_ROOT = experiment_directory(__file__).parents[1]
MODEL_DIR = REPO_ROOT / 'models' / 'M0'
sys.path.insert(0, str(MODEL_DIR))
from config import FEATURE_CACHE_DIR, MAX_SEQ_LENGTH, NUM_FEATURES, TEST_CSV, TRAIN_CSV
BASE_DIR = experiment_directory(__file__)
RESULTS_DIR = BASE_DIR / 'results'
CNN_RNN_CHECKPOINT = MODEL_DIR / 'checkpoints' / 'best_model' / 'best_model.keras'
LINE_DETECTOR_CHECKPOINT = REPO_ROOT / 'data' / 'resnet_frame_play_detectors' / 'checkpoints' / 'line' / 'model.keras'
SCRUM_DETECTOR_CHECKPOINT = REPO_ROOT / 'data' / 'resnet_frame_play_detectors' / 'checkpoints' / 'scrum' / 'model.keras'
EXTRACTOR = 'resnet50'
LABELS = ['line', 'other', 'scrum']
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}

def cache_path(video_name: str) -> Path:
    return Path(FEATURE_CACHE_DIR) / EXTRACTOR / f'{Path(video_name).stem}.npz'

def load_cached_features(video_name: str) -> tuple[np.ndarray, np.ndarray]:
    path = cache_path(video_name)
    if not _resolve_model_artifact(path).exists():
        raise FileNotFoundError(f'Missing feature cache for {video_name}: {path}')
    data = _load_model_result_array(path)
    frame_features = data['frame_features'].astype(np.float32, copy=False)
    frame_masks = data['frame_masks'].astype(bool, copy=False)
    if frame_features.shape != (MAX_SEQ_LENGTH, NUM_FEATURES):
        raise ValueError(f'Unexpected feature shape for {video_name}: {frame_features.shape}')
    if frame_masks.shape != (MAX_SEQ_LENGTH,):
        raise ValueError(f'Unexpected mask shape for {video_name}: {frame_masks.shape}')
    return (frame_features, frame_masks)

def clean_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(_resolve_model_artifact(path)).copy()
    df['tag'] = df['tag'].astype(str).str.strip().str.lower()
    df = df[df['tag'].isin(LABELS)].reset_index(drop=True)
    return df

def predict_cnn_rnn(model: keras.Model, df: pd.DataFrame, batch_size: int=32) -> pd.DataFrame:
    rows = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        features = np.zeros((len(batch), MAX_SEQ_LENGTH, NUM_FEATURES), dtype=np.float32)
        masks = np.zeros((len(batch), MAX_SEQ_LENGTH), dtype=bool)
        for idx, (_, row) in enumerate(batch.iterrows()):
            frame_features, frame_masks = load_cached_features(row['video_name'])
            features[idx] = frame_features
            masks[idx] = frame_masks
        probs = model.predict([features, masks], verbose=0)
        for local_idx, (_, row) in enumerate(batch.iterrows()):
            prob = probs[local_idx]
            pred_idx = int(np.argmax(prob))
            rows.append({'video_name': row['video_name'], 'true_label': row['tag'], 'cnn_p_line': float(prob[LABEL_TO_INDEX['line']]), 'cnn_p_other': float(prob[LABEL_TO_INDEX['other']]), 'cnn_p_scrum': float(prob[LABEL_TO_INDEX['scrum']]), 'cnn_pred': LABELS[pred_idx], 'cnn_confidence': float(prob[pred_idx])})
        print(f'CNN-RNN probabilities: {min(start + batch_size, len(df))}/{len(df)}')
    return pd.DataFrame(rows)

def predict_detector(model: keras.Model, df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    prob_col = f'detector_p_{target}'
    for idx, row in df.iterrows():
        frame_features, frame_masks = load_cached_features(row['video_name'])
        valid_frames = frame_features[frame_masks]
        if len(valid_frames) == 0:
            probability = 0.0
        else:
            probability = float(np.mean(model.predict(valid_frames, batch_size=1024, verbose=0).reshape(-1)))
        rows.append({'video_name': row['video_name'], prob_col: probability})
        if (idx + 1) % 100 == 0 or idx + 1 == len(df):
            print(f'{target} detector probabilities: {idx + 1}/{len(df)}')
    return pd.DataFrame(rows)

def build_score_table(df: pd.DataFrame, cnn_model: keras.Model, line_model: keras.Model, scrum_model: keras.Model, split: str) -> pd.DataFrame:
    cached_path = RESULTS_DIR / f'{split}_predictions_with_scores.csv'
    if _resolve_model_artifact(cached_path).exists():
        return pd.read_csv(_resolve_model_artifact(cached_path))
    cnn = predict_cnn_rnn(cnn_model, df)
    line = predict_detector(line_model, df, 'line')
    scrum = predict_detector(scrum_model, df, 'scrum')
    merged = cnn.merge(line, on='video_name').merge(scrum, on='video_name')
    merged.to_csv(cached_path, index=False)
    return merged

def y_true_from_scores(scores: pd.DataFrame) -> np.ndarray:
    return scores['true_label'].map(LABEL_TO_INDEX).to_numpy(dtype=np.int32)

def labels_from_indices(indices: np.ndarray) -> list[str]:
    return [LABELS[int(index)] for index in indices]

def baseline_predictions(scores: pd.DataFrame) -> np.ndarray:
    return scores['cnn_pred'].map(LABEL_TO_INDEX).to_numpy(dtype=np.int32)

def probability_matrix(scores: pd.DataFrame) -> np.ndarray:
    return scores[['cnn_p_line', 'cnn_p_other', 'cnn_p_scrum']].to_numpy(dtype=np.float32)

def late_fusion_predictions(scores: pd.DataFrame, alpha: float, beta: float) -> np.ndarray:
    probs = probability_matrix(scores).copy()
    probs[:, LABEL_TO_INDEX['line']] += alpha * scores['detector_p_line'].to_numpy(dtype=np.float32)
    probs[:, LABEL_TO_INDEX['scrum']] += beta * scores['detector_p_scrum'].to_numpy(dtype=np.float32)
    return np.argmax(probs, axis=1)

def rerank_predictions(scores: pd.DataFrame, line_threshold: float, scrum_threshold: float, uncertainty_threshold: float) -> np.ndarray:
    base = baseline_predictions(scores)
    cnn_conf = scores['cnn_confidence'].to_numpy(dtype=np.float32)
    line_prob = scores['detector_p_line'].to_numpy(dtype=np.float32)
    scrum_prob = scores['detector_p_scrum'].to_numpy(dtype=np.float32)
    other_idx = LABEL_TO_INDEX['other']
    line_idx = LABEL_TO_INDEX['line']
    scrum_idx = LABEL_TO_INDEX['scrum']
    pred = base.copy()
    for i in range(len(pred)):
        can_change = pred[i] == other_idx or cnn_conf[i] < uncertainty_threshold
        if not can_change:
            continue
        candidates = []
        if line_prob[i] >= line_threshold:
            candidates.append((line_prob[i], line_idx))
        if scrum_prob[i] >= scrum_threshold:
            candidates.append((scrum_prob[i], scrum_idx))
        if candidates:
            pred[i] = max(candidates, key=lambda item: item[0])[1]
    return pred

def metric_dict(name: str, y_true: np.ndarray, y_pred: np.ndarray, params: dict | None=None) -> dict:
    row = {'model': name, 'accuracy': round(float(accuracy_score(y_true, y_pred)), 6), 'f1_macro': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 6), 'f1_weighted': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 6), 'precision_line': round(float(precision_score(y_true, y_pred, labels=[LABEL_TO_INDEX['line']], average='macro', zero_division=0)), 6), 'recall_line': round(float(recall_score(y_true, y_pred, labels=[LABEL_TO_INDEX['line']], average='macro', zero_division=0)), 6), 'precision_other': round(float(precision_score(y_true, y_pred, labels=[LABEL_TO_INDEX['other']], average='macro', zero_division=0)), 6), 'recall_other': round(float(recall_score(y_true, y_pred, labels=[LABEL_TO_INDEX['other']], average='macro', zero_division=0)), 6), 'precision_scrum': round(float(precision_score(y_true, y_pred, labels=[LABEL_TO_INDEX['scrum']], average='macro', zero_division=0)), 6), 'recall_scrum': round(float(recall_score(y_true, y_pred, labels=[LABEL_TO_INDEX['scrum']], average='macro', zero_division=0)), 6)}
    row['params'] = json.dumps(params or {}, sort_keys=True)
    return row

def tune_late_fusion(train_scores: pd.DataFrame) -> dict:
    y_true = y_true_from_scores(train_scores)
    best = None
    for alpha in np.linspace(0.0, 1.0, 21):
        for beta in np.linspace(0.0, 1.0, 21):
            pred = late_fusion_predictions(train_scores, float(alpha), float(beta))
            score = f1_score(y_true, pred, average='macro', zero_division=0)
            if best is None or score > best['score']:
                best = {'alpha': float(alpha), 'beta': float(beta), 'score': float(score)}
    return best or {'alpha': 0.0, 'beta': 0.0, 'score': 0.0}

def tune_rerank(train_scores: pd.DataFrame) -> dict:
    y_true = y_true_from_scores(train_scores)
    best = None
    for line_threshold in np.linspace(0.5, 0.95, 10):
        for scrum_threshold in np.linspace(0.5, 0.95, 10):
            for uncertainty_threshold in np.linspace(0.45, 0.8, 8):
                pred = rerank_predictions(train_scores, float(line_threshold), float(scrum_threshold), float(uncertainty_threshold))
                score = f1_score(y_true, pred, average='macro', zero_division=0)
                if best is None or score > best['score']:
                    best = {'line_threshold': float(line_threshold), 'scrum_threshold': float(scrum_threshold), 'uncertainty_threshold': float(uncertainty_threshold), 'score': float(score)}
    return best or {'line_threshold': 0.5, 'scrum_threshold': 0.5, 'uncertainty_threshold': 0.5, 'score': 0.0}

def train_meta_classifier(train_scores: pd.DataFrame) -> LogisticRegression:
    x_train = train_scores[['cnn_p_line', 'cnn_p_other', 'cnn_p_scrum', 'detector_p_line', 'detector_p_scrum']].to_numpy(dtype=np.float32)
    y_train = y_true_from_scores(train_scores)
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(x_train, y_train)
    return model

def save_reports(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    report = classification_report(y_true, y_pred, target_names=LABELS, zero_division=0)
    (RESULTS_DIR / f'{name}_classification_report.txt').write_text(report, encoding='utf-8')
    for normalize, suffix in [(None, 'raw'), ('true', 'normalized')]:
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))), normalize=normalize)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{name} ({suffix})')
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f'{name}_confusion_matrix_{suffix}.png', dpi=100)
        plt.close(fig)

def write_markdown_report(rows: list[dict], late_params: dict, rerank_params: dict) -> None:
    report_path = RESULTS_DIR / 'REPORT.md'
    with _resolve_model_artifact(report_path).open('w', encoding='utf-8') as handle:
        handle.write('# CNN-RNN + detector fusion comparison\n\n')
        handle.write('The baseline is the unmodified CNN-RNN best model. Detector-assisted variants use the frame-level ResNet binary detector probabilities as auxiliary signals.\n\n')
        handle.write('## Tuned parameters\n\n')
        handle.write(f'- Late fusion: `{json.dumps(late_params, sort_keys=True)}`\n')
        handle.write(f'- Candidate re-ranking: `{json.dumps(rerank_params, sort_keys=True)}`\n\n')
        handle.write('## Comparative table\n\n')
        columns = ['model', 'accuracy', 'f1_macro', 'f1_weighted', 'precision_line', 'recall_line', 'precision_other', 'recall_other', 'precision_scrum', 'recall_scrum']
        handle.write('| ' + ' | '.join(columns) + ' |\n')
        handle.write('|' + '|'.join(['---'] + ['---:'] * (len(columns) - 1)) + '|\n')
        for row in rows:
            handle.write('| ' + ' | '.join((str(row[column]) for column in columns)) + ' |\n')
        handle.write('\n## Notes\n\n')
        handle.write('- Late fusion and re-ranking are tuned on the training split and evaluated on the held-out test split.\n')
        handle.write('- The meta-classifier is trained on prediction-level features from the training split and evaluated on the same held-out test split.\n')
        handle.write('- A useful fusion should improve target-play recall without sharply degrading macro-F1 or `other` recall.\n')

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading models...')
    cnn_model = keras.models.load_model(_resolve_model_artifact(CNN_RNN_CHECKPOINT))
    line_model = keras.models.load_model(_resolve_model_artifact(LINE_DETECTOR_CHECKPOINT))
    scrum_model = keras.models.load_model(_resolve_model_artifact(SCRUM_DETECTOR_CHECKPOINT))
    train_df = clean_df(TRAIN_CSV)
    test_df = clean_df(TEST_CSV)
    print(f'Train videos: {len(train_df)} | Test videos: {len(test_df)}')
    train_scores = build_score_table(train_df, cnn_model, line_model, scrum_model, 'train')
    test_scores = build_score_table(test_df, cnn_model, line_model, scrum_model, 'test')
    y_test = y_true_from_scores(test_scores)
    late_params = tune_late_fusion(train_scores)
    rerank_params = tune_rerank(train_scores)
    meta_model = train_meta_classifier(train_scores)
    predictions = {'cnn_rnn_baseline': baseline_predictions(test_scores), 'late_fusion_tuned': late_fusion_predictions(test_scores, late_params['alpha'], late_params['beta']), 'candidate_rerank_tuned': rerank_predictions(test_scores, rerank_params['line_threshold'], rerank_params['scrum_threshold'], rerank_params['uncertainty_threshold'])}
    x_test = test_scores[['cnn_p_line', 'cnn_p_other', 'cnn_p_scrum', 'detector_p_line', 'detector_p_scrum']].to_numpy(dtype=np.float32)
    predictions['meta_logistic'] = meta_model.predict(x_test)
    rows = []
    prediction_output = test_scores.copy()
    for name, pred in predictions.items():
        params = {}
        if name == 'late_fusion_tuned':
            params = late_params
        elif name == 'candidate_rerank_tuned':
            params = rerank_params
        rows.append(metric_dict(name, y_test, pred, params=params))
        prediction_output[f'{name}_pred'] = labels_from_indices(pred)
        save_reports(name, y_test, pred)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / 'comparison_table.csv', index=False)
    prediction_output.to_csv(RESULTS_DIR / 'test_predictions_with_fusion_outputs.csv', index=False)
    write_markdown_report(rows, late_params, rerank_params)
    print('\nComparison')
    print(comparison.to_string(index=False))
    print(f"\nReport: {RESULTS_DIR / 'REPORT.md'}")
if __name__ == '__main__':
    main()
