"""
Evaluation script for InceptionV3-GRU model on test dataset.
This script loads a trained model and evaluates it on the test set.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import os
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from models.M0.data_utils import load_video
from models.M0.config import IMG_SIZE, MAX_SEQ_LENGTH, NUM_FEATURES

def prepare_batch_for_inference(frames_list, feature_extractor_model):
    """Prepare a batch of videos for inference."""
    batch_size = len(frames_list)
    frame_masks = np.zeros(shape=(batch_size, MAX_SEQ_LENGTH), dtype='bool')
    frame_features = np.zeros(shape=(batch_size, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')
    for idx, frames in enumerate(frames_list):
        video_length = len(frames)
        length = min(MAX_SEQ_LENGTH, video_length)
        if video_length > 0:
            frame_features[idx, :length, :] = feature_extractor_model.predict(frames[:length], verbose=0)
            frame_masks[idx, :length] = 1
    return (frame_features, frame_masks)

def evaluate_model(model_dir, test_csv_path, videos_dir, batch_size=8):
    """Evaluate the trained model on test dataset."""
    print('=' * 70)
    print('InceptionV3-GRU Model Evaluation')
    print('=' * 70)
    print(f'\nLoading test data from: {test_csv_path}')
    test_df = pd.read_csv(_resolve_model_artifact(test_csv_path))
    print(f'Total test videos: {len(test_df)}')
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    label_vocab_path = os.path.join(checkpoint_dir, 'label_vocab.npy')
    if not os.path.exists(_resolve_model_artifact(label_vocab_path)):
        raise FileNotFoundError(f'Label vocabulary not found: {label_vocab_path}')
    label_vocab = _load_model_result_array(label_vocab_path, allow_pickle=True)
    print(f'Label vocabulary: {label_vocab}')
    label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}
    feature_extractors = [f for f in os.listdir(checkpoint_dir) if f.startswith('feature_extractor_')]
    if not feature_extractors:
        raise FileNotFoundError(f'No feature extractor found in {checkpoint_dir}')
    feature_extractor_path = os.path.join(checkpoint_dir, feature_extractors[0])
    print(f'\nLoading feature extractor from: {feature_extractor_path}')
    feature_extractor = keras.models.load_model(_resolve_model_artifact(feature_extractor_path))
    sequence_models = [f for f in os.listdir(checkpoint_dir) if f.startswith('sequence_model_')]
    if not sequence_models:
        raise FileNotFoundError(f'No sequence model found in {checkpoint_dir}')
    sequence_model_path = os.path.join(checkpoint_dir, sequence_models[0])
    print(f'Loading sequence model from: {sequence_model_path}')
    sequence_model = keras.models.load_model(_resolve_model_artifact(sequence_model_path))
    print(f"\n{'=' * 70}")
    print('Running predictions on test set...')
    print(f"{'=' * 70}\n")
    all_predictions = []
    all_true_labels = []
    all_video_names = []
    num_batches = (len(test_df) + batch_size - 1) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_df))
        batch_df = test_df.iloc[start_idx:end_idx]
        print(f'Processing batch {batch_idx + 1}/{num_batches} (videos {start_idx + 1}-{end_idx})...', end=' ')
        frames_list = []
        batch_true_labels = []
        batch_video_names = []
        for _, row in batch_df.iterrows():
            video_name = row['video_name']
            true_label = row['tag']
            video_path = os.path.join(videos_dir, video_name)
            if not os.path.exists(_resolve_model_artifact(video_path)):
                print(f'\nWarning: Video not found: {video_path}')
                continue
            frames = load_video(video_path)
            frames_list.append(frames)
            batch_true_labels.append(label_to_idx[true_label])
            batch_video_names.append(video_name)
        if not frames_list:
            print('No valid videos in batch, skipping.')
            continue
        frame_features, frame_masks = prepare_batch_for_inference(frames_list, feature_extractor)
        predictions = sequence_model.predict([frame_features, frame_masks], verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        all_predictions.extend(predicted_classes)
        all_true_labels.extend(batch_true_labels)
        all_video_names.extend(batch_video_names)
        print(f'Done ({len(frames_list)} videos processed)')
    y_true = np.array(all_true_labels)
    y_pred = np.array(all_predictions)
    print(f"\n{'=' * 70}")
    print('EVALUATION RESULTS')
    print(f"{'=' * 70}\n")
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Overall Accuracy: {accuracy * 100:.2f}%\n')
    print('Classification Report:')
    print('-' * 70)
    report = classification_report(y_true, y_pred, target_names=label_vocab, digits=4)
    print(report)
    print('\nConfusion Matrix:')
    print('-' * 70)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{'':12}", end='')
    for label in label_vocab:
        print(f'{label:>12}', end='')
    print()
    print('-' * (12 + 12 * len(label_vocab)))
    for i, label in enumerate(label_vocab):
        print(f'{label:12}', end='')
        for j in range(len(label_vocab)):
            print(f'{cm[i][j]:>12}', end='')
        print()
    print(f"\n{'=' * 70}")
    print('Per-Class Metrics:')
    print(f"{'=' * 70}")
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(len(label_vocab)))
    print(f"\n{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print('-' * 70)
    for i, label in enumerate(label_vocab):
        print(f'{label:<15} {precision[i]:>12.4f} {recall[i]:>12.4f} {f1[i]:>12.4f} {support[i]:>12}')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print('-' * 70)
    print(f"{'Macro Avg':<15} {precision_macro:>12.4f} {recall_macro:>12.4f} {f1_macro:>12.4f}")
    print(f"{'Weighted Avg':<15} {precision_weighted:>12.4f} {recall_weighted:>12.4f} {f1_weighted:>12.4f}")
    print(f"\n{'=' * 70}")
    print('Sample Misclassifications (first 10):')
    print(f"{'=' * 70}")
    misclassified = []
    for idx, (true_label, pred_label, video_name) in enumerate(zip(y_true, y_pred, all_video_names)):
        if true_label != pred_label:
            misclassified.append((video_name, label_vocab[true_label], label_vocab[pred_label]))
    if misclassified:
        print(f'\nTotal misclassifications: {len(misclassified)}')
        print(f"\n{'Video Name':<40} {'True Label':<15} {'Predicted':<15}")
        print('-' * 70)
        for video_name, true_label, pred_label in misclassified[:10]:
            print(f'{video_name:<40} {true_label:<15} {pred_label:<15}')
        if len(misclassified) > 10:
            print(f'\n... and {len(misclassified) - 10} more misclassifications')
    else:
        print('\nNo misclassifications! Perfect score!')
    print(f"\n{'=' * 70}")
    print('Evaluation Complete')
    print(f"{'=' * 70}\n")
    return {'accuracy': accuracy, 'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro, 'confusion_matrix': cm, 'y_true': y_true, 'y_pred': y_pred}

def main():
    """Main evaluation function."""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate InceptionV3-GRU model on test set')
    parser.add_argument('--model-dir', type=str, default='.', help='Directory containing trained model (default: current directory)')
    parser.add_argument('--test-csv', type=str, default='test.csv', help='Path to test CSV file')
    parser.add_argument('--videos-dir', type=str, default='D:/Tesina/entregas/repositorio_tesina_20261022/data/raw', help='Directory containing test videos')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing (default: 8)')
    args = parser.parse_args()
    if not os.path.exists(_resolve_model_artifact(args.test_csv)):
        print(f'Error: Test CSV not found: {args.test_csv}')
        return
    if not os.path.exists(_resolve_model_artifact(args.videos_dir)):
        print(f'Error: Videos directory not found: {args.videos_dir}')
        return
    results = evaluate_model(model_dir=args.model_dir, test_csv_path=args.test_csv, videos_dir=args.videos_dir, batch_size=args.batch_size)
    print(f"\nFinal Accuracy: {results['accuracy'] * 100:.2f}%")
if __name__ == '__main__':
    main()
