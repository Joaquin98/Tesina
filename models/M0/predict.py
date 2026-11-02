"""Inference script for trained video classification models."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings, load_array as _load_model_result_array
import os
import sys
import numpy as np
import keras
import imageio
from models.M0.data_utils import load_video
from models.M0.config import IMG_SIZE, MAX_SEQ_LENGTH, NUM_FEATURES

def prepare_single_video_for_inference(frames, feature_extractor_model):
    """Prepare a single video for inference by extracting features."""
    frames = frames[None, ...]
    video_length = frames.shape[1]
    length = min(MAX_SEQ_LENGTH, video_length)
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype='bool')
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32')
    if video_length > 0:
        frames_to_process = frames[0, :length, :]
        if frames_to_process.shape[0] > 0:
            frame_features[0, :length, :] = feature_extractor_model.predict(frames_to_process, verbose=0)
            frame_mask[0, :length] = 1
    return (frame_features, frame_mask)

def predict_video(video_path, sequence_model, feature_extractor_model, label_vocab):
    """Predict the class of a video."""
    print(f'\nPredicting: {video_path}')
    frames = load_video(video_path)
    print(f'Loaded {len(frames)} frames')
    frame_features, frame_mask = prepare_single_video_for_inference(frames, feature_extractor_model)
    if frame_features.shape[1] > 0:
        probabilities = sequence_model.predict([frame_features, frame_mask], verbose=0)[0]
        print(f'\nPredictions:')
        top_indices = np.argsort(probabilities)[::-1]
        for i in top_indices:
            print(f'  {label_vocab[i]}: {probabilities[i] * 100:5.2f}%')
        return (probabilities, frames)
    else:
        print('No frames processed. Cannot make prediction.')
        return (None, frames)

def to_gif(images, output_path='animation.gif', duration=100):
    """Save frames as an animated GIF."""
    converted_images = images.astype(np.uint8)
    imageio.mimsave(output_path, converted_images, duration=duration)
    print(f'GIF saved to {output_path}')

def load_model_and_predict(model_dir, video_path, save_gif=False):
    """Load a trained model and predict on a video."""
    print('=' * 60)
    print('Video Classification Inference')
    print('=' * 60)
    label_vocab_path = os.path.join(model_dir, 'checkpoints', 'label_vocab.npy')
    if not os.path.exists(_resolve_model_artifact(label_vocab_path)):
        raise FileNotFoundError(f'Label vocabulary not found: {label_vocab_path}')
    label_vocab = _load_model_result_array(label_vocab_path, allow_pickle=True)
    print(f'Loaded label vocabulary: {label_vocab}')
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    feature_extractors = [f for f in os.listdir(checkpoint_dir) if f.startswith('feature_extractor_')]
    if not feature_extractors:
        raise FileNotFoundError(f'No feature extractor found in {checkpoint_dir}')
    feature_extractor_path = os.path.join(checkpoint_dir, feature_extractors[0])
    print(f'Loading feature extractor: {feature_extractor_path}')
    feature_extractor = keras.models.load_model(_resolve_model_artifact(feature_extractor_path))
    sequence_models = [f for f in os.listdir(checkpoint_dir) if f.startswith('sequence_model_')]
    if not sequence_models:
        raise FileNotFoundError(f'No sequence model found in {checkpoint_dir}')
    if len(sequence_models) > 1:
        print('\nAvailable models:')
        for i, model_name in enumerate(sequence_models):
            print(f'  {i + 1}. {model_name}')
        choice = input('Select model number (default: 1): ').strip()
        idx = int(choice) - 1 if choice.isdigit() else 0
        sequence_model_path = os.path.join(checkpoint_dir, sequence_models[idx])
    else:
        sequence_model_path = os.path.join(checkpoint_dir, sequence_models[0])
    print(f'Loading sequence model: {sequence_model_path}')
    sequence_model = keras.models.load_model(_resolve_model_artifact(sequence_model_path))
    probabilities, frames = predict_video(video_path, sequence_model, feature_extractor, label_vocab)
    if save_gif and probabilities is not None:
        gif_path = os.path.splitext(video_path)[0] + '_prediction.gif'
        to_gif(frames[:MAX_SEQ_LENGTH], gif_path)
    return (probabilities, label_vocab)

def main():
    """Main inference function."""
    import argparse
    parser = argparse.ArgumentParser(description='Predict video class using trained model')
    parser.add_argument('--model-dir', type=str, default='models/M0', help='Directory containing trained model')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--save-gif', action='store_true', help='Save prediction as animated GIF')
    args = parser.parse_args()
    if not os.path.exists(_resolve_model_artifact(args.video)):
        print(f'Error: Video file not found: {args.video}')
        sys.exit(1)
    load_model_and_predict(args.model_dir, args.video, args.save_gif)
if __name__ == '__main__':
    main()
