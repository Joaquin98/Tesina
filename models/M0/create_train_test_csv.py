"""Generate train.csv and test.csv from processed videos."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
random.seed(42)
VIDEO_DIR = '../../data/videos/clips_224'
LABEL_MAPPING = {'scrum': 'scrum', 'line': 'line'}

def get_label_from_filename(filename):
    """Determine label from video filename."""
    filename_lower = filename.lower()
    if 'scrum' in filename_lower:
        return 'scrum'
    elif 'line' in filename_lower:
        return 'line'
    else:
        return 'other'

def main():
    print('Scanning video directory...')
    video_dir = VIDEO_DIR
    if not os.path.exists(_resolve_model_artifact(video_dir)):
        print(f'ERROR: Video directory not found: {video_dir}')
        return
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    print(f'Found {len(video_files)} videos')
    data = []
    for video_file in video_files:
        label = get_label_from_filename(video_file)
        data.append({'video_name': video_file, 'tag': label})
    df = pd.DataFrame(data)
    print('\nLabel distribution:')
    print(df['tag'].value_counts())
    print(f'\nTotal videos: {len(df)}')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['tag'])
    print(f'\nTrain set: {len(train_df)} videos')
    print('Train label distribution:')
    print(train_df['tag'].value_counts())
    print(f'\nTest set: {len(test_df)} videos')
    print('Test label distribution:')
    print(test_df['tag'].value_counts())
    train_csv_path = 'train.csv'
    test_csv_path = 'test.csv'
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    print(f'\n✓ Created {train_csv_path}')
    print(f'✓ Created {test_csv_path}')
    print('\nSample from train.csv:')
    print(train_df.head(10))
    print('\nTraining entry point: python train.py')
if __name__ == '__main__':
    main()
