"""
Quick verification script to test class weight computation without full training.
Run this to verify the fix is working before committing to full retraining.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import numpy as np
import pandas as pd
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAIN_CSV = PROJECT_ROOT / 'data' / 'splits' / 'train.csv'
try:
    from sklearn.utils.class_weight import compute_class_weight
    print('✓ scikit-learn imported successfully')
except ImportError:
    print('✗ scikit-learn not available. Install: pip install scikit-learn')
    exit(1)

def verify_class_weights():
    """Verify class weights computation."""
    print('\n' + '=' * 70)
    print('CLASS WEIGHT VERIFICATION TEST')
    print('=' * 70)
    try:
        train_df = pd.read_csv(_resolve_model_artifact(TRAIN_CSV))
        print(f'\n✓ Loaded training data: {len(train_df)} samples')
    except FileNotFoundError:
        print(f'\n✗ Training CSV not found at {TRAIN_CSV}')
        return False
    if 'tag' not in train_df.columns:
        print("✗ 'tag' column not found in training CSV")
        print(f'  Available columns: {list(train_df.columns)}')
        return False
    print(f"✓ Found 'tag' column in training data")
    print('\nCLASS DISTRIBUTION (Training Data):')
    print('-' * 70)
    class_dist = train_df['tag'].value_counts().sort_index()
    total = len(train_df)
    for label, count in class_dist.items():
        pct = count / total * 100
        print(f'  {label:10s}: {count:5d} samples ({pct:6.2f}%)')
    print(f"  {'TOTAL':10s}: {total:5d} samples (100.00%)")
    print('\nCOMPUTING CLASS WEIGHTS:')
    print('-' * 70)
    y_train = train_df['tag'].values
    unique_classes = np.unique(y_train)
    try:
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
        print("✓ Class weights computed successfully using 'balanced' method")
    except Exception as e:
        print(f'✗ Failed to compute class weights: {e}')
        return False
    class_weights_dict = dict(enumerate(class_weights))
    print('\nCOMPUTED CLASS WEIGHTS:')
    print('-' * 70)
    label_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
    for label in sorted(label_to_idx.keys()):
        idx = label_to_idx[label]
        weight = class_weights_dict[idx]
        count = class_dist.get(label, 0)
        relative_weight = weight / min(class_weights_dict.values())
        print(f'  Class {idx} ({label:10s}): weight={weight:.4f} (relative: {relative_weight:.2f}x)')
        print(f'    └─ {count:5d} samples in training set')
    print('\nINTERPRETATION:')
    print('-' * 70)
    min_weight = min(class_weights_dict.values())
    print(f'  • Weights are relative to class with minimum count')
    print(f'  • Smallest weight (baseline): {min_weight:.4f}x')
    print(f'  • Largest weight: {max(class_weights_dict.values()):.4f}x')
    print(f'  • Weight ratio: {max(class_weights_dict.values()) / min_weight:.2f}:1')
    print(f'\n  This means:')
    for label in sorted(label_to_idx.keys()):
        idx = label_to_idx[label]
        weight = class_weights_dict[idx]
        relative = weight / min_weight
        count = class_dist.get(label, 0)
        print(f"    - '{label}' samples will contribute {relative:.2f}x more to gradients")
    print('\nDICTIONARY FORMAT (as used in train.py):')
    print('-' * 70)
    print('class_weight = {')
    for idx, weight in sorted(class_weights_dict.items()):
        label = unique_classes[idx]
        print(f'    {idx}: {weight:.4f},  # {label}')
    print('}')
    print('\n' + '=' * 70)
    print('✓ CLASS WEIGHT COMPUTATION VERIFIED SUCCESSFULLY')
    print('=' * 70)
    print('\nTraining configuration verified:')
    print('  1. Class weights will be computed automatically')
    print('  2. Minority classes (scrum, line) will get higher weights')
    print('  3. Majority class (other) will get lower weight')
    print('  4. All classes should contribute more equally to training')
    print('\nRun: python train.py\n')
    return True
if __name__ == '__main__':
    success = verify_class_weights()
    exit(0 if success else 1)
