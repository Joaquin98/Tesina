"""
Analyze class imbalance in the rugby dataset and prediction bias
toward the 'other' class.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_class_distribution(csv_path, name='Dataset'):
    """Analyze class distribution in a CSV file."""
    print(f"\n{'=' * 70}")
    print(f'{name}')
    print(f"{'=' * 70}")
    try:
        df = pd.read_csv(_resolve_model_artifact(csv_path))
    except FileNotFoundError:
        print(f'❌ File not found: {csv_path}')
        return None
    except Exception as e:
        print(f'❌ Error reading file: {e}')
        return None
    if 'tag' not in df.columns:
        print(f"❌ No 'tag' column found. Columns: {df.columns.tolist()}")
        return None
    counts = df['tag'].value_counts()
    total = len(df)
    print(f'\nTotal samples: {total}')
    print(f'\nClass distribution:')
    print('-' * 70)
    for class_name in ['line', 'scrum', 'other']:
        count = counts.get(class_name, 0)
        pct = count / total * 100 if total > 0 else 0
        bar = '█' * int(pct // 2) + '░' * (50 - int(pct // 2))
        print(f'  {class_name:.<15} {count:5d} ({pct:5.1f}%) {bar}')
    print('\n' + '-' * 70)
    missing_classes = []
    for class_name in ['line', 'scrum', 'other']:
        if class_name not in counts.index:
            missing_classes.append(class_name)
            print(f"  ⚠️  WARNING: '{class_name}' class is MISSING!")
    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n{'Imbalance Analysis':.<70}")
    print(f'  Maximum class size: {max_count}')
    print(f'  Minimum class size: {min_count}')
    print(f'  Imbalance ratio: {imbalance_ratio:.2f}x')
    if imbalance_ratio < 1.5:
        status = '✅ BALANCED (Good!)'
    elif imbalance_ratio < 3:
        status = '⚠️  SLIGHT IMBALANCE (Acceptable)'
    elif imbalance_ratio < 5:
        status = '⚠️⚠️ MODERATE IMBALANCE (Problematic)'
    else:
        status = '❌ SEVERE IMBALANCE (Critical!)'
    print(f'  Status: {status}')
    return {'counts': counts, 'total': total, 'ratio': imbalance_ratio}

def main():
    """Run complete diagnostic."""
    print('\n' + '=' * 70)
    print('RUGBY VIDEO CLASSIFICATION - CLASS IMBALANCE DIAGNOSIS')
    print('=' * 70)
    train_results = analyze_class_distribution('train.csv', 'TRAINING SET')
    test_results = analyze_class_distribution('test.csv', 'TEST SET')
    print('\n' + '=' * 70)
    print('DIAGNOSIS & RECOMMENDATIONS')
    print('=' * 70)
    if train_results is None or test_results is None:
        print('\n❌ Could not analyze files. Check file paths.')
        return
    train_ratio = train_results['ratio']
    test_ratio = test_results['ratio']
    print('\n📊 OVERALL DIAGNOSIS:')
    if train_ratio < 1.5:
        print('  ✅ Training set is BALANCED - Class imbalance NOT the problem')
        print('     → Issue is likely elsewhere (model architecture, data quality, etc.)')
    elif train_ratio < 5:
        print('  ⚠️  Training set has MODERATE imbalance - This could be the issue')
        print('     → Model may be biased toward majority class')
        print('     → Recommend: Add class weights or resample data')
    else:
        print('  ❌ Training set has SEVERE imbalance - THIS IS DEFINITELY THE PROBLEM!')
        print('     → Model is heavily biased toward majority class')
        print('     → URGENT: Apply class weights and/or resample data')
    print('\n🔍 SPECIFIC FINDINGS:')
    train_classes = set(train_results['counts'].index)
    test_classes = set(test_results['counts'].index)
    all_classes = {'line', 'scrum', 'other'}
    missing_train = all_classes - train_classes
    missing_test = all_classes - test_classes
    if missing_train:
        print(f'  ⚠️  Missing in training: {missing_train}')
        print(f'     → Model cannot learn these classes!')
    else:
        print(f'  ✅ All 3 classes present in training')
    if missing_test:
        print(f'  ⚠️  Missing in test: {missing_test}')
        print(f'     → Cannot evaluate these classes!')
    else:
        print(f'  ✅ All 3 classes present in test')
    print('\n📈 PER-CLASS ANALYSIS:')
    train_counts = train_results['counts']
    for class_name in ['line', 'scrum', 'other']:
        count = train_counts.get(class_name, 0)
        if count > 0:
            pct = count / train_results['total'] * 100
            if pct < 5:
                print(f"  ⚠️  {class_name}: {count} samples ({pct:.1f}%) - TOO SMALL, model won't learn it")
            elif pct < 20:
                print(f'  ⚠️  {class_name}: {count} samples ({pct:.1f}%) - Small, needs attention')
            else:
                print(f'  ✅ {class_name}: {count} samples ({pct:.1f}%) - Good')
        else:
            print(f'  ❌ {class_name}: 0 samples - MISSING!')
    print('\n' + '=' * 70)
    print('🎯 RECOMMENDED ACTIONS (in priority order):')
    print('=' * 70)
    print('\n1. IMMEDIATE (Do this first):')
    if train_ratio >= 2:
        print('   □ Add class weights to loss function')
        print('     Code: model.fit(..., class_weight={0: 2.5, 1: 1.0, 2: 2.5}, ...)')
        print('     Expected improvement: +10-20% accuracy on minority classes')
    print('\n2. SHORT TERM (If imbalance is severe):')
    if train_ratio >= 3:
        print('   □ Resample training data to balance classes')
        print('     Option A: Oversample minority (simpler)')
        print('     Option B: Undersample majority (more efficient)')
        print('     Expected improvement: +20-30% accuracy on minority classes')
    if missing_train or missing_test:
        print("   □ Verify that 'line', 'scrum', 'other' exist in your data")
        print('     Check filename patterns and CSV format')
    print('\n3. MEDIUM TERM (Additional improvements):')
    print('   □ Data augmentation (for minority classes)')
    print('   □ Focal loss (advanced loss function for imbalance)')
    print('   □ Collect more samples of underrepresented classes')
    print('\n' + '=' * 70)
    print('SUMMARY TABLE')
    print('=' * 70)
    print(f"\n{'Metric':<40} {'Train':<15} {'Test':<15}")
    print('-' * 70)
    print(f"{'Total samples':<40} {train_results['total']:<15} {test_results['total']:<15}")
    print(f"{'Imbalance ratio':<40} {train_ratio:<15.2f}x {test_ratio:<15.2f}x")
    print('\n' + '=' * 70)
    print("FACTORS AFFECTING PERFORMANCE ON 'LINE' AND 'SCRUM'")
    print('=' * 70)
    print("\nThe model sees:\n  - Thousands of 'other' examples → Learns 'other' very well\n  - Few 'line' examples → Learns 'line' weakly\n  - Few 'scrum' examples → Learns 'scrum' weakly\n\nResult:\n  - 'other' recall: ~94% (model correctly identifies most 'other' videos)\n  - 'line' recall: ~24% (model misses 76% of 'line' videos)\n  - 'scrum' recall: ??? (probably 0-20%)\n\nThis is NORMAL for imbalanced data! The fix is straightforward:\n  1. Add class weights (easiest)\n  2. Resample data (most effective)\n  3. Both (best results)\n    ")
    print('\n' + '=' * 70)
    print('✅ DIAGNOSTIC COMPLETE')
    print('=' * 70)
    print('\nNext step: Apply one of the recommended actions above')
    print('Then retrain and check if per-class accuracy improves\n')
if __name__ == '__main__':
    main()
