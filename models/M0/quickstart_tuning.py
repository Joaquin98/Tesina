"""
Quick start hyperparameter tuning for rugby video classification.

This script provides an easy way to get started with tuning.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os
import sys
import json
from datetime import datetime

def print_banner(text):
    """Print a formatted banner."""
    print('\n' + '=' * 70)
    print(f'  {text}')
    print('=' * 70 + '\n')

def main():
    """Interactive tuning setup."""
    print_banner('HYPERPARAMETER TUNING - QUICK START')
    print('Hyperparameter search for the video classification model.\n')
    print('📚 BACKGROUND')
    print('-' * 70)
    print('\nHyperparameter tuning tests different combinations of settings:\n  • batch_size: How many videos to process at once\n  • learning_rate: How fast the model learns\n  • epochs: How many times to train through the data\n  • optimizer: The algorithm to update weights\n  • And more...\n\nThe script trains multiple models and finds which configuration works best.\n    ')
    print('\n🎯 CHOOSE YOUR APPROACH:\n')
    print('1️⃣  QUICK TEST (30 min)')
    print('   ✓ Fast: ~8 configurations, 5 epochs each')
    print('   ✓ Good for: Testing setup, quick feedback')
    print("   ✗ Limited: Won't find optimal parameters")
    print('\n2️⃣  FOCUSED TUNING (2-3 hours)')
    print('   ✓ Balanced: ~20 random configurations, 20 epochs each')
    print('   ✓ Good for: Finding good parameters quickly')
    print('   ✓ Practical for: Most situations')
    print('\n3️⃣  COMPREHENSIVE TUNING (8-16 hours)')
    print('   ✓ Thorough: ~144 configurations, 50 epochs each')
    print('   ✓ Good for: Production models, thesis results')
    print('   ✗ Slow: Requires significant time/resources')
    print('\n4️⃣  CUSTOM TUNING')
    print('   ✓ Flexible: Define your own parameter grid')
    print('   ✓ Good for: Targeted optimization')
    print('\n' + '-' * 70)
    choice = input('\nSelect approach (1/2/3/4): ').strip()
    if choice == '1':
        run_quick_test()
    elif choice == '2':
        run_focused_tuning()
    elif choice == '3':
        run_comprehensive_tuning()
    elif choice == '4':
        run_custom_tuning()
    else:
        print('❌ Invalid choice. Using quick test.')
        run_quick_test()

def run_quick_test():
    """Quick test setup."""
    print_banner('QUICK TEST SETUP')
    param_grid = {'batch_size': [8, 16], 'learning_rate': [0.0001, 0.0005], 'weight_decay': [1e-05], 'epochs': [5, 10], 'optimizer': ['adam'], 'sequence_model': ['lstm', 'gru_small'], 'feature_extractor': ['resnet50'], 'dropout_rate': [0.5]}
    print(f'Configuration: 2 × 2 × 2 = 8 experiments')
    print(f'Estimated time: 30 minutes\n')
    print('Parameters to test:')
    for param, values in param_grid.items():
        print(f'  • {param}: {values}')
    if confirm_and_run(param_grid):
        save_config('quick_test', param_grid)

def run_focused_tuning():
    """Focused tuning with random search."""
    print_banner('FOCUSED TUNING SETUP')
    param_grid = {'batch_size': [8, 16, 32], 'learning_rate': [0.0001, 0.0005, 0.001], 'weight_decay': [1e-05], 'epochs': [20], 'optimizer': ['adam'], 'sequence_model': ['lstm', 'gru_small'], 'feature_extractor': ['resnet50'], 'dropout_rate': [0.3, 0.5]}
    print(f'Configuration: Random search (20 random combinations)')
    print(f'Estimated time: 2-3 hours\n')
    print('Parameters to test:')
    for param, values in param_grid.items():
        print(f'  • {param}: {values}')
    if confirm_and_run(param_grid, search_type='random'):
        save_config('focused_tuning', param_grid)

def run_comprehensive_tuning():
    """Comprehensive grid search."""
    print_banner('COMPREHENSIVE TUNING SETUP')
    param_grid = {'batch_size': [8, 16, 32], 'learning_rate': [0.0001, 0.0005, 0.001], 'weight_decay': [1e-05, 5e-05], 'epochs': [50], 'optimizer': ['adam', 'adamw'], 'sequence_model': ['lstm', 'gru_small', 'gru_big'], 'feature_extractor': ['resnet50', 'efficientnetb0'], 'dropout_rate': [0.3, 0.5]}
    print(f'Configuration: 3 × 3 × 2 × 2 × 2 × 3 × 2 = 144 experiments')
    print(f'Estimated time: 8-16 hours\n')
    print('Parameters to test:')
    for param, values in param_grid.items():
        print(f'  • {param}: {values}')
    confirm = input('\n⚠️  This will take 8-16+ hours. Continue? (y/n): ').strip().lower()
    if confirm == 'y':
        if confirm_and_run(param_grid):
            save_config('comprehensive_tuning', param_grid)
    else:
        print('Cancelled.')

def run_custom_tuning():
    """Custom parameter grid."""
    print_banner('CUSTOM TUNING SETUP')
    print('Enter your custom parameter ranges.\n')
    param_grid = {}
    print('Batch Size options (comma-separated, e.g., 8,16,32):')
    bs_input = input('  > ').strip()
    if bs_input:
        param_grid['batch_size'] = [int(x.strip()) for x in bs_input.split(',')]
    else:
        param_grid['batch_size'] = [16]
    print('\nLearning Rate options (comma-separated, e.g., 1e-4,1e-3):')
    lr_input = input('  > ').strip()
    if lr_input:
        param_grid['learning_rate'] = [float(x.strip()) for x in lr_input.split(',')]
    else:
        param_grid['learning_rate'] = [0.0001]
    print('\nEpochs to train (comma-separated, e.g., 20,50):')
    epochs_input = input('  > ').strip()
    if epochs_input:
        param_grid['epochs'] = [int(x.strip()) for x in epochs_input.split(',')]
    else:
        param_grid['epochs'] = [50]
    print('\nSequence models (comma-separated):')
    print('  Options: lstm, gru_small, gru_big')
    models_input = input('  > ').strip()
    if models_input:
        param_grid['sequence_model'] = [x.strip() for x in models_input.split(',')]
    else:
        param_grid['sequence_model'] = ['lstm']
    print('\nFeature extractors (comma-separated):')
    print('  Options: resnet50, efficientnetb0, inceptionv3')
    fe_input = input('  > ').strip()
    if fe_input:
        param_grid['feature_extractor'] = [x.strip() for x in fe_input.split(',')]
    else:
        param_grid['feature_extractor'] = ['resnet50']
    param_grid['weight_decay'] = [1e-05]
    param_grid['optimizer'] = ['adam']
    param_grid['dropout_rate'] = [0.5]
    total_configs = 1
    for values in param_grid.values():
        total_configs *= len(values)
    print_banner('CUSTOM CONFIGURATION SUMMARY')
    print(f'Total configurations: {total_configs}\n')
    for param, values in param_grid.items():
        print(f'  • {param}: {values}')
    if total_configs > 30:
        print(f'\n⚠️  {total_configs} configurations is large.')
        search_type = input('Use random search instead of grid? (y/n): ').strip().lower()
        if search_type == 'y':
            search_type = 'random'
        else:
            search_type = 'grid'
    else:
        search_type = 'grid'
    if confirm_and_run(param_grid, search_type=search_type):
        save_config('custom_tuning', param_grid)

def confirm_and_run(param_grid, search_type='grid'):
    """Confirm and run tuning."""
    print('\n' + '-' * 70)
    confirm = input('Ready to start? (y/n): ').strip().lower()
    if confirm != 'y':
        print('Cancelled.')
        return False
    print('\n🚀 Starting tuning...')
    print('This may take a while. Monitor progress in the console.\n')
    script_content = f'\nimport os\nimport sys\nsys.path.insert(0, os.getcwd())\n\nfrom hyperparameter_tuning import HyperparameterTuner\n\nparam_grid = {param_grid}\ntuner = HyperparameterTuner(search_type="{search_type}", seed=42)\nresults = tuner.tune(param_grid)\n\nprint("\\n✓ Tuning complete!")\nprint(f"Results saved to: {{tuner.tuning_dir}}")\nprint(f"Best accuracy: {{tuner.best_score:.4f}}")\n'
    with open(_resolve_model_artifact('_run_tuning_temp.py'), 'w') as f:
        f.write(script_content)
    os.system('python _run_tuning_temp.py')
    os.remove('_run_tuning_temp.py')
    return True

def save_config(name, param_grid):
    """Save configuration for reference."""
    config = {'name': name, 'timestamp': datetime.now().isoformat(), 'param_grid': {k: str(v) for k, v in param_grid.items()}}
    config_file = f'tuning_config_{name}.json'
    with open(_resolve_model_artifact(config_file), 'w') as f:
        json.dump(config, f, indent=2)
    print(f'\n✓ Configuration saved to {config_file}')
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n❌ Tuning cancelled by user.')
        sys.exit(1)
    except Exception as e:
        print(f'\n\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
