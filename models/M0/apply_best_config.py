"""
Apply best hyperparameter configuration from tuning to config.py

Usage:
  python apply_best_config.py [tuning_results_dir]
  python apply_best_config.py results/tuning_20260216_143022
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
import shutil
from pathlib import Path

def apply_best_config(tuning_dir):
    """Load best config and apply to config.py."""
    best_config_file = os.path.join(tuning_dir, 'best_config.json')
    if not os.path.exists(_resolve_model_artifact(best_config_file)):
        print(f'❌ Best config file not found: {best_config_file}')
        return False
    with open(_resolve_model_artifact(best_config_file), 'r') as f:
        best_config = json.load(f)
    print('\n' + '=' * 70)
    print('APPLYING BEST HYPERPARAMETER CONFIGURATION')
    print('=' * 70 + '\n')
    print('Best configuration found:')
    for key, value in best_config.items():
        print(f'  {key}: {value}')
    config_file = 'config.py'
    with open(_resolve_model_artifact(config_file), 'r') as f:
        config_content = f.read()
    backup_file = f'{config_file}.backup'
    shutil.copy(config_file, backup_file)
    print(f'\n✓ Backed up original config to {backup_file}')
    updates = {'BATCH_SIZE': best_config.get('batch_size'), 'LEARNING_RATE': best_config.get('learning_rate'), 'WEIGHT_DECAY': best_config.get('weight_decay'), 'EPOCHS': best_config.get('epochs'), 'OPTIMIZER_NAME': f'''"{best_config.get('optimizer')}"''', 'SEQUENCE_MODEL': f'''"{best_config.get('sequence_model')}"''', 'FEATURE_EXTRACTOR': f'''"{best_config.get('feature_extractor')}"'''}
    for param, value in updates.items():
        if value is None:
            continue
        import re
        pattern = f'^{param}\\s*=\\s*.*$'
        new_assignment = f'{param} = {value}'
        config_content = re.sub(pattern, new_assignment, config_content, flags=re.MULTILINE)
    dropout_rate = best_config.get('dropout_rate')
    if dropout_rate is not None:
        if 'DROPOUT_RATE' not in config_content:
            config_content = config_content.replace('# Play categories', f'# Dropout rate\nDROPOUT_RATE = {dropout_rate}  # Regularization (0.0-1.0)\n\n# Play categories')
    with open(_resolve_model_artifact(config_file), 'w') as f:
        f.write(config_content)
    print(f'\n✓ Updated {config_file} with best parameters')
    print('\nUpdated parameters:')
    for param, value in updates.items():
        if value is not None:
            print(f'  {param} = {value}')
    if dropout_rate is not None:
        print(f'  DROPOUT_RATE = {dropout_rate}')
    print('\n' + '-' * 70)
    print('✨ Configuration updated!')
    print('-' * 70)
    print('\nNext steps:')
    print('1. Review the changes in config.py')
    print('2. Run: python train.py')
    print('3. Train your final model with the best parameters')
    summary_file = os.path.join(tuning_dir, 'applied_config.txt')
    with open(_resolve_model_artifact(summary_file), 'w') as f:
        f.write('BEST HYPERPARAMETER CONFIGURATION APPLIED\n')
        f.write('=' * 70 + '\n\n')
        f.write('Applied to: config.py\n')
        f.write('Backup saved to: config.py.backup\n\n')
        f.write('Configuration:\n')
        for key, value in best_config.items():
            f.write(f'  {key}: {value}\n')
    print(f'\n✓ Summary saved to {summary_file}')
    return True

def find_latest_tuning_dir():
    """Find the latest tuning directory."""
    results_dir = 'results'
    if not os.path.exists(_resolve_model_artifact(results_dir)):
        return None
    tuning_dirs = [d for d in os.listdir(results_dir) if d.startswith('tuning_') and os.path.isdir(_resolve_model_artifact(os.path.join(results_dir, d)))]
    if not tuning_dirs:
        return None
    latest = sorted(tuning_dirs)[-1]
    return os.path.join(results_dir, latest)

def main():
    """Main script."""
    if len(sys.argv) > 1:
        tuning_dir = sys.argv[1]
    else:
        tuning_dir = find_latest_tuning_dir()
        if tuning_dir:
            print(f'Found latest tuning directory: {tuning_dir}')
        else:
            print('❌ No tuning directory found. Usage:')
            print('   python apply_best_config.py [tuning_results_dir]')
            return
    if not os.path.isdir(_resolve_model_artifact(tuning_dir)):
        print(f'❌ Directory not found: {tuning_dir}')
        return
    print(f'\nApplying best configuration from: {tuning_dir}')
    confirm = input('Continue? (y/n): ').strip().lower()
    if confirm != 'y':
        print('Cancelled.')
        return
    apply_best_config(tuning_dir)
if __name__ == '__main__':
    main()
