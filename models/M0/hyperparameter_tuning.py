"""Tune sequence classifiers using current train, validation and test CSV files."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from models.common.legacy_training import DATASET as CURRENT_DATASET, auxiliary_directory, cached_rgb as current_rgb, current_frames, validation_data as current_validation
import os
import sys
import time
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from pathlib import Path
from datetime import datetime
from itertools import product
import shutil
from models.M0.config import *
from models.M0.model import FEATURE_EXTRACTORS, SEQUENCE_MODELS, build_optimizer
from models.M0.data_utils import load_or_prepare_data, prepare_all_videos_data
try:
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print('⚠️  scikit-learn not available. Install it for detailed metrics: pip install scikit-learn')
    SKLEARN_AVAILABLE = False

class HyperparameterTuner:
    """Hyperparameter tuning orchestrator."""

    def __init__(self, search_type='grid', seed=42):
        """
        Initialize the tuner.
        
        Args:
            search_type: "grid" for grid search, "random" for random search
            seed: Random seed for reproducibility
        """
        self.search_type = search_type
        self.seed = seed
        self.results = []
        self.best_model_config = None
        self.best_score = -np.inf
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tuning_dir = os.path.join(RESULTS_DIR, f'tuning_{timestamp}')
        os.makedirs(self.tuning_dir, exist_ok=True)
        self.log_file = os.path.join(self.tuning_dir, 'tuning_log.txt')
        self.results_csv = os.path.join(self.tuning_dir, 'results.csv')
        self.best_config_file = os.path.join(self.tuning_dir, 'best_config.json')
        np.random.seed(seed)

    def log(self, message):
        """Log a message to both console and file."""
        print(message)
        with open(_resolve_model_artifact(self.log_file), 'a') as f:
            f.write(message + '\n')

    def run_single_experiment(self, config, experiment_num, total_experiments):
        """
        Run a single training experiment with given hyperparameters.
        
        Returns:
            Dictionary with results
        """
        self.log(f"\n{'=' * 70}")
        self.log(f'Experiment {experiment_num}/{total_experiments}')
        self.log(f'Configuration: {json.dumps(config, indent=2)}')
        self.log(f"{'=' * 70}\n")
        batch_size = config['batch_size']
        learning_rate = config['learning_rate']
        weight_decay = config['weight_decay']
        epochs = config['epochs']
        optimizer_name = config['optimizer']
        sequence_model_name = config['sequence_model']
        feature_extractor_name = config['feature_extractor']
        dropout_rate = config.get('dropout_rate', 0.5)
        start_time = time.time()
        result = {**config, 'experiment_num': experiment_num, 'start_time': datetime.now().isoformat(), 'status': 'running'}
        try:
            self.log('Loading training and test data...')
            train_df = pd.read_csv(_resolve_model_artifact(TRAIN_CSV))
            test_df = pd.read_csv(_resolve_model_artifact(TEST_CSV))
            if CLASSES_TO_USE is not None:
                train_df = train_df[train_df['tag'].isin(CLASSES_TO_USE)].copy()
                test_df = test_df[test_df['tag'].isin(CLASSES_TO_USE)].copy()
                self.log(f'Filtered to classes: {CLASSES_TO_USE}')
            self.log(f'Training samples: {len(train_df)}, Test samples: {len(test_df)}')
            if feature_extractor_name not in FEATURE_EXTRACTORS:
                raise ValueError(f"Feature extractor '{feature_extractor_name}' not found")
            from models.common.legacy_training import _rgb_source
            feature_extractor = _rgb_source(feature_extractor_name).feature_extractor
            self.log(f'Built feature extractor: {feature_extractor_name}')
            label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(train_df['tag']))
            label_vocab = label_processor.get_vocabulary()
            self.log(f'Label vocabulary: {label_vocab}')
            self.log(f'Number of classes: {len(label_vocab)}')
            self.log('Loading/preparing data...')
            from models.common.legacy_training import rgb_dataset
            (train_data, train_masks), train_labels = rgb_dataset(train_df, label_vocab, feature_extractor_name)
            (test_data, test_masks), test_labels = rgb_dataset(test_df, label_vocab, feature_extractor_name)
            self.log(f'Train shape: {train_data.shape}, Test shape: {test_data.shape}')
            optimizer = build_optimizer(optimizer_name, learning_rate, weight_decay)
            self.log(f'Built optimizer: {optimizer_name}, lr={learning_rate}, wd={weight_decay}')
            if sequence_model_name not in SEQUENCE_MODELS:
                raise ValueError(f"Sequence model '{sequence_model_name}' not found")
            num_classes = len(label_vocab)
            seq_model_builder = SEQUENCE_MODELS[sequence_model_name]
            try:
                seq_model = seq_model_builder(num_classes, optimizer, dropout_rate=dropout_rate)
            except TypeError:
                seq_model = seq_model_builder(num_classes, optimizer)
            self.log(f'Built sequence model: {sequence_model_name}')
            self.log(f'Model summary:\n{seq_model.summary()}')
            self.log('Starting training...')
            history = seq_model.fit([train_data, train_masks], train_labels, validation_data=current_validation(label_vocab, feature_extractor_name), epochs=epochs, batch_size=batch_size, verbose=1)
            self.log('Evaluating on test set...')
            test_loss, test_accuracy = seq_model.evaluate([test_data, test_masks], test_labels, verbose=0)
            result['test_loss'] = float(test_loss)
            result['test_accuracy'] = float(test_accuracy)
            result['train_loss'] = float(history.history['loss'][-1])
            result['train_accuracy'] = float(history.history['accuracy'][-1])
            if 'val_loss' in history.history:
                result['val_loss'] = float(history.history['val_loss'][-1])
            if 'val_accuracy' in history.history:
                result['val_accuracy'] = float(history.history['val_accuracy'][-1])
            if SKLEARN_AVAILABLE:
                y_prob = seq_model.predict([test_data, test_masks], verbose=0)
                y_pred = np.argmax(y_prob, axis=1)
                y_true = test_labels
                report_dict = classification_report(y_true, y_pred, target_names=label_vocab, output_dict=True, zero_division=0)
                for class_name in label_vocab:
                    if class_name in report_dict:
                        result[f'{class_name}_precision'] = float(report_dict[class_name]['precision'])
                        result[f'{class_name}_recall'] = float(report_dict[class_name]['recall'])
                        result[f'{class_name}_f1'] = float(report_dict[class_name]['f1-score'])
            model_path = os.path.join(self.tuning_dir, f'model_exp{experiment_num}.keras')
            seq_model.save(model_path)
            result['model_path'] = model_path
            self._save_history_plot(history, experiment_num, sequence_model_name)
            result['status'] = 'completed'
            result['duration_seconds'] = time.time() - start_time
            self.log(f'✓ Experiment completed. Test accuracy: {test_accuracy:.4f}')
            validation_score = float(history.history.get('val_accuracy', [0.])[-1])
            result['validation_accuracy'] = validation_score
            if validation_score > self.best_score:
                self.best_score = validation_score
                self.best_model_config = config
                self.log(f'Best validation accuracy: {validation_score:.4f}')
        except Exception as e:
            self.log(f'✗ Experiment failed: {str(e)}')
            result['status'] = 'failed'
            result['error'] = str(e)
            result['duration_seconds'] = time.time() - start_time
            import traceback
            self.log(traceback.format_exc())
        return result

    def _save_history_plot(self, history, exp_num, model_name):
        """Save training history plot."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            axes[0].plot(history.history.get('loss', []), label='train_loss', linewidth=2)
            if 'val_loss' in history.history:
                axes[0].plot(history.history.get('val_loss', []), label='val_loss', linewidth=2)
            axes[0].set_title(f'Loss - Exp{exp_num} ({model_name})')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(history.history.get('accuracy', []), label='train_acc', linewidth=2)
            if 'val_accuracy' in history.history:
                axes[1].plot(history.history.get('val_accuracy', []), label='val_acc', linewidth=2)
            axes[1].set_title(f'Accuracy - Exp{exp_num} ({model_name})')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            plot_path = os.path.join(self.tuning_dir, f'history_exp{exp_num}.png')
            plt.savefig(plot_path, dpi=100)
            plt.close(fig)
            self.log(f'Saved history plot to {plot_path}')
        except Exception as e:
            self.log(f'Failed to save history plot: {e}')

    def _generate_configs(self, param_grid):
        """Generate configurations for tuning."""
        if self.search_type == 'grid':
            return self._grid_search(param_grid)
        elif self.search_type == 'random':
            return self._random_search(param_grid)
        else:
            raise ValueError(f'Unknown search type: {self.search_type}')

    def _grid_search(self, param_grid):
        """Generate all combinations of parameters (grid search)."""
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        configs = []
        for combination in product(*values):
            config = dict(zip(keys, combination))
            configs.append(config)
        return configs

    def _random_search(self, param_grid, n_iter=20):
        """Generate random combinations of parameters (random search)."""
        configs = []
        for _ in range(n_iter):
            config = {}
            for key, values in param_grid.items():
                if isinstance(values, list):
                    config[key] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    config[key] = np.random.uniform(values[0], values[1])
                else:
                    config[key] = np.random.choice(values)
            configs.append(config)
        return configs

    def tune(self, param_grid):
        """
        Run hyperparameter tuning.
        
        Args:
            param_grid: Dictionary of parameter names to lists of values to try
                       Example:
                       {
                           'batch_size': [8, 16, 32],
                           'learning_rate': [1e-4, 5e-4, 1e-3],
                           'epochs': [10, 20],
                           'optimizer': ['adam', 'adamw'],
                           'sequence_model': ['lstm', 'gru_small'],
                           'feature_extractor': ['resnet50', 'efficientnetb0'],
                           'dropout_rate': [0.3, 0.5]
                       }
        """
        self.log(f"\n{'=' * 70}")
        self.log(f'Starting Hyperparameter Tuning ({self.search_type.upper()} SEARCH)')
        self.log(f'Search Type: {self.search_type}')
        self.log(f'Random Seed: {self.seed}')
        self.log(f'Tuning Directory: {self.tuning_dir}')
        self.log(f"{'=' * 70}\n")
        configs = self._generate_configs(param_grid)
        self.log(f'Generated {len(configs)} configurations to test\n')
        for i, config in enumerate(configs, 1):
            result = self.run_single_experiment(config, i, len(configs))
            self.results.append(result)
            self._save_results()
        self._print_summary()
        return self.results

    def _save_results(self):
        """Save results to CSV."""
        if not self.results:
            return
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_csv, index=False)
        self.log(f'Results saved to {self.results_csv}')
        if self.best_model_config:
            with open(_resolve_model_artifact(self.best_config_file), 'w') as f:
                json.dump(self.best_model_config, f, indent=2)
            self.log(f'Best config saved to {self.best_config_file}')

    def _print_summary(self):
        """Print and save summary of tuning results."""
        self.log(f"\n{'=' * 70}")
        self.log('TUNING SUMMARY')
        self.log(f"{'=' * 70}\n")
        df = pd.DataFrame(self.results)
        df_sorted = df.sort_values('validation_accuracy', ascending=False)
        self.log('\nTop 5 Best Configurations:\n')
        top_n = min(5, len(df_sorted))
        for i, (_, row) in enumerate(df_sorted.head(top_n).iterrows(), 1):
            self.log(f"{i}. Test Accuracy: {row['test_accuracy']:.4f}")
            self.log(f"   Config: {json.dumps({k: v for k, v in row.items() if k not in ['status', 'model_path', 'duration_seconds', 'start_time']}, indent=6)}")
            self.log(f"   Duration: {row['duration_seconds']:.1f}s\n")
        self.log(f'\nBest Overall Accuracy: {self.best_score:.4f}')
        self.log(f'Best Configuration: {json.dumps(self.best_model_config, indent=2)}')
        self._create_summary_plots(df)
        self.log(f"\n{'=' * 70}")
        self.log(f'Tuning completed! Results saved to {self.tuning_dir}')
        self.log(f"{'=' * 70}\n")

    def _create_summary_plots(self, df):
        """Create visualization of tuning results."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes[0, 0].plot(df['experiment_num'], df['test_accuracy'], 'o-', linewidth=2, markersize=6)
            axes[0, 0].axhline(df['test_accuracy'].max(), color='r', linestyle='--', alpha=0.5, label='Best')
            axes[0, 0].set_xlabel('Experiment')
            axes[0, 0].set_ylabel('Test Accuracy')
            axes[0, 0].set_title('Test Accuracy by Experiment')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            axes[0, 1].scatter(df['test_loss'], df['test_accuracy'], s=100, alpha=0.6)
            axes[0, 1].set_xlabel('Test Loss')
            axes[0, 1].set_ylabel('Test Accuracy')
            axes[0, 1].set_title('Loss vs Accuracy')
            axes[0, 1].grid(True, alpha=0.3)
            axes[1, 0].bar(df['experiment_num'], df['duration_seconds'])
            axes[1, 0].set_xlabel('Experiment')
            axes[1, 0].set_ylabel('Duration (seconds)')
            axes[1, 0].set_title('Training Duration by Experiment')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            if 'learning_rate' in df.columns:
                unique_lrs = df['learning_rate'].unique()
                if len(unique_lrs) > 1:
                    for lr in unique_lrs:
                        subset = df[df['learning_rate'] == lr]
                        axes[1, 1].scatter(subset['epoch'], subset['test_accuracy'], label=f'lr={lr}', s=100, alpha=0.6)
                    axes[1, 1].set_xlabel('Epochs')
                    axes[1, 1].set_ylabel('Test Accuracy')
                    axes[1, 1].set_title('Learning Rate Impact')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
            fig.tight_layout()
            plot_path = os.path.join(self.tuning_dir, 'tuning_summary.png')
            plt.savefig(plot_path, dpi=100)
            plt.close(fig)
            self.log(f'Saved summary plot to {plot_path}')
        except Exception as e:
            self.log(f'Failed to create summary plots: {e}')

def main():
    """Main tuning script."""
    param_grid_small = {'batch_size': [8, 16], 'learning_rate': [0.0001, 0.0005], 'weight_decay': [1e-05], 'epochs': [5, 10], 'optimizer': ['adam'], 'sequence_model': ['lstm', 'gru_small'], 'feature_extractor': ['resnet50'], 'dropout_rate': [0.3, 0.5]}
    param_grid_comprehensive = {'batch_size': [8, 16, 32], 'learning_rate': [0.0001, 0.0005, 0.001], 'weight_decay': [1e-05, 5e-05], 'epochs': [20, 50], 'optimizer': ['adam', 'adamw'], 'sequence_model': ['lstm', 'gru_small', 'gru_big'], 'feature_extractor': ['resnet50', 'efficientnetb0', 'inceptionv3'], 'dropout_rate': [0.3, 0.5, 0.7]}
    print('Hyperparameter Tuning Options:')
    print('1. Quick Test (small grid, ~8 configurations)')
    print('2. Comprehensive Tuning (large grid, ~144 configurations)')
    print('3. Random Search (20 random configurations)')
    choice = input('Select option (1/2/3): ').strip()
    if choice == '1':
        param_grid = param_grid_small
        search_type = 'grid'
    elif choice == '2':
        param_grid = param_grid_comprehensive
        search_type = 'grid'
    elif choice == '3':
        param_grid = param_grid_comprehensive
        search_type = 'random'
    else:
        print('Invalid choice. Using quick test.')
        param_grid = param_grid_small
        search_type = 'grid'
    tuner = HyperparameterTuner(search_type=search_type, seed=42)
    results = tuner.tune(param_grid)
    print(f'\n✓ Tuning completed! Results saved to: {tuner.tuning_dir}')
    print(f'Best model accuracy: {tuner.best_score:.4f}')
    print(f'Best configuration:\n{json.dumps(tuner.best_model_config, indent=2)}')
if __name__ == '__main__':
    main()
