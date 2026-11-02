"""
Utility script to analyze hyperparameter tuning results.
Run this after tuning to visualize and compare results.
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_tuning_results(tuning_dir):
    """Analyze and visualize tuning results."""
    results_file = os.path.join(tuning_dir, 'results.csv')
    best_config_file = os.path.join(tuning_dir, 'best_config.json')
    if not os.path.exists(_resolve_model_artifact(results_file)):
        print(f'❌ Results file not found: {results_file}')
        return
    df = pd.read_csv(_resolve_model_artifact(results_file))
    print('\n' + '=' * 70)
    print('HYPERPARAMETER TUNING ANALYSIS')
    print('=' * 70 + '\n')
    print('📊 OVERALL STATISTICS')
    print('-' * 70)
    print(f'Total experiments: {len(df)}')
    print(f"Successful: {len(df[df['status'] == 'completed'])}")
    print(f"Failed: {len(df[df['status'] == 'failed'])}")
    print(f"Best test accuracy: {df['test_accuracy'].max():.4f}")
    print(f"Mean test accuracy: {df['test_accuracy'].mean():.4f}")
    print(f"Worst test accuracy: {df['test_accuracy'].min():.4f}")
    print(f"Std dev: {df['test_accuracy'].std():.4f}\n")
    print('🏆 TOP 10 CONFIGURATIONS')
    print('-' * 70)
    df_sorted = df.sort_values('test_accuracy', ascending=False)
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        print(f"\n{i}. Test Accuracy: {row['test_accuracy']:.4f}")
        print(f"   Batch Size: {row['batch_size']}")
        print(f"   Learning Rate: {row['learning_rate']:.2e}")
        print(f"   Epochs: {int(row['epochs'])}")
        print(f"   Optimizer: {row['optimizer']}")
        print(f"   Sequence Model: {row['sequence_model']}")
        print(f"   Feature Extractor: {row['feature_extractor']}")
        print(f"   Dropout: {row['dropout_rate']}")
        print(f"   Duration: {row['duration_seconds']:.1f}s")
    print('\n\n📈 PARAMETER IMPACT ANALYSIS')
    print('-' * 70)
    if 'batch_size' in df.columns:
        print('\n▶ Batch Size Impact:')
        bs_analysis = df.groupby('batch_size')['test_accuracy'].agg(['mean', 'max', 'count'])
        print(bs_analysis)
    if 'learning_rate' in df.columns:
        print('\n▶ Learning Rate Impact:')
        lr_analysis = df.groupby('learning_rate')['test_accuracy'].agg(['mean', 'max', 'count'])
        print(lr_analysis)
    if 'sequence_model' in df.columns:
        print('\n▶ Sequence Model Impact:')
        sm_analysis = df.groupby('sequence_model')['test_accuracy'].agg(['mean', 'max', 'count'])
        print(sm_analysis)
    if 'feature_extractor' in df.columns:
        print('\n▶ Feature Extractor Impact:')
        fe_analysis = df.groupby('feature_extractor')['test_accuracy'].agg(['mean', 'max', 'count'])
        print(fe_analysis)
    if 'optimizer' in df.columns:
        print('\n▶ Optimizer Impact:')
        opt_analysis = df.groupby('optimizer')['test_accuracy'].agg(['mean', 'max', 'count'])
        print(opt_analysis)
    if 'dropout_rate' in df.columns:
        print('\n▶ Dropout Rate Impact:')
        dr_analysis = df.groupby('dropout_rate')['test_accuracy'].agg(['mean', 'max', 'count'])
        print(dr_analysis)
    print('\n\n📋 PER-CLASS PERFORMANCE (Best Model)')
    print('-' * 70)
    best_row = df_sorted.iloc[0]
    for col in df.columns:
        if '_f1' in col:
            class_name = col.replace('_f1', '')
            precision = best_row.get(f'{class_name}_precision', None)
            recall = best_row.get(f'{class_name}_recall', None)
            f1 = best_row.get(f'{class_name}_f1', None)
            if f1 is not None:
                print(f'  {class_name.upper()}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
    print('\n\n🔗 CORRELATION WITH TEST ACCURACY')
    print('-' * 70)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['test_accuracy'].sort_values(ascending=False)
    print(correlations.head(10))
    print('\n\n✨ BEST CONFIGURATION')
    print('-' * 70)
    best_config = {k: v for k, v in best_row.items() if k in ['batch_size', 'learning_rate', 'weight_decay', 'epochs', 'optimizer', 'sequence_model', 'feature_extractor', 'dropout_rate']}
    for key, value in best_config.items():
        print(f'  {key}: {value}')
    if os.path.exists(_resolve_model_artifact(best_config_file)):
        with open(_resolve_model_artifact(best_config_file), 'r') as f:
            saved_config = json.load(f)
        print(f'\nSaved to: {best_config_file}')
    best_config_out = os.path.join(tuning_dir, 'best_config_analysis.json')
    with open(_resolve_model_artifact(best_config_out), 'w') as f:
        json.dump(best_config, f, indent=2)
    print('\n\n📊 Generating comparison visualizations...')
    create_comparison_plots(df, tuning_dir)
    print('\n' + '=' * 70)
    print('✓ Analysis complete!')
    print('=' * 70 + '\n')

def create_comparison_plots(df, tuning_dir):
    """Create detailed comparison plots."""
    try:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :2])
        df_sorted = df.sort_values('experiment_num')
        ax1.plot(df_sorted['experiment_num'], df_sorted['test_accuracy'], 'o-', linewidth=2, markersize=6)
        ax1.fill_between(df_sorted['experiment_num'], df_sorted['test_accuracy'], alpha=0.3)
        ax1.axhline(df['test_accuracy'].max(), color='r', linestyle='--', alpha=0.5, label='Best')
        ax1.axhline(df['test_accuracy'].mean(), color='orange', linestyle='--', alpha=0.5, label='Mean')
        ax1.set_xlabel('Experiment Number')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Accuracy Trend Across Experiments')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(df['test_accuracy'], bins=15, color='skyblue', edgecolor='black')
        ax2.axvline(df['test_accuracy'].mean(), color='r', linestyle='--', label='Mean')
        ax2.set_xlabel('Test Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Accuracy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        if 'batch_size' in df.columns:
            ax3 = fig.add_subplot(gs[1, 0])
            df.boxplot(column='test_accuracy', by='batch_size', ax=ax3)
            ax3.set_title('Test Accuracy by Batch Size')
            ax3.set_xlabel('Batch Size')
            ax3.set_ylabel('Test Accuracy')
            plt.sca(ax3)
            plt.xticks(rotation=0)
        if 'learning_rate' in df.columns:
            ax4 = fig.add_subplot(gs[1, 1])
            lr_groups = df.groupby('learning_rate')['test_accuracy'].apply(list)
            ax4.boxplot(lr_groups.values, labels=[f'{x:.0e}' for x in lr_groups.index])
            ax4.set_title('Test Accuracy by Learning Rate')
            ax4.set_xlabel('Learning Rate')
            ax4.set_ylabel('Test Accuracy')
            ax4.tick_params(axis='x', rotation=45)
        if 'sequence_model' in df.columns:
            ax5 = fig.add_subplot(gs[1, 2])
            model_groups = df.groupby('sequence_model')['test_accuracy'].apply(list)
            ax5.boxplot(model_groups.values, labels=model_groups.index)
            ax5.set_title('Test Accuracy by Sequence Model')
            ax5.set_xlabel('Model')
            ax5.set_ylabel('Test Accuracy')
            ax5.tick_params(axis='x', rotation=45)
        if 'feature_extractor' in df.columns:
            ax6 = fig.add_subplot(gs[2, 0])
            fe_groups = df.groupby('feature_extractor')['test_accuracy'].apply(list)
            ax6.boxplot(fe_groups.values, labels=fe_groups.index)
            ax6.set_title('Test Accuracy by Feature Extractor')
            ax6.set_xlabel('Extractor')
            ax6.set_ylabel('Test Accuracy')
            ax6.tick_params(axis='x', rotation=45)
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.scatter(df['duration_seconds'], df['test_accuracy'], s=100, alpha=0.6)
        ax7.set_xlabel('Training Duration (seconds)')
        ax7.set_ylabel('Test Accuracy')
        ax7.set_title('Training Time vs Accuracy')
        ax7.grid(True, alpha=0.3)
        ax8 = fig.add_subplot(gs[2, 2])
        scatter = ax8.scatter(df['test_loss'], df['test_accuracy'], c=df['learning_rate'], cmap='viridis', s=100, alpha=0.6)
        ax8.set_xlabel('Test Loss')
        ax8.set_ylabel('Test Accuracy')
        ax8.set_title('Loss vs Accuracy')
        plt.colorbar(scatter, ax=ax8, label='Learning Rate')
        ax8.grid(True, alpha=0.3)
        plot_path = os.path.join(tuning_dir, 'detailed_analysis.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f'✓ Saved detailed analysis to {plot_path}')
    except Exception as e:
        print(f'⚠️ Failed to create detailed plots: {e}')

def main():
    """Main analysis script."""
    if len(sys.argv) > 1:
        tuning_dir = sys.argv[1]
    else:
        results_base = 'results'
        if os.path.exists(_resolve_model_artifact(results_base)):
            tuning_dirs = [d for d in os.listdir(results_base) if d.startswith('tuning_')]
            if tuning_dirs:
                latest = sorted(tuning_dirs)[-1]
                tuning_dir = os.path.join(results_base, latest)
                print(f'Found latest tuning directory: {tuning_dir}')
            else:
                print(f'No tuning directories found in {results_base}')
                return
        else:
            print(f'No {results_base} directory found')
            return
    analyze_tuning_results(tuning_dir)
if __name__ == '__main__':
    main()
