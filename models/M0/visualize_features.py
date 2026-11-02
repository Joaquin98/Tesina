"""Visualize features extracted by the feature extractor model."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import keras
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from models.M0.config import IMG_SIZE, MAX_SEQ_LENGTH, NUM_FEATURES, CHECKPOINTS_DIR, DATA_ROOT
from models.M0.data_utils import load_video, prepare_single_video_data

def load_feature_extractor():
    """Load the saved feature extractor model."""
    feature_extractor_path = os.path.join(CHECKPOINTS_DIR, 'feature_extractor_resnet50.keras')
    if not os.path.exists(_resolve_model_artifact(feature_extractor_path)):
        print(f'ERROR: Feature extractor not found at {feature_extractor_path}')
        return None
    print(f'Loading feature extractor from {feature_extractor_path}')
    return keras.models.load_model(_resolve_model_artifact(feature_extractor_path))

def extract_features_from_video(video_path, feature_extractor):
    """Extract raw features for each frame in a video."""
    print(f'\nExtracting features from: {video_path}')
    if not os.path.exists(_resolve_model_artifact(video_path)):
        print(f'ERROR: Video file not found: {video_path}')
        return (None, None)
    frames = load_video(video_path)
    if len(frames) == 0:
        print(f'ERROR: Could not load any frames from {video_path}')
        return (None, None)
    print(f'Loaded {len(frames)} frames')
    features = feature_extractor.predict(frames, verbose=0)
    print(f'Extracted features shape: {features.shape}')
    return (frames, features)

def plot_feature_statistics(features, title='Feature Statistics'):
    """Plot statistics about the extracted features."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    mean_features = np.mean(features, axis=0)
    axes[0, 0].bar(range(min(100, len(mean_features))), mean_features[:100])
    axes[0, 0].set_title('Mean Feature Values (first 100 features)')
    axes[0, 0].set_ylabel('Mean activation')
    var_features = np.var(features, axis=0)
    axes[0, 1].hist(var_features, bins=50, edgecolor='black')
    axes[0, 1].set_title('Distribution of Feature Variance Across Frames')
    axes[0, 1].set_xlabel('Variance')
    axes[0, 1].set_ylabel('Frequency')
    mean_per_frame = np.mean(features, axis=1)
    axes[1, 0].plot(mean_per_frame, linewidth=2, color='steelblue')
    axes[1, 0].set_title('Mean Activation Per Frame')
    axes[1, 0].set_xlabel('Frame number')
    axes[1, 0].set_ylabel('Mean activation')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].hist(features.flatten(), bins=100, edgecolor='black', color='coral')
    axes[1, 1].set_title('Overall Feature Value Distribution')
    axes[1, 1].set_xlabel('Feature value')
    axes[1, 1].set_ylabel('Frequency')
    plt.tight_layout()
    return fig

def plot_pca_visualization(features, frames=None, title='PCA Visualization of Features'):
    """Use PCA to reduce features to 2D and visualize."""
    print('\nPerforming PCA dimensionality reduction...')
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    print(f'PCA explained variance ratio: {pca.explained_variance_ratio_}')
    print(f'Total variance explained: {sum(pca.explained_variance_ratio_):.2%}')
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=range(len(features)), cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(title)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Frame number (temporal progression)')
    for i in range(len(features_2d) - 1):
        ax.annotate('', xy=features_2d[i + 1], xytext=features_2d[i], arrowprops=dict(arrowstyle='->', lw=0.5, alpha=0.3, color='gray'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return (fig, pca)

def plot_feature_heatmap(features, title='Feature Heatmap'):
    """Visualize features as a heatmap (frames x features)."""
    features_norm = (features - features.min()) / (features.max() - features.min())
    fig, ax = plt.subplots(figsize=(16, 6))
    step = max(1, NUM_FEATURES // 200)
    features_sampled = features_norm[:, ::step]
    im = ax.imshow(features_sampled.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Frame number')
    ax.set_ylabel(f'Feature index (step={step})')
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized activation')
    plt.tight_layout()
    return fig

def plot_top_features(features, frames=None, top_n=16, title='Top Variable Features'):
    """Visualize the most variable features (most informative)."""
    feature_variance = np.var(features, axis=0)
    top_indices = np.argsort(feature_variance)[-top_n:]
    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    axes_flat = axes.flatten()
    for idx, feature_idx in enumerate(top_indices):
        ax = axes_flat[idx]
        ax.plot(features[:, feature_idx], linewidth=2, color='steelblue')
        ax.set_title(f'Feature {feature_idx}\n(var={feature_variance[feature_idx]:.2f})')
        ax.set_ylabel('Activation')
        ax.grid(True, alpha=0.3)
        if idx >= 12:
            ax.set_xlabel('Frame')
    plt.tight_layout()
    return fig

def create_feature_video_comparison(frames, features, output_path='feature_comparison.mp4'):
    """Create a video showing frames alongside their top features (visualization)."""
    print(f'\nCreating comparison visualization...')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for frame_idx in range(min(10, len(frames))):
        frame = frames[frame_idx]
        feature_vec = features[frame_idx]
        frame_display = frame.astype(np.uint8)
        axes[0].clear()
        axes[0].imshow(frame_display)
        axes[0].set_title(f'Frame {frame_idx}')
        axes[0].axis('off')
        axes[1].clear()
        top_features = np.argsort(feature_vec)[-20:]
        top_values = feature_vec[top_features]
        axes[1].barh(range(len(top_features)), top_values, color='steelblue')
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels([f'F{i}' for i in top_features])
        axes[1].set_xlabel('Activation value')
        axes[1].set_title(f'Top 20 Features')
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[2].clear()
        axes[2].hist(feature_vec, bins=50, edgecolor='black', color='coral')
        axes[2].axvline(np.mean(feature_vec), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(feature_vec):.2f}')
        axes[2].axvline(np.median(feature_vec), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(feature_vec):.2f}')
        axes[2].set_xlabel('Feature value')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'Feature Distribution (Frame {frame_idx})')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.pause(0.1)
    plt.close(fig)

def main():
    """Main visualization pipeline."""
    print('=' * 60)
    print('Feature Extractor Visualization Tool')
    print('=' * 60)
    feature_extractor = load_feature_extractor()
    if feature_extractor is None:
        return
    test_videos = ['f1_ataque_000_cut.mp4', 'F3_line_014_cut.mp4', 'F10_scrum_007_cut.mp4']
    video_path = None
    for video_name in test_videos:
        candidate = os.path.join(DATA_ROOT, video_name)
        if os.path.exists(_resolve_model_artifact(candidate)):
            video_path = candidate
            print(f'\nUsing video: {video_name}')
            break
    if video_path is None:
        print(f'ERROR: None of the test videos found in {DATA_ROOT}')
        print('Available videos in that directory:')
        if os.path.exists(_resolve_model_artifact(DATA_ROOT)):
            videos = [f for f in os.listdir(DATA_ROOT) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            for v in videos[:10]:
                print(f'  - {v}')
            if len(videos) > 10:
                print(f'  ... and {len(videos) - 10} more')
        return
    frames, features = extract_features_from_video(video_path, feature_extractor)
    if frames is None or features is None:
        return
    print('\nGenerating visualizations...')
    fig1 = plot_feature_statistics(features, title=f'Feature Statistics - {os.path.basename(video_path)}')
    fig1.savefig(os.path.join(CHECKPOINTS_DIR, 'feature_statistics.png'), dpi=100, bbox_inches='tight')
    print('Saved: feature_statistics.png')
    fig2, pca = plot_pca_visualization(features, frames, title=f'PCA Visualization - {os.path.basename(video_path)}')
    fig2.savefig(os.path.join(CHECKPOINTS_DIR, 'feature_pca.png'), dpi=100, bbox_inches='tight')
    print('Saved: feature_pca.png')
    fig3 = plot_feature_heatmap(features, title=f'Feature Heatmap - {os.path.basename(video_path)}')
    fig3.savefig(os.path.join(CHECKPOINTS_DIR, 'feature_heatmap.png'), dpi=100, bbox_inches='tight')
    print('Saved: feature_heatmap.png')
    fig4 = plot_top_features(features, frames, top_n=16, title=f'Top 16 Variable Features - {os.path.basename(video_path)}')
    fig4.savefig(os.path.join(CHECKPOINTS_DIR, 'top_variable_features.png'), dpi=100, bbox_inches='tight')
    print('Saved: top_variable_features.png')
    print('\nAll visualizations saved to:', CHECKPOINTS_DIR)
    print('\nShowing plots...')
    plt.show()
if __name__ == '__main__':
    main()
