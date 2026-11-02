"""Dataset classes for Two-Stream Network training."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from optical_flow import compute_optical_flow
from config import NUM_FLOW_FRAMES, IMG_SIZE
logger = logging.getLogger(__name__)

class TwoStreamDataset(Dataset):
    """
    Dataset for Two-Stream Network.
    
    Loads videos and computes optical flow on-the-fly.
    Provides RGB frame and stacked optical flow for each video.
    """

    def __init__(self, csv_file, video_dir, split='train', num_flow_frames=NUM_FLOW_FRAMES, img_size=IMG_SIZE, transform=None, cache_flow=False):
        """
        Args:
            csv_file: Path to CSV file with columns: video_name, tag
            video_dir: Directory containing video files
            split: 'train' or 'test'
            num_flow_frames: Number of optical flow frames to extract
            img_size: Target image size
            transform: Torchvision transforms to apply
            cache_flow: If True, cache optical flow in memory (useful for small datasets)
        """
        self.annotations = pd.read_csv(_resolve_model_artifact(csv_file))
        self.video_dir = Path(video_dir)
        self.num_flow_frames = num_flow_frames
        self.img_size = img_size
        self.transform = transform
        self.cache_flow = cache_flow
        self.class_to_idx = {'line': 0, 'other': 1, 'scrum': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.flow_cache = {}
        self.rgb_cache = {}
        logger.info(f'Loaded dataset with {len(self.annotations)} samples from {csv_file}')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - 'rgb': RGB frame, shape (3, H, W), values in [0, 1]
                - 'flow': Stacked optical flow, shape (2*num_flow_frames, H, W), values in [-1, 1]
                - 'label': Class index
                - 'video_name': Video filename
        """
        row = self.annotations.iloc[idx]
        video_name = row['video_name']
        label = self.class_to_idx[row['tag']]
        video_path = self.video_dir / video_name
        if self.cache_flow and video_name in self.flow_cache:
            flow_stack = self.flow_cache[video_name]
            rgb_frame = self.rgb_cache[video_name]
        else:
            try:
                flow_stack, rgb_frame = compute_optical_flow(video_path, num_flow_frames=self.num_flow_frames, target_size=(self.img_size, self.img_size))
                if self.cache_flow:
                    self.flow_cache[video_name] = flow_stack
                    self.rgb_cache[video_name] = rgb_frame
            except Exception as e:
                logger.error(f'Error loading video {video_path}: {e}')
                flow_stack = np.zeros((self.num_flow_frames, self.img_size, self.img_size, 2), dtype=np.float32)
                rgb_frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        flow_tensor = self._flow_to_tensor(flow_stack)
        rgb_tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1)
        if self.transform:
            rgb_tensor, flow_tensor = self.transform(rgb_tensor, flow_tensor)
        return {'rgb': rgb_tensor, 'flow': flow_tensor, 'label': torch.tensor(label, dtype=torch.long), 'video_name': video_name}

    @staticmethod
    def _flow_to_tensor(flow_stack):
        """
        Convert flow stack to tensor.
        
        Args:
            flow_stack: (num_flow_frames, H, W, 2)
        
        Returns:
            flow_tensor: (2*num_flow_frames, H, W)
        """
        num_frames, h, w, _ = flow_stack.shape
        flow_tensor = np.zeros((2 * num_frames, h, w), dtype=np.float32)
        for i in range(num_frames):
            flow_tensor[2 * i, :, :] = flow_stack[i, :, :, 0]
            flow_tensor[2 * i + 1, :, :] = flow_stack[i, :, :, 1]
        return torch.from_numpy(flow_tensor)

    def get_class_distribution(self):
        """Get class distribution in dataset."""
        distribution = self.annotations['tag'].value_counts().to_dict()
        return {self.class_to_idx[k]: v for k, v in distribution.items()}

    def get_class_weights(self):
        """
        Compute class weights for handling imbalance.
        
        Returns:
            weights: Tensor of shape (num_classes,)
        """
        distribution = self.get_class_distribution()
        total = sum(distribution.values())
        weights = torch.zeros(len(self.class_to_idx))
        for class_idx, count in distribution.items():
            weights[class_idx] = total / (len(self.class_to_idx) * (count + 1))
        return weights

    def clear_cache(self):
        """Clear cached flows and RGB frames."""
        self.flow_cache.clear()
        self.rgb_cache.clear()
        logger.info('Cache cleared')

class TwoStreamDataLoader:
    """Utility class for creating data loaders."""

    @staticmethod
    def create_dataloaders(train_csv, test_csv, video_dir, batch_size=16, num_workers=4, cache_flow=False, num_flow_frames=NUM_FLOW_FRAMES):
        """
        Create training and testing data loaders.
        
        Args:
            train_csv: Path to training CSV
            test_csv: Path to testing CSV
            video_dir: Directory containing videos
            batch_size: Batch size
            num_workers: Number of workers for data loading
            cache_flow: Whether to cache optical flows
            num_flow_frames: Number of flow frames
        
        Returns:
            train_loader, test_loader, class_weights
        """
        train_dataset = TwoStreamDataset(csv_file=train_csv, video_dir=video_dir, split='train', num_flow_frames=num_flow_frames, cache_flow=cache_flow)
        test_dataset = TwoStreamDataset(csv_file=test_csv, video_dir=video_dir, split='test', num_flow_frames=num_flow_frames, cache_flow=cache_flow)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        class_weights = train_dataset.get_class_weights()
        return (train_loader, test_loader, class_weights)

class TwoStreamCollater:
    """Custom collate function for Two-Stream dataset."""

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for batch processing.
        
        Args:
            batch: List of samples from dataset
        
        Returns:
            Dictionary with batched tensors
        """
        rgb_list = []
        flow_list = []
        labels_list = []
        video_names_list = []
        for sample in batch:
            rgb_list.append(sample['rgb'])
            flow_list.append(sample['flow'])
            labels_list.append(sample['label'])
            video_names_list.append(sample['video_name'])
        return {'rgb': torch.stack(rgb_list), 'flow': torch.stack(flow_list), 'label': torch.stack(labels_list), 'video_names': video_names_list}
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dataset = TwoStreamDataset(csv_file='train.csv', video_dir='videos/', num_flow_frames=10)
    print(f'Dataset size: {len(dataset)}')
    print(f'Class distribution: {dataset.get_class_distribution()}')
    print(f'Class weights: {dataset.get_class_weights()}')
    sample = dataset[0]
    print(f"RGB shape: {sample['rgb'].shape}")
    print(f"Flow shape: {sample['flow'].shape}")
    print(f"Label: {sample['label']}")
