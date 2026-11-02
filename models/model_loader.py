"""
Model loader utility for web interface.

This module handles loading different model architectures from checkpoints.
"""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[1]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from pathlib import Path
import torch
import torch.nn as nn

def load_model_from_checkpoint(checkpoint: dict, device: torch.device):
    """
    Load model from checkpoint dictionary.
    
    Args:
        checkpoint: Checkpoint dictionary from torch.load()
        device: Device to load model to
    
    Returns:
        Loaded model
    """
    config = checkpoint.get('config', {})
    model_type = config.get('model', {}).get('name', 'resnet3d18')
    if 'resnet' in model_type.lower():
        model = load_resnet3d_model(config, device)
    elif 'lstm' in model_type.lower():
        model = load_lstm_model(config, device)
    elif 'transformer' in model_type.lower():
        model = load_transformer_model(config, device)
    else:
        model = load_resnet3d_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def load_resnet3d_model(config: dict, device: torch.device):
    """Load ResNet3D model."""
    try:
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=False, num_classes=3)
        return model
    except Exception as e:
        print(f'Error loading ResNet3D: {e}')
        return SimpleCNN3D()

def load_lstm_model(config: dict, device: torch.device):
    """Load LSTM model."""
    return SimpleLSTM()

def load_transformer_model(config: dict, device: torch.device):
    """Load Transformer model."""
    return SimpleTransformer()

class SimpleCNN3D(nn.Module):
    """Simple 3D CNN for video classification."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 2 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class SimpleLSTM(nn.Module):
    """Simple LSTM for video classification."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.lstm = nn.LSTM(128 * 56 * 56, 512, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch, c, t, h, w = x.shape
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch * t, c, h, w)
        x = self.cnn(x)
        x = x.view(batch, t, -1)
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

class SimpleTransformer(nn.Module):
    """Simple Transformer for video classification."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, 64 * 112 * 112))
        encoder_layer = nn.TransformerEncoderLayer(d_model=64 * 112 * 112, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64 * 112 * 112, num_classes)

    def forward(self, x):
        batch, c, t, h, w = x.shape
        x = x.transpose(1, 2).contiguous().view(batch * t, c, h, w)
        x = self.cnn(x)
        x = x.view(batch, t, -1)
        x = x + self.pos_encoding[:, :t, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
