"""Two-Stream Network model architecture for video classification."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = _DeliveryPath(__file__).resolve().parents[2]
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import NUM_CLASSES, BACKBONE, FUSION_TYPE, NUM_FLOW_FRAMES, DROPOUT_RATE, PRETRAINED

class SpatialStream(nn.Module):
    """
    Spatial Stream Network - processes RGB frames.
    Uses ResNet-50/101 backbone for appearance feature extraction.
    """

    def __init__(self, num_classes=NUM_CLASSES, backbone=BACKBONE, pretrained=PRETRAINED):
        super(SpatialStream, self).__init__()
        if backbone == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(self.feature_dim, num_classes))

    def forward(self, x):
        """
        Args:
            x: RGB frame, shape (B, 3, H, W) with values in [0, 1]
        
        Returns:
            features: Global average pooled features, shape (B, feature_dim)
            logits: Classification logits, shape (B, num_classes)
        """
        x = self.features(x)
        x = x.squeeze(-1).squeeze(-1)
        features = x
        logits = self.classifier(x)
        return (features, logits)

    def freeze_backbone(self):
        """Freeze backbone parameters (transfer learning)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters (fine-tuning)."""
        for param in self.features.parameters():
            param.requires_grad = True

class TemporalStream(nn.Module):
    """
    Temporal Stream Network - processes optical flow.
    Uses modified ResNet-50/101 to handle 2*num_flow_frames input channels.
    """

    def __init__(self, num_classes=NUM_CLASSES, num_flow_frames=NUM_FLOW_FRAMES, backbone=BACKBONE, pretrained=PRETRAINED):
        super(TemporalStream, self).__init__()
        self.num_flow_frames = num_flow_frames
        if backbone == 'resnet50':
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')
        input_channels = 2 * num_flow_frames
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            original_weight = resnet.conv1.weight.data
            repeated_weight = original_weight.repeat(1, (input_channels + 2) // 3, 1, 1)
            self.conv1.weight.data = repeated_weight[:, :input_channels, :, :]
            self.conv1.weight.data = self.conv1.weight.data / (input_channels / 3.0)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.feature_dim = resnet.fc.in_features
        self.classifier = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(self.feature_dim, num_classes))

    def forward(self, x):
        """
        Args:
            x: Stacked optical flow, shape (B, 2*num_flow_frames, H, W) with values in [-1, 1]
        
        Returns:
            features: Global average pooled features, shape (B, feature_dim)
            logits: Classification logits, shape (B, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        features = x
        logits = self.classifier(x)
        return (features, logits)

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        layers_to_freeze = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        layers = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

class TwoStreamNetwork(nn.Module):
    """
    Two-Stream Network combining spatial and temporal streams.
    
    Supports two fusion strategies:
    - Late fusion: Average predictions from both streams
    - Early fusion: Concatenate features before classification
    """

    def __init__(self, num_classes=NUM_CLASSES, num_flow_frames=NUM_FLOW_FRAMES, backbone=BACKBONE, fusion_type=FUSION_TYPE, pretrained=PRETRAINED):
        super(TwoStreamNetwork, self).__init__()
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.spatial_stream = SpatialStream(num_classes, backbone, pretrained)
        self.temporal_stream = TemporalStream(num_classes, num_flow_frames, backbone, pretrained)
        if fusion_type == 'late':
            self.spatial_weight = nn.Parameter(torch.tensor(0.5))
            self.temporal_weight = nn.Parameter(torch.tensor(0.5))
        elif fusion_type == 'early':
            feature_dim = self.spatial_stream.feature_dim * 2
            self.fusion_layer = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(feature_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(DROPOUT_RATE), nn.Linear(1024, num_classes))
        else:
            raise ValueError(f'Unsupported fusion type: {fusion_type}')

    def forward(self, rgb, flow):
        """
        Forward pass through two-stream network.
        
        Args:
            rgb: RGB frame, shape (B, 3, H, W), values in [0, 1]
            flow: Stacked optical flow, shape (B, 2*num_flow_frames, H, W), values in [-1, 1]
        
        Returns:
            output: Classification logits, shape (B, num_classes)
            spatial_logits: Spatial stream predictions (for auxiliary loss)
            temporal_logits: Temporal stream predictions (for auxiliary loss)
        """
        spatial_feat, spatial_logits = self.spatial_stream(rgb)
        temporal_feat, temporal_logits = self.temporal_stream(flow)
        if self.fusion_type == 'late':
            alpha = torch.sigmoid(self.spatial_weight)
            beta = torch.sigmoid(self.temporal_weight)
            output = alpha * spatial_logits + (1 - alpha) * temporal_logits
        elif self.fusion_type == 'early':
            combined_feat = torch.cat([spatial_feat, temporal_feat], dim=1)
            output = self.fusion_layer(combined_feat)
        return (output, spatial_logits, temporal_logits)

    def get_stream_outputs(self, rgb, flow):
        """Get individual stream outputs (useful for analysis)."""
        _, spatial_logits = self.spatial_stream(rgb)
        _, temporal_logits = self.temporal_stream(flow)
        return (spatial_logits, temporal_logits)

    def freeze_spatial(self):
        """Freeze spatial stream (train only temporal)."""
        for param in self.spatial_stream.parameters():
            param.requires_grad = False

    def unfreeze_spatial(self):
        """Unfreeze spatial stream."""
        for param in self.spatial_stream.parameters():
            param.requires_grad = True

    def freeze_temporal(self):
        """Freeze temporal stream (train only spatial)."""
        for param in self.temporal_stream.parameters():
            param.requires_grad = False

    def unfreeze_temporal(self):
        """Unfreeze temporal stream."""
        for param in self.temporal_stream.parameters():
            param.requires_grad = True

def create_two_stream_model(num_classes=NUM_CLASSES, num_flow_frames=NUM_FLOW_FRAMES, backbone=BACKBONE, fusion_type=FUSION_TYPE, pretrained=PRETRAINED):
    """Factory function to create Two-Stream Network."""
    return TwoStreamNetwork(num_classes=num_classes, num_flow_frames=num_flow_frames, backbone=backbone, fusion_type=fusion_type, pretrained=pretrained)
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_two_stream_model()
    model = model.to(device)
    rgb = torch.randn(2, 3, 224, 224).to(device)
    flow = torch.randn(2, 2 * 10, 224, 224).to(device)
    output, spatial_logits, temporal_logits = model(rgb, flow)
    print(f'RGB input shape: {rgb.shape}')
    print(f'Flow input shape: {flow.shape}')
    print(f'Output shape: {output.shape}')
    print(f'Spatial logits shape: {spatial_logits.shape}')
    print(f'Temporal logits shape: {temporal_logits.shape}')
    print(f'\nModel created successfully!')
