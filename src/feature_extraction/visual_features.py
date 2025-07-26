import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

class VisualFeatureExtractor(nn.Module):
    """CNN-based visual feature extractor for lip regions."""
    
    def __init__(self, feature_dim: int = 256):
        """Initialize feature extractor.
        
        Args:
            feature_dim: Dimension of output features
        """
        super().__init__()
        
        # Use a pretrained ResNet18 as backbone
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add a custom head for our feature dimension
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from lip region images.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Extracted features of shape (batch_size, feature_dim)
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        return self.head(features)