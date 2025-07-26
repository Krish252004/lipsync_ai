import torch
import torch.nn as nn
from typing import Tuple

class TemporalEncoder(nn.Module):
    """Temporal encoder for sequence of visual features."""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, num_layers: int = 2):
        """Initialize temporal encoder.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden states
            num_layers: Number of LSTM layers
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Projection layer to get back to hidden_dim (removes bidirectional)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process sequence of visual features.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of:
            - Output features of shape (batch_size, seq_len, hidden_dim)
            - Tuple of final hidden and cell states
        """
        outputs, (hidden, cell) = self.lstm(x)
        outputs = self.proj(outputs)  # Combine bidirectional features
        return outputs, (hidden, cell)