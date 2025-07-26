import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class Vocoder(nn.Module):
    """Converts mel-spectrograms to raw audio waveforms."""
    
    def __init__(self, n_mels: int = 80):
        """Initialize vocoder.
        
        Args:
            n_mels: Number of mel frequency bins in input spectrograms
        """
        super().__init__()
        
        # Simplified WaveNet-like architecture
        self.conv1 = nn.Conv1d(n_mels, 512, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, dilation=2**i, padding=2**i),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=1),
                nn.ReLU()
            ) for i in range(4)
        ])
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(256, 1, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert mel-spectrogram to waveform.
        
        Args:
            x: Input mel-spectrogram of shape (batch_size, n_mels, seq_len)
            
        Returns:
            Waveform of shape (batch_size, 1, seq_len)
        """
        x = self.conv1(x)
        
        # Residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
            
        x = self.conv2(x)
        x = self.conv3(x)
        return self.tanh(x)