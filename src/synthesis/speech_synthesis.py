import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class SpeechSynthesizer(nn.Module):
    """Converts temporal features to mel-spectrograms."""
    
    def __init__(self, input_dim: int = 512, n_mels: int = 80):
        """Initialize speech synthesizer.
        
        Args:
            input_dim: Dimension of input temporal features
            n_mels: Number of mel frequency bins
        """
        super().__init__()
        
        self.prenet = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Simplified attention-based decoder
        self.attention = nn.Linear(256 + 256, 1)
        self.rnn = nn.GRU(256, 256, batch_first=True)
        
        self.mel_proj = nn.Linear(256, n_mels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert temporal features to mel-spectrogram.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Predicted mel-spectrogram of shape (batch_size, seq_len, n_mels)
        """
        batch_size, seq_len, _ = x.size()
        
        # Prenet
        x = self.prenet(x)
        
        # Initialize decoder state
        decoder_input = torch.zeros(batch_size, 1, 256, device=x.device)
        hidden = torch.zeros(1, batch_size, 256, device=x.device)
        
        # Simplified attention mechanism
        outputs = []
        for i in range(seq_len):
            # Compute attention weights
            energy = self.attention(torch.cat([
                hidden.transpose(0, 1).expand(-1, seq_len, -1),
                x
            ], dim=-1)).squeeze(-1)
            attention_weights = torch.softmax(energy, dim=1)
            
            # Context vector
            context = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
            
            # RNN step
            decoder_output, hidden = self.rnn(decoder_input, hidden)
            
            # Project to mel-spectrogram
            mel_output = self.mel_proj(decoder_output.squeeze(1))
            outputs.append(mel_output)
            
            # Next input (teacher forcing would be used in training)
            decoder_input = decoder_output
            
        return torch.stack(outputs, dim=1)