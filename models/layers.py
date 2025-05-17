"""
Various neural network layers used in the models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit implementation.
    """
    def __init__(self, input_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(input_size, input_size * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return F.glu(x, dim=-1)
        

class AddNorm(nn.Module):
    """
    Add and Normalize module.
    """
    def __init__(self, input_size: int, trainable_add: bool = False):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add
        
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)
    
    def forward(self, x, skip):
        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0
        return self.norm(x + skip)


class GateAddNorm(nn.Module):
    """
    Gated Add & Norm module.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.0, trainable_add: bool = False):
        super().__init__()
        self.dropout = dropout
        self.glu = GatedLinearUnit(hidden_size, dropout=dropout)
        self.add_norm = AddNorm(hidden_size, trainable_add=trainable_add)
        
    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class StaticContextEnrichment(nn.Module):
    """
    Static context enrichment layer that combines the input with a context vector.
    Similar to the implementation in TFT.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, hidden_size]
        # context: [batch_size, seq_len, hidden_size]
        x_norm = self.norm(x)
        # Concatenate along the feature dimension
        combined = torch.cat([x_norm, context], dim=-1)
        enriched = self.ff(combined)
        # Add residual connection
        return x + enriched 