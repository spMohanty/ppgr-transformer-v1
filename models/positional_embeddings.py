"""
Positional embedding implementations for sequence models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) as proposed in https://arxiv.org/abs/2104.09864.
    
    This implementation supports offset indices to handle negative positions.
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        offset: int = 0
    ) -> None:
        """
        Initialize rotary positional embeddings.
        
        Args:
            dim: Hidden dimension size. Must be divisible by 2.
            max_seq_len: Maximum sequence length the model is expected to handle.
                         This determines the size of the position cache.
            base: Base value for the frequency calculations.
            offset: Position offset to handle negative indices (centering).
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.offset = offset
        
        assert dim % 2 == 0, "RoPE dim must be divisible by 2"
        self.rope_init()

    def rope_init(self):
        # Precompute frequency bands
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes from 0 to max_seq_len-1
        seq_idx = torch.arange(
            max_seq_len, dtype=torch.float, device=self.theta.device
        )

        # Outer product of theta and position index
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta)
        
        # Cache includes both the cos and sin components
        # Shape: [max_seq_len, dim//2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            positions: Position indices of shape [batch_size, seq_len]
            
        Returns:
            Tensor with rotary embeddings applied
        """
        # Ensure positions include offset (for negative positions handling)
        positions = positions + self.offset
        
        # Ensure position indices are within bounds
        positions = torch.clamp(positions, 0, self.max_seq_len - 1)
        
        # Get batch size and sequence length
        batch_size, seq_len = positions.shape
        
        # Get cached embeddings for the given positions
        # Shape after indexing: [batch_size, seq_len, dim//2, 2]
        rope_cache = self.cache[positions]
        
        # Reshape input for rotation
        # From [batch_size, seq_len, dim] to [batch_size, seq_len, dim//2, 2]
        x_reshaped = x.float().reshape(batch_size, seq_len, -1, 2)
        
        # Apply rotary transformation using the cached values
        # cos(θ)x - sin(θ)y and sin(θ)x + cos(θ)y
        x_out = torch.stack(
            [
                x_reshaped[..., 0] * rope_cache[..., 0] - 
                x_reshaped[..., 1] * rope_cache[..., 1],
                
                x_reshaped[..., 1] * rope_cache[..., 0] + 
                x_reshaped[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        
        # Reshape back to original shape - ensure this matches the input shape
        x_out = x_out.reshape(batch_size, seq_len, self.dim)
        
        # Return with the same dtype as the input
        return x_out.type_as(x) 