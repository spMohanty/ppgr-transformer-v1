"""
Transformer building blocks for the glucose forecasting model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import copy
from typing import Dict, Tuple, Optional

class TransformerEncoderLayer(nn.Module):
    """
    A single Transformer encoder layer with self-attention + feed-forward.
    Returns (output, attn_weights) when need_weights=True.
    """
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        activation: str = "relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation function
        self.activation_fn = F.relu if activation == "relu" else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights=False):
        # 1) Self-Attention
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True  # Always compute weights, then use as needed
        )            
        
        # Dropout plus residual
        src = src + self.dropout_attn(attn_output)
        src = self.norm1(src)

        # 2) Feed-forward
        ff = self.linear2(self.dropout_ffn(self.activation_fn(self.linear1(src))))
        src = src + ff
        src = self.norm2(src)

        return src, attn_weights


def _get_clones(module, N, share_weights=False):
    """
    Create N identical layers or reuse the same one if share_weights=True
    """
    if share_weights:
        return nn.ModuleList([module for _ in range(N)])
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    """
    Stacks multiple TransformerEncoderLayer blocks. 
    If need_weights=True, returns the final layer's attn_weights.
    If layers_share_weights=True, the same layer is used for all positions,
    otherwise independent copies are created for each position.
    """
    def __init__(self, encoder_layer, num_layers, norm=None, layers_share_weights=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers, layers_share_weights)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, need_weights=False):
        output = src
        attn_weights = None
        
        for layer in self.layers:
            # If need_weights, get them from the last layer only
            output, attn_weights_layer = layer(
                output, 
                src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask, 
                need_weights=(need_weights and layer is self.layers[-1])
            )
            if need_weights and layer is self.layers[-1]:
                attn_weights = attn_weights_layer
        
        if self.norm is not None:
            output = self.norm(output)
            
        return output, attn_weights


class CustomTransformerDecoderLayer(nn.Module):
    """
    Custom TransformerDecoderLayer implementation that allows returning attention weights.
    Based on PyTorch's TransformerDecoderLayer but modified to return attention weights.
    """
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        activation: str = "relu",
        batch_first: bool = True
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Activation
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ):
        """
        Forward pass with option to return attention weights.
        
        Args:
            tgt: Target sequence
            memory: Source sequence from encoder
            tgt_mask: Target sequence mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            return_attn: Whether to return attention weights
            
        Returns:
            output: Decoder layer output
            attn: Attention weights if return_attn=True
        """
        # Self-attention block
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            tgt2, tgt2, tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        # Cross-attention block
        tgt2 = self.norm2(tgt)
        tgt2, attn_weights = self.multihead_attn(
            tgt2, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)

        # Feed-forward block
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        if return_attn:
            return tgt, attn_weights
        else:
            return tgt


class TransformerDecoderLayer(CustomTransformerDecoderLayer):
    """
    A single Transformer decoder layer with self-attention + cross-attention + feed-forward.
    Returns cross-attn weights if return_attn=True.
    
    Inherits from CustomTransformerDecoderLayer for compatibility with existing code.
    """
    pass


class TransformerDecoder(nn.Module):
    """
    A stack of transformer decoder layers with support for returning attention weights.
    Optionally shares weights across layers for parameter efficiency.
    """
    def __init__(
        self, 
        decoder_layer: nn.Module, 
        num_layers: int, 
        layers_share_weights: bool = True,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers_share_weights = layers_share_weights
        
        if layers_share_weights:
            # Use shared decoder with weight sharing
            self.decoder = SharedTransformerDecoder(decoder_layer, num_layers)
        else:
            # Use decoder with separate weights for each layer
            self.layers = nn.ModuleList([
                TransformerDecoderLayer(
                    d_model=decoder_layer.self_attn.embed_dim,
                    nhead=decoder_layer.self_attn.num_heads,
                    dim_feedforward=decoder_layer.linear1.out_features,
                    dropout=decoder_layer.dropout1.p,
                    activation="relu" if isinstance(decoder_layer.activation, type(F.relu)) else "gelu"
                )
                for _ in range(num_layers)
            ])
        
        self.norm = norm
        
    def forward(
        self, 
        tgt: torch.Tensor, 
        memory: torch.Tensor, 
        tgt_mask: Optional[torch.Tensor] = None, 
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None, 
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """
        Forward pass through the decoder stack.
        
        Args:
            tgt: Target sequence (decoder input)
            memory: Source sequence from encoder
            tgt_mask: Mask for target sequence (usually causal)
            memory_mask: Mask for memory sequence
            tgt_key_padding_mask: Padding mask for target sequence
            memory_key_padding_mask: Padding mask for memory sequence
            return_attn: Whether to return attention weights
            
        Returns:
            output: Decoder output
            attn: Attention weights from the last layer if return_attn=True
        """
        output = tgt
        cross_attn_weights = None
        
        if self.layers_share_weights:
            # Use shared decoder
            if return_attn:
                output, cross_attn_weights = self.decoder(
                    output, memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attn=True
                )
            else:
                output = self.decoder(
                    output, memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
        else:
            # Process through separate layers
            for i, layer in enumerate(self.layers):
                if return_attn and i == len(self.layers) - 1:
                    # Only return attention from the last layer
                    output, cross_attn_weights = layer(
                        output, memory,
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        return_attn=True
                    )
                else:
                    output, _ = layer(
                        output, memory,
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        return_attn=False
                    )
        
        if self.norm is not None:
            output = self.norm(output)
            
        if return_attn:
            return output, cross_attn_weights
        else:
            return output


class CrossModalFusionBlock(nn.Module):
    """
    Cross-modal fusion using multihead self-attention.
    
    This block fuses information from multiple modalities using a multihead
    self-attention mechanism followed by a feed-forward network.
    """
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.1,
        fusion_mode: str = "query_token"
    ):
        super().__init__()
        
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.dropout = dropout
        self.fusion_mode = fusion_mode
        self.mode_token = None
        
        # Projections for each modality
        self.projections = nn.ModuleDict()
        for modality, input_size in input_sizes.items():
            # Project to hidden size if input size differs
            if input_size != hidden_size:
                self.projections[modality] = nn.Linear(input_size, hidden_size)
        
        # Multi-head self-attention for cross-modal fusion
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=n_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Normalization and feed-forward
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # For query token mode
        if fusion_mode == "query_token":
            # Create a learnable query token for fusion
            self.query_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Special initialization for the query token
        if hasattr(self, 'query_token'):
            nn.init.xavier_normal_(self.query_token)
    
    def forward(self, inputs: Dict[str, torch.Tensor], static_context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CrossModalFusionBlock.
        
        Args:
            inputs: Dictionary mapping modality names to tensors [B, T, H]
            static_context: Optional static context tensor [B, T, H]
            
        Returns:
            fused: Fused representation [B, T, H]
            weights: Attention weights [B, T, M]
        """
        batch_size = next(iter(inputs.values())).size(0)
        seq_len = next(iter(inputs.values())).size(1)
        
        # Project each modality to hidden size if needed
        modality_list = []
        modality_names = []
        
        for modality, tensor in inputs.items():
            if modality in self.projections:
                # Project if input size differs from hidden size
                projected = self.projections[modality](tensor)
            else:
                projected = tensor
            
            modality_list.append(projected)
            modality_names.append(modality)
        
        # Track weights for each modality
        weights = torch.zeros(batch_size, seq_len, len(modality_list), device=modality_list[0].device)
        
        # Combine all modalities into a single sequence for attention
        if self.fusion_mode == "mean_pooling":
            # Stack all modalities [B, T, M, H] where M is the number of modalities
            stacked = torch.stack(modality_list, dim=2)
            
            # Apply self-attention across modalities
            for t in range(seq_len):
                # Get the t-th timestep for all modalities [B, M, H]
                time_slice = stacked[:, t]
                
                # Condition with static context if provided
                if static_context is not None:
                    # Add static context for this timestep
                    time_slice = time_slice + static_context[:, t].unsqueeze(1)
                
                # Apply self-attention
                attn_out, attn_weights = self.self_attn(
                    time_slice, time_slice, time_slice
                )
                
                # Update weights for this timestep
                weights[:, t] = attn_weights
                
                # Mean pooling across modalities (dimension 1)
                time_out = attn_out.mean(dim=1)
                
                # Update the original tensor
                if t == 0:
                    fused = time_out.unsqueeze(1)
                else:
                    fused = torch.cat([fused, time_out.unsqueeze(1)], dim=1)
        
        elif self.fusion_mode == "query_token":
            # Similar to VSN approach using a query token
            # Stack all modalities [B, T, M, H]
            stacked = torch.stack(modality_list, dim=2)
            
            # Initialize output tensor
            fused = torch.zeros(batch_size, seq_len, self.hidden_size, device=stacked.device)
            
            # Process each timestep
            for t in range(seq_len):
                # Get modalities at this timestep [B, M, H]
                time_slice = stacked[:, t]
                
                # Expand query token to batch size
                query = self.query_token.expand(batch_size, -1, -1)
                
                # Apply static context if provided
                if static_context is not None:
                    # Add static context to the query
                    query = query + static_context[:, t].unsqueeze(1)
                
                # Apply attention with query token as query
                # and modalities as keys/values
                attn_out, attn_weights = self.self_attn(
                    query=query,
                    key=time_slice,
                    value=time_slice
                )
                
                # Store the attention weights
                weights[:, t] = attn_weights.squeeze(1)
                
                # Store the output for this timestep
                fused[:, t] = attn_out.squeeze(1)
        
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
        
        # Apply layer norm
        fused = self.norm1(fused)
        
        # Apply feed-forward network with residual connection
        ff_output = self.ff(fused)
        fused = self.norm2(fused + ff_output)
        
        return fused, weights

class TransformerVariableSelectionNetwork(nn.Module):
    """
    Transformer-based Variable Selection Network.
    
    Implements a variable selection mechanism using a transformer architecture.
    This allows for modeling dynamic relationships between variables.
    """
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.num_variables = len(input_sizes)
        
        # Ensure all inputs have the same hidden size after projection
        self.projections = nn.ModuleDict()
        for var_name, input_size in input_sizes.items():
            if input_size != hidden_size:
                self.projections[var_name] = nn.Linear(input_size, hidden_size)
        
        # Create an embedding for the CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network for processing attention output
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )
        
        # Final projection layer to get attention weights
        self.attn_weights_layer = nn.Linear(hidden_size, self.num_variables)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'cls_token' in name:
                nn.init.xavier_normal_(param)
    
    def forward(self, inputs: Dict[str, torch.Tensor], static_context: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TransformerVariableSelectionNetwork.
        
        Args:
            inputs: Dictionary mapping variable names to tensors [B, T, H]
            static_context: Optional static context tensor [B, T, H]
            
        Returns:
            vs_output: Weighted sum of variables [B, T, H]
            weights: Selection weights for each variable [B, T, num_variables]
        """
        batch_size = next(iter(inputs.values())).size(0)
        seq_len = next(iter(inputs.values())).size(1)
        device = next(iter(inputs.values())).device
        
        # Project variables to hidden_size if needed
        var_list = []
        var_names = []
        
        for var_name, var_tensor in inputs.items():
            if var_name in self.projections:
                projected = self.projections[var_name](var_tensor)
            else:
                projected = var_tensor
            
            var_list.append(projected)
            var_names.append(var_name)
        
        # Initialize the output tensor
        vs_output = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
        
        # Initialize attention weights tensor
        weights = torch.zeros(batch_size, seq_len, self.num_variables, device=device)
        
        # Process each timestep separately to maintain variable dependencies
        for t in range(seq_len):
            # Get the variables for this timestep
            time_vars = [var[:, t].unsqueeze(1) for var in var_list]  # [B, 1, H]
            
            # Concatenate the CLS token with the variables
            # CLS token serves as the query for attention
            cls = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, H]
            
            # Condition with static context if provided
            if static_context is not None:
                # Add static context to the CLS token
                cls = cls + static_context[:, t].unsqueeze(1)
            
            # Concatenate all variables for this timestep [B, num_vars, H]
            time_vars_cat = torch.cat(time_vars, dim=1)
            
            # Apply self-attention using CLS token as query and variables as keys/values
            attn_output, _ = self.self_attn(
                query=cls,
                key=time_vars_cat,
                value=time_vars_cat
            )
            
            # Process through feed-forward network
            processed = self.ff(attn_output).squeeze(1)  # [B, H]
            
            # Generate attention weights for variables
            time_weights = self.attn_weights_layer(processed)  # [B, num_vars]
            time_weights = F.softmax(time_weights, dim=-1)  # [B, num_vars]
            
            # Store the weights for this timestep
            weights[:, t] = time_weights
            
            # Weighted sum of variables for this timestep
            for i, var in enumerate(var_list):
                vs_output[:, t] += var[:, t] * time_weights[:, i].unsqueeze(-1)
        
        return vs_output, weights


class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) as proposed in https://arxiv.org/abs/2104.09864.
    
    This implementation supports offset indices to handle negative positions,
    similar to TimeEmbedding.
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        offset: int = 0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.offset = offset
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

class SharedTransformerEncoder(nn.Module):
    def __init__(self, layer: nn.TransformerEncoderLayer, num_layers: int):
        """
        A transformer encoder that applies the same layer (weight sharing) repeatedly.
        
        Args:
            layer (nn.TransformerEncoderLayer): The transformer encoder layer to be shared.
            num_layers (int): How many times to apply the layer.
        """
        super().__init__()
        self.shared_layer = layer
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for _ in range(self.num_layers):
            output = self.shared_layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        return output


class SharedTransformerDecoder(nn.Module):
    def __init__(self, layer: nn.TransformerDecoderLayer, num_layers: int):
        """
        A transformer decoder that applies the same layer (weight sharing) repeatedly.
        
        Args:
            layer (nn.TransformerDecoderLayer): The transformer decoder layer to be shared.
            num_layers (int): How many times to apply the layer.
        """
        super().__init__()
        self.shared_layer = layer
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, 
                memory_mask=None, tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, return_attn=False):
        """
        Forward pass through the shared decoder layer.
        
        Args:
            tgt: Target sequence
            memory: Source sequence from encoder
            tgt_mask: Target sequence mask
            memory_mask: Memory mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            return_attn: Whether to return attention weights
            
        Returns:
            output: Decoder output
            attn: Attention weights if return_attn=True
        """
        output = tgt
        for i in range(self.num_layers):
            # Only return attention on the last layer if requested
            if return_attn and i == self.num_layers - 1:
                output, attn = self.shared_layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attn=True
                )
                return output, attn
            else:
                output = self.shared_layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
        
        # If we didn't return earlier, return output without attention
        return output