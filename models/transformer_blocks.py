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


class TransformerDecoderLayer(nn.Module):
    """
    A single Transformer decoder layer with self-attention + cross-attention + feed-forward.
    Returns cross-attn weights if return_attn=True.
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
        # Self-attn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Cross-attn
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Activation function
        self.activation_fn = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        tgt, 
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        return_attn=False
    ):
        """
        Forward pass of a transformer decoder layer.
        
        Args:
            tgt: Target sequence (queries)
            memory: Source sequence (keys/values)
            tgt_mask: Mask applied to self-attention
            memory_mask: Mask applied to cross-attention
            tgt_key_padding_mask: Padding mask for target sequence
            memory_key_padding_mask: Padding mask for memory sequence
            return_attn: Whether to return attention weights
            
        Returns:
            x: Output tensor
            cross_attn_weights: Cross-attention weights if return_attn=True
        """
        # 1) Self-Attn
        x = tgt
        sa_out, _ = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout_attn(sa_out)
        x = self.norm1(x)

        # 2) Cross-Attn
        cross_attn_weights = None
        ca_out, ca_weights = self.cross_attn(
            query=x, 
            key=memory, 
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True
        )
        x = x + self.dropout_cross(ca_out)
        x = self.norm2(x)

        if return_attn:
            cross_attn_weights = ca_weights  # shape [B, T_tgt, T_mem]

        # 3) Feed-forward
        ff = self.linear2(self.dropout_ffn(self.activation_fn(self.linear1(x))))
        x = x + ff
        x = self.norm3(x)

        return x, cross_attn_weights


class TransformerDecoder(nn.Module):
    """
    Stacks multiple TransformerDecoderLayer blocks.
    If return_attn=True, returns the final layer's cross-attn weights.
    If layers_share_weights=True, the same layer is used for all positions,
    otherwise independent copies are created for each position.
    """
    def __init__(self, decoder_layer, num_layers, norm=None, layers_share_weights=False):
        super().__init__()
        if layers_share_weights:
            # Use the same layer instance for all positions
            self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        else:
            # Create independent copies of the layer for each position
            self.layers = nn.ModuleList([deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm
        self.num_layers = num_layers

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        return_attn=False
    ):
        """
        Forward pass through the transformer decoder stack.
        
        Args:
            tgt: Target sequence (queries)
            memory: Source sequence (keys/values from encoder)
            tgt_mask: Mask for target sequence (self-attention)
            memory_mask: Mask for memory sequence (cross-attention)
            tgt_key_padding_mask: Padding mask for target sequence
            memory_key_padding_mask: Padding mask for memory sequence
            return_attn: Whether to return attention weights
            
        Returns:
            output: Decoder output
            cross_attn_weights: Attention weights if return_attn=True
        """
        output = tgt
        cross_attn_weights = None

        for layer_idx, layer in enumerate(self.layers):
            is_last = (layer_idx == len(self.layers) - 1)
            output, cross_attn_weights = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                return_attn=(return_attn and is_last)
            )            

        if self.norm is not None:
            output = self.norm(output)

        return output, cross_attn_weights
    
    
class CrossModalFusionBlock(nn.Module):
    """
    Fuses multiple modality embeddings using multi-head self-attention.
    Unlike the original, this version uses a dictionary-based API like TransformerVariableSelectionNetwork.
    
    Two fusion modes are supported:
    1. "mean_pooling": Apply self-attention across modalities then mean-pool (original approach)
    2. "query_token": Use a learnable query token to attend to all modalities (similar to VSN)
    """
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        n_heads: int = 4,
        input_embedding_flags: Dict[str, bool] = None,
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, nn.Module] = None,
        prescalers: Dict[str, nn.Module] = None,
        fusion_mode: str = "query_token"
    ):
        """
        Initialize a cross-modal fusion block with VSN-compatible API.
        
        Args:
            input_sizes: Dictionary mapping variable names to their input dimensions 
                         (currently ignored as we assume all inputs are already of hidden_size).
            hidden_size: The hidden dimension of the model.
            n_heads: Number of attention heads.
            input_embedding_flags: (Ignored, for VSN compatibility).
            dropout: Dropout probability.
            context_size: (Ignored, for VSN compatibility).
            single_variable_grns: (Ignored, for VSN compatibility).
            prescalers: Optional dict mapping variable names to a prescaler module.
            fusion_mode: Fusion approach: "mean_pooling" or "query_token".
        """
        super().__init__()
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.dropout = dropout
        self.fusion_mode = fusion_mode
        self.n_vars = len(input_sizes)
        self.single_variable = (self.n_vars == 1)
        
        # If provided, use the given prescalers
        if prescalers is not None:
            self.prescalers = nn.ModuleDict(prescalers)
        else:
            self.prescalers = nn.ModuleDict()
        
        # Multi-head attention for fusion
        self.attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Add a learnable query token if using query_token mode
        if fusion_mode == "query_token":
            # Create a learnable query token [1, 1, hidden_size] - will be expanded to [B, T, hidden_size]
            self.query_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            
            # Add feed-forward network for processing query token output
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
            )

    def forward(
        self, x: Dict[str, torch.Tensor], context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Dictionary of variable names to tensors.
               Each tensor should be of shape [batch, time, hidden_size].
            context: (Ignored, for VSN compatibility).

        Returns:
            output: The fused variable embedding of shape [batch, time, hidden_size].
            selection_weights: The variable selection weights, of shape [batch, time, n_vars].
        """
        # Handle single variable case
        if self.single_variable:
            var_name = list(self.input_sizes.keys())[0]
            var_tensor = x[var_name]
            
            # Apply prescaler if present
            if var_name in self.prescalers:
                var_tensor = self.prescalers[var_name](var_tensor)
                
            # Create trivial selection weights
            B, T, _ = var_tensor.shape
            weights = torch.ones(B, T, 1, device=var_tensor.device)
            
            return var_tensor, weights
        
        # Get batch and time dimensions from the first variable
        sample_tensor = next(iter(x.values()))
        B, T, D = sample_tensor.shape
        
        # Process each variable and collect embeddings
        embedded_vars = []
        var_names = []
        
        for name in self.input_sizes.keys():
            var_tensor = x[name]
            # Apply prescaler if exists
            if name in self.prescalers:
                var_tensor = self.prescalers[name](var_tensor)
            embedded_vars.append(var_tensor)
            var_names.append(name)
        
        # Stack variables along a new dimension
        tokens = torch.stack(embedded_vars, dim=2)  # [B, T, n_vars, D]
        
        # --- Fusion processing ---
        if self.fusion_mode == "mean_pooling":
            # Original approach: mean pooling after self-attention
            # Reshape for batch processing: [B*T, n_vars, D]
            tokens_reshaped = tokens.reshape(B * T, self.n_vars, D)
            
            # Self-attention across variables
            attn_output, attn_weights = self.attn(
                tokens_reshaped, tokens_reshaped, tokens_reshaped,
                need_weights=True
            )  # [B*T, n_vars, D], [B*T, n_vars, n_vars]
            
            # Mean-pool across variables
            fused = attn_output.mean(dim=1)  # [B*T, D]
            fused = self.norm(fused)
            fused = self.dropout_layer(fused)
            
            # Reshape back to [B, T, D]
            fused = fused.reshape(B, T, D)
            
            # Reshape attention weights to [B, T, n_vars, n_vars]
            # Then average across the first variable dimension to get [B, T, n_vars]
            selection_weights = attn_weights.reshape(B, T, self.n_vars, self.n_vars)
            selection_weights = selection_weights.mean(dim=2)  # [B, T, n_vars]
            
        elif self.fusion_mode == "query_token":
            # Query token approach: similar to VSN
            # Reshape tokens for processing: [B*T, n_vars, D]
            tokens_reshaped = tokens.reshape(B * T, self.n_vars, D)
            
            # Expand query token
            query = self.query_token.expand(B * T, 1, D)  # [B*T, 1, D]
            
            # Apply cross-attention: query attends to variables
            attn_output, attn_weights = self.attn(
                query=query,          # [B*T, 1, D]
                key=tokens_reshaped,  # [B*T, n_vars, D]
                value=tokens_reshaped,  # [B*T, n_vars, D]
                need_weights=True
            )  # [B*T, 1, D], [B*T, 1, n_vars]
            
            # Apply feed-forward with residual connection
            ff_output = self.feed_forward(attn_output)
            fused = self.norm(attn_output + ff_output)  # [B*T, 1, D]
            
            # Remove singleton dimension and reshape
            fused = fused.squeeze(1)  # [B*T, D]
            fused = self.dropout_layer(fused)
            fused = fused.reshape(B, T, D)  # [B, T, D]
            
            # Reshape attention weights to [B, T, n_vars]
            selection_weights = attn_weights.reshape(B, T, self.n_vars)
        
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
        
        return fused, selection_weights

class TransformerVariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        n_heads: int,
        input_embedding_flags: Dict[str, bool] = None,
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, nn.Module] = None,
        prescalers: Dict[str, nn.Module] = None,
    ):
        """
        Transformer-based variable selection network.
        
        Args:
            input_sizes: Dictionary mapping variable names to their input dimensions.
            hidden_size: The hidden dimension (this is used as the transformer model dimension).
            input_embedding_flags: (Ignored in this implementation, but left here for compatibility with the original code.)
            dropout: Dropout probability.
            context_size: If provided, context will be projected and added to a learnable CLS token.
            single_variable_grns: (Ignored in this implementation. - as here All variables are treated as "tokens" in a single attention mechanism)
            prescalers: Optional dict mapping variable names to a prescaler (e.g. an nn.Linear)
                        that should be applied before the main variable embedding.
        """
        super().__init__()
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.dropout = dropout
        self.context_size = context_size
        self.input_embedding_flags = input_embedding_flags if input_embedding_flags is not None else {}
        self.n_vars = len(input_sizes)
        self.single_variable = (self.n_vars == 1)

        # If provided, use the given prescalers; otherwise, initialize an empty ModuleDict.
        if prescalers is not None:
            self.prescalers = nn.ModuleDict(prescalers)
        else:
            self.prescalers = nn.ModuleDict()

        # Build a simple per-variable embedding layer mapping from the variable's input dim to hidden_size.
        self.variable_embeddings = nn.ModuleDict()
        for name, size in input_sizes.items():
            self.variable_embeddings[name] = nn.Linear(size, hidden_size)

        # If context is provided, project it to hidden_size.
        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size)
        else:
            self.context_proj = None
        
        # Learned CLS token used as the query for variable selection.
        # Shape: [1, 1, hidden_size] — later expanded to match the batch (and time) dimensions.
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Multi-head attention layer.
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=self.n_heads, dropout=dropout, batch_first=True
        )

        # An optional feed-forward network and layer norm following the attention layer.
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, x: Dict[str, torch.Tensor], context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Dictionary of variable names to tensors.
               Each tensor can be of shape [batch, input_dim] or [batch, time, input_dim].
            context: Optional context tensor of shape [batch, context_size] or [batch, time, context_size].

        Returns:
            output: The combined variable embedding, of shape [batch, hidden_size] (if no time dimension)
                    or [batch, time, hidden_size] (if inputs are time-varying).
            sparse_weights: The variable selection weights (attention weights),
                            of shape [batch, n_vars] or [batch, time, n_vars].
        """
        # Determine whether inputs are time-varying based on one sample.
        sample_tensor = next(iter(x.values()))
        has_time = (sample_tensor.dim() == 3)

        # --- Case 1: Single variable input ---
        if self.single_variable:
            var_name = list(self.input_sizes.keys())[0]
            var_tensor = x[var_name]
            if var_tensor.dim() == 2:
                var_tensor = var_tensor.unsqueeze(1)  # Shape: [B, 1, input_dim]
            if var_name in self.prescalers:
                var_tensor = self.prescalers[var_name](var_tensor)
            # Embed the variable.
            output = self.variable_embeddings[var_name](var_tensor)  # [B, T, hidden_size]
            B, T, _ = output.shape
            # Create trivial sparse weights (all ones).
            sparse_weights = torch.ones(B, T, 1, device=output.device)
            # If time dimension is 1, squeeze it.
            if T == 1:
                output = output.squeeze(1)
                sparse_weights = sparse_weights.squeeze(1)
            return output, sparse_weights

        # --- Case 2: Multiple variable inputs ---
        # Process each variable: apply prescaler (if provided) then the embedding.
        embedded_vars = []
        for name, size in self.input_sizes.items():
            var_tensor = x[name]
            if var_tensor.dim() == 2:
                var_tensor = var_tensor.unsqueeze(1)  # [B, 1, input_dim]
            if name in self.prescalers:
                var_tensor = self.prescalers[name](var_tensor)
            embedded = self.variable_embeddings[name](var_tensor)  # [B, T, hidden_size]
            embedded_vars.append(embedded)

        # Stack along a new variable dimension so that tokens shape becomes [B, T, n_vars, hidden_size].
        tokens = torch.stack(embedded_vars, dim=2)
        B, T, n_vars, H = tokens.shape
        # Merge batch and time dimensions for transformer processing: [B*T, n_vars, H].
        tokens_reshaped = tokens.view(B * T, n_vars, H)

        # Create a CLS token for each instance.
        cls_tokens = self.cls_token.expand(B * T, -1, -1)  # [B*T, 1, H]
        # If a context tensor is provided, project and add it to the CLS token.
        if context is not None:
            if context.dim() == 3:
                # [B, T, context_size] -> [B*T, 1, context_size]
                context = context.reshape(B * T, 1, -1)
            elif context.dim() == 2:
                # [B, context_size] -> [B, 1, context_size] then repeat for each time step.
                context = context.unsqueeze(1).expand(B, T, -1).reshape(B * T, 1, -1)
            if self.context_proj is not None:
                context = self.context_proj(context)
            cls_tokens = cls_tokens + context

        # Concatenate the CLS token with the variable tokens.
        # attn_input: [B*T, 1+n_vars, H]
        attn_input = torch.cat([cls_tokens, tokens_reshaped], dim=1)

        # Use the CLS token as the query, and the full sequence as key/value.
        query = attn_input[:, :1, :]  # [B*T, 1, H]
        key = attn_input             # [B*T, 1+n_vars, H]
        value = attn_input

        # Apply multi-head attention.
        attn_output, attn_weights = self.mha(query, key, value)
        # attn_weights shape: [B*T, 1, 1+n_vars]. Discard the first column (self-attention of CLS).
        variable_selection_weights = attn_weights[:, :, 1:]  # [B*T, 1, n_vars]

        # pass the output through a feed-forward network and add a residual connection.
        ff = self.feed_forward(attn_output)
        output = self.layer_norm(attn_output + ff)  # [B*T, 1, H]

        # Reshape back to [B, T, H].
        output = output.view(B, T, H)
        variable_selection_weights = variable_selection_weights.view(B, T, n_vars)

        # If time dimension is 1, squeeze it.
        if T == 1:
            output = output.squeeze(1)
            variable_selection_weights = variable_selection_weights.squeeze(1)

        return output, variable_selection_weights


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