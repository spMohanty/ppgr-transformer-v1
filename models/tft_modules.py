"""
Core Temporal Fusion Transformer (TFT) modules.

Based on the implementation from the ppgr-tft-v0 project, with minor modifications
to ensure compatibility with the current project.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = 0.0):
        """
        Applies a linear layer followed by dropout and then splits the output
        into two halves. which is then gated by a GLU activation.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size,  self.hidden_size * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x

class AddNorm(nn.Module):
    def __init__(
        self, input_size: int, skip_size: int = None, dropout: float = 0.0, trainable_add: bool = True
    ):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        if self.input_size != self.skip_size:
            self.resample = nn.Linear(skip_size, input_size)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)
        return output

# The GateAddNorm module combines a GLU and an AddNorm.
class GateAddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: Optional[int] = None,
        skip_size: Optional[int] = None,
        trainable_add: bool = False,
        dropout: float = 0.0,
    ):
        """
        First applies a GatedLinearUnit, then adds the skip connection
        and normalizes the result.
        """
        super().__init__()
        hidden_size = hidden_size or input_size
        skip_size = skip_size or hidden_size
        self.glu = GatedLinearUnit(input_size, hidden_size, dropout)
        self.add_norm = AddNorm(hidden_size, skip_size, dropout, trainable_add)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.add_norm(self.glu(x), skip)

class PreNormResidualBlock(nn.Module):
    """
    A pre-norm residual block that applies LayerNorm before the two-layer feed-forward
    network. Uses GELU activation and dropout.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, context_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Optional projection for residual connection if input and output dimensions differ
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
        
        self.context_layer = nn.Linear(context_dim, hidden_dim) if context_dim is not None else None
    
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # Apply layer norm before feed-forward block
        x_norm = self.norm1(x)
        hidden = self.fc1(x_norm)
        if context is not None and self.context_layer is not None:
            hidden = hidden + self.context_layer(context)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        out = self.fc2(hidden)
        out = self.dropout(out)
        
        # Apply projection for residual if dimensions don't match
        if self.skip_proj is not None:
            return self.skip_proj(x) + out
        else:
            return x + out  # residual connection


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
        attn = None
        
        for i in range(self.num_layers):
            # Only return attention on the last layer if requested
            if return_attn and i == self.num_layers - 1:
                try:
                    output, layer_attn = self.shared_layer(
                        output,
                        memory,
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        return_attn=True
                    )
                    attn = layer_attn
                except Exception as e:
                    print(f"Warning: Failed to get attention weights: {e}")
                    # Create a fallback attention with proper shape
                    batch_size = tgt.shape[0]
                    seq_len_tgt = tgt.shape[1]
                    seq_len_mem = memory.shape[1]
                    attn = torch.ones(batch_size, seq_len_tgt, seq_len_mem, device=tgt.device) / seq_len_mem
                    
                    # Continue processing without attention
                    output = self.shared_layer(
                        output,
                        memory,
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        return_attn=False
                    )
            else:
                output = self.shared_layer(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
        
        # If we're returning attention but didn't get any, create a fallback
        if return_attn and attn is None:
            print("Warning: No attention weights returned from decoder layers")
            batch_size = tgt.shape[0]
            seq_len_tgt = tgt.shape[1]
            seq_len_mem = memory.shape[1]
            attn = torch.ones(batch_size, seq_len_tgt, seq_len_mem, device=tgt.device) / seq_len_mem
            
        # Return output with or without attention
        if return_attn:
            return output, attn
        return output

class TransformerVariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        n_heads: int,
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, nn.Module] = None,
    ):
        """
        Variable selection network using self-attention.
        
        Args:
            input_sizes: Dictionary mapping variable names to their dimensions
            hidden_size: Size of hidden layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            context_size: Size of optional conditioning context
            single_variable_grns: Optional pre-initialized variable-wise GRNs
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.n_heads = n_heads
        self.dropout = dropout
        
        # Create a single variable grn for each variable (or use provided ones)
        self.single_variable_grns = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if single_variable_grns is not None and name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            else:
                self.single_variable_grns[name] = PreNormResidualBlock(
                    input_dim=input_size,
                    hidden_dim=self.hidden_size,
                    output_dim=self.hidden_size,
                    dropout=self.dropout
                )
        
        # Create the variable selection weights using a GRN with optional context
        self.var_selection_grn = PreNormResidualBlock(
            input_dim=len(self.input_sizes) * self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=len(self.input_sizes),
            context_dim=context_size,
            dropout=self.dropout
        )
        
    def forward(
        self, x: Dict[str, torch.Tensor], context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Variable Selection Network.
        
        Args:
            x: Dictionary of input variables {var_name: tensor of shape [batch, time, input_size]}
            context: Optional context tensor for conditioning
        
        Returns:
            Tuple of (processed variables tensor, variable selection weights)
        """
        # Get batch size and sequence length from first input
        first_var = list(x.values())[0]
        batch_size, seq_len = first_var.shape[:2]
        device = first_var.device
        
        # Process each variable independently
        var_outputs = {}
        for k, v in x.items():
            var_outputs[k] = self.single_variable_grns[k](v)
        
        # Concatenate variables for self-attention
        var_concat = torch.cat([var_outputs[k] for k in self.input_sizes.keys()], dim=-1)
        
        # Compute variable selection weights
        if context is not None:
            # Ensure context has the right shape for sequence data
            if context.ndim == 2:
                # Add sequence dimension if missing
                context = context.unsqueeze(1).expand(-1, seq_len, -1)
                
            # Apply the GRN with context to get variable weights
            weights = self.var_selection_grn(var_concat, context)
        else:
            # Apply the GRN without context
            weights = self.var_selection_grn(var_concat)
        
        # Apply softmax to get variable selection weights
        weights = F.softmax(weights, dim=-1)
        
        # Construct weighted combination of variables
        var_list = [var_outputs[k] for k in self.input_sizes.keys()]
        var_stacked = torch.stack(var_list, dim=-2)  # [batch, time, num_vars, hidden]
        
        # Apply weights to get the final output
        # weights: [batch, time, num_vars]
        # var_stacked: [batch, time, num_vars, hidden]
        # -> Need to expand weights to [batch, time, num_vars, 1]
        weights_expanded = weights.unsqueeze(-1)
        combined = (var_stacked * weights_expanded).sum(dim=-2)  # [batch, time, hidden]
        
        return combined, weights
        
# Custom TransformerDecoderLayer that supports returning attention weights
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
        try:
            tgt2, attn_weights = self.multihead_attn(
                tgt2, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
            
            # Validate the attention weights
            if attn_weights is None or torch.isnan(attn_weights).any() or torch.all(attn_weights == 0):
                # Create uniform attention as fallback
                batch_size = tgt.size(0)
                tgt_len = tgt.size(1)
                src_len = memory.size(1)
                attn_weights = torch.ones(batch_size, tgt_len, src_len, device=tgt.device) / src_len
                
        except Exception as e:
            print(f"Warning: Error computing cross-attention: {e}")
            # Create uniform attention as fallback
            batch_size = tgt.size(0)
            tgt_len = tgt.size(1)
            src_len = memory.size(1)
            attn_weights = torch.ones(batch_size, tgt_len, src_len, device=tgt.device) / src_len
            
            # Apply a simple attention computation as fallback
            tgt2 = torch.matmul(attn_weights, memory)
            
        tgt = tgt + self.dropout2(tgt2)

        # Feed-forward block
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        if return_attn:
            return tgt, attn_weights
        else:
            return tgt
            
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as described in "Attention Is All You Need"
    
    Computes attention weights using scaled dot product between query and key,
    then applies these weights to the values.
    """
    def __init__(self, dropout: float = 0.0, scale: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.softmax = nn.Softmax(dim=-1)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        """
        Computes scaled dot-product attention.
        
        Args:
            q: Query tensor [batch_size, seq_len_q, d_k]
            k: Key tensor [batch_size, seq_len_k, d_k]
            v: Value tensor [batch_size, seq_len_v, d_v] (seq_len_k == seq_len_v)
            mask: Optional mask tensor [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            output: Attention output [batch_size, seq_len_q, d_v]
            attn: Attention weights [batch_size, seq_len_q, seq_len_k]
        """
        # Calculate dot product attention
        attn = torch.bmm(q, k.transpose(-2, -1))  # [batch, seq_len_q, seq_len_k]
        
        # Scale attention scores
        if self.scale:
            d_k = k.size(-1)
            attn = attn / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Convert mask to proper format
            mask_expanded = ~mask if mask.dtype == torch.bool else mask
            attn = attn.masked_fill(mask_expanded == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn = self.softmax(attn)
        
        # Apply dropout if defined
        if self.dropout is not None:
            attn = self.dropout(attn)
        
        # Apply attention weights to values
        output = torch.bmm(attn, v)  # [batch, seq_len_q, d_v]
        
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-head Self-Attention layer.
    
    Based on the implementation from pytorch_forecasting's TFT model.
    Uses separate projection layers for each head and processes them independently
    before combining the results.
    """
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)
        
        # Shared value projection layer
        self.v_layer = nn.Linear(self.d_model, self.d_v)
        
        # Separate projection layers for each head
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(n_head)])
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_v, self.d_model, bias=False)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize linear layers using Xavier uniform initialization
        for name, p in self.named_parameters():
            if "bias" not in name:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
    
    def forward(self, q, k, v, mask=None):
        """
        Forward pass for interpretable multi-head attention.
        
        Args:
            q: Query tensor [batch_size, seq_len_q, d_model]
            k: Key tensor [batch_size, seq_len_k, d_model]
            v: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional mask tensor [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            output: Attention output [batch_size, seq_len_q, d_model]
            attn: Attention weights [batch_size, n_head, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)
        
        # Project value once since it's shared across heads
        vs = self.v_layer(v)
        
        # Process each head separately
        heads = []
        attns = []
        
        for i in range(self.n_head):
            # Project queries and keys for this head
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            
            # Apply attention
            head, attn = self.attention(qs, ks, vs, mask)
            
            # Apply dropout
            head = self.dropout(head)
            
            # Collect outputs
            heads.append(head)
            attns.append(attn)
        
        # Combine heads
        if self.n_head > 1:
            # For multiple heads, average the outputs
            head = torch.stack(heads, dim=2)  # [batch, seq_len_q, n_head, d_v]
            attn = torch.stack(attns, dim=1)  # [batch, n_head, seq_len_q, seq_len_k]
            
            # Take mean across heads dimension
            output = torch.mean(head, dim=2)  # [batch, seq_len_q, d_v]
        else:
            # For a single head, just use the first head's output
            output = heads[0]  # [batch, seq_len_q, d_v]
            attn = attns[0].unsqueeze(1)  # [batch, 1, seq_len_q, seq_len_k]
        
        # Project to output dimension
        output = self.out_proj(output)  # [batch, seq_len_q, d_model]
        
        # Final dropout
        output = self.dropout(output)
        
        return output, attn 