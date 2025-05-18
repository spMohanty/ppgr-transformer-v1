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
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super(GatedLinearUnit, self).__init__()

        if hidden_size is None:
            hidden_size = input_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout

    def forward(self, x):
        if torch.isnan(x).any():
            # Replace NaN values with zeros
            x = torch.nan_to_num(x, nan=0.0)
            
        # Calculate gating mechanism
        sig = self.sigmoid(self.fc1(x))
        
        # Check for and handle NaN values in sigmoid output
        if torch.isnan(sig).any():
            sig = torch.nan_to_num(sig, nan=0.5)  # Use 0.5 as neutral value for sigmoid
            
        # Apply linear transformation
        x_tilde = self.fc2(x)
        
        # Check for and handle NaN values in linear output
        if torch.isnan(x_tilde).any():
            x_tilde = torch.nan_to_num(x_tilde, nan=0.0)
            
        # Apply gating mechanism
        output = sig * x_tilde
        
        # Final NaN check
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
            
        if self.dropout is not None:
            output = self.dropout(output)
        return output


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        input_embedding_flags: Dict[str, bool] = {},
        dropout: float = None,
        context_size: int = None,
    ):
        """
        Variable selection network.
        
        Args:
            input_sizes: Dict with input sizes for each input variable
            hidden_size: Size of hidden layers
            input_embedding_flags: Dict indicating if variables need an embedding
            dropout: Dropout rate
            context_size: Size of context vector
        """
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout

        if len(self.input_sizes) > 1:
            self.flattened_grn = GatedResidualNetwork(
                input_size=sum(self.input_sizes.values()),
                hidden_size=hidden_size,
                output_size=len(self.input_sizes),
                dropout=dropout,
                context_size=context_size,
                residual=False,
            )

        self.single_variable_grns = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = TimeDistributedEmbeddingBag(
                    input_size, hidden_size
                )
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                )

        # Set for holding cached variable encodings
        self.cached_var_encodings = {}
        
    def forward(self, x: Dict[str, torch.Tensor], context: Optional[torch.Tensor] = None):
        """
        Forward pass of the variable selection network.
        
        Args:
            x: Dict with input variables
            context: Optional context vector
            
        Returns:
            Tuple of processed variable tensor and attention weights
        """
        # Check that input sizes are correct
        for name, tensor in x.items():
            if name not in self.input_sizes:
                raise ValueError(f"Input variable {name} not found in input_sizes")
        
        # Handle NaN values in inputs by replacing with zeros
        for name, tensor in x.items():
            if torch.isnan(tensor).any():
                x[name] = torch.nan_to_num(tensor, nan=0.0)
                
        if len(self.input_sizes) > 1:
            # Concatenate inputs
            var_encodings = []
            weight_inputs = []

            # Apply transformations
            for name, input_network in self.single_variable_grns.items():
                if name in x:
                    var_encodings.append(input_network(x[name]))
                    weight_inputs.append(x[name])
                else:
                    # Handle missing variables with zeros
                    batch_size = next(iter(x.values())).size(0)
                    time_steps = next(iter(x.values())).size(1) if len(next(iter(x.values())).size()) > 1 else 1
                    zero_tensor = torch.zeros((batch_size, time_steps, self.input_sizes[name]), 
                                            device=next(iter(x.values())).device)
                    var_encodings.append(input_network(zero_tensor))
                    weight_inputs.append(zero_tensor)

            # Check for NaN values in var_encodings and fix them
            for i, encoding in enumerate(var_encodings):
                if torch.isnan(encoding).any():
                    print(f"WARNING: NaN values detected in variable encoding {i}")
                    var_encodings[i] = torch.nan_to_num(encoding, nan=0.0)

            # Cache variable encodings (useful for interpretation)
            self.cached_var_encodings = {name: var_encodings[i] for i, name in enumerate(self.single_variable_grns.keys())}

            # Concatenate weight inputs
            if len(weight_inputs[0].shape) == 3:
                # For temporal inputs
                flatten = torch.cat(weight_inputs, dim=2)
            else:
                # For static inputs
                flatten = torch.cat(weight_inputs, dim=1)

            # Check for NaN values in flattened input
            if torch.isnan(flatten).any():
                flatten = torch.nan_to_num(flatten, nan=0.0)

            # Calculate variable weights
            sparse_weights = self.flattened_grn(flatten, context)
            
            # Check for NaN values in weights
            if torch.isnan(sparse_weights).any():
                print(f"WARNING: NaN values detected in variable selection weights")
                sparse_weights = torch.nan_to_num(sparse_weights, nan=1.0 / len(self.input_sizes))

            # Ensure weights are valid probabilities
            sparse_weights = F.softmax(sparse_weights, dim=-1)

            # Stack variable encodings
            if len(var_encodings[0].shape) == 3:
                # For temporal inputs
                var_encodings = torch.stack(var_encodings, dim=-1)
                # Apply variable selection weights
                outputs = var_encodings * sparse_weights.unsqueeze(2)
                outputs = outputs.sum(dim=-1)
            else:
                # For static inputs
                var_encodings = torch.stack(var_encodings, dim=1)
                # Apply variable selection weights
                outputs = var_encodings * sparse_weights.unsqueeze(2)
                outputs = outputs.sum(dim=1)

            # Final NaN check
            if torch.isnan(outputs).any():
                print(f"WARNING: NaN values detected in variable selection outputs")
                outputs = torch.nan_to_num(outputs, nan=0.0)
                
            return outputs, sparse_weights
        else:
            # For single variable, just apply the transformation
            name = next(iter(self.single_variable_grns.keys()))
            var_encoding = self.single_variable_grns[name](x[name])
            
            # Handle NaN values
            if torch.isnan(var_encoding).any():
                var_encoding = torch.nan_to_num(var_encoding, nan=0.0)
                
            # Create dummy weights
            if len(var_encoding.shape) == 3:
                batch_size, time_steps, _ = var_encoding.shape
                sparse_weights = torch.ones((batch_size, time_steps, 1), device=var_encoding.device)
            else:
                batch_size, _ = var_encoding.shape
                sparse_weights = torch.ones((batch_size, 1), device=var_encoding.device)
                
            return var_encoding, sparse_weights

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
            single_variable_grns: (Ignored in this implementation. - as here All variables are treated as “tokens” in a single attention mechanism)
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

        # Build a simple per-variable embedding layer mapping from the variable’s input dim to hidden_size.
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
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.as_tensor(
                k.size(-1), dtype=attn.dtype, device=attn.device
            ).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)]
        )
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn
    
class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = None,
        context_size: Optional[int] = None,
        residual: bool = True,
    ):
        """
        Gated Residual Network as described in the TFT paper.
        
        Args:
            input_size: Input size
            hidden_size: Hidden layer size
            output_size: Output size
            dropout: Dropout rate
            context_size: External context size
            residual: Whether to use residual connection
        """
        super(GatedResidualNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.residual = residual

        # Setup layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        
        # Optional context layer for conditioning
        if context_size is not None:
            self.context_layer = nn.Linear(context_size, hidden_size, bias=False)
            
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        
        # Skip connection mapping if input and output have different sizes
        if self.residual and (input_size != output_size):
            self.skip_layer = nn.Linear(input_size, output_size)
        
        # GLU for gates
        self.gate = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
        # Layer normalization for output
        self.norm = nn.LayerNorm(output_size)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for GRN.
        
        Args:
            x: Input tensor
            context: Optional context tensor for conditioning
            
        Returns:
            Processed tensor
        """
        # Handle NaN values in input
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
            
        # Get residual connection ready
        if self.residual:
            residual = x
            if hasattr(self, "skip_layer"):
                residual = self.skip_layer(x)
        
        # Main network
        output = self.fc1(x)
        
        # Check for NaN values after first layer
        if torch.isnan(output).any():
            print(f"WARNING: NaN values detected in GRN after fc1")
            output = torch.nan_to_num(output, nan=0.0)
        
        # Add context if provided
        if self.context_size is not None and context is not None:
            # Handle NaN values in context
            if torch.isnan(context).any():
                context = torch.nan_to_num(context, nan=0.0)
                
            context_output = self.context_layer(context)
            
            # Check for NaN values after context layer
            if torch.isnan(context_output).any():
                print(f"WARNING: NaN values detected in GRN after context_layer")
                context_output = torch.nan_to_num(context_output, nan=0.0)
                
            output = output + context_output
        
        # Apply activation and second layer
        output = self.elu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        
        # Check for NaN values after second layer
        if torch.isnan(output).any():
            print(f"WARNING: NaN values detected in GRN after fc2")
            output = torch.nan_to_num(output, nan=0.0)
        
        # Gating mechanism
        gate = self.sigmoid(self.gate(x))
        
        # Check for NaN values in gate
        if torch.isnan(gate).any():
            print(f"WARNING: NaN values detected in GRN gate")
            gate = torch.nan_to_num(gate, nan=0.5)  # Neutral value for sigmoid
            
        # Apply gate
        output = gate * output
        
        # Add residual if needed
        if self.residual:
            output = output + residual
            
        # Check for NaN values after residual
        if torch.isnan(output).any():
            print(f"WARNING: NaN values detected in GRN after residual")
            output = torch.nan_to_num(output, nan=0.0)
        
        # Apply normalization
        output = self.norm(output)
        
        # Final NaN check
        if torch.isnan(output).any():
            print(f"WARNING: NaN values detected in GRN output after norm")
            output = torch.nan_to_num(output, nan=0.0)
            
        return output 