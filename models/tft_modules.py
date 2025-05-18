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
            self.resample = TimeDistributedInterpolation(
                self.input_size, batch_first=True, trainable=False
            )

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
        
        self.context_layer = nn.Linear(context_dim, hidden_dim) if context_dim is not None else None
    
    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # Apply layer norm before feed-forward block
        x_norm = self.norm1(x)
        hidden = self.fc1(x_norm)
        if context is not None:
            hidden = hidden + self.context_layer(context)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        out = self.fc2(hidden)
        out = self.dropout(out)
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
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for _ in range(self.num_layers):
            output = self.shared_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
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
    
