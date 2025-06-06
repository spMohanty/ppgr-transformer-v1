"""
Transformer building blocks for the glucose forecasting model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import copy


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
            x, memory, memory,
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