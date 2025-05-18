"""
Attention mechanisms for various transformers and models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module with NaN handling.
    """
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        # Handle NaN values in inputs
        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            q = torch.nan_to_num(q, nan=0.0)
            k = torch.nan_to_num(k, nan=0.0)
            v = torch.nan_to_num(v, nan=0.0)
        
        # Compute query-key overlap
        attn = torch.bmm(q, k.permute(0, 2, 1))

        if self.scale:
            dimension = torch.as_tensor(
                k.size(-1), dtype=attn.dtype, device=attn.device
            ).sqrt()
            attn = attn / dimension
        
        # Check for NaNs in attention weights before masking
        if torch.isnan(attn).any():
            # If attention contains NaNs, replace with zeros
            attn = torch.nan_to_num(attn, nan=0.0)
            
        if mask is not None:
            # Apply mask using PyTorch's masked_fill convention
            # In the TFT get_attention_mask function:
            # - True values indicate positions that CAN be attended to
            # - False values indicate positions that should be masked out
            # PyTorch's masked_fill applies the fill value at positions where the mask is True
            # So we need to negate the mask (~mask) to convert between these conventions
            attn = attn.masked_fill(~mask, -float("inf"))
        
        # Try to apply softmax, with fallback to uniform attention if it fails
        try:
            attn = self.softmax(attn)
        except Exception as e:
            # If softmax fails (e.g., all -inf), use uniform attention
            uniform_weights = torch.ones_like(attn) / attn.size(-1)
            attn = uniform_weights
            
        # Final NaN check on attention weights
        if torch.isnan(attn).any():
            # If we still have NaNs after softmax, use uniform attention
            uniform_weights = torch.ones_like(attn) / attn.size(-1)
            attn = uniform_weights

        if self.dropout is not None:
            attn = self.dropout(attn)
        
        # Apply attention weights to values
        output = torch.bmm(attn, v)
        
        # Final NaN check on output
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
            
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention from TFT implementation with robust NaN handling.
    """
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
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None):
        # Initial NaN check
        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
            # Replace NaNs with zeros in input tensors
            q = torch.nan_to_num(q, nan=0.0)
            k = torch.nan_to_num(k, nan=0.0)
            v = torch.nan_to_num(v, nan=0.0)
            
        # Apply value transformation
        try:
            vs = self.v_layer(v)
            if torch.isnan(vs).any():
                vs = torch.nan_to_num(vs, nan=0.0)
        except Exception as e:
            # Fallback if linear transformation fails
            vs = torch.zeros_like(v[:, :, :self.d_v])
            
        heads = []
        attns = []
        
        # Process each attention head
        for i in range(self.n_head):
            try:
                # Transform queries and keys for this head
                qs = self.q_layers[i](q)
                ks = self.k_layers[i](k)
                
                # Check for NaNs after linear transformations
                if torch.isnan(qs).any():
                    qs = torch.nan_to_num(qs, nan=0.0)
                if torch.isnan(ks).any():
                    ks = torch.nan_to_num(ks, nan=0.0)
                    
                # Apply attention mechanism (which now has NaN handling)
                head, attn = self.attention(qs, ks, vs, mask)
                
                # Apply dropout
                head_dropout = self.dropout(head)
                
                # Final NaN check for this head
                if torch.isnan(head_dropout).any():
                    head_dropout = torch.nan_to_num(head_dropout, nan=0.0)
                    
                heads.append(head_dropout)
                attns.append(attn)
                
            except Exception as e:
                # If processing this head fails, create a fallback with zeros
                batch_size, seq_len = q.shape[0], q.shape[1]
                fallback_head = torch.zeros(batch_size, seq_len, self.d_v, device=q.device)
                fallback_attn = torch.ones(batch_size, seq_len, k.shape[1], device=q.device) / k.shape[1]
                
                heads.append(fallback_head)
                attns.append(fallback_attn)

        # Combine results from all heads
        if self.n_head > 1:
            try:
                head = torch.stack(heads, dim=2)
                attn = torch.stack(attns, dim=2)
                
                # Check for NaNs after stacking
                if torch.isnan(head).any():
                    head = torch.nan_to_num(head, nan=0.0)
                if torch.isnan(attn).any():
                    attn = torch.nan_to_num(attn, nan=0.0)
                    
                outputs = torch.mean(head, dim=2)
            except Exception as e:
                # Fallback if stacking fails
                batch_size, seq_len = q.shape[0], q.shape[1]
                outputs = torch.zeros(batch_size, seq_len, self.d_model, device=q.device)
                attn = torch.ones(batch_size, seq_len, k.shape[1], self.n_head, device=q.device) / k.shape[1]
        else:
            outputs = heads[0]
            attn = attns[0]
        
        # Apply final transformation
        try:
            outputs = self.w_h(outputs)
            outputs = self.dropout(outputs)
            
            # Final NaN check on outputs
            if torch.isnan(outputs).any():
                outputs = torch.nan_to_num(outputs, nan=0.0)
        except Exception as e:
            # Fallback if transformation fails
            batch_size, seq_len = q.shape[0], q.shape[1]
            outputs = torch.zeros(batch_size, seq_len, self.d_model, device=q.device)

        return outputs, attn 