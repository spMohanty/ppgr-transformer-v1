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


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network as used in the Temporal Fusion Transformer.
    
    This is a key component that improves gradient flow and allows for better training dynamics.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.context_size = context_size
        self.residual = residual
        
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Optional context projection
        if context_size is not None:
            self.context = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context = None
        
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Residual connection handling
        if (input_size != output_size) and self.residual:
            self.residual_proj = nn.Linear(input_size, output_size)
        else:
            self.residual_proj = None
        
        # Gating layer
        self.gate = nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize the weights of the network."""
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name or "gate" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)
    
    def forward(self, x, context=None):
        """
        Forward pass through the Gated Residual Network.
        
        Args:
            x: Input tensor
            context: Optional context tensor for conditioning
            
        Returns:
            Output tensor
        """
        # Handle residual connection
        if self.residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
        else:
            residual = 0
        
        # Main branch
        hidden = self.fc1(x)
        
        # Add context if provided
        if self.context is not None and context is not None:
            context_projection = self.context(context)
            hidden = hidden + context_projection
        
        # Apply non-linearity
        hidden = F.elu(hidden)
        hidden = self.dropout_layer(hidden)
        
        # Second layer
        outputs = self.fc2(hidden)
        
        # Gating mechanism
        gate = torch.sigmoid(self.gate(x))
        outputs = outputs * gate
        
        # Residual connection and normalization
        outputs = self.norm(outputs + residual)
        
        return outputs


class StaticContextEnrichment(nn.Module):
    """
    Apply static context enrichment to input features.
    This is a key component in the TFT model architecture.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size
        )
    
    def forward(self, x, static_context):
        """
        Args:
            x: Input tensor [batch_size, time_steps, hidden_size]
            static_context: Static context [batch_size, time_steps, hidden_size] or [batch_size, hidden_size]
            
        Returns:
            Enriched tensor [batch_size, time_steps, hidden_size]
        """
        # Ensure static context is properly expanded if needed
        if static_context.dim() == 2:
            batch_size, time_steps, _ = x.shape
            # Expand static context across time dimension to match x
            static_context = static_context.unsqueeze(1).expand(batch_size, time_steps, -1)
        
        # Flatten batch and time dimensions for more efficient processing
        original_shape = x.shape
        batch_size, time_steps, hidden_size = original_shape
        
        # Reshape to [batch_size * time_steps, hidden_size]
        x_flat = x.reshape(-1, hidden_size)
        static_context_flat = static_context.reshape(-1, hidden_size)
        
        # Apply GRN
        outputs_flat = self.grn(x_flat, static_context_flat)
        
        # Reshape back to original shape
        outputs = outputs_flat.reshape(original_shape)
        
        return outputs


class PreNormResidualBlock(nn.Module):
    """
    A pre-norm residual block as used in the Temporal Fusion Transformer.
    Applies layer normalization before the main computation for better gradient flow.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, context_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Optional context conditioning
        self.context_layer = nn.Linear(context_dim, hidden_dim) if context_dim is not None else None
        
        # Initialize weights properly for stable training
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with standard initialization techniques for stability."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        if self.context_layer is not None:
            nn.init.xavier_uniform_(self.context_layer.weight)
            nn.init.zeros_(self.context_layer.bias)
    
    def forward(self, x, context=None):
        """
        Forward pass with optional context injection.
        
        Args:
            x: Input tensor of shape [batch_size, ..., input_dim]
            context: Optional context tensor for conditioning
            
        Returns:
            Output tensor of shape [batch_size, ..., output_dim]
        """
        # Normalize input for better gradient flow
        x_norm = self.norm1(x)
        
        # First linear layer
        hidden = self.fc1(x_norm)
        
        # Inject context if provided
        if context is not None and self.context_layer is not None:
            # Make sure context has compatible shape with hidden
            if context.dim() != hidden.dim():
                # Add singleton dimensions to match hidden
                for _ in range(hidden.dim() - context.dim()):
                    context = context.unsqueeze(1)
                # Expand context if needed
                if context.shape[1] == 1 and hidden.shape[1] > 1:
                    context = context.expand(-1, hidden.shape[1], -1)
            
            # Add context to hidden state
            context_hidden = self.context_layer(context)
            hidden = hidden + context_hidden
        
        # Apply activation and second linear layer
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        out = self.fc2(hidden)
        
        # Apply dropout and residual connection
        out = self.dropout(out)
        
        # Check for potential numerical issues
        if torch.isnan(out).any() or torch.isinf(out).any():
            # Fall back to identity if numerical issues detected
            return x
        
        # Add residual connection (output dim must match input dim)
        if x.shape[-1] == out.shape[-1]:
            out = out + x
        
        return out 