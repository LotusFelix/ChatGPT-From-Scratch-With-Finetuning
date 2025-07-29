
"""
Transformer block: attention + feedforward + residual connections + layer norms.
"""
import torch
from torch.nn import Module, Dropout
from .attention import MultiHeadAttention
from .layers    import LayerNorm, FeedForward

class TransformerBlock(Module):
    def __init__(self, config):
        """
        Args:
            config (dict): must contain:
                - embed_dim, n_heads, context_length, drop_rate, qkv_bias
        """
        super().__init__()
        self.config = config
        # submodules
        self.attn  = MultiHeadAttention(config)
        self.ff    = FeedForward(config)
        self.ln1   = LayerNorm(config)
        self.ln2   = LayerNorm(config)
        self.drop1 = Dropout(config["drop_rate"])
        self.drop2 = Dropout(config["drop_rate"])

    def forward(self, x):
        # Pre-norm attention
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.drop1(x)
        x = x + residual

        # Pre-norm feed-forward
        residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.drop2(x)
        x = x + residual
        return x