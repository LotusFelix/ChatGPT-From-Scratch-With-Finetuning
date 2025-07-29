
"""
Multi-Head Self-Attention module.
"""
import math
import torch
from torch.nn import Module, Linear, Dropout

class MultiHeadAttention(Module):
    def __init__(self, config):
        """
        Args:
            config (dict): must contain:
                - embed_dim: model embedding size
                - n_heads: number of attention heads
                - context_length: max sequence length
                - drop_rate: dropout probability
                - qkv_bias: bool, include bias in Q/K/V projections
        """
        super().__init__()
        # Save config values
        self.embed_dim = config["embed_dim"]
        self.n_heads = config["n_heads"]
        self.context_length = config["context_length"]
        self.drop_rate = config["drop_rate"]
        self.qkv_bias = config["qkv_bias"]

        # Ensure divisibility for multi-head split
        assert self.embed_dim % self.n_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})"
        self.dim_head = self.embed_dim // self.n_heads

        # Linear projections for queries, keys, values
        self.query_proj = Linear(self.embed_dim, self.embed_dim, bias=self.qkv_bias)
        self.key_proj   = Linear(self.embed_dim, self.embed_dim, bias=self.qkv_bias)
        self.value_proj = Linear(self.embed_dim, self.embed_dim, bias=self.qkv_bias)

        # Causal mask: upper triangular, to prevent attending to future tokens
        mask = torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)

        # Output projection and dropout
        self.out_proj = Linear(self.embed_dim, self.embed_dim)
        self.dropout  = Dropout(self.drop_rate)

    def forward(self, x):
        """
        Compute multi-head self-attention.

        Args:
            x (Tensor): shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor: shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.size()

        # Project inputs to Q, K, V and reshape to (batch, heads, seq, head_dim)
        q = self.query_proj(x).view(batch_size, seq_len, self.n_heads, self.dim_head).transpose(1, 2)
        k = self.key_proj(x).view(batch_size, seq_len, self.n_heads, self.dim_head).transpose(1, 2)
        v = self.value_proj(x).view(batch_size, seq_len, self.n_heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dim_head)
        # Apply causal mask (same for all batches and heads)
        mask = self.mask[:seq_len, :seq_len].to(x.device)
        scores = scores.masked_fill(mask, float('-inf'))
        attn  = torch.softmax(scores, dim=-1)
        attn  = self.dropout(attn)

        # Aggregate values
        context = attn @ v  # (batch, heads, seq, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Final linear projection
        out = self.out_proj(context)
        return out