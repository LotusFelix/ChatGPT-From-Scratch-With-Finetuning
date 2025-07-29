"""
GPT-style model: embeddings + stack of Transformer blocks + final projection.
"""
import torch
from torch.nn import Module, Embedding, Dropout, Linear
from .block import TransformerBlock
from .layers import LayerNorm

class GPTModel(Module):
    def __init__(self, config):
        """
        Args:
            config (dict): must contain:
                - vocab_size, context_length, embed_dim, n_layers, n_heads, drop_rate, qkv_bias
        """
        super().__init__()
        self.config = config
        self.vocab_size     = config["vocab_size"]
        self.context_length = config["context_length"]
        self.embed_dim      = config["embed_dim"]
        self.n_layers       = config["n_layers"]
        self.drop_rate      = config["drop_rate"]

        # token & position embeddings
        self.token_emb = Embedding(self.vocab_size, self.embed_dim)
        self.pos_emb   = Embedding(self.context_length, self.embed_dim)
        self.dropout   = Dropout(self.drop_rate)

        # stack of transformer blocks
        blocks = [TransformerBlock(config) for _ in range(self.n_layers)]
        self.blocks = torch.nn.Sequential(*blocks)

        # final layer norm & language head
        self.ln_f = LayerNorm(config)
        self.head = Linear(self.embed_dim, self.vocab_size, bias=False)

    def forward(self, idx):
        """
        Forward pass through GPTModel.

        Args:
            idx (LongTensor): token indices, shape (batch, seq_len)
        Returns:
            logits (Tensor): shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = idx.size()
        # token & position embeddings
        tok_emb = self.token_emb(idx)                       # (batch, seq_len, embed_dim)
        pos_ids = torch.arange(seq_len, device=idx.device)  # (seq_len,)
        pos_emb = self.pos_emb(pos_ids).unsqueeze(0)        # (1, seq_len, embed_dim)

        x = tok_emb + pos_emb
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits