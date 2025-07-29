
"""
Core layers: LayerNorm and FeedForward network.
"""
import torch
from torch.nn import Module, Parameter, Linear, GELU

class LayerNorm(Module):
    def __init__(self, config, eps: float = 1e-5):
        """
        Args:
            config (dict): must contain 'embed_dim'.
            eps (float): stability constant.
        """
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.eps = eps
        # learnable gain and bias
        self.weight = Parameter(torch.ones(self.embed_dim))
        self.bias   = Parameter(torch.zeros(self.embed_dim))

    def forward(self, x):
        # compute mean and variance across last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias

class FeedForward(Module):
    def __init__(self, config):
        """
        Args:
            config (dict): must contain 'embed_dim' and 'drop_rate'.
        """
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.drop_rate = config["drop_rate"]
        # two-layer MLP with GELU activation
        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.act = GELU()
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x