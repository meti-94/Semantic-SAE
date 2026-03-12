"""
ReLU Sparse Autoencoder (ReLUSAE) for activation bottleneck.
Encoder: hidden_dim -> d_sae (16K) -> ReLU; Decoder: d_sae -> hidden_dim.
Forward preserves shape (B, L, H) for stacking in the latent_qa pipeline.
"""

import torch
import torch.nn as nn


class ReLUSAE(nn.Module):
    """ReLU SAE: encoder -> ReLU -> decoder, with 16K intermediate dimension by default."""

    def __init__(
        self,
        hidden_size: int,
        d_sae: int = 16384,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_sae = d_sae
        self.encoder = nn.Linear(hidden_size, d_sae, dtype=dtype)
        self.decoder = nn.Linear(d_sae, hidden_size, dtype=dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, hidden_size) -> (batch, seq_len, hidden_size)."""
        shape = x.shape
        x_flat = x.view(-1, shape[-1])
        latent = torch.relu(self.encoder(x_flat))
        out = self.decoder(latent)
        return out.view(shape)


class TopKSAE(nn.Module):
    """
    Top-k SAE: encoder -> ReLU -> top-k mask -> decoder.

    For each row in the latent (batch * seq_len, d_sae), only the top
    `topk_percent` fraction of activations are kept; the rest are zeroed.
    """

    def __init__(
        self,
        hidden_size: int,
        d_sae: int = 16384,
        topk_percent: float = 0.01,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        assert 0.0 < topk_percent <= 1.0
        self.hidden_size = hidden_size
        self.d_sae = d_sae
        self.topk_percent = topk_percent
        self.encoder = nn.Linear(hidden_size, d_sae, dtype=dtype)
        self.decoder = nn.Linear(d_sae, hidden_size, dtype=dtype)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, hidden_size) -> (batch, seq_len, hidden_size)."""
        shape = x.shape
        x_flat = x.view(-1, shape[-1])

        # Encoder + ReLU
        latent = torch.relu(self.encoder(x_flat))

        # Compute k per row (at least 1, at most d_sae)
        k = max(1, int(self.d_sae * self.topk_percent))
        k = min(k, self.d_sae)

        if k == self.d_sae:
            sparse_latent = latent
        else:
            # Top-k mask per row
            values, indices = torch.topk(latent, k, dim=-1)
            mask = torch.zeros_like(latent)
            mask.scatter_(1, indices, 1.0)
            sparse_latent = latent * mask

        out = self.decoder(sparse_latent)
        return out.view(shape)
