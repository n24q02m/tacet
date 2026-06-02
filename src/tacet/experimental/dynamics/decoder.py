"""Edge-probability decoder for Layer 4: latent z -> P(edge active at t+1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class EdgeProbabilityHead(nn.Module):
    """Score candidate (s, r, o) triples from a predicted latent z.

    score(s,r,o,z) = MLP( z || E_re[s] || R_re[r] || E_re[o] ) -> logit
    Uses the real-part embeddings from a fitted TorchComplEx as fixed
    (frozen) features; only the MLP is trained.
    """

    def __init__(self, z_dim: int, kge_dim: int, hidden: int = 256) -> None:
        super().__init__()
        in_dim = z_dim + 3 * kge_dim
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(
        self, z: torch.Tensor, s_emb: torch.Tensor, r_emb: torch.Tensor, o_emb: torch.Tensor
    ) -> torch.Tensor:
        """z: (B, z_dim); s_emb/r_emb/o_emb: (B, kge_dim). Returns (B,) logits."""
        x = torch.cat([z, s_emb, r_emb, o_emb], dim=-1)
        return self.mlp(x).squeeze(-1)
