"""Dreamer-V3-style recurrent state-space model for graph dynamics (Layer 4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RSSMConfig:
    z_categories: int = 32
    z_per_state: int = 32
    h_dim: int = 256
    state_in_dim: int = 400
    event_in_dim: int = 280
    hidden: int = 256


class RSSM(nn.Module):
    """Recurrent state-space model: predicts next latent z_{t+1} from
    previous hidden h_t and event e_t (prior), or refines with the
    observed state s_t (posterior).  z is a flattened categorical
    distribution (softmax over z_categories, repeated z_per_state times).
    """

    def __init__(self, cfg: RSSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        z_dim = cfg.z_categories * cfg.z_per_state
        self.gru = nn.GRUCell(input_size=cfg.event_in_dim + z_dim, hidden_size=cfg.h_dim)
        self.prior_head = nn.Sequential(
            nn.Linear(cfg.h_dim, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, z_dim)
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(cfg.h_dim + cfg.state_in_dim, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, z_dim),
        )

    def _categorical(self, logits: torch.Tensor, unimix: float = 0.01) -> torch.Tensor:
        b = logits.shape[0]
        z = logits.view(b, self.cfg.z_per_state, self.cfg.z_categories)
        p = torch.softmax(z, dim=-1)
        if unimix > 0:  # DreamerV3 unimix: floor every class to avoid dead latents
            p = (1.0 - unimix) * p + unimix / self.cfg.z_categories
        return p.reshape(b, self.cfg.z_per_state * self.cfg.z_categories)

    def initial(
        self,
        batch_size: int,
        device=None,  # noqa: ANN001
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (h_0, z_0) zeros for the start of a rollout."""
        z_dim = self.cfg.z_categories * self.cfg.z_per_state
        return (
            torch.zeros(batch_size, self.cfg.h_dim, device=device),
            torch.zeros(batch_size, z_dim, device=device),
        )

    def prior(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        e: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step prior: h_t = GRU([z_{t-1}, e_t], h_{t-1}); z_t ~ prior_head(h_t)."""
        h = self.gru(torch.cat([z_prev, e], dim=-1), h_prev)
        z = self._categorical(self.prior_head(h))
        return z, h

    def posterior(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        e: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step posterior: same GRU update, but z refined with observed state s."""
        h = self.gru(torch.cat([z_prev, e], dim=-1), h_prev)
        z = self._categorical(self.posterior_head(torch.cat([h, s], dim=-1)))
        return z, h
