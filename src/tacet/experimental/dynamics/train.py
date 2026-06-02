"""Teacher-forced training loop for Layer 4 dynamics (ELBO-style)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tacet.experimental.dynamics.decoder import EdgeProbabilityHead
from tacet.experimental.dynamics.encoders import EventEncoder, GraphStateEncoder
from tacet.experimental.dynamics.rssm import RSSM
from tacet.experimental.dynamics.trajectory import Trajectory


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-6
    kl_weight: float = 1.0
    neg_per_pos: int = 4
    seed: int = 0
    # P2.2 multi-step BPTT + DreamerV3 stabilizers
    bptt_window: int = 1  # >1 backprops through this many transitions
    # free_bits clamps the KL floor; DreamerV3 uses 1.0 for high-dim latents but
    # that over-regularises this small latent (it zeroes the KL gradient and
    # halved full-ranking MRR), so the default is off and it is tuned per run.
    free_bits: float = 0.0
    kl_alpha: float = 0.8  # KL-balancing weight on the prior side
    # P2.4 latent consistency
    consistency_weight: float = 0.0


def _kl_balanced(
    post,
    prior,
    z_per,
    z_cat,  # noqa: ANN001
    free_bits: float = 1.0,
    alpha: float = 0.8,
):
    """DreamerV3 KL-balancing + free-bits between two flattened categoricals.

    alpha weights the prior-side term (stop-grad on posterior) vs the
    posterior-side term (stop-grad on prior); free_bits clamps the result so
    the latent is not over-regularised early.
    """
    import torch

    def kl(a, b):
        pa = a.view(a.shape[0], z_per, z_cat).clamp_min(1e-8)
        pb = b.view(b.shape[0], z_per, z_cat).clamp_min(1e-8)
        return (pa * (pa.log() - pb.log())).sum(-1).mean()

    kl_prior = kl(post.detach(), prior)  # train the prior toward the posterior
    kl_post = kl(post, prior.detach())  # train the posterior toward the prior
    val = alpha * kl_prior + (1.0 - alpha) * kl_post
    return torch.clamp(val, min=free_bits)


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a 1-D numpy float32 array to a (1, D) tensor on device."""
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _kl_categorical(
    post: torch.Tensor,
    prior: torch.Tensor,
    z_per_state: int,
    z_categories: int,
) -> torch.Tensor:
    """Batch KL divergence between two flattened categorical latents.

    Both tensors are softmax distributions of shape (B, z_per_state*z_categories).
    Reshape to (B, z_per_state, z_categories), compute cross-entropy-style KL
    element-wise, then sum over categories and mean over slots and batch.

    KL(post || prior) = sum_k post_k * (log post_k - log prior_k)
    """
    b = post.shape[0]
    p = post.view(b, z_per_state, z_categories)
    q = prior.view(b, z_per_state, z_categories)
    # clamp for numerical safety
    log_p = torch.log(p.clamp_min(1e-8))
    log_q = torch.log(q.clamp_min(1e-8))
    kl = (p * (log_p - log_q)).sum(-1)  # (B, z_per_state)
    return kl.mean()


def train_layer4(  # noqa: PLR0912, PLR0914, PLR0915
    traj: Trajectory,
    kge,  # noqa: ANN001
    state_enc: GraphStateEncoder,
    event_enc: EventEncoder,
    rssm: RSSM,
    decoder: EdgeProbabilityHead,
    cfg: TrainConfig,
) -> list[float]:
    """Teacher-forced ELBO training loop over a Trajectory.

    Parameters
    ----------
    traj:
        Sequence of (G_t, EventBatch_t, G_{t+1}) transitions.
    kge:
        Fitted TorchComplEx — provides entity/relation embeddings and vocab.
    state_enc:
        Numpy graph encoder (frozen, no grad).
    event_enc:
        Numpy event encoder (frozen, no grad).
    rssm:
        RSSM module to train.
    decoder:
        EdgeProbabilityHead to train.
    cfg:
        Hyper-parameters for the training loop.

    Returns
    -------
    list[float]
        Per-epoch mean loss values (length == cfg.epochs).
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device: torch.device = kge.device
    n_ent = len(kge.ent)
    n_rel = len(kge.rel)

    # Collect all trainable params: RSSM + decoder MLP.
    # KGE embeddings are frozen (we index them with no_grad at inference).
    params = list(rssm.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    z_cat = rssm.cfg.z_categories
    z_per = rssm.cfg.z_per_state

    loss_history: list[float] = []

    for _epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_steps = 0

        # Reset hidden state at the start of each epoch (batch_size=1)
        h, z = rssm.initial(1, device=device)
        window_loss = torch.zeros((), device=device)
        window_count = 0
        opt.zero_grad()

        for t in range(len(traj)):
            g_t = traj.at(t)
            e_batch = traj.event_at(t)
            g_next = traj.at(t + 1)

            # --- encode state and events (numpy, no grad) ---
            s_np = state_enc.encode(g_t)
            e_np = event_enc.encode(e_batch)

            s_t = _to_tensor(s_np, device)  # (1, state_in_dim)
            e_t = _to_tensor(e_np, device)  # (1, event_in_dim)

            # --- RSSM forward (with grad) ---
            z_post, h_new = rssm.posterior(h, z, e_t, s_t)  # (1, z_dim)
            z_prior, _ = rssm.prior(h, z, e_t)  # prior target (with grad), same h/z

            # advance hidden state — KEEP attached so BPTT flows across the
            # window; detach happens only at window boundaries below
            h, z = h_new, z_post

            # --- collect positive edges: the NEW edges of G_{t+1} only ---
            # Snapshots are cumulative, so g_next ⊇ g_t; training the dynamics
            # head on persistent edges is both trivial (they already held) and
            # grows quadratically with t.  Score the change G_{t+1} \ G_t.
            prev_edges = {(e.source, e.relation, e.target) for e in g_t.edges}
            pos_edges = [
                (e.source, e.relation, e.target)
                for e in g_next.edges
                if (e.source, e.relation, e.target) not in prev_edges
                and e.source in kge.ent
                and e.relation in kge.rel
                and e.target in kge.ent
            ]

            if not pos_edges:
                continue

            n_pos = len(pos_edges)

            # --- build positive embedding indices ---
            s_idx = torch.tensor([kge.ent[s] for s, _, _ in pos_edges], device=device)  # (n_pos,)
            r_idx = torch.tensor([kge.rel[r] for _, r, _ in pos_edges], device=device)
            o_idx = torch.tensor([kge.ent[o] for _, _, o in pos_edges], device=device)

            with torch.no_grad():
                s_emb_pos = kge._E_re[s_idx]  # (n_pos, kge_dim)  # noqa: SLF001
                r_emb_pos = kge._R_re[r_idx]  # noqa: SLF001
                o_emb_pos = kge._E_re[o_idx]  # noqa: SLF001

            # --- negative sampling: cfg.neg_per_pos random (s,r,o) per positive ---
            n_neg = n_pos * cfg.neg_per_pos
            ns_idx_np = rng.integers(0, n_ent, n_neg)
            nr_idx_np = rng.integers(0, n_rel, n_neg)
            no_idx_np = rng.integers(0, n_ent, n_neg)

            ns_idx = torch.from_numpy(ns_idx_np).to(device)
            nr_idx = torch.from_numpy(nr_idx_np).to(device)
            no_idx = torch.from_numpy(no_idx_np).to(device)

            with torch.no_grad():
                s_emb_neg = kge._E_re[ns_idx]  # noqa: SLF001
                r_emb_neg = kge._R_re[nr_idx]  # noqa: SLF001
                o_emb_neg = kge._E_re[no_idx]  # noqa: SLF001

            # --- broadcast z_post to match positive and negative batches ---
            z_pos = z_post.expand(n_pos, -1)  # (n_pos, z_dim)
            z_neg = z_post.expand(n_neg, -1)  # (n_neg, z_dim)

            # --- decode ---
            logits_pos = decoder(z_pos, s_emb_pos, r_emb_pos, o_emb_pos)  # (n_pos,)
            logits_neg = decoder(z_neg, s_emb_neg, r_emb_neg, o_emb_neg)  # (n_neg,)

            # --- BCE loss ---
            logits_all = torch.cat([logits_pos, logits_neg], dim=0)
            labels = torch.cat(
                [
                    torch.ones(n_pos, device=device),
                    torch.zeros(n_neg, device=device),
                ]
            )
            bce = bce_loss(logits_all, labels)

            # --- KL divergence (balanced + free-bits) ---
            kl = _kl_balanced(z_post, z_prior, z_per, z_cat, cfg.free_bits, cfg.kl_alpha)
            step_loss = bce + cfg.kl_weight * kl

            # --- latent consistency (P2.4): pull the prior toward the posterior
            # in L2, complementing the KL; reduces multi-step rollout drift ---
            if cfg.consistency_weight > 0:
                consistency = ((z_prior - z_post.detach()) ** 2).mean()
                step_loss = step_loss + cfg.consistency_weight * consistency

            window_loss = window_loss + step_loss
            epoch_loss += float(step_loss.item())
            n_steps += 1

            # --- BPTT window boundary: one backward per window, then truncate ---
            window_count += 1
            if window_count >= cfg.bptt_window:
                if window_loss.requires_grad:
                    window_loss.backward()
                    opt.step()
                opt.zero_grad()
                h, z = h.detach(), z.detach()
                window_loss = torch.zeros((), device=device)
                window_count = 0

        # flush a partial final window
        if window_count > 0 and window_loss.requires_grad:
            window_loss.backward()
            opt.step()
            opt.zero_grad()

        loss_history.append(epoch_loss / max(n_steps, 1))

    return loss_history
