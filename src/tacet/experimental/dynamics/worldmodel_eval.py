"""Multi-step world-model evaluation: rollout fidelity + calibration.

Single-step MRR does not test a *world model* — it tests a one-step ranker.
These metrics roll the prior forward and measure whether imagined dynamics
stay faithful (rollout Hits@k vs the true future) and calibrated (ECE).
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:  # noqa: ANN001
    """Expected calibration error of predicted edge probabilities.

    Bins predictions by confidence; ECE is the weighted mean gap between bin
    confidence and bin accuracy.  0 = perfectly calibrated.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        ece += (m.sum() / probs.size) * abs(conf - acc)
    return float(ece)


def eval_rollout_fidelity(
    traj,
    kge,
    state_enc,
    event_enc,
    rssm,
    decoder,  # noqa: ANN001
    horizons=(1, 3, 5, 10),
    start_t: int = 0,
    candidates: int = 200,
    max_edges: int = 200,
) -> dict:
    """Roll the PRIOR forward and score the true new edges of G_{t+h} against
    sampled negatives.  Returns {h: Hits@10} per horizon — the curve that
    shows whether imagined dynamics stay faithful as the horizon grows.
    """
    import torch

    device = kge.device
    h, z = rssm.initial(1, device=device)
    s0 = torch.from_numpy(state_enc.encode(traj.at(start_t))).unsqueeze(0).to(device)
    e0 = torch.from_numpy(event_enc.encode(traj.event_at(start_t))).unsqueeze(0).to(device)
    with torch.no_grad():
        z, h = rssm.posterior(h, z, e0, s0)

    out: dict[int, float] = {}
    ezero = torch.zeros(1, rssm.cfg.event_in_dim, device=device)
    rng = np.random.default_rng(0)
    for step in range(1, max(horizons) + 1):
        with torch.no_grad():
            z, h = rssm.prior(h, z, ezero)
        if step not in horizons:
            continue
        ti = min(start_t + step, len(traj))
        g_t = traj.at(ti)
        g_p = traj.at(max(ti - 1, 0))
        prev = {(e.source, e.relation, e.target) for e in g_p.edges}
        new = [e for e in g_t.edges if (e.source, e.relation, e.target) not in prev]
        hit: list[float] = []
        for e in new[:max_edges]:
            if e.source not in kge.ent or e.relation not in kge.rel or e.target not in kge.ent:
                continue
            with torch.no_grad():
                cand = torch.from_numpy(rng.integers(0, len(kge.ent), candidates)).to(device)
                cand[0] = kge.ent[e.target]
                zc = z.expand(candidates, -1)
                s_e = kge._E_re[kge.ent[e.source]].unsqueeze(0).expand(candidates, -1)  # noqa: SLF001
                r_e = kge._R_re[kge.rel[e.relation]].unsqueeze(0).expand(candidates, -1)  # noqa: SLF001
                sc = decoder(zc, s_e, r_e, kge._E_re[cand]).cpu().numpy()  # noqa: SLF001
            rank = int((sc > sc[0]).sum()) + 1
            hit.append(1.0 if rank <= 10 else 0.0)
        out[step] = float(np.mean(hit)) if hit else 0.0
    return out
