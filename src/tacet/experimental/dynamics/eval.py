"""Evaluation for Layer 4 dynamics: P1.A MRR, P1.B rollout coherence, P1.C latency."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence


def filtered_rank(scores, gold_idx, filter_idx) -> int:  # noqa: ANN001
    """1-based filtered rank of the gold entity among all candidates.

    ``scores`` is indexed by entity id; entities in ``filter_idx`` (other true
    answers at the same timestamp) are excluded from the count, per the
    standard time-aware filtered protocol. Ties are broken optimistically
    (minimum rank): only strictly-greater scores count as better. This is
    harmless for the continuous neural scores used here, where exact ties do
    not occur.
    """
    gold = scores[gold_idx]
    better = 0
    for i, s in enumerate(scores):
        if i == gold_idx or i in filter_idx:
            continue
        if s > gold:
            better += 1
    return better + 1


# ---------------------------------------------------------------------------
# P1.A — filtered MRR / Hits@k on next-edge prediction
# ---------------------------------------------------------------------------


def eval_single_step(  # noqa: PLR0912, PLR0914
    traj,  # noqa: ANN001
    kge,  # noqa: ANN001
    state_enc,  # noqa: ANN001
    event_enc,  # noqa: ANN001
    rssm,  # noqa: ANN001
    decoder,  # noqa: ANN001
    candidates_per_query: int = 200,
    zero_latent: bool = False,
) -> dict:
    """P1.A — filtered MRR / Hits@1 / Hits@3 / Hits@10 on next-edge prediction.

    For each transition t: roll the posterior to z_t, then for each TRUE
    new edge (s, r, ?) in G_{t+1}, rank the gold o against
    candidates_per_query sampled negative tails by decoder logit; compute
    the filtered rank.  Returns {'MRR':, 'Hits@1':, 'Hits@3':, 'Hits@10':, 'n':}.

    If ``zero_latent`` is True the recurrence still advances normally but the
    latent fed to the decoder is zeroed — an ablation that isolates how much
    the temporal dynamics latent contributes to ranking over and above the
    frozen ComplEx embeddings the decoder also consumes.
    """
    import numpy as np
    import torch

    device = kge.device
    ent_idx_all = np.fromiter(kge.ent.values(), dtype=np.int64)
    rng = np.random.default_rng(0)

    h, z = rssm.initial(1, device=device)

    reciprocal_ranks: list[float] = []
    hits: dict[int, list[float]] = {1: [], 3: [], 10: []}

    for t in range(len(traj)):
        g_t = traj.at(t)
        e_batch = traj.event_at(t)
        g_next = traj.at(t + 1)

        # Encode and roll posterior
        s_np = state_enc.encode(g_t)
        e_np = event_enc.encode(e_batch)
        s_t = torch.from_numpy(s_np).unsqueeze(0).to(device)
        e_t = torch.from_numpy(e_np).unsqueeze(0).to(device)

        with torch.no_grad():
            z_post, h_new = rssm.posterior(h, z, e_t, s_t)

        h = h_new.detach()
        z = z_post.detach()

        # The latent fed to the decoder; recurrence above still uses the real
        # z_post, so the ablation only removes the dynamics signal at scoring.
        z_decode = torch.zeros_like(z_post) if zero_latent else z_post

        # A dynamics model is scored on the NEW edges of G_{t+1} (the change
        # G_{t+1} \ G_t), not on persistent edges that already held in G_t:
        # snapshots are cumulative, so scoring all of g_next would both be
        # trivial (persistent edges) and grow quadratically with t.
        prev_edges = {(e.source, e.relation, e.target) for e in g_t.edges}
        new_edges = [e for e in g_next.edges if (e.source, e.relation, e.target) not in prev_edges]

        # True tails per (s, r) among the new edges, for filtered ranking.
        true_tails: dict[tuple[str, str], set[str]] = {}
        for edge in new_edges:
            true_tails.setdefault((edge.source, edge.relation), set()).add(edge.target)

        # Evaluate each new edge that is within the KGE vocabulary
        for edge in new_edges:
            s, r, o = edge.source, edge.relation, edge.target
            if s not in kge.ent or r not in kge.rel or o not in kge.ent:
                continue

            gold_idx = kge.ent[o]

            # Sample candidates_per_query negative tails, excluding the true
            # tails for this (s, r) pair (filtered protocol).  Sampling from a
            # precomputed index array is O(candidates), not O(entities).
            true_idx = {kge.ent[e] for e in true_tails.get((s, r), ()) if e in kge.ent}
            n_draw = min(candidates_per_query + len(true_idx), len(ent_idx_all))
            drawn = rng.choice(ent_idx_all, size=n_draw, replace=False)
            neg_indices = [int(i) for i in drawn if int(i) not in true_idx][:candidates_per_query]

            # All candidate indices: gold + negatives
            candidate_indices = [gold_idx] + neg_indices

            cand_t = torch.tensor(candidate_indices, device=device)
            with torch.no_grad():
                s_emb = kge._E_re[kge.ent[s]].unsqueeze(0).expand(len(candidate_indices), -1)  # noqa: SLF001
                r_emb = kge._R_re[kge.rel[r]].unsqueeze(0).expand(len(candidate_indices), -1)  # noqa: SLF001
                o_emb = kge._E_re[cand_t]  # noqa: SLF001
                z_exp = z_decode.expand(len(candidate_indices), -1)
                logits = decoder(z_exp, s_emb, r_emb, o_emb)  # (n_cands,)

            # Rank of the gold (index 0) among all candidates (descending logit)
            logits_np = logits.cpu().numpy()
            gold_score = logits_np[0]
            rank = int((logits_np > gold_score).sum()) + 1  # 1-based

            reciprocal_ranks.append(1.0 / rank)
            for k in (1, 3, 10):
                hits[k].append(1.0 if rank <= k else 0.0)

    n = len(reciprocal_ranks)
    if n == 0:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "n": 0}

    return {
        "MRR": float(sum(reciprocal_ranks) / n),
        "Hits@1": float(sum(hits[1]) / n),
        "Hits@3": float(sum(hits[3]) / n),
        "Hits@10": float(sum(hits[10]) / n),
        "n": n,
    }


def eval_single_step_full_ranking(  # noqa: PLR0912, PLR0914, PLR0915
    traj,  # noqa: ANN001
    kge,  # noqa: ANN001
    state_enc,  # noqa: ANN001
    event_enc,  # noqa: ANN001
    rssm,  # noqa: ANN001
    decoder,  # noqa: ANN001
) -> dict:
    """P1.A under the standard TKG-forecasting protocol: time-aware filtered,
    ranking against ALL entities, mean over tail- and head-prediction.

    This is the number comparable to RE-GCN / TiRGN etc.  Unlike
    ``eval_single_step`` (which ranks against a sample of negatives and is
    therefore optimistic), every new edge (s, r, o) of G_{t+1} is scored by
    ranking the gold against all entities, filtering out other true answers
    at the same timestamp.  Returns {'MRR','Hits@1','Hits@3','Hits@10','n'}.
    """
    import torch

    device = kge.device
    n_ent = len(kge.ent)
    all_ent = torch.arange(n_ent, device=device)

    h, z = rssm.initial(1, device=device)
    rr: list[float] = []
    hits: dict[int, list[float]] = {1: [], 3: [], 10: []}

    for t in range(len(traj)):
        g_t = traj.at(t)
        e_batch = traj.event_at(t)
        g_next = traj.at(t + 1)

        s_t = torch.from_numpy(state_enc.encode(g_t)).unsqueeze(0).to(device)
        e_t = torch.from_numpy(event_enc.encode(e_batch)).unsqueeze(0).to(device)
        with torch.no_grad():
            z_post, h_new = rssm.posterior(h, z, e_t, s_t)
        h = h_new.detach()
        z = z_post.detach()

        prev_edges = {(e.source, e.relation, e.target) for e in g_t.edges}
        new_edges = [e for e in g_next.edges if (e.source, e.relation, e.target) not in prev_edges]

        tails: dict[tuple[str, str], set[str]] = {}
        heads: dict[tuple[str, str], set[str]] = {}
        for e in new_edges:
            tails.setdefault((e.source, e.relation), set()).add(e.target)
            heads.setdefault((e.relation, e.target), set()).add(e.source)

        for e in new_edges:
            s, r, o = e.source, e.relation, e.target
            if s not in kge.ent or r not in kge.rel or o not in kge.ent:
                continue
            with torch.no_grad():
                zc = z_post.expand(n_ent, -1)
                r_emb = kge._R_re[kge.rel[r]].unsqueeze(0).expand(n_ent, -1)  # noqa: SLF001
                # tail prediction: fix (s, r), score all candidate objects
                s_emb = kge._E_re[kge.ent[s]].unsqueeze(0).expand(n_ent, -1)  # noqa: SLF001
                st = decoder(zc, s_emb, r_emb, kge._E_re[all_ent]).cpu().numpy()  # noqa: SLF001
                ft = {kge.ent[x] for x in tails[(s, r)] if x in kge.ent} - {kge.ent[o]}
                rt = filtered_rank(st, kge.ent[o], ft)
                # head prediction: fix (r, o), score all candidate subjects
                o_emb = kge._E_re[kge.ent[o]].unsqueeze(0).expand(n_ent, -1)  # noqa: SLF001
                sh = decoder(zc, kge._E_re[all_ent], r_emb, o_emb).cpu().numpy()  # noqa: SLF001
                fh = {kge.ent[x] for x in heads[(r, o)] if x in kge.ent} - {kge.ent[s]}
                rh = filtered_rank(sh, kge.ent[s], fh)
            for rank in (rt, rh):
                rr.append(1.0 / rank)
                for k in (1, 3, 10):
                    hits[k].append(1.0 if rank <= k else 0.0)

    n = len(rr)
    if n == 0:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "n": 0}
    return {
        "MRR": float(sum(rr) / n),
        "Hits@1": float(sum(hits[1]) / n),
        "Hits@3": float(sum(hits[3]) / n),
        "Hits@10": float(sum(hits[10]) / n),
        "n": n,
    }


# ---------------------------------------------------------------------------
# P1.B — ontology-coherence rate over k-step prior rollouts
# ---------------------------------------------------------------------------


def eval_rollout_coherence(  # noqa: PLR0914
    traj,  # noqa: ANN001
    kge,  # noqa: ANN001
    state_enc,  # noqa: ANN001
    event_enc,  # noqa: ANN001
    rssm,  # noqa: ANN001
    decoder,  # noqa: ANN001
    ontology,  # noqa: ANN001
    ks: Sequence[int] = (3, 5, 10),
    edges_per_step: int = 50,
) -> dict:
    """P1.B — fraction of predicted top edges that are ontology-consistent
    over k-step rollouts.  Returns {3: rate, 5: rate, 10: rate}.

    Roll the PRIOR forward k steps from some start t; at each step decode the
    top edges_per_step (s,r,o) by logit; check ontology.allows(...) on each;
    coherence = (consistent predicted edges) / (total predicted edges).
    """
    import torch

    device = kge.device
    ent_list = list(kge.ent.keys())
    rel_list = list(kge.rel.keys())
    n_ent = len(ent_list)
    n_rel = len(rel_list)

    # Pick start t = 0 (earliest); roll up to max(ks) steps using prior
    max_k = max(ks)
    start_t = 0

    # Warm up hidden state with one posterior step at start_t if we have data
    h, z = rssm.initial(1, device=device)
    if len(traj) > 0:
        g0 = traj.at(start_t)
        e0 = traj.event_at(start_t)
        s_np = state_enc.encode(g0)
        e_np = event_enc.encode(e0)
        s0 = torch.from_numpy(s_np).unsqueeze(0).to(device)
        e0t = torch.from_numpy(e_np).unsqueeze(0).to(device)
        with torch.no_grad():
            z, h = rssm.posterior(h, z, e0t, s0)

    # Pre-compute entity / relation embeddings for candidate generation
    with torch.no_grad():
        all_e_emb = kge._E_re  # (n_ent, dim)  # noqa: SLF001
        all_r_emb = kge._R_re  # (n_rel, dim)  # noqa: SLF001

    # We roll the prior forward max_k steps and record per-step coherence
    step_coherences: list[float] = []

    for step in range(max_k):
        # Use a dummy zero event embedding for the prior rollout
        e_zero = torch.zeros(1, rssm.cfg.event_in_dim, device=device)
        with torch.no_grad():
            z_prior, h = rssm.prior(h, z, e_zero)
        z = z_prior.detach()

        # Decode a sample of (s, r, o) candidates and pick top edges_per_step
        # Strategy: sample a batch of random triples and rank by decoder logit
        sample_size = min(edges_per_step * 20, n_ent * n_rel)
        if sample_size == 0:
            step_coherences.append(1.0)
            continue

        import numpy as np

        rng = np.random.default_rng(step)
        s_idx_np = rng.integers(0, n_ent, sample_size)
        r_idx_np = rng.integers(0, n_rel, sample_size)
        o_idx_np = rng.integers(0, n_ent, sample_size)

        s_idx = torch.from_numpy(s_idx_np).to(device)
        r_idx = torch.from_numpy(r_idx_np).to(device)
        o_idx = torch.from_numpy(o_idx_np).to(device)

        with torch.no_grad():
            s_emb = all_e_emb[s_idx]
            r_emb = all_r_emb[r_idx]
            o_emb = all_e_emb[o_idx]
            z_exp = z_prior.expand(sample_size, -1)
            logits = decoder(z_exp, s_emb, r_emb, o_emb)

        # Pick top edges_per_step by logit
        n_top = min(edges_per_step, sample_size)
        top_k = torch.topk(logits, n_top).indices.cpu().numpy()

        # Check ontology coherence for each selected triple
        # We need a reference graph — use the last available snapshot
        ref_graph = traj.at(min(start_t + step, len(traj) - 1))

        consistent = 0
        for idx in top_k:
            s_name = ent_list[int(s_idx_np[idx])]
            r_name = rel_list[int(r_idx_np[idx])]
            o_name = ent_list[int(o_idx_np[idx])]
            try:
                ok = ontology.allows(ref_graph, s_name, r_name, o_name)
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                consistent += 1

        step_coherences.append(consistent / n_top)

    # Aggregate: coherence at k = mean of first k steps
    result: dict = {}
    for k in ks:
        steps = step_coherences[:k]
        result[k] = float(sum(steps) / len(steps)) if steps else 0.0
    return result


# ---------------------------------------------------------------------------
# P1.C — single-step Layer-4 inference latency
# ---------------------------------------------------------------------------


def eval_latency(
    traj,  # noqa: ANN001
    kge,  # noqa: ANN001
    state_enc,  # noqa: ANN001
    event_enc,  # noqa: ANN001
    rssm,  # noqa: ANN001
    decoder,  # noqa: ANN001
    tier2_call: Callable[[], None] | None = None,
    n_queries: int = 100,
) -> dict:
    """P1.C — mean single-step Layer-4 inference latency, and the ratio to a
    Tier-2 (KGE predict_tail) call if tier2_call is provided.  Returns
    {'layer4_ms':, 'tier2_ms':, 'ratio':}.
    """
    import torch

    device = kge.device

    # Warm up: run one pass to avoid cold-start JIT / cuDNN overhead
    h, z = rssm.initial(1, device=device)
    if len(traj) > 0:
        g0 = traj.at(0)
        e0 = traj.event_at(0)
        s_np = state_enc.encode(g0)
        e_np = event_enc.encode(e0)
        s0 = torch.from_numpy(s_np).unsqueeze(0).to(device)
        e0t = torch.from_numpy(e_np).unsqueeze(0).to(device)
        with torch.no_grad():
            z, h = rssm.posterior(h, z, e0t, s0)

    # Prepare a fixed input from the first available step
    t_idx = 0
    g_t = traj.at(t_idx)
    e_batch = traj.event_at(t_idx)
    s_fixed = torch.from_numpy(state_enc.encode(g_t)).unsqueeze(0).to(device)
    e_fixed = torch.from_numpy(event_enc.encode(e_batch)).unsqueeze(0).to(device)

    # Measure Layer-4 single-step latency
    h0, z0 = rssm.initial(1, device=device)
    start = time.perf_counter()
    for _ in range(n_queries):
        with torch.no_grad():
            _z, _h = rssm.posterior(h0, z0, e_fixed, s_fixed)
    elapsed_s = time.perf_counter() - start
    layer4_ms = elapsed_s * 1000.0 / n_queries

    # Measure Tier-2 latency if a callable is provided
    tier2_ms: float | None = None
    ratio: float | None = None
    if tier2_call is not None:
        t2_start = time.perf_counter()
        for _ in range(n_queries):
            tier2_call()
        t2_elapsed = time.perf_counter() - t2_start
        tier2_ms = t2_elapsed * 1000.0 / n_queries
        ratio = layer4_ms / tier2_ms if tier2_ms > 0 else None

    return {
        "layer4_ms": layer4_ms,
        "tier2_ms": tier2_ms,
        "ratio": ratio,
    }
