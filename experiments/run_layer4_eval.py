"""Layer 4 dynamics end-to-end eval — ICEWS14 trajectory, ComplEx KGE, full eval suite.

Loads an ICEWS14 trajectory slice, fits TorchComplEx, trains the RSSM + decoder
briefly, then runs P1.A (MRR), P1.B (rollout coherence) and P1.C (latency) and
writes a JSON report.

Designed to run on a Modal GPU worker (not locally — torch DLL missing on Windows).

Example::

    python experiments/run_layer4_eval.py \\
        --root data/icews14 \\
        --limit-days 30 \\
        --epochs 20 \\
        --dim 64 \\
        --out experiments/results/layer4_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402  (after thread-env so numpy honours the limits)

from tacet.core.ontology import Ontology
from tacet.experimental.dynamics.decoder import EdgeProbabilityHead
from tacet.experimental.dynamics.encoders import (
    ActionAsNodeEventEncoder,
    BagOfTypesEventEncoder,
    ComplexN3PooledEncoder,
)
from tacet.experimental.dynamics.eval import (
    eval_latency,
    eval_rollout_coherence,
    eval_single_step,
    eval_single_step_full_ranking,
)
from tacet.experimental.dynamics.loaders import load_icews14_trajectory
from tacet.experimental.dynamics.rssm import RSSM, RSSMConfig
from tacet.experimental.dynamics.rssm_worldmodel import RSSMWorldModel
from tacet.experimental.dynamics.train import TrainConfig, train_layer4
from tacet.experimental.dynamics.worldmodel_eval import eval_rollout_fidelity
from tacet.kge.kge_torch import TorchComplEx, TorchKGEConfig


def _slice_trajectory(traj, limit_days: int | None):
    """Return a trajectory sliced to at most limit_days steps."""
    if limit_days is None or limit_days >= len(traj):
        return traj
    from tacet.experimental.dynamics.trajectory import Trajectory

    n = limit_days
    return Trajectory(
        snapshots=traj.snapshots[: n + 1],
        event_batches=traj.event_batches[:n],
    )


def main() -> None:  # noqa: PLR0914, PLR0915
    ap = argparse.ArgumentParser(description="Layer 4 dynamics eval — ICEWS14, torch backend.")
    ap.add_argument("--root", required=True, help="ICEWS14 dataset root (must contain train.txt)")
    ap.add_argument(
        "--split",
        choices=("train", "valid", "test"),
        default="train",
        help="ICEWS14 split to evaluate on",
    )
    ap.add_argument(
        "--limit-days",
        type=int,
        default=None,
        help="Cap trajectory to this many day-steps (faster iteration)",
    )
    ap.add_argument("--dim", type=int, default=64, help="KGE embedding dimension")
    ap.add_argument("--kge-epochs", type=int, default=50, help="KGE training epochs")
    ap.add_argument("--epochs", type=int, default=20, help="Layer-4 RSSM + decoder training epochs")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kl-weight", type=float, default=1.0)
    ap.add_argument("--neg-per-pos", type=int, default=4)
    ap.add_argument("--z-categories", type=int, default=32)
    ap.add_argument("--z-per-state", type=int, default=32)
    ap.add_argument("--h-dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument(
        "--candidates",
        type=int,
        default=200,
        help="Negative candidates per query for MRR eval (P1.A)",
    )
    ap.add_argument(
        "--edges-per-step",
        type=int,
        default=50,
        help="Top edges decoded per step for coherence eval (P1.B)",
    )
    ap.add_argument(
        "--latency-queries",
        type=int,
        default=100,
        help="Repetitions for latency measurement (P1.C)",
    )
    ap.add_argument(
        "--full-ranking",
        action="store_true",
        help="Also run the standard all-entity time-aware-filtered "
        "P1.A (comparable to RE-GCN/TiRGN; slower)",
    )
    ap.add_argument(
        "--event-encoder",
        choices=("bag", "action-node"),
        default="bag",
        help="bag-of-types (default) or action-as-node event encoding",
    )
    ap.add_argument(
        "--bptt-window",
        type=int,
        default=1,
        help="multi-step BPTT window (>1 backprops through time)",
    )
    ap.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="latent prior<->posterior L2 consistency weight",
    )
    ap.add_argument(
        "--free-bits",
        type=float,
        default=0.0,
        help="KL free-bits floor (0 = off; 1.0 over-regularises here)",
    )
    ap.add_argument(
        "--kl-alpha", type=float, default=0.8, help="KL-balancing weight on the prior side"
    )
    ap.add_argument(
        "--rollout-eval",
        action="store_true",
        help="run the multi-step world-model rollout fidelity suite",
    )
    ap.add_argument(
        "--planning-demo",
        action="store_true",
        help="run the P5 planning demo (RSSMWorldModel imagined rollout)",
    )
    ap.add_argument("--device", default="cuda", help="PyTorch device — 'cuda' recommended on Modal")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/results/layer4_eval.json")
    args = ap.parse_args()

    t_total = time.time()

    # ------------------------------------------------------------------
    # 1. Load trajectory
    # ------------------------------------------------------------------
    print(f"Loading ICEWS14 ({args.split}) from {args.root!r} ...")
    t0 = time.time()
    traj_full = load_icews14_trajectory(args.root, split=args.split)
    traj = _slice_trajectory(traj_full, args.limit_days)
    load_time = time.time() - t0
    print(f"  Trajectory: {len(traj)} steps ({len(traj_full)} total). Loaded in {load_time:.1f}s.")

    # ------------------------------------------------------------------
    # 2. Collect all triples across snapshots for KGE fitting
    # ------------------------------------------------------------------
    all_triples = list({t for snap in traj.snapshots for t in snap.triples()})
    print(f"  Unique triples for KGE: {len(all_triples)}")

    # ------------------------------------------------------------------
    # 3. Fit TorchComplEx
    # ------------------------------------------------------------------
    print(f"Fitting TorchComplEx dim={args.dim} epochs={args.kge_epochs} ...")
    t0 = time.time()
    kge_cfg = TorchKGEConfig(
        dim=args.dim,
        epochs=args.kge_epochs,
        device=args.device,
        batch_size=1024,
        negatives=16,
        seed=args.seed,
    )
    kge = TorchComplEx(kge_cfg)
    kge.fit(all_triples)
    kge_time = time.time() - t0
    print(f"  KGE trained in {kge_time:.1f}s. Entities: {len(kge.ent)}, Relations: {len(kge.rel)}")

    # ------------------------------------------------------------------
    # 4. Build encoders + RSSM + decoder
    # ------------------------------------------------------------------
    kge_dim = kge.cfg.dim
    state_in_dim = 2 * kge_dim  # ComplexN3PooledEncoder: re || im

    # Discover event types from all batches
    event_types = sorted({e.type for batch in traj.event_batches for e in batch.events})
    suffix = "..." if len(event_types) > 10 else ""
    print(f"Event types ({len(event_types)}): {event_types[:10]}{suffix}")

    state_enc = ComplexN3PooledEncoder(kge)
    if args.event_encoder == "action-node":
        event_enc = ActionAsNodeEventEncoder(event_types or ["_none"], kge)
    else:
        event_enc = BagOfTypesEventEncoder(event_types or ["_none"])
    event_in_dim = event_enc.dim
    print(f"Event encoder: {args.event_encoder} (event_in_dim={event_in_dim})")

    z_dim = args.z_categories * args.z_per_state
    rssm_cfg = RSSMConfig(
        z_categories=args.z_categories,
        z_per_state=args.z_per_state,
        h_dim=args.h_dim,
        state_in_dim=state_in_dim,
        event_in_dim=event_in_dim,
        hidden=args.hidden,
    )
    # Modules must live on the same device as the frozen KGE embeddings and
    # the data tensors (kge.device); training + eval move data there but not
    # the module parameters, so place them here at construction.
    rssm = RSSM(rssm_cfg).to(kge.device)
    decoder = EdgeProbabilityHead(
        z_dim=z_dim,
        kge_dim=kge_dim,
        hidden=args.hidden,
    ).to(kge.device)

    # ------------------------------------------------------------------
    # 5. Train Layer-4 (RSSM + decoder)
    # ------------------------------------------------------------------
    print(f"Training Layer-4 RSSM + decoder for {args.epochs} epochs ...")
    t0 = time.time()
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        kl_weight=args.kl_weight,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
        bptt_window=args.bptt_window,
        consistency_weight=args.consistency_weight,
        free_bits=args.free_bits,
        kl_alpha=args.kl_alpha,
    )
    loss_history = train_layer4(
        traj=traj,
        kge=kge,
        state_enc=state_enc,
        event_enc=event_enc,
        rssm=rssm,
        decoder=decoder,
        cfg=train_cfg,
    )
    train_time = time.time() - t0
    print(
        f"  Training done in {train_time:.1f}s. "
        f"Loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"
    )

    # ------------------------------------------------------------------
    # 6. Induce ontology from the full cumulative graph
    # ------------------------------------------------------------------
    ref_graph = traj.at(len(traj))  # last snapshot
    onto = Ontology.induce(ref_graph)

    # ------------------------------------------------------------------
    # 7. Run evals
    # ------------------------------------------------------------------
    print("Running P1.A: filtered MRR / Hits@k ...")
    t0 = time.time()
    mrr_result = eval_single_step(
        traj,
        kge,
        state_enc,
        event_enc,
        rssm,
        decoder,
        candidates_per_query=args.candidates,
    )
    mrr_time = time.time() - t0
    print(
        f"  MRR={mrr_result['MRR']:.4f}  Hits@1={mrr_result['Hits@1']:.4f}"
        f"  Hits@10={mrr_result['Hits@10']:.4f}  n={mrr_result['n']}"
        f"  ({mrr_time:.1f}s)"
    )

    # Ablation: zero the dynamics latent at decode time.  The gap between this
    # and the full MRR is the contribution of the temporal latent over and
    # above the frozen ComplEx embeddings the decoder also consumes — the
    # honest test that the world model, not just the KGE, is doing the work.
    print("Running P1.A ablation: latent zeroed ...")
    mrr_no_latent = eval_single_step(
        traj,
        kge,
        state_enc,
        event_enc,
        rssm,
        decoder,
        candidates_per_query=args.candidates,
        zero_latent=True,
    )
    print(
        f"  MRR(no-latent)={mrr_no_latent['MRR']:.4f}  "
        f"delta={mrr_result['MRR'] - mrr_no_latent['MRR']:+.4f}"
    )

    # Standard TKG protocol: all-entity time-aware-filtered ranking (the number
    # comparable to RE-GCN / TiRGN). Slower; only on request.
    mrr_full = None
    if args.full_ranking:
        print("Running P1.A FULL RANKING (all entities, time-aware filtered) ...")
        t0 = time.time()
        mrr_full = eval_single_step_full_ranking(
            traj,
            kge,
            state_enc,
            event_enc,
            rssm,
            decoder,
        )
        print(
            f"  MRR(full-rank)={mrr_full['MRR']:.4f}  "
            f"Hits@1={mrr_full['Hits@1']:.4f}  Hits@10={mrr_full['Hits@10']:.4f}"
            f"  n={mrr_full['n']}  ({time.time() - t0:.1f}s)"
        )

    print("Running P1.B: rollout coherence ...")
    t0 = time.time()
    coh_result = eval_rollout_coherence(
        traj,
        kge,
        state_enc,
        event_enc,
        rssm,
        decoder,
        onto,
        ks=(3, 5, 10),
        edges_per_step=args.edges_per_step,
    )
    coh_time = time.time() - t0
    print(
        f"  Coherence k=3:{coh_result[3]:.4f}  k=5:{coh_result[5]:.4f}"
        f"  k=10:{coh_result[10]:.4f}  ({coh_time:.1f}s)"
    )

    print("Running P1.C: latency ...")
    t0 = time.time()
    lat_result = eval_latency(
        traj,
        kge,
        state_enc,
        event_enc,
        rssm,
        decoder,
        tier2_call=None,  # no Tier-2 baseline wired here; can be added post-run
        n_queries=args.latency_queries,
    )
    lat_time = time.time() - t0
    print(f"  Layer4 latency: {lat_result['layer4_ms']:.3f} ms/query  ({lat_time:.1f}s)")

    # P2.5: multi-step world-model rollout fidelity (the curve that shows
    # whether imagined dynamics stay faithful as the horizon grows).
    rollout = None
    if args.rollout_eval:
        print("Running P2.5: multi-step rollout fidelity ...")
        t0 = time.time()
        rollout = eval_rollout_fidelity(
            traj,
            kge,
            state_enc,
            event_enc,
            rssm,
            decoder,
            horizons=(1, 3, 5, 10),
        )
        print(f"  Rollout Hits@10 by horizon: {rollout}  ({time.time() - t0:.1f}s)")

    # P5: planning demo — the RSSMWorldModel lets the planner imagine futures
    # with learned dynamics.  Observe a real start state, imagine the next k
    # ticks under their actual events, and report the per-step proxy reward
    # (fraction of decoded top edges the ontology admits) — a sound signal a
    # goal-directed planner maximises.
    planning = None
    if args.planning_demo:
        print("Running P5: RSSMWorldModel imagined rollout ...")
        t0 = time.time()
        wm = RSSMWorldModel(
            rssm,
            decoder,
            state_enc,
            event_enc,
            kge,
            ontology=onto,
            edges_per_step=args.edges_per_step,
        )
        # Observe a real POPULATED late state (snapshot[0] is the empty
        # initial state) and imagine its actual next ticks forward.
        start_t = max(1, len(traj) - 9)
        start = wm.observe(traj.at(start_t))
        plan = [traj.event_at(t) for t in range(start_t, min(start_t + 8, len(traj)))]
        wm_traj = wm.rollout(start, plan)
        planning = {
            "horizon": len(plan),
            "stepwise_ontology_reward": [round(r, 4) for r in wm_traj.rewards],
            "mean_reward": round(float(np.mean(wm_traj.rewards)), 4) if wm_traj.rewards else 0.0,
        }
        print(
            f"  imagined {len(plan)}-step rollout; ontology reward/step="
            f"{planning['stepwise_ontology_reward']}  "
            f"mean={planning['mean_reward']}  ({time.time() - t0:.1f}s)"
        )

    # ------------------------------------------------------------------
    # 8. Write report
    # ------------------------------------------------------------------
    report = {
        "dataset": {
            "root": args.root,
            "split": args.split,
            "limit_days": args.limit_days,
            "trajectory_steps": len(traj),
            "unique_triples": len(all_triples),
            "n_entities": len(kge.ent),
            "n_relations": len(kge.rel),
            "n_event_types": event_in_dim,
        },
        "kge": {
            "dim": args.dim,
            "epochs": args.kge_epochs,
            "device": args.device,
            "train_time_s": round(kge_time, 2),
        },
        "layer4": {
            "z_categories": args.z_categories,
            "z_per_state": args.z_per_state,
            "h_dim": args.h_dim,
            "hidden": args.hidden,
            "epochs": args.epochs,
            "lr": args.lr,
            "kl_weight": args.kl_weight,
            "neg_per_pos": args.neg_per_pos,
            "train_time_s": round(train_time, 2),
            "loss_first": round(loss_history[0], 4) if loss_history else None,
            "loss_last": round(loss_history[-1], 4) if loss_history else None,
        },
        "p1a_mrr": {k: round(v, 4) if isinstance(v, float) else v for k, v in mrr_result.items()},
        "p1a_mrr_no_latent": {
            k: round(v, 4) if isinstance(v, float) else v for k, v in mrr_no_latent.items()
        },
        "p1a_mrr_full_ranking": (
            {k: round(v, 4) if isinstance(v, float) else v for k, v in mrr_full.items()}
            if mrr_full is not None
            else None
        ),
        "p2_rollout_hits10": (
            {str(k): round(v, 4) for k, v in rollout.items()} if rollout is not None else None
        ),
        "p5_planning": planning,
        "event_encoder": args.event_encoder,
        "bptt_window": args.bptt_window,
        "consistency_weight": args.consistency_weight,
        "p1b_coherence": {str(k): round(v, 4) for k, v in coh_result.items()},
        "p1c_latency": {
            "layer4_ms": round(lat_result["layer4_ms"], 4),
            "tier2_ms": lat_result["tier2_ms"],
            "ratio": lat_result["ratio"],
            "n_queries": args.latency_queries,
        },
        "total_time_s": round(time.time() - t_total, 2),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {out}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
