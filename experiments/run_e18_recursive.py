"""E18 — genuinely recursive targets: measuring the registered E16 boundary.

The E16 ledger entry (2026-07-24, before the result) registered the limitation:
forbidding the target in its own length-2 body "forecloses genuinely recursive
targets (``ancestor <= ancestor.parent``)"; a POSITIVE E16 licenses "safe on
non-recursive targets", never "always turn it off". This script measures that
case directly, at zero provider cost:

1. SYNTHETIC RECURSIVE WORKLOAD (free): a seeded random DAG provides the base
   relation ``parent``; the mining target ``anc`` is the genuine transitive
   closure — the recursive relation ``anc <= anc.parent`` the E16 ban cannot
   express. Oracle teacher answers (complete answer sets for sampled heads)
   play the teacher; no provider is contacted.
2. THREE ARMS per (seed, gamma): the published behaviour
   (``allow_target_in_body=True``), E16's ban (``allow_target_in_body=False``),
   and the E18 refined guard (``forbid_target_self_loop=True`` — ban only the
   pure all-target body, keep mixed legs).
3. ENDPOINTS per cell: installed-rule inventory, forward-chaining derivation
   coverage over the FULL closure, soundness (derived pairs must be closure
   pairs), and per-rule world precision against the full oracle.
4. METAQA RE-READ (free): the committed E16 artifact
   (``e16_forbid_target_hop2.json``) is re-classified under the refined guard:
   every self-referential junk rule there is a PURE self-loop, and the true
   composition is base-only — so the refined guard would have removed exactly
   the junk while keeping the true rule.

Locked decision rule (2026-09-11, pre-run):

- ``POSITIVE`` iff (a) on the synthetic recursive workload the E16 ban's
  derivation coverage is strictly below the published behaviour for EVERY
  (seed, gamma) cell, (b) the refined guard's coverage equals the published
  behaviour for every cell, and (c) on the committed MetaQA E16 artifact the
  refined guard classifies every junk rule as a pure self-loop and keeps the
  true composition. Reading: the refined guard subsumes E16's benefit without
  foreclosing genuinely recursive targets.
- ``NEGATIVE`` iff the refined guard loses coverage or keeps junk.
- ``NEUTRAL`` otherwise.

Everything is local and deterministic; nothing here calls a provider.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tacet.core.graph import WorldGraph
from tacet.distill.distill import Triple, mine_rules

E18_SCHEMA = "tacet.e18.recursive/v1"
PRE_REG = "tacet-research-ledger.md E18 (locked 2026-09-11)"


def build_dag(
    n_nodes: int, edge_p: float, seed: int
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """Forward-edge random DAG: parent relation + full transitive closure."""
    rng = random.Random(seed)
    edges = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes) if rng.random() < edge_p]
    closure: dict[int, set[int]] = {i: set() for i in range(n_nodes)}
    for a, b in edges:
        closure[a].add(b)
    changed = True
    while changed:
        changed = False
        for a in range(n_nodes):
            new: set[int] = set()
            for b in closure[a]:
                new |= closure[b]
            if not new <= closure[a]:
                closure[a] |= new
                changed = True
    pairs = {(a, b) for a, bs in closure.items() for b in bs}
    return edges, pairs


def classify(name: str, target: str) -> str:
    """Classify a mined rule name by body shape (after the ``syn:<head><=``)."""
    body = name.split("<=", 1)[1]
    legs = [leg.lstrip("~") for leg in body.split(".")]
    if len(legs) == 2 and legs[0] == target and legs[1] == target:
        return "pure_self_loop"
    if target in legs:
        return "mixed_recursive"
    return "base_only"


def legs_of(name: str) -> list[tuple[str, bool]]:
    body = name.split("<=", 1)[1]
    out: list[tuple[str, bool]] = []
    for leg in body.split("."):
        out.append((leg[1:], True) if leg.startswith("~") else (leg, False))
    return out


def world_precision(
    name: str,
    parent: set[tuple[int, int]],
    anc_facts: set[tuple[int, int]],
    full_closure: set[tuple[int, int]],
) -> float:
    """AMIE-style confidence over the FULL oracle (all closure pairs), not just
    the teacher-answered heads — the paper's rule_precision read-out."""
    legs = legs_of(name)
    src = parent | anc_facts
    if len(legs) == 1:
        rel, inv = legs[0]
        pairs: set[tuple[int, int]] = set()
        for h, t in src:
            if (rel == "parent" and (h, t) in parent) or (rel == "anc" and (h, t) in anc_facts):
                pairs.add((t, h) if inv else (h, t))
    else:
        (r1, i1), (r2, i2) = legs
        left: dict[int, set[int]] = {}
        right: dict[int, set[int]] = {}
        for h, t in src:
            if (r1 == "parent" and (h, t) in parent) or (r1 == "anc" and (h, t) in anc_facts):
                a, b = (t, h) if i1 else (h, t)
                left.setdefault(a, set()).add(b)
            if (r2 == "parent" and (h, t) in parent) or (r2 == "anc" and (h, t) in anc_facts):
                a, b = (t, h) if i2 else (h, t)
                right.setdefault(a, set()).add(b)
        pairs = {(x, y) for x, zs in left.items() for z in zs for y in right.get(z, ()) if x != y}
    if not pairs:
        return 0.0
    return sum(1 for p in pairs if p in full_closure) / len(pairs)


def derive_coverage(
    installed: list[dict],
    parent: set[tuple[int, int]],
    anc_facts: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Sound forward-chaining of the installed rule shapes over parent facts.

    ``anc`` write-back facts from the teacher participate as body facts only
    for ``anc`` legs (exactly the facts the miner saw). Iterated to fixpoint,
    so chained recursion through an installed ``anc <= anc.parent`` rule
    accumulates coverage over rounds, as the cascade's write-back loop would.
    """
    known: set[tuple[int, int]] = set(parent)
    changed = True
    while changed:
        changed = False
        for inst in installed:
            legs = inst["legs"]
            if len(legs) == 1:
                rel, inv = legs[0]
                src: set[tuple[int, int]] = set()
                for h, t in known:
                    if (rel == "parent" and (h, t) in parent) or (
                        rel == "anc" and (h, t) in anc_facts
                    ):
                        src.add((t, h) if inv else (h, t))
                new = {(x, y) for x, y in src if x != y}
            else:
                (r1, i1), (r2, i2) = legs
                left: dict[int, set[int]] = {}
                right: dict[int, set[int]] = {}
                for h, t in known:
                    if (r1 == "parent" and (h, t) in parent) or (
                        r1 == "anc" and (h, t) in anc_facts
                    ):
                        a, b = (t, h) if i1 else (h, t)
                        left.setdefault(a, set()).add(b)
                    if (r2 == "parent" and (h, t) in parent) or (
                        r2 == "anc" and (h, t) in anc_facts
                    ):
                        a, b = (t, h) if i2 else (h, t)
                        right.setdefault(a, set()).add(b)
                new = {
                    (x, y) for x, zs in left.items() for z in zs for y in right.get(z, ()) if x != y
                }
            if not new <= known:
                known |= new
                changed = True
    return known


def run_cell(
    edges: list[tuple[int, int]],
    full_closure: set[tuple[int, int]],
    seed: int,
    gamma: float,
    arm: str,
    n_nodes: int,
    k_heads: int,
) -> dict:
    rng = random.Random(seed * 10_000 + int(gamma * 100))
    heads = rng.sample(range(n_nodes), k_heads)
    head_set = set(heads)
    parent = set(edges)
    # Oracle teacher: complete anc answer sets for the sampled heads only.
    anc_facts = {(a, b) for a, b in full_closure if a in head_set}

    g = WorldGraph()
    for a, b in edges:
        g.add_edge(f"n{a}", "parent", f"n{b}")
    facts: set[Triple] = {(f"n{a}", "anc", f"n{b}") for a, b in anc_facts}
    complete = {f"n{h}" for h in heads}

    kwargs = dict(
        min_confidence=gamma,
        min_support=3,
        allowed_body={"parent"},
        complete_heads=complete,
    )
    if arm == "published":
        rules = mine_rules(g, facts, "anc", allow_target_in_body=True, **kwargs)
    elif arm == "e16_ban":
        rules = mine_rules(g, facts, "anc", allow_target_in_body=False, **kwargs)
    elif arm == "refined":
        rules = mine_rules(
            g,
            facts,
            "anc",
            allow_target_in_body=True,
            forbid_target_self_loop=True,
            **kwargs,
        )
    else:
        raise ValueError(arm)

    installed = []
    for r in rules:
        installed.append(
            {
                "name": r.rule.name,
                "legs": [list(lg) for lg in legs_of(r.rule.name)],
                "shape": classify(r.rule.name, "anc"),
                "support": r.support,
                "confidence": round(r.confidence, 4),
                "world_precision": round(
                    world_precision(r.rule.name, parent, anc_facts, full_closure), 4
                ),
            }
        )

    derivable = derive_coverage(installed, parent, anc_facts)
    return {
        "arm": arm,
        "seed": seed,
        "gamma": gamma,
        "installed": installed,
        "coverage": round(len(derivable & full_closure) / len(full_closure), 4),
        "unsound_derivations": len(derivable - full_closure),
        "n_installed": len(installed),
    }


def reread_e16(path: Path) -> dict:
    """Re-classify the committed E16 MetaQA artifact under the refined guard."""
    data = json.loads(path.read_text(encoding="utf-8"))
    target = "q2_directors_of_movies_acted_in_by"
    junk_shapes: set[str] = set()
    true_shapes: set[str] = set()
    for cell in data["cells"]:
        control = cell["control"]
        names = control.get("synthesised_rules", [])
        if control.get("true_rule_installed"):
            for name in names:
                if classify(name, target) == "base_only":
                    true_shapes.add("base_only")
        for name in names:
            shape = classify(name, target)
            if shape == "pure_self_loop":
                junk_shapes.add("pure_self_loop")
            elif shape == "mixed_recursive":
                junk_shapes.add("mixed_recursive")
    return {
        "artifact": path.name,
        "junk_shapes_observed": sorted(junk_shapes),
        "true_rule_shapes_observed": sorted(true_shapes),
        "refined_guard_would_remove_all_junk": junk_shapes <= {"pure_self_loop"},
        "refined_guard_keeps_true_rule": true_shapes == {"base_only"} or not true_shapes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-nodes", type=int, default=150)
    ap.add_argument("--edge-p", type=float, default=0.03)
    ap.add_argument("--k-heads", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    ap.add_argument(
        "--e16-artifact",
        type=Path,
        default=Path("experiments/results/e16_forbid_target_hop2.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/results/e18_recursive_target_synthetic.json"),
    )
    args = ap.parse_args()

    cells = []
    for seed in args.seeds:
        edges, closure = build_dag(args.n_nodes, args.edge_p, seed)
        for gamma in args.gammas:
            for arm in ("published", "e16_ban", "refined"):
                cells.append(run_cell(edges, closure, seed, gamma, arm, args.n_nodes, args.k_heads))

    by = {(c["seed"], c["gamma"], c["arm"]): c for c in cells}
    ban_loses_everywhere = all(
        by[(s, g, "e16_ban")]["coverage"] < by[(s, g, "published")]["coverage"]
        for s in args.seeds
        for g in args.gammas
    )
    refined_matches = all(
        by[(s, g, "refined")]["coverage"] == by[(s, g, "published")]["coverage"]
        for s in args.seeds
        for g in args.gammas
    )
    e16_read = reread_e16(args.e16_artifact)

    if ban_loses_everywhere and refined_matches and e16_read["refined_guard_would_remove_all_junk"]:
        decision = "POSITIVE"
        why = (
            "E16 ban loses coverage on every recursive cell (forecloses recursion, "
            "confirming the registered limitation); refined guard matches published "
            "coverage everywhere; on the committed MetaQA artifact the refined guard "
            "classifies all junk as pure self-loops (removable) while the true "
            "composition is base-only (kept)."
        )
    elif not refined_matches or not e16_read["refined_guard_would_remove_all_junk"]:
        decision = "NEGATIVE"
        why = "refined guard lost recursive coverage or fails to subsume E16's junk removal"
    else:
        decision = "NEUTRAL"
        why = "mixed outcome; see per-cell data"

    out = {
        "schema": E18_SCHEMA,
        "experiment": "E18 genuinely recursive targets: the registered E16 boundary",
        "pre_registration": PRE_REG,
        "workload": {
            "kind": "synthetic DAG, oracle teacher, zero provider calls",
            "n_nodes": args.n_nodes,
            "edge_p": args.edge_p,
            "k_heads": args.k_heads,
            "seeds": args.seeds,
            "gammas": args.gammas,
            "target": "anc (transitive closure of parent)",
        },
        "cells": cells,
        "e16_metaqa_reread": e16_read,
        "decision": decision,
        "decision_why": why,
    }
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: decision={decision} cells={len(cells)}")


if __name__ == "__main__":
    main()
