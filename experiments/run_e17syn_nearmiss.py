"""E17-SYN — synthetic near-miss-path workload: admission-test preflight.

BACKLOG B4, folded into the E17 campaign as a pre-funding task (ledger design
entry ``1b085c11``, .private ``superpower/tacet/tacet-research-ledger.md``).
Answers, under oracle conditions, the question the funded PrivaCI-GDPR run
cannot isolate: does the locked held-out estimator separate near-miss leakage
from a true composition, and how does separation degrade with leak strength?

1. SYNTHETIC WORKLOAD (free): a seeded world plants two length-2 candidate
   bodies sharing the second relation — TRUE ``r_a ∘ r_shared ⇒ target`` holds
   on every supported pair (world precision 1.0) and LEAK ``r_b ∘ r_shared ⇒
   target`` (``r_b`` a declared ontology sibling of ``r_a``) fires on a tunable
   fraction ``theta`` of unseen heads with per-fire correctness ``1 - theta``.
   The oracle teacher (complete gold sets) plays the teacher; no provider is
   contacted.
2. ESTIMATOR under test = the locked E17 machinery: 300-case Zipf-head stream
   split 50/50 by ``sha256(seed:case_id)`` parity (the split function is
   IMPORTED from ``run_e17_admission``, never copied); per-rule validation-half
   precision; the locked margins/thresholds.
3. DUAL VERDICT per (theta, seed) cell: admission classification is computed
   BOTH against the held-out estimate (what E17 can observe when paying) and
   against exact world precision over the full stream (free here, unknown
   there) — every cell reports estimator error next to ground truth.
4. LOCKED DECISION RULE (ledger 1457, operative only once the executing
   session declares the prereg LOCKED): POSITIVE iff held-out Delta-hat >= 0.20
   in every theta <= 0.40 cell AND estimator-based admission matches
   oracle-exact admission in >= 90% of cells; NEGATIVE iff Delta-hat < 0.05 in
   any theta cell; NEUTRAL otherwise. The theta-response curve IS the
   reportable result.

Scope limits (stated pre-execution in the ledger): oracle teacher says nothing
about teacher-noise robustness; single-target workloads only; MetaQA-shaped
Zipf streams only.

Everything is local and deterministic; nothing here calls a provider, and
``main`` refuses to run when a provider endpoint is resolvable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_e17_admission import (  # noqa: E402
    DELTA_NEGATIVE,
    DELTA_POSITIVE,
    KEEP_COMPOSITION_MIN,
    REJECT_COMPOSITION_MAX,
    REJECT_LEAKAGE_MIN,
    _half,
)

E17SYN_SCHEMA = "tacet.e17syn.nearmiss/v1"

#: Design-entry ledger commit (.private superpower/tacet/tacet-research-ledger.md).
DESIGN_LEDGER_SHA = "1b085c11"

N_CASES = 300
N_ENTITIES = 40
ZIPF_A = 1.4
THETAS = (0.10, 0.25, 0.40, 0.60)
SEEDS = (0, 1, 2)


def _world_rng(theta: float, seed: int) -> np.random.Generator:
    """Deterministic numpy Generator: sha256 of the world key -> PCG64 seed."""
    digest = hashlib.sha256(f"e17syn:{theta}:{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def build_world(theta: float, seed: int) -> dict:
    """Deterministic synthetic world for one (theta, seed) cell.

    Relations are pair-sets (AMIE-style, as in ``run_e18_recursive``). The
    composition ``r_a ∘ r_shared`` is exact on every head it covers; the leak
    body ``r_b ∘ r_shared`` fires on a Bernoulli(theta) per-case draw and a
    fired answer is correct with probability ``1 - theta``. Head entities are
    drawn Zipfian with rejection (MetaQA-shaped: hot heads repeat as distinct
    cases); case ids are stable across the theta sweep for a given seed so
    validation halves line up cell-to-cell.
    """
    rng = _world_rng(theta, seed)
    ents = range(N_ENTITIES)

    # r_shared: functional successor chain; every entity has one image.
    r_shared = {(x, (x * 7 + 3) % N_ENTITIES) for x in ents}
    # r_a: sparse body relation; every (x, y) in r_a yields a true target via
    # r_shared, so composition world precision is exactly 1 on its supports.
    r_a = {(x, (x * 11 + 5) % N_ENTITIES) for x in ents if (x * 11 + 5) % N_ENTITIES != x}
    # r_b: ontology sibling of r_a — semantically adjacent, structurally
    # different; enters only the leak body.
    r_b = {(x, (x * 11 + 6) % N_ENTITIES) for x in ents if (x * 11 + 6) % N_ENTITIES != x}

    comp_answer: dict[int, int] = {}
    for x, y in sorted(r_a):
        for y2, z in r_shared:
            if y == y2 and x != z:
                comp_answer[x] = z  # functional: one true answer per head

    cases: list[dict] = []
    for i in range(N_CASES):
        draw = int(rng.zipf(ZIPF_A))
        while draw > N_ENTITIES:  # rejection keeps the head law truly Zipfian
            draw = int(rng.zipf(ZIPF_A))
        entity = draw - 1
        cid = f"s{seed}c{i:03d}"
        fired = bool(rng.random() < theta)
        ok = bool(rng.random() < (1.0 - theta))
        wrong = (entity + 1 + int(rng.integers(0, N_ENTITIES - 1))) % N_ENTITIES
        cases.append(
            {
                "case_id": cid,
                "entity": entity,
                "gold": comp_answer.get(entity),
                "comp_covers": entity in comp_answer,
                # The composition rule's own answer: the true target where it
                # covers the head, silent otherwise.
                "comp_answer": comp_answer.get(entity),
                "leak_fired": fired,
                # Fired leak answer: gold when the fire is correct; on
                # uncovered heads gold is empty so any answer is wrong.
                "leak_answer": (
                    (comp_answer.get(entity) if ok and entity in comp_answer else wrong)
                    if fired
                    else None
                ),
            }
        )

    return {
        "theta": theta,
        "seed": seed,
        "r_a": r_a,
        "r_b": r_b,
        "r_shared": r_shared,
        "cases": cases,
    }


def run_cell(world: dict) -> dict:
    """Estimator vs oracle-exact endpoints for one (theta, seed) world."""
    theta, seed = world["theta"], world["seed"]
    cases = world["cases"]

    val = [c for c in cases if _half(seed, c["case_id"]) == "validation"]
    fit_n = len(cases) - len(val)

    # Held-out (estimator) precisions — what the E17 admission test observes.
    val_comp = [c for c in val if c["comp_covers"]]
    val_leak = [c for c in val if c["leak_fired"]]
    comp_est = (
        sum(1 for c in val_comp if c["comp_answer"] == c["gold"]) / len(val_comp)
        if val_comp
        else None
    )
    leak_est = (
        sum(1 for c in val_leak if c["leak_answer"] == c["gold"]) / len(val_leak)
        if val_leak
        else None
    )
    delta_hat = comp_est - leak_est if comp_est is not None and leak_est is not None else None

    # Exact (oracle) precisions over the FULL stream.
    all_leak = [c for c in cases if c["leak_fired"]]
    comp_exact = 1.0  # the composition is exact on every supported head by construction
    leak_exact = (
        sum(1 for c in all_leak if c["leak_answer"] == c["gold"]) / len(all_leak)
        if all_leak
        else None
    )
    delta_exact = comp_exact - leak_exact if leak_exact is not None else None

    # Locked classifier thresholds applied to both verdicts.
    def admit_verdict(comp_p: float, leak_p: float | None) -> str:
        if leak_p is None:
            return "NO_LEAK_EVIDENCE"
        if leak_p < REJECT_LEAKAGE_MIN and comp_p >= KEEP_COMPOSITION_MIN:
            return "ADMIT_COMPOSITION_REJECT_LEAK"
        if comp_p < REJECT_COMPOSITION_MAX:
            return "REJECT_COMPOSITION"
        return "DEAD_ZONE"

    est_verdict = admit_verdict(comp_est, leak_est)
    exact_verdict = admit_verdict(comp_exact, leak_exact)

    return {
        "theta": theta,
        "seed": seed,
        "fit_heads": fit_n,
        "validation_heads": len(val),
        "comp_fires_validation": len(val_comp),
        "leak_fires_total": len(all_leak),
        "leak_fires_validation": len(val_leak),
        "estimator": {
            "comp_precision": round(comp_est, 4) if comp_est is not None else None,
            "leak_precision": round(leak_est, 4) if leak_est is not None else None,
            "delta": round(delta_hat, 4) if delta_hat is not None else None,
            "verdict": est_verdict,
        },
        "oracle_exact": {
            "comp_precision": round(comp_exact, 4),
            "leak_precision": round(leak_exact, 4) if leak_exact is not None else None,
            "delta": round(delta_exact, 4) if delta_exact is not None else None,
            "verdict": exact_verdict,
        },
        "estimator_matches_oracle": est_verdict == exact_verdict,
        "leak_world_precision_nominal": round(1.0 - theta, 4),
    }


def decide_campaign(cells: list[dict]) -> tuple[str, str]:
    """Locked decision rule over the (theta, seed) cell grid (ledger 1457)."""
    low_theta = [c for c in cells if c["theta"] <= 0.40]
    all_low_positive = bool(low_theta) and all(
        c["estimator"]["delta"] is not None and c["estimator"]["delta"] >= DELTA_POSITIVE
        for c in low_theta
    )
    match_frac = (
        sum(1 for c in cells if c["estimator_matches_oracle"]) / len(cells) if cells else 0.0
    )

    if all_low_positive and match_frac >= 0.90:
        return "POSITIVE", (
            f"delta_hat >= {DELTA_POSITIVE} in every theta<=0.40 cell; "
            f"estimator/oracle verdict match {match_frac:.0%} >= 90%"
        )
    if any(
        c["estimator"]["delta"] is not None and c["estimator"]["delta"] < DELTA_NEGATIVE
        for c in cells
    ):
        return "NEGATIVE", (
            f"delta_hat < {DELTA_NEGATIVE} in at least one theta cell — the held-out "
            "estimator is blind even under oracle conditions"
        )
    return "NEUTRAL", (
        f"neither locked POSITIVE (verdict match {match_frac:.0%}) nor NEGATIVE "
        "condition met; the theta-response curve is the reportable result"
    )


def assert_provider_unreachable() -> None:
    """Fail closed unless no LLM provider endpoint resolves (E18 discipline).

    Experimental integrity: this preflight is meaningful only as a $0 oracle
    run; any reachable provider is treated as an environment error, not a
    warning.
    """
    hosts = ("api.deepseek.com", "api.openai.com", "api.anthropic.com")
    for host in hosts:
        try:
            socket.getaddrinfo(host, 443)
        except socket.gaierror:
            continue
        raise SystemExit(
            f"E17-SYN abort: provider endpoint {host} is resolvable; this run must "
            "execute with providers unreachable (E18 replay discipline)."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="artifact path (JSON)")
    ap.add_argument(
        "--prereg-lock-sha",
        required=True,
        help="ledger commit SHA of the PREREG LOCK entry (fail-closed: the runner "
        "refuses to execute without a declared lock)",
    )
    args = ap.parse_args()

    assert_provider_unreachable()

    cells = [run_cell(build_world(t, s)) for t in THETAS for s in SEEDS]
    decision, why = decide_campaign(cells)

    # Registered prediction (ledger 1458), checked transparently — advisory
    # only; the locked rule above is the verdict.
    prediction_holds = all(
        c["estimator"]["delta"] is not None
        and c["estimator"]["delta"] >= DELTA_POSITIVE
        and c["estimator_matches_oracle"]
        for c in cells
        if c["theta"] <= 0.40
    )

    def _mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    artifact = {
        "schema": E17SYN_SCHEMA,
        "experiment": "E17-SYN synthetic near-miss-path admission preflight",
        "pre_registration": {
            "design_ledger_sha": DESIGN_LEDGER_SHA,
            "lock_ledger_sha": args.prereg_lock_sha,
            "decision_rule": "ledger 1457 (POSITIVE/NEGATIVE/NEUTRAL, locked)",
            "registered_prediction": "POSITIVE at theta<=0.40, estimator within ~0.05",
        },
        "operating_point": {
            "n_cases": N_CASES,
            "n_entities": N_ENTITIES,
            "zipf_a": ZIPF_A,
            "thetas": list(THETAS),
            "seeds": list(SEEDS),
            "gamma": 0.50,
            "min_support": 3,
            "max_body_length": 2,
            "x_ne_y_guard": True,
            "split_rule": "sha256(seed:case_id) parity, IMPORTED from run_e17_admission",
        },
        "cells": cells,
        "theta_response": [
            {
                "theta": t,
                "mean_delta_hat": _mean(
                    [
                        c["estimator"]["delta"]
                        for c in cells
                        if c["theta"] == t and c["estimator"]["delta"] is not None
                    ]
                ),
                "mean_leak_precision_exact": _mean(
                    [
                        c["oracle_exact"]["leak_precision"]
                        for c in cells
                        if c["theta"] == t and c["oracle_exact"]["leak_precision"] is not None
                    ]
                ),
                "estimator_oracle_match_frac": _mean(
                    [
                        1.0 if c["estimator_matches_oracle"] else 0.0
                        for c in cells
                        if c["theta"] == t
                    ]
                ),
            }
            for t in THETAS
        ],
        "prediction_check": {
            "registered": "POSITIVE at theta<=0.40, estimator within ~0.05 of exact",
            "observed_consistent": prediction_holds,
            "note": "advisory only; the locked decision rule above is the verdict",
        },
        "decision": decision,
        "decision_why": why,
        "cost_usd": 0.0,
        "provider_calls": 0,
    }

    Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"{decision}: {why}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
