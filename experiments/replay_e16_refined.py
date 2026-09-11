"""E18-W2 — classification replay of the committed E16 MetaQA grid under the
refined pure-self-loop guard (ledger E18-W2, locked 2026-09-11).

Reads the committed E16 artifact, re-classifies every ``synthesised_rules``
entry of all 22 cells (control + forbid arms) under the refined taxonomy
(``pure_self_loop`` / ``mixed_recursive`` / ``base_only``), reconciles the
totals with the published aggregates and emits a per-cell decision artifact.

The GENERATIVE half of E18-W2 (re-mining the cells under
``forbid_target_self_loop=True``) requires the uncommitted E11 recorded answer
sessions and stays ``TECHNICAL_EXTERNAL`` while they are absent; see the
ledger entry. This script implements half (i) only and claims nothing about
half (ii).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aggregate_e16_forbid import DEFAULT_COMPOSITION
from aggregate_e16_forbid import classify as classify_published
from run_e18_recursive import classify

SCHEMA = "tacet.e18.e16replay/v1"
PRE_REG = "tacet-research-ledger.md E18-W2 (locked 2026-09-11)"
TARGET = "q2_directors_of_movies_acted_in_by"
ARMS = ("control", "forbid")
SHAPES = ("pure_self_loop", "mixed_recursive", "base_only")

# Published aggregates of the committed E16 grid (the numbers the artifact
# itself carries; the replay must reconcile with them exactly).
PUBLISHED = {
    "true_rule_installs": {"control": 10, "forbid": 10},
    "self_referential_rules": {"control": 14, "forbid": 0},
    "cells_installing_self_referential": {"control": 7, "forbid": 0},
    "other_rules": {"control": 0, "forbid": 0},
}


def _body_matches_composition(rule: str) -> bool:
    """True iff the rule body is exactly the published composition shape."""
    head, _, body = rule.partition("<=")
    body = body or head
    return all(rel in body for rel in DEFAULT_COMPOSITION)


def replay(data: dict, recording_sha256: str) -> dict:
    """Classify every rule of every cell and decide per the locked rule."""
    rows: list[dict] = []
    refined = {
        arm: {
            "rules": 0,
            "by_shape": {shape: 0 for shape in SHAPES},
            "cells_with_target_rule": 0,
            "true_installed_cells": 0,
            "true_installed_cells_with_base_only_rule": 0,
        }
        for arm in ARMS
    }
    parsed_published = {
        "true_rule_installs": {},
        "self_referential_rules": {},
        "cells_installing_self_referential": {},
        "other_rules": {},
    }
    per_cell_mismatches: list[dict] = []
    for arm in ARMS:
        true_cells = self_ref_rules = self_ref_cells = other_rules = 0
        for cell in data["cells"]:
            armdata = cell[arm]
            names = armdata.get("synthesised_rules", [])
            true_installed, self_ref, other = classify_published(names, DEFAULT_COMPOSITION)
            true_cells += int(true_installed)
            self_ref_rules += self_ref
            self_ref_cells += int(self_ref > 0)
            other_rules += other

            # Per-cell count reconciliation: the artifact stores the
            # published junk/other counts next to the rule lists; the lists
            # and the counts must agree cell-by-cell, or an aggregate-level
            # subsumes claim would be vacuous (under/over-count hiding in
            # individual cells).
            recorded_self_ref = armdata.get("self_referential_rules")
            recorded_other = armdata.get("other_rules")
            recorded_true = armdata.get("true_rule_installed")
            cell_mismatch = (
                (recorded_self_ref is not None and recorded_self_ref != self_ref)
                or (recorded_other is not None and recorded_other != other)
                or (recorded_true is not None and bool(recorded_true) != bool(true_installed))
            )
            if cell_mismatch:
                per_cell_mismatches.append(
                    {
                        "slug": cell["slug"],
                        "seed": cell["seed"],
                        "arm": arm,
                        "recorded": {
                            "self_referential_rules": recorded_self_ref,
                            "other_rules": recorded_other,
                            "true_rule_installed": recorded_true,
                        },
                        "classified": {
                            "self_referential_rules": self_ref,
                            "other_rules": other,
                            "true_rule_installed": bool(true_installed),
                        },
                    }
                )

            target_rules_in_cell = 0
            base_only_in_cell = False
            for name in names:
                shape = classify(name, TARGET)
                rows.append(
                    {
                        "slug": cell["slug"],
                        "seed": cell["seed"],
                        "gamma": cell["gamma"],
                        "arm": arm,
                        "rule": name,
                        "shape": shape,
                        "is_true_composition": _body_matches_composition(name),
                    }
                )
                refined[arm]["rules"] += 1
                refined[arm]["by_shape"][shape] += 1
                if shape in ("pure_self_loop", "mixed_recursive"):
                    target_rules_in_cell += 1
                if shape == "base_only":
                    base_only_in_cell = True
            if target_rules_in_cell:
                refined[arm]["cells_with_target_rule"] += 1
            if armdata.get("true_rule_installed"):
                refined[arm]["true_installed_cells"] += 1
                if base_only_in_cell:
                    refined[arm]["true_installed_cells_with_base_only_rule"] += 1
        parsed_published["true_rule_installs"][arm] = true_cells
        parsed_published["self_referential_rules"][arm] = self_ref_rules
        parsed_published["cells_installing_self_referential"][arm] = self_ref_cells
        parsed_published["other_rules"][arm] = other_rules

    mismatched = {
        key: {"published": PUBLISHED[key], "parsed": parsed_published[key]}
        for key in PUBLISHED
        if PUBLISHED[key] != parsed_published[key]
    }
    aggregate_clean = not mismatched and not per_cell_mismatches
    true_composition_rules = [row for row in rows if row["is_true_composition"]]
    true_all_base_only = all(row["shape"] == "base_only" for row in true_composition_rules)
    forbid_clean = (
        refined["forbid"]["by_shape"]["pure_self_loop"] == 0
        and refined["forbid"]["by_shape"]["mixed_recursive"] == 0
    )
    control_no_mixed = refined["control"]["by_shape"]["mixed_recursive"] == 0

    if not aggregate_clean or not true_all_base_only:
        decision = "TECHNICAL_EXTERNAL"
        why = (
            "recording mismatch: parsed totals disagree with the published "
            f"aggregates {mismatched}; per-cell count mismatches: "
            f"{per_cell_mismatches} (true-composition rules all base-only: "
            f"{true_all_base_only}); the artifact cannot support the replay "
            "claim. A data-availability failure, never a boundary claim."
        )
    elif not control_no_mixed or not forbid_clean:
        decision = "MIXED_RECURSIVE_RESIDUE"
        control_mixed = refined["control"]["by_shape"]["mixed_recursive"]
        forbid_target = (
            refined["forbid"]["by_shape"]["pure_self_loop"]
            + refined["forbid"]["by_shape"]["mixed_recursive"]
        )
        why = (
            "target-in-body residue beyond pure self-loops "
            f"(control mixed_recursive={control_mixed}, "
            f"forbid target rules={forbid_target}); "
            "NEUTRAL + follow-up entry required."
        )
    else:
        decision = "REFINED-GUARD-SUBSUMES"
        why = (
            "every control target-containing rule is a pure self-loop, the "
            "forbid arm is clean, every true composition is base-only and "
            "all totals reconcile with the published aggregates at both the "
            "aggregate and the per-cell level."
        )

    return {
        "schema": SCHEMA,
        "experiment": (
            "E18-W2 refined-guard classification replay of the 22 recorded E16 MetaQA cells"
        ),
        "pre_registration": PRE_REG,
        "recording": {
            "artifact": "e16_forbid_target_hop2.json",
            "sha256": recording_sha256,
        },
        "generative_half": (
            "TECHNICAL_EXTERNAL: E11 recorded answer sessions are not committed "
            "and not present on this machine; registered in the ledger prereg."
        ),
        "cells": rows,
        "per_cell_reconciliation": per_cell_mismatches,
        "refined_totals": refined,
        "published": PUBLISHED,
        "parsed_published": parsed_published,
        "decision": decision,
        "decision_why": why,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--e16-artifact",
        type=Path,
        default=Path("experiments/results/e16_forbid_target_hop2.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/results/e16_metaqa_reread_refined_guard.json"),
    )
    args = ap.parse_args()
    raw = args.e16_artifact.read_bytes()
    data = json.loads(raw)
    if len(data["cells"]) != 22:
        raise SystemExit(f"expected the committed 22-cell E16 grid, got {len(data['cells'])} cells")
    out = replay(data, hashlib.sha256(raw).hexdigest())
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: decision={out['decision']} rows={len(out['cells'])}")


if __name__ == "__main__":
    main()
