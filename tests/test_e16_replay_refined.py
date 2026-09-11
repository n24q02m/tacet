"""E18-W2 — the classification replay must decide exactly per the locked rule.

Three behaviours, one per locked decision branch:

* ``REFINED-GUARD-SUBSUMES`` — a faithful 22-cell recording (pure self-loops
  only in the control arm, clean forbid arm, true compositions base-only,
  totals reconciling with the published aggregates);
* ``MIXED_RECURSIVE_RESIDUE`` — any control rule that carries the target in
  a mixed body survives the refined guard (NEUTRAL + follow-up);
* ``TECHNICAL_EXTERNAL`` — a recording whose totals disagree with the
  published aggregates can support no claim at all.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from replay_e16_refined import PUBLISHED, replay


def _rule_true() -> str:
    return "syn:q2_directors_of_movies_acted_in_by<=~starred_actors.directed_by"


def _rule_pure(inverted: bool = False) -> str:
    t = "q2_directors_of_movies_acted_in_by"
    first = f"~{t}" if inverted else t
    return f"syn:{t}<={first}.{t}"


def _rule_mixed() -> str:
    t = "q2_directors_of_movies_acted_in_by"
    return f"syn:{t}<={t}.starred_actors"


def _grid(control_rules: list[str], forbid_rules: list[str]) -> dict:
    """A two-cell grid whose counts match the published aggregates when the
    caller supplies the canonical rule set; callers perturb it to hit the
    residue and mismatch branches."""
    cells = []
    for i, (c_rules, f_rules) in enumerate([(control_rules, forbid_rules)] * 2):
        cells.append(
            {
                "slug": f"vendor/model_hop2_lim300_s{i}",
                "seed": i,
                "gamma": 0.5,
                "control": {
                    "slug": cells_i_slug(i),
                    "seed": i,
                    "synthesised_rules": c_rules,
                    "true_rule_installed": _rule_true() in c_rules,
                },
                "forbid": {
                    "slug": cells_i_slug(i),
                    "seed": i,
                    "synthesised_rules": f_rules,
                    "true_rule_installed": _rule_true() in f_rules,
                },
                "savings_identical": True,
            }
        )
    return {"cells": cells}


def cells_i_slug(i: int) -> str:
    return f"vendor/model_hop2_lim300_s{i}"


class TestLockedDecisionRule(unittest.TestCase):
    def test_subsumes_on_faithful_grid(self) -> None:
        # Canonical rule census: 14 pure-self-loop junk rules across 7 cells
        # is the published control shape at full scale; at test scale the
        # published constants are patched to the grid's own totals.
        control = [_rule_pure(), _rule_pure(True), _rule_true()]
        forbid = [_rule_true()]
        grid = _grid(control, forbid)
        published = {
            "true_rule_installs": {"control": 2, "forbid": 2},
            "self_referential_rules": {"control": 4, "forbid": 0},
            "cells_installing_self_referential": {"control": 2, "forbid": 0},
            "other_rules": {"control": 0, "forbid": 0},
        }
        old = dict(PUBLISHED)
        PUBLISHED.clear()
        PUBLISHED.update(published)
        try:
            out = replay(grid, "0" * 64)
        finally:
            PUBLISHED.clear()
            PUBLISHED.update(old)
        self.assertEqual(out["decision"], "REFINED-GUARD-SUBSUMES")
        self.assertTrue(out["refined_totals"]["forbid"]["by_shape"]["pure_self_loop"] == 0)

    def test_mixed_recursive_residue(self) -> None:
        # One control rule carries the target in a MIXED body: the refined
        # guard cannot remove it, so the locked rule demands the residue
        # branch (NEUTRAL + follow-up), never a silent subsumes claim.
        control = [_rule_pure(), _rule_true(), _rule_mixed()]
        forbid = [_rule_true()]
        grid = _grid(control, forbid)
        published = {
            "true_rule_installs": {"control": 2, "forbid": 2},
            "self_referential_rules": {"control": 4, "forbid": 0},
            "cells_installing_self_referential": {"control": 2, "forbid": 0},
            "other_rules": {"control": 0, "forbid": 0},
        }
        old = dict(PUBLISHED)
        PUBLISHED.clear()
        PUBLISHED.update(published)
        try:
            out = replay(grid, "0" * 64)
        finally:
            PUBLISHED.clear()
            PUBLISHED.update(old)
        self.assertEqual(out["decision"], "MIXED_RECURSIVE_RESIDUE")

    def test_per_cell_count_mismatch_is_technical_external(self) -> None:
        # The artifact records the published junk count NEXT TO each cell's
        # rule list. If the recorded per-cell count disagrees with what the
        # list actually classifies to, an aggregate-level subsumes claim
        # would be vacuous: the locked rule demands TECHNICAL_EXTERNAL.
        control = [_rule_pure(), _rule_true()]
        forbid = [_rule_true()]
        grid = _grid(control, forbid)
        for cell in grid["cells"]:
            # List carries 1 junk rule, recording claims 2.
            cell["control"]["self_referential_rules"] = 2
            cell["control"]["other_rules"] = 0
            cell["forbid"]["self_referential_rules"] = 0
            cell["forbid"]["other_rules"] = 0
        published = {
            "true_rule_installs": {"control": 2, "forbid": 2},
            "self_referential_rules": {"control": 4, "forbid": 0},
            "cells_installing_self_referential": {"control": 2, "forbid": 0},
            "other_rules": {"control": 0, "forbid": 0},
        }
        old = dict(PUBLISHED)
        PUBLISHED.clear()
        PUBLISHED.update(published)
        try:
            out = replay(grid, "0" * 64)
        finally:
            PUBLISHED.clear()
            PUBLISHED.update(old)
        self.assertEqual(out["decision"], "TECHNICAL_EXTERNAL")
        self.assertEqual(len(out["per_cell_reconciliation"]), 2)
        self.assertIn("per-cell count mismatches", out["decision_why"])

    def test_recording_mismatch_is_technical_external(self) -> None:
        # A recording that lost its junk rules reconciles with nothing: the
        # locked rule demands TECHNICAL_EXTERNAL, never a boundary claim.
        control = [_rule_true()]
        forbid = [_rule_true()]
        grid = _grid(control, forbid)
        old = dict(PUBLISHED)
        PUBLISHED.clear()
        PUBLISHED.update(
            {
                "true_rule_installs": {"control": 2, "forbid": 2},
                "self_referential_rules": {"control": 14, "forbid": 0},
                "cells_installing_self_referential": {"control": 7, "forbid": 0},
                "other_rules": {"control": 0, "forbid": 0},
            }
        )
        try:
            out = replay(grid, "0" * 64)
        finally:
            PUBLISHED.clear()
            PUBLISHED.update(old)
        self.assertEqual(out["decision"], "TECHNICAL_EXTERNAL")
        self.assertIn("recording mismatch", out["decision_why"])


if __name__ == "__main__":
    unittest.main()
