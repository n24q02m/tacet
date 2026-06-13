"""Guard: the committed synthetic-grid macros stay in sync with summary.json.

`docs/reproduce.md` claims the paper's numbers regenerate from the committed
artifacts. This pins the regen-equivalence for the macros `experiments/analyze.py`
produces: recomputing the six synthetic-grid cost macros from the committed
`experiments/results/summary.json` must reproduce exactly the values committed in
`paper/results/macros.tex` (the file `main.tex` reads). A drift between the two
(e.g. an edited summary that was never re-analysed, or a hand-edited macro) fails
here rather than shipping a number the data does not support.

The rule-precision / audit / ProofWriter macros are produced by their own
runners (ProofWriter additionally needs the uncommitted dataset), so they are
out of scope for this self-contained check.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "experiments" / "results" / "summary.json"
MACROS = ROOT / "paper" / "results" / "macros.tex"


def _macro_values(text: str) -> dict[str, str]:
    return dict(re.findall(r"\\(?:renew|new)command\{\\(\w+)\}\{([^}]*)\}", text))


class TestMacrosReproducible(unittest.TestCase):
    def test_cost_macros_match_committed_summary(self) -> None:
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        e1 = summary["E1"]
        ak, llm, cache = e1["tacet"], e1["llm_only"], e1["cache_cascade"]
        seeds = summary.get("_meta", {}).get("seeds")
        # the exact formulas analyze.py uses to emit the synthetic-grid macros
        expected = {
            "akCost": f"{ak['cost']['mean']:.2f}",
            "akAcc": f"{ak['accuracy']['mean']:.3f}",
            "llmCost": f"{llm['cost']['mean']:.1f}",
            "cacheCost": f"{cache['cost']['mean']:.1f}",
            "costFactor": f"{llm['cost']['mean'] / ak['cost']['mean']:.1f}",
            "seeds": str(seeds),
        }
        committed = _macro_values(MACROS.read_text(encoding="utf-8"))
        for name, value in expected.items():
            self.assertEqual(
                committed.get(name),
                value,
                f"macro \\{name}: committed {committed.get(name)!r} != regen {value!r} "
                "(re-run experiments/analyze.py --out paper after updating summary.json)",
            )


if __name__ == "__main__":
    unittest.main()
