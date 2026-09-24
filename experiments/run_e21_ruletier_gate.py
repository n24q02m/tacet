"""E21 arm (d) — rule-tier admission gate (prereg v2 LOCKED 2026-09-24, user-ratified).

Replays the exact 132 rule-states E19 persisted (same cells, same rules, read
from the same E17 artifacts) through a DETERMINISTIC rule-tier gate — no API
calls, $0 spend. Inverse direction of E19: can TACET's own symbolic tier emit
admit/confidence + leakage_risk calibrated enough for the 3-gate contract?

Locked v2 mapping (user-ratified 2026-09-24, amendment #1 of 1 allowed):
  confidence := rule miner confidence (persisted, no rescale)
  risk       := 1 - match_validation_vs_teacher / fired_validation
                (teacher-agreement on held-out validation; fired_validation==0
                -> risk = 1.0, fail-expensive; teacher abstains count as
                non-matches because only matches enter the numerator)
Gate (identical to E19/E22 for fairness):
  route symbolic iff confidence >= 0.55 AND risk <= 0.5; else escalate.

Contamination disclosure (recorded in artifact): the v2 mapping was selected
AFTER an offline probe showed the three schema-faithful readings of the v1
prereg (support/fit_heads, 1-miner_confidence, fired_validation/
validation_heads) all degenerate (0 or ~132 admits of 132), and after the v2
candidate's delta distribution had been observed (0.16-0.95, mean ~0.52).
The user ratified v2 WITH this disclosure on 2026-09-24.

Decision rule (LOCKED at ratify):
  TACET-TIER-SUFFICIENT   iff delta_ruletier >= delta_jev - 0.02  (cost leg
                          auto-wins: $0 < $0.0000252/call)
  TACET-TIER-INSUFFICIENT iff delta_ruletier <  delta_jev - 0.10
  else NEUTRAL. delta_jev baseline = 0.9707366666666667 (E19 aggregate).
Invalidation: >20% of the 132 rule-states missing required fields ->
  INFEASIBLE_SCHEMA, not a negative result.
"""

from __future__ import annotations

import glob
import json
import os
import sys

DELTA_JEV = 0.9707366666666667  # E19 aggregate, artifact e19_jev_gate.json
RESULTS = "experiments/results"
E19 = os.path.join(RESULTS, "e19_jev_gate.json")
OUT = os.path.join(RESULTS, "e21_ruletier_gate.json")
EXCLUDE = ("0731", ".partial")


def gate_rule(r: dict) -> tuple[bool, float, float]:
    """Locked v2 gate. Returns (admitted, confidence, risk)."""
    conf = r.get("confidence")
    fv = r.get("fired_validation")
    mv = r.get("match_validation_vs_teacher")
    if conf is None or fv is None or mv is None:
        return False, 0.0, 1.0  # missing field = escalate (fail-open = fail-expensive)
    risk = 1.0 if fv == 0 else 1.0 - mv / fv
    admitted = conf >= 0.55 and risk <= 0.5
    return admitted, conf, risk


def cell_stats(rules: list[dict]) -> dict:
    comp = [r for r in rules if r.get("class") == "composition"]
    leak = [r for r in rules if r.get("class") == "leakage"]
    adm = [r for r in rules if r["ruletier_admitted"]]
    acomp = [r for r in adm if r.get("class") == "composition"]
    aleak = [r for r in adm if r.get("class") == "leakage"]
    vp = [r["val_precision"] for r in acomp if r.get("val_precision") is not None]
    vl = [r["val_precision"] for r in aleak if r.get("val_precision") is not None]
    agree = sum(1 for r in rules if r["ruletier_admitted"] == r.get("jev_admitted"))
    return {
        "n_rules": len(rules),
        "reject_leakage": (len(leak) - len(aleak)) / len(leak) if leak else None,
        "keep_composition": len(acomp) / len(comp) if comp else None,
        "delta_ruletier": (
            (sum(vp) / len(vp) - sum(vl) / len(vl))
            if vp and vl
            else (sum(vp) / len(vp) if vp else None)
        ),
        "jev_agreement_frac": agree / len(rules) if rules else None,
    }


def main() -> int:
    with open(E19, encoding="utf-8") as fh:
        e19 = json.load(fh)
    jev_by_name = {}
    for c in e19["cells"]:
        for r in c["rules"]:
            jev_by_name[(c["artifact"], r["name"])] = r["jev_admitted"]

    cells = []
    missing = 0
    total = 0
    for p in sorted(glob.glob(os.path.join(RESULTS, "e17_admission_*.json"))):
        if any(g in p for g in EXCLUDE):
            continue
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("truncated_by_budget"):
            continue
        art = os.path.basename(p)
        recs = []
        for r in d.get("rules") or []:
            total += 1
            admitted, conf, risk = gate_rule(r)
            if r.get("confidence") is None or r.get("fired_validation") is None:
                missing += 1
            recs.append(
                {
                    "name": r.get("name"),
                    "target": r.get("target"),
                    "class": r.get("class"),
                    "val_precision": r.get("val_precision"),
                    "confidence": conf,
                    "risk": round(risk, 6),
                    "e17_admitted": r.get("admitted"),
                    "jev_admitted": jev_by_name.get((art, r.get("name"))),
                    "ruletier_admitted": admitted,
                }
            )
        cells.append(
            {
                "artifact": art,
                "slug": d.get("slug"),
                "seed": d.get("seed"),
                "rules": recs,
            }
        )

    if total and missing / total > 0.20:
        verdict = "INFEASIBLE_SCHEMA"
    else:
        for c in cells:
            c["stats"] = cell_stats(c["rules"])
        deltas = [
            c["stats"]["delta_ruletier"] for c in cells if c["stats"]["delta_ruletier"] is not None
        ]
        d_mean = sum(deltas) / len(deltas) if deltas else None
        if d_mean is None:
            verdict = "INFEASIBLE_SCHEMA"
        elif d_mean >= DELTA_JEV - 0.02:
            verdict = "TACET-TIER-SUFFICIENT"
        elif d_mean < DELTA_JEV - 0.10:
            verdict = "TACET-TIER-INSUFFICIENT"
        else:
            verdict = "NEUTRAL"

    leak_rej = [
        c["stats"]["reject_leakage"] for c in cells if c["stats"].get("reject_leakage") is not None
    ]
    comp_keep = [
        c["stats"]["keep_composition"]
        for c in cells
        if c["stats"].get("keep_composition") is not None
    ]
    deltas = [
        c["stats"]["delta_ruletier"] for c in cells if c["stats"].get("delta_ruletier") is not None
    ]
    agree = [
        c["stats"]["jev_agreement_frac"]
        for c in cells
        if c["stats"].get("jev_agreement_frac") is not None
    ]

    out = {
        "schema": "tacet.e21.ruletier_gate/v1",
        "prereg": (
            "v2 LOCKED 2026-09-24 (user-ratified, amendment #1/1): "
            "conf=miner confidence; risk=1-match_validation_vs_teacher/"
            "fired_validation (fired=0->1.0); gates conf>=0.55 & risk<=0.5; "
            "132 rule-states of E19; $0"
        ),
        "contamination_disclosure": (
            "v2 mapping selected after offline probes showed all three "
            "schema-faithful v1 readings degenerate (0 or ~132/132 admits) "
            "and after v2's delta distribution was observed (0.16-0.95); "
            "ratified with disclosure 2026-09-24"
        ),
        "spend_usd": 0.0,
        "delta_jev_baseline": DELTA_JEV,
        "missing_field_frac": round(missing / total, 4) if total else None,
        "verdict": verdict,
        "cells": cells,
        "aggregate": {
            "cells": len(cells),
            "rules": total,
            "delta_ruletier_mean": sum(deltas) / len(deltas) if deltas else None,
            "reject_leakage_mean": sum(leak_rej) / len(leak_rej) if leak_rej else None,
            "keep_composition_mean": sum(comp_keep) / len(comp_keep) if comp_keep else None,
            "jev_agreement_mean": sum(agree) / len(agree) if agree else None,
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"[e21] wrote {OUT}: rules={total} missing={missing} verdict={verdict}")
    print(f"[e21] aggregate: {json.dumps(out['aggregate'], default=str)[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
