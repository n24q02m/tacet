"""Aggregate the E16 forbid-target-in-body grid into one publishable artifact.

E16 asks whether the self-referential junk rule the low-gamma regime installs is
a property of rule distillation or of how candidates are enumerated. The control
arm is the gamma=0.50 slice of the recorded E11 replays; the treatment arm is the
same cells replayed with ``--forbid-target-in-body``.

The per-cell replay reports carry a misleading label: in replay mode the runner
does not rewrite ``teacher_model_called`` from the answers file, so every report
claims the default model. The model is recovered from the FILENAME, which is
derived from the answers path and is therefore authoritative. Cells are matched
between the arms by filename for the same reason.

Usage::

    python experiments/aggregate_e16_forbid.py \\
        --control <dir of E11 replays> --forbid <dir of forbid replays> \\
        --out experiments/results/e16_forbid_target_hop2.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

GAMMA = 0.50
TRUE_RULE_ATOMS = ("starred_actors", "directed_by")

# Filenames are either `<vendor>_<model>_hop2_lim300_s<seed>[_forbid]` or one of
# two short forms used for the first two probe cells.
SHORT_FORMS = {
    "glm_s2": ("z-ai/glm-5.2", 2),
    "sonnet_s0": ("anthropic/claude-sonnet-5", 0),
}


def parse_cell(stem: str) -> tuple[str, int]:
    """(slug, seed) from a filename. The metadata inside the file cannot be trusted."""
    stem = stem.removesuffix("_forbid")
    if stem in SHORT_FORMS:
        return SHORT_FORMS[stem]
    m = re.match(r"^(.+?)_hop2_lim300_s(\d+)$", stem)
    if not m:
        raise SystemExit(f"cannot parse cell name: {stem}")
    raw, seed = m.group(1), int(m.group(2))
    if "__" in raw:
        vendor, model = raw.split("__", 1)
    else:
        vendor, model = raw.split("_", 1)
    return f"{vendor}/{model}", seed


def match_key(cell: tuple[str, int]) -> tuple[str, int]:
    """A key both arms agree on.

    The forbid filenames were written with dots stripped, so the control's
    `google/gemini-3.5-flash` and the forbid arm's `google/gemini-35-flash`
    name the same cell. Dropping dots makes them meet; the readable slug is
    still taken from the control side.
    """
    slug, seed = cell
    return slug.replace(".", ""), seed


def classify(rules: list[str]) -> tuple[bool, int]:
    """(installed the true composition, number of junk rules).

    Classified from the rule BODY rather than from any stored verdict: a rule
    whose body is the two base relations is the world-correct composition, and a
    rule whose body mentions the mining target is the self-referential junk.
    """
    true_installed = False
    junk = 0
    for r in rules:
        body = r.split("<=", 1)[1] if "<=" in r else r
        if all(a in body for a in TRUE_RULE_ATOMS):
            true_installed = True
        else:
            junk += 1
    return true_installed, junk


def read_control(path: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for f in sorted(path.glob("*.json")):
        for cell in json.load(f.open(encoding="utf-8")):
            if abs(cell.get("gamma", 0) - GAMMA) > 1e-9:
                continue
            key = match_key(parse_cell(f.stem))
            slug, seed = parse_cell(f.stem)
            rules = cell.get("synthesised_rules") or []
            true_installed, junk = classify(rules)
            out[key] = {
                "slug": slug,
                "seed": seed,
                "calls_saved_pct": cell.get("calls_saved_pct"),
                "rule_world_precision": cell.get("rule_world_precision"),
                "synthesised_rules": rules,
                "cache_accuracy": cell.get("cache_accuracy"),
                "full_accuracy": cell.get("full_accuracy"),
                "true_rule_installed": true_installed,
                "junk_rules": junk,
            }
    return out


def read_forbid(path: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for f in sorted(path.glob("*.json")):
        d = json.load(f.open(encoding="utf-8"))
        v = d.get("verdict") or {}
        arms = {a["arm"]: a for a in d.get("arms", [])}
        rules = (arms.get("full_distillation") or {}).get("synthesised_rules") or []
        true_installed, junk = classify(rules)
        out[match_key(parse_cell(f.stem))] = {
            "calls_saved_pct": v.get("calls_saved_pct"),
            "rule_world_precision": v.get("rule_world_precision"),
            "synthesised_rules": rules,
            "cache_accuracy": v.get("accuracy_cache"),
            "full_accuracy": v.get("accuracy_full"),
            "true_rule_installed": true_installed,
            "junk_rules": junk,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--forbid", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    control, forbid = read_control(args.control), read_forbid(args.forbid)
    missing = sorted(set(control) - set(forbid))
    if missing:
        raise SystemExit(f"forbid arm is missing cells: {missing}")

    cells = []
    for key in sorted(control):
        c, f = control[key], forbid[key]
        slug, seed = c["slug"], c["seed"]
        cells.append(
            {
                "slug": slug,
                "seed": seed,
                "gamma": GAMMA,
                "control": c,
                "forbid": f,
                "savings_identical": (
                    c["true_rule_installed"]
                    and f["true_rule_installed"]
                    and abs((c["calls_saved_pct"] or 0) - (f["calls_saved_pct"] or 0)) < 1e-9
                ),
            }
        )

    true_c = sum(x["control"]["true_rule_installed"] for x in cells)
    true_f = sum(x["forbid"]["true_rule_installed"] for x in cells)
    junk_c = sum(x["control"]["junk_rules"] for x in cells)
    junk_f = sum(x["forbid"]["junk_rules"] for x in cells)
    junk_cells_c = sum(1 for x in cells if x["control"]["junk_rules"])
    identical = sum(1 for x in cells if x["savings_identical"])
    lost = [
        x["slug"]
        for x in cells
        if x["forbid"]["full_accuracy"] is not None
        and x["forbid"]["cache_accuracy"] is not None
        and x["forbid"]["full_accuracy"] < x["forbid"]["cache_accuracy"] - 1e-9
    ]
    delta = sum(
        (x["forbid"]["calls_saved_pct"] or 0) - (x["control"]["calls_saved_pct"] or 0)
        for x in cells
    )

    report = {
        "experiment": "E16 - is the junk rule an artifact of candidate enumeration?",
        "gamma": GAMMA,
        "n_cells": len(cells),
        "control_arm": "recorded E11 replays at gamma=0.50",
        "forbid_arm": "same cells replayed with --forbid-target-in-body",
        "note_on_model_labels": (
            "The per-cell reports carry teacher_model_called from the runner default, "
            "not from the replayed answers; the model here comes from the filename, "
            "which is derived from the answers path."
        ),
        "true_rule_installs": {"control": true_c, "forbid": true_f},
        "cells_with_identical_savings": identical,
        "junk_rules": {"control": junk_c, "forbid": junk_f},
        "cells_installing_junk": {"control": junk_cells_c, "forbid": 0},
        "cells_losing_accuracy": lost,
        "total_savings_delta_pp": round(delta, 4),
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out} ({len(cells)} cells)")
    print(f"  true rule installs : control {true_c} -> forbid {true_f}")
    print(f"  identical savings  : {identical}/{true_c}")
    print(f"  junk rules         : control {junk_c} across {junk_cells_c} cells -> forbid {junk_f}")
    print(f"  cells losing acc   : {len(lost)}")
    print(f"  savings delta      : {delta:+.2f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
