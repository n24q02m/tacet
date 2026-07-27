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

# Defaults describe the committed E16 grid. They are defaults, not assumptions:
# every one is overridable, and the dataset is checked against the arm reports so
# the script refuses to run on a workload it would mislabel. See `classify`.
DEFAULT_GAMMA = 0.50
DEFAULT_COMPOSITION = ("starred_actors", "directed_by")
DEFAULT_DATASET = "MetaQA-2hop-test"

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


def classify(
    rules: list[str], composition: tuple[str, ...] = DEFAULT_COMPOSITION
) -> tuple[bool, int, int]:
    """(installed the true composition, self-referential rules, everything else).

    Classified from the rule BODY rather than from any stored verdict. Three
    classes, not two, because the paper's claim is about one of them:

    * the world-correct composition -- body is the base relations in
      `composition`;
    * the SELF-REFERENTIAL junk -- body names the mining target, which is what
      forbidding the target in the body removes;
    * anything else -- a body of legitimate base relations that is nonetheless
      not the composition. That is the near-functional leakage the paper scopes
      out explicitly, and `--forbid-target-in-body` does nothing about it.

    Folding the third class into "junk" happens to give the same counts on this
    grid because it is empty here. On a workload that produces a leaky rule it
    would report junk going to zero under an option that cannot remove it, which
    is precisely the conflation the paper's scope limit warns against.

    `composition` is a parameter for the same reason. Hard-coding MetaQA's two
    relations makes the mislabelling run the other way on another workload: a
    correct composition mined over different relations lands in `other`, and
    `true_rule_installs` silently reads 0. The caller states which relations
    compose, and `main` additionally refuses to run when the arm reports name a
    dataset other than the one it was told to expect.
    """
    true_installed = False
    self_referential = 0
    other = 0
    for r in rules:
        head, _, body = r.partition("<=")
        body = body or head
        target = head.removeprefix("syn:").strip()
        if all(a in body for a in composition):
            true_installed = True
        elif target and target in body:
            self_referential += 1
        else:
            other += 1
    return true_installed, self_referential, other


def read_control(
    path: Path, gamma: float = DEFAULT_GAMMA, composition: tuple[str, ...] = DEFAULT_COMPOSITION
) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for f in sorted(path.glob("*.json")):
        for cell in json.load(f.open(encoding="utf-8")):
            if abs(cell.get("gamma", 0) - gamma) > 1e-9:
                continue
            key = match_key(parse_cell(f.stem))
            slug, seed = parse_cell(f.stem)
            rules = cell.get("synthesised_rules") or []
            true_installed, self_ref, other = classify(rules, composition)
            out[key] = {
                "slug": slug,
                "seed": seed,
                "calls_saved_pct": cell.get("calls_saved_pct"),
                "rule_world_precision": cell.get("rule_world_precision"),
                "synthesised_rules": rules,
                "cache_accuracy": cell.get("cache_accuracy"),
                "full_accuracy": cell.get("full_accuracy"),
                "true_rule_installed": true_installed,
                "self_referential_rules": self_ref,
                "other_rules": other,
            }
    return out


def read_forbid(
    path: Path,
    composition: tuple[str, ...] = DEFAULT_COMPOSITION,
    dataset: str = DEFAULT_DATASET,
) -> dict[tuple[str, int], dict]:
    """Read the treatment arm, refusing to run on a workload it would mislabel.

    The forbid reports carry `dataset`, so the check costs nothing and turns a
    silently wrong `true_rule_installs: 0` into a stop. The control replays carry
    no such field, which is why the guard lives here.
    """
    out: dict[tuple[str, int], dict] = {}
    for f in sorted(path.glob("*.json")):
        d = json.load(f.open(encoding="utf-8"))
        found = d.get("dataset")
        if found is not None and found != dataset:
            raise SystemExit(
                f"{f.name}: dataset is {found!r}, expected {dataset!r}. Classifying it with "
                f"the composition {composition} would report that workload's true rule as "
                "leakage and its install count as zero. Pass --dataset and --composition."
            )
        v = d.get("verdict") or {}
        arms = {a["arm"]: a for a in d.get("arms", [])}
        rules = (arms.get("full_distillation") or {}).get("synthesised_rules") or []
        true_installed, self_ref, other = classify(rules, composition)
        out[match_key(parse_cell(f.stem))] = {
            "calls_saved_pct": v.get("calls_saved_pct"),
            "rule_world_precision": v.get("rule_world_precision"),
            "synthesised_rules": rules,
            "cache_accuracy": v.get("accuracy_cache"),
            "full_accuracy": v.get("accuracy_full"),
            "true_rule_installed": true_installed,
            "self_referential_rules": self_ref,
            "other_rules": other,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--forbid", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument(
        "--composition",
        default=",".join(DEFAULT_COMPOSITION),
        help="comma-separated base relations whose conjunction is the world-correct rule",
    )
    ap.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="expected `dataset` in the forbid reports; a mismatch stops the run",
    )
    args = ap.parse_args()

    composition = tuple(x.strip() for x in args.composition.split(",") if x.strip())
    if not composition:
        raise SystemExit("--composition needs at least one relation")

    control = read_control(args.control, args.gamma, composition)
    forbid = read_forbid(args.forbid, composition, args.dataset)
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
                "gamma": args.gamma,
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
    selfref_c = sum(x["control"]["self_referential_rules"] for x in cells)
    selfref_f = sum(x["forbid"]["self_referential_rules"] for x in cells)
    selfref_cells_c = sum(1 for x in cells if x["control"]["self_referential_rules"])
    other_c = sum(x["control"]["other_rules"] for x in cells)
    other_f = sum(x["forbid"]["other_rules"] for x in cells)
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
        "gamma": args.gamma,
        "dataset": args.dataset,
        "composition": list(composition),
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
        "self_referential_rules": {"control": selfref_c, "forbid": selfref_f},
        "cells_installing_self_referential": {
            "control": selfref_cells_c,
            "forbid": sum(1 for x in cells if x["forbid"]["self_referential_rules"]),
        },
        "other_rules": {
            "control": other_c,
            "forbid": other_f,
            "note": (
                "Bodies of legitimate base relations that are not the composition -- the "
                "near-functional leakage the paper scopes out. --forbid-target-in-body does "
                "not address them, so they are counted apart from the self-referential rules "
                "rather than folded in. Empty on this grid."
            ),
        },
        "cells_losing_accuracy": lost,
        "total_savings_delta_pp": round(delta, 4),
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out} ({len(cells)} cells)")
    print(f"  true rule installs : control {true_c} -> forbid {true_f}")
    print(f"  identical savings  : {identical}/{true_c}")
    print(
        f"  self-referential   : control {selfref_c} across {selfref_cells_c} cells "
        f"-> forbid {selfref_f}"
    )
    print(f"  other (leakage)    : control {other_c} -> forbid {other_f}")
    print(f"  cells losing acc   : {len(lost)}")
    print(f"  savings delta      : {delta:+.2f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
