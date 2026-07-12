"""E11 two-axis teacher-quality ladder — read-only analysis of the JSON artifacts.

The E11 study asks whether rule distillation beats caching as a function of teacher
quality, sweeping the miner's confidence threshold ``gamma``. Every cell carries TWO
teacher-quality metrics that DIVERGE for a real LLM teacher:

* **arm accuracy** (``cache_accuracy``) — the cascade cache-arm's per-query benchmark
  accuracy. This is the DECISION axis (it is what the paper reports, e.g. 0.5367 for
  grok-4.3): the quantity a practitioner reads off to decide distil-vs-cache.
* **sub-gold accuracy** (``teacher_answer_accuracy``) — the fraction of distinct
  ``(head, relation)`` teacher answers that are a subset of gold. Exact for the noise
  oracle, but it collapses to ~0.02-0.04 for a real teacher that returns
  plausible-but-not-subset lists. This is an INTERNAL axis: it matches the oracle's
  noise dial, not the benchmark the decision is made on.

This module ONLY reads the two on-disk artifacts (the free oracle
``gamma x error_rate`` sweep and the real-teacher replay ladder). It never touches
Modal, MetaQA or any provider; it re-uses :func:`run_oracle_noise_sweep.wilson_ci`
and :func:`run_oracle_noise_sweep.cliff_teacher_accuracy` so the statistics match the
sweep exactly. ``main`` is a thin argparse wrapper over the pure functions below.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_oracle_noise_sweep import cliff_teacher_accuracy, wilson_ci  # noqa: E402

#: Wilson confidence for P(rule installed) — matches the sweep's pre-registered level.
WILSON_CONFIDENCE = 0.95

#: The decision axis (what the paper reports) vs. the internal axis (oracle noise dial).
DECISION_AXIS = "mean_arm_accuracy"
INTERNAL_AXIS = "mean_sub_gold_accuracy"

#: Map an aggregate accuracy axis to the field that carries it on a REAL ladder row.
_AXIS_REAL_FIELD: dict[str, str] = {
    "mean_arm_accuracy": "cache_accuracy",
    "mean_sub_gold_accuracy": "teacher_answer_accuracy",
}

#: Human labels for the printed header (which ruler is the decision one).
_AXIS_LABEL: dict[str, str] = {
    "mean_arm_accuracy": "arm accuracy (cascade cache-arm benchmark accuracy) -- DECISION axis",
    "mean_sub_gold_accuracy": "sub-gold accuracy (teacher_answer_accuracy) -- INTERNAL axis",
}

#: ``--axis`` CLI value -> aggregate axis key.
_CLI_AXIS: dict[str, str] = {"arm": DECISION_AXIS, "sub_gold": INTERNAL_AXIS}


def aggregate_oracle(cells: list[dict]) -> list[dict]:
    """Group oracle cells by ``(gamma, error_rate)`` and emit BOTH quality axes.

    For each group the row carries the mean of each ruler (``mean_arm_accuracy`` from
    ``cache_accuracy``, ``mean_sub_gold_accuracy`` from ``teacher_answer_accuracy``),
    the install probability ``p_install`` (fraction of cells with a non-empty
    ``synthesised_rules``) with a Wilson 95% interval, the mean matched-accuracy
    saving, the seed count, the count of invalid cells (``cell_valid is False``), and
    the mean rule-world precision over cells where it is not ``None``.

    Rows are sorted by ``(gamma, error_rate)`` so the output is deterministic.
    """
    groups: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for c in cells:
        groups[(c["gamma"], c["error_rate"])].append(c)

    out: list[dict] = []
    for (gamma, error_rate), cs in sorted(groups.items()):
        n = len(cs)
        n_installed = sum(1 for c in cs if c.get("synthesised_rules"))
        saved = [c["calls_saved_pct"] for c in cs if c.get("calls_saved_pct") is not None]
        rwp = [c["rule_world_precision"] for c in cs if c.get("rule_world_precision") is not None]
        lo, hi = wilson_ci(n_installed, n, WILSON_CONFIDENCE)
        out.append(
            {
                "gamma": gamma,
                "error_rate": error_rate,
                "n_seeds": n,
                "mean_arm_accuracy": fmean(c["cache_accuracy"] for c in cs),
                "mean_sub_gold_accuracy": fmean(c["teacher_answer_accuracy"] for c in cs),
                "p_install": n_installed / n if n else 0.0,
                "p_install_wilson95": [lo, hi],
                "mean_calls_saved_pct": fmean(saved) if saved else None,
                "mean_rule_world_precision": fmean(rwp) if rwp else None,
                "n_invalid": sum(1 for c in cs if c.get("cell_valid") is False),
            }
        )
    return out


def cliff(aggregates: list[dict], axis: str) -> list[dict]:
    """Per ``gamma``, the ``axis`` accuracy at which ``p_install`` crosses 0.5.

    ``axis`` is an aggregate accuracy key (``"mean_arm_accuracy"`` or
    ``"mean_sub_gold_accuracy"``) — this is the H2 cliff test expressed on either
    ruler. Within each gamma the aggregates are sorted by ``error_rate`` ascending (so
    accuracy runs high->low as noise rises) and the crossing is linearly interpolated
    on ``axis`` by the shared :func:`run_oracle_noise_sweep.cliff_teacher_accuracy`.

    Returns one ``{gamma, axis, cliff_accuracy}`` per gamma; ``cliff_accuracy`` is
    ``None`` when ``p_install`` never crosses 0.5 within the swept grid.
    """
    by_gamma: dict[float, list[dict]] = defaultdict(list)
    for a in aggregates:
        by_gamma[a["gamma"]].append(a)

    out: list[dict] = []
    for gamma in sorted(by_gamma):
        rows = sorted(by_gamma[gamma], key=lambda a: a["error_rate"])
        accs = [r[axis] for r in rows]
        ps = [r["p_install"] for r in rows]
        out.append(
            {"gamma": gamma, "axis": axis, "cliff_accuracy": cliff_teacher_accuracy(accs, ps)}
        )
    return out


def _oracle_p_install_at_accuracy(
    points: Sequence[tuple[float | None, float | None]], target: float
) -> float | None:
    """Oracle ``p_install`` at a given ``target`` accuracy, along one gamma's curve.

    ``points`` are ``(axis_accuracy, p_install)`` pairs from the oracle's error_rate
    sweep at a fixed gamma. They are sorted by accuracy and ``p_install`` is linearly
    interpolated in accuracy at ``target``. When ``target`` falls OUTSIDE the swept
    accuracy range the nearest endpoint's ``p_install`` is returned (the curve is not
    extrapolated past its measured ends, which would push a probability out of
    ``[0, 1]``). ``None`` when there is no usable point.
    """
    pts = [(a, p) for a, p in points if a is not None and p is not None]
    if not pts:
        return None
    pts.sort(key=lambda ap: ap[0])
    if len(pts) == 1 or target <= pts[0][0]:
        return pts[0][1]
    if target >= pts[-1][0]:
        return pts[-1][1]
    for (a0, p0), (a1, p1) in zip(pts, pts[1:], strict=False):
        if a0 <= target <= a1:
            if a1 == a0:
                return p0
            frac = (target - a0) / (a1 - a0)
            return p0 + frac * (p1 - p0)
    return None  # unreachable: the clamps above cover every target


def locate_on_curve(oracle_aggregates: list[dict], real_rows: list[dict], axis: str) -> list[dict]:
    """Place each real teacher on the oracle's i.i.d.-noise curve at matched accuracy.

    For every real ``(slug, gamma)`` group this computes the real mean ``p_install``
    and the real mean accuracy on ``axis`` (``cache_accuracy`` for the arm ruler,
    ``teacher_answer_accuracy`` for the sub-gold ruler), then reads the oracle's
    ``p_install`` at the SAME accuracy and gamma by interpolating along the oracle's
    error_rate sweep (:func:`_oracle_p_install_at_accuracy`).

    Emits ``{slug, gamma, real_accuracy, real_p_install,
    oracle_p_install_at_same_accuracy, delta}`` where
    ``delta = real_p_install - oracle_p_install_at_same_accuracy``.

    A NEGATIVE delta means the real teacher installs a rule LESS often than i.i.d.
    noise of the *same accuracy* would: its errors are STRUCTURED (correlated,
    systematic) rather than independent, and structured errors are harder for the
    miner to distil than i.i.d. noise at matched accuracy. A delta near 0 means the
    real teacher behaves like i.i.d. noise on that ruler.
    """
    real_field = _AXIS_REAL_FIELD[axis]
    oracle_by_gamma: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for a in oracle_aggregates:
        oracle_by_gamma[a["gamma"]].append((a[axis], a["p_install"]))

    real_groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for r in real_rows:
        real_groups[(r["slug"], r["gamma"])].append(r)

    out: list[dict] = []
    for (slug, gamma), rows in sorted(real_groups.items()):
        n = len(rows)
        real_p = (sum(1 for r in rows if r.get("synthesised_rules")) / n) if n else 0.0
        real_acc = fmean(r[real_field] for r in rows)
        oracle_p = _oracle_p_install_at_accuracy(oracle_by_gamma.get(gamma, []), real_acc)
        out.append(
            {
                "slug": slug,
                "gamma": gamma,
                "real_accuracy": real_acc,
                "real_p_install": real_p,
                "oracle_p_install_at_same_accuracy": oracle_p,
                "delta": None if oracle_p is None else real_p - oracle_p,
            }
        )
    return out


# --------------------------------------------------------------------- printing
def _fmt(x: float | None, nd: int = 4) -> str:
    return "None" if x is None else f"{x:.{nd}f}"


def _print_aggregate_table(aggregates: list[dict]) -> None:
    print("  aggregate  (per gamma x error_rate; both rulers side by side)")
    print(
        f"    {'gamma':>5} {'err':>5} {'arm_acc':>8} {'sub_gold':>8} "
        f"{'p_inst':>6} {'wilson95':>17} {'saved%':>7} {'n':>3} {'inval':>5} {'mean_rwp':>8}"
    )
    for a in sorted(aggregates, key=lambda a: (a["gamma"], a["error_rate"])):
        lo, hi = a["p_install_wilson95"]
        print(
            f"    {a['gamma']:>5.2f} {a['error_rate']:>5.2f} "
            f"{_fmt(a['mean_arm_accuracy']):>8} {_fmt(a['mean_sub_gold_accuracy']):>8} "
            f"{a['p_install']:>6.3f} {'[' + _fmt(lo, 3) + ',' + _fmt(hi, 3) + ']':>17} "
            f"{_fmt(a['mean_calls_saved_pct'], 2):>7} {a['n_seeds']:>3} {a['n_invalid']:>5} "
            f"{_fmt(a['mean_rule_world_precision'], 3):>8}"
        )


def _print_cliff_table(cliffs: list[dict]) -> None:
    print("  H2 cliff  (accuracy at which p_install crosses 0.5, on this axis)")
    print(f"    {'gamma':>5} {'cliff_accuracy':>15}")
    for c in cliffs:
        print(f"    {c['gamma']:>5.2f} {_fmt(c['cliff_accuracy'], 6):>15}")


def _print_localisation_table(locs: list[dict]) -> None:
    print("  localisation  (real teacher vs oracle i.i.d.-noise curve at matched accuracy)")
    print(
        f"    {'slug':>16} {'gamma':>5} {'real_acc':>8} {'real_p':>7} {'oracle_p':>8} {'delta':>8}"
    )
    for r in locs:
        d = r["delta"]
        dstr = "None" if d is None else f"{d:+.4f}"
        print(
            f"    {r['slug']:>16} {r['gamma']:>5.2f} {_fmt(r['real_accuracy']):>8} "
            f"{r['real_p_install']:>7.4f} "
            f"{_fmt(r['oracle_p_install_at_same_accuracy']):>8} {dstr:>8}"
        )
    print(
        "    delta < 0  => the real teacher installs LESS than i.i.d. noise of the same "
        "accuracy (structured errors are harder to distil than i.i.d. noise)."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", default="experiments/results/oracle_gamma_sweep_hop2.json")
    ap.add_argument("--real", default="experiments/results/real_ladder_hop2.json")
    ap.add_argument("--axis", choices=["arm", "sub_gold", "both"], default="both")
    args = ap.parse_args()

    oracle_path = Path(args.oracle)
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    aggregates = aggregate_oracle(oracle["cells"])

    real_path = Path(args.real)
    real_rows: list[dict] | None = None
    if real_path.exists():
        real_rows = json.loads(real_path.read_text(encoding="utf-8"))["rows"]

    axes = [DECISION_AXIS, INTERNAL_AXIS] if args.axis == "both" else [_CLI_AXIS[args.axis]]

    print("=== E11 teacher-quality ladder (two rulers) ===")
    print(f"oracle: {oracle_path}  ({len(oracle['cells'])} cells, {len(aggregates)} groups)")
    if real_rows is not None:
        print(f"real:   {real_path}  ({len(real_rows)} rows)")
    else:
        print(f"real:   {real_path}  (ABSENT -- oracle-only analysis)")
    print(
        "arm accuracy = DECISION axis (the paper's reported accuracy); "
        "sub-gold accuracy = INTERNAL axis (the oracle noise dial)."
    )
    print()
    _print_aggregate_table(aggregates)

    for axis in axes:
        print()
        print(f"axis = {axis}  [{_AXIS_LABEL[axis]}]")
        _print_cliff_table(cliff(aggregates, axis))
        if real_rows is not None:
            _print_localisation_table(locate_on_curve(aggregates, real_rows, axis))


if __name__ == "__main__":
    main()
