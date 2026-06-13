"""Full PrivaCI-Bench compliance amortisation matrix (5 arms x seeds x teacher).

Orchestrates the controlled-runner arms over the whole experiment matrix in ONE
process per (teacher, seed) so a SINGLE SharedComplianceCache is reused across
every arm-config of that seed — the frontier teacher is called at most once per
distinct case for the entire seed (stream arms + lambda sweep + offline
contrast), not once per config. This both removes the teacher-stochasticity
confound across configs and keeps real spend near the single-run floor.

Per (teacher, seed):
  - stream arms (full 300-case stream): llm_only, cache, full, nl_strategy@lam.
Seed 0 only (the frontiers the paper needs to rebut "under-tuned baseline"):
  - nl_strategy lambda sweep over {0.5..0.9}.
  - offline contrast: for K in {0.3,0.5,0.7}, full (suffix-scored) vs
    compile_once (frozen after K) on the identical test suffix.

Run (keys in skret /tacet/prod)::

    MSYS_NO_PATHCONV=1 skret run -e prod --path=/tacet/prod -- \
        uv run python experiments/run_privaci_matrix.py --teachers gemini,grok \
        --seeds 0,1,2 --n 300 --budget-usd 2.0
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_privaci_controlled import (  # noqa: E402
    NL_STRATEGY_PROMPT_TEMPLATE,
    BudgetExceededError,
    BudgetGuard,
    MeteredTeacher,
    PriceTable,
    SharedComplianceCache,
    _build_teacher,
    _case_atoms,
    _load_embedder,
    _run_cache,
    _run_compile_once,
    _run_full,
    _run_llm_only,
    _run_nl_strategy,
)

from tacet.data.privaci import load_privaci  # noqa: E402
from tacet.data.privaci_graph import build_compliance_benchmark  # noqa: E402
from tacet.data.privaci_vocab import load_vocab, normalize_case  # noqa: E402

TAUS = (0.3, 0.5, 0.7)
PREFIX_FRACS = (0.3, 0.5, 0.7)
STREAM_ARMS = ("llm_only", "cache", "full", "nl_strategy")


def _run_seed(
    teacher: str,
    seed: int,
    *,
    n: int,
    privaci: str,
    split: str,
    budget_usd: float,
    consolidate_every: int,
    min_support: int,
    min_confidence: float,
    nl_tau: float,
    do_sweeps: bool,
) -> dict:
    cases = load_privaci(privaci, split=split)
    random.Random(seed).shuffle(cases)
    cases = cases[:n]
    vocab = load_vocab()
    bench = build_compliance_benchmark(cases, vocab)
    atoms = {c.case_id: _case_atoms(normalize_case(c, vocab)) for c in cases}
    distinct = len(set(atoms.values()))

    model, raw_teacher = _build_teacher(teacher)
    metered = MeteredTeacher(raw_teacher, PriceTable.default(), model=model)
    guard = BudgetGuard(budget_usd)
    shared = SharedComplianceCache(metered, guard, bench)
    # no routing: nl_strategy re-prompts the SAME model as the frontier
    weak_model, weak_raw = _build_teacher(teacher, NL_STRATEGY_PROMPT_TEMPLATE)
    weak = MeteredTeacher(weak_raw, PriceTable.default(), model=weak_model)
    embed = _load_embedder()

    print(f"\n=== {teacher} seed={seed} n={len(cases)} distinct={distinct} model={model} ===")
    out: dict = {
        "teacher": teacher,
        "model": model,
        "seed": seed,
        "n": len(cases),
        "distinct_patterns": distinct,
        "stream": {},
        "tau_sweep": [],
        "offline_contrast": [],
        "truncated_by_budget": False,
    }
    try:
        out["stream"]["llm_only"] = _run_llm_only(bench, shared)
        out["stream"]["cache"] = _run_cache(bench, shared, atoms)
        out["stream"]["full"] = _run_full(
            bench,
            shared,
            atoms,
            consolidate_every=consolidate_every,
            min_support=min_support,
            min_confidence=min_confidence,
        )
        out["stream"]["nl_strategy"] = _run_nl_strategy(
            bench, shared, weak_metered=weak, embed=embed, guard=guard, tau=nl_tau
        )
        for arm in STREAM_ARMS:
            r = out["stream"][arm]
            print(
                f"  {arm}: acc={r['verdict_acc']} f1={r['article_micro_f1']} "
                f"calls={r['teacher_calls']} cost=${r['total_cost_usd']:.4f}"
            )

        if do_sweeps:
            print("  -- nl_strategy tau sweep --")
            for tau in TAUS:
                r = _run_nl_strategy(
                    bench, shared, weak_metered=weak, embed=embed, guard=guard, tau=tau
                )
                out["tau_sweep"].append(r)
                print(
                    f"    tau={tau}: acc={r['verdict_acc']} defers={r['defers']} "
                    f"accepts={r['accepts']} cost=${r['total_cost_usd']:.4f}"
                )
            print("  -- offline contrast (full suffix vs compile_once) --")
            for frac in PREFIX_FRACS:
                k = int(round(frac * len(cases)))
                full_suffix = _run_full(
                    bench,
                    shared,
                    atoms,
                    consolidate_every=consolidate_every,
                    min_support=min_support,
                    min_confidence=min_confidence,
                    score_from=k,
                )
                comp = _run_compile_once(
                    bench,
                    shared,
                    atoms,
                    prefix_k=k,
                    min_support=min_support,
                    min_confidence=min_confidence,
                )
                out["offline_contrast"].append(
                    {
                        "prefix_frac": frac,
                        "prefix_k": k,
                        "full_suffix": full_suffix,
                        "compile_once": comp,
                    }
                )
                print(
                    f"    K={k}: full acc={full_suffix['verdict_acc']} "
                    f"cost=${full_suffix['total_cost_usd']:.4f} | compile_once "
                    f"acc={comp['verdict_acc']} cost=${comp['total_cost_usd']:.4f} "
                    f"hits={comp['engine_hits']}"
                )
    except BudgetExceededError as e:
        out["truncated_by_budget"] = True
        print(f"  [HARD STOP] {e}")

    out["real_teacher_calls"] = shared.real_calls
    out["total_measured_spend_usd"] = round(guard.spent_usd, 6)
    return out


def _amort(stream: dict) -> dict:
    """Amortisation + iso-accuracy summary vs llm_only for one seed's stream arms."""
    llm = stream["llm_only"]
    base = llm["total_cost_usd"]
    summ = {}
    for arm in ("cache", "full", "nl_strategy"):
        a = stream[arm]
        summ[arm] = {
            "amortisation_vs_llm": round(base / a["total_cost_usd"], 3)
            if a["total_cost_usd"]
            else None,
            "verdict_acc": a["verdict_acc"],
            "article_f1": a["article_micro_f1"],
            "verdict_acc_gap_vs_llm": round(a["verdict_acc"] - llm["verdict_acc"], 4),
        }
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--privaci", default="../PrivaCI-Bench")
    ap.add_argument("--split", default="GDPR")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--teachers", default="gemini,grok")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--budget-usd", type=float, default=2.0)
    ap.add_argument("--consolidate-every", type=int, default=25)
    ap.add_argument("--min-support", type=int, default=5)
    ap.add_argument("--min-confidence", type=float, default=0.9)
    ap.add_argument("--nl-tau", type=float, default=0.5)
    ap.add_argument("--sweep-seed", type=int, default=0, help="seed that also runs tau/K sweeps")
    ap.add_argument("--out", default="experiments/results/privaci_matrix.json")
    args = ap.parse_args()

    teachers = [t.strip() for t in args.teachers.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for teacher in teachers:
        for seed in seeds:
            r = _run_seed(
                teacher,
                seed,
                n=args.n,
                privaci=args.privaci,
                split=args.split,
                budget_usd=args.budget_usd,
                consolidate_every=args.consolidate_every,
                min_support=args.min_support,
                min_confidence=args.min_confidence,
                nl_tau=args.nl_tau,
                do_sweeps=(seed == args.sweep_seed),
            )
            runs.append(r)
            # incremental dump: a mid-matrix rate-limit death keeps completed seeds
            (outdir / f"privaci_matrix_partial_{teacher}_seed{seed}.json").write_text(
                json.dumps(r, indent=2), encoding="utf-8"
            )

    # aggregate amortisation mean+/-std per teacher across seeds (stream arms)
    aggregate: dict = {}
    for teacher in teachers:
        per = [_amort(r["stream"]) for r in runs if r["teacher"] == teacher and r["stream"]]
        if not per:
            continue
        agg = {}
        for arm in ("cache", "full", "nl_strategy"):
            am = [p[arm]["amortisation_vs_llm"] for p in per if p[arm]["amortisation_vs_llm"]]
            ac = [p[arm]["verdict_acc"] for p in per]
            f1 = [p[arm]["article_f1"] for p in per]
            agg[arm] = {
                "amortisation_mean": round(statistics.fmean(am), 3) if am else None,
                "amortisation_std": round(statistics.pstdev(am), 3) if len(am) > 1 else 0.0,
                "verdict_acc_mean": round(statistics.fmean(ac), 4),
                "article_f1_mean": round(statistics.fmean(f1), 4),
                "n_seeds": len(per),
            }
        aggregate[teacher] = agg

    report = {
        "dataset": f"PrivaCI-Bench-{args.split}",
        "n": args.n,
        "teachers": teachers,
        "seeds": seeds,
        "nl_strategy_model": "same-as-frontier (no routing: weak == frontier per run)",
        "runs": runs,
        "aggregate": aggregate,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    total = sum(r["total_measured_spend_usd"] for r in runs)
    print("\n=== AGGREGATE ===")
    for teacher, agg in aggregate.items():
        for arm, a in agg.items():
            print(
                f"  {teacher} {arm}: {a['amortisation_mean']}x +/-{a['amortisation_std']} "
                f"(acc {a['verdict_acc_mean']}, f1 {a['article_f1_mean']}, {a['n_seeds']} seeds)"
            )
    print(f"[spend] total measured across all runs: ${total:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
