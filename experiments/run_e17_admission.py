"""E17 — held-out admission test for near-functional leakage (PrivaCI-Bench GDPR).

One paid phase, one free phase, per (model, seed) cell:

1. RECORD (paid): one real teacher answer per workload case with per-head
   measured cost and provider usage — the shared-answer recording whose absence
   of per-head logs made the endpoint incomputable from committed artifacts
   (E17 feasibility verdict, 2026-08-23).
2. ADMISSION (free): split the answered heads 50/50 into fit/validation by a
   stable hash of ``(seed, case_id)``; mine rules from fit-half write-backs
   ONLY at the locked operating point (min_support=5, gamma_candidate=0.90);
   score every candidate rule on the validation half against the SAME teacher
   answers (verdict-match, the benchmark accuracy criterion); compute world
   precision over the full workload oracle exactly as paper Section
   ``rule_precision``; classify post-hoc composition-class (world >= 0.95) vs
   leakage-class (world < 0.85); report the locked endpoint

       Delta = mean val_precision(composition) - mean val_precision(leakage)

   and the locked decision rule POSITIVE / NEGATIVE / NEUTRAL (see
   ``tacet-research-ledger.md`` E17 PRE-REGISTRATION, locked 2026-08-23).

Per-head teacher answers, per-rule firing counts, and per-call provider costs
are persisted so the endpoint is recomputable from the artifact alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_privaci_controlled import _engine_answer  # noqa: E402
from run_real_kg_amortization import BudgetExceededError, BudgetGuard  # noqa: E402

from tacet.core.symbolic import RuleEngine  # noqa: E402
from tacet.data.privaci import load_privaci  # noqa: E402
from tacet.data.privaci_graph import build_compliance_benchmark  # noqa: E402
from tacet.data.privaci_vocab import load_vocab, normalize_case  # noqa: E402
from tacet.distill.compliance_miner import LabeledCase, mine_compliance_rules  # noqa: E402
from tacet.llm.metering import MeteredTeacher, PriceTable, price_key_for_slug  # noqa: E402
from tacet.llm.teachers.compliance import (  # noqa: E402
    COMPLIANCE_PROMPT_TEMPLATE,
    parse_compliance_answer,
)
from tacet.llm.teachers.llm import OpenRouterTeacher  # noqa: E402

Answer = tuple[str, tuple[str, ...]]

#: Locked operating point (E17 pre-registration): candidate-rule confidence
#: threshold for mining and the admission cut on validation precision.
GAMMA_CANDIDATE = 0.90
MIN_SUPPORT = 5
#: Post-hoc world-precision class thresholds, exactly as Section rule_precision.
WORLD_COMPOSITION = 0.95
WORLD_LEAKAGE = 0.85
#: Locked decision-rule margins.
DELTA_POSITIVE = 0.20
DELTA_NEGATIVE = 0.05
REJECT_LEAKAGE_MIN = 0.90
KEEP_COMPOSITION_MIN = 0.90
REJECT_COMPOSITION_MAX = 0.10


def _half(seed: int, case_id: str) -> str:
    """Stable fit/validation assignment: sha256(seed:case_id) parity."""
    digest = hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()
    return "fit" if int(digest, 16) % 2 == 0 else "validation"


def _match(pred: Answer | None, gold: Answer) -> bool:
    """Benchmark accuracy criterion: verdict equality (articles logged too)."""
    return pred is not None and pred[0] == gold[0]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def record_answers(
    bench, workload: list[str], metered: MeteredTeacher, guard: BudgetGuard
) -> tuple[dict, bool]:
    """Paid phase: one teacher answer per workload case, per-head cost+usage."""
    heads: dict[str, dict] = {}
    truncated = False
    t0 = time.time()
    try:
        for i, case_id in enumerate(workload):
            resp = metered.answer(None, bench.case_content[case_id], "verdict")
            answer = parse_compliance_answer(resp.answers)
            usage = dict(getattr(metered, "last_usage", None) or {})
            heads[case_id] = {
                "verdict": answer[0],
                "articles": list(answer[1]),
                "cost_usd": round(metered.last_cost_usd, 8),
                "provider_cost_usd": usage.get("cost"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "cached_tokens": usage.get("cached_tokens"),
            }
            guard.add(metered.last_cost_usd)
            if (i + 1) % 25 == 0:
                print(
                    f"  recorded {i + 1}/{len(workload)} "
                    f"spend=${guard.spent_usd:.4f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    except BudgetExceededError as exc:  # pragma: no cover - budget path
        truncated = True
        print(f"[HARD STOP] {exc}")
    return heads, truncated


def _score_rule(bench, workload: list[str], rule, seed: int, answered: dict) -> dict:
    """Fire ONE rule in isolation over the whole workload; count matches."""
    engine = RuleEngine(bench.ontology)
    if not engine.add_rule(rule):
        return {"gate_rejected": True}
    engine.materialise(bench.graph)
    fired_all = match_all = fired_val = match_val = abstain_val = 0
    for case_id in workload:
        if case_id not in answered:
            continue
        pred = _engine_answer(engine, case_id)
        if pred is None:
            continue
        fired_all += 1
        if _match(pred, bench.oracle[case_id]):
            match_all += 1
        if _half(seed, case_id) == "validation":
            fired_val += 1
            if answered[case_id]["verdict"] == "abstain":
                abstain_val += 1
            elif _match(pred, (answered[case_id]["verdict"], tuple(answered[case_id]["articles"]))):
                match_val += 1
    scorables = fired_val - abstain_val
    world_p = match_all / fired_all if fired_all else None
    val_p = match_val / scorables if scorables else None
    if world_p is None:
        cls = "never_fires"
    elif world_p >= WORLD_COMPOSITION:
        cls = "composition"
    elif world_p < WORLD_LEAKAGE:
        cls = "leakage"
    else:
        cls = "middle"
    return {
        "gate_rejected": False,
        "fired_all": fired_all,
        "match_all_vs_oracle": match_all,
        "world_precision": round(world_p, 4) if world_p is not None else None,
        "fired_validation": fired_val,
        "teacher_abstain_validation": abstain_val,
        "match_validation_vs_teacher": match_val,
        "val_precision": round(val_p, 4) if val_p is not None else None,
        "class": cls,
        "admitted": (val_p >= GAMMA_CANDIDATE) if val_p is not None else None,
    }


def run_admission(
    bench,
    workload: list[str],
    heads: dict,
    *,
    seed: int,
    min_support: int,
    min_confidence: float,
    atoms: dict,
) -> dict:
    """Free phase: fit-half mining, validation-half scoring, locked decision."""
    answered = {c: (v["verdict"], tuple(v["articles"])) for c, v in heads.items()}
    fit_ids = [c for c in workload if c in answered and _half(seed, c) == "fit"]
    val_ids = [c for c in workload if c in answered and _half(seed, c) == "validation"]

    labeled = [
        LabeledCase(case_id=c, atoms=atoms[c], verdict=answered[c][0], articles=answered[c][1])
        for c in fit_ids
        if answered[c][0] in ("permit", "prohibit")
    ]
    mined = mine_compliance_rules(labeled, min_support=min_support, min_confidence=min_confidence)

    rule_rows: list[dict] = []
    for m in mined:
        row = {
            "name": m.rule.name,
            "target": m.target,
            "confidence": round(m.confidence, 4),
            "support": m.support,
        }
        row.update(_score_rule(bench, workload, m.rule, seed, heads))
        rule_rows.append(row)

    # Full-engine hybrid arm over the validation half: rule verdict where the
    # combined rule set fires, teacher answer where it does not (the `full`
    # arm semantics), scored against the oracle like every arm.
    full_engine = RuleEngine(bench.ontology)
    for m in mined:
        full_engine.add_rule(m.rule)
    full_engine.materialise(bench.graph)
    hybrid_n = hybrid_ok = cache_ok = 0
    for case_id in val_ids:
        gold = bench.oracle[case_id]
        pred = _engine_answer(full_engine, case_id)
        if pred is None:
            pred = answered[case_id]
        hybrid_n += 1
        hybrid_ok += int(_match(pred, gold))
        cache_ok += int(_match(answered[case_id], gold))
    hybrid_acc = hybrid_ok / hybrid_n if hybrid_n else None
    cache_acc = cache_ok / hybrid_n if hybrid_n else None

    scored = [r for r in rule_rows if r.get("val_precision") is not None]
    comp = [r for r in scored if r["class"] == "composition"]
    leak = [r for r in scored if r["class"] == "leakage"]
    middle = [r for r in rule_rows if r.get("class") == "middle"]
    mean_comp = _mean([r["val_precision"] for r in comp])
    mean_leak = _mean([r["val_precision"] for r in leak])
    delta = mean_comp - mean_leak if mean_comp is not None and mean_leak is not None else None
    reject_leak = sum(1 for r in leak if not r["admitted"]) / len(leak) if leak else None
    keep_comp = sum(1 for r in comp if r["admitted"]) / len(comp) if comp else None
    reject_comp = sum(1 for r in comp if not r["admitted"]) / len(comp) if comp else None

    if delta is None:
        decision = "INDETERMINATE_EMPTY_CLASS"
        decision_why = (
            f"composition_rules={len(comp)} leakage_rules={len(leak)}; "
            "the locked POSITIVE/NEGATIVE conditions are undefined without both classes"
        )
    else:
        positive = (
            delta >= DELTA_POSITIVE
            and reject_leak is not None
            and reject_leak >= REJECT_LEAKAGE_MIN
            and keep_comp is not None
            and keep_comp >= KEEP_COMPOSITION_MIN
            and hybrid_acc is not None
            and cache_acc is not None
            and hybrid_acc >= cache_acc
        )
        negative = delta < DELTA_NEGATIVE or (
            reject_comp is not None and reject_comp > REJECT_COMPOSITION_MAX
        )
        if positive:
            decision = "POSITIVE"
        elif negative:
            decision = "NEGATIVE"
        else:
            decision = "NEUTRAL"
        decision_why = (
            f"delta={delta:.4f} reject_leak={reject_leak} keep_comp={keep_comp} "
            f"reject_comp={reject_comp} hybrid_acc={hybrid_acc} cache_acc={cache_acc}"
        )

    return {
        "fit_heads": len(fit_ids),
        "validation_heads": len(val_ids),
        "mined_candidates": len(mined),
        "rules": rule_rows,
        "class_counts": {
            "composition": len(comp),
            "leakage": len(leak),
            "middle": len(middle),
            "never_fires": sum(1 for r in rule_rows if r.get("class") == "never_fires"),
        },
        "endpoint": {
            "delta": round(delta, 4) if delta is not None else None,
            "mean_val_precision_composition": (
                round(mean_comp, 4) if mean_comp is not None else None
            ),
            "mean_val_precision_leakage": (round(mean_leak, 4) if mean_leak is not None else None),
        },
        "admission_test": {
            "gamma_candidate": GAMMA_CANDIDATE,
            "reject_leakage_frac": (round(reject_leak, 4) if reject_leak is not None else None),
            "keep_composition_frac": (round(keep_comp, 4) if keep_comp is not None else None),
            "reject_composition_frac": (round(reject_comp, 4) if reject_comp is not None else None),
        },
        "arm_accuracy_validation": {
            "hybrid_rule_arm": round(hybrid_acc, 4) if hybrid_acc is not None else None,
            "cache_arm_teacher": round(cache_acc, 4) if cache_acc is not None else None,
        },
        "decision": decision,
        "decision_why": decision_why,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--privaci", default="../PrivaCI-Bench")
    ap.add_argument("--split", default="GDPR")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slug", required=True, help="OpenRouter model id, e.g. z-ai/glm-5.3")
    ap.add_argument(
        "--effort",
        default=None,
        help="OpenRouter reasoning effort (e.g. max/xhigh/high); omit for provider default",
    )
    ap.add_argument("--budget-usd", type=float, default=0.35)
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    ap.add_argument("--min-confidence", type=float, default=GAMMA_CANDIDATE)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = load_privaci(args.privaci, split=args.split)
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    cases = cases[: args.n]
    vocab = load_vocab()
    bench = build_compliance_benchmark(cases, vocab)
    atoms = {c.case_id: _case_atoms(normalize_case(c, vocab)) for c in cases}
    workload = list(bench.workload)
    print(
        f"[setup] {args.split} n={len(cases)} seed={args.seed} "
        f"slug={args.slug} effort={args.effort or 'provider-default'} "
        f"budget=${args.budget_usd:.2f}",
        flush=True,
    )

    price_key = price_key_for_slug(args.slug)
    base_url = os.environ.get("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1"
    teacher = OpenRouterTeacher(
        os.environ["TACET_OPENROUTER_API_KEY"],
        model=args.slug,
        prompt_template=COMPLIANCE_PROMPT_TEMPLATE,
        reasoning_effort=args.effort,
        base_url=base_url,
    )
    metered = MeteredTeacher(teacher, PriceTable.default(), model=price_key)
    guard = BudgetGuard(args.budget_usd)

    heads, truncated = record_answers(bench, workload, metered, guard)
    abstains = sum(1 for v in heads.values() if v["verdict"] == "abstain")
    print(
        f"[record] heads={len(heads)} abstains={abstains} "
        f"spend=${guard.spent_usd:.4f} truncated={truncated}",
        flush=True,
    )

    admission = run_admission(
        bench,
        workload,
        heads,
        seed=args.seed,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        atoms=atoms,
    )
    print(
        f"[admission] classes={admission['class_counts']} "
        f"delta={admission['endpoint']['delta']} decision={admission['decision']}",
        flush=True,
    )
    print(f"  {admission['decision_why']}")

    out = Path(
        args.out
        or (
            f"experiments/results/e17_admission_"
            f"{args.slug.replace('/', '__')}_s{args.seed}"
            f"{'' if args.effort is None else '_' + args.effort}.json"
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": "tacet.e17.admission/v1",
                "experiment": "E17 held-out admission test for near-functional leakage",
                "pre_registration": "tacet-research-ledger.md E17 (locked 2026-08-23)",
                "dataset": f"PrivaCI-Bench-{args.split}",
                "n": len(cases),
                "seed": args.seed,
                "slug": args.slug,
                "reasoning_effort": args.effort,
                "price_key": price_key,
                "min_support": args.min_support,
                "min_confidence": args.min_confidence,
                "split_rule": "sha256(seed:case_id) parity, 50/50 fit/validation",
                "world_precision_reference": (
                    "paper Section rule_precision: firings matching the workload "
                    "oracle over the full n-case ground truth"
                ),
                "spend_usd": round(guard.spent_usd, 6),
                "real_teacher_calls": len(heads),
                "truncated_by_budget": truncated,
                "teacher_abstains": abstains,
                "heads": heads,
                **admission,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[spend] ${guard.spent_usd:.4f} for {len(heads)} calls; wrote {out}")


def _case_atoms(slots: dict[str, tuple[str, ...]]) -> frozenset[tuple[str, str]]:
    return frozenset((slot, v) for slot, values in slots.items() for v in values)


if __name__ == "__main__":
    main()
