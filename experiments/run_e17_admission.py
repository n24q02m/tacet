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
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_privaci_controlled import _engine_answer  # noqa: E402
from run_real_kg_amortization import BudgetExceededError, BudgetGuard  # noqa: E402
from run_real_kg_controlled import _AnswerLog, _atomic_write_text  # noqa: E402

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
E17_SCHEMA = "tacet.e17.admission/v1"
E17_PARTIAL_SCHEMA = "tacet.e17.admission.partial/v1"


def load_resume_artifact(
    path: Path,
    *,
    dataset: str,
    n: int,
    seed: int,
    slug: str,
    reasoning_effort: str | None,
    price_key: str,
    min_support: int,
    min_confidence: float,
) -> tuple[dict[str, dict], float]:
    """Restore paid heads from one compatible budget-truncated E17 artifact.

    A cell's answers are expensive and stochastic, so resume never re-buys a
    recorded head. It also refuses every stream-shaping provenance mismatch:
    mixing even one different model, effort, seed, or split would invalidate
    the locked fit/validation endpoint.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != E17_SCHEMA:
        raise ValueError(
            f"resume artifact {path} has schema {record.get('schema')!r}, expected {E17_SCHEMA!r}"
        )
    if not record.get("truncated_by_budget"):
        raise ValueError(f"resume artifact {path} is not budget-truncated")

    expected = {
        "dataset": dataset,
        "n": n,
        "seed": seed,
        "slug": slug,
        "reasoning_effort": reasoning_effort,
        "price_key": price_key,
        "min_support": min_support,
        "min_confidence": min_confidence,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(
                f"resume artifact {path} {field}={record.get(field)!r}, expected {value!r}"
            )

    heads = record.get("heads")
    if not isinstance(heads, dict):
        raise ValueError(f"resume artifact {path} heads is not an object")
    if any(not isinstance(head, dict) or "cost_usd" not in head for head in heads.values()):
        raise ValueError(f"resume artifact {path} has an invalid paid-head row")
    return heads, sum(float(head["cost_usd"]) for head in heads.values())


def _partial_log_path(path: Path) -> Path:
    return Path(f"{path}.partial")


def resume_or_start_partial_log(path: Path, provenance: dict) -> tuple[_AnswerLog, list[dict]]:
    """Open an E17 write-ahead log and warm-load its validated durable prefix."""
    log = _AnswerLog(_partial_log_path(path))
    if not log.exists():
        log.start({"schema": E17_PARTIAL_SCHEMA, "provenance": provenance})
        return log, []

    header, rows = log.read()
    if header is None:
        log.start({"schema": E17_PARTIAL_SCHEMA, "provenance": provenance})
        return log, []
    if header.get("schema") != E17_PARTIAL_SCHEMA:
        raise ValueError(
            f"partial log {log.path} has schema {header.get('schema')!r}, "
            f"expected {E17_PARTIAL_SCHEMA!r}"
        )
    logged_provenance = header.get("provenance")
    if not isinstance(logged_provenance, dict):
        raise ValueError(f"partial log {log.path} has no provenance object")
    for field, value in provenance.items():
        if logged_provenance.get(field) != value:
            raise ValueError(
                f"partial log {log.path} {field}={logged_provenance.get(field)!r}, "
                f"expected {value!r}"
            )
    for row in rows:
        if not isinstance(row.get("case_id"), str) or not isinstance(row.get("head"), dict):
            raise ValueError(f"partial log {log.path} has an invalid paid-head row")

    # Drop any crash-truncated trailing line before appending again.
    log.rewrite(header, rows)
    return log, rows


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
    bench,
    workload: list[str],
    metered: MeteredTeacher,
    guard: BudgetGuard,
    *,
    heads: dict[str, dict] | None = None,
    checkpoint: Callable[[str, dict], None] | None = None,
) -> tuple[dict, bool]:
    """Record only heads not already paid for by a compatible prior artifact."""
    heads = dict(heads or {})
    resumed = len(heads)
    fresh = 0
    truncated = False
    t0 = time.time()
    try:
        for case_id in workload:
            if case_id in heads:
                continue
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
            fresh += 1
            if checkpoint is not None:
                checkpoint(case_id, heads[case_id])
            guard.add(metered.last_cost_usd)
            if fresh % 25 == 0:
                print(
                    f"  recorded {len(heads)}/{len(workload)} "
                    f"(+{fresh} new, {resumed} warm) "
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


def incomplete_admission(*, heads: int, expected_heads: int) -> dict:
    """Mark a partial paid recording as resumable, never as a result cell."""
    return {
        "fit_heads": None,
        "validation_heads": None,
        "mined_candidates": 0,
        "rules": [],
        "class_counts": {
            "composition": 0,
            "leakage": 0,
            "middle": 0,
            "never_fires": 0,
        },
        "endpoint": {
            "delta": None,
            "mean_val_precision_composition": None,
            "mean_val_precision_leakage": None,
        },
        "admission_test": {
            "gamma_candidate": GAMMA_CANDIDATE,
            "reject_leakage_frac": None,
            "keep_composition_frac": None,
            "reject_composition_frac": None,
        },
        "arm_accuracy_validation": {
            "hybrid_rule_arm": None,
            "cache_arm_teacher": None,
        },
        "decision": "INCOMPLETE_RECORDING",
        "decision_why": (
            f"recorded {heads}/{expected_heads} heads; no locked endpoint decision "
            "is valid until all planned heads are present"
        ),
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


def _output_path(out_arg: str | None, slug: str, seed: int, effort: str | None) -> Path:
    return Path(
        out_arg
        or (
            f"experiments/results/e17_admission_{slug.replace('/', '__')}_s{seed}"
            f"{'' if effort is None else '_' + effort}.json"
        )
    )


def _merge_warm_rows(heads: dict[str, dict], rows: list[dict]) -> dict[str, dict]:
    """Merge the canonical artifact and a crash-surviving WAL without conflicts."""
    merged = dict(heads)
    for row in rows:
        case_id = row["case_id"]
        head = row["head"]
        if case_id in merged and merged[case_id] != head:
            raise ValueError(f"resume has conflicting durable records for {case_id!r}")
        merged[case_id] = head
    return merged


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
    ap.add_argument(
        "--budget-usd",
        type=float,
        default=0.35,
        help="fresh-spend hard cap; on --resume this caps only newly bought heads",
    )
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    ap.add_argument("--min-confidence", type=float, default=GAMMA_CANDIDATE)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="warm-load a compatible truncated artifact and only buy missing heads",
    )
    args = ap.parse_args()

    out = _output_path(args.out, args.slug, args.seed, args.effort)
    partial_path = _partial_log_path(out)
    if not args.resume and (out.exists() or partial_path.exists()):
        raise SystemExit(f"output state exists for {out}; rerun with --resume or choose --out")
    if args.resume and not out.exists() and not partial_path.exists():
        raise SystemExit(f"no resumable artifact or partial log exists for {out}")

    cases = load_privaci(args.privaci, split=args.split)
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    cases = cases[: args.n]
    vocab = load_vocab()
    bench = build_compliance_benchmark(cases, vocab)
    atoms = {c.case_id: _case_atoms(normalize_case(c, vocab)) for c in cases}
    workload = list(bench.workload)
    price_key = price_key_for_slug(args.slug)
    dataset = f"PrivaCI-Bench-{args.split}"
    provenance = {
        "dataset": dataset,
        "n": len(cases),
        "seed": args.seed,
        "slug": args.slug,
        "reasoning_effort": args.effort,
        "price_key": price_key,
        "min_support": args.min_support,
        "min_confidence": args.min_confidence,
    }

    warm_heads: dict[str, dict] = {}
    if args.resume and out.exists():
        warm_heads, _ = load_resume_artifact(out, **provenance)
    log, warm_rows = resume_or_start_partial_log(out, provenance)
    heads = _merge_warm_rows(warm_heads, warm_rows)
    unknown_heads = set(heads) - set(workload)
    if unknown_heads:
        raise ValueError(
            f"resume artifact has heads outside this workload: {sorted(unknown_heads)!r}"
        )
    resumed_heads = len(heads)

    print(
        f"[setup] {args.split} n={len(cases)} seed={args.seed} "
        f"slug={args.slug} effort={args.effort or 'provider-default'} "
        f"new-budget=${args.budget_usd:.2f} resumed={resumed_heads}",
        flush=True,
    )
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

    def checkpoint(case_id: str, head: dict) -> None:
        log.append_row({"case_id": case_id, "head": head})

    heads, truncated = record_answers(
        bench,
        workload,
        metered,
        guard,
        heads=heads,
        checkpoint=checkpoint,
    )
    total_spend = sum(float(head["cost_usd"]) for head in heads.values())
    complete = not truncated and len(heads) == len(workload)
    abstains = sum(1 for head in heads.values() if head["verdict"] == "abstain")
    print(
        f"[record] heads={len(heads)} abstains={abstains} "
        f"new-spend=${guard.spent_usd:.4f} total=${total_spend:.4f} complete={complete}",
        flush=True,
    )

    admission = (
        run_admission(
            bench,
            workload,
            heads,
            seed=args.seed,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            atoms=atoms,
        )
        if complete
        else incomplete_admission(heads=len(heads), expected_heads=len(workload))
    )
    print(
        f"[admission] classes={admission['class_counts']} "
        f"delta={admission['endpoint']['delta']} decision={admission['decision']}",
        flush=True,
    )
    print(f"  {admission['decision_why']}")

    payload = {
        "schema": E17_SCHEMA,
        "experiment": "E17 held-out admission test for near-functional leakage",
        "pre_registration": "tacet-research-ledger.md E17 (locked 2026-08-23)",
        **provenance,
        "split_rule": "sha256(seed:case_id) parity, 50/50 fit/validation",
        "world_precision_reference": (
            "paper Section rule_precision: firings matching the workload oracle over "
            "the full n-case ground truth"
        ),
        "spend_semantics": (
            "spend_usd is the cumulative measured cost of every persisted head; "
            "spend_this_process_usd is only the fresh API spend in this invocation"
        ),
        "spend_usd": round(total_spend, 6),
        "spend_this_process_usd": round(guard.spent_usd, 6),
        "real_teacher_calls": len(heads),
        "new_teacher_calls": len(heads) - resumed_heads,
        "resumed_heads": resumed_heads,
        "truncated_by_budget": not complete,
        "recording_state": "complete" if complete else "budget-truncated",
        "teacher_abstains": abstains,
        "heads": heads,
        **admission,
    }
    _atomic_write_text(out, json.dumps(payload, indent=2) + "\n")
    log.discard()
    print(
        f"[spend] new=${guard.spent_usd:.4f} cumulative=${total_spend:.4f} "
        f"for {len(heads)} calls; wrote {out}"
    )


def _case_atoms(slots: dict[str, tuple[str, ...]]) -> frozenset[tuple[str, str]]:
    return frozenset((slot, v) for slot, values in slots.items() for v in values)


if __name__ == "__main__":
    main()
