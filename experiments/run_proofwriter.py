"""ProofWriter deductive-reasoning evaluation with proof validity.

    python experiments/run_proofwriter.py [--limit 300] [--depths 0 1 2 3 5]

What it does
------------
For each ProofWriter-CWA depth split it:

1. Loads the theories (``tacet.data.proofwriter``). Theories whose rules
   use negation-as-failure (the ``~`` polarity marker) are outside positive
   Datalog; they are *excluded* and *counted* per depth (the expressivity
   boundary is reported, never silently dropped).
2. For every expressible theory, builds a Tier-1 ``RuleEngine`` (empty ontology
   --- ProofWriter has no structural axioms) with the theory's rules and
   materialises the deductive closure of its facts.
3. Answers every question under the closed-world assumption: a positive query
   ``(a, v, b, "+")`` is predicted True iff ``b`` is derivable for ``(a, v)``; a
   negated query ``(a, v, b, "-")`` is predicted True iff ``b`` is *not*
   derivable (negation-as-failure). Predictions are scored against the gold
   boolean answer.
4. For each positively-answered question (the queried atom is derivable), scores
   the proof with ``tacet.eval.eval_audit``: ``proof_coverage`` (the answer ships a
   proof) and ``proof_validity`` (every proof step grounds in a base fact or a
   known rule).

It writes ``experiments/results/tab_proofwriter.tex`` (one row per depth) and
updates ``experiments/results/macros.tex``.

Honesty note: accuracy reflects answer correctness against the ProofWriter-CWA
oracle; ``proof_validity`` audits provenance integrity (nothing unsupported
entered a derivation), not answer correctness. On this dataset the gold labels
are themselves a forward-chaining closure, so a sound+complete positive-Datalog
engine is expected to score at the ceiling on the *expressible* subset --- the
contribution is the checkable proof attached to every deductive answer, not a
novel accuracy number.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tacet.core.ontology import Ontology
from tacet.core.symbolic import RuleEngine
from tacet.data.proofwriter import load_proofwriter
from tacet.eval.eval_audit import proof_coverage, proof_validity

_RESULTS = Path(__file__).resolve().parents[1] / "experiments" / "results"


@dataclass
class DepthResult:
    depth: int
    n_theories: int
    n_excluded: int
    n_questions: int
    accuracy: float
    majority_accuracy: float
    proof_coverage: float
    proof_validity: float
    n_answered: int


def _evaluate_depth(depth: int, split: str, limit: int | None) -> DepthResult:
    bench = load_proofwriter(depth=depth, split=split, limit=limit)

    n_questions = 0
    n_correct = 0
    gold_labels: Counter[bool] = Counter()

    coverage_sum = 0.0
    validity_sum = 0.0
    n_answered = 0  # positively-answered (derivable) questions, for proof metrics

    for theory in bench.expressible_theories():
        engine = RuleEngine(Ontology(), list(theory.rules))
        engine.materialise(theory.graph)
        for q in theory.questions:
            arg0, verb, arg1, polarity = q.atom
            result = engine.query(arg0, verb)
            derivable = result.answered and (arg1 in result.answers)
            predicted = derivable if polarity == "+" else (not derivable)

            n_questions += 1
            gold_labels[q.answer] += 1
            if predicted == q.answer:
                n_correct += 1

            # Proof metrics are defined on positively-derived answers: the
            # engine produced a proof tree we can audit for grounding.
            if derivable:
                n_answered += 1
                coverage_sum += proof_coverage(engine, result)
                validity_sum += proof_validity(engine, result)

    accuracy = n_correct / n_questions if n_questions else 0.0
    majority = max(gold_labels.values()) / n_questions if n_questions else 0.0
    coverage = coverage_sum / n_answered if n_answered else 0.0
    validity = validity_sum / n_answered if n_answered else 0.0

    return DepthResult(
        depth=depth,
        n_theories=len(bench.theories),
        n_excluded=bench.n_excluded,
        n_questions=n_questions,
        accuracy=accuracy,
        majority_accuracy=majority,
        proof_coverage=coverage,
        proof_validity=validity,
        n_answered=n_answered,
    )


def _write_table(path: Path, rows: list[DepthResult]) -> None:
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Depth & $n$ & Excl.\ & Acc. & Proof-cov. & Proof-val. \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r.depth} & {r.n_questions} & {r.n_excluded} & "
            f"{r.accuracy:.3f} & {r.proof_coverage:.3f} & {r.proof_validity:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_macros(path: Path, macros: dict[str, str]) -> None:
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [ln for ln in existing if not any(f"{{{name}}}" in ln for name in macros)]
    for name, value in macros.items():
        kept.append(f"\\renewcommand{{{name}}}{{{value}}}")
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    ap.add_argument("--split", default="dev")
    ap.add_argument(
        "--limit", type=int, default=300, help="cap theories per depth (None-like 0 = all)"
    )
    args = ap.parse_args()
    limit = args.limit if args.limit > 0 else None

    rows: list[DepthResult] = []
    tot_theories = 0
    tot_excluded = 0
    tot_questions = 0
    tot_correct = 0.0
    for depth in args.depths:
        r = _evaluate_depth(depth, args.split, limit)
        rows.append(r)
        tot_theories += r.n_theories
        tot_excluded += r.n_excluded
        tot_questions += r.n_questions
        tot_correct += r.accuracy * r.n_questions
        print(
            f"depth {depth}: theories={r.n_theories} excluded={r.n_excluded} "
            f"({r.n_excluded / r.n_theories:.1%}) questions={r.n_questions} "
            f"acc={r.accuracy:.3f} majority={r.majority_accuracy:.3f} "
            f"proof_cov={r.proof_coverage:.3f} proof_val={r.proof_validity:.3f} "
            f"(answered={r.n_answered})"
        )

    overall_acc = tot_correct / tot_questions if tot_questions else 0.0
    overall_excl_frac = tot_excluded / tot_theories if tot_theories else 0.0
    # macros aggregate over expressible questions only (the evaluated set).
    overall_cov = (
        sum(r.proof_coverage * r.n_answered for r in rows) / sum(r.n_answered for r in rows)
        if any(r.n_answered for r in rows)
        else 0.0
    )
    overall_val = (
        sum(r.proof_validity * r.n_answered for r in rows) / sum(r.n_answered for r in rows)
        if any(r.n_answered for r in rows)
        else 0.0
    )

    print(
        f"OVERALL: acc={overall_acc:.3f} "
        f"excluded={tot_excluded}/{tot_theories} ({overall_excl_frac:.1%}) "
        f"proof_cov={overall_cov:.3f} proof_val={overall_val:.3f}"
    )

    _RESULTS.mkdir(parents=True, exist_ok=True)
    _write_table(_RESULTS / "tab_proofwriter.tex", rows)
    _update_macros(
        _RESULTS / "macros.tex",
        {
            r"\pwAccOverall": f"{overall_acc:.3f}",
            r"\pwProofCov": f"{overall_cov:.3f}",
            r"\pwProofVal": f"{overall_val:.3f}",
            r"\pwExclFrac": f"{overall_excl_frac:.3f}",
            r"\pwExclPct": f"{overall_excl_frac * 100:.1f}",
        },
    )
    print(f"wrote {_RESULTS / 'tab_proofwriter.tex'}")
    print(f"updated {_RESULTS / 'macros.tex'}")


if __name__ == "__main__":
    main()
