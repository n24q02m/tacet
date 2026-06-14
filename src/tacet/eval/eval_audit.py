"""Proof-tree validity and auditability evaluation for the symbolic tier (Task 3).

Two metrics over a :class:`~tacet.core.symbolic.SymbolicResult`:

* :func:`proof_validity` — the fraction of an answered query's proof that is
  *grounded*: every top-level derivation reduces, through its proof tree, to a
  base fact or a known/named rule registered in the engine. A score of ``1.0``
  means no dangling or unsupported step; ``0.0`` means unanswered or no proof.
* :func:`proof_coverage` — ``1.0`` when the result is answered with a non-empty
  proof, else ``0.0``. Aggregated across a workload it reports the share of
  answers that ship an explanation.

Honesty note: proof validity measures the *grounding* (provenance integrity) of
the proof the engine produced, **not** the ground-truth correctness of the
answer. A perfectly grounded proof of a wrong fact still scores ``1.0`` — the
metric audits that nothing was invented, which is exactly Tier-1's soundness
guarantee, not that the world graph itself is correct.
"""

from __future__ import annotations

from tacet.core.symbolic import RuleEngine, SymbolicResult, Triple

# A proof step renders as ``FACT     h -r-> t`` or ``DERIVED  h -r-> t   [rule]``;
# indentation (two spaces per depth) marks premises. The top-level entries — one
# per answered triple — carry no leading whitespace.
_ARROW = "-"


def _parse_conclusion(line: str) -> Triple | None:
    """Parse one un-indented proof line ``<KIND><pad>h -r-> t[...]`` into a triple.

    Returns ``None`` when the line does not parse as a conclusion. We split on the
    ``-r->`` arrow so entity names containing other characters survive.
    """
    body = line.split(None, 1)
    if len(body) != 2:
        return None
    payload = body[1]
    # drop a trailing rule annotation "   [rule]" if present
    if payload.rstrip().endswith("]") and "[" in payload:
        payload = payload[: payload.rindex("[")]
    # payload is now "h -r-> t"; the relation is wrapped as "-r->"
    if f" {_ARROW}" not in payload or f"{_ARROW}> " not in payload:
        return None
    head, rest = payload.split(f" {_ARROW}", 1)
    rel, tail = rest.split(f"{_ARROW}> ", 1)
    return (head.strip(), rel.strip(), tail.strip())


def _top_level_conclusions(proof: list[str]) -> list[Triple | None]:
    """Recover every depth-0 conclusion from a proof string list.

    Only un-indented, non-blank lines are top-level conclusions (indented lines
    are premises). Each entry is the parsed triple, or ``None`` when the line is
    a top-level conclusion that fails to parse. A non-parsing conclusion is kept
    as ``None`` rather than dropped, so the validity metric counts it as
    *ungrounded* (it lands in the denominator but never the numerator): a
    conclusion the auditor cannot even read is not evidence of soundness, and
    silently excluding it would let a malformed proof score a perfect 1.0. On the
    clean synthetic / ProofWriter proofs used here every conclusion parses, so
    this is invariant on those workloads and only hardens the metric against
    malformed input.
    """
    return [
        _parse_conclusion(line) for line in proof if not (line[:1].isspace() or not line.strip())
    ]


def proof_validity(engine: RuleEngine, result: SymbolicResult) -> float:
    """Fraction of the answered query's proof that is grounded.

    Returns ``0.0`` when the result is unanswered or carries no proof. Otherwise
    recovers the top-level proof conclusions and checks each one structurally
    against the engine's derivation tree and known rule set: the score is the
    fraction whose every proof step reduces to a base fact or a known rule.
    """
    if not result.answered or not result.proof:
        return 0.0
    conclusions = _top_level_conclusions(result.proof)
    if not conclusions:
        return 0.0
    # the denominator is every top-level conclusion line; a line that fails to
    # parse (``None``) counts as ungrounded, never as grounded.
    grounded = sum(1 for t in conclusions if t is not None and engine.proof_is_grounded(t))
    return grounded / len(conclusions)


def proof_coverage(engine: RuleEngine, result: SymbolicResult) -> float:
    """``1.0`` if the result is answered *and* ships a non-empty proof, else ``0.0``.

    The ``engine`` argument is accepted for a uniform metric signature; coverage
    is a property of the result alone.
    """
    return 1.0 if result.answered and result.proof else 0.0


def proof_derivation_list(result: SymbolicResult) -> list[str]:
    """Flat, expert-checkable view of the proof for the auditability rubric.

    Returns the proof lines stripped of indentation, in derivation order, so a
    human auditor can scan every FACT/DERIVED step without reconstructing tree
    depth. Empty for an unanswered result.
    """
    if not result.answered:
        return []
    return [line.strip() for line in result.proof if line.strip()]
