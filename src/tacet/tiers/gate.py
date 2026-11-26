"""Entailment gate decorator verifying tier predictions against `RuleEngine`."""

from __future__ import annotations

from tacet.core.symbolic import EntailmentCheck, RuleEngine
from tacet.tiers.base import Tier, TierResult


class EntailmentGate:
    """Wraps an inner `Tier` and validates its answers via `RuleEngine.entails`."""

    def __init__(self, inner: Tier, engine: RuleEngine, strict: bool = True) -> None:
        self.inner = inner
        self.engine = engine
        self.strict = strict
        self.name = f"gate({inner.name})"

    @property
    def tier_id(self) -> int:
        return self.inner.tier_id

    def can_handle(self, head: str, relation: str) -> bool:
        return self.inner.can_handle(head, relation)

    def answer(self, head: str, relation: str) -> TierResult | None:
        res = self.inner.answer(head, relation)
        if res is None:
            return None

        entailed_answers: list[str] = []
        checks: list[EntailmentCheck] = []
        proof_lines: list[str] = []
        seen_lines: set[str] = set()
        seen_answers: set[str] = set()

        for ans in res.answers:
            chk = self.engine.entails(head, relation, ans)
            if chk.entailed:
                if ans not in seen_answers:
                    seen_answers.add(ans)
                    entailed_answers.append(ans)
                    checks.append(chk)
                for line in chk.proof:
                    if line not in seen_lines:
                        seen_lines.add(line)
                        proof_lines.append(line)

        if not entailed_answers:
            if self.strict:
                return None
            return TierResult(
                tier=res.tier,
                answers=list(res.answers),
                confidence=res.confidence,
                proof=list(res.proof),
                grounded=False,
                note="unentailed",
            )

        grounded = bool(entailed_answers) and all(chk.grounded for chk in checks)
        return TierResult(
            tier=res.tier,
            answers=entailed_answers,
            confidence=res.confidence,
            proof=proof_lines,
            grounded=grounded,
            note=res.note or "entailed",
        )
