"""Compliance-domain prompt + answer parsing for PrivaCI-Bench teachers.

The teachers' transport/metering is reused unchanged: the compliance prompt
asks for a JSON LIST whose first element is the verdict and the rest are
violated GDPR article numbers, so ``_parse_json_list`` already handles it.
"""

from __future__ import annotations

import re

COMPLIANCE_PROMPT_TEMPLATE = """You are a GDPR compliance auditor.

Read the data-processing scenario below and decide whether the described
processing is permitted or prohibited under the GDPR, and which GDPR
articles are violated (empty if none).

Return ONLY a JSON list of strings, no commentary:
- first element: "permit" or "prohibit"
- remaining elements: violated articles as "artN" (e.g. "art6", "art32")

Scenario:
{head}

Answer:"""


#: NL-strategy (Inter-Cascade-style) weak-model prompt. ``{head}`` is built by
#: the arm as a guidelines block followed by the scenario, so retrieved
#: strategies condition the weak model in-context. The contract stays the same
#: JSON LIST as the frontier teacher, so ``parse_compliance_answer`` is reused
#: unchanged.
NL_STRATEGY_PROMPT_TEMPLATE = """You are a GDPR compliance auditor. Guidelines
distilled from earlier cases may precede the scenario; use them only when
relevant.

{head}

Return ONLY a JSON list of strings, no commentary:
- first element: "permit" or "prohibit"
- remaining elements: violated articles as "artN" (e.g. "art6", "art32")

Answer:"""


def _base_article(label: str) -> str | None:
    m = re.search(r"art(?:icle)?\s*(\d+)", str(label).lower())
    return f"art{m.group(1)}" if m else None


def parse_compliance_answer(answers: list[str]) -> tuple[str, tuple[str, ...]]:
    """Map a teacher's JSON-list answer to ``(verdict, violated_articles)``.

    Unparseable or empty answers become ``("abstain", ())`` so the cascade can
    treat them as a teacher failure rather than a verdict.
    """
    if not answers:
        return ("abstain", ())
    verdict = str(answers[0]).strip().lower()
    if verdict not in ("permit", "prohibit"):
        return ("abstain", ())
    arts = sorted({a for a in (_base_article(x) for x in answers[1:]) if a})
    return (verdict, tuple(arts))


__all__ = [
    "COMPLIANCE_PROMPT_TEMPLATE",
    "NL_STRATEGY_PROMPT_TEMPLATE",
    "parse_compliance_answer",
]
