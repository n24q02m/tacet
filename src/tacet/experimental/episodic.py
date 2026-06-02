"""Episodic memory — record every interaction; learn from feedback over time.

The cascade's deductive closure is *semantic memory* — what is true, plus the
rules that derive what else is true. It is timeless. TACET agents also need
*episodic memory*: the trajectory of interactions — every query asked, the
tier that answered, the answer returned, whether the user (or downstream
process) judged the answer correct, and the latency / cost it cost.

Two concrete uses for this:

1. **Audit and explainability over time.** Compliance settings (e.g.\\
   an "audit log per request") need the answer record itself,
   not just the proof tree. ``EpisodicStore`` is that record.
2. **Feedback-driven rule retirement.** When the user marks an answer wrong,
   the offending rule (or fact) is identified and demoted: its trust score
   drops; if it accumulates enough negative feedback it is *retired* from
   the active rule set. Rules with sustained positive feedback are kept.
   This is how TACET learns from real-world use, not just from the teacher.

The store is in-memory by default; persistence is a one-method extension
(``save_jsonl`` / ``load_jsonl`` are included).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Episode:
    """One recorded interaction with the cascade."""

    id: int
    timestamp: float
    head: str
    relation: str
    tier: int
    answers: list[str]
    cost: float
    latency_ms: float
    note: str = ""
    proof_rules: list[str] = field(default_factory=list)
    feedback: dict[str, object] = field(default_factory=dict)

    def mark_correct(self, by: str = "user") -> None:
        self.feedback = {"correct": True, "by": by, "at": time.time()}

    def mark_wrong(self, by: str = "user", reason: str = "") -> None:
        self.feedback = {"correct": False, "by": by, "at": time.time(), "reason": reason}


class EpisodicStore:
    """An append-only log of `Episode`s with simple query helpers."""

    def __init__(self) -> None:
        self._episodes: list[Episode] = []

    # ---- ingest ----------------------------------------------------------
    def record(
        self,
        head: str,
        relation: str,
        tier: int,
        answers: list[str],
        cost: float,
        latency_ms: float,
        note: str = "",
        proof_rules: list[str] | None = None,
    ) -> Episode:
        ep = Episode(
            id=len(self._episodes),
            timestamp=time.time(),
            head=head,
            relation=relation,
            tier=tier,
            answers=list(answers),
            cost=cost,
            latency_ms=latency_ms,
            note=note,
            proof_rules=list(proof_rules or []),
        )
        self._episodes.append(ep)
        return ep

    # ---- read ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self):
        return iter(self._episodes)

    def all(self) -> list[Episode]:
        return list(self._episodes)

    def for_query(self, head: str, relation: str) -> list[Episode]:
        return [e for e in self._episodes if e.head == head and e.relation == relation]

    def in_window(self, start: float, end: float) -> list[Episode]:
        return [e for e in self._episodes if start <= e.timestamp < end]

    def with_feedback(self) -> list[Episode]:
        return [e for e in self._episodes if e.feedback]

    # ---- persistence ------------------------------------------------------
    def save_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        with p.open("w", encoding="utf-8") as fh:
            for e in self._episodes:
                fh.write(json.dumps(asdict(e), default=str) + "\n")

    def load_jsonl(self, path: str | Path) -> None:
        self._episodes = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            self._episodes.append(Episode(**d))

    # ---- summary ----------------------------------------------------------
    def summary(self) -> dict[str, object]:
        n = len(self._episodes)
        if n == 0:
            return {"queries": 0}
        by_tier = defaultdict(int)
        total_cost = total_lat = 0.0
        marked = correct = 0
        for e in self._episodes:
            by_tier[e.tier] += 1
            total_cost += e.cost
            total_lat += e.latency_ms
            if e.feedback:
                marked += 1
                if e.feedback.get("correct"):
                    correct += 1
        return {
            "queries": n,
            "tier_counts": dict(by_tier),
            "avg_cost": total_cost / n,
            "avg_latency_ms": total_lat / n,
            "feedback_received": marked,
            "feedback_accuracy": correct / marked if marked else None,
        }


# --- feedback-driven rule curation ------------------------------------------
@dataclass
class RuleScore:
    """Running trust score for a rule, learnt from user feedback."""

    rule: str
    positives: int = 0
    negatives: int = 0

    @property
    def trust(self) -> float:
        """Wilson-lower-bound-ish score in [0, 1] — conservative for small n."""
        n = self.positives + self.negatives
        if n == 0:
            return 0.5
        p = self.positives / n
        # cheap pessimistic adjustment so a single +1 doesn't jump trust to 1.0
        return max(0.0, p - 1.0 / (n + 2))


class FeedbackCurator:
    """Promote / retire synthesised rules from the episodic feedback signal.

    Every episode whose answer came through Tier-1 carries the names of the
    rules whose firing produced that answer. Aggregated user judgements on
    those episodes give each rule a trust score; rules below ``retire_below``
    after at least ``min_observations`` judgements are removed from the
    active rule set.
    """

    def __init__(self, retire_below: float = 0.5, min_observations: int = 5) -> None:
        self.retire_below = retire_below
        self.min_observations = min_observations
        self.scores: dict[str, RuleScore] = {}

    def absorb(self, episodes: Iterable[Episode]) -> None:
        for e in episodes:
            if not e.feedback or "correct" not in e.feedback:
                continue
            for rule in e.proof_rules:
                score = self.scores.setdefault(rule, RuleScore(rule))
                if e.feedback["correct"]:
                    score.positives += 1
                else:
                    score.negatives += 1

    def rules_to_retire(self) -> list[str]:
        return [
            r
            for r, s in self.scores.items()
            if (s.positives + s.negatives) >= self.min_observations and s.trust < self.retire_below
        ]

    def trusted_rules(self, threshold: float = 0.7) -> list[str]:
        return [
            r
            for r, s in self.scores.items()
            if (s.positives + s.negatives) >= self.min_observations and s.trust >= threshold
        ]


__all__ = [
    "Episode",
    "EpisodicStore",
    "FeedbackCurator",
    "RuleScore",
]
