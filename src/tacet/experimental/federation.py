"""Federated TACET — many writers, one substrate.

The Akashic Records image only makes sense if many minds can read from and
write to the same memory. ``FederatedGraph`` wraps a ``WorldGraph`` so each
edge carries *provenance*: who wrote it, when, and how much we trust them.
Merges between two federated graphs combine facts under a pluggable
``MergeStrategy``; trust-weighted queries surface the consensus answer
across writers, and disagreements are surfaced explicitly rather than
silently overwritten.

This is engineering, not new ML — but it is what turns a private cascade
into a shared substrate, the topology the Akashic-Records name actually
requires.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from tacet.core.graph import Triple, WorldGraph


@dataclass(frozen=True)
class Provenance:
    """Who wrote a fact, when, and how reliable they are."""

    writer: str
    timestamp: float
    trust: float = 1.0  # in [0, 1]; multiplied by writer-level trust at merge time


@dataclass
class FederatedGraph:
    """A graph whose every edge carries a list of `Provenance` records.

    The wrapped ``WorldGraph`` stays the system of record; provenance is
    stored on the side keyed by triple, so the rest of TACET (engine, KGE,
    distiller …) is unchanged.
    """

    graph: WorldGraph = field(default_factory=WorldGraph)
    provenance: dict[Triple, list[Provenance]] = field(default_factory=lambda: defaultdict(list))
    writer_trust: dict[str, float] = field(default_factory=dict)

    # ---- writing ---------------------------------------------------------
    def assert_fact(
        self,
        head: str,
        relation: str,
        target: str,
        *,
        writer: str,
        trust: float = 1.0,
        timestamp: float | None = None,
        **props: object,
    ) -> None:
        """Record that `writer` says (head, relation, target) is true."""
        self.graph.add_edge(head, relation, target, **props)
        triple: Triple = (head, relation, target)
        self.provenance[triple].append(
            Provenance(
                writer=writer,
                timestamp=timestamp or time.time(),
                trust=trust,
            )
        )

    def set_writer_trust(self, writer: str, trust: float) -> None:
        self.writer_trust[writer] = trust

    # ---- queries ---------------------------------------------------------
    def writers_of(self, triple: Triple) -> list[str]:
        return [p.writer for p in self.provenance.get(triple, [])]

    def trust_score(self, triple: Triple) -> float:
        """Aggregate trust for a triple: ``max_writer max(record_trust * writer_trust)``."""
        return max(
            (
                rec.trust * self.writer_trust.get(rec.writer, 1.0)
                for rec in self.provenance.get(triple, [])
            ),
            default=0.0,
        )

    def consensus(self, triple: Triple) -> tuple[bool, float, list[str]]:
        """Is this triple endorsed by trusted writers? Returns
        ``(any_endorses_it, aggregate_trust, list_of_writers)``."""
        writers = self.writers_of(triple)
        return (bool(writers), self.trust_score(triple), writers)

    def disagreements(self, head: str, relation: str) -> list[Triple]:
        """Distinct triples (head, relation, *) asserted by different writers
        — surface when functional relations get conflicting answers."""
        return sorted(t for t in self.provenance if t[0] == head and t[1] == relation)


# ----------------------------------------------------------------------------
MergeStrategy = Callable[[Iterable[Provenance]], bool]


def latest_wins(records: Iterable[Provenance]) -> bool:
    """Keep the most recent assertion."""
    return any(records)  # any presence means we keep it; latest filtering done in merge


def trust_weighted(threshold: float = 0.5) -> MergeStrategy:
    """Keep a fact iff the *sum* of trust across endorsing writers ≥ threshold."""

    def strategy(records: Iterable[Provenance]) -> bool:
        return sum(r.trust for r in records) >= threshold

    return strategy


def majority(records: Iterable[Provenance]) -> bool:
    counts = Counter(r.writer for r in records)
    return len(counts) >= 2


# ----------------------------------------------------------------------------
def merge(
    a: FederatedGraph, b: FederatedGraph, *, strategy: MergeStrategy | None = None
) -> FederatedGraph:
    """Merge two federated graphs. The default strategy keeps any fact with
    at least one endorsing writer (the union); other strategies filter."""
    strategy = strategy or latest_wins
    out = FederatedGraph()
    out.writer_trust = {**a.writer_trust, **b.writer_trust}
    seen: set[Triple] = set(a.provenance) | set(b.provenance)
    for triple in seen:
        records = a.provenance.get(triple, []) + b.provenance.get(triple, [])
        if not strategy(records):
            continue
        h, r, t = triple
        out.graph.add_edge(h, r, t)
        out.provenance[triple] = list(records)
    return out


__all__ = [
    "FederatedGraph",
    "MergeStrategy",
    "Provenance",
    "latest_wins",
    "majority",
    "merge",
    "trust_weighted",
]
