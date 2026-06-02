"""Auto-ingestion — turn a stream of documents into a `WorldGraph`.

Two extractor kinds cover the spectrum:

* ``RuleBasedExtractor`` — deterministic regex/pattern extraction. Useful for
  reproducible demos and tests, for highly templated text (logs, structured
  prose), and as a cheap first pass that an LLM can audit.
* ``CallableExtractor`` — wraps any ``(text) -> list[Triple]`` function. This
  is the production hook: drop a Gemini / Grok / GPT NER + relation extractor
  in here and the rest of the pipeline is unchanged. Pair it with the
  ontology, optionally, and ``KGBuilder`` will reject triples whose relation
  is not declared or whose endpoints fail the type / functional checks.

The pipeline is intentionally thin. Production-grade KG construction is a
research area in itself (GraphRAG, Diffbot, REBEL, …); this module provides
the surface area so a richer extractor can be slotted in without disturbing
the cascade above it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology

Triple = tuple[str, str, str]


class Extractor(Protocol):
    def extract(self, text: str) -> list[Triple]: ...


# ---------------------------------------------------------------------------
@dataclass
class Pattern:
    """A regex with named groups ``head`` and ``tail`` mapped to a relation.

    Optional ``head_type`` / ``tail_type`` annotate the entity types so the
    resulting graph carries typed nodes.

    Example::

        Pattern(r"(?P<head>\\w+) borders (?P<tail>\\w+)",
                relation="borders",
                head_type="Country", tail_type="Country")
    """

    regex: str
    relation: str
    head_type: str = "Entity"
    tail_type: str = "Entity"
    flags: int = re.IGNORECASE
    _compiled: re.Pattern[str] | None = field(default=None, init=False, repr=False)

    def matches(self, text: str) -> list[tuple[str, str]]:
        if self._compiled is None:
            self._compiled = re.compile(self.regex, self.flags)
        return [(m.group("head"), m.group("tail")) for m in self._compiled.finditer(text)]


class RuleBasedExtractor:
    """Deterministic offline extractor — applies a list of patterns to text."""

    def __init__(self, patterns: list[Pattern]) -> None:
        self.patterns = patterns

    def extract(self, text: str) -> list[Triple]:
        out: list[Triple] = []
        for pat in self.patterns:
            for h, t in pat.matches(text):
                out.append((h.strip(), pat.relation, t.strip()))
        return out


class CallableExtractor:
    """Wraps any ``(text) -> list[Triple]`` callable — the LLM integration hook.

    Example (production)::

        from google import genai
        client = genai.Client()

        def gemini(text):
            ...  # call client, parse a JSON list of [head, relation, tail]
            return parsed_triples

        extractor = CallableExtractor(gemini)
    """

    def __init__(self, fn: Callable[[str], list[Triple]]) -> None:
        self._fn = fn

    def extract(self, text: str) -> list[Triple]:
        return list(self._fn(text))


# ---------------------------------------------------------------------------
@dataclass
class IngestionReport:
    documents: int = 0
    triples_extracted: int = 0
    triples_accepted: int = 0
    triples_rejected_unknown_relation: int = 0
    triples_rejected_type_violation: int = 0
    nodes_added: int = 0
    edges_added: int = 0
    # Populated only when an ingestion run uses a ``PIIRedactor``.  We
    # initialise lazily in __post_init__ because ``RedactionReport`` is
    # defined below this class.
    pii_redactions: RedactionReport | None = None

    def __post_init__(self) -> None:
        if self.pii_redactions is None:
            self.pii_redactions = RedactionReport()

    def __str__(self) -> str:
        return (
            f"IngestionReport(docs={self.documents}, "
            f"extracted={self.triples_extracted}, "
            f"accepted={self.triples_accepted}, "
            f"added={self.edges_added} edges / {self.nodes_added} nodes, "
            f"rejected unknown_rel={self.triples_rejected_unknown_relation}, "
            f"type_violation={self.triples_rejected_type_violation})"
        )


# ---------------------------------------------------------------------------
# PII redaction (G2.4) — strip emails / phones / SSNs / credit cards from
# incoming text *before* the extractor sees it.  Replacing each match with a
# tagged placeholder keeps the surrounding context intact for relation
# extraction (e.g. "person works at <COMPANY>" stays a usable triple even
# after "alice@example.com" → "<EMAIL_1>").
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # RFC-5322-lite email — unambiguous (the ``@`` rules out the other
    # numeric patterns), so we run it first.
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # US SSN — strict ``NNN-NN-NNNN`` shape; matched before PHONE so
    # the more permissive phone pattern doesn't capture it first.
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Luhn-shaped credit-card numbers — 13-19 digits with optional
    # ``-`` / `` `` separators.  We don't verify the checksum, just
    # blank anything that *looks* like a card so it never reaches the
    # graph.  Matched before PHONE to win on long card numbers.
    ("CC", re.compile(r"\b(?:\d[ -]?){13,19}\d\b")),
    # E.164 phone numbers + the most common North-American / EU forms.
    # Anchored at word boundaries so it doesn't bite into longer
    # numeric runs that are CCs.
    ("PHONE", re.compile(r"\b\+?\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b")),
]


@dataclass
class RedactionReport:
    """Per-document tally of redactions; surfaces in ``IngestionReport``."""

    counts: dict[str, int] = field(default_factory=dict)

    def merge(self, other: RedactionReport) -> RedactionReport:
        for k, v in other.counts.items():
            self.counts[k] = self.counts.get(k, 0) + v
        return self


class PIIRedactor:
    """Strip e-mails / phones / SSNs / credit cards from raw text.

    Each detected span is replaced with ``<KIND_N>`` so the surrounding
    extractor still sees a unique placeholder per occurrence (useful when
    the extractor wants to recover order).  ``redact`` returns the cleaned
    text *and* a ``RedactionReport`` so callers can audit how much PII
    crossed the wire.

    The redactor is conservative — it has zero entity-level intelligence
    (no Presidio / spaCy NER) and runs in pure-regex Python.  For tighter
    PII coverage (names, locations, custom regulated fields), wrap a
    Presidio analyser around it; the redactor's interface is just
    ``redact(text) -> (cleaned, report)``.
    """

    def __init__(self, patterns: list[tuple[str, re.Pattern[str]]] | None = None) -> None:
        self.patterns = patterns if patterns is not None else _PII_PATTERNS

    def redact(self, text: str) -> tuple[str, RedactionReport]:
        report = RedactionReport()
        out = text
        for kind, pat in self.patterns:
            count_holder = [0]

            def _sub(
                match: re.Match[str], kind_: str = kind, counter: list[int] = count_holder
            ) -> str:
                counter[0] += 1
                return f"<{kind_}_{counter[0]}>"

            out = pat.sub(_sub, out)
            if count_holder[0]:
                report.counts[kind] = report.counts.get(kind, 0) + count_holder[0]
        return out, report


# ---------------------------------------------------------------------------
class KGBuilder:
    """Run an extractor across a corpus and write the result into a `WorldGraph`.

    When an ``Ontology`` is supplied, every extracted triple is checked: the
    relation must be declared, and (after assigning the patterned head/tail
    types) the edge must satisfy the ontology's domain / range / functional
    constraints. Rejected triples are counted in the returned report.

    Optionally accepts a ``PIIRedactor`` to strip PII from raw text *before*
    the extractor runs — required when ingesting user-submitted content
    (BYOD compliance, GDPR §32, etc.).  The redactor leaves placeholders
    in place so the extractor still sees a structured sentence shape.
    """

    def __init__(
        self,
        extractor: Extractor,
        ontology: Ontology | None = None,
        type_hints: dict[str, tuple[str, str]] | None = None,
        redactor: PIIRedactor | None = None,
    ) -> None:
        self.extractor = extractor
        self.ontology = ontology
        # type_hints: relation -> (head_type, tail_type)
        self.type_hints = type_hints or {}
        self.redactor = redactor

    def ingest(
        self, texts: Iterable[str], into: WorldGraph | None = None
    ) -> tuple[WorldGraph, IngestionReport]:
        graph = into if into is not None else WorldGraph(name="ingested")
        report = IngestionReport()
        for text in texts:
            report.documents += 1
            if self.redactor is not None:
                text, red = self.redactor.redact(text)
                report.pii_redactions.merge(red)
            triples = self.extractor.extract(text)
            report.triples_extracted += len(triples)
            for head, relation, tail in triples:
                report.triples_accepted += 1
                if self.ontology is not None and self.ontology.relation(relation) is None:
                    report.triples_rejected_unknown_relation += 1
                    report.triples_accepted -= 1
                    continue
                h_type, t_type = self.type_hints.get(relation, ("Entity", "Entity"))
                nodes_before = graph.stats()["nodes"]
                graph.add_node(head, h_type)
                graph.add_node(tail, t_type)
                if self.ontology is not None and not self.ontology.allows(
                    graph, head, relation, tail
                ):
                    report.triples_rejected_type_violation += 1
                    report.triples_accepted -= 1
                    continue
                edges_before = graph.stats()["edges"]
                graph.add_edge(head, relation, tail)
                edges_after = graph.stats()["edges"]
                report.nodes_added += graph.stats()["nodes"] - nodes_before
                report.edges_added += edges_after - edges_before
        return graph, report


__all__ = [
    "CallableExtractor",
    "Extractor",
    "IngestionReport",
    "KGBuilder",
    "Pattern",
    "PIIRedactor",
    "RedactionReport",
    "RuleBasedExtractor",
]
