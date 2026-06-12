"""ProofWriter loader — synthetic deductive-reasoning benchmark (Tafjord et al.,
Findings of ACL 2021).

ProofWriter ships small *theories* — a handful of facts and IF-THEN rules
expressed in templated English plus a structured logical form — together with
boolean questions, each labelled with the gold answer and a full proof chain.
Each question has a known proof depth (``QDep``), so a system can be scored not
only on answer accuracy but on whether it *produces a checkable proof*. That is
exactly what TACET's Tier-1 symbolic engine does: it computes the deductive
closure of the facts under the rules and returns, for every answer, a proof tree
grounded in base facts.

We use the **CWA** (closed-world assumption) variant. Under CWA an atom that the
engine cannot derive is taken to be false (negation-as-failure at the query
level), which is precisely the semantics of ProofWriter-CWA's boolean answers.

Expressivity boundary (honesty-critical)
-----------------------------------------
TACET's symbolic tier is **positive Datalog**: function-free, range-restricted
Horn rules with no negation in the rule body. ProofWriter-CWA expresses the
queried answer's polarity with ``+`` / ``-`` on the *question*, but a subset of
theories also use negation-as-failure *inside a rule body* (polarity marker
``~``, e.g. "if someone is white and not blue then they are young"). Those rules
are **not** expressible in positive Datalog. The loader detects any theory whose
rule atoms (body or head) carry the negation-as-failure marker ``~`` and marks it
``expressible=False`` (the ``-`` marker is query polarity, not a rule head, so it
plays no part in this rule-expressivity test).
The run *excludes* such theories and *reports* the excluded fraction per depth;
nothing is silently dropped.

The data is expected to be already extracted under ``DATA_ROOT`` with the
official directory layout::

    <DATA_ROOT>/CWA/depth-<d>/meta-<split>.jsonl

so this loader performs no download.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tacet.core.graph import WorldGraph
from tacet.core.symbolic import Pattern, Rule

# Repo-local default root for the extracted ProofWriter V2020.12.3 release.
DATA_ROOT = (
    Path(__file__).resolve().parents[3] / "data" / "proofwriter" / "proofwriter-dataset-V2020.12.3"
)

# Tokens that act as the universally-quantified variable in a rule. ProofWriter
# templates use both "something" and "someone"; both map to the single rule
# variable ``?x`` (rules in this dataset are unary in their variable).
_VARIABLE_TOKENS = frozenset({"something", "someone", "somebody", "things"})
_VAR = "?x"

# An atom renders as ("arg0" "verb" "arg1" "polarity"); polarity is one of
# '+' (holds), '~' (negation-as-failure inside a rule), or '-' (negated query).
_ATOM_RE = re.compile(r'\("([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\)')

Atom = tuple[str, str, str, str]  # (arg0, verb, arg1, polarity)


def _parse_atom(s: str) -> Atom:
    """Parse a single lisp-y atom string into ``(arg0, verb, arg1, polarity)``.

    Raises ``ValueError`` if ``s`` does not contain exactly one atom.
    """
    m = _ATOM_RE.search(s)
    if m is None:
        raise ValueError(f"not a ProofWriter atom: {s!r}")
    return (m.group(1), m.group(2), m.group(3), m.group(4))


def _to_pattern(atom: Atom) -> Pattern:
    """Map a positive atom to a symbolic pattern ``(arg0, verb, arg1)``.

    A variable token (``something`` / ``someone``) becomes ``?x``; everything
    else is a constant. Polarity is dropped (callers handle it separately).
    """
    arg0, verb, arg1, _pol = atom
    head = _VAR if arg0 in _VARIABLE_TOKENS else arg0
    tail = _VAR if arg1 in _VARIABLE_TOKENS else arg1
    return (head, verb, tail)


def _parse_rule(rep: str) -> tuple[tuple[Pattern, ...], Pattern, bool]:
    """Parse a rule representation into ``(body_patterns, head_pattern, has_neg)``.

    The representation has the form ``(((body_atom)...) -> (head_atom))``. The
    body is the parenthesised list before ``->``; the head is the single atom
    after it. ``has_neg`` is True when any atom (body or head) carries the ``~``
    negation-as-failure polarity — such a rule is outside positive Datalog.
    """
    lhs, _sep, rhs = rep.partition("->")
    body_atoms = [_parse_atom(m.group(0)) for m in _ATOM_RE.finditer(lhs)]
    head_atoms = [_parse_atom(m.group(0)) for m in _ATOM_RE.finditer(rhs)]
    if not head_atoms:
        raise ValueError(f"rule has no head: {rep!r}")
    has_neg = any(a[3] == "~" for a in body_atoms + head_atoms)
    body = tuple(_to_pattern(a) for a in body_atoms)
    head = _to_pattern(head_atoms[0])
    return body, head, has_neg


@dataclass
class ProofWriterQuestion:
    """A single boolean question and its gold answer under CWA."""

    qid: str
    text: str
    atom: Atom  # (arg0, verb, arg1, polarity) of the queried fact
    answer: bool  # gold boolean under the closed-world assumption
    depth: int  # QDep: proof depth (0 = base fact)


@dataclass
class ProofWriterTheory:
    """One ProofWriter theory: a world graph, its rules, and its questions.

    ``expressible`` is False when any rule uses negation-as-failure (``~``),
    putting it outside positive Datalog; such theories are excluded from a run.
    """

    tid: str
    graph: WorldGraph
    rules: tuple[Rule, ...]
    questions: list[ProofWriterQuestion]
    expressible: bool

    def question_atoms(self) -> list[Atom]:
        return [q.atom for q in self.questions]


@dataclass
class ProofWriterBenchmark:
    """A ProofWriter (depth, split) slice ready to feed the symbolic tier."""

    name: str
    depth: int
    split: str
    theories: list[ProofWriterTheory]
    n_excluded: int = field(default=0)

    def expressible_theories(self) -> list[ProofWriterTheory]:
        return [t for t in self.theories if t.expressible]

    def stats(self) -> dict[str, int]:
        n_q = sum(len(t.questions) for t in self.theories)
        return {
            "theories": len(self.theories),
            "excluded": self.n_excluded,
            "questions": n_q,
        }


def _build_theory(record: dict) -> ProofWriterTheory:
    """Turn one parsed JSONL record into a :class:`ProofWriterTheory`."""
    tid = record["id"]

    # Facts: every triple is a positive "+" fact under CWA. The atom maps to the
    # edge (arg0)-[verb]->(arg1).
    graph = WorldGraph(name=tid)
    for triple in record.get("triples", {}).values():
        arg0, verb, arg1, _pol = _parse_atom(triple["representation"])
        graph.add_edge(arg0, verb, arg1)

    # Rules: parse each; flag negation-as-failure as an expressivity violation.
    rules: list[Rule] = []
    expressible = True
    for key, rule in record.get("rules", {}).items():
        body, head, has_neg = _parse_rule(rule["representation"])
        if has_neg:
            expressible = False
            continue  # not a positive-Datalog rule; theory will be excluded
        rules.append(Rule(name=f"{tid}:{key}", body=body, head=head))

    # Questions: keep the queried atom (with its +/- polarity) and gold answer.
    questions: list[ProofWriterQuestion] = []
    for qid, q in record.get("questions", {}).items():
        atom = _parse_atom(q["representation"])
        questions.append(
            ProofWriterQuestion(
                qid=qid,
                text=q.get("question", ""),
                atom=atom,
                answer=bool(q["answer"]),
                depth=int(q.get("QDep", 0)),
            )
        )

    return ProofWriterTheory(
        tid=tid,
        graph=graph,
        rules=tuple(rules),
        questions=questions,
        expressible=expressible,
    )


def load_proofwriter(
    root: str | Path = DATA_ROOT,
    *,
    depth: int = 2,
    split: str = "dev",
    limit: int | None = None,
) -> ProofWriterBenchmark:
    """Load a ProofWriter-CWA ``(depth, split)`` slice.

    ``root`` is the extracted ``proofwriter-dataset-V2020.12.3`` directory. The
    CWA meta file is read from ``<root>/CWA/depth-<depth>/meta-<split>.jsonl``.
    ``limit`` caps the number of theories read (for fast tests / bounded runs).
    """
    root = Path(root)
    meta_path = root / "CWA" / f"depth-{depth}" / f"meta-{split}.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Expected the extracted ProofWriter "
            f"V2020.12.3 release under {root}."
        )

    theories: list[ProofWriterTheory] = []
    with meta_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            theories.append(_build_theory(record))
            if limit is not None and len(theories) >= limit:
                break

    n_excluded = sum(1 for t in theories if not t.expressible)
    return ProofWriterBenchmark(
        name=f"ProofWriter-CWA-depth{depth}-{split}",
        depth=depth,
        split=split,
        theories=theories,
        n_excluded=n_excluded,
    )


__all__ = [
    "DATA_ROOT",
    "ProofWriterBenchmark",
    "ProofWriterQuestion",
    "ProofWriterTheory",
    "load_proofwriter",
]
