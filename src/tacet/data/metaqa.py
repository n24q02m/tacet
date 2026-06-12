"""MetaQA loader — Movie-domain KGQA benchmark (Zhang et al., AAAI 2018).

MetaQA ships a knowledge base over movies plus natural-language questions
at three hop levels (1-hop, 2-hop, 3-hop). It is the canonical NL→KGQA
benchmark for cost / accuracy studies because (a) the KG is small enough
to fit on a single machine and (b) every question has a gold-standard
entity-set answer, so cascade and LLM-only systems can be compared
directly on accuracy.

Download (one-shot)::

    git clone https://github.com/yuyuz/MetaQA.git data/MetaQA

Then::

    from tacet.data.metaqa import load_metaqa
    bench = load_metaqa("data/MetaQA", hop=1, split="dev")

The loader is permissive: it tolerates both the v1 (``kb.txt``) and v1.1
(``kb_entity_dict.txt`` + ``kb.txt``) folder layouts, and the
``question\\tanswer1|answer2`` line format used by the test/dev/train splits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from tacet.core.graph import WorldGraph

Triple = tuple[str, str, str]
_TOPIC_RE = re.compile(r"\[(.+?)\]")


@dataclass
class MetaQAQuestion:
    question: str  # raw natural-language sentence with [head] marker
    head: str  # the topic entity extracted from [...]
    answers: list[str]  # gold tail entity set
    hop: int  # 1, 2, or 3


@dataclass
class MetaQABenchmark:
    """A MetaQA hop-level split, ready to feed into the cascade."""

    name: str
    hop: int
    split: str  # train | dev | test
    kg: WorldGraph
    questions: list[MetaQAQuestion]
    # entity / relation lookup tables built from kb.txt
    entities: set[str] = field(default_factory=set)
    relations: set[str] = field(default_factory=set)

    def stats(self) -> dict[str, int]:
        return {
            "kg_triples": len(self.kg.edges),
            "kg_entities": len(self.entities),
            "kg_relations": len(self.relations),
            "questions": len(self.questions),
        }


def load_kb(root: str | Path) -> tuple[WorldGraph, set[str], set[str]]:
    """Load MetaQA's ``kb.txt`` (head|relation|tail, one per line)."""
    root = Path(root)
    kb_path = next((p for p in (root / "kb.txt", root / "MetaQA" / "kb.txt") if p.exists()), None)
    if kb_path is None:
        raise FileNotFoundError(f"kb.txt not found under {root}")
    g = WorldGraph(name="MetaQA-KB")
    entities: set[str] = set()
    relations: set[str] = set()
    with kb_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            h, r, t = parts[0].strip(), parts[1].strip(), parts[2].strip()
            g.add_edge(h, r, t)
            entities.add(h)
            entities.add(t)
            relations.add(r)
    return g, entities, relations


def _parse_question_line(line: str, hop: int) -> MetaQAQuestion | None:
    line = line.strip()
    if not line:
        return None
    # standard MetaQA line: "question text [head]?\tanswer1|answer2|..."
    if "\t" not in line:
        return None
    q_part, a_part = line.split("\t", 1)
    m = _TOPIC_RE.search(q_part)
    if not m:
        return None
    head = m.group(1).strip()
    answers = [a.strip() for a in a_part.split("|") if a.strip()]
    return MetaQAQuestion(question=q_part.strip(), head=head, answers=answers, hop=hop)


def load_metaqa(root: str | Path, *, hop: int = 1, split: str = "dev") -> MetaQABenchmark:
    """Load a (hop, split) slice with its shared KB.

    ``root`` is the path you cloned MetaQA into (the repo's top-level
    directory, or the ``MetaQA`` subdirectory thereof — both work).
    """
    if hop not in (1, 2, 3):
        raise ValueError("hop must be 1, 2, or 3")
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be 'train', 'dev', or 'test'")
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} does not exist. Clone with: "
            f"`git clone https://github.com/yuyuz/MetaQA.git {root}`"
        )

    kg, entities, relations = load_kb(root)

    # the v1.0 layout puts questions under `1-hop/`, `2-hop/`, `3-hop/`;
    # try both layouts.
    candidates = [
        root / f"{hop}-hop" / f"qa_{split}.txt",
        root / "MetaQA" / f"{hop}-hop" / f"qa_{split}.txt",
        root / "qa" / f"qa_{split}_{hop}hop.txt",
        root / f"qa_{split}_{hop}hop.txt",
    ]
    q_path = next((p for p in candidates if p.exists()), None)
    if q_path is None:
        raise FileNotFoundError(
            f"qa file for hop={hop} split={split} not found; tried: "
            + ", ".join(str(c) for c in candidates)
        )

    questions: list[MetaQAQuestion] = []
    with q_path.open(encoding="utf-8") as fh:
        for line in fh:
            parsed = _parse_question_line(line, hop)
            if parsed is not None:
                questions.append(parsed)

    return MetaQABenchmark(
        name=f"MetaQA-{hop}hop-{split}",
        hop=hop,
        split=split,
        kg=kg,
        questions=questions,
        entities=entities,
        relations=relations,
    )


__all__ = ["MetaQABenchmark", "MetaQAQuestion", "load_kb", "load_metaqa"]
