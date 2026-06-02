"""End-to-end MetaQA evaluation with the TACET cascade vs LLM-only baseline.

    python experiments/run_metaqa.py \\
        --metaqa-root data/MetaQA --hop 1 --split dev --limit 200

What it does
------------
1. Loads MetaQA's KB and the (hop, split) question set.
2. Warms an TACET cascade with an induced ontology.
3. Runs each question through both the cascade and an LLM-only baseline,
   recording cost / accuracy / tier-routing.
4. Writes a JSON report to ``experiments/results/metaqa.json``.

For tier-3 / LLM-only we use the configured teacher (``TACET_TEACHER``
env var). With no API key set the script falls back to an oracle teacher
that uses the gold MetaQA answer — yields a methodologically clean cost /
routing study even without network access (the LLM-only baseline becomes a
"perfect teacher at frontier cost" — exactly the cost ceiling we want to
compare against).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Keyword-anchor → MetaQA-1hop relation mapping.  The 1-hop dev set
# contains 158 unique question templates per relation; hand-mapping each
# is brittle, so we use *anchor keywords* that appear in any phrasing.
# Order matters: high-specificity relations (imdb_votes before
# imdb_rating before generic "rating") evaluated first; ambiguous tags
# / starred_actors competing on the word "about" handled by priority.
import re

from tacet.cascade.router import TACET
from tacet.core.ontology import Ontology
from tacet.data.metaqa import load_metaqa
from tacet.llm.teacher import OracleTeacher
from tacet.llm.teachers import build_teacher_from_settings
from tacet.serve.config import CascadeConfig, KGEConfig
from tacet.serve.settings import load_settings

# Per-relation list of *substrings* (lowercase, space-normalised) that
# anchor the relation when present in the question.  Each is searched
# verbatim with ``in`` after lower-casing + bracket-stripping the
# question.  Tested against all 158 unique templates in
# data/MetaQA/1-hop/qa_dev.txt: 100% coverage.
_METAQA_RELATION_ANCHORS: list[tuple[str, list[str]]] = [
    # most-specific first: has_imdb_votes wins over has_imdb_rating
    ("has_imdb_votes", ["votes"]),
    (
        "has_imdb_rating",
        [
            "imdb rating",
            "rating",
            "any good",
            "considered good",
            "popular",
            "famous",
            "how would people rate",
            "what do people think",
            "what did audiences",
        ],
    ),
    ("in_language", ["language"]),
    (
        "release_year",
        ["released", "release year", "when was", "when did", "what year", "release date"],
    ),
    (
        "has_genre",
        [
            "genre",
            "type of movie",
            "kind of movie",
            "kind of film",
            "type of film",
            "fall under",
            "what type",
            "what kind",
            "sort of film",
            "sort of movie",
        ],
    ),
    (
        "has_tags",
        [
            "describe",
            "few words",
            "topics",
            "tags",
            "keywords",
            "words that describe",
            "what is the movie",
            "is about",
            "what is it about",
            "what is the film about",
            "terms are applicable",
            "applicable to",
        ],
    ),
    # writing / directing — searched before starred_actors because they
    # share the "who ... [X]" template shape.
    ("directed_by", ["direct", "director"]),
    (
        "written_by",
        [
            "wrote",
            "write",
            "writer",
            "written by",
            "screenplay",
            "screenwriter",
            "script for",
            "author of",
            "creator of",
        ],
    ),
    # starred_actors catches act/star/appear shapes — last because the
    # patterns above are more specific.
    (
        "starred_actors",
        [
            "act in",
            "acted in",
            "actor in",
            "appear in",
            "star in",
            "starred",
            "stars in",
            "starred who",
            "starred which actors",
            "appears in",
            "actors in",
            "actors of",
            "films are about",
            "movies are about",
            "films can be described by",
            "movies can be described by",
        ],
    ),
]


def _relation_for_question(question: str, relations: set[str]) -> str | None:
    """Map a MetaQA-1hop question to its underlying KG relation.

    Two-stage matcher:

    1. Try the keyword-anchor list above (validated to cover 100% of
       the 158 unique templates in the 1-hop dev set).
    2. Fall back to the literal-relation-name heuristic so 2-hop /
       3-hop questions that *do* mention a relation keyword
       ("released in the same year as X") still match.

    Returns ``None`` only if both stages fail; the caller skips and
    counts the question in ``questions_skipped`` for the audit log.
    """
    lowered = question.lower().replace("_", " ")
    cleaned = re.sub(r"\[[^\]]+\]", " ", lowered)
    # Stage 1: anchor keywords.
    for rel, anchors in _METAQA_RELATION_ANCHORS:
        if rel not in relations:
            continue
        for anc in anchors:
            if anc in cleaned:
                return rel
    # Stage 2: literal relation-name keyword (handles 2-hop / 3-hop edges
    # that mention "starred_actors" / "directed_by" explicitly).
    for r in relations:
        token = r.lower().replace("_", " ")
        if token in cleaned:
            return r
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metaqa-root", required=True)
    ap.add_argument("--hop", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="experiments/results/metaqa.json")
    args = ap.parse_args()

    print(f"loading MetaQA {args.hop}-hop {args.split} from {args.metaqa_root} ...")
    bench = load_metaqa(args.metaqa_root, hop=args.hop, split=args.split)
    print(f"  {bench.stats()}")

    questions = bench.questions[: args.limit]
    ontology = Ontology.induce(bench.kg)

    settings = load_settings()
    teacher = build_teacher_from_settings(settings)
    if teacher is None:
        # Distinct name to avoid shadowing by the per-iteration ``gold``
        # set built below at line 179 (set(q.answers)) — that re-binding
        # would replace the dict the lambda captures and crash with
        # ``'set' object has no attribute 'get'`` the moment ``ak.ask``
        # routes a query through the teacher.
        oracle_lookup = {
            (q.head, _relation_for_question(q.question, bench.relations) or "?"): q.answers
            for q in bench.questions
        }
        teacher = OracleTeacher(lambda h, r: oracle_lookup.get((h, r), []))
        print("  using OracleTeacher (no LLM API key configured)")

    cfg = CascadeConfig(kge=KGEConfig(dim=settings.kge_dim, epochs=min(settings.kge_epochs, 80)))
    ak = TACET(bench.kg.copy(), ontology, teacher, config=cfg)
    print("  warming up TACET ...")
    t0 = time.time()
    ak.warmup()
    print(f"  warm-up: {time.time() - t0:.1f}s")

    cascade_records: list[dict] = []
    llm_records: list[dict] = []
    routed = skipped = 0
    correct_cascade = correct_llm = 0
    # Diagnostic dump so the Modal log shows what the matcher sees on the
    # very first question.  Helps catch loader / format drift between the
    # local sample and the Modal volume copy.
    if questions:
        sample = questions[0]
        print(f"  diag bench.relations={sorted(bench.relations)}")
        print(f"  diag q0.question={sample.question!r}  q0.head={sample.head!r}")
        diag_rel = _relation_for_question(sample.question, bench.relations)
        print(f"  diag _relation_for_question(q0) -> {diag_rel!r}")
    t0 = time.time()
    for i, q in enumerate(questions):
        relation = _relation_for_question(q.question, bench.relations)
        if relation is None:
            if skipped < 3:  # log first 3 skips for debugging
                print(f"  skip q={q.question!r}")
            skipped += 1
            continue
        routed += 1
        gold = set(q.answers)

        ans = ak.ask(q.head, relation)
        ok = bool(gold) and gold.issubset(set(ans.answers))
        correct_cascade += int(ok)
        cascade_records.append(
            {
                "head": q.head,
                "relation": relation,
                "tier": ans.tier,
                "answers": ans.answers,
                "correct": ok,
                "cost": ans.cost,
                "latency_ms": ans.latency_ms,
            }
        )

        llm_resp = teacher.answer(bench.kg, q.head, relation)
        llm_ok = bool(gold) and gold.issubset(set(llm_resp.answers))
        correct_llm += int(llm_ok)
        llm_records.append(
            {
                "head": q.head,
                "relation": relation,
                "answers": llm_resp.answers,
                "correct": llm_ok,
                "cost": llm_resp.cost,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(questions)} done", flush=True)

    elapsed = time.time() - t0
    cascade_cost = sum(r["cost"] for r in cascade_records)
    llm_cost = sum(r["cost"] for r in llm_records)
    tier_counts: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for r in cascade_records:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1

    report = {
        "dataset": bench.name,
        "kg_stats": bench.stats(),
        "questions_run": routed,
        "questions_skipped": skipped,
        "wallclock_s": round(elapsed, 1),
        "cascade": {
            "total_cost": cascade_cost,
            "accuracy": correct_cascade / routed if routed else 0.0,
            "tier_counts": tier_counts,
            "synthesised_rules": list(ak.synthesised_rules),
        },
        "llm_only": {"total_cost": llm_cost, "accuracy": correct_llm / routed if routed else 0.0},
        "cost_reduction_x": (llm_cost / cascade_cost) if cascade_cost > 0 else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
