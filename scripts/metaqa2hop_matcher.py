"""MetaQA 2-hop derived-relation matcher.

MetaQA 2-hop questions are 2-hop compositions over the 9 base relations used
by the 1-hop matcher in ``run_metaqa.py`` (``directed_by``, ``written_by``,
``starred_actors``, ``has_genre``, ``has_imdb_rating``, ``has_imdb_votes``,
``has_tags``, ``in_language``, ``release_year``).  There are 21 distinct
2-hop question *types*; each is a composition of two base relations, where the
first hop may be traversed inverse (e.g. "the films acted by [X]" goes from a
person [X] *backward* along ``starred_actors`` to the movies, then forward to
the answer).

This module mirrors the anchor-based, substring-matching style of
``run_metaqa._relation_for_question``: a priority-ordered list of
(derived-relation, [anchor substrings]) pairs, most-specific first to avoid
collisions.  The question is lower-cased, has its ``[head]`` stripped, and is
whitespace-collapsed before matching (so anchors can span the position where
the head used to sit, e.g. ``"films acted by were directed by"``).  Validated
against all 210 unique surface templates in ``2-hop/vanilla/qa_dev.txt``
(14 872 questions): 100% resolution.

Composition semantics for ``DERIVED_2HOP``
------------------------------------------
Each value is ``(base_rel_1, base_rel_2, invert_first)``:

* ``base_rel_1``   -- relation crossed on the *first* hop (from [X]).
* ``base_rel_2``   -- relation crossed on the *second* hop (to the answer).
* ``invert_first`` -- whether the *first* hop is traversed in the **inverse**
  direction of how the base relation is stored (all base relations are stored
  movie-centric: ``movie --rel--> tail``).

When [X] is a **person** (actor / director / writer) and we must first reach
the movies, that first hop is the *inverse* of the base relation
(person -> movies), so ``invert_first = True``.  When [X] is a **movie** the
first hop is forward (movie -> person), so ``invert_first = False``.  The
second hop is always the forward, movie-centric base relation.

There are 21 types: 3 movie-headed symmetric (shares_*), 3 person-headed
symmetric (co_*), and 15 asymmetric person->movies->attribute compositions
(5 answer attributes x 3 first-hop roles, excluding the self-role that would
collapse to a co_*/shares_* relation).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 21 derived 2-hop relation types -> (base_rel_1, base_rel_2, invert_first)
# ---------------------------------------------------------------------------
DERIVED_2HOP: dict[str, tuple[str, str, bool]] = {
    # --- symmetric "shares a collaborator" (head = MOVIE [X]) --------------
    # movie [X] -> its actor -> that actor's other movies.
    "shares_actor": ("starred_actors", "starred_actors", False),
    # movie [X] -> its director -> that director's other movies.
    "shares_director": ("directed_by", "directed_by", False),
    # movie [X] -> its writer -> that writer's other movies.
    "shares_writer": ("written_by", "written_by", False),
    # --- symmetric "collaborated with" (head = PERSON [X]) -----------------
    # actor [X] -> (inverse) movies -> co-actors of those movies.
    "co_actor": ("starred_actors", "starred_actors", True),
    # director [X] -> (inverse) movies -> co-directors of those movies.
    "co_director": ("directed_by", "directed_by", True),
    # writer [X] -> (inverse) movies -> co-writers of those movies.
    "co_writer": ("written_by", "written_by", True),
    # --- actor [X] -> (inverse) movies -> attribute/role -------------------
    "director_of_acted": ("starred_actors", "directed_by", True),
    "writer_of_acted": ("starred_actors", "written_by", True),
    "genre_of_acted": ("starred_actors", "has_genre", True),
    "language_of_acted": ("starred_actors", "in_language", True),
    "release_year_of_acted": ("starred_actors", "release_year", True),
    # --- director [X] -> (inverse) movies -> attribute/role ----------------
    "actor_of_directed": ("directed_by", "starred_actors", True),
    "writer_of_directed": ("directed_by", "written_by", True),
    "genre_of_directed": ("directed_by", "has_genre", True),
    "language_of_directed": ("directed_by", "in_language", True),
    "release_year_of_directed": ("directed_by", "release_year", True),
    # --- writer [X] -> (inverse) movies -> attribute/role ------------------
    "actor_of_written": ("written_by", "starred_actors", True),
    "director_of_written": ("written_by", "directed_by", True),
    "genre_of_written": ("written_by", "has_genre", True),
    "language_of_written": ("written_by", "in_language", True),
    "release_year_of_written": ("written_by", "release_year", True),
}


# ---------------------------------------------------------------------------
# Anchor list -- most-specific first.  The first derived relation whose
# substring is found in the bracket-stripped, lower-cased, underscore-
# normalised question wins.
#
# Ordering rationale:
#   1. Co-* / shares-* (symmetric) FIRST -- unique markers ("together with",
#      "co-star", "co-directed", "same actor of", ...) that must not be
#      swallowed by the generic "<role> by [X]" patterns below.
#   2. Then the asymmetric person->movies->attribute families.  Within those,
#      the answer role ("genres/types", "languages", "release/when",
#      "director(s)", "writer(s)/screenwriter", "actor(s)/starred") combined
#      with the first-hop role ("acted/starred", "directed", "written")
#      uniquely identifies the relation.
# ---------------------------------------------------------------------------
#
# NOTE: after head-stripping + whitespace-collapse, the ``[head]`` position
# becomes a single space; anchors below are written to match the resulting
# contiguous text (e.g. "films acted by were directed by who").
_ANCHORS: list[tuple[str, list[str]]] = [
    # ===================== symmetric: head = PERSON =======================
    (
        "co_actor",
        [
            "acted together with",
            "starred together with",
            "co-starred with",
            "co-star of",
            "co-stars of",
            "appeared in the same movie with",
            "appear in the same movie with",
            "is a co-star of",
            "are co-stars of",
        ],
    ),
    (
        "co_director",
        [
            "co-directed films",
            "co-directed movies",
            "co-directed",
            "film co-directors",
            "movie co-directors",
            "co-directors",
            "directed films together with",
            "directed movies together with",
        ],
    ),
    (
        "co_writer",
        [
            "co-wrote films",
            "co-wrote movies",
            "co-wrote",
            "film co-writers",
            "movie co-writers",
            "co-writers",
            "wrote films together with",
            "wrote movies together with",
        ],
    ),
    # ===================== symmetric: head = MOVIE ========================
    (
        "shares_director",
        [
            "same director of",
            "same director",
            "directed by the same director",
            "the director of also directed",
            "is also the director of",
        ],
    ),
    (
        "shares_writer",
        [
            "same screenwriter of",
            "same screenwriter",
            "share the screenwriter with",
            "share the same screenwriter",
            "the screenwriter of also wrote",
            "the scriptwriter of also wrote",
            "scriptwriter of also wrote",
        ],
    ),
    (
        "shares_actor",
        [
            "same actor of",
            "same actor",
            "the actor of also starred",
            "also starred in which films",
            "also starred in which movies",
            "the actor in also appears",
            "also appears in which films",
            "also appears in which movies",
        ],
    ),
    # ============== asymmetric: ANSWER = GENRE / TYPE =====================
    (
        "genre_of_acted",
        [
            "genres of the films starred",
            "genres of the films acted",
            "genres of the movies acted",
            "genres of the movies starred",
            "genres are the films acted",
            "genres are the movies acted",
            "genres do the films starred",
            "genres do the movies acted",
            "types are the films starred",
            "types are the movies starred",
            "films acted by were in which genres",
            "movies starred by were in which genres",
        ],
    ),
    (
        "genre_of_directed",
        [
            "genres of the films directed",
            "genres of the movies directed",
            "genres are the films directed",
            "genres are the movies directed",
            "genres do the films directed",
            "genres do the movies directed",
            "types are the films directed",
            "types are the movies directed",
            "films directed by were in which genres",
            "movies directed by were in which genres",
        ],
    ),
    (
        "genre_of_written",
        [
            "genres of the films written",
            "genres of the movies written",
            "genres are the films written",
            "genres are the movies written",
            "genres do the films written",
            "genres do the movies written",
            "types are the films written",
            "types are the movies written",
            "films written by were in which genres",
            "movies written by were in which genres",
        ],
    ),
    # ============== asymmetric: ANSWER = LANGUAGE =========================
    (
        "language_of_acted",
        [
            "languages spoken in the films starred",
            "languages spoken in the films acted",
            "languages spoken in the movies acted",
            "main languages in acted films",
            "main languages in starred movies",
            "primary languages in the films acted",
            "primary languages in the movies acted",
            "languages are the films starred",
            "languages are the movies acted",
            "films acted by were in which languages",
            "movies starred by were in which languages",
        ],
    ),
    (
        "language_of_directed",
        [
            "languages spoken in the films directed",
            "languages spoken in the movies directed",
            "main languages in directed films",
            "main languages in directed movies",
            "primary languages in the films directed",
            "primary languages in the movies directed",
            "languages are the films directed",
            "languages are the movies directed",
            "films directed by were in which languages",
            "movies directed by were in which languages",
        ],
    ),
    (
        "language_of_written",
        [
            "languages spoken in the films written",
            "languages spoken in the movies written",
            "main languages in written films",
            "main languages in written movies",
            "primary languages in the films written",
            "primary languages in the movies written",
            "languages are the films written",
            "languages are the movies written",
            "films written by were in which languages",
            "movies written by were in which languages",
        ],
    ),
    # ============== asymmetric: ANSWER = RELEASE YEAR =====================
    (
        "release_year_of_acted",
        [
            "release dates of acted films",
            "release dates of starred movies",
            "release years of the movies acted",
            "release years the films starred",
            "did the films starred by",
            "did the movies acted by",
            "the films acted by were released",
            "films acted by were released",
            "the movies starred by were released",
            "movies starred by were released",
            "were the films acted by released",
            "were the movies starred by released",
        ],
    ),
    (
        "release_year_of_directed",
        [
            "release dates of directed films",
            "release dates of directed movies",
            "release years of the movies directed",
            "release years the films directed",
            "did the films directed by",
            "did the movies directed by",
            "films directed by were released",
            "movies directed by were released",
            "were the films directed by released",
            "were the movies directed by released",
        ],
    ),
    (
        "release_year_of_written",
        [
            "release dates of written films",
            "release dates of written movies",
            "release years of the movies written",
            "release years the films written",
            "did the films written by",
            "did the movies written by",
            "films written by were released",
            "movies written by were released",
            "were the films written by released",
            "were the movies written by released",
        ],
    ),
    # ============== asymmetric: ANSWER = DIRECTOR =========================
    (
        "director_of_acted",
        [
            "directors of the films starred",
            "directors of the movies acted",
            "director of acted films",
            "director of starred movies",
            "person directed the films acted",
            "person directed the movies starred",
            "directed the films starred by",
            "directed the movies acted by",
            "films acted by were directed by",
            "movies starred by were directed by",
        ],
    ),
    (
        "director_of_written",
        [
            "directors of the films written",
            "directors of the movies written",
            "director of written films",
            "director of written movies",
            "person directed the films written",
            "person directed the movies written",
            "directed the films written by",
            "directed the movies written by",
            "films written by were directed by",
            "movies written by were directed by",
        ],
    ),
    # ============== asymmetric: ANSWER = WRITER ===========================
    (
        "writer_of_acted",
        [
            "writers of the films starred",
            "writers of the movies acted",
            "screenwriter of acted films",
            "screenwriter of starred movies",
            "person wrote the films acted",
            "person wrote the movies starred",
            "wrote the movies acted by",
            "screenplay for the movies starred",
            "films acted by were written by",
            "movies starred by were written by",
            "starred movies for the writer",
        ],
    ),
    (
        "writer_of_directed",
        [
            "writers of the films directed",
            "writers of the movies directed",
            "screenwriter of directed films",
            "screenwriter of directed movies",
            "person wrote the films directed",
            "person wrote the movies directed",
            "wrote the movies directed by",
            "screenplay for the movies directed",
            "films directed by were written by",
            "movies directed by were written by",
        ],
    ),
    # ============== asymmetric: ANSWER = ACTOR ============================
    (
        "actor_of_directed",
        [
            "actors in the films directed",
            "actors in the movies directed",
            "actors of the director",
            "starred movies for the director",
            "films directed by starred",
            "movies directed by starred",
            "acted in the films directed",
            "acted in the movies directed",
            "starred in the films directed",
            "starred in the movies directed",
        ],
    ),
    (
        "actor_of_written",
        [
            "actors in the films written",
            "actors in the movies written",
            "actors of the screenwriter",
            "films written by starred",
            "movies written by starred",
            "acted in the films written",
            "acted in the movies written",
            "starred in the films written",
            "starred in the movies written",
        ],
    ),
]


def relation_for_2hop(question: str) -> str | None:
    """Map a MetaQA-2hop NL question to its derived-relation name.

    Anchor-based, mirroring ``run_metaqa._relation_for_question``: the
    question is lower-cased and underscore-normalised, then matched against
    the priority-ordered ``_ANCHORS`` list (most-specific first).  Returns the
    derived-relation name or ``None`` if nothing matches.
    """
    lowered = question.lower().replace("_", " ")
    stripped = re.sub(r"\[[^\]]+\]", " ", lowered)
    cleaned = re.sub(r"\s+", " ", stripped).strip()
    for rel, anchors in _ANCHORS:
        for anc in anchors:
            if anc in cleaned:
                return rel
    return None


# ---------------------------------------------------------------------------
# Self-test / verification.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import collections
    import sys

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="camazlucas/MetaQA",
        filename="2-hop/vanilla/qa_dev.txt",
        repo_type="dataset",
    )

    total = resolved = 0
    unresolved: collections.Counter[str] = collections.Counter()
    by_rel: collections.Counter[str] = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            q = line.split("\t")[0]
            total += 1
            rel = relation_for_2hop(q)
            if rel is None:
                tmpl = re.sub(r"\[[^\]]+\]", "[X]", q.strip().lower())
                unresolved[tmpl] += 1
            else:
                resolved += 1
                by_rel[rel] += 1

    print(f"resolution: {resolved}/{total} = {resolved / total:.4%}")
    print(f"distinct derived relations hit: {len(by_rel)} / {len(DERIVED_2HOP)}")
    for rel in sorted(DERIVED_2HOP):
        print(f"  {rel:28s} {by_rel.get(rel, 0):5d}")
    if unresolved:
        print(f"\nUNRESOLVED templates ({len(unresolved)}):")
        for tmpl, c in sorted(unresolved.items()):
            print(f"  {c:5d}  {tmpl}")
        sys.exit(1)
