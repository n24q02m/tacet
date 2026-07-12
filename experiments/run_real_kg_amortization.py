"""Real-LLM, real-cost amortization experiment on MetaQA (paper §13 central evidence).

This is the **real-money** counterpart to ``run_metaqa_costdecay.py`` (which uses
the paper's *simulated* tier-cost units). Here every Tier-3 answer is a genuine
Grok-4.3 API call wrapped in :class:`tacet.llm.metering.MeteredTeacher`, so the
reported USD is the provider's *measured* spend (token counts x real price, or
xAI's authoritative ``cost_in_usd_ticks`` when present). Symbolic / KGE tiers
cost ~0 USD — we record their wall-clock latency instead.

The question this answers
-------------------------
At *matched accuracy*, three ways of serving the same streamed query workload:

(a) **LLM-only** — every query to Grok. The cost ceiling, and the accuracy
    reference (a frontier model genuinely knows some MetaQA movie facts).
(b) **cache-cascade** — write-back only (``rule_synthesis=False`` /
    ``kge_augment=False``): a degenerate cache that answers *exact repeats*
    cheaply but must re-query the teacher for every unseen (head, relation).
(c) **full distillation cascade** — write-back + rule synthesis (+ KGE aug):
    a synthesised Horn rule answers *unseen heads*, so distillation can be
    cheaper than a cache on a workload with many distinct heads per relation.

The honest hypothesis under test: does (c) beat (b) at matched accuracy?
On MetaQA 1-hop the relations are flat, independent movie attributes
(``directed_by``, ``starred_actors``, ``has_genre`` ...), which have **no
compositional Datalog structure** for the AMIE-style miner to exploit, so a
negative result (full ~= cache) is a plausible and *valid* finding — that is
exactly the cold-start / sparse-structure limitation reviewers asked us to
report rather than paper over.

Spend control
-------------
* ``--limit`` caps the workload (pilot 50, capped main <= 300). The cap is
  logged in the output JSON (``workload_cap``).
* ``--budget-usd`` (default 2.0) is a HARD stop: a budget-guarded teacher
  raises once cumulative *measured* spend crosses it, and the run exits after
  writing whatever it has. Each of the three arms calls the teacher, so the
  effective spend is roughly (LLM-only + cache + full) teacher calls; the guard
  is shared across all three.

Run (set your own xAI API key)::

    export TACET_XAI_API_KEY=<your xAI API key>
    export TACET_TEACHER=grok
    python experiments/run_real_kg_amortization.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from run_metaqa import _relation_for_question  # noqa: E402

from tacet.cascade.router import TACET  # noqa: E402
from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.core.ontology import Ontology, RelationType  # noqa: E402
from tacet.data.metaqa import load_metaqa  # noqa: E402
from tacet.llm.metering import (  # noqa: E402
    DEFAULT_PRICES,
    MeteredTeacher,
    PriceTable,
    price_key_for_slug,
)
from tacet.llm.teacher import OracleTeacher, Teacher, TeacherResponse  # noqa: E402
from tacet.llm.teachers import build_teacher_from_settings  # noqa: E402
from tacet.serve.config import CascadeConfig, KGEConfig  # noqa: E402
from tacet.serve.settings import load_settings  # noqa: E402

# ---------------------------------------------------------------------------
# Multi-hop COMPOSITION support (the 2-hop / 3-hop distillation differentiator).
#
# MetaQA's KB ships only *base* relations (directed_by, starred_actors, ...).
# A 2-hop question's answer is reachable only by *composing* two base relations
# from the head, so there is no single base edge ``(head, R, gold)`` — the 1-hop
# harness (single-relation lookup) cannot express it. Here we synthesise the
# 2-hop workload directly from the KB:
#
#   * a COMPOSED relation ``target`` is DEFINED as a base-relation chain
#     ``R1 ∘ R2`` (each leg optionally inverted), e.g. "directors of movies that
#     X acted in" = ``~starred_actors . directed_by``;
#   * gold tails are computed EXACTLY by composing those base edges over the KB,
#     so accuracy is measured against provable ground truth;
#   * the teacher is asked the composed question in NATURAL LANGUAGE (the
#     ``CompositionTeacher`` rewrites the synthetic relation token into a real
#     2-hop English question), so a frontier model genuinely answers it;
#   * the base edges stay in the graph, so the AMIE-style miner can recover the
#     length-2 Horn rule ``R1(x,z) & R2(z,y) => target(x,y)`` once enough heads
#     are seen — and that rule then fires Tier-1 (free) on UNSEEN heads, which a
#     write-back cache cannot do. THAT is the distillation-beats-caching claim,
#     and a compositional workload is exactly where it can hold.
#
# The 2-hop / 3-hop NL question files from the original MetaQA distribution are
# not part of the cloned ``yuyuz/MetaQA`` git repo (only the KB + 1-hop QA were
# downloaded); building the workload from the KB composition is both available
# and *cleaner*, because gold is the provable composition rather than a parsed
# NL string.
# ---------------------------------------------------------------------------

#: A composition leg: ``(relation, inverse)``.  ``inverse=True`` traverses the
#: edge backwards (``into`` instead of ``out``), so a movie->person base edge
#: can be walked person->movie.
Leg = tuple[str, bool]

#: name -> (legs, nl_question_template, kg_relation_token).
#: ``nl_question_template`` is a ``str.format``-style template with a ``{head}``
#: slot, phrasing the composed query as a natural 2-hop / 3-hop question.
COMPOSITIONS: dict[str, dict] = {
    # 2-hop, canonical MetaQA pattern, low fan-out (median answer set = 1),
    # ~9.2k distinct heads -> a large held-out pool for rule generalisation.
    "directors_of_acted": {
        "legs": [("starred_actors", True), ("directed_by", False)],
        "kg_relation": "q2_directors_of_movies_acted_in_by",
        "nl": (
            "Who are the directors of the movies that the actor {head} starred in? "
            "List every director."
        ),
        "hop": 2,
    },
    # 2-hop alternative: genres of movies an actor starred in (median answer 2).
    "genres_of_acted": {
        "legs": [("starred_actors", True), ("has_genre", False)],
        "kg_relation": "q2_genres_of_movies_acted_in_by",
        "nl": (
            "What are the genres of the movies that the actor {head} starred in? List every genre."
        ),
        "hop": 2,
    },
    # 3-hop: actors who co-starred with X = actors of movies that X acted in,
    # ``~starred_actors . starred_actors . ...`` would loop; instead use
    # actors of movies directed by directors of movies X acted in (true 3-leg).
    "coactors_via_director": {
        "legs": [
            ("starred_actors", True),
            ("directed_by", False),
            # third leg re-expanded at build time: director -> movies -> actors
        ],
        "kg_relation": "q3_actors_of_movies_by_directors_of_acted",
        "nl": (
            "Consider the directors of the movies that the actor {head} starred in. "
            "Who are all the actors that starred in any movie by those directors? "
            "List every actor."
        ),
        "hop": 3,
        "third_leg": ("starred_actors", False),  # director-movies via ~directed_by then actors
        "third_pivot": ("directed_by", True),
    },
}


def _leg_adj(graph: WorldGraph, relation: str, inverse: bool) -> dict[str, set[str]]:
    """Adjacency ``x -> {y}`` for one composition leg over the base KB."""
    adj: dict[str, set[str]] = defaultdict(set)
    for e in graph.edges:
        if e.relation != relation:
            continue
        if inverse:
            adj[e.target].add(e.source)
        else:
            adj[e.source].add(e.target)
    return adj


def _compose_gold(graph: WorldGraph, spec: dict) -> dict[str, set[str]]:
    """Compute, for every head, the EXACT composed answer set over the KB.

    Two-leg compositions walk ``head -L1-> z -L2-> y``; the optional three-leg
    form (``coactors_via_director``) walks one extra hop. Self-answers (``y==head``)
    are dropped so the workload is a genuine multi-hop traversal.
    """
    l1, l2 = spec["legs"][0], spec["legs"][1]
    a1 = _leg_adj(graph, l1[0], l1[1])
    a2 = _leg_adj(graph, l2[0], l2[1])
    gold: dict[str, set[str]] = {}
    if spec.get("hop") == 3:
        # head -L1-> z1 -L2-> z2 (directors); then directors -~directed_by-> movies
        # -starred_actors-> actors.
        piv = spec["third_pivot"]  # (directed_by, True): director -> movies
        leg3 = spec["third_leg"]  # (starred_actors, False): movie -> actors
        a_piv = _leg_adj(graph, piv[0], piv[1])
        a_leg3 = _leg_adj(graph, leg3[0], leg3[1])
        for x, z1s in a1.items():
            directors: set[str] = set()
            for z1 in z1s:
                directors |= a2.get(z1, set())
            ys: set[str] = set()
            for d in directors:
                for mv in a_piv.get(d, set()):
                    ys |= a_leg3.get(mv, set())
            ys.discard(x)
            if ys:
                gold[x] = ys
        return gold
    for x, zs in a1.items():
        ys: set[str] = set()
        for z in zs:
            ys |= a2.get(z, set())
        ys.discard(x)
        if ys:
            gold[x] = ys
    return gold


def _build_composed_workload(
    graph: WorldGraph, spec: dict, limit_pool: int, rng, max_answer: int = 25
):  # noqa: ANN001, ANN201
    """Resolve the composition into a balanced pool of ``(head, kg_relation, gold)``.

    Heads with an unmanageably large answer set (> ``max_answer``) are dropped so
    a frontier teacher has a realistic chance of returning the full gold set —
    keeping LLM-only accuracy interpretable (we are testing cost at *matched*
    accuracy, not stress-testing recall on 100-way answers).

    Returns ``(pool, full_gold)``. ``pool`` is that filtered / shuffled / truncated
    workload; ``full_gold`` is the COMPLETE ``_compose_gold`` map (every head,
    unfiltered) — the composed relation's TRUE edges over the whole KB, which the
    controlled runner needs to score installed-rule world precision against ALL
    entities the rule body fires on, not just the pool's teacher-answered heads.
    It is returned here so callers do not recompute ``_compose_gold``.
    """
    kg_rel = spec["kg_relation"]
    gold_map = _compose_gold(graph, spec)
    items = [(h, kg_rel, frozenset(ys)) for h, ys in gold_map.items() if 0 < len(ys) <= max_answer]
    rng.shuffle(items)
    return items[:limit_pool], gold_map


class CompositionTeacher(Teacher):
    """Rewrite a synthetic composed-relation token into a natural 2-hop question.

    The base ``GrokTeacher`` interpolates ``relation`` verbatim into its prompt
    (``Head: X, Relation: <relation>``); a token like
    ``q2_directors_of_movies_acted_in_by`` would be opaque. This wrapper swaps in
    the composition's English ``nl`` template so the model answers the genuine
    multi-hop question, then returns the parsed answer unchanged. ``last_usage``
    is proxied so :class:`MeteredTeacher` still meters the real token cost.
    """

    def __init__(self, wrapped: Teacher, nl_template: str) -> None:
        self.wrapped = wrapped
        self._nl = nl_template

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:  # noqa: ARG002
        question = self._nl.format(head=head)
        resp = self.wrapped.answer(graph, head, question)
        # proxy provider usage so the meter reads the real token counts
        self.last_usage = getattr(self.wrapped, "last_usage", None)
        return resp


class BudgetExceededError(RuntimeError):
    """Raised when cumulative measured spend crosses the hard cap."""


class BudgetGuard:
    """Shared across all arms: tracks total measured USD, trips at the cap."""

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd = budget_usd
        self.spent_usd = 0.0

    def add(self, usd: float) -> None:
        self.spent_usd += usd
        if self.spent_usd > self.budget_usd:
            raise BudgetExceededError(
                f"measured spend ${self.spent_usd:.4f} exceeded hard cap "
                f"${self.budget_usd:.2f} — stopping."
            )


class GuardedMeteredTeacher(Teacher):
    """Wrap a ``MeteredTeacher`` and feed each call's measured cost to a shared
    :class:`BudgetGuard`, so spend is bounded across all three arms.

    Also records a (cumulative_usd) trajectory point per delegated call.
    """

    def __init__(self, metered: MeteredTeacher, guard: BudgetGuard) -> None:
        self.metered = metered
        self.guard = guard

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        resp = self.metered.answer(graph, head, relation)
        self.guard.add(self.metered.last_cost_usd)
        return resp


def _build_workload(bench, limit_pool: int, rng) -> list[tuple[str, str, frozenset[str]]]:  # noqa: ANN001
    """Resolve 1-hop questions into (head, relation, gold_tails) triples.

    Only questions whose head is a real KG entity and whose relation the
    validated matcher can resolve are kept, so LLM-only accuracy is measured
    against genuine gold answers over real movie / person names. The pool is
    **shuffled and round-robin balanced across relations** so the workload is
    not dominated by a single relation (MetaQA's test file is grouped by
    question type) — a relation-diverse stream is what lets rule synthesis
    have a fair shot at finding cross-relation structure.
    """
    by_rel: dict[str, list[tuple[str, str, frozenset[str]]]] = {}
    seen: set[tuple[str, str]] = set()
    for q in bench.questions:
        rel = _relation_for_question(q.question, bench.relations)
        if rel is None or q.head not in bench.entities:
            continue
        key = (q.head, rel)
        if key in seen:
            continue
        seen.add(key)
        by_rel.setdefault(rel, []).append((q.head, rel, frozenset(q.answers)))
    for items in by_rel.values():
        rng.shuffle(items)
    # round-robin across relations for a balanced pool
    out: list[tuple[str, str, frozenset[str]]] = []
    cursors = dict.fromkeys(by_rel, 0)
    rels = sorted(by_rel)
    while len(out) < limit_pool and any(cursors[r] < len(by_rel[r]) for r in rels):
        for r in rels:
            if cursors[r] < len(by_rel[r]):
                out.append(by_rel[r][cursors[r]])
                cursors[r] += 1
                if len(out) >= limit_pool:
                    break
    return out


def _zipf_stream(pool, n, a, rng):  # noqa: ANN001
    """Sample n query instances Zipfian over the distinct pool so hot queries
    repeat (cache hits) while many distinct heads per relation still appear
    (the rule-generalisation opportunity)."""
    out: list[tuple[str, str, frozenset[str]]] = []
    while len(out) < n:
        draw = rng.zipf(a, size=n)
        draw = draw[draw <= len(pool)] - 1
        out.extend(pool[i] for i in draw)
    return out[:n]


def _oracle_gold_from_pool(
    pool: list[tuple[str, str, frozenset[str]]],
) -> dict[str, frozenset[str]]:
    """Ground-truth ``(head, relation) -> gold tails`` map for the oracle teacher.

    Keyed on ``f"{head}\\t{relation}"`` (the key the ``OracleTeacher`` lookup in
    :func:`_new_metered` expects) over the WHOLE pool -- a superset of the streamed
    queries -- so the oracle answers seen and UNSEEN held-out heads alike, which is
    the whole point of the mechanism test. Shared with the controlled runner so both
    designs build the gold map identically.
    """
    return {f"{h}\t{r}": g for h, r, g in pool}


def _kg_without(bench, stream) -> WorldGraph:
    """A copy of the MetaQA KB with every queried (head, relation) edge removed.

    Holding out the workload's edges forces the cascade to genuinely query the
    teacher the first time it sees a (head, relation): Tier 1 can no longer read
    the answer straight off the base graph. Write-back then re-caches it; a
    synthesised rule (if any) re-derives it for *other* heads of that relation.
    """
    held = {(h, r) for h, r, _ in stream}
    g = WorldGraph(name=f"{bench.kg.name}-heldout")
    for n in bench.kg.nodes:
        g.add_node(n.id, n.type, **n.props)
    for e in bench.kg.edges:
        if (e.source, e.relation) in held:
            continue
        g.add_edge(e.source, e.relation, e.target, **e.props)
    return g


def _accuracy(gold: frozenset[str], answers: list[str]) -> bool:
    """Gold-subset match — the same metric as run_metaqa.py."""
    return bool(gold) and gold.issubset(set(answers))


def _run_llm_only(stream, bench, metered, guard) -> dict:
    """Arm (a): every query straight to the metered Grok teacher."""
    guarded = GuardedMeteredTeacher(metered, guard)
    correct = 0
    traj: list[list[float]] = []  # [query_index, cumulative_usd]
    t0 = time.time()
    for i, (h, r, gold) in enumerate(stream):
        resp = guarded.answer(bench.kg, h, r)
        correct += int(_accuracy(gold, resp.answers))
        traj.append([i + 1, round(metered.total_cost_usd, 6)])
    return {
        "arm": "llm_only",
        "n": len(stream),
        "total_cost_usd": round(metered.total_cost_usd, 6),
        "total_prompt_tokens": metered.total_prompt_tokens,
        "total_completion_tokens": metered.total_completion_tokens,
        "teacher_calls": metered.n_calls,
        "accuracy": round(correct / len(stream), 4) if stream else 0.0,
        "tier_pct": {"3": 100.0},  # every query hits the teacher
        "wallclock_s": round(time.time() - t0, 1),
        "cost_trajectory": traj,
    }


def _run_cascade(name, stream, bench, ontology, metered, guard, cfg) -> dict:
    """Arms (b)/(c): stream through TACET whose teacher is the metered Grok.

    Measured USD is read from the meter (only Tier-3 teacher calls spend money);
    the cascade's simulated ``Answer.cost`` is ignored for the real-cost figure.
    """
    guarded = GuardedMeteredTeacher(metered, guard)
    ak = TACET(_kg_without(bench, stream), ontology, guarded, config=cfg)
    ak.warmup()
    correct = 0
    tiers = {1: 0, 2: 0, 3: 0}
    traj: list[list[float]] = []
    t0 = time.time()
    for i, (h, r, gold) in enumerate(stream):
        ans = ak.ask(h, r)
        correct += int(_accuracy(gold, ans.answers))
        tiers[ans.tier] = tiers.get(ans.tier, 0) + 1
        traj.append([i + 1, round(metered.total_cost_usd, 6)])
    n = len(stream)
    return {
        "arm": name,
        "n": n,
        "total_cost_usd": round(metered.total_cost_usd, 6),
        "total_prompt_tokens": metered.total_prompt_tokens,
        "total_completion_tokens": metered.total_completion_tokens,
        "teacher_calls": metered.n_calls,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "tier_counts": tiers,
        "tier_pct": {str(t): round(100.0 * tiers[t] / n, 1) if n else 0.0 for t in (1, 2, 3)},
        "synthesised_rules": list(ak.synthesised_rules),
        "wallclock_s": round(time.time() - t0, 1),
        "cost_trajectory": traj,
    }


def _new_metered(
    settings,
    model,
    nl_template: str | None = None,
    oracle_gold: dict[str, frozenset[str]] | None = None,
    error_rate: float = 0.0,
    seed: int = 0,
) -> MeteredTeacher:
    """A fresh per-arm MeteredTeacher (isolated so spend / call-count is per-arm).

    With ``TACET_TEACHER=oracle`` (``oracle_gold`` supplied) the teacher is a free,
    instant ground-truth oracle over the composed gold map — a zero-cost stand-in
    used to test the *mechanism* (does a synthesised rule cut teacher calls on
    UNSEEN heads?) before spending real money on Grok. ``MeteredTeacher`` still
    counts every delegated call (``n_calls``), and the oracle exposes no
    ``last_usage`` so the measured USD stays 0 — the decisive signal is the
    per-arm *call count*, not the dollar figure. ``error_rate`` (from
    ``TACET_ORACLE_ERROR_RATE``) turns the perfect oracle into a noisy one,
    corrupting that fraction of answers into a plausible wrong entity drawn from
    the workload's own gold tails, with ``seed`` making the corruption
    reproducible — the imperfect-teacher regime the noise sweep needs.

    When ``nl_template`` is given (multi-hop composition runs) the real teacher is
    first wrapped in a :class:`CompositionTeacher` so the synthetic
    composed-relation token is rewritten into a natural-language multi-hop
    question; the meter reads the proxied provider usage either way.
    """
    if settings.teacher == "oracle":
        if oracle_gold is None:
            raise SystemExit("TACET_TEACHER=oracle but no gold map was built for the workload.")
        _gold = oracle_gold
        # A corrupted answer must be a PLAUSIBLE wrong entity, so draw the noise
        # pool from the workload's own gold tails (the answer entities).
        entity_pool = sorted({tail for tails in _gold.values() for tail in tails})
        teacher: Teacher = OracleTeacher(
            lambda h, r: sorted(_gold.get(f"{h}\t{r}", ())),
            error_rate=error_rate,
            entity_pool=entity_pool,
            seed=seed,
            # Arms call the teacher a different number of times (LLM-only every
            # query, cache/full only on misses); per_key keys corruption on
            # (seed, head, relation) so the same question gets the same answer
            # across arms, independent of call order/count.
            noise_mode="per_key",
        )
        return MeteredTeacher(teacher, PriceTable.default(), model=model)
    teacher = build_teacher_from_settings(settings)
    if teacher is None:
        raise SystemExit(
            "No real teacher configured. Set TACET_TEACHER=grok and "
            "TACET_XAI_API_KEY to your xAI API key."
        )
    if model not in DEFAULT_PRICES:
        raise SystemExit(f"no price for model {model!r}; add it to DEFAULT_PRICES")
    if nl_template is not None:
        teacher = CompositionTeacher(teacher, nl_template)
    return MeteredTeacher(teacher, PriceTable.default(), model=model)


def resolve_price_key(settings) -> str:  # noqa: ANN001
    """The DEFAULT_PRICES key to meter this run's teacher against.

    An explicit ``TACET_PRICE_MODEL`` always wins (unchanged). Otherwise, for the
    OpenRouter teacher the price key is derived from its model slug via
    :func:`price_key_for_slug` -- so a run of an E11 ladder model no longer needs
    ``TACET_PRICE_MODEL`` set by hand -- and every other teacher keeps defaulting
    to ``grok-4.3``.
    """
    explicit = os.environ.get("TACET_PRICE_MODEL")
    if explicit:
        return explicit
    if getattr(settings, "teacher", None) == "openrouter":
        return price_key_for_slug(settings.openrouter_model)
    return "grok-4.3"


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--metaqa-root", default="data/MetaQA")
    ap.add_argument("--hop", type=int, default=1)
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--limit",
        type=int,
        default=50,
        help="WORKLOAD CAP — number of streamed queries (pilot 50, capped main <=300)",
    )
    ap.add_argument("--zipf-a", type=float, default=1.5, help="Zipf exponent for query repeats")
    ap.add_argument("--budget-usd", type=float, default=2.0, help="HARD spend stop (shared)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--composition",
        default=None,
        choices=sorted(COMPOSITIONS),
        help=(
            "for --hop>=2: which base-relation chain to compose into the workload "
            "(default picks one matching --hop)"
        ),
    )
    ap.add_argument("--out", default="experiments/results/real_kg_amortization.json")
    args = ap.parse_args()

    # Oracle-teacher noise dial: fraction of oracle answers corrupted into a
    # plausible wrong entity (0.0 = perfect oracle); enables the noise sweep.
    oracle_error_rate = float(os.environ.get("TACET_ORACLE_ERROR_RATE", "0.0"))
    rng = np.random.default_rng(args.seed)

    print(f"[cap] workload capped at {args.limit} queries; hard budget ${args.budget_usd:.2f}")
    # The KB is shared across all hop levels; the 1-hop QA file is always
    # present, so we load the bench at hop=1 for its KB and (for 1-hop) its
    # questions. Multi-hop workloads are SYNTHESISED from the KB by composition
    # (the original distribution's 2-hop/3-hop NL question files are not part of
    # the cloned yuyuz/MetaQA git repo).
    kb_hop = 1
    print(f"loading MetaQA KB (+1-hop questions) {args.split} from {args.metaqa_root} ...")
    bench = load_metaqa(args.metaqa_root, hop=kb_hop, split=args.split)
    print(f"  kg stats: {bench.stats()}")

    settings = load_settings()
    oracle_mode = settings.teacher == "oracle"
    openrouter_mode = settings.teacher == "openrouter"
    # The price key must be the real one for the USD to mean anything; resolve it
    # from the teacher (OpenRouter derives it from the slug, so TACET_PRICE_MODEL
    # need not be set by hand).
    model = resolve_price_key(settings)
    called_model = settings.openrouter_model if openrouter_mode else settings.xai_model
    if oracle_mode:
        # Free, instant ground-truth teacher — mechanism test, no API cost.
        print("  teacher=ORACLE (ground-truth, $0, instant) — mechanism test, no Grok cost")
    else:
        key = settings.openrouter_api_key if openrouter_mode else settings.xai_api_key
        if not key:
            raise SystemExit(
                "no teacher key set — export TACET_XAI_API_KEY (grok) or "
                "TACET_OPENROUTER_API_KEY (openrouter) first."
            )
        print(f"  teacher={settings.teacher} model={called_model} (priced as {model})")

    # --- workload: 1-hop (single-relation lookup) vs multi-hop (composition) --
    nl_template: str | None = None
    composed_relation: str | None = None
    #: (head, relation) -> gold tails, for the oracle teacher. Keyed on the pair
    #: so it answers any UNSEEN held-out head exactly (ground truth), which is the
    #: whole point of the mechanism test.
    oracle_gold: dict[str, frozenset[str]] = {}
    if args.hop == 1:
        # Pool of distinct resolvable (head, relation, gold) -> a Zipfian stream.
        pool = _build_workload(bench, limit_pool=max(args.limit, 400), rng=rng)
        ontology = Ontology.induce(bench.kg)
    else:
        comp_name = args.composition or next(
            (k for k, v in COMPOSITIONS.items() if v.get("hop") == args.hop), None
        )
        if comp_name is None:
            raise SystemExit(f"no composition registered for hop={args.hop}")
        spec = COMPOSITIONS[comp_name]
        if spec.get("hop") != args.hop:
            raise SystemExit(
                f"composition {comp_name!r} is hop={spec.get('hop')}, not --hop={args.hop}"
            )
        nl_template = spec["nl"]
        composed_relation = spec["kg_relation"]
        legs_desc = " . ".join(f"{'~' if inv else ''}{r}" for r, inv in spec["legs"])
        print(
            f"  COMPOSITION {comp_name!r} (hop={args.hop}): target {composed_relation} "
            f":= {legs_desc}{' . (+3rd leg)' if spec.get('hop') == 3 else ''}"
        )
        pool, _ = _build_composed_workload(bench.kg, spec, limit_pool=max(args.limit, 400), rng=rng)
        if not pool:
            raise SystemExit(
                f"composition {comp_name!r} produced an empty pool — check the KB / legs."
            )
        # Declare the composed relation in the ontology so the synthesised
        # length-2 Horn rule (whose HEAD is this relation) passes the engine's
        # ontology-consistency gate. MetaQA nodes are all type 'Entity', so the
        # base legs and the composed head share the same {Entity}->{Entity}
        # domain/range and impose no type contradiction.
        ontology = Ontology.induce(bench.kg)
        ontology.add_relation_type(
            RelationType(
                name=composed_relation, domain=frozenset({"Entity"}), range=frozenset({"Entity"})
            )
        )

    stream = _zipf_stream(pool, args.limit, args.zipf_a, rng)
    if oracle_mode:
        # Ground truth for every (head, relation) in the pool (a superset of the
        # stream), so the oracle answers seen and UNSEEN held-out heads alike.
        oracle_gold = _oracle_gold_from_pool(pool)
    distinct = len({(h, r) for h, r, _ in stream})
    print(
        f"  pool={len(pool)} stream={len(stream)} distinct={distinct} "
        f"({100 * distinct / len(stream):.0f}% unique)"
    )
    # MetaQA's KB is large (~133k triples, ~43k entities). The default
    # full-batch KGE fit (``batch_size=0``) allocates per-epoch arrays over the
    # whole triple set and exhausts RAM; mini-batching is the documented path at
    # this scale (KGEConfig: "set 2048+ for FB15k-scale data"). Tier-2 rarely
    # fires on MetaQA's flat attribute relations, so a compact, mini-batched
    # embedding is ample for this cost study.
    kge_cfg = KGEConfig(
        dim=min(settings.kge_dim, 32),
        epochs=min(settings.kge_epochs, 15),
        batch_size=4096,
    )
    guard = BudgetGuard(args.budget_usd)

    arms: list[dict] = []
    truncated = False
    try:
        # (a) LLM-only — also the accuracy reference.
        print("\narm (a) LLM-only — every query to Grok ...")
        m_a = _new_metered(
            settings, model, nl_template, oracle_gold, error_rate=oracle_error_rate, seed=args.seed
        )
        r_a = _run_llm_only(stream, bench, m_a, guard)
        arms.append(r_a)
        print(
            f"  acc={r_a['accuracy']}  cost=${r_a['total_cost_usd']:.4f}  "
            f"cost/query=${r_a['total_cost_usd'] / r_a['n']:.5f}  calls={r_a['teacher_calls']}"
        )

        # (b) cache-cascade — write-back only (degenerate cache).
        print("\narm (b) cache-cascade — write-back only (no rule synthesis / no KGE aug) ...")
        cfg_cache = CascadeConfig(
            kge=kge_cfg, rule_synthesis=False, kge_augment=False, write_back=True
        )
        m_b = _new_metered(
            settings, model, nl_template, oracle_gold, error_rate=oracle_error_rate, seed=args.seed
        )
        r_b = _run_cascade("cache_cascade", stream, bench, ontology, m_b, guard, cfg_cache)
        arms.append(r_b)
        print(
            f"  acc={r_b['accuracy']}  cost=${r_b['total_cost_usd']:.4f}  "
            f"tiers={r_b['tier_pct']}  calls={r_b['teacher_calls']}"
        )

        # (c) full distillation cascade — write-back + rule synthesis + KGE aug.
        print("\narm (c) full distillation — write-back + rule synthesis + KGE aug ...")
        cfg_full = CascadeConfig(
            kge=kge_cfg, rule_synthesis=True, kge_augment=True, write_back=True
        )
        m_c = _new_metered(
            settings, model, nl_template, oracle_gold, error_rate=oracle_error_rate, seed=args.seed
        )
        r_c = _run_cascade("full_distillation", stream, bench, ontology, m_c, guard, cfg_full)
        arms.append(r_c)
        print(
            f"  acc={r_c['accuracy']}  cost=${r_c['total_cost_usd']:.4f}  "
            f"tiers={r_c['tier_pct']}  rules={r_c['synthesised_rules']}  "
            f"calls={r_c['teacher_calls']}"
        )
    except BudgetExceededError as e:
        truncated = True
        print(f"\n[HARD STOP] {e}")

    # --- verdict: full vs cache at matched accuracy ---------------------
    by = {a["arm"]: a for a in arms}
    verdict: dict[str, object] = {}
    if "cache_cascade" in by and "full_distillation" in by:
        cache, full = by["cache_cascade"], by["full_distillation"]
        # The MECHANISM signal is teacher-CALL count, not USD: a synthesised rule
        # that fires Tier-1 on an UNSEEN head removes one teacher call regardless
        # of token price. USD = calls x price, so with a free oracle teacher
        # (price 0) the dollar figure is identically 0 for every arm and cannot
        # discriminate the arms — the call count is what proves generalisation.
        cache_calls = cache["teacher_calls"]
        full_calls = full["teacher_calls"]
        verdict = {
            "cache_cost_usd": cache["total_cost_usd"],
            "full_cost_usd": full["total_cost_usd"],
            "cache_teacher_calls": cache_calls,
            "full_teacher_calls": full_calls,
            "calls_saved_vs_cache": cache_calls - full_calls,
            "calls_saved_pct": (
                round(100.0 * (cache_calls - full_calls) / cache_calls, 2)
                if cache_calls > 0
                else None
            ),
            # Decisive, price-independent verdict: did installed rules answer
            # unseen heads, cutting teacher calls below the pure cache?
            "full_fewer_calls_than_cache": full_calls < cache_calls,
            "cache_accuracy": cache["accuracy"],
            "full_accuracy": full["accuracy"],
            "accuracy_matched": abs(cache["accuracy"] - full["accuracy"]) < 1e-9,
            "full_cheaper_than_cache": full["total_cost_usd"] < cache["total_cost_usd"],
            "savings_usd": round(cache["total_cost_usd"] - full["total_cost_usd"], 6),
            "savings_pct": (
                round(
                    100.0
                    * (cache["total_cost_usd"] - full["total_cost_usd"])
                    / cache["total_cost_usd"],
                    2,
                )
                if cache["total_cost_usd"] > 0
                else None
            ),
            "synthesised_rules": full.get("synthesised_rules", []),
        }

    report = {
        "dataset": f"MetaQA-{args.hop}hop-{args.split}",
        "hop": args.hop,
        "composition": (
            {
                "name": args.composition
                or next((k for k, v in COMPOSITIONS.items() if v.get("hop") == args.hop), None),
                "kg_relation": composed_relation,
                "legs": COMPOSITIONS[
                    args.composition
                    or next(k for k, v in COMPOSITIONS.items() if v.get("hop") == args.hop)
                ]["legs"]
                if args.hop >= 2
                else None,
                "nl_template": nl_template,
            }
            if args.hop >= 2
            else None
        ),
        "kg_stats": bench.stats(),
        "real_llm": True,
        "teacher_model_called": called_model,
        "priced_as_model": model,
        "price_per_1k_usd": {"input": DEFAULT_PRICES[model][0], "output": DEFAULT_PRICES[model][1]},
        "price_source": "https://openrouter.ai/x-ai/grok-4.3 + https://docs.x.ai (2026-06-03)",
        "workload_cap": args.limit,
        "budget_usd_hard_cap": args.budget_usd,
        "zipf_a": args.zipf_a,
        "seed": args.seed,
        "stream_len": len(stream),
        "distinct_queries": distinct,
        "truncated_by_budget": truncated,
        "total_measured_spend_usd": round(guard.spent_usd, 6),
        "arms": arms,
        "verdict_full_vs_cache": verdict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\n[spend] total measured across all arms: ${guard.spent_usd:.4f} "
        f"(cap ${args.budget_usd:.2f})"
    )
    print(f"wrote {out}")
    if verdict:
        # The decisive, price-independent verdict is the teacher-CALL count: a
        # synthesised rule that fires on an unseen head removes a teacher call
        # whatever the token price. With the free oracle teacher every arm reads
        # $0, so the call count is the only metric that can show generalisation.
        if verdict["full_fewer_calls_than_cache"]:
            print(
                f"VERDICT: full distillation makes FEWER teacher calls than cache "
                f"({verdict['full_teacher_calls']} vs {verdict['cache_teacher_calls']}, "
                f"-{verdict['calls_saved_vs_cache']} = {verdict['calls_saved_pct']}%) "
                f"at matched accuracy={verdict['accuracy_matched']}. The synthesised "
                f"rule answers UNSEEN heads. rules={verdict['synthesised_rules']}. "
                + (
                    f"USD savings ${verdict['savings_usd']:.4f} ({verdict['savings_pct']}%)."
                    if verdict["cache_cost_usd"] > 0
                    else "(oracle teacher: USD=0 by construction; cost follows the call count.)"
                )
            )
        else:
            tail = (
                "MetaQA 1-hop has no compositional Datalog structure for the miner to exploit."
                if args.hop == 1
                else (
                    f"on MetaQA {args.hop}-hop the miner synthesised "
                    f"{len(verdict['synthesised_rules'])} rule(s) but they did not "
                    "reduce teacher calls below the cache on this workload."
                )
            )
            print(
                "VERDICT: full distillation does NOT beat cache on this workload "
                f"(calls full={verdict['full_teacher_calls']} vs cache="
                f"{verdict['cache_teacher_calls']}); "
                f"synthesised_rules={verdict['synthesised_rules']}. "
                f"Honest negative result — {tail}"
            )


if __name__ == "__main__":
    main()
