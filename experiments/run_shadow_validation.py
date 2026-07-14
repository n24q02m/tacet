"""E12 shadow-validation — label-free promotion of mined rules from the stream.

E11 established that at a candidate confidence gamma_candidate = 0.50 the miner
installs the true latent rule on most 2026 teachers (large teacher-call savings)
but can also install a junk self-referential rule; telling the two apart needed
gold labels that a production stream does not carry. E12 closes that gap using
the stream itself, with no gold:

* The cascade mines rules at gamma_candidate = 0.50. Every installed rule enters
  SHADOW mode: it PREDICTS but never ROUTES, so the arm keeps calling
  cache / teacher exactly as the pure cache arm would.
* On each subsequent teacher call for a ``(head, relation)`` the shadow rule
  COVERS (predicts a non-empty answer for) and that was NOT among the rule's
  training write-backs (an unseen head), the rule's prediction is compared with
  the teacher's answer under the same gold-subset criterion the benchmark uses
  for accuracy (:func:`run_real_kg_amortization._accuracy`).
* A rule PROMOTES after ``k`` distinct agreeing unseen heads with zero
  disagreements; from promotion onward it routes, replacing teacher calls for
  covered pairs. A rule DEMOTES permanently on the first disagreement.
* A junk self-rule only reproduces facts already written back, so on an unseen
  head (whose fact is not yet written back) it predicts nothing, is never
  checked, and is rejected structurally -- without a single gold label.

The harness reuses the controlled runner's record/replay and shared-answer
machinery unchanged (:mod:`run_real_kg_controlled`): the pure cache arm is its
``_replay_cascade`` verbatim, and the shadow arm streams through the same
``TACET`` + ``ReplayTeacher`` with the shadow bookkeeping wired around it. The
teacher answers are served from a recorded ladder in replay mode or a free,
optionally noisy oracle in ``TACET_TEACHER=oracle`` mode, so a gamma-fixed E12
run pays the teacher at most once per distinct pair (or nothing, for the oracle).

Run::

    # free oracle (mechanism check)
    export TACET_TEACHER=oracle
    uv run python experiments/run_shadow_validation.py --slug oracle --seed 0

    # replay a recorded teacher ladder (no API calls)
    uv run python experiments/run_shadow_validation.py \
        --answers experiments/results/ladder --slug grok-4.3 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from run_real_kg_amortization import (  # noqa: E402
    COMPOSITIONS,
    BudgetGuard,
    _accuracy,
    _build_composed_workload,
    _build_workload,
    _kg_without,
    _new_metered,
    _oracle_gold_from_pool,
    _zipf_stream,
    resolve_price_key,
)
from run_real_kg_controlled import (  # noqa: E402
    ANSWERS_RECORD_SCHEMA,
    TIER2_OFF,
    AnswerRecordError,
    ReplayAnswerCache,
    ReplayTeacher,
    SharedAnswerCache,
    _check_replay_provenance,
    _load_answers_record,
    _replay_cascade,
)

from tacet.cascade.router import TACET  # noqa: E402
from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.core.ontology import Ontology, RelationType  # noqa: E402
from tacet.core.symbolic import Rule  # noqa: E402
from tacet.data.metaqa import load_metaqa  # noqa: E402
from tacet.distill.distill import MinedRule, mine_rules  # noqa: E402
from tacet.serve.config import CascadeConfig, KGEConfig  # noqa: E402
from tacet.serve.settings import load_settings  # noqa: E402

#: The pre-registered candidate confidence the shadow miner runs at.
GAMMA_CANDIDATE = 0.50


def _is_var(token: str) -> bool:
    return token.startswith("?")


def _is_self_rule(rule: Rule) -> bool:
    """True if the rule's head relation also appears in its body (self-referential).

    A junk self-rule composes the target relation with itself (via a decoy or an
    identity edge), so on unseen heads its body -- which depends on written-back
    target facts -- cannot fire. Detecting this shape lets the report flag the
    failure mode ``junk_promoted`` should such a rule ever slip through.
    """
    return rule.head[1] in {atom[1] for atom in rule.body}


def _shadow_prediction(rule: Rule, graph: WorldGraph, head: str) -> set[str]:
    """Tails the mined ``rule`` derives for a single ``head`` over ``graph``.

    A bounded, single-seed forward evaluation of the rule body (the miner emits
    length-1 and length-2 range-restricted bodies chained from ``?x`` through an
    intermediate ``?z`` to ``?y``), reusing the graph's O(degree) adjacency
    instead of materialising the whole closure per check. It binds ``?x = head``,
    propagates each body atom along the graph edges (forward on a bound subject,
    backward on a bound object), applies the rule's ``distinct`` guards, and
    returns the set of ``?y`` bindings. An empty result means the rule does not
    cover ``(head, target)`` on this graph.
    """
    hx, _, hy = rule.head
    bindings: list[dict[str, str]] = [{hx: head}]
    for s, rel, o in rule.body:
        nxt: list[dict[str, str]] = []
        for b in bindings:
            s_bound = (not _is_var(s)) or (s in b)
            o_bound = (not _is_var(o)) or (o in b)
            s_val = b.get(s, s) if _is_var(s) else s
            o_val = b.get(o, o) if _is_var(o) else o
            if s_bound and not o_bound:
                for t in graph.out(s_val, rel):
                    nxt.append({**b, o: t})
            elif o_bound and not s_bound:
                for src in graph.into(o_val, rel):
                    nxt.append({**b, s: src})
            elif s_bound and o_bound:
                if o_val in graph.out(s_val, rel):
                    nxt.append(dict(b))
            else:
                raise ValueError(
                    f"rule {rule.name!r} has an unconnected body atom {(s, rel, o)!r}; "
                    f"the miner only emits bodies chained from the head variable."
                )
        bindings = nxt
    tails: set[str] = set()
    for b in bindings:
        if any(b.get(a) == b.get(c) for a, c in rule.distinct):
            continue
        y = b.get(hy)
        if y is not None:
            tails.add(y)
    return tails


@dataclass
class _ShadowRule:
    """Per-rule shadow bookkeeping: state machine plus the agreeing unseen heads.

    ``training_heads`` are the heads the teacher had answered when the rule was
    mined; an unseen head is any head absent from that set. ``agreements`` is the
    set of distinct unseen heads whose prediction matched the teacher; the rule
    promotes once it reaches ``k`` with no disagreement, and demotes on the first
    disagreement.
    """

    mined: MinedRule
    training_heads: frozenset[str]
    state: str = "shadow"  # "shadow" -> "promoted" | "demoted"
    agreements: set[str] = field(default_factory=set)


def _replay_shadow_arm(
    stream,  # noqa: ANN001
    bench,  # noqa: ANN001
    ontology: Ontology,
    shared: SharedAnswerCache,
    *,
    gamma_candidate: float,
    k: int,
    kge_cfg: KGEConfig,
) -> dict:
    """Stream through a cache cascade with shadow rule-validation wired around it.

    The cascade routes exactly like the pure cache arm (write-back only, Tier-2
    off, no immediate rule install). In parallel the arm mines rules at
    ``gamma_candidate`` -- once per relation, at the same complete-heads trigger
    the full arm uses -- and holds them in SHADOW mode. On each teacher call for
    an unseen head a shadow rule covers, it compares the rule's prediction with
    the teacher's answer; the k-th agreement promotes the rule (it is added to the
    routing engine and the closure re-materialised), a disagreement demotes it.
    Promotion latency is real: the k agreeing checks are still teacher calls, so
    savings begin only after promotion.
    """
    replay = ReplayTeacher(shared)
    cfg = CascadeConfig(
        kge=kge_cfg,
        rule_synthesis=False,
        kge_augment=False,
        write_back=True,
        l2_threshold=TIER2_OFF,
    )
    ak = TACET(_kg_without(bench, stream), ontology, replay, config=cfg)
    ak.warmup()

    shadow: dict[str, _ShadowRule] = {}
    mined_relations: set[str] = set()
    promoted: list[str] = []
    demoted: list[str] = []
    shadow_checks = 0
    correct = 0

    for h, r, gold in stream:
        # A teacher call happens only when Tier 1 (write-back cache + any promoted
        # rule) cannot answer. Detect it before the ask so the shadow prediction is
        # taken over the graph BEFORE this head's own write-back is applied -- a
        # junk self-rule must not see the fact the teacher is about to write back.
        teacher_event = not ak.engine.query(h, r).answered
        predictions: dict[str, set[str]] = {}
        if teacher_event:
            for name, sr in shadow.items():
                if sr.state != "shadow" or sr.mined.rule.head[1] != r:
                    continue
                if h in sr.training_heads or h in sr.agreements:
                    continue  # seen at mining time, or already counted for this rule
                pred = _shadow_prediction(sr.mined.rule, ak.graph, h)
                if pred:  # a non-empty prediction is what "covers" means
                    predictions[name] = pred

        ans = ak.ask(h, r)
        correct += int(_accuracy(gold, ans.answers))

        if teacher_event and ans.tier == 3:
            teacher_answer = frozenset(ans.answers)  # the shared teacher answer
            for name, pred in predictions.items():
                sr = shadow[name]
                if sr.state != "shadow":
                    continue
                shadow_checks += 1
                if _accuracy(teacher_answer, sorted(pred)):
                    sr.agreements.add(h)
                    if len(sr.agreements) >= k:
                        sr.state = "promoted"
                        promoted.append(name)
                        if ak.engine.add_rule(sr.mined.rule):
                            ak.engine.materialise(ak.graph)
                else:
                    sr.state = "demoted"
                    demoted.append(name)

            # Mine each relation once, at the full arm's complete-heads trigger, but
            # at the candidate confidence and WITHOUT installing (shadow, not route).
            complete = ak.distiller._complete_heads.get(r, set())
            if r not in mined_relations and len(complete) >= ak.distiller.synth_trigger:
                mined_relations.add(r)
                training = frozenset(complete)
                for m in mine_rules(
                    ak.graph,
                    ak.distiller.teacher_facts,
                    r,
                    min_confidence=gamma_candidate,
                    min_support=ak.distiller.min_support,
                    complete_heads=complete,
                    allowed_body=ak.distiller.base_relations or None,
                ):
                    shadow.setdefault(m.rule.name, _ShadowRule(m, training))

    n = len(stream)
    return {
        "teacher_calls": replay.n_calls,
        # Rounded to the same precision as the cache arm's accuracy
        # (run_real_kg_controlled._replay_cascade), so two arms that behave
        # identically report identical accuracy instead of diverging on the
        # rounding alone.
        "accuracy": round(correct / n, 4) if n else 0.0,
        "promoted_rules": promoted,
        "demoted_rules": demoted,
        "shadow_rules_mined": sorted(shadow),
        "shadow_checks_used": shadow_checks,
        "junk_promoted": any(_is_self_rule(shadow[name].mined.rule) for name in promoted),
    }


def _build_shared(
    *,
    replay_mode: bool,
    answers_path: str | None,
    settings,  # noqa: ANN001
    model: str,
    nl_template: str | None,
    gold_map: dict[str, frozenset[str]],
    guard: BudgetGuard,
    kg: WorldGraph,
    oracle_error_rate: float,
    seed: int,
    hop: int,
    split: str,
    limit: int,
    zipf_a: float,
    composed_relation: str | None,
) -> tuple[SharedAnswerCache, int | None]:
    """The one teacher-answer cache both arms share, plus the record's answer cap.

    In replay mode the record's provenance is validated against this run's stream
    parameters -- including ``composed_relation`` for a hop>=2 ladder -- so a record
    made for a different stream is refused, never silently mis-served. The answer-
    discipline cap ``response_format_max_items`` is INHERITED from the record rather
    than matched: a replay calls no live teacher, so there is no run-side cap to
    compare against (matching one would make a structured record impossible to replay).
    The cap is returned so the report can echo it -- the record's value on replay
    (``None`` for a legacy/unconstrained record), or ``None`` in the oracle path, where
    no teacher cap applies.
    """
    if replay_mode:
        record = _load_answers_record(answers_path)
        schema = record.get("schema")
        if schema != ANSWERS_RECORD_SCHEMA:
            raise AnswerRecordError(
                f"answers record schema {schema!r} != expected {ANSWERS_RECORD_SCHEMA!r}; "
                f"re-record with the current code."
            )
        _check_replay_provenance(
            record,
            hop=hop,
            split=split,
            limit=limit,
            zipf_a=zipf_a,
            seed=seed,
            composed_relation=composed_relation,
            # Inherit the record's cap instead of matching it: a replay declares no
            # run-side cap (no live teacher), so comparing would refuse any structured
            # record. Only this field is inherited; all others still match strictly.
            inherit_fields=frozenset({"response_format_max_items"}),
        )
        max_items = record.get("provenance", {}).get("response_format_max_items")
        return ReplayAnswerCache.from_record(record, guard, kg), max_items
    metered = _new_metered(
        settings, model, nl_template, gold_map, error_rate=oracle_error_rate, seed=seed
    )
    return SharedAnswerCache(metered, guard, kg), None


def _resolve_composition(composition: str | None, hop: int) -> tuple[dict, str]:
    """The controlled runner's composition lookup: an explicit name, else by hop."""
    comp_name = composition or next(
        (name for name, spec in COMPOSITIONS.items() if spec.get("hop") == hop), None
    )
    if comp_name is None:
        raise SystemExit(f"no composition for hop={hop}")
    return COMPOSITIONS[comp_name], comp_name


def shadow_report(
    slug: str,
    seed: int,
    *,
    hop: int = 1,
    split: str = "test",
    limit: int = 300,
    zipf_a: float = 1.5,
    budget_usd: float = 1e9,
    k: int = 3,
    gamma_candidate: float = GAMMA_CANDIDATE,
    oracle_error_rate: float = 0.0,
    answers_path: str | None = None,
    composition: str | None = None,
    composed_relation: str | None = None,
    metaqa_root: str = "data/MetaQA",
    settings=None,  # noqa: ANN001
    bench=None,  # noqa: ANN001
    stream=None,  # noqa: ANN001
    verbose: bool = False,
) -> dict:
    """Run the E12 shadow-validation pipeline and return its report.

    Reuses the controlled runner's design (Tier-2 disabled, one deterministic
    teacher answer shared across arms) to compare a pure cache arm against a
    shadow-validation arm on the same stream. ``net_calls_saved_pct`` is the
    relative reduction in teacher calls of the shadow arm versus the pure cache
    arm over the full stream; it is non-negative by construction (a shadow rule
    only ever adds Tier-1 coverage, never removes it).

    ``hop`` selects the workload. ``hop == 1`` streams real 1-hop relations;
    ``hop >= 2`` streams a synthetic COMPOSED relation built exactly as the
    controlled runner does -- resolved from :data:`COMPOSITIONS` (or an explicit
    ``composition`` name), its edges materialised by ``_build_composed_workload``,
    and its type added to the induced ontology so a mined composed rule can
    install. The E11 recorded ladder is hop 2 on the ``q2`` composed relation, so
    this path is what replays those records. ``composed_relation`` names that
    relation for the replay provenance guard; on the default (non-injected) path
    it is taken from the composition spec.

    ``bench`` and ``settings`` are injectable so tests can drive a tiny synthetic
    KG without MetaQA, and ``stream`` (a list of ``(head, relation, gold)``) can
    be supplied directly to pin the exact query order; an injected stream whose
    relation is absent from the induced ontology (a synthetic composed relation)
    has an ``Entity -> Entity`` type added for it, mirroring the default path.
    With ``settings.teacher == 'oracle'`` a free ground-truth oracle feeds the
    shared cache; when ``answers_path`` names an existing record the teacher
    answers are replayed from it (no API calls, provenance-checked). ``k`` is the
    promotion threshold (registered default 3; 2 and 5 are the descriptive
    sensitivity values). The returned dict carries, among context fields, the
    aggregation keys ``net_calls_saved_pct``, ``promoted_rules``,
    ``demoted_rules``, ``shadow_checks_used``, ``cache_accuracy``,
    ``shadow_accuracy`` and ``junk_promoted``.
    """
    if settings is None:
        settings = load_settings()
    if bench is None:
        bench = load_metaqa(metaqa_root, hop=1, split=split)

    replay_mode = answers_path is not None and Path(answers_path).exists()
    rng = np.random.default_rng(seed)
    ontology = Ontology.induce(bench.kg)
    nl_template: str | None = None

    if stream is None:
        if hop == 1:
            pool = _build_workload(bench, limit_pool=max(limit, 400), rng=rng)
        else:
            spec, comp_name = _resolve_composition(composition, hop)
            nl_template = spec["nl"]
            composed_relation = spec["kg_relation"]
            pool, _full_gold = _build_composed_workload(
                bench.kg, spec, limit_pool=max(limit, 400), rng=rng
            )
            if not pool:
                raise SystemExit(f"composition {comp_name!r} produced an empty pool")
            ontology.add_relation_type(
                RelationType(composed_relation, frozenset({"Entity"}), frozenset({"Entity"}))
            )
        stream = _zipf_stream(pool, limit, zipf_a, rng)
        gold_map = _oracle_gold_from_pool(pool)
    else:
        stream = list(stream)
        gold_map = {f"{h}\t{r}": g for h, r, g in stream}
        # Give any streamed relation the induced ontology does not know (a synthetic
        # composed relation) an open Entity->Entity type, exactly as the default
        # hop>=2 path adds the composition's kg_relation, so a mined rule on it can
        # pass the engine's ontology-consistency check when it promotes.
        for rel in sorted({r for _, r, _ in stream}):
            if ontology.relation(rel) is None:
                ontology.add_relation_type(
                    RelationType(rel, frozenset({"Entity"}), frozenset({"Entity"}))
                )

    distinct = len({(h, r) for h, r, _ in stream})
    kge_cfg = KGEConfig(
        dim=min(getattr(settings, "kge_dim", 8), 32),
        epochs=min(getattr(settings, "kge_epochs", 2), 15),
        batch_size=4096,
    )
    guard = BudgetGuard(budget_usd)
    model = resolve_price_key(settings)
    shared, response_format_max_items = _build_shared(
        replay_mode=replay_mode,
        answers_path=answers_path,
        settings=settings,
        model=model,
        nl_template=nl_template,
        gold_map=gold_map,
        guard=guard,
        kg=bench.kg,
        oracle_error_rate=oracle_error_rate,
        seed=seed,
        hop=hop,
        split=split,
        limit=limit,
        zipf_a=zipf_a,
        composed_relation=composed_relation,
    )

    if verbose:
        print(f"[e12] slug={slug} seed={seed} k={k} gamma_candidate={gamma_candidate}")
        print(f"  stream={len(stream)} distinct={distinct}")

    cfg_cache = CascadeConfig(
        kge=kge_cfg,
        rule_synthesis=False,
        kge_augment=False,
        write_back=True,
        l2_threshold=TIER2_OFF,
    )
    cache = _replay_cascade("cache_cascade", stream, bench, ontology, shared, cfg_cache)
    shadow = _replay_shadow_arm(
        stream, bench, ontology, shared, gamma_candidate=gamma_candidate, k=k, kge_cfg=kge_cfg
    )

    cache_calls = cache["teacher_calls"]
    shadow_calls = shadow["teacher_calls"]
    net = round(100.0 * (cache_calls - shadow_calls) / cache_calls, 4) if cache_calls > 0 else 0.0

    report = {
        "experiment": "E12-shadow-validation",
        "slug": slug,
        "seed": seed,
        "hop": hop,
        # Label the actual workload hop, matching run_real_kg_controlled: the bench is
        # loaded at hop=1 for its KB even on a hop-2 run, so bench.name would say 1hop.
        "dataset": f"MetaQA-{hop}hop-{split}",
        "composed_relation": composed_relation,
        "k": k,
        "gamma_candidate": gamma_candidate,
        # The answer-discipline cap the replayed record's answers were made under,
        # inherited from the record on replay (None for a legacy/unconstrained record or
        # the oracle path), so a future structured E12 replay is distinguishable from an
        # unconstrained one.
        "response_format_max_items": response_format_max_items,
        "teacher_kind": (
            "replay"
            if replay_mode
            else ("oracle" if getattr(settings, "teacher", None) == "oracle" else model)
        ),
        "stream_len": len(stream),
        "distinct_queries": distinct,
        "cache_teacher_calls": cache_calls,
        "shadow_teacher_calls": shadow_calls,
        "net_calls_saved_pct": net,
        "promoted_rules": shadow["promoted_rules"],
        "demoted_rules": shadow["demoted_rules"],
        "shadow_rules_mined": shadow["shadow_rules_mined"],
        "shadow_checks_used": shadow["shadow_checks_used"],
        "cache_accuracy": cache["accuracy"],
        "shadow_accuracy": shadow["accuracy"],
        "junk_promoted": shadow["junk_promoted"],
    }
    if verbose:
        print(
            f"  cache_calls={cache_calls} shadow_calls={shadow_calls} "
            f"net_saved={net}% promoted={report['promoted_rules']} "
            f"demoted={report['demoted_rules']} checks={report['shadow_checks_used']}"
        )
    return report


def _resolve_answers_path(answers: str | None, slug: str, seed: int) -> str | None:
    """Resolve the recorded-answers file for ``slug`` / ``seed`` from ``--answers``.

    ``answers`` may name the record file directly or a directory holding one file
    per ``(slug, seed)`` as ``<slug>_seed<seed>.json``. ``None`` (no ``--answers``)
    keeps the oracle path.
    """
    if answers is None:
        return None
    path = Path(answers)
    if path.is_dir():
        return str(path / f"{slug}_seed{seed}.json")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metaqa-root", default="data/MetaQA")
    ap.add_argument("--hop", type=int, default=1)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--zipf-a", type=float, default=1.5)
    ap.add_argument("--budget-usd", type=float, default=1.5)
    ap.add_argument("--slug", required=True, help="teacher identifier stamped into the report")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--k", type=int, default=3, help="distinct agreeing unseen heads required to promote"
    )
    ap.add_argument(
        "--gamma-candidate",
        type=float,
        default=GAMMA_CANDIDATE,
        help="confidence the shadow miner runs at (pre-registered 0.50)",
    )
    ap.add_argument(
        "--composition",
        default=None,
        choices=sorted(COMPOSITIONS),
        help="composition name for hop>=2 (default: the first matching the --hop value)",
    )
    ap.add_argument(
        "--answers",
        default=None,
        help=(
            "recorded-answers file or a directory of <slug>_seed<seed>.json records "
            "to replay (no API calls); omit for the free oracle teacher"
        ),
    )
    ap.add_argument("--out", default=None, help="write the JSON report here (default: stdout)")
    args = ap.parse_args()

    oracle_error_rate = float(os.environ.get("TACET_ORACLE_ERROR_RATE", "0.0"))
    answers_path = _resolve_answers_path(args.answers, args.slug, args.seed)
    report = shadow_report(
        args.slug,
        args.seed,
        hop=args.hop,
        split=args.split,
        limit=args.limit,
        zipf_a=args.zipf_a,
        budget_usd=args.budget_usd,
        k=args.k,
        gamma_candidate=args.gamma_candidate,
        oracle_error_rate=oracle_error_rate,
        answers_path=answers_path,
        composition=args.composition,
        metaqa_root=args.metaqa_root,
        verbose=True,
    )
    text = json.dumps(report, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
