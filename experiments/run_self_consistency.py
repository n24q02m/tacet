"""E13 self-consistency — label-free rule promotion from a k-sample majority.

E12 (``run_shadow_validation.py``) ran a mined rule in SHADOW at
gamma_candidate = 0.50 and promoted it only after it agreed, on ``k`` distinct
unseen heads, with a SINGLE teacher answer. That signal was inert: a rule
installs precisely where the teacher is least accurate, so a world-correct rule
there CORRECTS the teacher and therefore DISAGREES with its single sample. E13
tests the label-free alternative the E12 write-up pre-registered as untested --
multi-sample self-consistency:

* The cascade mines rules at gamma_candidate = 0.50 and holds every installed
  rule in SHADOW (it predicts but never routes), exactly as E12.
* The teacher answer for each distinct ``(head, relation)`` is now a ``k``-sample
  record: ``k`` answer-sets (in the real run, ``k`` queries at temperature; here,
  ``k`` answer-sets replayed from a record). The MAJORITY answer set is the
  entities present in at least ``ceil(k / 2)`` of the samples
  (:func:`run_real_kg_controlled.majority`).
* On each teacher call for an unseen head a shadow rule covers, the rule's
  prediction is compared with the MAJORITY (not a single sample) under the SAME
  gold-subset criterion the benchmark uses (:func:`run_real_kg_amortization._accuracy`).
* A rule PROMOTES after ``k_prime`` distinct agreeing unseen heads with zero
  disagreements; from promotion onward it routes. A rule DEMOTES permanently on
  the first majority-disagreement. A junk self-rule predicts nothing on unseen
  heads, is never checked, and is rejected structurally -- exactly as in E12.

The harness reuses the E12 shadow arm unchanged, injecting only the majority
``signal`` (:func:`run_shadow_validation._replay_shadow_arm`), and the controlled
runner's record/replay + shared-answer machinery
(:mod:`run_real_kg_controlled`). A ``k``-sample record is replayed through
:class:`run_real_kg_controlled.KSampleReplayAnswerCache`, which serves the primary
sample for routing (single-sample base cost, identical to E12) and votes the
majority for the promotion check. A ``k == 1`` record's majority is its lone
sample, so E13 reproduces the E12 report exactly.

Run::

    # free oracle (mechanism check; the oracle's one answer is its own majority)
    export TACET_TEACHER=oracle
    uv run python experiments/run_self_consistency.py --slug oracle --seed 0

    # replay a recorded k-sample teacher ladder (no API calls)
    uv run python experiments/run_self_consistency.py \
        --answers experiments/results/ladder_ksample --slug grok-4.3 --seed 0 --k 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from run_real_kg_amortization import (  # noqa: E402
    COMPOSITIONS,
    BudgetGuard,
    _build_composed_workload,
    _build_workload,
    _new_metered,
    _oracle_gold_from_pool,
    _zipf_stream,
    resolve_price_key,
)
from run_real_kg_controlled import (  # noqa: E402
    ANSWERS_RECORD_SCHEMA,
    TIER2_OFF,
    AnswerRecordError,
    KSampleReplayAnswerCache,
    SharedAnswerCache,
    _check_replay_provenance,
    _load_answers_record,
    _replay_cascade,
)
from run_shadow_validation import (  # noqa: E402
    GAMMA_CANDIDATE,
    _replay_shadow_arm,
    _resolve_composition,
)

from tacet.core.ontology import Ontology, RelationType  # noqa: E402
from tacet.data.metaqa import load_metaqa  # noqa: E402
from tacet.serve.config import CascadeConfig, KGEConfig  # noqa: E402
from tacet.serve.settings import load_settings  # noqa: E402


def _majority_signal(shared, head, relation):  # noqa: ANN001
    """E13 validation target: the k-sample self-consistency majority for a pair.

    Read from the shared cache AFTER the pair has been served, so it never triggers
    or re-charges a teacher call. The base :class:`SharedAnswerCache` (oracle path)
    returns its single served answer -- its own majority -- so the mechanism check
    degenerates cleanly to the E12 single-sample signal.
    """
    return shared.majority(head, relation)


def _build_shared(
    *,
    replay_mode: bool,
    answers_path: str | None,
    settings,  # noqa: ANN001
    model: str,
    nl_template: str | None,
    gold_map: dict[str, frozenset[str]],
    guard: BudgetGuard,
    kg,  # noqa: ANN001
    oracle_error_rate: float,
    seed: int,
    hop: int,
    split: str,
    limit: int,
    zipf_a: float,
    composed_relation: str | None,
) -> tuple[SharedAnswerCache, int | None, int | None]:
    """The one teacher cache both arms share, plus the record's answer cap and ``k``.

    Mirrors :func:`run_shadow_validation._build_shared` but, on replay, returns a
    :class:`KSampleReplayAnswerCache` so the shadow arm can vote the record's ``k``
    samples: the primary sample is served for routing (single-sample base cost,
    identical to E12) while :meth:`majority` votes over all ``k``. The record's ``k``
    (its samples-per-pair) is returned so the report can price the extra teacher cost
    of self-consistency; it is ``None`` in the oracle path, where the caller's ``--k``
    stands in. As in E12, the ``response_format_max_items`` cap is INHERITED from the
    record on replay (no live teacher declares one) and every stream-fixing provenance
    field is still matched strictly -- a ``k``-sample record made for a different stream
    is refused, never mis-served.
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
            inherit_fields=frozenset({"response_format_max_items"}),
        )
        prov = record.get("provenance", {})
        max_items = prov.get("response_format_max_items")
        k_record = prov.get("k", 1)
        return KSampleReplayAnswerCache.from_record(record, guard, kg), max_items, k_record
    metered = _new_metered(
        settings, model, nl_template, gold_map, error_rate=oracle_error_rate, seed=seed
    )
    return SharedAnswerCache(metered, guard, kg), None, None


def self_consistency_report(
    slug: str,
    seed: int,
    *,
    hop: int = 1,
    split: str = "test",
    limit: int = 300,
    zipf_a: float = 1.5,
    budget_usd: float = 1e9,
    k: int = 3,
    k_prime: int = 3,
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
    """Run the E13 self-consistency pipeline and return its report.

    Mirrors :func:`run_shadow_validation.shadow_report` -- same controlled design
    (Tier-2 disabled, one shared teacher record across arms), same pure-cache arm,
    same injectable ``bench`` / ``settings`` / ``stream`` for tiny synthetic tests --
    but validates each shadow rule against the k-sample MAJORITY. ``k`` is the samples
    per pair (the majority denominator; on replay the record's own ``k`` overrides it),
    and ``k_prime`` is the distinct agreeing unseen heads required to promote (the E12
    ``k`` role). ``net_calls_saved_pct`` is the relative reduction in single-sample
    teacher calls of the shadow arm versus the pure cache arm; it is non-negative by
    construction. ``self_consistency_calls`` = ``k`` x ``shadow_checks_used`` is the
    EXTRA teacher cost self-consistency pays (``k`` samples at each shadow check), which
    the analysis weighs against the savings.

    ``hop`` selects the workload exactly as in E12: ``hop == 1`` streams real 1-hop
    relations; ``hop >= 2`` streams the synthetic COMPOSED relation. ``answers_path``,
    when it names an existing ``k``-sample record, replays it (no API calls,
    provenance-checked); otherwise the free oracle feeds the shared cache and every
    ``k`` samples coincide. The returned dict carries, among context fields, the
    aggregation keys ``net_calls_saved_pct``, ``promoted_rules``, ``demoted_rules``,
    ``shadow_checks_used``, ``self_consistency_calls``, ``cache_accuracy``,
    ``shadow_accuracy``, ``junk_promoted``, ``k`` and ``k_prime``.
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
    shared, response_format_max_items, k_record = _build_shared(
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
    # On replay the record's own samples-per-pair is authoritative (it fixes how many
    # samples the majority votes over); the oracle path has no record, so the caller's
    # --k stands in as the intended sample count for the cost accounting.
    effective_k = k_record if k_record is not None else k

    if verbose:
        print(
            f"[e13] slug={slug} seed={seed} k={effective_k} k_prime={k_prime} "
            f"gamma_candidate={gamma_candidate}"
        )
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
        stream,
        bench,
        ontology,
        shared,
        gamma_candidate=gamma_candidate,
        k=k_prime,
        kge_cfg=kge_cfg,
        signal=_majority_signal,
    )

    cache_calls = cache["teacher_calls"]
    shadow_calls = shadow["teacher_calls"]
    net = round(100.0 * (cache_calls - shadow_calls) / cache_calls, 4) if cache_calls > 0 else 0.0
    shadow_checks = shadow["shadow_checks_used"]

    report = {
        "experiment": "E13-self-consistency",
        "slug": slug,
        "seed": seed,
        "hop": hop,
        "dataset": f"MetaQA-{hop}hop-{split}",
        "composed_relation": composed_relation,
        # k = samples voted per pair; k_prime = distinct agreeing unseen heads to promote.
        "k": effective_k,
        "k_prime": k_prime,
        "gamma_candidate": gamma_candidate,
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
        "shadow_checks_used": shadow_checks,
        # The extra teacher cost self-consistency pays: k samples at each shadow check.
        "self_consistency_calls": effective_k * shadow_checks,
        "cache_accuracy": cache["accuracy"],
        "shadow_accuracy": shadow["accuracy"],
        "junk_promoted": shadow["junk_promoted"],
    }
    if verbose:
        print(
            f"  cache_calls={cache_calls} shadow_calls={shadow_calls} "
            f"net_saved={net}% promoted={report['promoted_rules']} "
            f"demoted={report['demoted_rules']} checks={shadow_checks} "
            f"self_consistency_calls={report['self_consistency_calls']}"
        )
    return report


def _resolve_answers_path(answers: str | None, slug: str, seed: int) -> str | None:
    """Resolve the recorded-answers file for ``slug`` / ``seed`` from ``--answers``.

    ``answers`` may name the record file directly or a directory holding one file per
    ``(slug, seed)`` as ``<slug>_seed<seed>.json``. ``None`` keeps the oracle path.
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
        "--k",
        type=int,
        default=3,
        help="teacher samples voted per pair (majority denominator; pre-registered 3)",
    )
    ap.add_argument(
        "--k-prime",
        type=int,
        default=3,
        help="distinct agreeing unseen heads required to promote (pre-registered 3)",
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
            "recorded k-sample answers file or a directory of <slug>_seed<seed>.json "
            "records to replay (no API calls); omit for the free oracle teacher"
        ),
    )
    ap.add_argument("--out", default=None, help="write the JSON report here (default: stdout)")
    args = ap.parse_args()

    oracle_error_rate = float(os.environ.get("TACET_ORACLE_ERROR_RATE", "0.0"))
    answers_path = _resolve_answers_path(args.answers, args.slug, args.seed)
    report = self_consistency_report(
        args.slug,
        args.seed,
        hop=args.hop,
        split=args.split,
        limit=args.limit,
        zipf_a=args.zipf_a,
        budget_usd=args.budget_usd,
        k=args.k,
        k_prime=args.k_prime,
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
