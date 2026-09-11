"""Plumbing tests for the E18 refined pure-self-loop guard (E18-W2 generative half).

The miner-level behaviour (``mine_rules(forbid_target_self_loop=True)`` bans only the
pure all-target body, keeps mixed legs) is pinned in ``test_e18_recursive.py``. These
tests pin the PLUMBING: the flag must reach the distiller through ``CascadeConfig`` →
``Router.warmup``, and the controlled runner must accept it and stamp it into the
report so an artifact records which guard produced it.

All fixtures are TINY and SYNTHETIC: MetaQA is never loaded and no network is used.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_real_kg_controlled import run_controlled  # noqa: E402

from tacet.cascade.router import TACET  # noqa: E402,N811
from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.core.ontology import Ontology  # noqa: E402
from tacet.data.metaqa import MetaQABenchmark, MetaQAQuestion  # noqa: E402
from tacet.serve.config import CascadeConfig  # noqa: E402


def _tiny_graph() -> WorldGraph:
    kg = WorldGraph(name="tiny-plumbing")
    for i in range(6):
        kg.add_edge(f"M{i}", "directed_by", f"D{i}")
        kg.add_edge(f"M{i}", "has_genre", "drama")
    return kg


def _tiny_bench() -> MetaQABenchmark:
    kg = _tiny_graph()
    questions = [
        MetaQAQuestion(question=f"who directed [M{i}]?", head=f"M{i}", answers=[f"D{i}"], hop=1)
        for i in range(6)
    ]
    return MetaQABenchmark(
        name="tiny-plumbing",
        hop=1,
        split="test",
        kg=kg,
        questions=questions,
        entities=set(kg.entities()),
        relations=kg.relations(),
    )


def _oracle_settings() -> SimpleNamespace:
    return SimpleNamespace(
        teacher="oracle", xai_model="grok-4.3", xai_api_key=None, kge_dim=8, kge_epochs=2
    )


# ------------------------------------------------- 1. config flag reaches the distiller
def test_cascade_config_flag_reaches_distiller() -> None:
    refined = TACET(
        _tiny_graph(), Ontology(), None, config=CascadeConfig(forbid_target_self_loop=True)
    ).warmup()
    assert refined.distiller.forbid_target_self_loop is True

    published = TACET(_tiny_graph(), Ontology(), None, config=CascadeConfig()).warmup()
    assert published.distiller.forbid_target_self_loop is False


# ------------------------------------------------------- 2. runner accepts and stamps it
def test_runner_accepts_and_stamps_the_flag() -> None:
    bench, settings = _tiny_bench(), _oracle_settings()
    common = dict(
        hop=1,
        split="test",
        limit=6,
        zipf_a=1.5,
        seed=0,
        oracle_error_rate=0.0,
        gamma=0.5,
        bench=bench,
        settings=settings,
        verbose=False,
    )
    published = run_controlled(**common)
    # Published defaults: target may appear in length-2 bodies, no pure-self-loop ban.
    assert published["allow_target_in_body"] is True
    assert published["forbid_target_self_loop"] is False
    refined = run_controlled(forbid_target_self_loop=True, **common)
    # The refined guard flips ONLY the self-loop ban, not the in-body allowance.
    assert refined["allow_target_in_body"] is True
    assert refined["forbid_target_self_loop"] is True

    # The published arm is untouched by the refined flag on this bench (no
    # self-referential rule exists here); both runs produce the full arm reports.
    assert refined["arms"] and published["arms"]
