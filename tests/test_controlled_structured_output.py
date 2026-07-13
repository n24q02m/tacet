"""Structured-output (--max-items) tests for the controlled runner (E11 aftermath).

Phase 2 of the E11 aftermath lets a recording run constrain the OpenRouter teacher
with an OpenAI-style json_schema that caps the answer list (``maxItems``), so a
later re-recorded ladder measures distillability with answer discipline enforced.
These tests pin:

* (a) the runner's builder threads the EXACT schema through to the OpenRouter
  client, and the default (``--max-items`` unset) path stays byte-identical to
  today (no ``response_format`` key emitted);
* (c) a structured record refuses to replay against an unstructured run and vice
  versa -- the provenance guard treats ``response_format_max_items`` like every
  other stream-fixing field, naming it in the error;
* (d) ``--max-items`` is refused for any non-openrouter teacher and in replay mode.

All fixtures are TINY and SYNTHETIC: MetaQA is never loaded. The teacher is a fake
(no network) except in (a), where a real ``OpenRouterTeacher`` is built and its SDK
client is swapped for a stub that captures the call kwargs.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import run_real_kg_controlled as rkc  # noqa: E402
from run_real_kg_controlled import (  # noqa: E402
    ProvenanceMismatchError,
    run_controlled,
)

from tacet.core.graph import WorldGraph  # noqa: E402
from tacet.data.metaqa import MetaQABenchmark, MetaQAQuestion  # noqa: E402
from tacet.llm.teacher import TeacherResponse  # noqa: E402


def _expected_schema(max_items: int) -> dict:
    """The exact OpenAI-style structured-output constraint the runner must build."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "kg_answers",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": max_items,
                    }
                },
                "required": ["answers"],
                "additionalProperties": False,
            },
        },
    }


# --------------------------------------------------------------- fake teacher (record path)
class _FakeMeteredTeacher:
    """Call-counting stand-in duck-typing the metered-teacher surface run_controlled reads."""

    def __init__(self, oracle_gold, counter, cost_per_call: float = 0.01) -> None:
        self._gold = oracle_gold or {}
        self._counter = counter
        self._cost = cost_per_call
        self.last_cost_usd = 0.0
        self.last_usage: dict | None = None

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        self._counter["calls"] += 1
        self.last_cost_usd = self._cost
        self.last_usage = {"prompt_tokens": 4, "completion_tokens": 2}
        return TeacherResponse(answers=sorted(self._gold.get(f"{head}\t{relation}", ())))


def _install_fake_teacher(monkeypatch, counter: dict) -> None:
    """Replace ``run_real_kg_controlled._new_metered`` with the counting fake.

    Signature mirrors the real ``_new_metered`` (including the new
    ``response_format`` keyword the builder forwards) so run_controlled's call site
    is unchanged; the fake ignores it because the provenance record -- not the fake
    teacher -- is what these tests inspect.
    """

    def factory(  # noqa: ANN001, ANN202
        settings, model, nl_template, oracle_gold=None, error_rate=0.0, seed=0, response_format=None
    ):
        return _FakeMeteredTeacher(oracle_gold, counter)

    monkeypatch.setattr(rkc, "_new_metered", factory)


# --------------------------------------------------------------- synthetic bench + settings
def _oracle_settings() -> SimpleNamespace:
    return SimpleNamespace(
        teacher="oracle", xai_model="grok-4.3", xai_api_key=None, kge_dim=8, kge_epochs=2
    )


def _openrouter_settings() -> SimpleNamespace:
    return SimpleNamespace(
        teacher="openrouter",
        openrouter_api_key="or-test-key",
        openrouter_model="x-ai/grok-4.3",
        xai_model="grok-4.3",
        kge_dim=8,
        kge_epochs=2,
    )


def _tiny_bench() -> MetaQABenchmark:
    """A few movies with directors + actors (flat 1-hop, no latent rule)."""
    kg = WorldGraph(name="tiny-movies")
    questions: list[MetaQAQuestion] = []
    directors = ["D1", "D2"]
    actors = ["A1", "A2", "A3"]
    for i in range(10):
        m = f"M{i}"
        d = directors[i % len(directors)]
        a = actors[i % len(actors)]
        kg.add_edge(m, "directed_by", d)
        kg.add_edge(m, "starred_actors", a)
        kg.add_edge(m, "has_genre", "drama")  # background: never queried
        questions.append(
            MetaQAQuestion(question=f"who directed [{m}]?", head=m, answers=[d], hop=1)
        )
        questions.append(
            MetaQAQuestion(question=f"who acted in [{m}]?", head=m, answers=[a], hop=1)
        )
    return MetaQABenchmark(
        name="tiny-movies",
        hop=1,
        split="test",
        kg=kg,
        questions=questions,
        entities=set(kg.entities()),
        relations=kg.relations(),
    )


def _common(bench, settings) -> dict:
    return dict(
        hop=1,
        split="test",
        limit=24,
        zipf_a=1.5,
        seed=0,
        oracle_error_rate=0.0,
        gamma=0.95,
        bench=bench,
        settings=settings,
        verbose=False,
    )


# ============================================================= (a) builder threads the schema
def test_runner_builder_passes_exact_max_items_schema() -> None:
    pytest.importorskip("openai")
    settings = _openrouter_settings()
    metered = rkc._build_metered_teacher(
        settings, "grok-4.3", None, {}, error_rate=0.0, seed=0, max_items=25
    )
    captured: list[dict] = []

    class _Completions:
        def create(self, **kwargs):  # noqa: ANN003, ANN201
            captured.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"answers": ["Belgium"]}'))
                ],
                usage=None,
            )

    # metered.wrapped is the OpenRouterTeacher (nl_template=None, so no CompositionTeacher).
    metered.wrapped._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    resp = metered.answer(None, "France", "borders")

    # The exact schema is sent to the provider ...
    assert captured[0]["response_format"] == _expected_schema(25)
    # ... and the object-rooted structured content is parsed back into a list.
    assert resp.answers == ["Belgium"]


def test_max_items_none_sends_no_response_format() -> None:
    pytest.importorskip("openai")
    settings = _openrouter_settings()
    metered = rkc._build_metered_teacher(
        settings, "grok-4.3", None, {}, error_rate=0.0, seed=0, max_items=None
    )
    captured: list[dict] = []

    class _Completions:
        def create(self, **kwargs):  # noqa: ANN003, ANN201
            captured.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='["Belgium"]'))],
                usage=None,
            )

    metered.wrapped._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    metered.answer(None, "France", "borders")
    # The default path stays byte-identical to today: the key is absent, not None.
    assert "response_format" not in captured[0]


# ============================================================= (c) provenance refusal both ways
def test_structured_record_refuses_replay_by_unstructured_run(tmp_path, monkeypatch) -> None:
    counter = {"calls": 0}
    _install_fake_teacher(monkeypatch, counter)
    bench, settings = _tiny_bench(), _openrouter_settings()
    path = tmp_path / "answers.json"

    run_controlled(max_items=25, answers_path=str(path), **_common(bench, settings))
    record = json.loads(path.read_text(encoding="utf-8"))
    # The record is self-describing: structured records carry the cap in provenance.
    assert record["provenance"]["response_format_max_items"] == 25

    calls_before_replay = counter["calls"]
    with pytest.raises(ProvenanceMismatchError) as excinfo:
        run_controlled(answers_path=str(path), **_common(bench, settings))
    assert "response_format_max_items" in str(excinfo.value)
    # refused before touching the teacher
    assert counter["calls"] == calls_before_replay


def test_provenance_check_flags_max_items_both_directions() -> None:
    base = dict(hop=1, split="test", limit=24, zipf_a=1.5, seed=0, composed_relation=None)
    structured = {"provenance": {**base, "response_format_max_items": 25}}
    unstructured = {"provenance": {**base, "response_format_max_items": None}}

    # a structured record (25) refuses a run not requesting it (None) ...
    with pytest.raises(ProvenanceMismatchError) as e1:
        rkc._check_replay_provenance(structured, **base, response_format_max_items=None)
    assert "response_format_max_items" in str(e1.value)

    # ... and vice versa: an E11 original (None) refuses a run requesting 25.
    with pytest.raises(ProvenanceMismatchError) as e2:
        rkc._check_replay_provenance(unstructured, **base, response_format_max_items=25)
    assert "response_format_max_items" in str(e2.value)

    # Matching values (both None = an E11 original replayed plainly) still replay.
    rkc._check_replay_provenance(unstructured, **base, response_format_max_items=None)


# ============================================================= (d) --max-items guards
def test_max_items_rejected_for_non_openrouter_teacher(monkeypatch) -> None:
    counter = {"calls": 0}
    _install_fake_teacher(monkeypatch, counter)
    bench, settings = _tiny_bench(), _oracle_settings()  # teacher="oracle"
    with pytest.raises(SystemExit) as excinfo:
        run_controlled(max_items=25, **_common(bench, settings))
    assert "openrouter" in str(excinfo.value).lower()
    assert counter["calls"] == 0  # refused before any teacher call


def test_max_items_rejected_in_replay_mode(tmp_path, monkeypatch) -> None:
    counter = {"calls": 0}
    _install_fake_teacher(monkeypatch, counter)
    bench, settings = _tiny_bench(), _openrouter_settings()
    path = tmp_path / "answers.json"
    # a plain recording first (no max_items) so the file exists -> the next run replays.
    run_controlled(answers_path=str(path), **_common(bench, settings))

    calls_after_record = counter["calls"]
    with pytest.raises(SystemExit) as excinfo:
        run_controlled(max_items=25, answers_path=str(path), **_common(bench, settings))
    msg = str(excinfo.value).lower()
    assert "replay" in msg or "record" in msg
    assert counter["calls"] == calls_after_record  # replay refused before any call


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__, "-v"]))
