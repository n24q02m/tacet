"""Regression tests for E17 paid-recording recovery."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import run_e17_admission as e17  # noqa: E402

#: The stream-shaping provenance of one E17 cell; a resume must match it exactly.
PROVENANCE = {
    "dataset": "PrivaCI-Bench-GDPR",
    "n": 300,
    "seed": 1,
    "slug": "deepseek/deepseek-v4-flash-0731",
    "reasoning_effort": "max",
    "price_key": "deepseek-v4-flash-0731",
    "min_support": 5,
    "min_confidence": 0.9,
}
PAID_HEAD = {"verdict": "permit", "articles": [], "cost_usd": 0.012345}
OTHER_SLUG = {**PROVENANCE, "slug": "openai/gpt-5.6-luna"}


def _partial_record() -> dict:
    return {
        "schema": "tacet.e17.admission/v1",
        **PROVENANCE,
        "truncated_by_budget": True,
        "spend_usd": 0.012345,
        "heads": {"case-a": PAID_HEAD},
    }


class _StubTeacher:
    """MeteredTeacher stand-in journalling every paid call in issue order."""

    def __init__(self, journal: list[str]) -> None:
        self.journal = journal
        self.last_cost_usd = 0.01
        self.last_usage = {"cost": 0.01}

    def answer(self, _graph, head: str, _relation: str):
        self.journal.append(f"call:{head}")
        return SimpleNamespace(answers=["permit"])


def test_load_resume_artifact_restores_paid_heads_only_for_matching_provenance(tmp_path):
    """A resume keeps paid heads, but cannot silently mix a different cell."""
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(_partial_record()), encoding="utf-8")

    heads, spent = e17.load_resume_artifact(path, **PROVENANCE)

    assert heads == {"case-a": PAID_HEAD}
    assert spent == pytest.approx(0.012345)

    with pytest.raises(ValueError, match="slug"):
        e17.load_resume_artifact(path, **OTHER_SLUG)


def test_record_answers_never_rebuys_warm_loaded_heads():
    """Resume calls the teacher only for cases absent from the paid artifact."""
    calls: list[str] = []
    bench = SimpleNamespace(case_content={"case-a": "already paid", "case-b": "buy me"})
    guard = e17.BudgetGuard(0.10)
    heads, truncated = e17.record_answers(
        bench,
        ["case-a", "case-b"],
        _StubTeacher(calls),
        guard,
        heads={"case-a": dict(PAID_HEAD)},
    )

    assert calls == ["call:buy me"]
    assert heads["case-a"] == PAID_HEAD
    assert heads["case-b"]["verdict"] == "permit"
    assert guard.spent_usd == pytest.approx(0.01)
    assert truncated is False


def test_record_answers_checkpoints_each_paid_head_before_the_next_call():
    """The durable write-ahead callback runs between consecutive API calls."""
    order: list[str] = []
    bench = SimpleNamespace(case_content={"case-a": "first", "case-b": "second"})
    e17.record_answers(
        bench,
        ["case-a", "case-b"],
        _StubTeacher(order),
        e17.BudgetGuard(0.10),
        checkpoint=lambda case_id, _head: order.append(f"checkpoint:{case_id}"),
    )

    assert order == [
        "call:first",
        "checkpoint:case-a",
        "call:second",
        "checkpoint:case-b",
    ]


def test_e17_partial_log_warm_loads_durable_heads_and_validates_provenance(tmp_path):
    """A killed recording resumes its exact durable prefix, never a different cell."""
    path = tmp_path / "cell.json"
    log, rows = e17.resume_or_start_partial_log(path, PROVENANCE)
    assert rows == []

    logged_row = {
        "case_id": "case-b",
        "head": {"verdict": "permit", "articles": [], "cost_usd": 0.01},
    }
    log.append_row(logged_row)

    _, rows = e17.resume_or_start_partial_log(path, PROVENANCE)
    assert rows == [logged_row]

    with pytest.raises(ValueError, match="slug"):
        e17.resume_or_start_partial_log(path, OTHER_SLUG)


def test_incomplete_recording_cannot_emit_a_locked_endpoint_decision():
    """A budget-truncated recording is resumable evidence, not a result cell."""
    result = e17.incomplete_admission(heads=142, expected_heads=300)

    assert result["decision"] == "INCOMPLETE_RECORDING"
    assert result["endpoint"]["delta"] is None
    assert "142/300" in result["decision_why"]
