"""Tests for real LLM token-cost metering.

The synthetic grid uses a *constant* per-tier cost (``CascadeConfig.tier_cost``).
For a real experiment we must measure the actual teacher cost in USD from the
token usage the provider reports. ``MeteredTeacher`` wraps any ``Teacher`` and
accumulates measured cost / token counts; ``PriceTable`` maps a model name to
its per-1k-token prices. These tests prove the recorded cost is *measured* (not
the simulated ``tier_cost`` constant) using a fake teacher with known usage.
"""

from __future__ import annotations

import unittest

from tacet.core.graph import WorldGraph
from tacet.llm.metering import MeteredTeacher, PriceTable, _read_usage
from tacet.llm.teacher import Teacher, TeacherResponse
from tacet.serve.config import TIER_COST


class _FakeTeacher(Teacher):
    """A teacher with a fixed answer and a fixed reported token usage.

    Mirrors the real REST adapters, which expose the provider-reported usage
    on a ``last_usage`` attribute after each ``answer`` call.
    """

    def __init__(self, answers: list[str], prompt_tokens: int, completion_tokens: int) -> None:
        self._answers = answers
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self.last_usage: dict | None = None

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        self.last_usage = {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._prompt_tokens + self._completion_tokens,
        }
        return TeacherResponse(answers=list(self._answers))


class PriceTableTest(unittest.TestCase):
    def test_lookup_returns_per_1k_prices(self) -> None:
        table = PriceTable({"grok-test": (2.0, 10.0)})
        p_in, p_out = table.price("grok-test")
        self.assertEqual(p_in, 2.0)
        self.assertEqual(p_out, 10.0)

    def test_default_table_has_grok_entry(self) -> None:
        table = PriceTable.default()
        p_in, p_out = table.price("grok-4.3")
        self.assertGreater(p_in, 0.0)
        self.assertGreater(p_out, 0.0)

    def test_price_lookup_fails_loud_for_unpriced_model(self) -> None:
        table = PriceTable({"grok-test": (2.0, 10.0)})
        with self.assertRaises(KeyError):
            table.price("unknown-model")


class UsageReaderTest(unittest.TestCase):
    def test_read_usage_handles_none_or_empty(self) -> None:
        self.assertEqual(_read_usage(None), (0, 0))
        self.assertEqual(_read_usage({}), (0, 0))

    def test_read_usage_prefers_openai_style_keys(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        self.assertEqual(_read_usage(usage), (10, 20))

    def test_read_usage_handles_gemini_style_keys(self) -> None:
        usage = {"prompt_token_count": 15, "candidates_token_count": 25}
        self.assertEqual(_read_usage(usage), (15, 25))

    def test_read_usage_includes_reasoning_tokens_in_completion(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "reasoning_tokens": 30}
        self.assertEqual(_read_usage(usage), (10, 50))


class MeteredTeacherTest(unittest.TestCase):
    def test_single_call_cost_is_measured_from_usage(self) -> None:
        # 100 prompt + 20 completion tokens; $2/1k prompt, $10/1k completion.
        # cost = 100/1000 * 2 + 20/1000 * 10 = 0.2 + 0.2 = 0.4 USD.
        fake = _FakeTeacher(["Belgium"], prompt_tokens=100, completion_tokens=20)
        table = PriceTable({"grok-test": (2.0, 10.0)})
        metered = MeteredTeacher(fake, table, model="grok-test")

        graph = WorldGraph()
        resp = metered.answer(graph, "France", "borders")

        self.assertEqual(resp.answers, ["Belgium"])
        self.assertAlmostEqual(metered.total_cost_usd, 0.4)
        self.assertEqual(metered.total_prompt_tokens, 100)
        self.assertEqual(metered.total_completion_tokens, 20)
        self.assertEqual(metered.n_calls, 1)

    def test_accumulates_over_three_calls(self) -> None:
        fake = _FakeTeacher(["Belgium"], prompt_tokens=100, completion_tokens=20)
        table = PriceTable({"grok-test": (2.0, 10.0)})
        metered = MeteredTeacher(fake, table, model="grok-test")

        graph = WorldGraph()
        for _ in range(3):
            metered.answer(graph, "France", "borders")

        # 3 * 0.4 = 1.2 USD measured.
        self.assertAlmostEqual(metered.total_cost_usd, 1.2)
        self.assertEqual(metered.total_prompt_tokens, 300)
        self.assertEqual(metered.total_completion_tokens, 60)
        self.assertEqual(metered.n_calls, 3)

    def test_measured_cost_is_not_the_constant_tier_cost(self) -> None:
        """Prove the cost is *measured*, not the simulated Tier-3 constant."""
        fake = _FakeTeacher(["Belgium"], prompt_tokens=100, completion_tokens=20)
        table = PriceTable({"grok-test": (2.0, 10.0)})
        metered = MeteredTeacher(fake, table, model="grok-test")

        graph = WorldGraph()
        metered.answer(graph, "France", "borders")

        self.assertNotAlmostEqual(metered.total_cost_usd, TIER_COST[3])
        # last_cost_usd of the single call also differs from the constant.
        self.assertNotAlmostEqual(metered.last_cost_usd, TIER_COST[3])

    def test_response_cost_reflects_measured_usd(self) -> None:
        """The returned ``TeacherResponse.cost`` carries the measured USD cost."""
        fake = _FakeTeacher(["Belgium"], prompt_tokens=100, completion_tokens=20)
        table = PriceTable({"grok-test": (2.0, 10.0)})
        metered = MeteredTeacher(fake, table, model="grok-test")

        graph = WorldGraph()
        resp = metered.answer(graph, "France", "borders")
        self.assertAlmostEqual(resp.cost, 0.4)

    def test_reasoning_tokens_are_billed_at_output_rate(self) -> None:
        """A reasoning model bills hidden reasoning tokens at the output rate;
        the meter must add them to the visible completion tokens so it does not
        systematically undercount."""

        class _ReasoningTeacher(Teacher):
            def __init__(self) -> None:
                self.last_usage: dict | None = None

            def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
                # 100 prompt + 20 visible completion + 480 reasoning tokens.
                self.last_usage = {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "reasoning_tokens": 480,
                    "total_tokens": 600,
                }
                return TeacherResponse(answers=["Belgium"])

        table = PriceTable({"grok-test": (2.0, 10.0)})
        metered = MeteredTeacher(_ReasoningTeacher(), table, model="grok-test")
        metered.answer(WorldGraph(), "France", "borders")
        # cost = 100/1000*2 + (20+480)/1000*10 = 0.2 + 5.0 = 5.2 USD.
        self.assertAlmostEqual(metered.total_cost_usd, 5.2)

    def test_provider_billed_ticks_preferred_over_token_estimate(self) -> None:
        """When the provider reports an authoritative billed cost (xAI
        ``cost_in_usd_ticks``, 1 tick = 1e-10 USD) the meter uses it verbatim
        instead of the token-times-price estimate."""

        class _TicksTeacher(Teacher):
            def __init__(self) -> None:
                self.last_usage: dict | None = None

            def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
                self.last_usage = {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cost_in_usd_ticks": 15_000_000,  # = $0.0015
                }
                return TeacherResponse(answers=["Belgium"])

        table = PriceTable({"grok-test": (2.0, 10.0)})
        metered = MeteredTeacher(_TicksTeacher(), table, model="grok-test")
        metered.answer(WorldGraph(), "France", "borders")
        self.assertAlmostEqual(metered.total_cost_usd, 0.0015)

    def test_missing_usage_records_zero_cost_but_counts_call(self) -> None:
        """A wrapped teacher with no usage info must not crash the meter."""

        class _NoUsageTeacher(Teacher):
            def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
                return TeacherResponse(answers=["X"])

        metered = MeteredTeacher(
            _NoUsageTeacher(), PriceTable({"grok-test": (2.0, 10.0)}), model="grok-test"
        )
        graph = WorldGraph()
        metered.answer(graph, "France", "borders")
        self.assertEqual(metered.total_cost_usd, 0.0)
        self.assertEqual(metered.total_prompt_tokens, 0)
        self.assertEqual(metered.n_calls, 1)


if __name__ == "__main__":
    unittest.main()


def test_default_prices_cover_phase1_teachers():
    """Both real teachers used by the v2 experiments must have price entries."""
    from tacet.llm.metering import DEFAULT_PRICES

    assert "grok-4.3" in DEFAULT_PRICES
    assert "gemini-3.5-flash" in DEFAULT_PRICES
    inp, out = DEFAULT_PRICES["gemini-3.5-flash"]
    assert 0 < inp < out
