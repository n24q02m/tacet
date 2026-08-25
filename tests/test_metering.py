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

import pytest

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
        # No declaration present -> preserve the historical behaviour (add).
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "reasoning_tokens": 30}
        self.assertEqual(_read_usage(usage), (10, 50))

    def test_read_usage_adds_reasoning_when_adapter_declares_excluded(self) -> None:
        # Direct xAI (GrokTeacher) declares reasoning is NOT part of the visible
        # completion, so it must be added to reach the billed output count.
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "reasoning_tokens": 5,
            "reasoning_included_in_completion": False,
        }
        self.assertEqual(_read_usage(usage), (100, 25))

    def test_read_usage_does_not_double_count_when_declared_included(self) -> None:
        # OpenRouter (OpenRouterTeacher) declares completion ALREADY includes
        # reasoning, so it must NOT be added again (measured 2026-07-10:
        # deepseek-v4-pro reported completion 60, reasoning 58 -> billed 60).
        usage = {
            "prompt_tokens": 20,
            "completion_tokens": 60,
            "reasoning_tokens": 58,
            "reasoning_included_in_completion": True,
        }
        self.assertEqual(_read_usage(usage), (20, 60))


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

    def test_openrouter_completion_tokens_not_double_counted(self) -> None:
        """On OpenRouter ``completion_tokens`` already includes reasoning tokens;
        the adapter declares that, so the meter must bill the completion count as
        reported rather than adding reasoning on top (which overcounted output by
        up to ~97% on the measured roster)."""

        class _ORTeacher(Teacher):
            def __init__(self) -> None:
                self.last_usage: dict | None = None

            def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
                # deepseek-v4-pro-shaped row: completion already folds in reasoning.
                self.last_usage = {
                    "prompt_tokens": 20,
                    "completion_tokens": 60,
                    "reasoning_tokens": 58,
                    "reasoning_included_in_completion": True,
                }
                return TeacherResponse(answers=["Belgium"])

        table = PriceTable({"or-test": (1.0, 2.0)})
        metered = MeteredTeacher(_ORTeacher(), table, model="or-test")
        metered.answer(WorldGraph(), "France", "borders")
        # billed completion = 60 (NOT 60 + 58 = 118)
        self.assertEqual(metered.total_completion_tokens, 60)
        self.assertEqual(metered.total_prompt_tokens, 20)
        # cost = 20/1000*1 + 60/1000*2 = 0.02 + 0.12 = 0.14 USD.
        self.assertAlmostEqual(metered.total_cost_usd, 0.14)

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

    def test_openrouter_cost_field_preferred_over_ticks_and_tokens(self) -> None:
        """OpenRouter reports the authoritative billed USD directly under
        ``cost``; it must win over both the xAI tick field and the token
        estimate (the three-way precedence, most authoritative first)."""

        class _CostTeacher(Teacher):
            def __init__(self) -> None:
                self.last_usage: dict | None = None

            def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
                # A tick figure and priced tokens are both present, but the
                # OpenRouter ``cost`` (already USD) must take precedence.
                self.last_usage = {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cost_in_usd_ticks": 999_999_999,
                    "cost": 0.42,
                }
                return TeacherResponse(answers=["Belgium"])

        table = PriceTable({"grok-test": (2.0, 10.0)})
        metered = MeteredTeacher(_CostTeacher(), table, model="grok-test")
        metered.answer(WorldGraph(), "France", "borders")
        self.assertAlmostEqual(metered.total_cost_usd, 0.42)

    def test_ticks_preferred_over_token_estimate_when_no_cost(self) -> None:
        """With no OpenRouter ``cost`` but an xAI tick figure, ticks win over the
        token estimate (unchanged xAI behaviour on the middle tier)."""

        class _TicksOnly(Teacher):
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
        metered = MeteredTeacher(_TicksOnly(), table, model="grok-test")
        metered.answer(WorldGraph(), "France", "borders")
        self.assertAlmostEqual(metered.total_cost_usd, 0.0015)

    def test_token_fallback_fails_loud_for_unpriced_model(self) -> None:
        """With neither ``cost`` nor ticks the meter prices via the table and
        must raise ``KeyError`` for an unpriced model -- never silently $0."""
        fake = _FakeTeacher(["Belgium"], prompt_tokens=100, completion_tokens=20)
        metered = MeteredTeacher(fake, PriceTable({"other": (1.0, 1.0)}), model="grok-test")
        with self.assertRaises(KeyError):
            metered.answer(WorldGraph(), "France", "borders")

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


def test_price_key_for_slug_maps_the_e11_ladder():
    """Every E11 ladder OpenRouter slug maps to its bare DEFAULT_PRICES key."""
    from tacet.llm.metering import DEFAULT_PRICES, price_key_for_slug

    ladder = {
        "x-ai/grok-4.3": "grok-4.3",
        "x-ai/grok-4.5": "grok-4.5",
        "anthropic/claude-sonnet-5": "claude-sonnet-5",
        "openai/gpt-5.6-luna": "gpt-5.6-luna",
        "z-ai/glm-5.2": "glm-5.2",
        "google/gemini-3.5-flash": "gemini-3.5-flash",
        "qwen/qwen3.7-max": "qwen3.7-max",
        "minimax/minimax-m3": "minimax-m3",
        "deepseek/deepseek-v4-pro": "deepseek-v4-pro",
        "moonshotai/kimi-k2.6": "kimi-k2.6",
        "xiaomi/mimo-v2.5-pro": "mimo-v2.5-pro",
    }
    for slug, key in ladder.items():
        assert price_key_for_slug(slug) == key
        assert key in DEFAULT_PRICES  # and the mapped key is actually priced


def test_price_key_for_slug_raises_on_unpriced_model():
    """An unknown slug fails loudly rather than defaulting to a $0 price."""
    from tacet.llm.metering import price_key_for_slug

    with pytest.raises(KeyError):
        price_key_for_slug("acme/nonexistent-model")


def test_default_prices_cover_phase1_teachers():
    """Both real teachers used by the v2 experiments must have price entries."""
    from tacet.llm.metering import DEFAULT_PRICES

    assert "grok-4.3" in DEFAULT_PRICES
    assert "gemini-3.5-flash" in DEFAULT_PRICES
    inp, out = DEFAULT_PRICES["gemini-3.5-flash"]
    assert 0 < inp < out


def test_default_prices_include_e11_model_roster():
    """The E11 teacher roster (2026-07-10 OpenRouter live prices, converted to
    per-1k) must all be priced, or a metered run over them raises / mis-meters."""
    from tacet.llm.metering import DEFAULT_PRICES

    expected = {
        "grok-4.5": (0.002, 0.006),
        "claude-sonnet-5": (0.002, 0.010),
        # Re-verified live 2026-08-23 (was $1.00/$6.00 on 2026-07-10): OpenAI
        # cut the price ~80%; the gateway billed the new rate on every call.
        "gpt-5.6-luna": (0.0002, 0.0012),
        "glm-5.2": (0.00054, 0.00176),
        "qwen3.7-max": (0.00125, 0.00375),
        "minimax-m3": (0.0003, 0.0012),
        "deepseek-v4-pro": (0.000435, 0.00087),
        "kimi-k2.6": (0.00066, 0.00341),
        "mimo-v2.5-pro": (0.000435, 0.00087),
    }
    for key, (inp, out) in expected.items():
        assert key in DEFAULT_PRICES, f"missing price entry: {key}"
        got_in, got_out = DEFAULT_PRICES[key]
        assert got_in == inp, (key, got_in, inp)
        assert got_out == out, (key, got_out, out)
        assert 0 < got_in < got_out  # input cheaper than output, both positive
