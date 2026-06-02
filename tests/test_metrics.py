"""Tests cho Prometheus metrics + /metrics endpoint (G2.2)."""

from __future__ import annotations

import unittest

from tacet.eval.metrics import (
    _HAS_PROM,
    make_metrics_endpoint,
    mark_llm_rotation,
    record_ask,
    record_distill,
    set_episodes_total,
)


@unittest.skipUnless(_HAS_PROM, "prometheus_client not installed")
class TestMetricsBackend(unittest.TestCase):
    def test_record_ask_increments_counter(self) -> None:
        from tacet.eval.metrics import QUERY_TOTAL

        before = QUERY_TOTAL.labels(tier="1")._value.get()
        record_ask(1, 12.5, 0.0)
        after = QUERY_TOTAL.labels(tier="1")._value.get()
        self.assertEqual(after - before, 1)

    def test_record_ask_cost_only_nonzero(self) -> None:
        from tacet.eval.metrics import QUERY_COST

        before = QUERY_COST.labels(tier="3")._value.get()
        record_ask(3, 1000.0, 0.0)  # zero cost → counter NOT bumped
        self.assertEqual(QUERY_COST.labels(tier="3")._value.get(), before)
        record_ask(3, 1000.0, 0.01)
        self.assertGreater(QUERY_COST.labels(tier="3")._value.get(), before)

    def test_distill_verdict_labels(self) -> None:
        from tacet.eval.metrics import DISTILL_TOTAL

        before = DISTILL_TOTAL.labels(verdict="correct")._value.get()
        record_distill("correct")
        self.assertEqual(DISTILL_TOTAL.labels(verdict="correct")._value.get(), before + 1)

    def test_llm_rotation_gauge_toggles(self) -> None:
        from tacet.eval.metrics import LLM_ROTATION_HEALTH

        mark_llm_rotation("gemini-2.5-flash", True)
        self.assertEqual(LLM_ROTATION_HEALTH.labels(model="gemini-2.5-flash")._value.get(), 1)
        mark_llm_rotation("gemini-2.5-flash", False)
        self.assertEqual(LLM_ROTATION_HEALTH.labels(model="gemini-2.5-flash")._value.get(), 0)


class TestMetricsAreOptional(unittest.TestCase):
    """Even without prometheus_client, the helpers must be no-ops, not raise."""

    def test_record_ask_noop_safe(self) -> None:
        record_ask(1, 1.0, 0.0)  # never raises
        record_distill("correct")
        mark_llm_rotation("gemma-4-26b-it", True)
        set_episodes_total(5)

    def test_endpoint_factory_returns_callable(self) -> None:
        h = make_metrics_endpoint()
        self.assertTrue(callable(h))


if __name__ == "__main__":
    unittest.main()
