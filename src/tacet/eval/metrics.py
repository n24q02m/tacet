"""Prometheus instrumentation for the TACET FastAPI service (G2.2).

Main deliverable: two objects for the ``server.py`` handlers to call
after each request:

* ``record_ask(tier, latency_ms, cost_usd)`` — bump the counter /
  histogram for each ``/ask`` query.
* ``mark_llm_rotation(model, ok)`` — gauge for the per-model health of
  the rotating teacher.

When ``prometheus_client`` is not installed, the functions become
no-ops (keeping the package always importable; the ``[service]`` extra
in pyproject adds prometheus_client for production).

The ``/metrics`` endpoint is exposed via ``make_metrics_endpoint`` for
the caller to mount into FastAPI:

    from tacet.eval.metrics import make_metrics_endpoint
    app.get("/metrics")(make_metrics_endpoint())
"""

from __future__ import annotations

try:
    from prometheus_client import (  # type: ignore[import-not-found]
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAS_PROM = True
except ImportError:  # pragma: no cover - optional
    _HAS_PROM = False


if _HAS_PROM:
    # We register on a *named* registry so re-importing the module in
    # tests or under uvicorn's --reload doesn't raise duplicate-time-series.
    _NS = "tacet"
    QUERY_TOTAL = Counter(
        f"{_NS}_query_total",
        "Total number of /ask queries served, labelled by served-by tier.",
        labelnames=("tier",),
    )
    QUERY_LATENCY = Histogram(
        f"{_NS}_query_duration_seconds",
        "Distribution of /ask query latency, labelled by served-by tier.",
        labelnames=("tier",),
        # Wide buckets so p99 covers genuine LLM round-trips (~10 s).
        buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    QUERY_COST = Counter(
        f"{_NS}_query_cost_usd_total",
        "Cumulative blended USD cost of /ask traffic, by served-by tier.",
        labelnames=("tier",),
    )
    DISTILL_TOTAL = Counter(
        f"{_NS}_distill_total",
        "Number of /distill calls that wrote back to the cache.",
        labelnames=("verdict",),  # correct | wrong | unspecified
    )
    LLM_ROTATION_HEALTH = Gauge(
        f"{_NS}_llm_rotation_health",
        "1 if model answered the latest call, 0 if it 429'd / timed out.",
        labelnames=("model",),
    )
    EPISODES_TOTAL = Gauge(
        f"{_NS}_episodes_total",
        "Number of episodes currently in the episodic store (post-flush).",
    )


def record_ask(tier: int, latency_ms: float, cost_usd: float) -> None:
    """Bump counters + histogram for one served ``/ask`` query."""
    if not _HAS_PROM:
        return
    label = str(tier)
    QUERY_TOTAL.labels(tier=label).inc()
    QUERY_LATENCY.labels(tier=label).observe(latency_ms / 1000.0)
    if cost_usd > 0:
        QUERY_COST.labels(tier=label).inc(cost_usd)


def record_distill(verdict: str) -> None:
    """``correct`` | ``wrong`` | ``unspecified`` — track feedback flow."""
    if not _HAS_PROM:
        return
    DISTILL_TOTAL.labels(verdict=verdict).inc()


def mark_llm_rotation(model: str, ok: bool) -> None:
    """Set per-model health gauge from the rotating-teacher path."""
    if not _HAS_PROM:
        return
    LLM_ROTATION_HEALTH.labels(model=model).set(1 if ok else 0)


def set_episodes_total(n: int) -> None:
    if not _HAS_PROM:
        return
    EPISODES_TOTAL.set(n)


def make_metrics_endpoint():
    """Return a FastAPI route handler that emits the Prometheus text format.

    If ``prometheus_client`` is not installed, the endpoint returns 503
    so monitoring tools can detect the absence of instrumentation
    without crashing the app.
    """
    if not _HAS_PROM:

        def _stub():
            from fastapi.responses import PlainTextResponse  # type: ignore[import-not-found]

            return PlainTextResponse(
                "prometheus_client not installed; install with `pip install tacet[service]`",
                status_code=503,
            )

        return _stub

    def _metrics():
        from fastapi.responses import Response  # type: ignore[import-not-found]

        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return _metrics


__all__ = [
    "make_metrics_endpoint",
    "mark_llm_rotation",
    "record_ask",
    "record_distill",
    "set_episodes_total",
]
