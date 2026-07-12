"""Real LLM token-cost metering — measured USD cost instead of a constant.

The synthetic grid reports a *simulated* per-tier cost (``CascadeConfig.tier_cost``);
that is intentional and stays unchanged. For a **real** experiment we must report
the actual money spent, which depends on the token usage the provider reports per
call. ``MeteredTeacher`` is the opt-in path: wrap any real ``Teacher`` and it

* reads the provider-reported usage (prompt / completion token counts),
* prices it via a ``PriceTable`` (per-1k-token rates for the model),
* accumulates the running total cost and token counts across a run.

The real REST adapters (``GrokTeacher`` / ``GeminiRestTeacher``) expose the raw
provider usage on a ``last_usage`` attribute set after each call; the meter reads
``wrapped.last_usage`` after delegating. Usage shapes handled:

* xAI / OpenAI-style: ``usage = {prompt_tokens, completion_tokens, total_tokens}``.
* Gemini: ``usage_metadata = {prompt_token_count, candidates_token_count, ...}``
  — normalised by the adapter into the ``prompt_tokens`` / ``completion_tokens``
  keys before being stored on ``last_usage``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tacet.core.graph import WorldGraph
from tacet.llm.teacher import Teacher, TeacherResponse

#: Default per-model prices as ``(usd_per_1k_prompt, usd_per_1k_completion)``.
#:
#: Each entry was verified live against the provider on the date in its source
#: comment; the metering is only as current as those figures, and a stale entry
#: silently mis-meters, so re-verify against the provider before quoting a real
#: cost.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    # xAI Grok 4.3: $1.25 / 1M input, $2.50 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/x-ai/grok-4.3 and https://docs.x.ai
    # (xAI standard <=200k-token-context tier), verified 2026-06-03.
    "grok-4.3": (0.00125, 0.0025),
    # Gemini 3.5 Flash: $1.50 / 1M input, $9.00 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/google/gemini-3.5-flash and
    # https://ai.google.dev/gemini-api/docs/pricing, verified 2026-06-12.
    "gemini-3.5-flash": (0.0015, 0.009),
    # Claude Sonnet 4.6: $3 / 1M input, $15 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/anthropic/claude-sonnet-4.6, verified
    # 2026-06-13. Used via OpenRouter BYOK (anthropic/claude-sonnet-4.6); the
    # 5% BYOK surcharge (waived for the first 1M requests/month) is not modelled
    # here, so reported cost is the Anthropic token price.
    "claude-sonnet-4.6": (0.003, 0.015),
    # update to current Gemini pricing
    "gemini-2.5-pro": (0.00125, 0.005),
    "gemini-2.5-flash": (0.0003, 0.0025),
    # xAI Grok 4.5: $2.00 / 1M input, $6.00 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/x-ai/grok-4.5, verified 2026-07-10.
    "grok-4.5": (0.002, 0.006),
    # Claude Sonnet 5: $2.00 / 1M input, $10.00 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/anthropic/claude-sonnet-5, verified 2026-07-10.
    "claude-sonnet-5": (0.002, 0.010),
    # OpenAI GPT-5.6 Luna: $1.00 / 1M input, $6.00 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/openai/gpt-5.6-luna, verified 2026-07-10.
    "gpt-5.6-luna": (0.001, 0.006),
    # Z.ai GLM-5.2: $0.54 / 1M input, $1.76 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/z-ai/glm-5.2, verified 2026-07-10.
    "glm-5.2": (0.00054, 0.00176),
    # Qwen3.7-Max: $1.25 / 1M input, $3.75 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/qwen/qwen3.7-max, verified 2026-07-10.
    "qwen3.7-max": (0.00125, 0.00375),
    # MiniMax M3: $0.30 / 1M input, $1.20 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/minimax/minimax-m3, verified 2026-07-10.
    "minimax-m3": (0.0003, 0.0012),
    # DeepSeek V4 Pro: $0.435 / 1M input, $0.87 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/deepseek/deepseek-v4-pro, verified 2026-07-10.
    "deepseek-v4-pro": (0.000435, 0.00087),
    # Moonshot Kimi K2.6: $0.66 / 1M input, $3.41 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/moonshotai/kimi-k2.6, verified 2026-07-10.
    "kimi-k2.6": (0.00066, 0.00341),
    # Xiaomi MiMo v2.5 Pro: $0.435 / 1M input, $0.87 / 1M output (= per-1k below).
    # Source: https://openrouter.ai/xiaomi/mimo-v2.5-pro, verified 2026-07-10.
    "mimo-v2.5-pro": (0.000435, 0.00087),
}


@dataclass
class PriceTable:
    """Maps a model name to ``(usd_per_1k_prompt, usd_per_1k_completion)``."""

    prices: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def default(cls) -> PriceTable:
        """A table seeded with placeholder prices (see ``DEFAULT_PRICES``)."""
        return cls(dict(DEFAULT_PRICES))

    def price(self, model: str) -> tuple[float, float]:
        """Return ``(usd_per_1k_prompt, usd_per_1k_completion)`` for ``model``.

        Raises ``KeyError`` for an unpriced model so a missing entry fails loud
        rather than silently metering a cost of zero.
        """
        return self.prices[model]

    def cost_usd(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Measured USD cost for a single call's token usage.

        ``completion_tokens`` should be the *billed* output-token count: for a
        reasoning model that means the visible completion tokens PLUS the
        reasoning tokens (both billed at the output rate). The caller
        (``MeteredTeacher``) is responsible for summing those before calling.
        """
        p_in, p_out = self.price(model)
        return prompt_tokens / 1000.0 * p_in + completion_tokens / 1000.0 * p_out


def price_key_for_slug(slug: str, prices: dict[str, tuple[float, float]] | None = None) -> str:
    """Map an OpenRouter model slug to its bare ``DEFAULT_PRICES`` key.

    OpenRouter ids carry a vendor prefix (``anthropic/claude-sonnet-5``), while
    the price table is keyed on the bare model name (``claude-sonnet-5``). The key
    is the id after the last ``/``. Raises ``KeyError`` for a model with no price
    entry, so an unpriced run fails loudly here rather than metering it at $0.
    """
    table = DEFAULT_PRICES if prices is None else prices
    key = slug.rsplit("/", 1)[-1]
    if key not in table:
        raise KeyError(
            f"no price for OpenRouter model {slug!r}: price key {key!r} is absent from "
            f"the price table; add it to DEFAULT_PRICES before metering this model."
        )
    return key


#: xAI returns the authoritative billed cost as an integer "ticks" field;
#: one tick is 1e-10 USD (empirically verified 2026-06-03 against the
#: per-token rate breakdown). When present this is preferred over the
#: token-times-price estimate because it already accounts for reasoning
#: tokens, cached-prompt discounts and any tool surcharges.
USD_PER_TICK = 1e-10


def _read_usage(usage: dict | None) -> tuple[int, int]:
    """Extract ``(prompt_tokens, billed_completion_tokens)`` from a usage dict.

    Tolerates both the xAI/OpenAI keys (``prompt_tokens`` / ``completion_tokens``)
    and the raw Gemini keys (``prompt_token_count`` / ``candidates_token_count``)
    so the meter works whether or not the adapter pre-normalised them. Missing
    usage yields ``(0, 0)``.

    Whether the reasoning tokens are already part of ``completion_tokens`` is a
    per-provider convention, so it is NOT inferred from the response shape: the
    adapter declares it on the usage dict via ``reasoning_included_in_completion``.
    Direct xAI reports a visible ``completion_tokens`` that EXCLUDES the (billed)
    reasoning tokens, so they are added; OpenRouter already folds reasoning into
    ``completion_tokens``, so adding again would double-count. When the flag is
    absent the historical behaviour is preserved (reasoning is added).
    """
    if not usage:
        return 0, 0
    prompt = usage.get("prompt_tokens", usage.get("prompt_token_count", 0)) or 0
    completion = usage.get("completion_tokens", usage.get("candidates_token_count", 0)) or 0
    reasoning = usage.get("reasoning_tokens", 0) or 0
    if usage.get("reasoning_included_in_completion"):
        return int(prompt), int(completion)
    return int(prompt), int(completion) + int(reasoning)


class MeteredTeacher(Teacher):
    """Wrap a ``Teacher`` and record measured USD cost + token usage per call.

    Parameters
    ----------
    wrapped:
        The real teacher to delegate to. After each call its ``last_usage``
        attribute (if present) is read for the provider-reported token counts.
    prices:
        The ``PriceTable`` used to convert token counts into USD.
    model:
        Model name used for the price lookup (must be a key in ``prices``).

    Accumulated state (read after a run)::

        total_cost_usd, total_prompt_tokens, total_completion_tokens, n_calls
        last_cost_usd  # cost of the most recent call
    """

    def __init__(self, wrapped: Teacher, prices: PriceTable, model: str) -> None:
        self.wrapped = wrapped
        self.prices = prices
        self.model = model
        self.total_cost_usd: float = 0.0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.n_calls: int = 0
        self.last_cost_usd: float = 0.0
        self.last_usage: dict | None = None

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        resp = self.wrapped.answer(graph, head, relation)
        usage = getattr(self.wrapped, "last_usage", None)
        prompt_tokens, completion_tokens = _read_usage(usage)

        # Cost precedence, most authoritative first:
        #   1. ``cost``               -- OpenRouter's billed spend, already in USD.
        #   2. ``cost_in_usd_ticks``  -- xAI's billed cost in 1e-10-USD ticks.
        #   3. tokens x PriceTable    -- a token-based estimate.
        cost_usd_field = (usage or {}).get("cost") if usage else None
        ticks = (usage or {}).get("cost_in_usd_ticks") if usage else None
        if cost_usd_field is not None:
            cost = float(cost_usd_field)
        elif ticks is not None:
            cost = float(ticks) * USD_PER_TICK
        else:
            # ``completion_tokens`` here is already the BILLED output count:
            # ``_read_usage`` honours the adapter's
            # ``reasoning_included_in_completion`` declaration, so reasoning is
            # added for direct xAI (excluded) and not for OpenRouter (included).
            cost = self.prices.cost_usd(self.model, prompt_tokens, completion_tokens)

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost_usd += cost
        self.n_calls += 1
        self.last_cost_usd = cost
        # Re-expose the raw provider usage so a caller (e.g. the record/replay
        # layer) can persist it for audit without reaching into ``wrapped``.
        self.last_usage = usage

        # Surface the measured USD cost on the response so a metered run can use
        # it directly; the answers/correctness are untouched.
        return TeacherResponse(answers=resp.answers, cost=cost, correct=resp.correct)


__all__ = [
    "DEFAULT_PRICES",
    "USD_PER_TICK",
    "MeteredTeacher",
    "PriceTable",
    "price_key_for_slug",
]
