"""GrokTeacher usage capture -- OpenRouter authoritative cost vs xAI ticks.

``OpenRouterTeacher`` reuses ``GrokTeacher.answer`` unchanged, but OpenRouter
reports its billed spend under ``cost`` / ``cost_details.upstream_inference_cost``
while xAI reports ``cost_in_usd_ticks``. Those field families are mutually
exclusive per provider, so ``answer`` must capture both and let ``getattr``
degrade the absent one to ``None`` without raising. No network: a fake client
returns a scripted response object shaped like each provider's usage.
"""

from types import SimpleNamespace

from tacet.llm.teachers.llm import GrokTeacher, OpenRouterTeacher


class _FakeCompletions:
    def __init__(self, response) -> None:  # noqa: ANN001
        self._response = response

    def create(self, **kwargs):  # noqa: ANN003, ANN201
        return self._response


class _FakeClient:
    def __init__(self, response) -> None:  # noqa: ANN001
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


def _response(content: str, usage) -> SimpleNamespace:  # noqa: ANN001
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


def _grok_with_client(response) -> GrokTeacher:  # noqa: ANN001
    t = GrokTeacher("test-key")  # builds the SDK client; the fake below replaces it
    t._client = _FakeClient(response)
    return t


def test_openrouter_cost_and_upstream_cost_captured() -> None:
    # OpenRouter usage object (shape from the usage-accounting docs): ``cost`` and
    # ``cost_details.upstream_inference_cost`` present; NO xAI ``cost_in_usd_ticks``.
    usage = SimpleNamespace(
        prompt_tokens=194,
        completion_tokens=2,
        total_tokens=196,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        prompt_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=100),
        cost=0.95,
        cost_details=SimpleNamespace(upstream_inference_cost=19),
    )
    t = _grok_with_client(_response('["Belgium"]', usage))
    resp = t.answer(None, "France", "borders")

    assert resp.answers == ["Belgium"]
    assert t.last_usage["cost"] == 0.95
    assert t.last_usage["upstream_inference_cost"] == 19
    # xAI-only field is absent on an OpenRouter response -> degrades to None.
    assert t.last_usage["cost_in_usd_ticks"] is None


def test_xai_response_has_no_openrouter_cost_fields() -> None:
    # xAI usage: ``cost_in_usd_ticks`` present but no ``cost`` / ``cost_details``.
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        cost_in_usd_ticks=15_000_000,
    )
    t = _grok_with_client(_response('["Belgium"]', usage))
    t.answer(None, "France", "borders")

    assert t.last_usage["cost_in_usd_ticks"] == 15_000_000
    # OpenRouter-only fields absent on an xAI response -> degrade to None.
    assert t.last_usage["cost"] is None
    assert t.last_usage["upstream_inference_cost"] is None


def test_reasoning_convention_is_declared_per_adapter() -> None:
    # The reasoning-token convention is DECLARED by the adapter (not inferred
    # from the response): direct xAI excludes reasoning from completion_tokens,
    # OpenRouter includes it. Each stamps that onto last_usage so the meter
    # honours it.
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        cost_in_usd_ticks=15_000_000,
    )

    grok = _grok_with_client(_response('["Belgium"]', usage))
    grok.answer(None, "France", "borders")
    assert grok.last_usage["reasoning_included_in_completion"] is False

    router = OpenRouterTeacher("test-key", model="deepseek/deepseek-v4-pro")
    router._client = _FakeClient(_response('["Belgium"]', usage))
    router.answer(None, "France", "borders")
    assert router.last_usage["reasoning_included_in_completion"] is True
