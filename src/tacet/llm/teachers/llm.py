"""Real LLM teachers — Gemini, Grok, and a fallback chain.

Drop-in replacements for ``OracleTeacher`` that hit a frontier model. Both
adapters import their SDKs lazily so the framework still works without them
installed (an ``ImportError`` is raised only when you instantiate one).

Conventions:

* Gemini via ``google-genai`` SDK.
* Grok via ``openai`` SDK pointed at ``https://api.x.ai/v1``.
* ``FallbackChainTeacher`` retries the next teacher on transient failure
  (rate limit, network) as a multi-provider fallback chain.

Each teacher returns its parsed answer as a ``TeacherResponse`` so the rest
of the cascade is identical to the oracle path. The prompt asks the model
to return JSON; parsing is permissive so a model that wraps in markdown
code fences still works.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable

from tacet.core.graph import WorldGraph
from tacet.llm.teacher import Teacher, TeacherResponse

log = logging.getLogger("tacet.llm.teachers.llm")

_PROMPT_TEMPLATE = """You are answering a structured knowledge-graph question.

Given the head entity and the relation, return the list of tail entities
that the head bears that relation to. Return only a JSON list of strings,
no commentary.

Head: {head}
Relation: {relation}

Example:
  Head: "France", Relation: "borders"
  Answer: ["Belgium", "Germany", "Spain", "Italy", "Switzerland"]

Now answer for:
  Head: {head}, Relation: {relation}
Answer:"""


def _parse_json_list(raw: str) -> list[str]:
    """Best-effort: strip markdown fences, then parse a JSON list of strings."""
    txt = raw.strip()
    fence = re.match(r"```(?:json)?\s*(.*?)\s*```", txt, flags=re.S)
    if fence:
        txt = fence.group(1).strip()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        # last resort: regex-pull any JSON array out of the text
        m = re.search(r"\[[^\[\]]*\]", txt, flags=re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data]


# ---------------------------------------------------------------------------
class GeminiTeacher(Teacher):
    """Gemini via the official ``google-genai`` SDK.

    Parameters
    ----------
    api_key:
        Gemini API key.
    model:
        Model identifier (e.g. ``"gemini-2.5-pro"``).
    cost:
        Reported cost per query; the cascade uses it for the cost trajectory.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro", cost: float = 0.05) -> None:
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - optional
            raise ImportError(
                "GeminiTeacher requires 'google-genai'. Install with `pip install google-genai`."
            ) from e
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._cost = cost

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        prompt = _PROMPT_TEMPLATE.format(head=head, relation=relation)
        try:
            resp = self._client.models.generate_content(model=self._model, contents=prompt)
            text = resp.text or ""
        except Exception as e:  # pragma: no cover - network
            log.warning("gemini call failed: %s", e)
            return TeacherResponse(answers=[], cost=self._cost, correct=False)
        return TeacherResponse(answers=_parse_json_list(text), cost=self._cost)


class GeminiRestTeacher(Teacher):
    """Gemini via the REST API directly (no SDK).  Light dependency footprint
    — just ``httpx``.  Useful when ``google-genai`` cannot be installed
    cleanly (e.g.\\ broken system ``cryptography``).
    """

    _ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    _VERTEX_ENDPOINT = (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent"
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        cost: float = 0.005,
        timeout: float = 30.0,
        max_retries: int = 5,
        qps: float | None = 9 / 60,
        endpoint: str = "generativelanguage",
        prompt_template: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        max_retries:
            Retry budget on 429 / 5xx with exponential backoff.
        qps:
            Soft client-side rate limit. ``9/60`` ≈ free-tier safe (≤ 9 req/min).
            Pass ``None`` to disable.
        endpoint:
            ``"generativelanguage"`` (default) or ``"vertex"`` for the Vertex AI
            express-mode endpoint — required when the API key is restricted to
            ``aiplatform.googleapis.com`` and generativelanguage returns 403.

        ``httpx`` is imported lazily on the first ``answer`` call, so the teacher
        can be constructed (and the rotation wired up) without the optional
        ``service`` extra installed; the ``ImportError`` is raised only when a
        call actually needs the HTTP client.
        """
        if endpoint not in ("generativelanguage", "vertex"):
            raise ValueError(f"unknown endpoint {endpoint!r}")
        #: Domain prompt; must contain ``{head}`` (and optionally ``{relation}``).
        self._prompt_template = prompt_template or _PROMPT_TEMPLATE
        self._timeout = timeout
        self._client = None  # built lazily on first answer() (needs httpx)
        template = self._VERTEX_ENDPOINT if endpoint == "vertex" else self._ENDPOINT
        self._url = template.format(model=model)
        self._api_key = api_key
        self._cost = cost
        self._max_retries = max_retries
        self._qps = qps
        self._last_call: float = 0.0
        #: Provider-reported token usage from the most recent call, normalised
        #: to the xAI/OpenAI keys (``prompt_tokens`` / ``completion_tokens`` /
        #: ``total_tokens``). ``None`` until the first successful call. Read by
        #: ``tacet.llm.metering.MeteredTeacher`` for real cost metering.
        self.last_usage: dict | None = None

    def _get_client(self):  # noqa: ANN202 - returns httpx.Client, imported lazily
        if self._client is None:
            try:
                import httpx  # type: ignore[import-not-found]
            except ImportError as e:  # pragma: no cover - optional
                raise ImportError(
                    "GeminiRestTeacher requires 'httpx'. Install with `pip install httpx`."
                ) from e
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def _throttle(self) -> None:
        if self._qps is None:
            return
        import time

        wait = (1.0 / self._qps) - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        import time

        prompt = self._prompt_template.format(head=head, relation=relation)
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        client = self._get_client()
        text = ""
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                # Pass the key in a header, not the query string, so it never
                # lands in the request URL (and thus not in httpx exception
                # messages / logs / redirect Referer headers).
                r = client.post(self._url, headers={"x-goog-api-key": self._api_key}, json=body)
                if r.status_code == 429 or r.status_code >= 500:
                    delay = min(2**attempt, 30)
                    log.info(
                        "gemini-rest %d, retry %d/%d in %ds",
                        r.status_code,
                        attempt + 1,
                        self._max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                r.raise_for_status()
                data = r.json()
                text = (
                    data["candidates"][0]["content"]["parts"][0]["text"]
                    if data.get("candidates")
                    else ""
                )
                # The REST API reports usage under camelCase ``usageMetadata``
                # (``promptTokenCount`` / ``candidatesTokenCount``); the
                # snake_case spelling only exists on google-genai SDK objects,
                # so accept both and normalise to the xAI/OpenAI key names so
                # the meter is provider-agnostic.
                um = data.get("usageMetadata") or data.get("usage_metadata") or {}
                if um:
                    self.last_usage = {
                        "prompt_tokens": um.get(
                            "promptTokenCount", um.get("prompt_token_count", 0)
                        ),
                        "completion_tokens": um.get(
                            "candidatesTokenCount", um.get("candidates_token_count", 0)
                        ),
                        "total_tokens": um.get("totalTokenCount", um.get("total_token_count", 0)),
                    }
                break
            except Exception as e:  # pragma: no cover - network
                if attempt == self._max_retries:
                    log.warning("gemini-rest exhausted retries: %s", e)
                    return TeacherResponse(answers=[], cost=self._cost, correct=False)
                time.sleep(min(2**attempt, 30))
        return TeacherResponse(answers=_parse_json_list(text), cost=self._cost)


class GrokTeacher(Teacher):
    """Grok via the ``openai`` SDK pointed at xAI's endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "grok-4.3",
        base_url: str = "https://api.x.ai/v1",
        cost: float = 0.05,
        prompt_template: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - optional
            raise ImportError(
                "GrokTeacher requires 'openai'. Install with `pip install openai`."
            ) from e
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._cost = cost
        #: Domain prompt; must contain ``{head}`` (and optionally ``{relation}``).
        self._prompt_template = prompt_template or _PROMPT_TEMPLATE
        #: xAI/OpenAI-style token usage from the most recent call
        #: (``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``).
        #: ``None`` until the first successful call. Read by
        #: ``tacet.llm.metering.MeteredTeacher`` for real cost metering.
        self.last_usage: dict | None = None

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        prompt = self._prompt_template.format(head=head, relation=relation)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            # xAI returns OpenAI-style ``usage`` with prompt/completion/total.
            # For *reasoning* Grok models (grok-4.x) the visible
            # ``completion_tokens`` EXCLUDES the (billed) reasoning tokens, which
            # appear under ``completion_tokens_details.reasoning_tokens``; a
            # portion of the prompt may be served from cache
            # (``prompt_tokens_details.cached_tokens``) at a cheaper rate. xAI
            # also returns the authoritative billed cost as
            # ``cost_in_usd_ticks`` (1 tick = 1e-10 USD). OpenRouter (which
            # reuses this method via ``OpenRouterTeacher``) instead reports its
            # authoritative billed spend in USD directly under ``cost`` (and the
            # upstream provider's charge under
            # ``cost_details.upstream_inference_cost``). The xAI and OpenRouter
            # cost fields are mutually exclusive per provider, so each is read
            # defensively and simply degrades to ``None`` on the provider that
            # does not emit it. All of these are surfaced so ``MeteredTeacher``
            # can meter the real money spent rather than undercount on the
            # visible completion tokens alone.
            usage = getattr(resp, "usage", None)
            if usage is not None:
                comp_details = getattr(usage, "completion_tokens_details", None)
                prompt_details = getattr(usage, "prompt_tokens_details", None)
                cost_details = getattr(usage, "cost_details", None)
                reasoning_tokens = (
                    getattr(comp_details, "reasoning_tokens", 0) if comp_details else 0
                ) or 0
                cached_tokens = (
                    getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
                ) or 0
                self.last_usage = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                    "reasoning_tokens": reasoning_tokens,
                    "cached_tokens": cached_tokens,
                    "cost_in_usd_ticks": getattr(usage, "cost_in_usd_ticks", None),
                    "cost": getattr(usage, "cost", None),
                    "upstream_inference_cost": (
                        getattr(cost_details, "upstream_inference_cost", None)
                        if cost_details
                        else None
                    ),
                }
        except Exception as e:  # pragma: no cover - network
            log.warning("grok call failed: %s", e)
            return TeacherResponse(answers=[], cost=self._cost, correct=False)
        return TeacherResponse(answers=_parse_json_list(text), cost=self._cost)


class OpenRouterTeacher(GrokTeacher):
    """OpenAI-compatible teacher via OpenRouter (https://openrouter.ai/api/v1).

    OpenRouter is OpenAI-compatible, so this reuses ``GrokTeacher.answer``
    unchanged (the xAI-only usage fields it probes are simply absent here, and
    ``getattr`` degrades them to ``None``/0 so ``MeteredTeacher`` falls back to
    the token-times-price estimate). The model slug is the OpenRouter id, e.g.
    ``anthropic/claude-sonnet-4.6``; with the matching provider key attached on
    the OpenRouter dashboard, BYOK routing spends that provider's own credit
    first and only then OpenRouter's shared capacity.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4.6",
        prompt_template: str | None = None,
    ) -> None:
        super().__init__(
            api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1",
            prompt_template=prompt_template,
        )


class RotatingTeacher(Teacher):
    """Cycle through a priority-ordered list of teachers, one per query.

    Designed for free-tier LLM rotations: each provider/model has its own
    rpm quota, so rotating spreads load and stretches the aggregate
    quota.  Models are listed *smartest first*; on a query the router
    picks the next non-cooldown model in cyclic order starting from the
    head, so the smartest still-available model is always preferred when
    several are due to fire.

    If a model returns an empty / failed response (the typical
    rate-limit signal for our REST teachers), it goes on cooldown for
    ``cooldown_s`` seconds and the next model is tried.  When every
    model is cooling down, the router waits out the shortest cooldown
    and retries once before giving up.
    """

    def __init__(self, teachers: list[Teacher], cooldown_s: float = 60.0) -> None:
        if not teachers:
            raise ValueError("RotatingTeacher needs at least one teacher")
        self._teachers = list(teachers)
        self._cooldown_s = cooldown_s
        self._cooldowns: dict[int, float] = {}
        self._cursor = 0

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        import time

        n = len(self._teachers)
        for stage in range(2):
            for offset in range(n):
                idx = (self._cursor + offset) % n
                deadline = self._cooldowns.get(idx, 0.0)
                if time.time() < deadline:
                    continue
                resp = self._teachers[idx].answer(graph, head, relation)
                if resp.answers:
                    self._cursor = (idx + 1) % n
                    return resp
                # empty answer = rate-limit / error from a REST teacher
                self._cooldowns[idx] = time.time() + self._cooldown_s
                self._cursor = (idx + 1) % n
            if stage == 0 and self._cooldowns:
                # everyone cooled down — wait for the soonest to expire, once.
                wake = min(self._cooldowns.values()) - time.time()
                if 0 < wake < self._cooldown_s * 2:
                    time.sleep(wake + 0.1)
                    continue
            break
        return TeacherResponse(answers=[], cost=0.0, correct=False)


class FallbackChainTeacher(Teacher):
    """Try each teacher in turn; if one returns empty/raises, fall through.

    A multi-provider fallback chain so a transient Gemini rate-limit
    doesn't stall the cascade.
    """

    def __init__(self, teachers: Iterable[Teacher]) -> None:
        self._teachers = list(teachers)
        if not self._teachers:
            raise ValueError("FallbackChainTeacher needs at least one teacher")

    def answer(self, graph: WorldGraph, head: str, relation: str) -> TeacherResponse:
        last: TeacherResponse | None = None
        for t in self._teachers:
            resp = t.answer(graph, head, relation)
            last = resp
            if resp.answers:
                return resp
        return last or TeacherResponse(answers=[], cost=0.0, correct=False)


#: Free-tier-rotation model list, **smartest first**.  Gemini full Flash
#: variants are placed ahead of Lite variants within each generation;
#: newer generations are placed ahead of older; Gemma open-weights models
#: trail Gemini Flash for instructed structured-QA work.  Override via
#: ``settings.rotating_models``.
DEFAULT_ROTATING_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2-flash-lite",
    "gemma-4-31b-it",
    "gemma-4-26b-it",
)


def build_teacher_from_settings(settings) -> Teacher | None:  # noqa: ANN001
    """Build the configured teacher; returns None for ``teacher=oracle`` so
    the caller can wire in its own ``OracleTeacher`` from a benchmark."""
    name = settings.teacher
    gemini_endpoint = getattr(settings, "gemini_endpoint", "generativelanguage")
    if name == "oracle":
        return None
    if name == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("TACET_GEMINI_API_KEY not set")
        return GeminiRestTeacher(
            settings.gemini_api_key, settings.gemini_model, endpoint=gemini_endpoint
        )
    if name == "grok":
        if not settings.xai_api_key:
            raise RuntimeError("TACET_XAI_API_KEY not set")
        return GrokTeacher(settings.xai_api_key, settings.xai_model, settings.xai_base_url)
    if name == "rotating":
        if not settings.gemini_api_key:
            raise RuntimeError("TACET_GEMINI_API_KEY not set (required for teacher=rotating)")
        models = getattr(settings, "rotating_models", None) or list(DEFAULT_ROTATING_MODELS)
        chain = [
            GeminiRestTeacher(
                settings.gemini_api_key,
                model=m,
                qps=getattr(settings, "rotating_qps_per_model", 9 / 60),
                endpoint=gemini_endpoint,
            )
            for m in models
        ]
        return RotatingTeacher(chain, cooldown_s=getattr(settings, "rotating_cooldown_s", 60.0))
    if name == "fallback":
        chain = []
        if settings.gemini_api_key:
            chain.append(
                GeminiRestTeacher(
                    settings.gemini_api_key, settings.gemini_model, endpoint=gemini_endpoint
                )
            )
        if settings.xai_api_key:
            chain.append(
                GrokTeacher(settings.xai_api_key, settings.xai_model, settings.xai_base_url)
            )
        if not chain:
            raise RuntimeError("teacher=fallback but no API key configured")
        return FallbackChainTeacher(chain)
    raise ValueError(f"unknown teacher: {name!r}")


__all__ = [
    "DEFAULT_ROTATING_MODELS",
    "FallbackChainTeacher",
    "GeminiRestTeacher",
    "GeminiTeacher",
    "GrokTeacher",
    "RotatingTeacher",
    "build_teacher_from_settings",
]
