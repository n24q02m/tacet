"""Compliance prompt plumbing: custom template + verdict/article parsing."""

import pytest

from tacet.llm.teachers.compliance import (
    COMPLIANCE_PROMPT_TEMPLATE,
    parse_compliance_answer,
)
from tacet.llm.teachers.llm import GeminiRestTeacher, OpenRouterTeacher


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": '["prohibit", "art32", "art6"]'}],
                    }
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 3,
                "totalTokenCount": 8,
            },
        }


class _FakeClient:
    def __init__(self):
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse()


def test_custom_prompt_template_used():
    t = GeminiRestTeacher(
        "k", endpoint="vertex", qps=None, prompt_template=COMPLIANCE_PROMPT_TEMPLATE
    )
    fake = _FakeClient()
    t._client = fake
    resp = t.answer(None, "Acme stored passwords unencrypted.", "verdict")
    sent = fake.calls[0]["json"]["contents"][0]["parts"][0]["text"]
    assert "Acme stored passwords unencrypted." in sent
    assert "GDPR" in sent
    verdict, articles = parse_compliance_answer(resp.answers)
    assert verdict == "prohibit"
    assert articles == ("art32", "art6")


def test_parse_compliance_answer_edge_cases():
    assert parse_compliance_answer([]) == ("abstain", ())
    assert parse_compliance_answer(["permit"]) == ("permit", ())
    assert parse_compliance_answer(["Prohibit", "Article 32", "art6(1)"]) == (
        "prohibit",
        ("art32", "art6"),
    )
    assert parse_compliance_answer(["weird"]) == ("abstain", ())


def test_openrouter_teacher_points_at_openrouter():
    # constructing the teacher pulls in the optional 'openai' SDK (lazy import in
    # GrokTeacher.__init__); skip without the 'llm' extra, matching the rest of
    # the suite's optional-dep convention so a no-extras checkout stays green.
    pytest.importorskip("openai")
    # thin subclass of GrokTeacher: OpenRouter base_url + the model slug, no network
    t = OpenRouterTeacher("k", model="anthropic/claude-sonnet-4.6", prompt_template="{head}")
    assert t._model == "anthropic/claude-sonnet-4.6"
    assert str(t._client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
