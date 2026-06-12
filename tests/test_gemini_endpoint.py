"""Tests for GeminiRestTeacher endpoint selection (vertex express vs generativelanguage)."""

from tacet.llm.teachers.llm import GeminiRestTeacher


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": '["x"]'}]}}],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 2,
                "totalTokenCount": 12,
            },
        }


class _FakeClient:
    def __init__(self):
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse()


def test_default_endpoint_is_generativelanguage():
    t = GeminiRestTeacher("k", model="gemini-3.5-flash", qps=None)
    assert t._url == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
    )


def test_vertex_endpoint_selected():
    t = GeminiRestTeacher("k", model="gemini-3.5-flash", endpoint="vertex", qps=None)
    assert t._url == (
        "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.5-flash:generateContent"
    )


def test_vertex_call_uses_header_auth_and_role():
    t = GeminiRestTeacher("sekrit", model="gemini-3.5-flash", endpoint="vertex", qps=None)
    fake = _FakeClient()
    t._client = fake
    resp = t.answer(None, "head", "relation")
    call = fake.calls[0]
    assert call["headers"]["x-goog-api-key"] == "sekrit"
    assert call["json"]["contents"][0]["role"] == "user"
    assert resp.answers == ["x"]
    assert t.last_usage["prompt_tokens"] == 10
