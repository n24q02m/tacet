"""Tests for LLM teacher wiring, structured-output parsing, and rotation routing."""

from __future__ import annotations

import importlib.util
import unittest
from dataclasses import dataclass

from tacet.llm.teachers import (
    FallbackChainTeacher,
    GeminiTeacher,
    GrokTeacher,
    build_teacher_from_settings,
)
from tacet.llm.teachers.llm import _parse_json_list


def _spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_HAS_GENAI = _spec("google.genai")
_HAS_OPENAI = _spec("openai")


@dataclass
class _SettingsStub:
    teacher: str = "oracle"
    gemini_endpoint: str = "generativelanguage"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    xai_api_key: str | None = None
    xai_model: str = "grok-4.3"
    xai_base_url: str = "https://api.x.ai/v1"
    openrouter_api_key: str | None = None
    openrouter_model: str = "x-ai/grok-4.3"
    rotating_models: list[str] | None = None
    rotating_qps_per_model: float = 9 / 60
    rotating_cooldown_s: float = 60.0


class TestLLMTeachers(unittest.TestCase):
    def test_parse_json_list_handles_fences(self) -> None:
        self.assertEqual(_parse_json_list('["a", "b"]'), ["a", "b"])
        self.assertEqual(_parse_json_list('```json\n["a"]\n```'), ["a"])
        self.assertEqual(_parse_json_list('answer is ["x", "y"] yes'), ["x", "y"])
        self.assertEqual(_parse_json_list("nothing useful"), [])

    def test_parse_json_list_unwraps_object_root(self) -> None:
        self.assertEqual(_parse_json_list('{"answers": ["a", "b"]}'), ["a", "b"])
        self.assertEqual(_parse_json_list('```json\n{"answers": ["x"]}\n```'), ["x"])
        self.assertEqual(_parse_json_list('{"foo": "bar"}'), [])
        self.assertEqual(_parse_json_list('{"a": ["x"], "b": ["y"]}'), [])

    def test_build_teacher_oracle_returns_none(self) -> None:
        self.assertIsNone(build_teacher_from_settings(_SettingsStub()))

    def test_build_teacher_unknown_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_teacher_from_settings(_SettingsStub(teacher="qwen"))

    def test_build_teacher_openrouter_missing_key_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            build_teacher_from_settings(_SettingsStub(teacher="openrouter"))

    def test_build_teacher_openrouter_targets_openrouter(self) -> None:
        if not _HAS_OPENAI:
            self.skipTest("openai not installed")
        from tacet.llm.teachers.llm import OpenRouterTeacher

        s = _SettingsStub(
            teacher="openrouter",
            openrouter_api_key="or-key",
            openrouter_model="deepseek/deepseek-v4-pro",
        )
        teacher = build_teacher_from_settings(s)
        self.assertIsInstance(teacher, OpenRouterTeacher)
        self.assertEqual(teacher._model, "deepseek/deepseek-v4-pro")
        self.assertIn("openrouter.ai", str(teacher._client.base_url))

    def test_build_teacher_grok_still_targets_xai(self) -> None:
        if not _HAS_OPENAI:
            self.skipTest("openai not installed")
        s = _SettingsStub(teacher="grok", xai_api_key="xai-key")
        teacher = build_teacher_from_settings(s)
        self.assertIsInstance(teacher, GrokTeacher)
        self.assertIn("api.x.ai", str(teacher._client.base_url))

    def test_openrouter_response_format_passthrough(self) -> None:
        if not _HAS_OPENAI:
            self.skipTest("openai not installed")
        from types import SimpleNamespace

        from tacet.llm.teachers.llm import OpenRouterTeacher

        class _FakeCompletions:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def create(self, **kwargs):  # noqa: ANN003, ANN201
                self.calls.append(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='["Belgium"]'))],
                    usage=None,
                )

        class _FakeClient:
            def __init__(self, completions: _FakeCompletions) -> None:
                self.chat = SimpleNamespace(completions=completions)

        schema = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "array", "maxItems": 5}},
        }
        with_schema = _FakeCompletions()
        t = OpenRouterTeacher("test-key", response_format=schema)
        t._client = _FakeClient(with_schema)
        t.answer(None, "France", "borders")
        self.assertEqual(with_schema.calls[0]["response_format"], schema)

        no_schema = _FakeCompletions()
        default_teacher = OpenRouterTeacher("test-key")
        default_teacher._client = _FakeClient(no_schema)
        default_teacher.answer(None, "France", "borders")
        self.assertNotIn("response_format", no_schema.calls[0])

    def test_gemini_teacher_missing_sdk_raises_at_construction(self) -> None:
        if _HAS_GENAI:
            self.skipTest("google-genai installed; cannot test missing-SDK path")
        with self.assertRaises(ImportError):
            GeminiTeacher(api_key="fake")

    def test_grok_teacher_missing_sdk_raises_at_construction(self) -> None:
        if _HAS_OPENAI:
            self.skipTest("openai installed; cannot test missing-SDK path")
        with self.assertRaises(ImportError):
            GrokTeacher(api_key="fake")

    def test_fallback_chain_requires_at_least_one(self) -> None:
        with self.assertRaises(ValueError):
            FallbackChainTeacher([])

    def test_gemini_rest_teacher_constructs_without_network(self) -> None:
        if not _spec("httpx"):
            self.skipTest("httpx not installed")
        from tacet.llm.teachers import GeminiRestTeacher

        t = GeminiRestTeacher("fake-key", model="gemini-2.5-flash", qps=None)
        self.assertIn("gemini-2.5-flash", t._url)

    def test_rotating_teacher_cycles_in_priority_order(self) -> None:
        from tacet.llm.teacher import Teacher, TeacherResponse
        from tacet.llm.teachers import RotatingTeacher

        class _Stub(Teacher):
            def __init__(self, name: str, answers: list[str]) -> None:
                self.name, self._answers, self.calls = name, answers, 0

            def answer(self, _g, _h, _r) -> TeacherResponse:
                self.calls += 1
                return TeacherResponse(answers=list(self._answers))

        rot = RotatingTeacher([_Stub("a", ["x"]), _Stub("b", ["y"]), _Stub("c", ["z"])])
        out = [rot.answer(None, "_", "_").answers[0] for _ in range(5)]
        self.assertEqual(out, ["x", "y", "z", "x", "y"])

    def test_rotating_teacher_cools_down_failed_models(self) -> None:
        from unittest import mock

        from tacet.llm.teacher import Teacher, TeacherResponse
        from tacet.llm.teachers import RotatingTeacher

        class _Empty(Teacher):
            def answer(self, _g, _h, _r) -> TeacherResponse:
                return TeacherResponse(answers=[])

        class _Good(Teacher):
            def __init__(self) -> None:
                self.calls = 0

            def answer(self, _g, _h, _r) -> TeacherResponse:
                self.calls += 1
                return TeacherResponse(answers=["ok"])

        empty, good = _Empty(), _Good()
        rot = RotatingTeacher([empty, good], cooldown_s=60.0)
        slept: list[float] = []
        with mock.patch("time.sleep", side_effect=lambda s: slept.append(s)):
            r1 = rot.answer(None, "_", "_")
            self.assertEqual(r1.answers, ["ok"])
            r2 = rot.answer(None, "_", "_")
            self.assertEqual(r2.answers, ["ok"])
        self.assertEqual(good.calls, 2)
        self.assertEqual(slept, [])

    def test_rotating_teacher_waits_once_when_all_models_cooling(self) -> None:
        from unittest import mock

        from tacet.llm.teacher import Teacher, TeacherResponse
        from tacet.llm.teachers import RotatingTeacher

        class _Empty(Teacher):
            def __init__(self) -> None:
                self.calls = 0

            def answer(self, _g, _h, _r) -> TeacherResponse:
                self.calls += 1
                return TeacherResponse(answers=[])

        e1, e2 = _Empty(), _Empty()
        rot = RotatingTeacher([e1, e2], cooldown_s=30.0)
        clock = {"t": 1000.0}
        slept: list[float] = []

        def _sleep(s: float) -> None:
            slept.append(s)
            clock["t"] += s

        with (
            mock.patch("time.time", side_effect=lambda: clock["t"]),
            mock.patch("time.sleep", side_effect=_sleep),
        ):
            resp = rot.answer(None, "_", "_")

        self.assertEqual(resp.answers, [])
        self.assertEqual(len(slept), 1)
        self.assertAlmostEqual(slept[0], 30.1)
        self.assertEqual(e1.calls, 2)
        self.assertEqual(e2.calls, 2)

    def test_rotating_teacher_requires_at_least_one(self) -> None:
        from tacet.llm.teachers import RotatingTeacher

        with self.assertRaises(ValueError):
            RotatingTeacher([])

    def test_build_rotating_from_settings(self) -> None:
        from tacet.llm.teachers import RotatingTeacher, build_teacher_from_settings

        s = _SettingsStub(
            teacher="rotating",
            gemini_api_key="fake-key",
            rotating_models=["gemini-3.5-flash", "gemini-2.5-flash-lite", "gemma-4-31b-it"],
        )
        teacher = build_teacher_from_settings(s)
        self.assertIsInstance(teacher, RotatingTeacher)
        self.assertEqual(len(teacher._teachers), 3)


if __name__ == "__main__":
    unittest.main()
