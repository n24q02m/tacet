"""Tests for the MVP layer: settings, real-LLM teacher wiring, MetaQA loader,
PyTorch backend availability, and the FastAPI service shape."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from tacet.data.metaqa import (
    _parse_question_line,
    load_metaqa,
)
from tacet.llm.teachers import (
    FallbackChainTeacher,
    GeminiTeacher,
    GrokTeacher,
    build_teacher_from_settings,
)
from tacet.llm.teachers.llm import _parse_json_list
from tacet.serve.settings import Settings, load_settings


def _spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_HAS_FASTAPI = _spec("fastapi")
_HAS_PYDANTIC = _spec("pydantic_settings")
_HAS_TORCH = _spec("torch")
_HAS_GENAI = _spec("google.genai")
_HAS_OPENAI = _spec("openai")


# --- settings -------------------------------------------------------------
_ENV_KEYS = (
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "TACET_GEMINI_API_KEY",
    "TACET_XAI_API_KEY",
    "TACET_TEACHER",
)


class TestSettings(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = {k: os.environ.pop(k, None) for k in _ENV_KEYS}

    def tearDown(self) -> None:
        for k, v in self._previous.items():
            if v is not None:
                os.environ[k] = v

    def test_default_load(self) -> None:
        s = load_settings()
        self.assertEqual(s.teacher, "oracle")
        self.assertEqual(s.kge_backend, "auto")
        self.assertFalse(s.has_real_teacher())

    def test_env_gemini_key_backfills_into_settings(self) -> None:
        os.environ["GEMINI_API_KEY"] = "test-gemini-key"
        s = load_settings()
        self.assertEqual(s.gemini_api_key, "test-gemini-key")
        # Oracle promotes to "grok" when an xAI key is present and otherwise
        # to "rotating" when only a Gemini key is around.  This test
        # exercises the Gemini-only path.
        self.assertEqual(s.teacher, "rotating")
        self.assertTrue(s.has_real_teacher())

    def test_env_xai_key_promotes_to_grok(self) -> None:
        os.environ["XAI_API_KEY"] = "test-xai-key"
        s = load_settings()
        self.assertEqual(s.xai_api_key, "test-xai-key")
        # xAI key present → Grok is the primary teacher per the
        # oracle→grok promotion rule.
        self.assertEqual(s.teacher, "grok")
        self.assertTrue(s.has_real_teacher())

    def test_openrouter_env_configures_first_class_teacher(self) -> None:
        keys = {
            "TACET_TEACHER": "openrouter",
            "TACET_OPENROUTER_API_KEY": "or-secret",
            "TACET_OPENROUTER_MODEL": "deepseek/deepseek-v4-pro",
        }
        for k, v in keys.items():
            os.environ[k] = v
        try:
            s = load_settings()
            self.assertEqual(s.teacher, "openrouter")
            self.assertEqual(s.openrouter_api_key, "or-secret")
            self.assertEqual(s.openrouter_model, "deepseek/deepseek-v4-pro")
            self.assertTrue(s.has_real_teacher())
        finally:
            for k in keys:
                os.environ.pop(k, None)

    def test_openrouter_model_defaults_to_published_anchor(self) -> None:
        os.environ.pop("TACET_OPENROUTER_MODEL", None)
        s = load_settings()
        self.assertEqual(s.openrouter_model, "x-ai/grok-4.3")
        # no key -> not a usable teacher unless one is configured
        self.assertIsNone(s.openrouter_api_key)

    def test_env_override(self) -> None:
        keys = {
            "TACET_TEACHER": "gemini",
            "TACET_GEMINI_API_KEY": "test-key",
            "TACET_KGE_DIM": "32",
            "TACET_PORT": "9000",
        }
        for k, v in keys.items():
            os.environ[k] = v
        try:
            s = load_settings()
            self.assertEqual(s.teacher, "gemini")
            self.assertEqual(s.gemini_api_key, "test-key")
            self.assertEqual(s.kge_dim, 32)
            self.assertEqual(s.port, 9000)
            self.assertTrue(s.has_real_teacher())
        finally:
            for k in keys:
                os.environ.pop(k, None)


# --- LLM teacher wiring ---------------------------------------------------
class TestLLMTeachers(unittest.TestCase):
    def test_parse_json_list_handles_fences(self) -> None:
        self.assertEqual(_parse_json_list('["a", "b"]'), ["a", "b"])
        self.assertEqual(_parse_json_list('```json\n["a"]\n```'), ["a"])
        self.assertEqual(_parse_json_list('answer is ["x", "y"] yes'), ["x", "y"])
        self.assertEqual(_parse_json_list("nothing useful"), [])

    def test_build_teacher_oracle_returns_none(self) -> None:
        s = Settings()  # default teacher="oracle"
        self.assertIsNone(build_teacher_from_settings(s))

    def test_build_teacher_unknown_raises(self) -> None:
        s = Settings()
        s.teacher = "qwen"
        with self.assertRaises(ValueError):
            build_teacher_from_settings(s)

    def test_build_teacher_openrouter_missing_key_raises(self) -> None:
        s = Settings()
        s.teacher = "openrouter"
        s.openrouter_api_key = None
        with self.assertRaises(RuntimeError):
            build_teacher_from_settings(s)

    def test_build_teacher_openrouter_targets_openrouter(self) -> None:
        if not _HAS_OPENAI:
            self.skipTest("openai not installed")
        from tacet.llm.teachers.llm import OpenRouterTeacher

        s = Settings()
        s.teacher = "openrouter"
        s.openrouter_api_key = "or-key"
        s.openrouter_model = "deepseek/deepseek-v4-pro"
        teacher = build_teacher_from_settings(s)
        self.assertIsInstance(teacher, OpenRouterTeacher)
        self.assertEqual(teacher._model, "deepseek/deepseek-v4-pro")
        # base_url points at OpenRouter, not xAI (do not call it).
        self.assertIn("openrouter.ai", str(teacher._client.base_url))

    def test_build_teacher_grok_still_targets_xai(self) -> None:
        if not _HAS_OPENAI:
            self.skipTest("openai not installed")
        s = Settings()
        s.teacher = "grok"
        s.xai_api_key = "xai-key"
        teacher = build_teacher_from_settings(s)
        self.assertIsInstance(teacher, GrokTeacher)
        self.assertIn("api.x.ai", str(teacher._client.base_url))

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

        # Constructing must not perform any network I/O.
        t = GeminiRestTeacher("fake-key", model="gemini-2.5-flash", qps=None)
        # Endpoint URL is computed from the model name.
        self.assertIn("gemini-2.5-flash", t._url)

    def test_rotating_teacher_cycles_in_priority_order(self) -> None:
        """Round-robin: each call picks the next non-cooldown teacher."""
        from tacet.llm.teacher import Teacher, TeacherResponse
        from tacet.llm.teachers import RotatingTeacher

        class _Stub(Teacher):
            def __init__(self, name: str, answers: list[str]) -> None:
                self.name, self._answers, self.calls = name, answers, 0

            def answer(self, _g, _h, _r) -> TeacherResponse:
                self.calls += 1
                return TeacherResponse(answers=list(self._answers))

        a = _Stub("a", ["x"])
        b = _Stub("b", ["y"])
        c = _Stub("c", ["z"])
        rot = RotatingTeacher([a, b, c])
        out = [rot.answer(None, "_", "_").answers[0] for _ in range(5)]
        # cycles through smartest-first: a, b, c, a, b
        self.assertEqual(out, ["x", "y", "z", "x", "y"])

    def test_rotating_teacher_cools_down_failed_models(self) -> None:
        """A teacher that returns empty answers is parked on cooldown -- and a
        single empty response must NOT stall a whole cooldown while a healthy
        model is available.

        Regression guard: the router indexed a cursor it reassigned mid-loop, so
        after the empty model it re-hit the same (now-cooling) model instead of
        the healthy one, then slept the full cooldown. ``time.sleep`` is patched
        to a no-op recorder, so this runs instantly on the buggy code too and
        fails on the recorded sleep rather than by taking a real minute.
        """
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
            # First call: empty fires (returns nothing), router falls STRAIGHT to
            # good in the same call -- no cooldown wait.
            r1 = rot.answer(None, "_", "_")
            self.assertEqual(r1.answers, ["ok"])
            # Second call: empty is on cooldown, router goes straight to good.
            r2 = rot.answer(None, "_", "_")
            self.assertEqual(r2.answers, ["ok"])
        self.assertEqual(good.calls, 2)
        # A healthy model was available on every call, so the router NEVER slept.
        # The unfixed code slept a full cooldown on the first call.
        self.assertEqual(slept, [])

    def test_rotating_teacher_waits_once_when_all_models_cooling(self) -> None:
        """When EVERY model is cooling the router waits out the soonest cooldown
        exactly once, retries the round, and then gives up.

        Uses a fake clock the patched ``sleep`` advances, so no real time passes.
        """
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
            clock["t"] += s  # advance the virtual clock so the cooldown expires

        with (
            mock.patch("time.time", side_effect=lambda: clock["t"]),
            mock.patch("time.sleep", side_effect=_sleep),
        ):
            resp = rot.answer(None, "_", "_")

        # gave up after one wait -> empty result
        self.assertEqual(resp.answers, [])
        # waited exactly once, for the soonest cooldown (+0.1 slack)
        self.assertEqual(len(slept), 1)
        self.assertAlmostEqual(slept[0], 30.1)
        # each model tried once per stage: stage 0 sets the cooldown, stage 1
        # retries after the single wait.
        self.assertEqual(e1.calls, 2)
        self.assertEqual(e2.calls, 2)

    def test_rotating_teacher_requires_at_least_one(self) -> None:
        from tacet.llm.teachers import RotatingTeacher

        with self.assertRaises(ValueError):
            RotatingTeacher([])

    def test_build_rotating_from_settings(self) -> None:
        from tacet.llm.teachers import RotatingTeacher, build_teacher_from_settings

        s = Settings()
        s.teacher = "rotating"
        s.gemini_api_key = "fake-key"
        s.rotating_models = ["gemini-3.5-flash", "gemini-2.5-flash-lite", "gemma-4-31b-it"]
        teacher = build_teacher_from_settings(s)
        self.assertIsInstance(teacher, RotatingTeacher)
        self.assertEqual(len(teacher._teachers), 3)


# --- MetaQA loader --------------------------------------------------------
class TestMetaQALoader(unittest.TestCase):
    def test_parse_question_line(self) -> None:
        q = _parse_question_line("Who starred in [Inception]?\tLeonardo DiCaprio|Tom Hardy", 1)
        assert q is not None
        self.assertEqual(q.head, "Inception")
        self.assertEqual(q.answers, ["Leonardo DiCaprio", "Tom Hardy"])
        self.assertEqual(q.hop, 1)

    def test_parse_handles_missing_topic(self) -> None:
        self.assertIsNone(_parse_question_line("no topic\tanswer", 1))

    def test_parse_handles_blank(self) -> None:
        self.assertIsNone(_parse_question_line("   ", 1))

    def test_loader_raises_on_missing_root(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_metaqa("/nonexistent/path", hop=1)

    def test_loader_reads_a_synthetic_layout(self) -> None:
        # Build a tiny MetaQA-shaped tree to exercise the loader.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "kb.txt").write_text(
                "Inception|directed_by|Christopher Nolan\n"
                "Inception|starred_actors|Leonardo DiCaprio\n",
                encoding="utf-8",
            )
            hop_dir = root / "1-hop"
            hop_dir.mkdir()
            (hop_dir / "qa_test.txt").write_text(
                "who directed [Inception]?\tChristopher Nolan\n", encoding="utf-8"
            )
            bench = load_metaqa(root, hop=1, split="test")
            self.assertEqual(bench.stats()["kg_triples"], 2)
            self.assertEqual(len(bench.questions), 1)
            self.assertEqual(bench.questions[0].answers, ["Christopher Nolan"])


# --- PyTorch backend availability ------------------------------------------
class TestTorchBackend(unittest.TestCase):
    def test_torch_backend_import_signals_correctly(self) -> None:
        from tacet.kge.kge_torch import _HAS_TORCH, TorchComplEx, TorchKGEConfig

        if _HAS_TORCH:
            m = TorchComplEx(TorchKGEConfig(dim=8, epochs=2))
            m.fit([("a", "rel", "b"), ("b", "rel", "c"), ("c", "rel", "d")])
            self.assertIn("a", m.ent)
        else:
            with self.assertRaises(ImportError):
                TorchComplEx(TorchKGEConfig(dim=8, epochs=2))

    def test_build_kge_from_settings_falls_back_to_numpy(self) -> None:
        from tacet.kge.kge_torch import build_kge_from_settings

        s = Settings()
        s.kge_backend = "numpy"
        model = build_kge_from_settings(s)
        # numpy backend exposes the same API
        self.assertTrue(hasattr(model, "fit"))
        self.assertTrue(hasattr(model, "predict_tail"))


# --- FastAPI service shape -------------------------------------------------
class TestServerShape(unittest.TestCase):
    def test_build_app_requires_fastapi(self) -> None:
        if not (_HAS_FASTAPI and _HAS_PYDANTIC):
            self.skipTest("fastapi + pydantic-settings not installed")
        from tacet.core.ontology import Ontology
        from tacet.data import load_worldgeo
        from tacet.llm.teacher import OracleTeacher
        from tacet.serve.server import build_app

        app = build_app(
            load_worldgeo(),
            Ontology.induce(load_worldgeo()),
            teacher=OracleTeacher(lambda _h, _r: []),
        )
        # OpenAPI schema is available and lists our endpoints.
        spec = app.openapi()
        paths = set(spec["paths"].keys())
        self.assertIn("/ask", paths)
        self.assertIn("/distill", paths)
        self.assertIn("/consolidate", paths)
        self.assertIn("/healthz", paths)


class TestServerEndpoints(unittest.TestCase):
    """End-to-end HTTP integration tests against `build_app(...)` via TestClient."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (_HAS_FASTAPI and _HAS_PYDANTIC):
            cls.skip = True
            return
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            cls.skip = True
            return
        from tacet.core.ontology import NodeType, Ontology, RelationType
        from tacet.data import load_worldgeo
        from tacet.llm.teacher import OracleTeacher
        from tacet.serve.server import build_app

        onto = Ontology()
        for t in ("Country", "City", "Subregion", "Continent", "Language", "Currency"):
            onto.add_node_type(NodeType(t))
        onto.add_relation_type(
            RelationType("borders", frozenset({"Country"}), frozenset({"Country"}), symmetric=True)
        )
        onto.add_relation_type(
            RelationType(
                "has_capital", frozenset({"Country"}), frozenset({"City"}), functional=True
            )
        )
        onto.add_relation_type(
            RelationType(
                "located_in",
                frozenset({"Country", "Subregion"}),
                frozenset({"Subregion", "Continent"}),
                transitive=True,
            )
        )
        cls.app = build_app(
            load_worldgeo(),
            onto,
            teacher=OracleTeacher(lambda _h, _r: ["UNKNOWN"]),
        )
        cls.client = TestClient(cls.app)
        cls.skip = False

    def setUp(self) -> None:
        if getattr(self, "skip", False):
            self.skipTest("fastapi not installed")

    def test_healthz_returns_ok(self) -> None:
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": True})

    def test_readyz_after_warmup(self) -> None:
        resp = self.client.get("/readyz")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ready"])

    def test_ask_returns_tier1_answer_for_symmetric_borders(self) -> None:
        # Belgium borders France/Germany/Netherlands — symmetric closure
        # is in the shipped ontology, so Tier-1 derives it.
        resp = self.client.post("/ask", json={"head": "Belgium", "relation": "borders"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["tier"], 1)
        self.assertIn("France", body["answers"])

    def test_ask_falls_through_to_teacher_on_unknown(self) -> None:
        resp = self.client.post("/ask", json={"head": "Atlantis", "relation": "borders"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["tier"], 3)

    def test_graph_edges_ingest_and_query(self) -> None:
        resp = self.client.post(
            "/graph/edges",
            json={
                "nodes": [["Liechtenstein", "Country"]],
                "triples": [["Liechtenstein", "borders", "Austria"]],
            },
        )
        self.assertEqual(resp.status_code, 200)
        # symmetric closure means Austria now borders Liechtenstein
        ask = self.client.post("/ask", json={"head": "Austria", "relation": "borders"})
        self.assertIn("Liechtenstein", ask.json()["answers"])

    def test_stats_returns_report(self) -> None:
        resp = self.client.get("/stats")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("graph", body)
        self.assertIn("episodes", body)


class TestServerAuthGate(unittest.TestCase):
    """Opt-in API-key gate on mutating endpoints (TACET_SERVER_API_KEY set)."""

    KEY = "test-secret-key"

    @classmethod
    def setUpClass(cls) -> None:
        if not (_HAS_FASTAPI and _HAS_PYDANTIC):
            cls.skip = True
            return
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            cls.skip = True
            return
        from tacet.core.ontology import NodeType, Ontology, RelationType
        from tacet.data import load_worldgeo
        from tacet.llm.teacher import OracleTeacher
        from tacet.serve.server import build_app
        from tacet.serve.settings import Settings

        onto = Ontology()
        for t in ("Country", "City", "Subregion", "Continent", "Language", "Currency"):
            onto.add_node_type(NodeType(t))
        onto.add_relation_type(
            RelationType("borders", frozenset({"Country"}), frozenset({"Country"}), symmetric=True)
        )
        cls.app = build_app(
            load_worldgeo(),
            onto,
            teacher=OracleTeacher(lambda _h, _r: []),
            settings=Settings(server_api_key=cls.KEY),
        )
        cls.client = TestClient(cls.app)
        cls.skip = False

    def setUp(self) -> None:
        if getattr(self, "skip", False):
            self.skipTest("fastapi not installed")

    def test_mutating_endpoint_rejects_missing_key(self) -> None:
        resp = self.client.post("/ask", json={"head": "Belgium", "relation": "borders"})
        self.assertEqual(resp.status_code, 401)

    def test_mutating_endpoint_rejects_wrong_key(self) -> None:
        resp = self.client.post(
            "/ask",
            headers={"X-API-Key": "wrong-key"},
            json={"head": "Belgium", "relation": "borders"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_mutating_endpoint_accepts_correct_key(self) -> None:
        resp = self.client.post(
            "/ask",
            headers={"X-API-Key": self.KEY},
            json={"head": "Belgium", "relation": "borders"},
        )
        self.assertEqual(resp.status_code, 200)

    def test_read_only_endpoint_is_ungated(self) -> None:
        # liveness / readiness must never require the key
        self.assertEqual(self.client.get("/readyz").status_code, 200)


class TestSettingsEnvParsing(unittest.TestCase):
    """CSV parsing of list env vars on the pydantic-settings path.

    Regression guard for the NoDecode + field_validator fix: pydantic-settings
    JSON-decodes ``list`` env values by default, which raises on a comma-
    separated string. These assert the validator splits CSV instead.
    """

    def setUp(self) -> None:
        if not _HAS_PYDANTIC:
            self.skipTest("pydantic-settings not installed")

    @staticmethod
    def _with_env(name: str, value: str):
        import os

        prev = os.environ.get(name)
        os.environ[name] = value

        def restore() -> None:
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev

        return restore

    def test_cors_origins_parses_comma_separated_env(self) -> None:
        from tacet.serve.settings import load_settings

        # trailing comma + surrounding whitespace must be stripped / dropped
        restore = self._with_env("TACET_CORS_ORIGINS", "https://a.com, https://b.com ,")
        self.addCleanup(restore)
        self.assertEqual(load_settings().cors_origins, ["https://a.com", "https://b.com"])

    def test_rotating_models_parses_comma_separated_env(self) -> None:
        from tacet.serve.settings import load_settings

        restore = self._with_env("TACET_ROTATING_MODELS", "m1,m2,m3")
        self.addCleanup(restore)
        self.assertEqual(load_settings().rotating_models, ["m1", "m2", "m3"])


if __name__ == "__main__":
    unittest.main()
