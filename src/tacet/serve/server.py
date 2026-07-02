"""FastAPI service — TACET as an HTTP endpoint.

    uvicorn tacet.serve.server:app --host 0.0.0.0 --port 8088

Or via the CLI::

    python -m tacet.serve.cli serve --port 8088

Endpoints:

* ``GET  /healthz``               — liveness probe.
* ``GET  /readyz``                — readiness (graph loaded, kge warmed).
* ``GET  /stats``                 — graph + cascade stats.
* ``POST /ask``                   — ``{head, relation}`` → ``Answer``.
* ``POST /distill``               — ``{head, relation, answers, correct?}``
                                    seed an episode + drive feedback.
* ``POST /consolidate``           — trigger a consolidation (rule mining,
                                    KGE warm-start) on demand.
* ``POST /graph/edges``           — bulk-ingest ``{triples: [[h,r,t], ...]}``.

Configuration via ``tacet.serve.settings`` (``TACET_*`` env vars).
FastAPI / Pydantic are optional dependencies; if not installed,
``build_app`` raises a clear ``ImportError``.
"""

from __future__ import annotations

import hmac
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - optional
    _HAS_FASTAPI = False

from tacet.cascade.router import TACET
from tacet.core.graph import WorldGraph
from tacet.core.ontology import Ontology
from tacet.experimental.episodic import EpisodicStore, FeedbackCurator
from tacet.llm.teacher import OracleTeacher, Teacher
from tacet.serve.config import CascadeConfig, KGEConfig
from tacet.serve.settings import Settings, load_settings

if TYPE_CHECKING:  # pragma: no cover
    pass

log = logging.getLogger("tacet.serve.server")


# ---- request / response schemas --------------------------------------------
if _HAS_FASTAPI:

    class AskRequest(BaseModel):
        head: str = Field(..., max_length=255)
        relation: str = Field(..., max_length=255)

    class AskResponse(BaseModel):
        tier: int
        answers: list[str]
        text: str
        confidence: float
        cost: float
        latency_ms: float
        note: str = ""

    class DistillRequest(BaseModel):
        head: str = Field(..., max_length=255)
        relation: str = Field(..., max_length=255)
        answers: list[str] = Field(..., max_length=100)
        correct: bool | None = None

    class GraphIngestRequest(BaseModel):
        triples: list[tuple[str, str, str]] = Field(default_factory=list, max_length=1000)
        nodes: list[tuple[str, str]] = Field(default_factory=list, max_length=1000)  # (id, type)


# ---- service ---------------------------------------------------------------
class TACETService:
    """Thin wrapper that owns the cascade state used by the HTTP layer.

    When ``settings.episodes_path`` is set, the service loads any existing
    JSONL log at start-up and persists episodes back to it on every
    ``/consolidate`` call (and on graceful shutdown via :meth:`flush`).
    """

    def __init__(
        self,
        engine: TACET,
        settings: Settings,
        episodes: EpisodicStore | None = None,
        curator: FeedbackCurator | None = None,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.episodes = episodes or EpisodicStore()
        self.curator = curator or FeedbackCurator()
        self._ready = False
        self._ep_path = settings.episodes_path

    def warmup(self) -> None:
        if self._ep_path and Path(self._ep_path).exists():
            self.episodes.load_jsonl(self._ep_path)
            log.info("loaded %d episodes from %s", len(self.episodes), self._ep_path)
        self.engine.warmup()
        self._ready = True

    def flush(self) -> None:
        """Persist episodes to ``episodes_path`` (no-op when not configured)."""
        if self._ep_path:
            self.episodes.save_jsonl(self._ep_path)

    # ------------------------------------------------------------------ G2.3
    def save_state(self, state_dir: str | Path) -> dict[str, Any]:
        """Persist cascade state to ``state_dir`` for warm-restart later.

        Writes three files into the directory:

        * ``rules.json`` — every Rule in the symbolic engine, including
          axiom rules and any rules synthesised at runtime.
        * ``graph.tsv`` — three-column TSV dump of the current world
          graph (subject, relation, target).
        * ``episodes.jsonl`` — episodic memory (if any), reusing the
          existing JSONL format.

        KGE embeddings are *not* persisted here; the Tier-2 model is
        rebuilt on warm-up by replaying the saved graph through the
        cascade.  This keeps the on-disk format small and avoids the
        framework-version-coupling that pickling Torch tensors implies.
        """
        out = Path(state_dir)
        out.mkdir(parents=True, exist_ok=True)
        # rules — serialise via the engine's JSON dump.
        from tacet.core.symbolic import save_rules_json

        save_rules_json(self.engine.engine.rules, out / "rules.json")
        # graph — three-column TSV is enough to round-trip.
        with (out / "graph.tsv").open("w", encoding="utf-8") as fh:
            for edge in self.engine.graph.edges:
                fh.write(f"{edge.source}\t{edge.relation}\t{edge.target}\n")
        # episodes — reuse the existing JSONL persistence path.
        self.episodes.save_jsonl(out / "episodes.jsonl")
        return {
            "rules": len(self.engine.engine.rules),
            "edges": len(self.engine.graph.edges),
            "episodes": len(self.episodes),
            "dir": str(out),
        }

    def load_state(self, state_dir: str | Path) -> dict[str, Any]:
        """Inverse of :meth:`save_state`.  Replaces rules / graph / episodes.

        After loading, callers should run ``self.engine.warmup()`` so the
        KGE retrains on the restored graph.  The Tier-1 engine inherits
        the rules straight away (no retraining needed there).
        """
        src = Path(state_dir)
        if not src.exists():
            raise FileNotFoundError(f"state dir not found: {src}")
        from tacet.core.symbolic import load_rules_json

        rules_file = src / "rules.json"
        if rules_file.exists():
            self.engine.engine.rules = load_rules_json(rules_file)
        graph_file = src / "graph.tsv"
        if graph_file.exists():
            with graph_file.open(encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) >= 3:
                        self.engine.graph.add_edge(parts[0], parts[1], parts[2])
        ep_file = src / "episodes.jsonl"
        if ep_file.exists():
            self.episodes.load_jsonl(ep_file)
        return {
            "rules": len(self.engine.engine.rules),
            "edges": len(self.engine.graph.edges),
            "episodes": len(self.episodes),
        }

    def ask(self, head: str, relation: str) -> dict[str, Any]:
        ans = self.engine.ask(head, relation)
        ep = self.episodes.record(
            head=head,
            relation=relation,
            tier=ans.tier,
            answers=ans.answers,
            cost=ans.cost,
            latency_ms=ans.latency_ms,
            note=ans.note,
            proof_rules=[ln.split()[1] for ln in ans.proof if ln.startswith("DERIVED")],
        )
        from tacet.eval.metrics import record_ask

        record_ask(ans.tier, ans.latency_ms, ans.cost)
        return {
            "tier": ans.tier,
            "answers": ans.answers,
            "text": ans.text,
            "confidence": ans.confidence,
            "cost": ans.cost,
            "latency_ms": ans.latency_ms,
            "note": ans.note,
            "episode_id": ep.id,
        }

    def distill(
        self, head: str, relation: str, answers: list[str], correct: bool | None
    ) -> dict[str, Any]:
        for ep in self.episodes.for_query(head, relation):
            if correct is True:
                ep.mark_correct()
            elif correct is False:
                ep.mark_wrong()
        # write the asserted facts back into the graph
        for t in answers:
            self.engine.graph.add_edge(head, relation, t, provenance="api")
            self.engine.engine.add_fact((head, relation, t))
        from tacet.eval.metrics import record_distill

        verdict = "correct" if correct is True else ("wrong" if correct is False else "unspecified")
        record_distill(verdict)
        return {"written": len(answers)}

    def consolidate(self) -> dict[str, Any]:
        self.engine.consolidate()
        self.curator.absorb(self.episodes)
        retire = self.curator.rules_to_retire()
        if retire:
            self.engine.engine.rules = [
                r for r in self.engine.engine.rules if r.name not in set(retire)
            ]
            self.engine.engine.materialise(self.engine.graph)
        # persist accumulated episodes after every consolidation
        self.flush()
        return {
            "retired_rules": retire,
            "stats": self.engine.report(),
            "episodes_persisted": bool(self._ep_path),
        }

    def ingest(
        self, triples: list[tuple[str, str, str]], nodes: list[tuple[str, str]]
    ) -> dict[str, Any]:
        for nid, ntype in nodes:
            self.engine.graph.add_node(nid, ntype)
        for h, r, t in triples:
            self.engine.graph.add_edge(h, r, t)
        self.engine.engine.materialise(self.engine.graph)
        return {"stats": self.engine.graph.stats()}

    def stats(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "report": self.engine.report(),
            "graph": self.engine.graph.stats(),
            "episodes": self.episodes.summary(),
        }


# ---- app construction ------------------------------------------------------
def build_service(
    graph: WorldGraph,
    ontology: Ontology,
    teacher: Teacher,
    *,
    settings: Settings | None = None,
    rules: list[Any] | None = None,
) -> TACETService:
    """Wire a configured TACET cascade into an `TACETService`."""
    s = settings or load_settings()
    cfg = CascadeConfig(
        l2_threshold=s.l2_threshold,
        synth_trigger=s.synth_trigger,
        min_confidence=s.min_confidence,
        min_support=s.min_support,
        kge=KGEConfig(dim=s.kge_dim, epochs=s.kge_epochs),
    )
    ak = TACET(graph, ontology, teacher, rules=rules, config=cfg)
    service = TACETService(ak, s)
    service.warmup()
    return service


def build_app(
    graph: WorldGraph,
    ontology: Ontology,
    teacher: Teacher | None = None,
    *,
    settings: Settings | None = None,
    rules: list[Any] | None = None,
) -> FastAPI:
    """Build a FastAPI application around the cascade."""
    if not _HAS_FASTAPI:  # pragma: no cover
        raise ImportError(
            "tacet.serve.server requires FastAPI + Pydantic. "
            "Install with `pip install fastapi uvicorn pydantic`."
        )
    s = settings or load_settings()
    if teacher is None:
        # Without a real teacher we fall back to an oracle that knows nothing —
        # the LLM should be wired in by the caller in production.
        teacher = OracleTeacher(lambda _h, _r: [])
    service = build_service(graph, ontology, teacher, settings=s, rules=rules)
    app = FastAPI(title="TACET", version="mvp")
    if s.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=s.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        """Gate the mutating / cost-incurring endpoints.

        Enforced only when ``TACET_SERVER_API_KEY`` is configured; otherwise a
        no-op so the offline demo and tests run without credentials. When set,
        callers must send a matching ``X-API-Key`` header.
        """
        if s.server_api_key and not (
            x_api_key is not None
            and hmac.compare_digest(x_api_key.encode(), s.server_api_key.encode())
        ):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    # Healthz also re-exports the episode-count Prometheus gauge so the
    # observability layer (Section 11/12 of the paper) stays fresh
    # between consolidation events.
    @app.get("/healthz")
    def healthz():
        from tacet.eval.metrics import set_episodes_total

        set_episodes_total(len(service.episodes._episodes))
        return {"ok": True}

    @app.get("/readyz")
    def readyz():
        return {"ready": service._ready}

    @app.get("/stats")
    def stats():
        return service.stats()

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(_require_api_key)])
    def ask(req: AskRequest):
        try:
            out = service.ask(req.head, req.relation)
        except Exception as e:  # pragma: no cover
            logging.error("Operation failed", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error") from e
        return AskResponse(**{k: v for k, v in out.items() if k != "episode_id"})

    @app.post("/distill", dependencies=[Depends(_require_api_key)])
    def distill(req: DistillRequest):
        return service.distill(req.head, req.relation, req.answers, req.correct)

    @app.post("/consolidate", dependencies=[Depends(_require_api_key)])
    def consolidate():
        return service.consolidate()

    @app.post("/graph/edges", dependencies=[Depends(_require_api_key)])
    def ingest(req: GraphIngestRequest):
        return service.ingest(list(req.triples), list(req.nodes))

    # /metrics — Prometheus scrape endpoint (no-op 503 when
    # prometheus_client is not installed, so dashboards stay aware of
    # the missing instrumentation).
    from tacet.eval.metrics import make_metrics_endpoint

    app.get("/metrics")(make_metrics_endpoint())

    return app


__all__ = ["TACETService", "build_app", "build_service"]


def _demo_app() -> FastAPI:
    """Bootstrap a service over the shipped world-geography KG.

    Used by ``python -m tacet.serve.server`` so the image starts answering
    questions out of the box. In production, build your own app::

        from tacet.serve.server import build_app
        from tacet.llm.teachers import build_teacher_from_settings
        from tacet.serve.settings import load_settings

        settings = load_settings()
        teacher = build_teacher_from_settings(settings)
        app = build_app(my_graph, my_ontology, teacher=teacher, settings=settings)
    """
    from tacet.core.ontology import NodeType, Ontology, RelationType
    from tacet.data import load_worldgeo
    from tacet.llm.teachers import build_teacher_from_settings

    settings = load_settings()
    onto = Ontology()
    for t in ("Country", "City", "Subregion", "Continent", "Language", "Currency"):
        onto.add_node_type(NodeType(t))
    onto.add_relation_type(
        RelationType(
            "located_in",
            frozenset({"Country", "Subregion"}),
            frozenset({"Subregion", "Continent"}),
            transitive=True,
        )
    )
    onto.add_relation_type(
        RelationType("borders", frozenset({"Country"}), frozenset({"Country"}), symmetric=True)
    )
    onto.add_relation_type(
        RelationType("has_capital", frozenset({"Country"}), frozenset({"City"}), functional=True)
    )
    onto.add_relation_type(
        RelationType(
            "official_language", frozenset({"Country"}), frozenset({"Language"}), functional=True
        )
    )
    onto.add_relation_type(
        RelationType(
            "uses_currency", frozenset({"Country"}), frozenset({"Currency"}), functional=True
        )
    )
    teacher = build_teacher_from_settings(settings) or OracleTeacher(lambda _h, _r: [])
    return build_app(load_worldgeo(), onto, teacher=teacher, settings=settings)


if _HAS_FASTAPI:  # pragma: no cover - server runtime
    app = None  # populated lazily; tests import the module without binding the app

    def _main() -> None:
        import uvicorn

        global app
        app = _demo_app()
        s = load_settings()
        uvicorn.run(app, host=s.host, port=s.port, log_level=s.log_level.lower())

    if __name__ == "__main__":
        _main()
