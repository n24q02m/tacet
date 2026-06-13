# TACET

**A self-distilling neuro-symbolic cascade that amortises LLM cost across knowledge-graph QA and regulatory-compliance checking, with auditable Datalog proof trees.**

[![CI](https://img.shields.io/github/actions/workflow/status/n24q02m/tacet/ci.yml?branch=main&label=CI)](https://github.com/n24q02m/tacet/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/n24q02m/tacet)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20621240.svg)](https://doi.org/10.5281/zenodo.20621240)

[![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org)
[![semantic-release](https://img.shields.io/badge/semantic--release-python-e10079?logo=semantic-release)](https://github.com/python-semantic-release/python-semantic-release)
[![Renovate](https://img.shields.io/badge/renovate-enabled-brightgreen?logo=renovate)](https://renovatebot.com)

<!-- BEGIN: AUTO-GENERATED-CROSS-PROMO -->
<details>
  <summary><strong>Sister projects from n24q02m</strong> (click to expand)</summary>

| Project | Tagline | Tag |
|---|---|---|
| [better-code-review-graph](https://github.com/n24q02m/better-code-review-graph) | Knowledge graph for token-efficient code reviews -- semantic search and call-... | MCP |
| [better-email-mcp](https://github.com/n24q02m/better-email-mcp) | IMAP/SMTP email for AI agents -- read, send, organize folders, and manage att... | MCP |
| [better-godot-mcp](https://github.com/n24q02m/better-godot-mcp) | Composite MCP server for Godot Engine -- 17 composite tools for AI-assisted g... | MCP |
| [better-notion-mcp](https://github.com/n24q02m/better-notion-mcp) | Markdown-first Notion for AI agents -- pages, databases, blocks, and comments... | MCP |
| [better-telegram-mcp](https://github.com/n24q02m/better-telegram-mcp) | Telegram for AI agents -- messages, chats, media, and contacts across both bo... | MCP |
| [claude-plugins](https://github.com/n24q02m/claude-plugins) | Claude Code plugin marketplace for the n24q02m MCP servers -- install web sea... | Marketplace |
| [imagine-mcp](https://github.com/n24q02m/imagine-mcp) | Image and video understanding + generation for AI agents -- across Gemini, Op... | MCP |
| [jules-task-archiver](https://github.com/n24q02m/jules-task-archiver) | Chrome Extension for bulk operations on Jules tasks via batchexecute API -- a... | Tooling |
| [mcp-core](https://github.com/n24q02m/mcp-core) | Shared foundation for building MCP servers -- Streamable HTTP transport, OAut... | MCP |
| [mnemo-mcp](https://github.com/n24q02m/mnemo-mcp) | Persistent AI memory with hybrid search and embedded sync. Open, free, unlimi... | MCP |
| [qwen3-embed](https://github.com/n24q02m/qwen3-embed) | Lightweight Qwen3 text embedding and reranking via ONNX Runtime and GGUF | Library |
| [skret](https://github.com/n24q02m/skret) | Secrets without the server. | CLI |
| [tacet](https://github.com/n24q02m/tacet) | TACET: a self-distilling neuro-symbolic cascade that amortises LLM cost in kn... | Tooling |
| [web-core](https://github.com/n24q02m/web-core) | Shared web infrastructure package for search, scraping, HTTP security, and st... | Library |
| [wet-mcp](https://github.com/n24q02m/wet-mcp) | Open-source MCP server for AI agents: web search, content extraction, and lib... | MCP |

</details>
<!-- END: AUTO-GENERATED-CROSS-PROMO -->


TACET answers knowledge-graph questions through a **three-tier cascade** —
deterministic rules, a learned graph embedding, and an LLM — and **distils the
LLM's reasoning down into the cheap tiers as it runs**. The blended cost per
query falls toward the symbolic floor while the answers stay explainable and,
on the symbolic tier, provably sound.

**TACET** is the system; the paper is
*"Cost-Amortised Reasoning via Self-Distilling Neuro-Symbolic Cascades:
From Knowledge-Graph QA to Regulatory-Compliance Checking."*  The paper source is in
[`paper/`](paper) (build with `tectonic paper/main.tex`), and the preprint
is published on Zenodo: [doi:10.5281/zenodo.20621240](https://doi.org/10.5281/zenodo.20621240).

> **Status:** the preprint is published on Zenodo and pending arXiv
> endorsement for cs.LG. If you have published in cs.LG and find the work
> sound, an endorsement would be warmly appreciated — please reach out at
> quangminh2402.dev@gmail.com.

```
            query
              │
   ┌──────────▼───────────┐  Tier 1 · SYMBOLIC   sound, explainable, ~$1e-4/query
   │ deductive rule engine │  forward-chaining closure + provenance proofs
   └──────────┬───────────┘  abstains when it cannot prove an answer
              │ abstain
   ┌──────────▼───────────┐  Tier 2 · KGE        learned, calibrated, ~$1e-3/query
   │  ComplEx link pred.   │  predicts missing edges; ontology-verified
   └──────────┬───────────┘  abstains below a confidence threshold
              │ abstain
   ┌──────────▼───────────┐  Tier 3 · LLM TEACHER answers anything, ~$5e-2/query
   │   LLM (or oracle)     │  ── distils ──▶ facts, rules, KGE data ──▶ Tiers 1–2
   └──────────────────────┘
```

## v0.1.0 (first public release)

- **Sound symbolic Tier 1** — forward-chaining Datalog with machine-checkable
  provenance proof trees (`tacet.core.symbolic`); abstains rather than guess.
- **KGE Tier 2** — ComplEx link prediction (NumPy + optional PyTorch backend),
  ontology-verified and confidence-calibrated (`tacet.kge`).
- **LLM teacher Tier 3** — pluggable real-LLM adapters (Gemini, Grok, GPT) and
  oracle/callable teachers (`tacet.llm`).
- **Causal identification** (`tacet.core.causal`) — Pearl-framework do-calculus,
  back-door / front-door criteria, instrumental-variable detection,
  counterfactuals via abduction-action-prediction.
- **Bi-temporal reasoning** (`tacet.core.temporal`) — `valid_from` / `valid_to`
  per edge, time-sliced queries, Allen interval relations.
- **Online rule distillation** (`tacet.distill`) — fact write-back, AMIE-style
  Horn-rule synthesis, KGE augmentation, plus FCA / concept-formation baselines.
- **Auditability evaluation** (`tacet.eval`) — proof-validity / proof-coverage
  metrics, ProofWriter + synthetic-KGQA benchmarks, the experiment grid, and
  Prometheus collectors + `/metrics` endpoint (Grafana dashboard in `extras/`).
- **`tacet.experimental`** — a graph forward-dynamics forecaster retained as a
  documented negative result, not a core contribution.

Detailed change log: [`CHANGELOG.md`](CHANGELOG.md).

## Why it matters

Production KG/RAG assistants send **every** question to an LLM. Yet most of a
real workload is repetitive and structurally regular — exactly what cheap,
deterministic machinery can handle. TACET routes each query to the cheapest
tier that can answer it, and the **distillation loop** moves knowledge *down*
the stack over time:

* **fact write-back** — a teacher answer becomes a graph edge (exact repeats → Tier 1);
* **rule synthesis** — an AMIE-style miner induces Horn rules from teacher
  answers, so a whole *pattern* — not just one fact — drops to the sound Tier 1;
* **KGE augmentation** — teacher facts join the embedding's training set.

Rule synthesis is what makes TACET beat a cache: a synthesised rule answers
**unseen** entities, a cache only answers exact repeats.

## Install

```bash
pip install -e .                       # core (numpy only)
pip install -e ".[experiments]"        # + matplotlib/scipy to reproduce the paper
pip install -e ".[service]"            # + fastapi/uvicorn/pydantic for the HTTP API
pip install -e ".[torch]"              # + PyTorch for the GPU KGE backend
pip install -e ".[llm]"                # + google-genai / openai for real LLM teachers
pip install -e ".[all]"                # everything
```

Requires Python 3.11+ (Python 3.13 in production).

## Run as a service

```bash
# Single-model teacher
export TACET_TEACHER=gemini
export TACET_GEMINI_API_KEY=...

# Or rotate across the free-tier flash family (default order, smartest first):
#   gemini-3.5-flash → gemini-3-flash → gemini-2.5-flash →
#   gemini-3.1-flash-lite → gemini-2.5-flash-lite →
#   gemma-4-31b-it → gemma-4-26b-it
# Each model has its own rpm quota; round-robin stretches throughput ~7×.
export TACET_TEACHER=rotating
export TACET_GEMINI_API_KEY=...
# (override the list)
# export TACET_ROTATING_MODELS="gemini-3.5-flash,gemini-2.5-flash,gemma-4-31b-it"

uvicorn tacet.serve.server:app --host 0.0.0.0 --port 8088
# or
docker build -t tacet .
docker run -p 8088:8088 -e TACET_TEACHER=gemini \
  -e TACET_GEMINI_API_KEY=$GEMINI_API_KEY tacet

curl -X POST localhost:8088/ask \
  -H 'content-type: application/json' \
  -d '{"head":"France","relation":"borders"}'
```

Endpoints: `POST /ask`, `POST /distill`, `POST /consolidate`, `POST /graph/edges`,
`GET /stats`, `GET /healthz`, `GET /readyz`. OpenAPI at `/docs`.

## Install from a registry

TACET plugs into any RAG pipeline as a *cost-saving layer in front of the
existing retriever* (load a project KG → cascade → fall through to the
retriever on Tier 3). Once published:

    pip install tacet
    # or from source:
    pip install 'tacet @ git+https://github.com/n24q02m/tacet'

Real MetaQA evaluation:

```bash
git clone https://github.com/yuyuz/MetaQA.git data/MetaQA
python experiments/run_metaqa.py --metaqa-root data/MetaQA --hop 1 --limit 200
```

## Quickstart

```python
from tacet import TACET, WorldGraph, Ontology
from tacet.llm.teacher import CallableTeacher

kg   = WorldGraph.from_json("mygraph.json")     # or .from_triples([...]) / .from_csv(...)
onto = Ontology.induce(kg)                       # or hand-build a typed ontology
ak   = TACET(kg, onto, teacher=CallableTeacher(my_llm_fn))
ak.warmup()                                      # compile rules, train the KGE tier

ans = ak.ask("France", "located_in")             # -> Answer(tier, answers, proof, cost)
print(ans.tier, ans.answers, ans.cost)

for head, rel in workload:
    ak.ask(head, rel)
ak.consolidate()                                 # absorb what the teacher taught
print(ak.report())
```

`CallableTeacher` wraps any `(head, relation) -> list[str]` function — that is
the single integration point for a real LLM (Gemini, Grok, GPT, …).

## Try it

```bash
python -m tacet.serve.cli demo      # 5 systems on a synthetic benchmark
python examples/real_kg_demo.py    # the cascade on a real geography KG
python examples/quickstart.py      # the API in 20 lines
python -m unittest discover tests  # 39 tests
```

## Reproduce the paper

```bash
python -m tacet.eval.experiment --out experiments/results --seeds 12
python experiments/analyze.py --results experiments/results --out experiments/results
```

This runs the full grid (E1 main comparison, E2 cost trajectory, E3 cost–accuracy
frontier, E4 ablations, E5 repetition sensitivity, E6 noisy-teacher robustness)
and emits the figures and LaTeX tables used by the paper.  See
[`docs/reproduce.md`](docs/reproduce.md) for the full reproduction recipe.

## Results (synthetic KGQA benchmark, 12 seeds)

After running the grid above, the per-seed figures land in `experiments/results/`;
the headline is a **~4–5× reduction in blended cost**
versus an LLM-only system at near-parity accuracy, and a strict cost win over
an exact-match cache because synthesised rules generalise to unseen entities.

## Modules

The package is grouped into subpackages: `core/` (graph + symbolic + causal +
temporal + ontology + ingest primitives), `kge/` (Tier-2 embeddings), `distill/`
(distillation + rule mining), `llm/` (Tier-3 teacher + adapters), `cascade/`
(the flagship `TACET` router), `eval/` (benchmark + baselines + experiment grid +
metrics + audit), `serve/` (FastAPI server + CLI + settings + config), `data/`
(dataset loaders), and `experimental/` (negative-result world-model / federation
/ planner code).

| Module | Role |
|--------|------|
| `core/graph.py` | `WorldGraph` — typed labeled-property graph + loaders + temporal slicing |
| `core/ontology.py` | typed ontology: axioms, validation, the Tier-2 verification gate, induction |
| `core/symbolic.py` | Tier 1 — forward-chaining rule engine, provenance proofs |
| `kge/kge.py` | Tier 2 — ComplEx embedding (NumPy), calibrated link prediction |
| `llm/teacher.py` | Tier 3 — LLM teacher adapters + narrator |
| `distill/distill.py` | distillation: fact write-back, AMIE-style rule mining, KGE augmentation |
| `cascade/router.py` | `TACET` — the cascade, routing and consolidation |
| `eval/benchmark.py` | controlled synthetic KGQA benchmark generator |
| `eval/baselines.py` | baseline systems + the unified run harness |
| `eval/experiment.py` | the experiment grid (process-parallel) |
| `core/temporal.py` *(Tier A)* | bi-temporal edges, time-sliced queries, Allen interval relations |
| `core/ingest.py` *(Tier A)* | text → graph extractors (rule-based + LLM adapter) + `KGBuilder` |
| `data/` *(Tier A)* | KGC dataset loaders (FB15k-237 / WN18RR layout) + curated `worldgeo.json` |
| `core/causal.py` *(Tier B)* | Pearl-framework SCM: observation / intervention / counterfactual + backdoor set |
| `experimental/episodic.py` *(Tier B)* | per-query episode log + feedback-driven rule curation |
| `experimental/agent.py` *(Tier C)* | STRIPS-style `Action` / `Goal` / `Planner` over the closure |
| `distill/concepts.py` *(Tier B)* | unsupervised induction of node types and length-2 relation candidates |
| `experimental/federation.py` *(Tier C)* | multi-writer `FederatedGraph` with provenance + trust + merge |
| `experimental/worldmodel.py` *(Tier C)* | `WorldModel` protocol — the integration seam for Dreamer / JEPA-class WMs |
| `serve/settings.py` *(MVP)* | env-driven `Settings` (`TACET_*`), pydantic-settings + dataclass fallback |
| `llm/teachers/llm.py` *(MVP)* | real LLM teacher adapters — `GeminiTeacher`, `GrokTeacher`, `FallbackChainTeacher` |
| `kge/kge_torch.py` *(MVP)* | GPU-trainable ComplEx / RotatE on PyTorch — same public API as the NumPy backend |
| `serve/server.py` *(MVP)* | FastAPI service: `/ask`, `/distill`, `/consolidate`, `/graph/edges`, `/stats` |
| `data/metaqa.py` *(MVP)* | MetaQA loader (KB + 1/2/3-hop QA splits) for end-to-end NL-KGQA eval |

## Roadmap status

This release ships all of **Tier A** and the tractable parts of
**Tier B + C** discussed in the paper §10:

- ✅ **Temporal extension** — `valid_from` / `valid_to` per edge; `slice_at`,
  `slice_between`, `TemporalEngine`, Allen interval relations.
- ✅ **Auto-ingestion from text** — `RuleBasedExtractor` for reproducible
  offline pipelines + `CallableExtractor` as the LLM hook.
- ✅ **Standard KGC dataset support** — FB15k-237 / WN18RR style loaders,
  `synthetic_kg_dataset`, `worldgeo`. The Tier-2 ComplEx attains
  **MRR≈0.39 / Hits@10≈0.58** on the synthetic-org benchmark
  (`experiments/kge_real_eval.py`).
- ✅ **Causal layer (Tier B)** — `CausalModel` with structural equations,
  `do`-interventions, `counterfactual()` via abduction-action-prediction,
  and `backdoor_set()`. Pearl-framework semantics on discrete DAGs.
- ✅ **Episodic memory + feedback loop (Tier B)** — `EpisodicStore` records
  every (query, route, answer, cost, latency) tuple; `FeedbackCurator`
  retires rules that accumulate negative user feedback.
- ✅ **Concept formation (Tier B)** — `induce_node_types` (k-means++ over
  structural signatures), `induce_relations` (length-2 path mining),
  `revise_ontology` to apply both in place.
- ✅ **Goal-directed agency (Tier C)** — STRIPS-style `Action` / `Goal`,
  best-first `Planner` over the rule-engine closure; goals may reference
  *derived* triples.
- ✅ **Federation (Tier C)** — `FederatedGraph` with per-edge `Provenance`,
  `trust_weighted` / `majority` / `latest_wins` merge strategies, conflict
  surfacing via `disagreements()`.
- ⚠️ **Continuous world model (Tier C)** — the `WorldModel` Protocol +
  `IdentityWorldModel` stand-in *only*; a real Dreamer / JEPA integration
  needs GPU + perception data and is out of scope.
- ⬜ Multi-modal entity embeddings — deferred.
- ⬜ Real KGQA benchmark eval with a frontier LLM teacher — needs API keys.

The full ID algorithm for semi-Markovian causal models, learnt federation
trust inference, and CRDT-grade federation infrastructure remain explicit
out-of-scope items.

## Limitations

* The KGE tier is a CPU NumPy ComplEx; a production deployment would use a
  GPU R-GCN/RotatE on PyTorch Geometric behind the same interface.
* The graph store is in-memory; the symbolic and graph APIs map onto FalkorDB.
* Experiments use a controlled *synthetic* benchmark and model the LLM as an
  (optionally noisy) oracle to isolate the routing/distillation dynamics from
  LLM answer quality — see the paper's *Limitations* section.

## License

MIT.
