# TACET architecture

TACET is an auditable causal-temporal neuro-symbolic reasoning engine. It answers
knowledge-graph questions through a three-tier cascade and distils teacher
knowledge down into the cheap, sound tiers as a workload streams. This document
is a one-page map of the engine; every component below names the module that
implements it.

## The three-tier cascade

The cascade is the flagship router, `TACET`, in `tacet.cascade.router`. Every query
is routed to the cheapest tier that can answer it; a tier that cannot answer
abstains and the query falls through to the next.

- **Tier 1 — symbolic** (`tacet.core.symbolic`). A forward-chaining Datalog rule
  engine (`RuleEngine`, `Rule`) computes the deductive closure over the graph and
  attaches a **machine-checkable proof tree** to every answer (provenance: each
  step reduces to a base fact or a named rule). The tier *abstains* rather than
  guess, which is what keeps its answers sound and auditable.
- **Tier 2 — KGE link prediction** (`tacet.kge`). A ComplEx embedding (`ComplEx`,
  NumPy backend in `kge.py`, optional PyTorch backend in `kge_torch.py`,
  text-attributed seam in `kge_textual.py`) predicts missing edges, calibrated and
  verified against the ontology gate before an answer is accepted. Abstains below
  a confidence threshold.
- **Tier 3 — LLM teacher** (`tacet.llm`). A pluggable teacher answers anything the
  cheap tiers could not. `tacet.llm.teacher` defines the `Teacher` protocol plus
  `OracleTeacher` / `CallableTeacher` / `Narrator`; `tacet.llm.teachers.llm` wraps
  real LLMs (`GeminiTeacher`, `GrokTeacher`, `FallbackChainTeacher`). Tier-3
  answers feed the distillation loop.

Per-tier cost and latency constants live in `tacet.serve.config` (`TIER_COST`,
`TIER_LATENCY_MS`, `CascadeConfig`). An `Answer` carries `(tier, answers, proof,
cost)`.

## Causal identification layer

`tacet.core.causal` implements Pearl-framework structural causal models
(`CausalModel`, `Variable`): observation / intervention (`do`) / counterfactuals
(via abduction-action-prediction), plus identification machinery —
`backdoor_set`, the front-door criterion, and instrumental-variable detection,
with bidirected edges for semi-Markovian models.

## Bi-temporal layer

`tacet.core.temporal` adds `valid_from` / `valid_to` validity intervals to edges
and a `TemporalEngine` for time-sliced queries (`slice_at`, `slice_between`),
`TemporalQuery`, and the full set of Allen interval relations (`allen_relation`,
`temporal_edge`).

## Online rule distillation

`tacet.distill` moves knowledge *down* the cascade so the blended cost drifts
toward the symbolic floor over time (`Distiller`, `mine_rules` in `distill.py`):

- **fact write-back** — a teacher answer becomes a graph edge (exact repeats then
  drop to Tier 1);
- **rule synthesis** — an AMIE-style miner (`tacet.distill.amie`) induces Horn
  rules from teacher answers, so a whole *pattern* — not just one fact — drops to
  the sound Tier 1; this is what lets TACET generalise to unseen entities where a
  cache cannot;
- **KGE augmentation** — teacher facts join Tier-2's training set;
- **concept / ontology formation** — `tacet.distill.concepts`
  (`induce_node_types`, `induce_relations`, `revise_ontology`) and the
  `tacet.distill.fca` (Formal Concept Analysis) baseline.

## Supporting modules

- `tacet.core.graph` — `WorldGraph`, a typed labeled-property graph with loaders.
- `tacet.core.ontology` — typed `Ontology`, axioms, induction, the Tier-2
  verification gate.
- `tacet.core.ingest` — text-to-graph extractors (`RuleBasedExtractor`,
  `CallableExtractor`) and `KGBuilder` with PII redaction.
- `tacet.eval` — auditability metrics (`eval_audit`: proof validity / coverage),
  the synthetic KGQA benchmark (`benchmark`), baseline systems (`baselines`), the
  process-parallel experiment grid (`experiment`), and Prometheus collectors
  (`metrics`, exposed via the `/metrics` endpoint of the service).
- `tacet.data` — dataset loaders: `metaqa`, `proofwriter`, and the curated
  `worldgeo.json`.
- `tacet.serve` — the FastAPI service (`server`), the `tacet` CLI (`cli`),
  env-driven `Settings` (`settings`), and `CascadeConfig` (`config`).

## experimental/ (negative result)

`tacet.experimental` is **not** part of the paper's core contribution. It retains a
graph forward-dynamics forecaster (a Dreamer-style RSSM in
`tacet.experimental.dynamics`) together with the `WorldModel` protocol seam
(`worldmodel`), a STRIPS-style planner (`agent`), episodic memory (`episodic`),
and a multi-writer federation prototype (`federation`). These are documented and
shipped for reproducibility of the reported negative result, not as features.
