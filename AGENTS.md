# Agent Collaboration

Instructions for AI assistants working on TACET, the reference implementation of
an auditable causal-temporal neuro-symbolic reasoning engine.

## Quick Reference

- Language: Python 3.13 (library supports 3.11+)
- Package: `tacet` (src layout under `src/tacet/`)
- Setup: `mise run setup` or `uv sync --all-extras`
- Test: `uv run pytest`
- Lint: `uv run ruff check .` and `uv run ruff format --check .`
- Auto-fix: `uv run ruff check --fix . && uv run ruff format .`

## Package Layout

TACET routes each query to the cheapest tier of a three-tier cascade and distils
the expensive teacher's knowledge down into the cheap tiers online.

- `core/` — `WorldGraph` (typed labeled-property graph), the forward-chaining
  symbolic rule engine with provenance proof trees, Pearl-framework causal
  identification, bi-temporal reasoning (Allen relations), the typed ontology,
  and text-to-graph ingestion primitives.
- `kge/` — Tier-2 knowledge-graph embeddings: ComplEx link prediction on NumPy
  plus an optional PyTorch backend (`kge_torch.py`) sharing the same API.
- `distill/` — the online distillation loop: fact write-back, AMIE-style Horn
  rule synthesis, KGE augmentation, and FCA / concept-formation baselines.
- `llm/` — Tier-3 teacher abstractions and real-LLM adapters (Gemini, Grok,
  GPT) plus oracle/callable teachers used in offline reproducible runs.
- `cascade/` — the flagship `TACET` router that wires the three tiers together
  and drives consolidation.
- `eval/` — the synthetic KGQA benchmark generator, baseline systems, the
  process-parallel experiment grid, auditability metrics, and the audit harness.
- `serve/` — the FastAPI service, CLI entry point, env-driven settings, and the
  cascade config.
- `data/` — dataset loaders (FB15k-237 / WN18RR layout, MetaQA, ProofWriter) and
  a curated `worldgeo.json`.
- `experimental/` — a graph forward-dynamics world model, federation, and a
  STRIPS-style planner, retained as documented negative-result / exploratory
  code, not core contributions.

## Conventions

- Match the existing style of each file. TACET uses upper-case index names
  (`H`, `T`, `R`, `X`, `A`, `B`) in the KGE and FCA modules to mirror the
  mathematical notation; these are whitelisted in `pyproject.toml` per-file
  ruff ignores. Do not "fix" them.
- Keep changes surgical: every changed line should trace to the task. No
  drive-by refactors.
- New behavior needs a test under `tests/` (mirrors the module path).

## Commit Convention

Only two prefixes are allowed for human commits:

- `feat:` — new features
- `fix:` — bug fixes

`chore(release):` is reserved for the automated python-semantic-release commit.
Do not use `chore`, `docs`, `refactor`, `ci`, `build`, `style`, `test`, or `perf`
prefixes, and never use the `!` breaking-change indicator.
