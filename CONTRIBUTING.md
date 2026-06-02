# Contributing to TACET

Thanks for your interest in improving TACET. This is a public research reference
implementation; contributions that improve correctness, reproducibility,
documentation, or coverage are very welcome.

## Development Setup

TACET uses [mise](https://mise.jdx.dev/) to pin the Python toolchain and
[uv](https://docs.astral.sh/uv/) for dependency management.

1. Install [mise](https://mise.jdx.dev/) (it provisions Python 3.13 and uv).
2. Run `mise run setup` — this installs the toolchain, syncs all extras, and
   installs the pre-commit hooks (including the `commit-msg` enforcer).

If you prefer not to use mise:

```bash
uv sync --all-extras
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

## Workflow

1. Create a feature branch off `main`.
2. Make your changes.
3. Auto-fix style: `mise run fix` (or `uv run ruff check --fix . && uv run ruff format .`).
4. Run the checks locally:
   - `mise run lint` — `uv run ruff check .` + `uv run ruff format --check .`
   - `mise run test` — `uv run pytest`
5. Commit (pre-commit hooks run automatically).
6. Open a pull request against `main`.

> The ProofWriter test cases skip automatically when the benchmark data under
> `data/` is absent (it is never committed), so a clean checkout still runs a
> green suite.

## Commit Convention

Only two prefixes are allowed:

- `feat:` — new features
- `fix:` — bug fixes

A `commit-msg` hook enforces this. The single exception is `chore(release):`,
which is reserved for the automated release tooling. Conventional Commit
messages drive the changelog, so write them carefully and keep commits atomic
(one logical change each).

## Pull Requests

- One PR per feature or fix.
- Tests are required for new behavior; keep the suite green.
- All CI checks (lint, test, dependency review, CodeQL) must pass.
- PRs are squash-merged, so the PR title must also follow the `feat:` / `fix:`
  convention.

## Reporting Issues

Use the bug report or feature request templates under
[`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/). For security issues, see
[`SECURITY.md`](SECURITY.md) — do not open a public issue.
