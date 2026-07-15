# Reproducing the paper's numbers

This guide lists the exact commands behind TACET's reported results. All commands
are run from the repository root.

## 1. Environment

```bash
uv sync --all-extras          # installs every optional extra (experiments/service/torch/llm)
# or, with pip:
pip install -e ".[all]"
```

Python 3.11+ is required (3.13 in production). The synthetic-benchmark and
ProofWriter results need only the `experiments` extra (numpy + matplotlib +
scipy); the real-LLM smokes additionally need the `llm` extra and an API key.

## 2. Synthetic KGQA grid (no external data)

The headline cost/accuracy results run entirely on the in-repo synthetic
benchmark generator — no download required:

```bash
python -m tacet.eval.experiment --out experiments/results --seeds 8
python experiments/analyze.py --results experiments/results --out paper
```

The first command runs the full grid (E1 main comparison, E2 cost trajectory,
E3 cost–accuracy frontier, E4 ablations, E5 repetition sensitivity, E6
noisy-teacher robustness) and writes the raw rows + `summary.json` under
`experiments/results/`. The second turns that summary into the paper's figures
(`paper/figures/`) and `\input`-able tables, and *merges* the synthetic-grid
macros into the shared `paper/results/macros.tex` (the file `main.tex` reads).
`--seeds 8` matches the committed artifacts.

## 3. ProofWriter (auditability benchmark) — requires a download

ProofWriter is **not** committed (`data/` is gitignored). Download the AllenAI
V2020.12.3 release and extract it into `data/` so the loader's default root
(`data/proofwriter/proofwriter-dataset-V2020.12.3/`) resolves:

```bash
mkdir -p data/proofwriter
curl -L -o data/proofwriter/proofwriter.zip \
  https://aristo-data-public.s3-us-west-2.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip
cd data/proofwriter && unzip proofwriter.zip && cd -
```

After extraction you should have
`data/proofwriter/proofwriter-dataset-V2020.12.3/CWA/depth-<d>/meta-<split>.jsonl`.
Then run the per-depth evaluation (proof validity / coverage / accuracy):

```bash
python experiments/run_proofwriter.py --split dev --depths 0 1 2 3 5 --limit 300
```

This writes `paper/results/tab_proofwriter.tex` and merges the ProofWriter
macros into `paper/results/macros.tex`. If `data/` is absent these ProofWriter
steps (and the corresponding tests) skip rather than fail.

## 4. Symbolic auditability on the synthetic workload

```bash
python experiments/run_audit_eval.py --seed 0 --workload-size 300
```

Prints mean proof coverage / validity and writes `paper/results/tab_audit.tex`
plus the merged `paper/results/macros.tex`.

## 5. Rule-mining precision (synthetic workload)

```bash
python experiments/run_rule_precision.py --seeds 5 --workload-size 300
```

`--seeds 5` runs seeds `0..4` and reports each count (proposed / installed /
world-correct rules) as mean +/- std across the seeds. This writes
`paper/results/tab_rule_precision.tex` and *merges* the rule-precision macros
(`\ruleProposed`, `\ruleInstalled`, `\ruleWorldCorrect`, `\ruleSeeds`, ...) into
the shared `paper/results/macros.tex`, so it must be run alongside the other
macro-contributing runners to keep `main.tex`'s numbers in sync.

## 6. Causal-layer demo

```bash
python experiments/run_causal_demo.py --seed 0 --samples 200000
```

Demonstrates `do`-interventions, back-door / front-door identification, IV
detection, and counterfactual estimation on a structural causal model.

## 7. KGE rotation smoke (Tier 2)

```bash
python experiments/run_rotation_smoke.py --out experiments/results/rotation_smoke.json
```

Writes a small RotatE/ComplEx link-prediction smoke result to the given path.

## 8. Real-LLM cost at matched accuracy (MetaQA, Section 8.1)

The controlled real-teacher study shares one deterministic teacher answer per
distinct `(head, relation)` across all arms (Tier 2 disabled), so the arms
differ only in routing. It needs MetaQA under `data/MetaQA` (see the README's
"Real MetaQA evaluation" for the clone command).

```bash
# real Grok teacher (metered USD)
export TACET_TEACHER=grok TACET_XAI_API_KEY=<key> TACET_KGE_BACKEND=numpy
uv run python experiments/run_real_kg_controlled.py --hop 1 --seed 0
uv run python experiments/run_real_kg_controlled.py --hop 2 --seed 0

# free (noisy) oracle — $0, no key
export TACET_TEACHER=oracle TACET_ORACLE_ERROR_RATE=0.2
uv run python experiments/run_real_kg_controlled.py --hop 2 --seed 0
```

Writes `experiments/results/real_kg_controlled.json` (per-arm teacher calls,
metered cost, and accuracy). Pass `--answers-path <file>` to record the teacher
answers on the first run and replay them byte-identically (no API calls) on
later runs. **API key:** `TACET_XAI_API_KEY` (xAI) for the real Grok teacher;
none for `TACET_TEACHER=oracle`.

## 9. Oracle noise sweep (E11) — teacher-accuracy vs. rule-advantage

Sweeps `OracleTeacher.error_rate` under the same controlled design ($0, no API
key), reporting the mean rule-over-cache saving with a bootstrap 95% CI and a
pre-registered verdict.

```bash
export TACET_TEACHER=oracle
uv run python experiments/run_oracle_noise_sweep.py --hop 2 --seeds 3 \
    --out experiments/results/oracle_noise_sweep.json
```

The default error-rate grid is `0.0..0.5` by `0.05`; add `--gammas 0.5 0.7 0.95`
for the 2-D noise x gamma sweep. **API key:** none.

## 10. Eleven-teacher threshold-gate sweep (H.3 / Table 9)

Each teacher's 2-hop MetaQA ladder is recorded once through the controlled
runner in OpenRouter mode, then replayed at the two thresholds; the committed
reference artifacts are `experiments/results/real_ladder_hop2.json` and the
rendered `paper/results/tab_threshold_gate.tex`.

```bash
export TACET_TEACHER=openrouter TACET_OPENROUTER_API_KEY=<key>
export TACET_OPENROUTER_MODEL=x-ai/grok-4.3    # set per teacher in the ladder

# record the teacher's answers at gamma=0.50, then replay the same answers at 0.95
uv run python experiments/run_real_kg_controlled.py --hop 2 --seed 0 --gamma 0.50 \
    --answers-path experiments/results/ladder_grok-4.3_hop2_seed0.json
uv run python experiments/run_real_kg_controlled.py --hop 2 --seed 0 --gamma 0.95 \
    --answers-path experiments/results/ladder_grok-4.3_hop2_seed0.json
```

Repeat with `TACET_OPENROUTER_MODEL` set to each of the eleven models. Summarise
the recorded ladder (read-only over the on-disk JSON):

```bash
uv run python experiments/analyze_teacher_ladder.py \
    --real experiments/results/real_ladder_hop2.json --axis both
```

**API key:** `TACET_OPENROUTER_API_KEY` (OpenRouter) for the teacher ladder.

## 11. Structured-output comparison (answer-length control)

Re-record each teacher with its answer list capped by an OpenAI-style
`json_schema` constraint (`--max-items`, OpenRouter only), replay at
`gamma=0.50`, and compare the install set with the uncapped ladder of Section 10.
The committed reference artifact is
`experiments/results/structured_output_comparison_hop2.json`.

```bash
export TACET_TEACHER=openrouter TACET_OPENROUTER_API_KEY=<key>
export TACET_OPENROUTER_MODEL=x-ai/grok-4.3    # set per teacher
uv run python experiments/run_real_kg_controlled.py --hop 2 --seed 0 --gamma 0.50 \
    --max-items 25 \
    --answers-path experiments/results/ladder_capped_grok-4.3_hop2_seed0.json \
    --out experiments/results/real_kg_controlled_capped.json
```

`--max-items` requires `TACET_TEACHER=openrouter` (it is a live structured-output
constraint, rejected on replay and on any non-OpenRouter teacher). **API key:**
`TACET_OPENROUTER_API_KEY`.

## 12. Shadow validation (E12) — label-free rule promotion

Runs the mined rule in shadow (it predicts but does not route) and promotes it
only after `k` distinct agreeing unseen heads, demoting on the first
disagreement — a gold-free promotion test.

```bash
# free oracle (mechanism check)
export TACET_TEACHER=oracle
uv run python experiments/run_shadow_validation.py --slug oracle --seed 0

# replay a recorded teacher ladder (no API calls)
uv run python experiments/run_shadow_validation.py \
    --answers experiments/results/ladder --slug grok-4.3 --seed 0
```

Sweep `k` with `--k 2`, `--k 3` (default), `--k 5`; pass `--out` to write the
JSON report (the committed reference is
`experiments/results/shadow_validation_hop2.json`, with zero promotions across
all cells). **API key:** none for the oracle; the recorded ladder replays
offline.

## 13. Compliance amortisation (PrivaCI-Bench GDPR, C5)

Three arms (`llm_only` / `cache` / `full`) over the same shuffled GDPR case
stream, one metered teacher call per distinct case shared across arms. Needs the
PrivaCI-Bench dataset (default root `../PrivaCI-Bench`) and a teacher key.

```bash
# keys in skret /tacet/prod
MSYS_NO_PATHCONV=1 skret run -e prod --path=/tacet/prod -- \
    uv run python experiments/run_privaci_controlled.py --n 300 --teacher gemini --seed 0
```

Repeat with `--teacher grok` and `--teacher claude`. Summarise across teachers
and seeds into the paper macros:

```bash
uv run python experiments/analyze_privaci.py \
    --results experiments/results --out paper/results
```

**API keys:** the selected teacher's key (`gemini` / `grok` / `claude`) from
skret `/tacet/prod`.

## Where outputs land

Raw experiment data (per-run rows and `summary.json`, plus the `*.json` result
files) is written under `experiments/results/`. The artifacts the paper actually
`\input`s — the `tab_*.tex` tables and the figures — are written under
`paper/results/` and `paper/figures/`, and every runner that contributes paper
macros (`analyze.py`, `run_rule_precision.py`, `run_audit_eval.py`,
`run_proofwriter.py`) *merges* its own macros into the single shared
`paper/results/macros.tex`, so the runners can be re-run in any order without
clobbering each other. All of these directories are tracked, so a fresh run
overwrites the committed sample artifacts in place. (The ProofWriter macros
require the `data/` download from Section 3; without it those macros keep their
committed values.)

## Building the paper

The LaTeX source for the paper lives in `paper/` (`main.tex`, `references.bib`,
the `\input`-ed tables under `results/`, and the figures under `figures/`).
Build the PDF with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
tectonic paper/main.tex
```

This writes `paper/main.pdf`; build artifacts are git-ignored.
