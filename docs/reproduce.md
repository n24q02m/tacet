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
python -m tacet.eval.experiment --out experiments/results --seeds 12
python experiments/analyze.py --results experiments/results --out experiments/results
```

This runs the full grid (E1 main comparison, E2 cost trajectory, E3
cost–accuracy frontier, E4 ablations, E5 repetition sensitivity, E6
noisy-teacher robustness) and writes the figures and LaTeX tables under
`experiments/results/`.

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

Outputs land in `experiments/results/` as `tab_proofwriter.tex` and the macro
file `macros.tex`. If `data/` is absent these ProofWriter steps (and the
corresponding tests) skip rather than fail.

## 4. Symbolic auditability on the synthetic workload

```bash
python experiments/run_audit_eval.py --seed 0 --workload-size 300
```

Prints mean proof coverage / validity and writes `experiments/results/tab_audit.tex`
plus `experiments/results/macros.tex`.

## 5. Causal-layer demo

```bash
python experiments/run_causal_demo.py --seed 0 --samples 200000
```

Demonstrates `do`-interventions, back-door / front-door identification, IV
detection, and counterfactual estimation on a structural causal model.

## 6. KGE rotation smoke (Tier 2)

```bash
python experiments/run_rotation_smoke.py --out experiments/results/rotation_smoke.json
```

Writes a small RotatE/ComplEx link-prediction smoke result to the given path.

## Where outputs land

Every runner writes into `experiments/results/`. That directory is tracked, so a
fresh run overwrites the committed sample artifacts in place.

## Building the paper

The LaTeX source for the paper lives in `paper/` (`main.tex`, `references.bib`,
the `\input`-ed tables under `results/`, and the figures under `figures/`).
Build the PDF with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
tectonic paper/main.tex
```

This writes `paper/main.pdf`; build artifacts are git-ignored.
