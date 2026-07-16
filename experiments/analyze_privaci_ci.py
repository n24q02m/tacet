"""Per-seed spread and bootstrap CIs for the PrivaCI compliance amortisation.

``analyze_privaci.py`` emits the per-teacher amortisation *mean* and std
(``\\privFull<Teacher>`` etc.). This companion adds, per teacher, the three
per-seed full-arm amortisation factors and a percentile bootstrap 95% CI over
those three seeds, so the compliance domain reports the same spread summary as
the oracle-noise sweep (``experiments/run_oracle_noise_sweep.py``).

The per-seed full-arm factor is EXACTLY the quantity ``analyze_privaci`` averages
into ``\\privFull<Teacher>``: the seed's LLM-only stream cost divided by its full
arm stream cost (see ``run_privaci_matrix._amort``). The script re-derives those
factors from the committed matrices, checks their mean reproduces the committed
macro to the last digit -- and aborts rather than emit a number the data does not
support if it does not -- then writes the whole file as ``analyze_privaci``'s base
macros followed by the CI block, so one command regenerates it::

    uv run python experiments/analyze_privaci_ci.py

The bootstrap is the SAME 10,000-replicate percentile method the oracle-noise
sweep uses, mirrored here (constants and function body identical to
``run_oracle_noise_sweep.bootstrap_ci``) so this data-only script needs no
MetaQA/torch import chain; ``tests/test_analyze_privaci_ci.py`` pins the mirror
to the canonical function.
"""

import argparse
import statistics
from pathlib import Path

import numpy as np
from analyze_privaci import TEACHERS, build_macros, load_matrices

# Mirror of experiments/run_oracle_noise_sweep.py: percentile bootstrap of the
# mean at a fixed resample count and rng seed so every reported CI is
# reproducible. Kept in sync by tests/test_analyze_privaci_ci.py, which asserts
# byte-identical output against the canonical function.
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_RNG_SEED = 12345
BOOTSTRAP_CI = 0.95

#: Seed index -> LaTeX-safe token (control words are letters only, so no digits).
SEED_TOKENS = ("Zero", "One", "Two")


def bootstrap_ci(values, rng, n_resamples=BOOTSTRAP_RESAMPLES, ci=BOOTSTRAP_CI):
    """Percentile bootstrap CI of the mean over ``values`` (resampling with
    replacement). Byte-for-byte the estimator in
    ``run_oracle_noise_sweep.bootstrap_ci``; deterministic given ``values`` and
    a freshly seeded ``rng``.
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100.0 * (1.0 - ci) / 2.0))
    hi = float(np.percentile(means, 100.0 * (1.0 + ci) / 2.0))
    return (lo, hi)


def per_seed_full_factors(data):
    """The per-seed full-arm amortisation factors (LLM-only cost / full cost),
    ordered by seed -- exactly ``run_privaci_matrix._amort``'s ``full`` entry.
    """
    factors = []
    for run in sorted(data["runs"], key=lambda r: r["seed"]):
        stream = run["stream"]
        base = stream["llm_only"]["total_cost_usd"]
        full = stream["full"]["total_cost_usd"]
        factors.append(round(base / full, 3))
    return factors


def build_ci_macros(matrices):
    """Return the CI + per-seed macro block appended after the base macros.

    Aborts (``SystemExit``) if a per-seed mean does not reproduce the committed
    ``\\privFull<Teacher>`` (the matrix ``amortisation_mean``), so a drifted or
    hand-edited matrix fails here rather than shipping an unsupported CI.
    """
    lines = ["% -- compliance per-seed factors + bootstrap CIs (analyze_privaci_ci.py) --"]
    for teacher, (token, _model) in TEACHERS.items():
        data = matrices[teacher]
        factors = per_seed_full_factors(data)
        if len(factors) != len(SEED_TOKENS):
            raise SystemExit(f"{teacher}: expected {len(SEED_TOKENS)} seeds, got {len(factors)}")
        mean = round(statistics.fmean(factors), 3)
        committed = data["aggregate"][teacher]["full"]["amortisation_mean"]
        if mean != committed:
            raise SystemExit(
                f"{teacher}: per-seed mean {mean} does not reproduce committed "
                f"amortisation_mean {committed}; refusing to emit CI macros"
            )
        rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
        lo, hi = bootstrap_ci(factors, rng)
        for seed_token, factor in zip(SEED_TOKENS, factors, strict=True):
            lines.append(f"\\newcommand{{\\privFullS{seed_token}{token}}}{{{factor:.3f}}}")
        lines.append(f"\\newcommand{{\\privFullCILo{token}}}{{{lo:.3f}}}")
        lines.append(f"\\newcommand{{\\privFullCIHi{token}}}{{{hi:.3f}}}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("experiments/results"))
    parser.add_argument("--out", type=Path, default=Path("paper/results"))
    args = parser.parse_args()

    matrices = load_matrices(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    body = build_macros(matrices) + build_ci_macros(matrices)
    (args.out / "macros_privaci.tex").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
