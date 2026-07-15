"""Test the PrivaCI compliance bootstrap-CI macro generator.

Pins three things: (1) the script emits the per-seed and CI macros on top of the
base ``analyze_privaci`` macros; (2) each teacher's three per-seed full-arm
factors reproduce the committed ``\\privFull<Teacher>`` mean to the last digit
(the correctness check the script itself enforces); and (3) the mirrored
bootstrap estimator is byte-for-byte the canonical one in
``run_oracle_noise_sweep`` -- so the compliance CI is the same 10,000-replicate
percentile method as the oracle sweep, not a different estimator.
"""

import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from analyze_privaci import load_matrices  # noqa: E402
from analyze_privaci_ci import (  # noqa: E402
    BOOTSTRAP_RNG_SEED,
    bootstrap_ci,
    per_seed_full_factors,
)

RESULTS = REPO / "experiments" / "results"


def test_ci_script_emits_per_seed_and_ci_macros(tmp_path):
    out = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "experiments" / "analyze_privaci_ci.py"),
            "--results",
            str(RESULTS),
            "--out",
            str(out),
        ],
        check=True,
    )
    macros = (out / "macros_privaci.tex").read_text(encoding="utf-8")
    # base macros are preserved (the CI block is appended, not a replacement)
    assert "privFullGemini" in macros
    for token in ("Gemini", "Claude", "Grok"):
        for seed in ("SZero", "SOne", "STwo"):
            assert f"privFull{seed}{token}" in macros, f"missing per-seed macro {seed}{token}"
        assert f"privFullCILo{token}" in macros, f"missing CI-lo macro for {token}"
        assert f"privFullCIHi{token}" in macros, f"missing CI-hi macro for {token}"


def test_per_seed_means_reproduce_committed_macro():
    """Each teacher's per-seed full-arm factors must average to the committed
    ``amortisation_mean`` (the number rounded into ``\\privFull<Teacher>``)."""
    matrices = load_matrices(RESULTS)
    for teacher in ("gemini", "claude", "grok"):
        data = matrices[teacher]
        factors = per_seed_full_factors(data)
        assert len(factors) == 3
        committed = data["aggregate"][teacher]["full"]["amortisation_mean"]
        assert round(statistics.fmean(factors), 3) == committed, (
            f"{teacher}: per-seed mean {statistics.fmean(factors)} != committed {committed}"
        )


def test_bootstrap_mirror_matches_canonical():
    """The mirrored bootstrap must be byte-identical to the oracle sweep's, on the
    real per-seed vectors -- guarding against estimator drift between the copies."""
    from run_oracle_noise_sweep import bootstrap_ci as canonical

    matrices = load_matrices(RESULTS)
    for teacher in ("gemini", "claude", "grok"):
        factors = per_seed_full_factors(matrices[teacher])
        mine = bootstrap_ci(factors, np.random.default_rng(BOOTSTRAP_RNG_SEED))
        theirs = canonical(factors, np.random.default_rng(BOOTSTRAP_RNG_SEED))
        assert mine == theirs, f"{teacher}: mirror {mine} != canonical {theirs}"
