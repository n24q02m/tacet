"""Regression tests for the E17-SYN near-miss-path preflight runner."""

import socket
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import run_e17_admission as e17  # noqa: E402
import run_e17syn_nearmiss as syn  # noqa: E402


def test_split_function_is_imported_not_copied():
    """The estimator split MUST be the locked E17 function (ledger 1456)."""
    assert syn._half is e17._half
    assert syn.DELTA_POSITIVE == e17.DELTA_POSITIVE
    assert syn.REJECT_LEAKAGE_MIN == e17.REJECT_LEAKAGE_MIN


def test_build_world_is_deterministic():
    a = syn.build_world(0.25, 1)
    b = syn.build_world(0.25, 1)
    assert [c["case_id"] for c in a["cases"]] == [c["case_id"] for c in b["cases"]]
    assert [c["entity"] for c in a["cases"]] == [c["entity"] for c in b["cases"]]
    assert [c["leak_fired"] for c in a["cases"]] == [c["leak_fired"] for c in b["cases"]]
    assert [c["leak_answer"] for c in a["cases"]] == [c["leak_answer"] for c in b["cases"]]


def test_case_ids_are_stable_across_theta_for_same_seed():
    """Validation halves must line up across the theta sweep (ledger 1455)."""
    ids_low = [c["case_id"] for c in syn.build_world(0.10, 0)["cases"]]
    ids_high = [c["case_id"] for c in syn.build_world(0.60, 0)["cases"]]
    assert ids_low == ids_high


def test_head_law_is_zipfian_not_uniform():
    """Rejection-sampled zipf heads concentrate: the hottest head dominates."""
    from collections import Counter

    counts = Counter(c["entity"] for c in syn.build_world(0.25, 0)["cases"])
    assert counts.most_common(1)[0][1] >= 30  # >=10% of the 300-case stream
    assert len(counts) < syn.N_ENTITIES  # the tail entities never all appear


def test_composition_is_exact_and_leak_tracks_one_minus_theta():
    for theta in (0.10, 0.25, 0.40, 0.60):
        for seed in (0, 1, 2):
            cell = syn.run_cell(syn.build_world(theta, seed))
            assert cell["oracle_exact"]["comp_precision"] == 1.0
            # Band widens at small theta: few fires, coarse granularity.
            band = 0.20 if theta == 0.10 else 0.15
            assert abs(cell["oracle_exact"]["leak_precision"] - (1.0 - theta)) <= band
            assert cell["leak_fires_total"] > 0


def test_estimator_vs_exact_dual_verdict_and_mismatch_recording():
    """Craft a world where estimator and oracle disagree on the verdict."""
    cases = []
    for i in range(20):
        cid = f"s0c{i:03d}"
        # Deterministic by construction: a fire is correct exactly on the
        # validation half, so the estimator sees 1.0 and the oracle 0.5.
        ok = e17._half(0, cid) == "validation"
        cases.append(
            {
                "case_id": cid,
                "entity": 0,
                "gold": 7,
                "comp_covers": True,
                "comp_answer": 7,
                "leak_fired": True,
                "leak_answer": 7 if ok else 9,
            }
        )
    cell = syn.run_cell({"theta": 0.10, "seed": 0, "cases": cases})
    n_val = sum(1 for c in cases if e17._half(0, c["case_id"]) == "validation")
    assert 0 < n_val < 20
    assert cell["oracle_exact"]["leak_precision"] == round(n_val / 20, 4)
    assert cell["estimator"]["verdict"] != cell["oracle_exact"]["verdict"]
    assert cell["estimator_matches_oracle"] is False


def test_decide_campaign_locked_rule_boundaries():
    def cell(theta, seed, delta, match=True):
        return {
            "theta": theta,
            "seed": seed,
            "estimator": {"delta": delta},
            "oracle_exact": {"delta": delta},
            "estimator_matches_oracle": match,
        }

    # POSITIVE: every theta<=0.40 cell >= 0.20 AND verdict match >= 90%.
    pos_cells = [
        cell(t, s, 0.25, match=(t, s) != (0.60, 2))
        for t in (0.10, 0.25, 0.40, 0.60)
        for s in (0, 1, 2)
    ]
    assert syn.decide_campaign(pos_cells)[0] == "POSITIVE"

    # NEUTRAL by match rate: all deltas fine but two cells mismatch (<90%).
    neutral_cells = [
        cell(t, s, 0.25, match=(t, s) not in ((0.10, 0), (0.25, 1)))
        for t in (0.10, 0.25, 0.40, 0.60)
        for s in (0, 1, 2)
    ]
    assert syn.decide_campaign(neutral_cells)[0] == "NEUTRAL"

    # NEUTRAL by delta: one low-theta cell sits between the margins.
    mid_cells = [cell(t, s, 0.25) for t in (0.10, 0.25, 0.40, 0.60) for s in (0, 1, 2)]
    mid_cells[0] = cell(0.10, 0, 0.15)
    assert syn.decide_campaign(mid_cells)[0] == "NEUTRAL"

    # NEGATIVE: any single cell below DELTA_NEGATIVE.
    neg_cells = [cell(t, s, 0.25) for t in (0.10, 0.25, 0.40, 0.60) for s in (0, 1, 2)]
    neg_cells[1] = cell(0.10, 1, 0.04)
    assert syn.decide_campaign(neg_cells)[0] == "NEGATIVE"


def test_provider_assert_fails_closed(monkeypatch):
    def resolves(host, port):
        return [(None, None, None, "", (host, port))]

    def refuses(host, port):
        raise socket.gaierror("no dns")

    monkeypatch.setattr(socket, "getaddrinfo", resolves)
    with pytest.raises(SystemExit, match="providers unreachable"):
        syn.assert_provider_unreachable()
    monkeypatch.setattr(socket, "getaddrinfo", refuses)
    syn.assert_provider_unreachable()  # must not raise


def test_runner_refuses_without_prereg_lock():
    """Fail-closed: no execution without a declared prereg lock (ledger 1452)."""
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent.parent / "experiments" / "run_e17syn_nearmiss.py"),
            "--out",
            "unused.json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode != 0
    assert "prereg-lock-sha" in proc.stderr
