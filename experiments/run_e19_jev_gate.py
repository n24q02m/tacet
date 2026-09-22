"""E19 — jev-tier admission gate (prereg LOCKED 2026-09-21, ledger `tacet-research-ledger.md` E19).

Replays per-rule admission decisions from recorded E17 artifacts through the jev
decision model. No teacher spend; jev budget hard stop $2.00.

Route (LOCKED): OpenRouter alpha/decisions `typesafe/jev-1.13`; native
api.typesafe.ai/v1/systemone `jev-latest` = fallback if OR route is dead.
Observed 2026-09-22: OR endpoint returns 404 -> harness probes once at start and
falls back to native for the whole run; route + failures recorded per call.

Locked gates: route symbolic iff admit.confidence >= 0.55 AND
leakage_risk.score/(n-1) <= 0.5; every other case (missing confidence, missing
score after one retry, non-200, timeout, malformed) escalates — fail-open = fail
EXPENSIVE (escalate). Never guess.

Grid (LOCKED): mọi E17 cell có per-head logs usable, tối thiểu 5 cells = 1
seed/model. `luna_s0_high` (August, high) STAYS VALID per v2.1 authorization
("Luna s0 stays valid"); truncated cells are invalid per v2.1.

Artifacts carry ALL primitives so any locked reading of the endpoints is
derivable at verdict time without re-calling jev.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.error
import urllib.request

API_OR = "https://openrouter.ai/api/v1/alpha/decisions"
API_NATIVE = "https://api.typesafe.ai/v1/systemone"
MODEL_OR = "typesafe/jev-1.13"
MODEL_NATIVE = "jev-latest"
TIMEOUT_S = 8.0  # measured native p50 ~1.14s; the 1.5s spec is the router port, not this harness
PRICE_PER_MTOK_IN = 0.042  # measured 21/09 (input-only billing)
RISK_LEVELS = [
    "clearly-composition",
    "mostly-composition",
    "unclear",
    "mostly-leakage",
    "clearly-leakage",
]
CALL_LOG: list[dict] = []

RISK_Q = {
    "type": "score",
    "instructions": (
        "How much does this mined Datalog rule risk leaking teacher-label "
        "information (fitting the teacher's answers) rather than capturing true "
        "composition semantics of the privacy-compliance domain?"
    ),
    "criteria": RISK_LEVELS,
}
ADMIT_Q = {
    "type": "choice",
    "instructions": (
        "Should this mined rule be ADMITTED to the symbolic tier (cheap, "
        "autonomous Datalog execution in the compliance cascade), or ESCALATED "
        "to the teacher tier (LLM judgment per case)?"
    ),
    "criteria": {
        "symbolic": (
            "Rule is precise, well-supported on the fit half, matches teacher "
            "and oracle on recorded firings, and composition semantics are clear."
        ),
        "escalate": (
            "Any doubt about correctness, support, or leakage risk — the rule "
            "belongs behind teacher judgment."
        ),
    },
}


def clip(s: str, n: int = 24_000) -> str:
    return s if len(s) <= n else s[: n - 200] + "\n…[clipped]"


# NOTE: artifact schema persists only rule NAME (atoms encoded) + stats — no
# Datalog body text (mined_candidates is a count). This clip is the max
# fidelity available without re-mining (re-mining forbidden: no new cells).
def rule_state(r: dict) -> str:
    return clip(
        "Domain: GDPR privacy-compliance cascade (PrivaCI bench). "
        "A Datalog rule was mined from the fit half of teacher-answered heads.\n"
        f"Rule (head atoms encoded in name): {r.get('name')}\n"
        f"Target article: {r.get('target')}\n"
        f"Miner confidence: {r.get('confidence')}\n"
        f"Support (fit half): {r.get('support')}\n"
        f"Fired on all heads: {r.get('fired_all')}; matches oracle: "
        f"{r.get('match_all_vs_oracle')}\n"
        f"Fired on validation half: {r.get('fired_validation')}; matches "
        f"teacher: {r.get('match_validation_vs_teacher')}; teacher abstains: "
        f"{r.get('teacher_abstain_validation')}\n"
    )


def call_jev(state: str, key: str, route: str) -> tuple[dict | None, float]:
    """One call, both questions. Returns (answers|None, spend_usd); errors go to CALL_LOG."""
    if route == "native":
        url, model = API_NATIVE, MODEL_NATIVE
    else:
        url, model = API_OR, MODEL_OR
    body = json.dumps(
        {
            "model": model,
            "state": state,
            "questions": {"admit": ADMIT_Q, "leakage_risk": RISK_Q},
        }
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            payload = json.loads(resp.read())
        CALL_LOG.append({"route": route, "ok": True})
    except urllib.error.HTTPError as e:
        CALL_LOG.append({"route": route, "ok": False, "http": e.code})
        return None, 0.0
    except Exception as e:
        CALL_LOG.append({"route": route, "ok": False, "err": type(e).__name__})
        return None, 0.0
    usage = payload.get("usage") or {}
    toks = usage.get("input_tokens") or 0
    return payload.get("answers"), toks / 1e6 * PRICE_PER_MTOK_IN


def gate(answers: dict | None, retry_state: str, key: str, route: str) -> tuple[bool, dict, float]:
    """Apply LOCKED gates; missing score retries once; anything else escalates."""
    spend = 0.0
    for attempt in range(2):
        if answers is None:
            break
        a = answers.get("admit") or {}
        conf = a.get("confidence")
        choice = a.get("choice")
        lr = answers.get("leakage_risk") or {}
        score = lr.get("score")
        if score is not None:
            risk = score / (len(RISK_LEVELS) - 1)
            rec = {
                "choice": choice,
                "confidence": conf if conf is not None else 0.0,
                "risk_raw": score,
                "risk_norm": risk,
                "attempt": attempt,
            }
            admitted = choice == "symbolic" and rec["confidence"] >= 0.55 and risk <= 0.5
            return admitted, rec, spend
        if attempt == 0:
            answers, s = call_jev(retry_state, key, route)
            spend += s
            continue
        break
    return False, {"choice": None, "confidence": 0.0, "risk_norm": None, "attempt": 2}, spend


def cell_stats(cs: dict) -> dict:
    rl = [r for r in cs.get("rules", []) if not cs.get("partial")]
    comp = [r for r in rl if r.get("class") == "composition"]
    leak = [r for r in rl if r.get("class") == "leakage"]
    jadm = [r for r in rl if r["jev_admitted"]]
    jcomp = [r for r in jadm if r.get("class") == "composition"]
    jleak = [r for r in jadm if r.get("class") == "leakage"]
    vp = [r["val_precision"] for r in jcomp if r.get("val_precision") is not None]
    vl = [r["val_precision"] for r in jleak if r.get("val_precision") is not None]
    return {
        "n_rules": len(rl),
        "reject_leakage": (len(leak) - len(jleak)) / len(leak) if leak else None,
        "keep_composition": len(jcomp) / len(comp) if comp else None,
        "admitted_val_prec_composition": vp,
        "admitted_val_prec_leakage": vl,
        "delta_jev": (
            (sum(vp) / len(vp) - sum(vl) / len(vl))
            if vp and vl
            else (sum(vp) / len(vp) if vp else None)
        ),
        "e17_reject_leakage": (cs.get("e17_admission_test") or {}).get("reject_leakage_frac"),
        "e17_keep_composition": (cs.get("e17_admission_test") or {}).get("keep_composition_frac"),
    }


def finish(args, cells, partial_rules, partial_artifact, spend, truncated, route) -> int:
    if partial_rules is not None:
        cells.append({"artifact": partial_artifact, "rules": partial_rules, "partial": True})
    out = {
        "schema": "tacet.e19.jev_gate/v1",
        "prereg": (
            "LOCKED 2026-09-21 (ledger E19); gates conf>=0.55 & risk<=0.5; fail-open=escalate"
        ),
        "rule_representation": (
            "E17 artifact schema tacet.e17.admission/v1 persists candidate rules "
            "as name-encoded atoms (target + body bindings) + stats; no full "
            "Datalog text exists in the artifact. State clips use the "
            "name-decoded representation — documented prereg-compatible."
        ),
        "route": route,
        "route_note": (
            "OR alpha/decisions 404 observed 2026-09-22; native "
            "api.typesafe.ai/v1/systemone per LOCKED fallback"
        ),
        "state_representation_note": (
            "artifact schema v1 carries no Datalog body text; rule head/body "
            "decoded from rules[].name (mined_<target>__<atom>=<value> pairs) — "
            "the only artifact-complete representation"
        ),
        "call_log_summary": {
            "total": len(CALL_LOG),
            "failed": sum(1 for c in CALL_LOG if not c.get("ok")),
            "failures": [c for c in CALL_LOG if not c.get("ok")][:20],
        },
        "budget_usd": args.budget_usd,
        "spend_usd": round(spend, 6),
        "truncated": truncated,
        "cells": [{**cs, "stats": cell_stats(cs)} for cs in cells if not cs.get("partial")],
    }
    leak_rej = [
        c["stats"]["reject_leakage"]
        for c in out["cells"]
        if c["stats"]["reject_leakage"] is not None
    ]
    comp_keep = [
        c["stats"]["keep_composition"]
        for c in out["cells"]
        if c["stats"]["keep_composition"] is not None
    ]
    deltas = [c["stats"]["delta_jev"] for c in out["cells"] if c["stats"]["delta_jev"] is not None]
    out["aggregate"] = {
        "cells": len(out["cells"]),
        "jev_spend_usd": round(spend, 6),
        "reject_leakage_mean": sum(leak_rej) / len(leak_rej) if leak_rej else None,
        "keep_composition_mean": sum(comp_keep) / len(comp_keep) if comp_keep else None,
        "delta_jev_mean": sum(deltas) / len(deltas) if deltas else None,
        "note": "verdict computed at analysis time from primitives per locked decision rule",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(
        f"[e19] wrote {args.out}: cells={out['aggregate']['cells']} "
        f"spend=${spend:.6f} truncated={truncated}"
    )
    print(f"[e19] aggregate: {json.dumps(out['aggregate'], default=str)[:400]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--pattern", default="e17_admission_*.json")
    ap.add_argument(
        "--exclude",
        default="0731",
        help="comma substrings to skip (luna_s0_high STAYS — v2.1 keeps it valid)",
    )
    ap.add_argument("--budget-usd", type=float, default=2.00)
    ap.add_argument("--out", default="experiments/results/e19_jev_gate.json")
    args = ap.parse_args()

    excl = [g.strip() for g in args.exclude.split(",") if g.strip()]
    # grid check FIRST — credentials are only required once calls will actually start
    paths = []
    for p in sorted(glob.glob(os.path.join(args.results_dir, args.pattern))):
        if any(g in p for g in excl) or p.endswith(".partial"):
            continue
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("truncated_by_budget"):
            print(f"[e19] skip invalid (truncated) cell {os.path.basename(p)}")
            continue
        paths.append((p, d))
    n_valid = len({d["slug"] for _, d in paths})
    if n_valid < 5 or len(paths) < 5:
        print(
            f"[e19] grid below prereg minimum (valid cells={len(paths)}, "
            f"models={n_valid}; need >=5 cells across >=5 models) — REFUSING"
        )
        return 2

    key_or = os.environ.get("TACET_OPENROUTER_API_KEY")
    key_native = os.environ.get("TACET_JEV_API_KEY")

    # route pick (LOCKED chain): probe OR once; 404/dead -> native for the run
    probe, _ = call_jev("route probe", key_or or "", route="or")
    route = "or" if probe is not None else "native"
    key = key_or if route == "or" else key_native
    if not key:
        print(f"[e19] route={route} but its credential is missing from env", file=sys.stderr)
        return 9
    print(
        f"[e19] route: {route} (probe {'ok' if probe is not None else 'failed -> native fallback'})"
    )

    spend_total = 0.0
    cells: list[dict] = []
    for p, d in paths:
        rec_rules = []
        for r in d.get("rules") or []:
            state = rule_state(r)
            answers, s = call_jev(state, key, route)
            spend_total += s
            admitted, rec, s2 = gate(answers, state, key, route)
            spend_total += s2
            rec.update(
                {
                    "name": r.get("name"),
                    "target": r.get("target"),
                    "class": r.get("class"),
                    "val_precision": r.get("val_precision"),
                    "e17_admitted": r.get("admitted"),
                    "jev_admitted": admitted,
                }
            )
            rec_rules.append(rec)
            if spend_total >= args.budget_usd:
                print("[e19] jev budget hard stop — marking truncated")
                return finish(args, cells, rec_rules, p, spend_total, truncated=True, route=route)
        cells.append(
            {
                "artifact": os.path.basename(p),
                "slug": d.get("slug"),
                "seed": d.get("seed"),
                "e17_admission_test": d.get("admission_test"),
                "e17_arm_accuracy": d.get("arm_accuracy_validation"),
                "rules": rec_rules,
            }
        )
    return finish(args, cells, None, None, spend_total, truncated=False, route=route)


if __name__ == "__main__":
    sys.exit(main())
