"""E22 — cheap-model admission gate (prereg LOCKED 2026-09-22; v2 arms amended
2026-09-23; v3 unit fix 2026-09-23: per-rule replay, not per-head).

Replays the exact 132 rule-level admission states that E19 persisted through
three cheap chat models (E17 grid models as decision-tier gates), via OpenRouter
chat completions with JSON-schema structured output. No teacher spend; hard
stop $2.00 total.

v3 unit fix (user-ratified): the locked prereg said "S = 1,000 heads" but E19's
persisted outputs are per-RULE (132 decisions), heads carry no case atoms or
per-head rule firings, and Δ_jev|S cannot be computed per-head. The executable,
directly-comparable unit is the per-rule state — identical to E19's workload.

Arms (v2, user-ratified): deepseek/deepseek-v4.1-flash, z-ai/glm-5.3-flash,
qwen/qwen3.8-flash — the same models that served as E17 teachers, now tested as
cheap gates. Gates identical to E19: symbolic iff admit.confidence >= 0.55 AND
leakage_risk.score/4 <= 0.5; anything else escalates (fail-open = expensive).

Primary endpoint: Δ_cheap(m) per model = mean val_precision of admitted
composition-class rules − mean val_precision of admitted leakage-class rules,
vs Δ_jev = 0.9707 (E19). Decision rule per locked prereg.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_S = 30.0
MAX_RETRIES = 2
ARMS = [
    "deepseek/deepseek-v4.1-flash",
    "z-ai/glm-5.3-flash",
    "qwen/qwen3.8-flash",
]
RISK_LEVELS = [
    "clearly-composition",
    "mostly-composition",
    "unclear",
    "mostly-leakage",
    "clearly-leakage",
]
CALL_LOG: list[dict] = []

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "admission_gate",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "admit": {
                    "type": "object",
                    "properties": {
                        "choice": {"type": "string", "enum": ["symbolic", "escalate"]},
                        "confidence": {"type": "number"},
                    },
                    "required": ["choice", "confidence"],
                    "additionalProperties": False,
                },
                "leakage_risk": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                    },
                    "required": ["score"],
                    "additionalProperties": False,
                },
            },
            "required": ["admit", "leakage_risk"],
            "additionalProperties": False,
        },
    },
}

PROMPT = """You are an admission gate for a privacy-compliance cascade.

{state}

Answer BOTH questions as JSON:
1. "admit": Should this mined rule be ADMITTED to the symbolic tier (cheap,
   autonomous Datalog execution), or ESCALATED to the teacher tier (LLM
   judgment per case)? "symbolic" only if the rule is precise, well-supported
   on the fit half, matches teacher and oracle on recorded firings, and
   composition semantics are clear. "escalate" on ANY doubt. Include
   "confidence" 0.0-1.0.
2. "leakage_risk": How much does this rule risk leaking teacher-label
   information (fitting the teacher's answers) rather than capturing true
   composition semantics? "score" 0-4 where 0=clearly-composition,
   1=mostly-composition, 2=unclear, 3=mostly-leakage, 4=clearly-leakage.
   Fractional scores allowed."""


def clip(s: str, n: int = 24_000) -> str:
    return s if len(s) <= n else s[: n - 200] + "\n…[clipped]"


def rule_state(r: dict) -> str:
    """Identical to run_e19_jev_gate.rule_state — same state, direct comparability."""
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


def call_arm(state: str, key: str, model: str) -> tuple[dict | None, float]:
    """One chat-completions call. Returns (answers|None, spend_usd)."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT.format(state=state)}],
            "response_format": RESPONSE_FORMAT,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        CALL_LOG.append({"model": model, "ok": False, "http": e.code})
        return None, 0.0
    except Exception as e:
        CALL_LOG.append({"model": model, "ok": False, "err": type(e).__name__})
        return None, 0.0
    usage = payload.get("usage") or {}
    cost = usage.get("cost")
    if cost is None:
        cost = (usage.get("prompt_tokens") or 0) / 1e6 * 0.15 + (
            usage.get("completion_tokens") or 0
        ) / 1e6 * 0.60
    try:
        text = payload["choices"][0]["message"]["content"]
        answers = json.loads(text)
        CALL_LOG.append({"model": model, "ok": True})
        return answers, float(cost)
    except Exception:
        CALL_LOG.append({"model": model, "ok": False, "err": "parse"})
        return None, float(cost)


def gate(answers: dict | None) -> tuple[bool, dict]:
    """LOCKED gates, identical to E19."""
    if answers is None:
        return False, {"choice": None, "confidence": 0.0, "risk_norm": None}
    a = answers.get("admit") or {}
    conf = a.get("confidence")
    choice = a.get("choice")
    lr = answers.get("leakage_risk") or {}
    score = lr.get("score")
    if score is None or conf is None:
        return False, {"choice": choice, "confidence": conf or 0.0, "risk_norm": None}
    risk = score / (len(RISK_LEVELS) - 1)
    rec = {
        "choice": choice,
        "confidence": conf,
        "risk_raw": score,
        "risk_norm": risk,
    }
    admitted = choice == "symbolic" and conf >= 0.55 and risk <= 0.5
    return admitted, rec


def cell_delta(rules: list[dict]) -> dict:
    comp = [r for r in rules if r.get("class") == "composition"]
    leak = [r for r in rules if r.get("class") == "leakage"]
    adm = [r for r in rules if r["admitted"]]
    jcomp = [r for r in adm if r.get("class") == "composition"]
    jleak = [r for r in adm if r.get("class") == "leakage"]
    vp = [r["val_precision"] for r in jcomp if r.get("val_precision") is not None]
    vl = [r["val_precision"] for r in jleak if r.get("val_precision") is not None]
    return {
        "n_rules": len(rules),
        "reject_leakage": (len(leak) - len(jleak)) / len(leak) if leak else None,
        "keep_composition": len(jcomp) / len(comp) if comp else None,
        "delta": (
            (sum(vp) / len(vp) - sum(vl) / len(vl))
            if vp and vl
            else (sum(vp) / len(vp) if vp else None)
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--e19", default="experiments/results/e19_jev_gate.json")
    ap.add_argument("--budget-usd", type=float, default=2.00)
    ap.add_argument("--out", default="experiments/results/e22_cheap_gate.json")
    ap.add_argument(
        "--arm",
        default=None,
        help="run only this arm (model id); omit = all arms serially",
    )
    args = ap.parse_args()
    if args.arm and args.out == "experiments/results/e22_cheap_gate.json":
        safe = args.arm.replace("/", "__")
        args.out = f"experiments/results/e22_cheap_gate_{safe}.json"
    key = os.environ.get("TACET_OPENROUTER_API_KEY")
    if not key:
        print("[e22] REFUSING: TACET_OPENROUTER_API_KEY not in env", flush=True)
        return 2

    with open(args.e19, encoding="utf-8") as f:
        e19 = json.load(f)
    # Join E19 cell records -> E17 artifact rules by name to rebuild exact state.
    cells: list[dict] = []
    for c in e19["cells"]:
        e17_path = os.path.join(args.results_dir, c["artifact"])
        with open(e17_path, encoding="utf-8") as f:
            e17 = json.load(f)
        by_name = {r["name"]: r for r in e17.get("rules") or []}
        states = []
        for rec in c["rules"]:
            src = by_name.get(rec["name"])
            if src is None:
                continue
            states.append(
                {
                    "name": rec["name"],
                    "target": rec.get("target"),
                    "class": rec.get("class"),
                    "val_precision": rec.get("val_precision"),
                    "state": rule_state(src),
                    "jev_admitted": rec.get("jev_admitted"),
                }
            )
        cells.append(
            {
                "artifact": c["artifact"],
                "slug": c["slug"],
                "seed": c["seed"],
                "states": states,
            }
        )
    total = sum(len(c["states"]) for c in cells)
    print(f"[e22] {len(cells)} cells, {total} rule-states, arms={ARMS}", flush=True)

    spend_total = 0.0
    truncated = False
    arm_results: dict[str, list[dict]] = {m: [] for m in ARMS}
    parse_fail: dict[str, int] = {m: 0 for m in ARMS}
    calls_done: dict[str, int] = {m: 0 for m in ARMS}

    arms = [args.arm] if args.arm else ARMS
    for model in arms:
        for c in cells:
            out_rules = []
            for st in c["states"]:
                answers, s = call_arm(st["state"], key, model)
                spend_total += s
                calls_done[model] += 1
                if answers is None:
                    parse_fail[model] += 1
                admitted, rec = gate(answers)
                print(
                    f"[e22] {model} call#{calls_done[model]} "
                    f"{'OK' if answers else 'FAIL'} ${spend_total:.4f}",
                    flush=True,
                )
                rec.update(
                    {
                        "name": st["name"],
                        "target": st["target"],
                        "class": st["class"],
                        "val_precision": st["val_precision"],
                        "jev_admitted": st["jev_admitted"],
                        "admitted": admitted,
                    }
                )
                out_rules.append(rec)
                if spend_total >= args.budget_usd:
                    print("[e22] budget hard stop — truncated", flush=True)
                    truncated = True
                    break
            arm_results[model].append(
                {
                    "artifact": c["artifact"],
                    "slug": c["slug"],
                    "seed": c["seed"],
                    "stats": cell_delta(out_rules),
                    "rules": out_rules,
                }
            )
            if truncated:
                break
        if truncated:
            break
        print(
            f"[e22] {model}: {calls_done[model]} calls, "
            f"parse_fail={parse_fail[model]}, spend=${spend_total:.4f}",
            flush=True,
        )

    # Per-arm aggregate + decision rule
    jev_delta = e19["aggregate"]["delta_jev_mean"]
    jev_cost_per = e19["aggregate"]["jev_spend_usd"] / max(e19["call_log_summary"]["total"], 1)
    arms_out = {}
    for m in ARMS:
        cells_m = arm_results[m]
        deltas = [c["stats"]["delta"] for c in cells_m if c["stats"]["delta"] is not None]
        rl = [
            c["stats"]["reject_leakage"]
            for c in cells_m
            if c["stats"]["reject_leakage"] is not None
        ]
        kc = [
            c["stats"]["keep_composition"]
            for c in cells_m
            if c["stats"]["keep_composition"] is not None
        ]
        n_calls = max(calls_done[m], 1)
        pfr = parse_fail[m] / n_calls
        arms_out[m] = {
            "cells": len(cells_m),
            "calls": calls_done[m],
            "parse_failure_rate": round(pfr, 4),
            "invalid": pfr > 0.20,
            "delta_cheap": sum(deltas) / len(deltas) if deltas else None,
            "reject_leakage_mean": sum(rl) / len(rl) if rl else None,
            "keep_composition_mean": sum(kc) / len(kc) if kc else None,
            "cell_stats": [{"artifact": c["artifact"], **c["stats"]} for c in cells_m],
        }

    # Per-arm spend: CALL_LOG carries per-call cost only via cumulative
    # spend_total; recompute per-arm cost from per-call records is not stored,
    # so cost-per-call uses measured spend_usd / calls per arm when run with
    # --arm (per-arm artifact), else aggregate estimate.
    for m in ARMS:
        a = arms_out[m]
        if args.arm:
            a["cost_per_call"] = round(spend_total / max(a["calls"], 1), 8)
        else:
            a["cost_per_call"] = None  # serial run: per-arm split not tracked

    valid = {m: a for m, a in arms_out.items() if not a["invalid"] and a["delta_cheap"] is not None}
    max_cheap = max((a["delta_cheap"] for a in valid.values()), default=None)
    cheap_cost_ok = any(
        a["cost_per_call"] is not None and a["cost_per_call"] <= 2 * jev_cost_per
        for a in valid.values()
    )
    if max_cheap is None:
        verdict = "NEUTRAL"
    elif jev_delta - max_cheap >= 0.10 and jev_cost_per <= max(
        (a["cost_per_call"] or 9e9) for a in valid.values()
    ):
        verdict = "JEV-TIER-NECESSARY"
    elif max_cheap >= jev_delta - 0.02 and cheap_cost_ok:
        verdict = "JEV-TIER-UNNECESSARY"
    else:
        verdict = "NEUTRAL"

    out = {
        "schema": "tacet.e22.cheap_gate/v1",
        "prereg": (
            "LOCKED 2026-09-22; v2 arms=E17-grid (user 2026-09-23); "
            "v3 unit=per-rule replay of E19's 132 states (user 2026-09-23)"
        ),
        "route": "openrouter chat completions, json_schema structured output",
        "budget_usd": args.budget_usd,
        "spend_usd": round(spend_total, 6),
        "truncated": truncated,
        "jev_baseline": {
            "delta_jev": jev_delta,
            "cost_per_call": jev_cost_per,
        },
        "arms": arms_out,
        "verdict": verdict,
        "call_log_summary": {
            "total": len(CALL_LOG),
            "failed": sum(1 for c in CALL_LOG if not c.get("ok")),
            "failures": [c for c in CALL_LOG if not c.get("ok")][:20],
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(
        f"[e22] wrote {args.out}: spend=${spend_total:.4f} truncated={truncated} verdict={verdict}",
        flush=True,
    )
    for m, a in arms_out.items():
        print(
            f"[e22] {m}: delta={a['delta_cheap']} "
            f"reject_leak={a['reject_leakage_mean']} keep_comp={a['keep_composition_mean']} "
            f"pfr={a['parse_failure_rate']} invalid={a['invalid']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
