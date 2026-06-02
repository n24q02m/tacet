"""Probe the Gemini / Gemma model list with a one-token call apiece.

Useful for verifying *which* of the rotating-router models exist on the
account / region the GEMINI_API_KEY is for.  Free-tier model availability
changes over time; the rotation gracefully cools any 404/400 out, but
running this once lets the operator confirm the configured list.

Usage::

    export GEMINI_API_KEY=...
    python scripts/probe_gemini_models.py
    python scripts/probe_gemini_models.py --models gemini-3.5-flash,gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import json
import os
import time

import httpx

from tacet.llm.teachers import DEFAULT_ROTATING_MODELS

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"


def probe(model: str, api_key: str, timeout: float = 15.0) -> dict:
    t0 = time.monotonic()
    try:
        r = httpx.post(
            ENDPOINT.format(m=model),
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": "reply OK"}]}]},
            timeout=timeout,
        )
        latency = time.monotonic() - t0
        if r.status_code == 200:
            text = (
                r.json()
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            return {
                "model": model,
                "ok": True,
                "status": 200,
                "latency_s": round(latency, 2),
                "sample": text[:40],
            }
        return {
            "model": model,
            "ok": False,
            "status": r.status_code,
            "latency_s": round(latency, 2),
            "error": r.text[:200],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "model": model,
            "ok": False,
            "status": 0,
            "latency_s": round(time.monotonic() - t0, 2),
            "error": repr(e)[:200],
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        default=",".join(DEFAULT_ROTATING_MODELS),
        help="comma-separated model list (default: rotating router order)",
    )
    ap.add_argument(
        "--qps", type=float, default=9 / 60, help="rate limit between probes (calls / sec)"
    )
    args = ap.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("TACET_GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY (or TACET_GEMINI_API_KEY).")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    interval = 1.0 / args.qps
    rows: list[dict] = []
    print(f"{'Model':30s} {'Status':6s}  {'Latency':>8s}  Note")
    print("-" * 80)
    for i, model in enumerate(models):
        if i > 0:
            time.sleep(interval)
        r = probe(model, api_key)
        mark = "OK" if r["ok"] else "FAIL"
        note = r.get("sample") or r.get("error", "")
        print(f"{model:30s} {mark} {r['status']:>4d}  {r['latency_s']:>7.2f}s  {note}")
        rows.append(r)
    available = [r["model"] for r in rows if r["ok"]]
    print(f"\nAvailable: {len(available)}/{len(models)}")
    if available:
        print(f"Suggested TACET_ROTATING_MODELS={','.join(available)}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
