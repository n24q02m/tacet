"""Probe every model in the rotating router against the live Gemini API.

For each of the 7 free-tier models the rotating router cycles through, send
a single trivial structured-QA prompt and record:

  - whether it answered at all,
  - the latency,
  - the first 80 chars of the answer.

The result tells us which models are currently reachable on the free tier
and therefore which slots in the rotation actually contribute throughput.
The whole probe sends 7 requests and finishes in well under a minute even
when several models 429 (each failure is bounded by a 10s timeout).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from tacet.core.graph import WorldGraph
from tacet.llm.teachers.llm import DEFAULT_ROTATING_MODELS, GeminiRestTeacher


def main(out_path: Path) -> dict:
    key = os.environ["GEMINI_API_KEY"]
    g = WorldGraph()
    g.add_node("France", "Country")
    g.add_node("Belgium", "Country")
    g.add_edge("France", "borders", "Belgium")

    records: list[dict] = []
    for model in DEFAULT_ROTATING_MODELS:
        teacher = GeminiRestTeacher(api_key=key, model=model, qps=10.0)
        t0 = time.perf_counter()
        try:
            resp = teacher.answer(g, "France", "borders")
            dt = time.perf_counter() - t0
            answered = bool(resp.answers)
            records.append(
                {
                    "model": model,
                    "answered": answered,  # True iff the model returned a parseable answer
                    "latency_s": round(dt, 3),
                    "answer_preview": str(resp.answers)[:80],
                    "cost": getattr(resp, "cost", None),
                }
            )
        except Exception as e:  # noqa: BLE001
            dt = time.perf_counter() - t0
            err = repr(e)[:160]
            records.append(
                {"model": model, "answered": False, "latency_s": round(dt, 3), "error": err}
            )
    summary = {
        "tested_models": len(records),
        "answered": sum(1 for r in records if r["answered"]),
        "records": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/results/rotation_smoke.json")
    args = p.parse_args()
    s = main(Path(args.out))
    print(json.dumps(s, indent=2))
