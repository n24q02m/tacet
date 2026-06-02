"""Generate `paper/results/tab_gemini.tex` from a `gemini_smoke.json` report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(data: dict) -> str:
    runs = data["runs"]
    rows = [
        (r"Oracle cascade", runs["oracle_cascade"]),
        (r"\textbf{Gemini cascade}", runs["gemini_cascade"]),
        (r"Gemini LLM-only", runs["gemini_llm_only"]),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{End-to-end run with a real Gemini~2.5~Flash Tier-3 teacher",
        f"on a {data['workload_size']}-query subset of the synthetic-org",
        f"benchmark (seed {data['seed']}).}}",
        r"\label{tab:gemini}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"System & Cost (USD) & Accuracy \\",
        r"\midrule",
    ]
    for name, r in rows:
        lines.append(f"{name} & {r['cost']:.3f} & {r['accuracy']:.3f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results/gemini_smoke.json")
    ap.add_argument("--out", default="experiments/results/tab_gemini.tex")
    args = ap.parse_args()
    data = json.loads(Path(args.results).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
