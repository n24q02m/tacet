"""Compliance smoke: 20 real PrivaCI-Bench GDPR cases against both teachers.

GATE for the v2 port (phase 1.B Task B): if neither teacher reaches ~0.7
verdict accuracy the teacher signal is too noisy to distil from and the
port stops here. PrivaCI-Bench reports GPT-4o-mini at 92% on GDPR.

Run (keys in skret /tacet/prod):

    MSYS_NO_PATHCONV=1 skret run -e prod --path=/tacet/prod -- \
        uv run python scripts/smoke_compliance.py [--n 20] [--privaci ../PrivaCI-Bench]
"""

import argparse
import json
import os
import random
from pathlib import Path

from tacet.data.privaci import load_privaci
from tacet.llm.metering import MeteredTeacher, PriceTable
from tacet.llm.teachers.compliance import COMPLIANCE_PROMPT_TEMPLATE, parse_compliance_answer
from tacet.llm.teachers.llm import GeminiRestTeacher, GrokTeacher


def evaluate(name, teacher, cases):
    metered = MeteredTeacher(teacher, prices=PriceTable.default(), model=name)
    verdict_ok = 0
    abstain = 0
    tp = fp = fn = 0
    rows = []
    for c in cases:
        resp = metered.answer(None, c.case_content, "verdict")
        verdict, arts = parse_compliance_answer(resp.answers)
        gold_v, gold_a = c.norm_type, set(c.violated_articles)
        pred_a = set(arts)
        if verdict == "abstain":
            abstain += 1
        if verdict == gold_v:
            verdict_ok += 1
        tp += len(pred_a & gold_a)
        fp += len(pred_a - gold_a)
        fn += len(gold_a - pred_a)
        rows.append(
            {
                "case_id": c.case_id,
                "gold": [gold_v, sorted(gold_a)],
                "pred": [verdict, sorted(pred_a)],
            }
        )
    n = len(cases)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "verdict_acc": verdict_ok / n,
        "n_abstain": abstain,
        "article_micro_p": round(p, 4),
        "article_micro_r": round(r, 4),
        "article_micro_f1": round(f1, 4),
        "usd": round(metered.total_cost_usd, 6),
        "prompt_tokens": metered.total_prompt_tokens,
        "completion_tokens": metered.total_completion_tokens,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--privaci", default="../PrivaCI-Bench")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cases = load_privaci(args.privaci, split="GDPR")
    sample = random.Random(args.seed).sample(cases, args.n)
    teachers = {
        "grok-4.3": GrokTeacher(
            os.environ["TACET_XAI_API_KEY"],
            "grok-4.3",
            prompt_template=COMPLIANCE_PROMPT_TEMPLATE,
        ),
        "gemini-3.5-flash": GeminiRestTeacher(
            os.environ["TACET_GEMINI_API_KEY"],
            model="gemini-3.5-flash",
            endpoint="vertex",
            qps=None,
            prompt_template=COMPLIANCE_PROMPT_TEMPLATE,
        ),
    }
    report = {"n": args.n, "seed": args.seed, "split": "GDPR", "teachers": {}}
    for name, t in teachers.items():
        res = evaluate(name, t, sample)
        report["teachers"][name] = res
        print(
            f"{name}: verdict_acc={res['verdict_acc']:.2f} "
            f"art_f1={res['article_micro_f1']:.3f} "
            f"(p={res['article_micro_p']:.3f} r={res['article_micro_r']:.3f}) "
            f"abstain={res['n_abstain']} usd=${res['usd']:.4f}"
        )
    out = Path("experiments/results/compliance_teacher_smoke.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
