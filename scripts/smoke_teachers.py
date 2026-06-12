"""One real call per phase-1 teacher (grok-4.3 + gemini-3.5-flash @ vertex).

Verifies, against the live APIs: both keys work, the Vertex express endpoint
accepts header auth, and MeteredTeacher sees provider-reported usage.

Run (keys live in skret /tacet/prod):

    MSYS_NO_PATHCONV=1 skret run -e prod --path=/tacet/prod -- \
        uv run python scripts/smoke_teachers.py
"""

import os

from tacet.llm.metering import MeteredTeacher, PriceTable
from tacet.llm.teachers.llm import GeminiRestTeacher, GrokTeacher


def main() -> None:
    prices = PriceTable.default()
    teachers = {
        "grok-4.3": GrokTeacher(os.environ["TACET_XAI_API_KEY"], "grok-4.3"),
        "gemini-3.5-flash": GeminiRestTeacher(
            os.environ["TACET_GEMINI_API_KEY"],
            model="gemini-3.5-flash",
            endpoint="vertex",
            qps=None,
        ),
    }
    for name, raw in teachers.items():
        metered = MeteredTeacher(raw, prices=prices, model=name)
        resp = metered.answer(None, "France", "borders")
        usage = getattr(raw, "last_usage", None)
        print(f"{name}: answers={resp.answers!r} usd={metered.total_cost_usd:.6f} usage={usage}")
        assert resp.answers, f"{name} returned no answers"
        assert metered.total_cost_usd > 0, f"{name} metering produced zero cost"
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
