"""Step 3 checkpoint — a MANUAL script, not a test. Runs the supervisor +
extractor over sample questions and prints what each produced, so you can eyeball
the two prompts. Makes real (cheap) LLM calls; needs GOOGLE_API_KEY.

    python -m src.graph.check
"""
from __future__ import annotations

from src.data.catalog import get_catalog
from src.graph.nodes import extractor, supervisor
from src.schemas import Clarification, DATA_INTENTS

SAMPLES = [
    "What is the total P&L for all my properties this year?",
    "Compare Building 120 and Building 180",
    "How does this quarter compare to the same period last year?",
    "Who are my top tenants, and is anything unusual in the numbers?",
    "Show me revenue by month for Building 160",
    "Which expense category is the largest?",
    "Break down 2024 P&L by building",
    "Tell me about Building 17",
    "What is the price of my asset at 123 Main St?",
    "How did Building 999 do?",
    "What data do you have?",
    "How is NOI calculated?",
    "How are things going?",
    "Write me a poem",
    "Compare Q1 and Q2 for Building 120, rank the tenants, flag anomalies, "
    "show revenue by month, and give me the portfolio total",
]


def main() -> None:
    cat = get_catalog()
    for question in SAMPLES:
        print("\n" + "=" * 80)
        print("Q:", question)
        plan = supervisor(question)
        print("  reasoning:", plan.reasoning)
        for i, sub in enumerate(plan.sub_questions, 1):
            print(f"  [{i}] ({sub.intent.value}) {sub.text}")
            if sub.intent not in DATA_INTENTS:
                print("      -> responder handles directly (no query)")
                continue
            out = extractor(sub, cat)
            if isinstance(out, Clarification):
                print(f"      -> clarify: {out.reason.value} — {out.detail}")
                if out.options:
                    print(f"         options: {list(out.options)}")
            else:
                print(f"      -> {out.operation} | {out.timeframe_label}"
                      f" | group_by={out.group_by} | top_n={out.top_n}"
                      f" | rank_by={out.rank_by} | direction={out.direction}"
                      f" | members={list(out.members)} | subject={out.subject}")
                print(f"         filter: {out.query.describe()}")
                for c in out.caveats:
                    print(f"         caveat: {c}")


if __name__ == "__main__":
    main()
