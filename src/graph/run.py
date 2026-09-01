"""One turn through the full graph, from the command line.

    python -m src.graph.run "Compare Building 120 and Building 180, and flag anything odd"

Makes real LLM calls (supervisor + extractor) — needs GOOGLE_API_KEY in .env.
"""
from __future__ import annotations

import sys
import uuid

from src.graph.build import build_graph

_BLANK = {"question": "", "history": [], "pending": [], "results": [], "trace": []}


def answer(question: str, thread_id: str | None = None) -> dict:
    graph = build_graph()
    cfg = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    return graph.invoke({**_BLANK, "question": question}, cfg)


def main(argv: list[str]) -> None:
    if not argv:
        print('usage: python -m src.graph.run "your question"')
        raise SystemExit(1)
    state = answer(" ".join(argv))
    print("\n=== TRACE ===")
    for line in state.get("trace", []):
        print(line)
    print("\n=== ANSWER ===")
    print(state.get("answer", "(no answer)"))


if __name__ == "__main__":
    main(sys.argv[1:])
