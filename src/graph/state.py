"""The graph's shared state and the per-sub-question result record.

AgentState is the "clipboard" every node reads and writes a slice of. BranchResult
is one sub-question's full outcome — the grounded query, or a clarification, plus
whatever the analyst computed and a one-line headline for the responder.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from src.schemas import Clarification, Intent, ResolvedQuery, SubQuestion


@dataclass
class BranchResult:
    text: str
    intent: Intent
    resolved: ResolvedQuery | None = None
    clarification: Clarification | None = None
    data: Any = None                       # Metric | Breakdown | Card | list[Finding]
    headline: str = ""                     # deterministic one-liner, grounding material
    caveats: tuple[str, ...] = ()
    trace: list[str] = field(default_factory=list)

    @property
    def needs_analyst(self) -> bool:
        return self.resolved is not None and self.clarification is None


class AgentState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]          # (role, text) earlier turns
    pending: list[SubQuestion]              # drained by the extract -> analyst loop
    results: list[BranchResult]
    dropped: int                            # sub-questions cut by MAX_SUB_QUESTIONS
    answer: str
    trace: Annotated[list[str], operator.add]
    error: str | None
