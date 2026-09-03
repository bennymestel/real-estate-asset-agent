"""StateGraph wiring: supervisor -> (extract <-> analyst loop) -> responder.

    supervisor   classify + split + cap at MAX_SUB_QUESTIONS
       |
    extract      pop one sub-question -> ResolvedQuery | Clarification | (non-data)
       |  \
       |   analyst   run the pandas tools for that sub-question
       |  /
    responder    write the answer with one LLM call, then fact-check every figure

Every node is wrapped by @safe_node: an exception is caught, written to
state['error'], and routing sends it straight to the responder.
"""
from __future__ import annotations

import functools
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.data.catalog import get_catalog
from src.data.loader import load_ledger
from src.graph import analyst, nodes, responder
from src.graph.state import AgentState, BranchResult
from src.schemas import Clarification, DATA_INTENTS


def safe_node(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(state: AgentState) -> dict:
        try:
            return fn(state)
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all -> responder
            label = f"{fn.__name__}: {exc}"
            return {"error": label, "trace": [f"[error] {label}"]}
    return wrapper


@safe_node
def supervisor_node(state: AgentState) -> dict:
    plan = nodes.supervisor(state["question"], state.get("history"))
    subs = plan.sub_questions
    cap = settings.max_sub_questions
    dropped = max(0, len(subs) - cap)
    subs = subs[:cap]
    trace = [f"supervisor: {len(subs)} sub-question(s) — {plan.reasoning}".rstrip(" —")]
    trace += [f"  [{i}] ({s.intent.value}) {s.text}" for i, s in enumerate(subs, 1)]
    if dropped:
        trace.append(f"  (+{dropped} part(s) over the cap of {cap} — will disclose)")
    return {"pending": list(subs), "results": [], "dropped": dropped, "trace": trace}


@safe_node
def extract_node(state: AgentState) -> dict:
    pending = list(state["pending"])
    sub = pending.pop(0)
    results = list(state["results"])

    if sub.intent not in DATA_INTENTS:
        results.append(BranchResult(text=sub.text, intent=sub.intent))
        return {"pending": pending, "results": results,
                "trace": [f"extract: ({sub.intent.value}) -> responder handles directly"]}

    out = nodes.extractor(sub, get_catalog())
    if isinstance(out, Clarification):
        results.append(BranchResult(text=sub.text, intent=sub.intent, clarification=out))
        line = f"extract: ({sub.intent.value}) -> clarify: {out.reason.value}"
    else:
        results.append(BranchResult(text=sub.text, intent=sub.intent,
                                    resolved=out, caveats=out.caveats,
                                    trace=list(out.trace)))
        line = f"extract: ({sub.intent.value}) -> {out.operation} | {out.timeframe_label}"
    return {"pending": pending, "results": results, "trace": [line]}


@safe_node
def analyst_node(state: AgentState) -> dict:
    results = list(state["results"])
    br = results[-1]
    data, headline = analyst.run(br.resolved, load_ledger(), get_catalog())
    br.data = data
    br.headline = headline
    tool_trace = list(getattr(data, "trace", []) or [])
    br.trace = list(br.trace) + tool_trace
    return {"results": results, "trace": [f"analyst: {headline}"]}


@safe_node
def responder_node(state: AgentState) -> dict:
    return responder.respond(state)


# --- routing -----------------------------------------------------------------


def _route_after_supervisor(state: AgentState) -> str:
    if state.get("error") or not state.get("pending"):
        return "responder"
    return "extract"


def _route_after_extract(state: AgentState) -> str:
    if state.get("error"):
        return "responder"
    if state["results"][-1].needs_analyst:
        return "analyst"
    return "extract" if state.get("pending") else "responder"


def _route_after_analyst(state: AgentState) -> str:
    if state.get("error"):
        return "responder"
    return "extract" if state.get("pending") else "responder"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("extract", extract_node)
    g.add_node("analyst", analyst_node)
    g.add_node("responder", responder_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", _route_after_supervisor,
                            {"extract": "extract", "responder": "responder"})
    g.add_conditional_edges("extract", _route_after_extract,
                            {"analyst": "analyst", "extract": "extract",
                             "responder": "responder"})
    g.add_conditional_edges("analyst", _route_after_analyst,
                            {"extract": "extract", "responder": "responder"})
    g.add_edge("responder", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
