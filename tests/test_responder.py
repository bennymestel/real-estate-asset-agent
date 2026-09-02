"""The responder in isolation: the numeric grounding check, the repair retry,
the fallback, and the canned/no-LLM path. The LLM call (_write) is always mocked.
"""
import pytest

from src.graph import responder
from src.graph.state import BranchResult
from src.schemas import Intent, Metric

_HEADLINE = "net P&L, 2025-Q1 = €361,810.32 (147 rows)"
_METRIC = Metric(361_810.32, 147, "net P&L, 2025-Q1",
                 ["filter: months=2025-M01..2025-M03", "matched 147 rows",
                  "sum(profit) = 361,810.32"])


def _pnl_state():
    br = BranchResult(text="P&L this year", intent=Intent.pnl_metric,
                      data=_METRIC, headline=_HEADLINE,
                      caveats=("2025 is partial (3 month(s) in ledger)",))
    return {"results": [br]}


def test_clean_prose_passes_through(monkeypatch):
    monkeypatch.setattr(responder, "_write",
                        lambda d, retry_flag=None: "Net P&L for 2025-Q1 was €361,810.32.")
    out = responder.respond(_pnl_state())
    assert out["answer"] == "Net P&L for 2025-Q1 was €361,810.32."
    assert "fell back" not in out["answer"]


def test_invented_number_triggers_retry_then_succeeds(monkeypatch):
    calls = []

    def fake(digest, retry_flag=None):
        calls.append(retry_flag)
        return "It was €999,000.00." if retry_flag is None else "It was €361,810.32."

    monkeypatch.setattr(responder, "_write", fake)
    out = responder.respond(_pnl_state())
    assert len(calls) == 2 and calls[1] == ["999,000.00"]
    assert "361,810.32" in out["answer"]
    assert "fell back" not in out["answer"]


def test_persistent_hallucination_falls_back(monkeypatch):
    monkeypatch.setattr(responder, "_write",
                        lambda d, retry_flag=None: "The portfolio earned €5,000,000 this year.")
    out = responder.respond(_pnl_state())
    assert "fell back" in out["answer"]
    assert "361,810.32" in out["answer"]          # the real headline
    assert "2025 is partial" in out["answer"]     # caveat preserved


def test_canned_parts_never_call_the_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM was called for a canned part")

    monkeypatch.setattr(responder, "_write", boom)
    for intent, needle in [(Intent.capability, "LEDGER SCHEMA"),
                           (Intent.unsupported, "valuation"),
                           (Intent.out_of_scope, "asset-management assistant"),
                           (Intent.vague, "be more specific")]:
        out = responder.respond({"results": [BranchResult(text="q", intent=intent)]})
        assert needle in out["answer"]


def test_error_mode(monkeypatch):
    monkeypatch.setattr(responder, "_write", lambda *a, **k: pytest.fail("no LLM in error mode"))
    out = responder.respond({"error": "analyst: boom", "results": []})
    assert "internal error" in out["answer"] and "analyst: boom" in out["answer"]


def test_dropped_parts_disclosed(monkeypatch):
    monkeypatch.setattr(responder, "_write", lambda d, retry_flag=None: "€361,810.32 for Q1.")
    state = _pnl_state()
    state["dropped"] = 2
    out = responder.respond(state)
    assert "2 more weren't covered" in out["answer"]


# --- the check's own unit behaviour ---------------------------------------


def test_numbers_and_ungrounded():
    allowed = responder._numbers(f"{_HEADLINE} sum(profit) = 361,810.32")
    assert responder._ungrounded("about €361,810 in Q1 2025", allowed) == []
    assert responder._ungrounded("roughly €0.36 million", allowed) == []      # abbreviation
    assert responder._ungrounded("that is 47% of revenue", allowed) == []     # percentage skipped
    assert responder._ungrounded("we booked €5,000,000", allowed) == ["5,000,000"]
    assert responder._ungrounded("147 rows matched", allowed) == []           # count is in payload


def test_a_spelled_date_is_not_an_invented_figure():
    """The responder names its own reporting date; the day must not be rejected."""
    allowed = responder._numbers(f"{_HEADLINE} sum(profit) = 361,810.32")
    assert responder._ungrounded("as of 31 March 2025", allowed) == []
    assert responder._ungrounded("March 31, 2025", allowed) == []
    assert responder._ungrounded("the 21st of June 2024", allowed) == []
    assert responder._ungrounded("we booked €31,000", allowed) == ["31,000"]
    assert responder._ungrounded("31 properties", allowed) == ["31"]
