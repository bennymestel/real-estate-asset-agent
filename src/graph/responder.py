"""The writer. Turns finished BranchResults into English via one LLM call, then
refuses any number it can't tie back to the analyst's payload.

Canned parts (capability, unsupported, out-of-scope, vague) never reach the LLM —
they are fixed, safe copy. A turn made only of those costs zero API calls.
"""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm, settings
from src.data.catalog import get_catalog
from src.graph import prompts
from src.graph.state import AgentState, BranchResult
from src.schemas import Breakdown, Intent, Metric
from src.tools.details import Card

_ERR = "Sorry — I hit an internal error ({err}). Try rephrasing the question."
_BUSY = ("The model is rate-limited right now. "
         "Give it a few seconds and ask again.")
# the provider's 429 body is a wall of quota ids, urls and retry timings; showing it
# raw makes an ordinary throttle look like a crash.
_BUSY_MARKERS = ("RESOURCE_EXHAUSTED", "429", "quota", "rate limit", "rate-limit")
# the analyst truncates long bucket lists for its trace line; the digest lists every
# bucket underneath, so leaving the marker in tells the writer data is missing.
_TRUNC = re.compile(r";\s*\(\+\d+ more\)")

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_TOL = 0.03  # relative slack for readable rounding / abbreviations
_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")
_ORDINAL = re.compile(r"^(?:st|nd|rd|th)?\s*(?:of\s+)?", re.I)


def _friendly_error(err: str) -> str:
    """A transient throttle is not an internal error — say so in one line."""
    return _BUSY if any(m in err for m in _BUSY_MARKERS) else _ERR.format(err=err)


def respond(state: AgentState) -> dict:
    if state.get("error"):
        return {"answer": _friendly_error(state["error"]), "trace": ["responder: error mode"]}

    results: list[BranchResult] = list(state.get("results") or [])
    llm_parts = [br for br in results if _needs_llm(br)]
    blocks: list[str] = []

    if llm_parts:
        digest = "\n\n".join(_part_block(br) for br in llm_parts)
        prose = _write(digest)
        if any(br.data is not None for br in llm_parts):
            allowed = _numbers(digest)
            bad = _ungrounded(prose, allowed)
            if bad:
                prose = _write(digest, retry_flag=bad)
                if _ungrounded(prose, allowed):
                    prose = _fallback(llm_parts)
                    prose += ("\n\n_(narration fell back to the raw figures — a generated "
                              "number could not be verified against the ledger)_")
        blocks.append(prose)

    blocks += [_canned(br) for br in results if not _needs_llm(br)]

    if state.get("dropped"):
        cap = settings.max_sub_questions
        blocks.append(f"_I answered the first {cap} part(s); {state['dropped']} more weren't "
                      "covered — ask again for those._")

    return {"answer": "\n\n".join(b for b in blocks if b),
            "trace": [f"responder: {_label(llm_parts)}"]}


def _needs_llm(br: BranchResult) -> bool:
    return (br.clarification is not None or br.data is not None
            or br.intent is Intent.general_knowledge)


def _label(llm_parts: list[BranchResult]) -> str:
    if not llm_parts:
        return "canned"
    if all(br.clarification is not None for br in llm_parts):
        return "clarify"
    if all(br.intent is Intent.general_knowledge for br in llm_parts):
        return "knowledge"
    return "answer"


# --- digest: what the LLM may draw on, and the source of the allowed numbers ---


def _part_block(br: BranchResult) -> str:
    if br.clarification is not None:
        c = br.clarification
        opts = f"\nOPTIONS: {' | '.join(c.options)}" if c.options else ""
        return f"[CLARIFY] {br.text}\nREASON: {c.detail}{opts}"
    if br.intent is Intent.general_knowledge:
        return f"[KNOWLEDGE] {br.text}"
    return f"[ANSWER] {br.text}\nDATA:\n{_data_digest(br)}"


def _data_digest(br: BranchResult) -> str:
    lines = [f"  {br.headline}"]
    d = br.data
    if isinstance(d, Metric):
        lines += [f"  {t}" for t in d.trace]
    elif isinstance(d, Breakdown):
        lines[0] = _TRUNC.sub("", lines[0])  # every bucket is listed below; nothing is elided
        lines.append(f"  total across buckets: €{d.total:,.2f}")
        lines += [f"  {b.key}: €{b.value:,.2f} ({b.rows} rows)" for b in d.buckets]
    elif isinstance(d, Card):
        lines.append(f"  net P&L: €{d.headline.value:,.2f} ({d.headline.rows} rows)")
        for name, bd in (("revenue/expenses", d.by_type), ("group", d.by_group),
                         ("top category", d.top_categories)):
            lines += [f"  {name} — {b.key}: €{b.value:,.2f}" for b in bd.buckets]
        if d.months_active:
            lines.append(f"  active {d.months_active[0]}..{d.months_active[-1]} "
                         f"({len(d.months_active)} months)")
        lines += [f"  note: {n}" for n in d.notes]
    elif isinstance(d, list):  # findings
        lines += [f"  {f.summary} (magnitude €{f.magnitude:,.2f}) — {f.detail}" for f in d]
    lines += [f"  caveat: {c}" for c in br.caveats]
    return "\n".join(lines)


# --- the LLM call ------------------------------------------------------------


def _write(digest: str, retry_flag: list[str] | None = None) -> str:
    msg = digest
    if retry_flag:
        msg += ("\n\n---\nYour previous draft used numbers that are NOT in the DATA above: "
                f"{', '.join(retry_flag)}. Rewrite using only the figures given.")
    resp = get_llm("smart").invoke(
        [SystemMessage(prompts.responder_system()), HumanMessage(msg)]
    )
    # resp.content can be a list of blocks (text + thinking/signature) rather than
    # a plain string; .text pulls out only the text blocks.
    return resp.text.strip()


def _fallback(llm_parts: list[BranchResult]) -> str:
    out: list[str] = []
    for br in llm_parts:
        if br.clarification is not None:
            c = br.clarification
            txt = f"**{br.text}**\n{c.detail}."
            if c.options:
                txt += "\nOptions: " + "; ".join(c.options) + "."
            out.append(txt)
        elif br.data is not None:
            txt = br.headline
            for cav in br.caveats:
                txt += f"\n_· {cav}_"
            out.append(txt)
        else:
            out.append(f'_(general knowledge on "{br.text}" is unavailable right now)_')
    return "\n\n".join(out)


# --- numeric grounding check ------------------------------------------------


def _numbers(text: str) -> set[float]:
    out: set[float] = set()
    for tok in _NUM.findall(text):
        try:
            out.add(round(abs(float(tok.replace(",", ""))), 2))
        except ValueError:
            continue
    return out


def _ungrounded(prose: str, allowed: set[float]) -> list[str]:
    bad: list[str] = []
    for m in _NUM.finditer(prose):
        tok = m.group()
        nxt = prose[m.end():m.end() + 1]
        if nxt == "%":
            continue
        raw = tok.replace(",", "")
        try:
            v = abs(float(raw))
        except ValueError:
            continue
        if v == 0:
            continue
        if raw.isdigit() and (1900 <= int(raw) <= 2100 or int(raw) <= 12):
            continue  # a year, or a small count / ordinal
        if _is_date_day(prose, m.start(), m.end(), raw):
            continue  # the day in a spelled-out date ("31 March 2025")
        v *= _scale(prose, m.end(), nxt)
        if not _grounded(v, allowed):
            bad.append(tok)
    return bad


def _is_date_day(prose: str, start: int, end: int, raw: str) -> bool:
    """True for the 13..31 in "31 March 2025" / "March 31, 2025" — a date, not a figure.

    Days 1..12 already pass as small counts; without this the responder's own
    reporting date ("as of 31 March 2025") is rejected as an invented number and
    every narrated comparison falls back to raw figures.
    """
    if not raw.isdigit() or not 1 <= int(raw) <= 31:
        return False
    after = _ORDINAL.sub("", prose[end:end + 16]).lower()
    if after.startswith(_MONTHS):
        return True
    before = prose[max(0, start - 16):start].lower().rstrip()
    return before.endswith(_MONTHS)


def _scale(prose: str, end: int, nxt: str) -> int:
    if nxt in ("k", "K"):
        return 1_000
    if nxt in ("m", "M") and not prose[end + 1:end + 2].isalpha():
        return 1_000_000
    word = prose[end:end + 12].lower().lstrip()
    if word.startswith("thousand"):
        return 1_000
    if word.startswith("million"):
        return 1_000_000
    if word.startswith("billion"):
        return 1_000_000_000
    return 1


def _grounded(v: float, allowed: set[float]) -> bool:
    for scale in (1, 1_000, 1_000_000):
        for a in allowed:
            if a and abs(v * scale - a) <= max(1.0, _TOL * a):
                return True
    return False


# --- canned parts: no LLM ---------------------------------------------------


def _canned(br: BranchResult) -> str:
    if br.intent is Intent.capability:
        return "Here's what I have to work with:\n\n" + get_catalog().schema_card()
    if br.intent is Intent.unsupported:
        return ("That needs data this ledger doesn't hold (valuation, cap rate, occupancy, "
                "address, floor area or lease terms). I can show net P&L, revenue, expenses, "
                "comparisons, rankings, breakdowns and a data-quality scan.")
    if br.intent is Intent.out_of_scope:
        return ("I'm an asset-management assistant for this portfolio, so I can't help with "
                "that — but ask me about P&L, tenants, expenses or anomalies in the ledger.")
    return ("Could you be more specific? I can show P&L for a property, tenant or period, "
            "compare two of them, rank tenants or categories, break a total down, or scan "
            "the ledger for data-quality issues.")
