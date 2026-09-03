"""Streamlit chat UI for the real-estate asset agent.

Thin: it owns no business logic — every turn is one `graph.invoke()` call.
Session state holds the chat transcript and a LangGraph thread_id, so
MemorySaver gives real follow-up memory ("...and what about last year?").
"""
from __future__ import annotations

import uuid

import altair as alt
import pandas as pd
import streamlit as st

from src.config import settings
from src.graph.build import build_graph
from src.graph.state import BranchResult
from src.schemas import Breakdown, Finding, Metric
from src.tools.details import Card

EXAMPLES = [
    "What is the total P&L for all my properties this year?",
    "Compare Building 120 and Building 180",
    "How does this quarter compare to the same period last year?",
    "Who are my top tenants, and is anything unusual in the numbers?",
    "What is the price of my asset at 123 Main St?",
    "How did Building 999 do?",
    "What data do you have?",
    "How is NOI calculated?",
]

st.set_page_config(page_title="Real Estate Asset Agent", page_icon="🏢", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph()


if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🏢 Real Estate Asset Agent")
st.caption(
    f"Reporting period: as of {settings.reporting_as_of:%d %b %Y} — every relative "
    "date (\"this year\", \"last quarter\") resolves against this, since the ledger "
    "closes there."
)

if not settings.google_api_key:
    st.error("`GOOGLE_API_KEY` is not set — add it to `.env` or the app's secrets.")
    st.stop()


# --- rendering helpers -------------------------------------------------------


DIM_LABEL = {
    "property_name": "Property", "tenant_name": "Tenant", "ledger_type": "Type",
    "ledger_group": "Group", "ledger_category": "Category", "month": "Month",
    "quarter": "Quarter", "year": "Year", "period": "Period",
}

_MONEY = st.column_config.NumberColumn("Value (EUR)", format="euro")


def _get(obj, name, default=None):
    """Read a field off a payload. These arrive as dataclasses; a checkpoint
    round-trip can hand them back as plain dicts, so read either shape."""
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _kind(d) -> str:
    """What the analyst returned, by structure rather than by identity — an
    isinstance check fails the moment the object survives a round-trip."""
    if isinstance(d, list):
        return "findings" if d and _get(d[0], "kind") is not None else ""
    if isinstance(d, (Card, Metric, Breakdown)):
        return {Card: "card", Metric: "metric", Breakdown: "breakdown"}[type(d)]
    if _get(d, "by_type") is not None:
        return "card"
    if _get(d, "buckets") is not None:
        return "breakdown"
    if _get(d, "value") is not None:
        return "metric"
    return ""


def bucket_frame(buckets, dimension: str) -> tuple[pd.DataFrame, str]:
    """One tidy frame per breakdown. The dimension gets a real column name —
    Altair cannot encode an unnamed field, and the table reads better for it."""
    col = DIM_LABEL.get(dimension, dimension.replace("_", " ").title() or "Key")
    df = pd.DataFrame([{col: _get(b, "key"), "value": _get(b, "value"),
                        "rows": _get(b, "rows")} for b in buckets])
    return df, col


def render_breakdown(bd, title: str, chart: bool = True) -> None:
    df, col = bucket_frame(_get(bd, "buckets", []), _get(bd, "dimension", ""))
    if df.empty:
        return
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"value": _MONEY, "rows": "Rows"})
    if chart and len(df) > 1:
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X(f"{col}:N", sort=None, title=None),
                y=alt.Y("value:Q", title="EUR"),
                tooltip=[col, "value", "rows"],
            ).properties(title=title),
            use_container_width=True,
        )


def render_result(br: BranchResult, answer: str = "") -> None:
    d = _get(br, "data")
    kind = _kind(d)
    if kind == "metric":
        pass  # the narrated answer already carries the one number; nothing extra to show
    elif kind == "breakdown":
        resolved = _get(br, "resolved")
        period = _get(resolved, "timeframe_label", "") if resolved else ""
        label = str(_get(d, "label", "Breakdown")).capitalize()
        with st.expander(f"📊 {label}" + (f" — {period}" if period else "")):
            render_breakdown(d, label)
    elif kind == "card":
        with st.expander(f"📇 {_get(d, 'title', 'Details')}"):
            for name, key in (("Revenue vs expenses", "by_type"), ("By group", "by_group"),
                              ("Top categories", "top_categories")):
                st.markdown(f"**{name}**")
                render_breakdown(_get(d, key), name, chart=False)
            if _get(d, "notes"):
                st.caption(" · ".join(_get(d, "notes")))
    elif kind == "findings":
        with st.expander(f"🔎 {len(d)} data-quality finding(s)"):
            st.dataframe(
                pd.DataFrame([{"Kind": _get(f, "kind"), "Summary": _get(f, "summary"),
                              "Magnitude": _get(f, "magnitude"),
                              "Detail": _get(f, "detail")} for f in d]),
                hide_index=True, use_container_width=True,
                column_config={"Magnitude": st.column_config.NumberColumn(format="euro")},
            )
    elif _get(br, "headline"):
        # unrecognised payload shape: the headline is a plain string and always
        # survives, so the figures still reach the reader.
        st.caption(_get(br, "headline"))
    # the writer is asked to weave caveats into the prose; only surface the ones
    # it left out, so the same sentence isn't printed twice.
    low = answer.lower()
    for c in _get(br, "caveats", ()) or ():
        if c.lower().strip(" .") not in low:
            st.caption(f"⚠️ {c}")


def render_extras(msg: dict) -> None:
    if msg.get("dropped"):
        st.warning(f"I answered the first {settings.max_sub_questions} part(s); "
                   f"{msg['dropped']} more weren't covered — ask again for those.")
    for br in msg.get("results", []):
        render_result(br, msg.get("content", ""))
    if msg.get("trace"):
        with st.expander("🧭 Agent trace"):
            st.code("\n".join(msg["trace"]), language=None)


# --- transcript ---------------------------------------------------------------


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_extras(msg)

with st.sidebar:
    st.markdown("**Example prompts**")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, use_container_width=True, key=f"ex_{i}"):
            st.session_state.pending_question = ex
            st.rerun()


# --- one turn -------------------------------------------------------------


question = st.chat_input("Ask about P&L, tenants, comparisons, or anomalies…")
if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
    cfg = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Working it out…"):
            state = get_graph().invoke(
                {"question": question, "history": history, "pending": [],
                 "results": [], "trace": []},
                cfg,
            )
        answer = state.get("answer", "(no answer)")
        st.markdown(answer)
        msg = {"role": "assistant", "content": answer,
              "results": state.get("results", []), "trace": state.get("trace", []),
              "dropped": state.get("dropped", 0)}
        render_extras(msg)

    st.session_state.messages.append(msg)
