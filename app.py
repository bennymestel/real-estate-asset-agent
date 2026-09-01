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


def render_result(br: BranchResult) -> None:
    d = br.data
    if d is None:
        return
    if isinstance(d, Metric):
        pass  # the narrated answer already carries the one number; nothing extra to show
    elif isinstance(d, Breakdown):
        df = pd.DataFrame([{"": b.key, "value": b.value, "rows": b.rows} for b in d.buckets])
        with st.expander(f"📊 {br.headline}"):
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.altair_chart(
                alt.Chart(df).mark_bar().encode(x=alt.X(":N", sort=None), y="value:Q"),
                use_container_width=True,
            )
    elif isinstance(d, Card):
        with st.expander(f"📇 {br.headline}"):
            for name, bd in (("Revenue vs expenses", d.by_type), ("By group", d.by_group),
                             ("Top categories", d.top_categories)):
                st.markdown(f"**{name}**")
                st.dataframe(
                    pd.DataFrame([{"": b.key, "value": b.value, "rows": b.rows} for b in bd.buckets]),
                    hide_index=True, use_container_width=True,
                )
            if d.notes:
                st.caption(" · ".join(d.notes))
    elif isinstance(d, list) and d and isinstance(d[0], Finding):
        with st.expander(f"🔎 {br.headline}"):
            st.dataframe(
                pd.DataFrame([{"kind": f.kind, "summary": f.summary,
                              "magnitude": f.magnitude, "detail": f.detail} for f in d]),
                hide_index=True, use_container_width=True,
            )
    for c in br.caveats:
        st.caption(f"⚠️ {c}")


def render_extras(msg: dict) -> None:
    if msg.get("dropped"):
        st.warning(f"I answered the first {settings.max_sub_questions} part(s); "
                   f"{msg['dropped']} more weren't covered — ask again for those.")
    for br in msg.get("results", []):
        render_result(br)
    if msg.get("trace"):
        with st.expander("🧭 Agent trace"):
            st.code("\n".join(msg["trace"]), language=None)


# --- transcript ---------------------------------------------------------------


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_extras(msg)

if not st.session_state.messages:
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, use_container_width=True, key=f"ex_{i}"):
            st.session_state.pending_question = ex


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
