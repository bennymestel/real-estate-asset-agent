"""The calculator. Turns one grounded ResolvedQuery into numbers by dispatching
to the pandas tools in src/tools. No LLM — same query, same result, every time.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from src.data.catalog import Catalog, get_catalog
from src.data.loader import load_ledger
from src.data.timeframe import parse_timeframe
from src.schemas import Breakdown, Bucket, ResolvedQuery
from src.tools.anomalies import scan
from src.tools.details import portfolio_card, property_card, tenant_card
from src.tools.metrics import breakdown, compare, pnl, timeseries, top_n

_TYPE_TO_WORD = {("revenue",): "revenue", ("expenses",): "expenses"}
_DIM_WORD = {"property_name": "property", "tenant_name": "tenant",
             "ledger_category": "category", "ledger_group": "group",
             "ledger_type": "type"}


def _metric_word(rq: ResolvedQuery) -> str:
    return _TYPE_TO_WORD.get(rq.query.ledger_types, "net P&L")


def _dim(name: str | None) -> str:
    return _DIM_WORD.get(name or "", name or "")


def _money(x: float) -> str:
    return f"€{x:,.2f}"


def _bucket_line(buckets: list[Bucket], limit: int = 6) -> str:
    shown = "; ".join(f"{b.key}: {_money(b.value)}" for b in buckets[:limit])
    return shown + (f"; (+{len(buckets) - limit} more)" if len(buckets) > limit else "")


def _periods(df: pd.DataFrame, rq: ResolvedQuery, cat: Catalog) -> Breakdown:
    buckets = []
    for label in rq.members:
        months = parse_timeframe(label, cat).months
        m = pnl(df, replace(rq.query, months=months), label=label)
        buckets.append(Bucket(label, m.value, m.rows))
    trace = [f"per-period {_metric_word(rq)}", f"members={list(rq.members)}"]
    return Breakdown("period", buckets, " vs ".join(rq.members), trace)


def run(rq: ResolvedQuery, df: pd.DataFrame | None = None,
        cat: Catalog | None = None) -> tuple[Any, str]:
    df = load_ledger() if df is None else df
    cat = cat or get_catalog()
    q, op, word = rq.query, rq.operation, _metric_word(rq)

    if op == "pnl":
        m = pnl(df, q, label=f"{word}, {rq.timeframe_label}")
        return m, f"{m.label} = {_money(m.value)} ({m.rows} rows)"

    if op == "breakdown":
        b = breakdown(df, q, rq.group_by, label=f"{word} by {_dim(rq.group_by)}", dropna=False)
        return b, f"{b.label} ({rq.timeframe_label}) — {_bucket_line(b.buckets)}"

    if op == "timeseries":
        by = rq.group_by if rq.group_by in ("month", "quarter", "year") else "month"
        b = timeseries(df, q, by=by, label=f"{word} by {by}")
        return b, f"{b.label} ({rq.timeframe_label}) — {_bucket_line(b.buckets, 12)}"

    if op == "top_n":
        n = rq.top_n or 5
        b = top_n(df, q, rq.group_by, n, rank_by=rq.rank_by,
                  label=f"top {n} {_dim(rq.group_by)} by {rq.rank_by}")
        return b, f"{b.label} ({rq.timeframe_label}) — {_bucket_line(b.buckets)}"

    if op == "compare":
        b = _periods(df, rq, cat) if rq.group_by == "period" \
            else compare(df, q, rq.group_by, list(rq.members),
                         label=f"{word}, {_dim(rq.group_by)}: {' vs '.join(rq.members)}")
        return b, f"{b.label} — {_bucket_line(b.buckets)}"

    if op == "details":
        card = _card(df, rq.subject, cat)
        span = (f"{card.months_active[0]}..{card.months_active[-1]}"
                if card.months_active else "no activity")
        return card, f"{card.title}: net P&L {_money(card.headline.value)}, active {span}"

    if op == "anomalies":
        findings = scan(df)
        head = "; ".join(f.summary for f in findings[:4]) or "no anomalies found"
        return findings, f"{len(findings)} data-quality finding(s): {head}"

    raise ValueError(f"analyst has no handler for operation {op!r}")


def _card(df: pd.DataFrame, subject: str | None, cat: Catalog):
    if subject and subject in cat.properties:
        return property_card(df, subject)
    if subject and subject in cat.tenants:
        return tenant_card(df, subject)
    return portfolio_card(df)
