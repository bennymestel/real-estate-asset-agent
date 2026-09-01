"""'Tell me about X' cards — one bundle of the common facts about a property,
a tenant, or the whole portfolio. Pure pandas; the responder narrates it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.schemas import Breakdown, LedgerQuery, Metric
from src.tools.metrics import breakdown, pnl


@dataclass(frozen=True)
class Card:
    title: str
    headline: Metric
    by_type: Breakdown          # revenue vs expenses
    by_group: Breakdown         # ledger_group split
    top_categories: Breakdown   # biggest ledger_category lines by absolute value
    months_active: tuple[str, ...]
    notes: list[str] = field(default_factory=list)


def _card(df: pd.DataFrame, q: LedgerQuery, title: str, notes: list[str]) -> Card:
    rows = q.apply(df)
    cats = breakdown(df, q, "ledger_category")
    cats.buckets.sort(key=lambda b: abs(b.value), reverse=True)
    months = tuple(rows.sort_values("period_ord")["month"].unique().tolist())
    return Card(
        title=title,
        headline=pnl(df, q, label=f"{title} — net P&L"),
        by_type=breakdown(df, q, "ledger_type", label="revenue vs expenses"),
        by_group=breakdown(df, q, "ledger_group", label="by ledger group"),
        top_categories=Breakdown("ledger_category", cats.buckets[:5], "top categories", cats.trace),
        months_active=months,
        notes=notes,
    )


def property_card(df: pd.DataFrame, name: str) -> Card:
    return _card(df, LedgerQuery(properties=(name,)), name,
                 ["excludes entity-level overhead (rows with no property)"])


def tenant_card(df: pd.DataFrame, name: str) -> Card:
    return _card(df, LedgerQuery(tenants=(name,)), name, [])


def portfolio_card(df: pd.DataFrame) -> Card:
    return _card(df, LedgerQuery(), "Portfolio",
                 ["includes entity-level overhead in the total"])
