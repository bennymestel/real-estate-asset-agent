"""Pandas aggregations over the ledger. No LLM. Every number is sum(profit).

pnl        - one signed total
breakdown  - that total split by a column
timeseries - that total per period (month/quarter/year)
top_n      - breakdown, ranked, truncated
compare    - breakdown restricted to named members, in the order asked
"""
from __future__ import annotations

import pandas as pd

from src.schemas import Breakdown, Bucket, LedgerQuery, Metric


def _round(x: float) -> float:
    return round(float(x), 2)


def pnl(df: pd.DataFrame, q: LedgerQuery, *, label: str = "net P&L") -> Metric:
    rows = q.apply(df)
    total = _round(rows["profit"].sum())
    trace = [
        f"filter: {q.describe()}",
        f"matched {len(rows)} rows",
        f"sum(profit) = {total:,.2f}",
    ]
    return Metric(total, len(rows), label, trace)


def breakdown(df: pd.DataFrame, q: LedgerQuery, by: str, *,
              label: str | None = None, dropna: bool = False) -> Breakdown:
    rows = q.apply(df)
    grp = rows.groupby(by, dropna=dropna)["profit"]
    buckets = [
        Bucket(key="(entity-level)" if pd.isna(k) else str(k),
               value=_round(v), rows=int(grp.size()[k]))
        for k, v in grp.sum().items()
    ]
    buckets.sort(key=lambda b: b.value, reverse=True)
    trace = [
        f"filter: {q.describe()}",
        f"group by {by}: {len(buckets)} bucket(s), {len(rows)} rows",
    ]
    return Breakdown(by, buckets, label or f"by {by}", trace)


def timeseries(df: pd.DataFrame, q: LedgerQuery, *, by: str = "month",
               label: str | None = None) -> Breakdown:
    sort_key = {
        "month": ["period_ord"],
        "quarter": ["year_num", "quarter_num"],
        "year": ["year_num"],
    }[by]
    rows = q.apply(df).sort_values(sort_key)
    grp = rows.groupby(by, sort=False)["profit"]
    buckets = [Bucket(str(k), _round(v), int(grp.size()[k])) for k, v in grp.sum().items()]
    trace = [f"filter: {q.describe()}", f"{len(buckets)} period(s) by {by}"]
    return Breakdown(by, buckets, label or f"over time by {by}", trace)


def top_n(df: pd.DataFrame, q: LedgerQuery, by: str, n: int = 5, *,
          label: str | None = None, dropna: bool = True,
          rank_by: str = "value") -> Breakdown:
    full = breakdown(df, q, by, dropna=dropna)
    buckets = full.buckets
    if rank_by == "magnitude":
        buckets = sorted(buckets, key=lambda b: abs(b.value), reverse=True)
    top = buckets[:n]
    return Breakdown(by, top, label or f"top {n} by {by}",
                     full.trace + [f"kept top {len(top)} of {len(full.buckets)} by {rank_by}"])


def compare(df: pd.DataFrame, q: LedgerQuery, by: str, members: list[str], *,
            label: str | None = None) -> Breakdown:
    full = breakdown(df, q, by)
    found = {b.key: b for b in full.buckets}
    ordered = [found.get(m, Bucket(m, 0.0, 0)) for m in members]
    trace = full.trace + [f"compared {by}: {members}"]
    return Breakdown(by, ordered, label or f"{by}: {' vs '.join(members)}", trace)
