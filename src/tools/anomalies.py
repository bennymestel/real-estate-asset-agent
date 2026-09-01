"""Data-quality scan. Surfaces oddities in the ledger instead of silently
'correcting' them, so every headline number still ties back to the raw file.

reversal          - a +X and -X posting in the same property/tenant/category/month
double_mapped_code - one ledger_code that fans out to two ledger_category values
concentration     - one tenant is an outsized share of tenant-attributed profit
coverage          - the ledger is a partial period
"""
from __future__ import annotations

import pandas as pd

from src.schemas import Finding

_REVERSAL_MIN = 10_000.0      # ignore small equal-and-opposite noise
_CONCENTRATION = 0.30         # flag a tenant above this share of tenant profit


def _reversals(df: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    for (p, t, c), g in df.groupby(["property_name", "tenant_name", "ledger_category"], dropna=False):
        hits: list[tuple[str, float]] = []
        for m, gm in g.groupby("month"):
            vals = gm["profit"].tolist()
            pos = [v for v in vals if v >= _REVERSAL_MIN]
            neg = [v for v in vals if v <= -_REVERSAL_MIN]
            hits += [(m, round(v, 2)) for v in pos if any(abs(v + n) < 0.01 for n in neg)]
        if hits:
            out.append(Finding(
                "reversal",
                f"{'entity-level' if pd.isna(p) else p} / {'—' if pd.isna(t) else t} / {c}: "
                f"{len(hits)} equal-and-opposite posting(s)",
                max(v for _, v in hits),
                "; ".join(f"{m} ±{v:,.2f}" for m, v in hits),
            ))
    return sorted(out, key=lambda f: f.magnitude, reverse=True)


def _double_mapped_codes(df: pd.DataFrame) -> list[Finding]:
    out: list[Finding] = []
    for code, g in df.groupby("ledger_code"):
        cats = sorted(g["ledger_category"].dropna().unique().tolist())
        if len(cats) > 1:
            out.append(Finding(
                "double_mapped_code",
                f"ledger_code {code} maps to {len(cats)} categories: {', '.join(cats)}",
                round(float(g["profit"].sum()), 2),
                f"{len(g)} rows; likely double-counts one set of postings",
            ))
    return out


def _concentration(df: pd.DataFrame) -> list[Finding]:
    by_tenant = df.dropna(subset=["tenant_name"]).groupby("tenant_name")["profit"].sum()
    pos = by_tenant[by_tenant > 0]
    if pos.empty:
        return []
    share = pos / pos.sum()
    top = share.idxmax()
    if share[top] >= _CONCENTRATION:
        return [Finding("concentration",
                        f"{top} is {share[top]:.0%} of tenant-attributed profit",
                        round(float(pos[top]), 2),
                        "revenue concentration risk")]
    return []


def _coverage(df: pd.DataFrame) -> list[Finding]:
    years = df.groupby("year")["month"].nunique()
    return [
        Finding("coverage",
                f"{y} is partial: {n} month(s) of data",
                float(n),
                "year-over-year comparisons need a like-for-like basis")
        for y, n in years.items() if n < 12
    ]


def scan(df: pd.DataFrame) -> list[Finding]:
    return (
        _reversals(df)
        + _double_mapped_codes(df)
        + _concentration(df)
        + _coverage(df)
    )
