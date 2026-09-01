"""Resolve a timeframe phrase to concrete months, relative to the pinned
reporting-as-of date (settings.reporting_as_of), not the wall clock.
"""
from __future__ import annotations

import re

from src.config import settings
from src.data.catalog import Catalog
from src.schemas import TimeRange

_QUARTER_RE = re.compile(r"(\d{4})[-\s]?q([1-4])", re.I)
_MONTH_RE = re.compile(r"(\d{4})[-\s]?m(\d{1,2})", re.I)
_YEAR_RE = re.compile(r"(?:fy\s*)?(\d{4})")
_LAST_N_RE = re.compile(r"(?:last|trailing|past)\s+(\d{1,2})\s+months?", re.I)

_YEAR_Q_RE = re.compile(r"\b(\d{4})[-\s]?q([1-4])\b", re.I)
_Q_YEAR_RE = re.compile(r"\bq([1-4])[-\s]?(\d{4})\b", re.I)
_YM_TOKEN_RE = re.compile(r"\b(\d{4})[-\s]?m(\d{1,2})\b", re.I)
_BARE_Q_RE = re.compile(r"\bq([1-4])\b", re.I)
_BARE_YEAR_RE = re.compile(r"\b(\d{4})\b")

_ALL = {"", "all", "all time", "all periods", "overall", "to date", "lifetime"}


def parse_timeframe(phrase: str, cat: Catalog) -> TimeRange:
    p = (phrase or "").strip().lower()
    as_of = settings.reporting_as_of
    year = as_of.year
    quarter = (as_of.month - 1) // 3 + 1

    if p in _ALL:
        return TimeRange.all_time(cat)

    m = _QUARTER_RE.fullmatch(p)
    if m:
        return TimeRange.quarter(f"{m.group(1)}-Q{m.group(2)}", cat)

    m = _MONTH_RE.fullmatch(p)
    if m and 1 <= int(m.group(2)) <= 12:
        mo = f"{m.group(1)}-M{int(m.group(2)):02d}"
        return TimeRange.from_months([mo], cat, mo)

    if p in {"this year", "current year", "ytd", "year to date"}:
        return TimeRange.year(year, cat)
    if p in {"last year", "prior year", "previous year"}:
        return TimeRange.year(year - 1, cat)
    if p in {"this quarter", "current quarter", "latest quarter", "most recent quarter"}:
        return TimeRange.quarter(f"{year}-Q{quarter}", cat)
    if p in {"last quarter", "prior quarter", "previous quarter"}:
        y, q = (year, quarter - 1) if quarter > 1 else (year - 1, 4)
        return TimeRange.quarter(f"{y}-Q{q}", cat)
    if p in {"same period last year", "same quarter last year", "year ago"}:
        return TimeRange.quarter(f"{year - 1}-Q{quarter}", cat)
    if p in {"last month", "latest month", "most recent month"}:
        return TimeRange.from_months([cat.months[-1]], cat, cat.months[-1])

    m = _LAST_N_RE.fullmatch(p)
    if m:
        n = int(m.group(1))
        return TimeRange.from_months(list(cat.months[-n:]), cat, f"last {n} months")

    m = _YEAR_RE.fullmatch(p)
    if m:
        return TimeRange.year(m.group(1), cat)

    return TimeRange(
        tuple(cat.months), "all periods",
        f"could not read timeframe '{phrase}', used all available data",
    )


def find_timeframes(text: str, cat: Catalog) -> list[str]:
    """Every explicit period token in a free-text sub-question, in order.
    Bare quarters ('Q1 and Q2') snap to the most recent year that covers them.
    """
    t = text or ""
    hits: list[tuple[int, str]] = []
    spans: list[tuple[int, int]] = []
    for m in _YEAR_Q_RE.finditer(t):
        hits.append((m.start(), f"{m.group(1)}-Q{m.group(2)}"))
        spans.append(m.span())
    for m in _Q_YEAR_RE.finditer(t):
        hits.append((m.start(), f"{m.group(2)}-Q{m.group(1)}"))
        spans.append(m.span())
    for m in _YM_TOKEN_RE.finditer(t):
        if 1 <= int(m.group(2)) <= 12:
            hits.append((m.start(), f"{m.group(1)}-M{int(m.group(2)):02d}"))
        spans.append(m.span())

    def free(i: int) -> bool:
        return not any(a <= i < b for a, b in spans)

    bare_q = [m for m in _BARE_Q_RE.finditer(t) if free(m.start())]
    if bare_q:
        year = _common_quarter_year([m.group(1) for m in bare_q], cat)
        hits += [(m.start(), f"{year}-Q{m.group(1)}") for m in bare_q]
    hits += [(m.start(), m.group(1)) for m in _BARE_YEAR_RE.finditer(t) if free(m.start())]

    out: list[str] = []
    for _, label in sorted(hits):
        if label not in out:
            out.append(label)
    return out


def _common_quarter_year(quarters: list[str], cat: Catalog) -> str:
    for y in sorted(set(cat.month_year.values()), reverse=True):
        if all(f"{y}-Q{q}" in cat.quarters for q in quarters):
            return y
    return cat.years[-1]


def compare_periods(spec_timeframes: list[str], text: str, cat: Catalog) -> list[TimeRange]:
    """The two+ periods a comparison sub-question is asking about, resolved."""
    out: list[TimeRange] = []
    seen: set[str] = set()
    for label in list(spec_timeframes) + find_timeframes(text, cat):
        tr = parse_timeframe(label, cat)
        unreadable = tr.note and "could not read" in tr.note
        if tr.months and not unreadable and tr.label not in seen:
            out.append(tr)
            seen.add(tr.label)
    return out
