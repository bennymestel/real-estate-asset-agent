"""Deterministic free-text -> canonical value resolution (rapidfuzz).

The extractor calls this to turn "b120" / "the oak building" into "Building 120"
or an explicit "not found, here are the real options" result. No LLM involved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from src.config import settings
from src.data.catalog import get_catalog


@dataclass(frozen=True)
class Resolution:
    query: str
    value: str | None          # canonical value, or None if unresolved
    score: float               # 0-100 match confidence
    candidates: list[str]      # top choices, for a clarifying question

    @property
    def ok(self) -> bool:
        return self.value is not None


def _normalise(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[_\-]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    # "b120" / "bldg 120" / "building120" -> "building 120"
    t = re.sub(r"\b(?:b|bldg|building)\s*0*(\d{1,3})\b", r"building \1", t)
    # "t7" / "tenant7" -> "tenant 7"
    t = re.sub(r"\b(?:t|tenant)\s*0*(\d{1,3})\b", r"tenant \1", t)
    return t


def resolve(query: str, choices: list[str], *, threshold: int | None = None,
            limit: int = 3) -> Resolution:
    thr = settings.fuzzy_threshold if threshold is None else threshold
    norm_query = _normalise(query)
    norm_map: dict[str, str] = {_normalise(c): c for c in choices}

    if norm_query in norm_map:
        return Resolution(query, norm_map[norm_query], 100.0, [norm_map[norm_query]])

    # bare number: "120" -> "building 120" if unambiguous
    if norm_query.isdigit():
        hits = [c for n, c in norm_map.items() if n.endswith(f" {norm_query}") or n == norm_query]
        if len(hits) == 1:
            return Resolution(query, hits[0], 100.0, [hits[0]])

    ranked = process.extract(
        norm_query, list(norm_map.keys()),
        scorer=fuzz.WRatio, limit=max(limit, 3),
    )
    candidates = [norm_map[name] for name, _score, _idx in ranked]
    best_name, best_score, _ = ranked[0]
    if best_score >= thr:
        return Resolution(query, norm_map[best_name], best_score, candidates)
    return Resolution(query, None, best_score, candidates)


def resolve_property(query: str, **kw) -> Resolution:
    return resolve(query, get_catalog().properties, **kw)


def resolve_tenant(query: str, **kw) -> Resolution:
    return resolve(query, get_catalog().tenants, **kw)


def resolve_ledger_category(query: str, **kw) -> Resolution:
    return resolve(query, get_catalog().ledger_categories, **kw)


def resolve_ledger_group(query: str, **kw) -> Resolution:
    return resolve(query, get_catalog().ledger_groups, **kw)
