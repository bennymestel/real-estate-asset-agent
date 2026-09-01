"""Shared data-holder types for the deterministic tool layer.

TimeRange = which months. LedgerQuery = which rows. Metric/Bucket/Breakdown/Finding
= what a tool hands back. LLM-facing models (Intent, QuerySpec) join this file in step 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.data.catalog import Catalog


@dataclass(frozen=True)
class TimeRange:
    months: tuple[str, ...]
    label: str
    note: str | None = None

    @classmethod
    def all_time(cls, cat: Catalog) -> "TimeRange":
        return cls(tuple(cat.months), "all periods")

    @classmethod
    def year(cls, y: int | str, cat: Catalog) -> "TimeRange":
        months = tuple(cat.months_in_year(y))
        note = None if len(months) == 12 else f"{y} is partial ({len(months)} month(s) in ledger)"
        return cls(months, str(y), note)

    @classmethod
    def quarter(cls, q: str, cat: Catalog) -> "TimeRange":
        return cls(tuple(cat.months_in_quarter(q)), q)

    @classmethod
    def from_months(cls, months: list[str], cat: Catalog, label: str | None = None) -> "TimeRange":
        valid = tuple(m for m in cat.months if m in set(months))
        return cls(valid, label or (f"{valid[0]}..{valid[-1]}" if valid else "no periods"))


@dataclass(frozen=True)
class LedgerQuery:
    properties: tuple[str, ...] = ()
    tenants: tuple[str, ...] = ()
    ledger_types: tuple[str, ...] = ()
    ledger_groups: tuple[str, ...] = ()
    ledger_categories: tuple[str, ...] = ()
    months: tuple[str, ...] = ()
    include_entity_level: bool = True

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series(True, index=df.index)
        for col, vals in (
            ("property_name", self.properties),
            ("tenant_name", self.tenants),
            ("ledger_type", self.ledger_types),
            ("ledger_group", self.ledger_groups),
            ("ledger_category", self.ledger_categories),
            ("month", self.months),
        ):
            if vals:
                mask &= df[col].isin(vals)
        if not self.include_entity_level:
            mask &= ~df["is_entity_level"]
        return df[mask]

    def describe(self) -> str:
        parts = [
            f"{name}={list(v)}"
            for name, v in (
                ("properties", self.properties), ("tenants", self.tenants),
                ("ledger_types", self.ledger_types), ("ledger_groups", self.ledger_groups),
                ("ledger_categories", self.ledger_categories),
            ) if v
        ]
        if self.months:
            parts.append(f"months={self.months[0]}..{self.months[-1]}")
        if not self.include_entity_level:
            parts.append("excl. entity-level")
        return "; ".join(parts) or "whole ledger"


@dataclass(frozen=True)
class Metric:
    value: float
    rows: int
    label: str
    trace: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Bucket:
    key: str
    value: float
    rows: int


@dataclass(frozen=True)
class Breakdown:
    dimension: str
    buckets: list[Bucket]
    label: str
    trace: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return round(sum(b.value for b in self.buckets), 2)


@dataclass(frozen=True)
class Finding:
    kind: str
    summary: str
    magnitude: float
    detail: str = ""
