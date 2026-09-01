"""Shared types.

Deterministic tool layer: TimeRange (which months), LedgerQuery (which rows),
Metric/Bucket/Breakdown/Finding (what a tool returns).

LLM layer: Intent, SubQuestion, SupervisorPlan, QuerySpec (Pydantic — the models
the supervisor/extractor emit), plus Clarification and ResolvedQuery (the grounded
results the extractor hands to the analyst).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

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


# --- LLM layer -----------------------------------------------------------------


class Intent(str, Enum):
    pnl_metric = "pnl_metric"
    comparison = "comparison"
    ranking = "ranking"
    entity_details = "entity_details"
    anomaly_scan = "anomaly_scan"
    general_knowledge = "general_knowledge"
    capability = "capability"
    unsupported = "unsupported"
    vague = "vague"
    out_of_scope = "out_of_scope"


DATA_INTENTS: frozenset[Intent] = frozenset({
    Intent.pnl_metric, Intent.comparison, Intent.ranking,
    Intent.entity_details, Intent.anomaly_scan,
})


class SubQuestion(BaseModel):
    text: str = Field(description="one self-contained question, no 'and'")
    intent: Intent


class SupervisorPlan(BaseModel):
    sub_questions: list[SubQuestion] = Field(
        description="the message split into atomic parts; always at least one, "
                    "each with its own intent",
    )
    reasoning: str = Field(default="", description="one short line explaining the split")


class QuerySpec(BaseModel):
    operation: Literal["pnl", "breakdown", "timeseries", "top_n", "compare",
                       "details", "anomalies"]
    metric: Literal["net_pnl", "revenue", "expenses"] = "net_pnl"
    properties: list[str] = Field(default_factory=list, description="names as written")
    tenants: list[str] = Field(default_factory=list)
    ledger_groups: list[str] = Field(default_factory=list)
    ledger_categories: list[str] = Field(default_factory=list)
    timeframe: str = Field(
        default="all",
        description="'all', a year '2024', a quarter '2025-Q1', a month '2025-M02', "
                    "or a relative phrase like 'this year' / 'same period last year'",
    )
    compare_timeframes: list[str] = Field(
        default_factory=list,
        description="two+ periods to compare, e.g. ['2024-Q1', '2024-Q2'] for "
                    "'compare Q1 and Q2'; leave empty for a single-period question",
    )
    group_by: Optional[Literal["property_name", "tenant_name", "ledger_type",
                               "ledger_group", "ledger_category",
                               "month", "quarter", "year"]] = None
    top_n: Optional[int] = None
    unsupported_field: Optional[str] = Field(
        default=None,
        description="set if the question needs a field the ledger lacks: price, "
                    "valuation, appraisal, cap rate, occupancy, address, area, lease terms",
    )


class ClarifyReason(str, Enum):
    unknown_entity = "unknown_entity"
    unsupported_field = "unsupported_field"
    uncovered_timeframe = "uncovered_timeframe"
    vague = "vague"


@dataclass(frozen=True)
class Clarification:
    reason: ClarifyReason
    detail: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedQuery:
    operation: str
    query: LedgerQuery
    timeframe_label: str
    group_by: str | None = None
    top_n: int | None = None
    rank_by: Literal["value", "magnitude"] = "value"
    members: tuple[str, ...] = ()
    subject: str | None = None       # for details: property / tenant / "Portfolio"
    caveats: tuple[str, ...] = ()
    trace: tuple[str, ...] = ()
