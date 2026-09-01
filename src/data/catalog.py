"""Distinct ledger values + a compact schema card for LLM prompts.

Run `python -m src.data.catalog` to print the schema card.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from src.config import settings
from src.data.loader import coverage, load_ledger

# Fields a reviewer will ask for that this ledger simply does not contain.
# Used by the supervisor (to short-circuit) and the responder (to explain the gap).
UNSUPPORTED_METRICS: dict[str, tuple[str, ...]] = {
    "valuation": ("price", "value", "worth", "valuation", "appraisal", "appraised", "market value"),
    "cap_rate": ("cap rate", "capitalisation rate", "capitalization rate", "yield on cost"),
    "occupancy": ("occupancy", "vacancy", "occupied", "vacant"),
    "address": ("address", "located", "location", "street", "postcode", "zip"),
    "area": ("sqm", "sq m", "square metres", "square meters", "square feet", "sqft", "floor area"),
    "lease_terms": ("lease term", "lease start", "lease end", "lease expiry", "wault", "break option"),
}


@dataclass(frozen=True)
class Catalog:
    entities: list[str]
    properties: list[str]
    tenants: list[str]
    ledger_types: list[str]
    ledger_groups: list[str]
    ledger_categories: list[str]
    ledger_codes: list[int]
    months: list[str]
    quarters: list[str]
    years: list[str]
    coverage_start: str
    coverage_end: str
    reporting_as_of: str
    category_descriptions: dict[str, str] = field(default_factory=dict)
    month_quarter: dict[str, str] = field(default_factory=dict)  # "2024-M01" -> "2024-Q1"
    month_year: dict[str, str] = field(default_factory=dict)     # "2024-M01" -> "2024"

    def months_in_year(self, year: str | int) -> list[str]:
        return [m for m in self.months if self.month_year[m] == str(year)]

    def months_in_quarter(self, quarter: str) -> list[str]:
        return [m for m in self.months if self.month_quarter[m] == quarter]

    def schema_card(self) -> str:
        cats = ", ".join(self.ledger_categories)
        return "\n".join([
            "LEDGER SCHEMA (the only data available)",
            f"- entity: {', '.join(self.entities)} (single entity)",
            f"- properties: {', '.join(self.properties)} "
            "(rows with no property = entity-level overhead)",
            f"- tenants: {len(self.tenants)} tenants, Tenant 1..Tenant 18",
            f"- ledger_type: {', '.join(self.ledger_types)}",
            f"- ledger_group: {', '.join(self.ledger_groups)}",
            f"- ledger_category: {cats}",
            f"- time: monthly {self.coverage_start}..{self.coverage_end}; "
            f"quarters {self.quarters[0]}..{self.quarters[-1]}; years {', '.join(self.years)}",
            "- measure: `profit` (signed EUR) — the ONLY numeric column. "
            "P&L = sum(profit); revenue vs expenses = split by ledger_type.",
            f"- reporting as of: {self.reporting_as_of} (relative dates resolve against this)",
            "NOT AVAILABLE: valuation/price, cap rate, occupancy, address, floor area, lease terms.",
        ])


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    df = load_ledger()
    start, end = coverage(df)

    def uniq(col: str) -> list[str]:
        return sorted(df[col].dropna().unique().tolist())

    descriptions = (
        df[["ledger_category", "ledger_description"]]
        .drop_duplicates("ledger_category")
        .set_index("ledger_category")["ledger_description"]
        .to_dict()
    )
    period_map = df[["month", "quarter", "year"]].drop_duplicates().set_index("month")
    return Catalog(
        entities=uniq("entity_name"),
        properties=uniq("property_name"),
        tenants=uniq("tenant_name"),
        ledger_types=uniq("ledger_type"),
        ledger_groups=uniq("ledger_group"),
        ledger_categories=uniq("ledger_category"),
        ledger_codes=sorted(df["ledger_code"].dropna().unique().tolist()),
        months=uniq("month"),
        quarters=uniq("quarter"),
        years=uniq("year"),
        coverage_start=start,
        coverage_end=end,
        reporting_as_of=settings.reporting_as_of.isoformat(),
        category_descriptions=descriptions,
        month_quarter=period_map["quarter"].to_dict(),
        month_year=period_map["year"].to_dict(),
    )


if __name__ == "__main__":
    print(get_catalog().schema_card())
