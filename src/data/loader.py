"""Load the ledger parquet once and add sortable period columns.

The returned frame is treated as immutable — tools filter and aggregate, never mutate.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.config import settings

RAW_COLUMNS = [
    "entity_name", "property_name", "tenant_name",
    "ledger_type", "ledger_group", "ledger_category", "ledger_code", "ledger_description",
    "month", "quarter", "year", "profit",
]


@lru_cache(maxsize=1)
def load_ledger(path: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path or settings.data_path)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"ledger is missing expected columns: {sorted(missing)}")

    df = df.copy()
    # "2024-M01" -> year_num 2024, month_num 1, period_ord 202401 (sortable int)
    parts = df["month"].str.extract(r"^(?P<y>\d{4})-M(?P<m>\d{2})$")
    df["year_num"] = parts["y"].astype(int)
    df["month_num"] = parts["m"].astype(int)
    df["period_ord"] = df["year_num"] * 100 + df["month_num"]
    # "2024-Q1" -> quarter_num 1
    df["quarter_num"] = df["quarter"].str.extract(r"Q(\d)$").astype(int)
    df["is_entity_level"] = df["property_name"].isna()
    return df


def coverage(df: pd.DataFrame | None = None) -> tuple[str, str]:
    """(first month, last month) present in the ledger, e.g. ('2024-M01', '2025-M03')."""
    df = load_ledger() if df is None else df
    ordered = df.sort_values("period_ord")["month"]
    return ordered.iloc[0], ordered.iloc[-1]
