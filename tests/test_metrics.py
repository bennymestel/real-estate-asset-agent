"""Golden-number tests: lock the deterministic math to values verified by hand
against data/cortex.parquet. Zero LLM calls in this file.
"""
import pytest

from src.data.catalog import get_catalog
from src.data.loader import load_ledger
from src.schemas import LedgerQuery, TimeRange
from src.tools.anomalies import scan
from src.tools.details import portfolio_card, property_card
from src.tools.metrics import breakdown, compare, pnl, timeseries, top_n


@pytest.fixture(scope="module")
def df():
    return load_ledger()


@pytest.fixture(scope="module")
def cat():
    return get_catalog()


def test_net_pnl(df):
    assert pnl(df, LedgerQuery()).value == 1_533_331.87


def test_pnl_by_year(df):
    ts = {b.key: b.value for b in timeseries(df, LedgerQuery(), by="year").buckets}
    assert ts == {"2024": 1_171_521.55, "2025": 361_810.32}


def test_pnl_2025_q1(df, cat):
    q = LedgerQuery(months=TimeRange.quarter("2025-Q1", cat).months)
    assert pnl(df, q).value == 361_810.32


def test_by_ledger_type(df):
    b = {x.key: (x.value, x.rows) for x in breakdown(df, LedgerQuery(), "ledger_type").buckets}
    assert b["revenue"] == (2_887_652.89, 3_135)
    assert b["expenses"] == (-1_354_321.02, 789)


def test_by_ledger_group(df):
    b = {x.key: x.value for x in breakdown(df, LedgerQuery(), "ledger_group").buckets}
    assert b == {
        "rental_income": 3_072_754.64,
        "general_expenses": -782_154.21,
        "management_fees": -471_587.49,
        "sales_discounts": -185_101.75,
        "taxes_and_insurances": -100_579.32,
    }


def test_by_property_includes_entity_level(df):
    b = {x.key: x.value for x in breakdown(df, LedgerQuery(), "property_name", dropna=False).buckets}
    assert b["Building 120"] == 850_567.42
    assert b["Building 140"] == 526_658.85
    assert b["Building 160"] == 713_065.13
    assert b["Building 17"] == 352_566.81
    assert b["Building 180"] == 384_900.03
    assert b["(entity-level)"] == -1_294_426.37
    assert round(sum(b.values()), 2) == 1_533_331.87


def test_by_quarter(df):
    ts = {b.key: b.value for b in timeseries(df, LedgerQuery(), by="quarter").buckets}
    assert ts == {
        "2024-Q1": 262_309.07, "2024-Q2": 317_892.76, "2024-Q3": 312_364.85,
        "2024-Q4": 278_954.87, "2025-Q1": 361_810.32,
    }
    assert list(ts) == sorted(ts)  # chronological order preserved


def test_top_tenants(df):
    top = top_n(df, LedgerQuery(), "tenant_name", 5)
    assert [(b.key, b.value) for b in top.buckets] == [
        ("Tenant 7", 880_512.18),
        ("Tenant 14", 391_490.29),
        ("Tenant 11", 292_531.00),
        ("Tenant 13", 274_344.48),
        ("Tenant 3", 204_788.42),
    ]


def test_compare_two_buildings_keeps_order(df):
    c = compare(df, LedgerQuery(), "property_name", ["Building 180", "Building 120"])
    assert [(b.key, b.value) for b in c.buckets] == [
        ("Building 180", 384_900.03),
        ("Building 120", 850_567.42),
    ]


def test_property_card_excludes_entity_level(df):
    card = property_card(df, "Building 120")
    assert card.headline.value == 850_567.42
    assert card.months_active[0] == "2024-M01"


def test_portfolio_card_ties_to_net(df):
    assert portfolio_card(df).headline.value == 1_533_331.87


# --- anomalies -------------------------------------------------------------

def test_scan_finds_the_big_reversal_pair(df):
    rev = [f for f in scan(df) if f.kind == "reversal"]
    b120 = next(f for f in rev if "Building 120" in f.summary and "Tenant 7" in f.summary)
    assert b120.magnitude == 154_415.07
    for m in ("2024-M06", "2024-M09", "2024-M12", "2025-M03"):
        assert m in b120.detail


def test_scan_finds_double_mapped_code_4650(df):
    dm = [f for f in scan(df) if f.kind == "double_mapped_code"]
    assert any("4650" in f.summary for f in dm)
    f4650 = next(f for f in dm if "4650" in f.summary)
    assert "bank_charges" in f4650.summary and "financial_expenses" in f4650.summary


def test_scan_flags_partial_2025(df):
    cov = [f for f in scan(df) if f.kind == "coverage"]
    assert any("2025" in f.summary for f in cov)


def test_scan_flags_tenant_7_concentration(df):
    con = [f for f in scan(df) if f.kind == "concentration"]
    assert any("Tenant 7" in f.summary for f in con)
