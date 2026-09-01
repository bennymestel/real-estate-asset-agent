"""parse_timeframe: phrase -> months, measured against reporting_as_of (2025-03-31).
No LLM.
"""
import pytest

from src.data.catalog import get_catalog
from src.data.timeframe import compare_periods, find_timeframes, parse_timeframe


@pytest.fixture(scope="module")
def cat():
    return get_catalog()


def months(phrase, cat):
    return parse_timeframe(phrase, cat).months


def test_all(cat):
    assert months("all", cat) == tuple(cat.months)
    assert months("", cat) == tuple(cat.months)


def test_explicit_year(cat):
    assert months("2024", cat) == tuple(f"2024-M{m:02d}" for m in range(1, 13))


def test_partial_year_carries_note(cat):
    tr = parse_timeframe("2025", cat)
    assert tr.months == ("2025-M01", "2025-M02", "2025-M03")
    assert tr.note and "partial" in tr.note


def test_explicit_quarter(cat):
    assert months("2024-Q2", cat) == ("2024-M04", "2024-M05", "2024-M06")
    assert months("2025 q1", cat) == ("2025-M01", "2025-M02", "2025-M03")


def test_explicit_month(cat):
    assert months("2025-M02", cat) == ("2025-M02",)


def test_relative_to_as_of(cat):
    assert months("this year", cat) == ("2025-M01", "2025-M02", "2025-M03")
    assert months("last year", cat) == tuple(f"2024-M{m:02d}" for m in range(1, 13))
    assert months("this quarter", cat) == ("2025-M01", "2025-M02", "2025-M03")
    assert months("last quarter", cat) == ("2024-M10", "2024-M11", "2024-M12")
    assert months("same period last year", cat) == ("2024-M01", "2024-M02", "2024-M03")
    assert months("last month", cat) == ("2025-M03",)


def test_trailing_n_months(cat):
    assert months("last 6 months", cat) == (
        "2024-M10", "2024-M11", "2024-M12", "2025-M01", "2025-M02", "2025-M03",
    )


def test_out_of_range_year_is_empty(cat):
    assert parse_timeframe("2023", cat).months == ()


def test_unparseable_falls_back_to_all_with_note(cat):
    tr = parse_timeframe("banana", cat)
    assert tr.months == tuple(cat.months)
    assert tr.note and "could not read" in tr.note


def test_find_timeframes_explicit_tokens(cat):
    assert find_timeframes("Compare Q1 2024 and Q2 2024 for Building 120", cat) == \
        ["2024-Q1", "2024-Q2"]
    assert find_timeframes("2024 vs 2025", cat) == ["2024", "2025"]
    assert find_timeframes("revenue in 2025-M02", cat) == ["2025-M02"]


def test_find_timeframes_bare_quarters_snap_to_common_year(cat):
    assert find_timeframes("compare q1 and q2 for building 120", cat) == \
        ["2024-Q1", "2024-Q2"]


def test_compare_periods_resolves_two_ranges(cat):
    trs = compare_periods([], "compare q1 and q2 for building 120", cat)
    assert [t.label for t in trs] == ["2024-Q1", "2024-Q2"]
    trs = compare_periods(["2024-Q1", "2024-Q2"], "no tokens here", cat)
    assert [t.label for t in trs] == ["2024-Q1", "2024-Q2"]
