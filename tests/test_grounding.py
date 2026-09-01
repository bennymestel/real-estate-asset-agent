"""ground(): a raw QuerySpec + sub-question -> ResolvedQuery or Clarification.
Feeds hand-built QuerySpecs, so no LLM is called.
"""
import pytest

from src.data.catalog import get_catalog
from src.graph.nodes import ground
from src.schemas import (
    Clarification,
    ClarifyReason,
    Intent,
    QuerySpec,
    ResolvedQuery,
    SubQuestion,
)


@pytest.fixture(scope="module")
def cat():
    return get_catalog()


def sub(text, intent):
    return SubQuestion(text=text, intent=intent)


def test_unsupported_field_from_spec(cat):
    out = ground(QuerySpec(operation="pnl", unsupported_field="valuation"),
                 sub("what is building 120 worth", Intent.pnl_metric), cat)
    assert isinstance(out, Clarification)
    assert out.reason is ClarifyReason.unsupported_field


def test_unsupported_field_from_text_scan(cat):
    out = ground(QuerySpec(operation="pnl"),
                 sub("what is the price of my asset at 123 Main St", Intent.pnl_metric), cat)
    assert isinstance(out, Clarification)
    assert out.reason is ClarifyReason.unsupported_field


def test_unknown_property_clarifies_with_options(cat):
    out = ground(QuerySpec(operation="pnl", properties=["Building 999"]),
                 sub("how did building 999 do", Intent.pnl_metric), cat)
    assert isinstance(out, Clarification)
    assert out.reason is ClarifyReason.unknown_entity
    assert any("Building" in c for c in out.options)


def test_fuzzy_property_resolves(cat):
    out = ground(QuerySpec(operation="pnl", properties=["b120"]),
                 sub("p&l for b120", Intent.pnl_metric), cat)
    assert isinstance(out, ResolvedQuery)
    assert out.query.properties == ("Building 120",)


def test_comparison_two_buildings(cat):
    out = ground(QuerySpec(operation="pnl", properties=["Building 120", "Building 180"]),
                 sub("compare building 120 and building 180", Intent.comparison), cat)
    assert isinstance(out, ResolvedQuery)
    assert out.operation == "compare"
    assert out.group_by == "property_name"
    assert out.members == ("Building 120", "Building 180")


def test_ranking_defaults_to_tenant(cat):
    out = ground(QuerySpec(operation="top_n", top_n=5),
                 sub("who are my top tenants", Intent.ranking), cat)
    assert out.operation == "top_n"
    assert out.group_by == "tenant_name"
    assert out.top_n == 5


def test_largest_expense_ranks_by_magnitude(cat):
    out = ground(QuerySpec(operation="top_n", metric="expenses",
                           group_by="ledger_category", top_n=1),
                 sub("which expense category is the largest", Intent.ranking), cat)
    assert out.operation == "top_n"
    assert out.rank_by == "magnitude"
    assert out.group_by == "ledger_category"


def test_period_comparison_with_scope(cat):
    out = ground(QuerySpec(operation="compare", properties=["Building 120"],
                           compare_timeframes=["2024-Q1", "2024-Q2"]),
                 sub("compare q1 and q2 for building 120", Intent.comparison), cat)
    assert isinstance(out, ResolvedQuery)
    assert out.operation == "compare"
    assert out.group_by == "period"
    assert out.members == ("2024-Q1", "2024-Q2")
    assert out.query.properties == ("Building 120",)
    assert out.query.months == ("2024-M01", "2024-M02", "2024-M03",
                                "2024-M04", "2024-M05", "2024-M06")


def test_bare_quarter_comparison_snaps_to_year(cat):
    out = ground(QuerySpec(operation="compare", properties=["Building 120"]),
                 sub("compare Q1 and Q2 for Building 120", Intent.comparison), cat)
    assert isinstance(out, ResolvedQuery)
    assert out.members == ("2024-Q1", "2024-Q2")


def test_comparison_without_two_things_clarifies(cat):
    out = ground(QuerySpec(operation="compare", properties=["Building 120"], timeframe="2024"),
                 sub("compare building 120", Intent.comparison), cat)
    assert isinstance(out, Clarification)
    assert out.reason is ClarifyReason.vague


def test_anomaly_scan(cat):
    out = ground(QuerySpec(operation="pnl"),
                 sub("flag anything unusual", Intent.anomaly_scan), cat)
    assert out.operation == "anomalies"


def test_entity_details_sets_subject(cat):
    out = ground(QuerySpec(operation="details", properties=["Building 17"]),
                 sub("tell me about building 17", Intent.entity_details), cat)
    assert out.operation == "details"
    assert out.subject == "Building 17"


def test_group_by_month_is_timeseries(cat):
    out = ground(QuerySpec(operation="pnl", properties=["Building 160"],
                           metric="revenue", group_by="month"),
                 sub("revenue by month for building 160", Intent.pnl_metric), cat)
    assert out.operation == "timeseries"
    assert out.query.ledger_types == ("revenue",)


def test_group_by_property_is_breakdown(cat):
    out = ground(QuerySpec(operation="pnl", timeframe="2024", group_by="property_name"),
                 sub("break down 2024 p&l by building", Intent.pnl_metric), cat)
    assert out.operation == "breakdown"
    assert out.timeframe_label == "2024"


def test_out_of_range_timeframe_clarifies(cat):
    out = ground(QuerySpec(operation="pnl", timeframe="2023"),
                 sub("p&l for 2023", Intent.pnl_metric), cat)
    assert isinstance(out, Clarification)
    assert out.reason is ClarifyReason.uncovered_timeframe


def test_partial_year_caveat(cat):
    out = ground(QuerySpec(operation="pnl", timeframe="2025"),
                 sub("p&l this year", Intent.pnl_metric), cat)
    assert any("partial" in c for c in out.caveats)


def test_per_property_excludes_entity_level(cat):
    out = ground(QuerySpec(operation="pnl", properties=["Building 120"]),
                 sub("p&l for building 120", Intent.pnl_metric), cat)
    assert out.query.include_entity_level is False
    assert any("entity-level" in c for c in out.caveats)
