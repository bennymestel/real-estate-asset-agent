"""Full-graph scenarios. The two LLM calls (supervisor split, extractor spec) are
monkeypatched with canned output; ground(), the analyst tools and routing all run
for real. Zero API calls.
"""
import pytest

from src.graph import nodes
from src.graph.build import build_graph
from src.schemas import Intent, QuerySpec, SubQuestion, SupervisorPlan


def plan(*subs: tuple[str, Intent]) -> SupervisorPlan:
    return SupervisorPlan(
        sub_questions=[SubQuestion(text=t, intent=i) for t, i in subs],
        reasoning="canned",
    )


@pytest.fixture
def run_graph(monkeypatch):
    def _run(sup_plan: SupervisorPlan, specs: dict[str, QuerySpec] | None = None):
        specs = specs or {}
        monkeypatch.setattr(nodes, "supervisor", lambda q, h=None: sup_plan)
        monkeypatch.setattr(nodes, "_extract_spec", lambda sub: specs[sub.text])
        graph = build_graph()
        return graph.invoke(
            {"question": "q", "history": [], "pending": [], "results": [], "trace": []},
            {"configurable": {"thread_id": "t"}},
        )
    return _run


def test_simple_pnl(run_graph):
    state = run_graph(
        plan(("total P&L this year", Intent.pnl_metric)),
        {"total P&L this year": QuerySpec(operation="pnl", timeframe="this year")},
    )
    br = state["results"][0]
    assert br.data.value == 361_810.32
    assert "361,810.32" in state["answer"]
    assert any("partial" in c for c in br.caveats)


def test_comparison_two_buildings(run_graph):
    state = run_graph(
        plan(("compare 120 and 180", Intent.comparison)),
        {"compare 120 and 180": QuerySpec(
            operation="compare", properties=["Building 120", "Building 180"])},
    )
    br = state["results"][0]
    assert br.resolved.operation == "compare"
    assert br.resolved.group_by == "property_name"
    assert [b.key for b in br.data.buckets] == ["Building 120", "Building 180"]


def test_period_comparison_uses_period_branch(run_graph):
    state = run_graph(
        plan(("compare Q1 and Q2 for Building 120", Intent.comparison)),
        {"compare Q1 and Q2 for Building 120": QuerySpec(
            operation="compare", properties=["Building 120"],
            compare_timeframes=["2024-Q1", "2024-Q2"])},
    )
    br = state["results"][0]
    assert br.resolved.group_by == "period"
    assert [b.key for b in br.data.buckets] == ["2024-Q1", "2024-Q2"]
    assert "vs" in state["answer"]


def test_compound_ranking_plus_anomaly(run_graph):
    state = run_graph(
        plan(("top tenants", Intent.ranking),
             ("anything unusual", Intent.anomaly_scan)),
        {"top tenants": QuerySpec(operation="top_n", top_n=5),
         "anything unusual": QuerySpec(operation="anomalies")},
    )
    assert len(state["results"]) == 2
    assert "Tenant 7" in state["answer"]
    assert isinstance(state["results"][1].data, list)  # findings


def test_unknown_entity_clarifies(run_graph):
    state = run_graph(
        plan(("how did Building 999 do", Intent.pnl_metric)),
        {"how did Building 999 do": QuerySpec(
            operation="pnl", properties=["Building 999"])},
    )
    br = state["results"][0]
    assert br.clarification is not None
    assert br.data is None
    assert "Building 120" in state["answer"]


def test_unsupported_metric_short_circuits(run_graph):
    state = run_graph(plan(("price of my asset at 123 Main St", Intent.unsupported)))
    assert "valuation" in state["answer"]
    assert state["results"][0].resolved is None


def test_capability_returns_schema_card(run_graph):
    state = run_graph(plan(("what data do you have", Intent.capability)))
    assert "LEDGER SCHEMA" in state["answer"]


def test_out_of_scope_redirect(run_graph):
    state = run_graph(plan(("write me a poem", Intent.out_of_scope)))
    assert "asset-management assistant" in state["answer"]


def test_overflow_is_disclosed(run_graph):
    subs = [(f"part {n}", Intent.pnl_metric) for n in range(7)]
    specs = {f"part {n}": QuerySpec(operation="pnl") for n in range(7)}
    state = run_graph(plan(*subs), specs)
    assert len(state["results"]) == 5
    assert state["dropped"] == 2
    assert "weren't covered" in state["answer"]


def test_node_error_routes_to_responder(monkeypatch):
    monkeypatch.setattr(nodes, "supervisor",
                        lambda q, h=None: plan(("break", Intent.pnl_metric)))

    def boom(sub):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(nodes, "_extract_spec", boom)

    out = build_graph().invoke(
        {"question": "q", "history": [], "pending": [], "results": [], "trace": []},
        {"configurable": {"thread_id": "err"}},
    )
    assert "internal error" in out["answer"]
    assert out["error"] is not None
    assert any("kaboom" in line for line in out["trace"])
