"""The two LLM nodes: supervisor (classify + split) and extractor (sub-question ->
grounded query). Arithmetic lives in src/tools; these only understand text.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_llm
from src.data.catalog import UNSUPPORTED_METRICS, Catalog, get_catalog
from src.data.resolver import (
    resolve_ledger_category,
    resolve_ledger_group,
    resolve_property,
    resolve_tenant,
)
from src.data.timeframe import compare_periods, parse_timeframe
from src.graph import prompts
from src.schemas import (
    Clarification,
    ClarifyReason,
    DATA_INTENTS,
    Intent,
    LedgerQuery,
    QuerySpec,
    ResolvedQuery,
    SubQuestion,
    SupervisorPlan,
    TimeRange,
)

_METRIC_TO_TYPES = {"net_pnl": (), "revenue": ("revenue",), "expenses": ("expenses",)}
_CLARIFY_OPTIONS = (
    "net P&L, revenue or expenses for a property/tenant/period",
    "a breakdown by property, tenant, month or category",
    "a data-quality scan",
)


def supervisor(question: str, history: list[tuple[str, str]] | None = None) -> SupervisorPlan:
    llm = get_llm("fast").with_structured_output(SupervisorPlan)
    msgs: list = [SystemMessage(prompts.supervisor_system())]
    for role, text in history or []:
        msgs.append(HumanMessage(f"[{role}] {text}"))
    msgs.append(HumanMessage(question))
    plan: SupervisorPlan = llm.invoke(msgs)
    if not plan.sub_questions:
        plan.sub_questions = [SubQuestion(text=question, intent=Intent.vague)]
    return plan


def extractor(sub: SubQuestion, cat: Catalog | None = None) -> ResolvedQuery | Clarification:
    cat = cat or get_catalog()
    if sub.intent not in DATA_INTENTS:
        raise ValueError(
            f"extractor called with non-data intent {sub.intent.value}; "
            "route these straight to the responder"
        )
    llm = get_llm("fast").with_structured_output(QuerySpec)
    spec: QuerySpec = llm.invoke([
        SystemMessage(prompts.extractor_system()),
        HumanMessage(sub.text),
    ])
    return ground(spec, sub, cat)


def ground(spec: QuerySpec, sub: SubQuestion, cat: Catalog) -> ResolvedQuery | Clarification:
    field = spec.unsupported_field or _scan_unsupported(sub.text)
    if field:
        return Clarification(
            ClarifyReason.unsupported_field,
            f"the ledger has no {field} data",
            options=_CLARIFY_OPTIONS,
        )

    props, clar = _resolve_all(spec.properties, resolve_property, "property")
    if clar:
        return clar
    tenants, clar = _resolve_all(spec.tenants, resolve_tenant, "tenant")
    if clar:
        return clar
    groups, clar = _resolve_all(spec.ledger_groups, resolve_ledger_group, "ledger group")
    if clar:
        return clar
    categories, clar = _resolve_all(spec.ledger_categories, resolve_ledger_category, "ledger category")
    if clar:
        return clar

    if sub.intent is Intent.comparison and len(props) < 2 and len(tenants) < 2:
        return _compare_over_periods(spec, sub, cat, props, tenants, groups, categories)

    tr = (TimeRange.all_time(cat) if sub.intent is Intent.anomaly_scan
          else parse_timeframe(spec.timeframe, cat))
    if not tr.months:
        return Clarification(
            ClarifyReason.uncovered_timeframe,
            f"'{spec.timeframe}' is outside the ledger "
            f"({cat.coverage_start}..{cat.coverage_end})",
        )

    caveats: list[str] = [tr.note] if tr.note else []
    op = spec.operation
    group_by = spec.group_by
    members: tuple[str, ...] = ()
    subject: str | None = None
    rank_by = "value"

    if sub.intent is Intent.anomaly_scan:
        op = "anomalies"
    elif sub.intent is Intent.entity_details:
        op = "details"
        subject = props[0] if props else tenants[0] if tenants else "Portfolio"
    elif sub.intent is Intent.comparison:
        op, group_by, members = "compare", "property_name", tuple(props)
        if len(tenants) >= 2:
            group_by, members = "tenant_name", tuple(tenants)
    elif sub.intent is Intent.ranking:
        op = "top_n"
        group_by = group_by or (
            "tenant_name" if tenants or "tenant" in sub.text.lower() else "property_name"
        )
        if spec.metric == "expenses":
            rank_by = "magnitude"
    elif group_by:
        op = "timeseries" if group_by in ("month", "quarter", "year") else "breakdown"

    include_entity = not props
    if props:
        caveats.append("per-property figures exclude entity-level overhead")

    query = LedgerQuery(
        properties=tuple(props),
        tenants=tuple(tenants),
        ledger_types=_METRIC_TO_TYPES[spec.metric],
        ledger_groups=tuple(groups),
        ledger_categories=tuple(categories),
        months=tr.months,
        include_entity_level=include_entity,
    )
    return ResolvedQuery(
        operation=op,
        query=query,
        timeframe_label=tr.label,
        group_by=group_by,
        top_n=spec.top_n,
        rank_by=rank_by,
        members=members,
        subject=subject,
        caveats=tuple(caveats),
        trace=(
            f"intent={sub.intent.value}",
            f"operation={op}",
            f"timeframe={tr.label}",
            f"filter: {query.describe()}",
        ),
    )


def _compare_over_periods(spec, sub, cat, props, tenants, groups, categories):
    periods = compare_periods(spec.compare_timeframes, sub.text, cat)
    if len(periods) < 2:
        return Clarification(
            ClarifyReason.vague,
            "a comparison needs two things — name two properties, two tenants, "
            "or two periods",
        )
    months = tuple(m for m in cat.months
                   if m in {mo for tr in periods for mo in tr.months})
    labels = tuple(tr.label for tr in periods)
    caveats = [tr.note for tr in periods if tr.note]
    if props:
        caveats.append("per-property figures exclude entity-level overhead")
    query = LedgerQuery(
        properties=tuple(props),
        tenants=tuple(tenants),
        ledger_types=_METRIC_TO_TYPES[spec.metric],
        ledger_groups=tuple(groups),
        ledger_categories=tuple(categories),
        months=months,
        include_entity_level=not props,
    )
    return ResolvedQuery(
        operation="compare",
        query=query,
        timeframe_label=" vs ".join(labels),
        group_by="period",
        members=labels,
        caveats=tuple(caveats),
        trace=(
            "intent=comparison",
            "operation=compare",
            f"periods={list(labels)}",
            f"filter: {query.describe()}",
        ),
    )


def _scan_unsupported(text: str) -> str | None:
    low = text.lower()
    for field, terms in UNSUPPORTED_METRICS.items():
        if any(term in low for term in terms):
            return field.replace("_", " ")
    return None


def _resolve_all(raw, resolver, kind: str):
    out: list[str] = []
    for term in raw:
        res = resolver(term)
        if not res.ok:
            return [], Clarification(
                ClarifyReason.unknown_entity,
                f"no {kind} matches '{term}'",
                options=tuple(res.candidates),
            )
        out.append(res.value)
    return out, None
