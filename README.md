# Real Estate Asset Agent

A LangGraph multi-agent assistant that answers natural-language asset-management questions
over a general-ledger dataset (`data/cortex.parquet`) — P&L, comparisons, rankings, entity
detail cards, and a data-quality anomaly scan — with a Streamlit chat UI.

**Live demo:** _add Streamlit Cloud URL here after deploy_
**Design record:** [PLAN.md](PLAN.md) — the full plan this was built from, including
rejected alternatives.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then set GOOGLE_API_KEY
pytest                       # 79 tests, zero API calls (LLM calls are mocked)
streamlit run app.py
```

CLI, for one-off questions without the UI:
```bash
python -m src.graph.run "Compare Building 120 and Building 180, and flag anything odd"
```

## The dataset, and why it drives the design

3,924 rows × 12 columns — a **general-ledger fact table**, not an asset register. Each row is
one posted line: an entity, an optional property, an optional tenant, a ledger
type/group/category/code/description, a `month`/`quarter`/`year`, and a signed `profit` —
the *only* numeric column. No price, valuation, appraisal date, address, floor area,
occupancy or lease terms exist anywhere in the data.

That matters because two of the example questions in the brief ("price of my asset at 123
Main St", "Last Appraisal Date") are **not answerable from this dataset**. The honest move
is to say so plainly and offer what the ledger *can* show, rather than inventing a number —
that's the core of the "handles unexpected input" requirement, and it's why the graph has a
dedicated grounding step between extraction and calculation (below).

Other facts that shaped the design, all backed by tests in `tests/test_metrics.py`:

- **Signs are messy.** `sales_discounts` is negative revenue; 16 expense categories
  (e.g. `interest_mortgage`, +442k) net *positive*. So "revenue = positive amounts" is false —
  every total is `sum(profit)`, split by `ledger_type`, never by sign.
- **Duplicate rows are real and must not be deduped.** 1,747 of 3,924 rows are exact repeats.
  Most are `profit = 0` and inert. The ones that carry value are posting-and-reversal
  chains — e.g. Building 180 / Tenant 14 / 2024-M06 is `+97,708.92 ×3` and `−97,708.92 ×2`,
  netting to exactly one month's rent. Deduping any of these deletes real revenue (a naive
  `drop_duplicates()` drops the ledger-wide net P&L by ~35%). Signed summation over every row
  is the only correct read.
- **One genuine data defect exists:** ledger code `4650` is mapped to two different
  `ledger_category` values (`bank_charges` and `financial_expenses`) with the same
  description, across 121 rows, double-counting −3,627.54 (0.24% of net P&L). The anomaly
  scan surfaces this as a finding rather than silently patching it, so every number the
  agent states still ties back to the raw ledger.
- **2025 is a partial year** (Q1 only, ledger ends 2025-M03) — every year-level figure that
  includes 2025 says so.

Golden numbers (used as regression tests): net P&L = **€1,533,331.87**
(2024: €1,171,521.55 · 2025-Q1: €361,810.32); revenue €2,887,652.89 / expenses
−€1,354,321.02.

### Assumptions

1. **`REPORTING_AS_OF = 2025-03-31`**, the ledger's last period. All relative dates
   ("this year", "last quarter") resolve against it, not the wall clock — the ledger is a
   closed book, and a literal clock read would return zero rows for most of the brief's
   example questions. The UI states the reporting date up front.
2. `profit` is the single signed measure; P&L = `sum(profit)`, revenue/expenses = the same
   sum split by `ledger_type`.
3. Rows with a null `property_name` are entity-level overhead — excluded from per-property
   figures, included in portfolio totals, always disclosed.
4. Duplicate rows are never deduped (evidence above); the one genuine double-count
   (code 4650) is reported as a finding, not silently corrected.
5. Price / valuation / cap rate / occupancy / address questions are structurally
   unsupported — the agent says so and offers the nearest supported answer.
6. `Building 17` is a real distinct property (not a typo for 170).
7. Single entity (`PropCo`) — entity filtering exists but is currently a no-op.

## Architecture

The LLM never touches data or does arithmetic. It only ever does two things: turn language
into a validated Pydantic object, or turn a validated result back into language. Everything
in between — grounding, filtering, summing, ranking, the numeric fact-check — is plain,
tested Python.

```
                       ┌────────────┐
   user turn ─────────▶│ supervisor │  classify intent + split compound question
                       └─────┬──────┘
                             │
              needs data     │     no data needed (knowledge / capability /
             ┌───────────────┴──── unsupported field / vague / out-of-scope)
             ▼                                      │
       ┌───────────┐   QuerySpec + grounding        │
       │ extractor │   unknown entity / unsupported  │
       └─────┬─────┘   field → Clarification ───────▶│
             ▼                                      │
       ┌───────────┐   pandas: pnl / breakdown /     │
       │  analyst  │   timeseries / top_n / compare / │
       └─────┬─────┘   details / anomalies            │
             │                                      │
             │◀── loop to extractor while           │
             │    sub-questions remain              │
             ▼                                      ▼
                    ┌─────────────────────────────────┐
                    │           responder             │
                    │  LLM prose + numeric grounding  │  → END
                    │  check with one repair retry    │
                    └─────────────────────────────────┘
```

| node | job | LLM? |
|---|---|---|
| `supervisor` | Classifies the message and splits a compound question into typed sub-questions (`pnl_metric`, `comparison`, `ranking`, `entity_details`, `anomaly_scan`, `general_knowledge`, `capability`, `unsupported`, `vague`, `out_of_scope`). Caps at `MAX_SUB_QUESTIONS = 5`; overflow is disclosed, never silently dropped. | yes — one structured-output call |
| `extractor` | Builds a validated `QuerySpec` (operation, properties/tenants/categories, timeframe, `group_by`, `top_n`) from one sub-question, then **deterministically** fuzzy-resolves names against the live catalog (rapidfuzz) and checks the timeframe/metric is actually answerable. Any failure returns a `Clarification` — never an LLM decision. | yes (structured) + deterministic grounding |
| `analyst` | Runs the grounded query through pandas — `pnl`, `breakdown`, `timeseries`, `top_n`, `compare`, `details`, `anomalies` — and returns numbers plus a calculation trace (filters, rows matched, subtotals). | no — pure pandas, same input always gives the same output |
| `responder` | The single voice of the system. Canned parts (capability schema card, unsupported-field explainer, out-of-scope redirect, vague clarifier) are fixed copy and cost zero API calls. Everything else — data answers, entity clarifications, general-knowledge asides — goes through one LLM call, followed by a **deterministic post-check**: every number in the prose must trace back to the analyst's payload (with tolerance for rounding/abbreviation, e.g. "€0.36 million"). One ungrounded figure triggers a repair retry with the offending number named; if it still doesn't ground, the response falls back to the raw analyst figures verbatim. | yes + deterministic numeric grounding check |

Full node-by-node rationale, the four-case short-circuit table, and the two design
alternatives that were considered and rejected (a single combined planner+extractor call;
a 2-node ReAct `agent ⇄ tools` graph) are in [PLAN.md](PLAN.md).

### State

Nodes never call each other directly — LangGraph passes a shared `AgentState` dict, each
node returns only the keys it changed, and `MemorySaver` checkpoints it per `thread_id` so
follow-ups ("...and last year?") resolve against the real conversation history.

```python
class AgentState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]     # (role, text) — earlier turns, for follow-ups
    pending: list[SubQuestion]         # drained by the extract -> analyst loop
    results: list[BranchResult]        # one entry per finished sub-question
    dropped: int                       # sub-questions cut by MAX_SUB_QUESTIONS
    answer: str
    trace: list[str]                   # rendered in the UI's Agent Trace panel
    error: str | None                  # set by @safe_node, routes straight to responder
```

### Errors

Every node is wrapped by `@safe_node`: an exception is caught, written to `state["error"]`,
and routing sends it straight to `responder`, which apologises with the underlying message
instead of crashing the app.

## Project structure

```
app.py                    # Streamlit chat UI
src/
  config.py                # settings + get_llm(tier) factory
  schemas.py                # Intent, QuerySpec, ResolvedQuery, Metric/Breakdown/Finding, ...
  data/
    loader.py                 # cached parquet load + period parsing
    catalog.py                 # distinct values, coverage window, schema card for prompts
    resolver.py                 # rapidfuzz canonical name resolution
    timeframe.py                 # relative-phrase -> TimeRange resolution
  tools/
    metrics.py                # pnl, breakdown, timeseries, top_n, compare
    details.py                 # property/tenant/portfolio detail cards
    anomalies.py                 # reversal pairs, code 4650, coverage gaps, concentration
  graph/
    state.py, build.py            # StateGraph wiring, conditional edges, checkpointer
    nodes.py                        # supervisor, extractor + deterministic grounding
    analyst.py                        # dispatches ResolvedQuery -> tools/
    responder.py                        # LLM writer + numeric grounding check
    prompts.py                            # system prompts (schema injected live)
    run.py                                  # CLI entrypoint
tests/                     # 79 tests — golden numbers, resolver, grounding, full-graph
                              # scenarios (LLM calls mocked, deterministic layers run for real)
```

## Challenges and how they were solved

- **The brief's own example questions aren't all answerable.** Rather than quietly
  hallucinating a price, `unsupported_field` short-circuits at the supervisor (known from
  the schema alone) or the extractor (needs the live catalog), and the responder names the
  gap plus what *is* available.
- **Deduping looked like an obvious data-cleaning step and was wrong.** Verified by tracing
  the largest repeat groups by hand (see dataset section above) before deciding not to touch
  them — a silent "fix" here would have understated net P&L by over a third.
- **Keeping the LLM out of arithmetic entirely.** The extractor only ever fills a typed
  `QuerySpec`; grounding and every computation are plain pandas functions with direct unit
  tests. This makes the same question deterministic across runs and testable without any
  API key (`tests/test_graph.py` runs the whole graph with the three LLM calls mocked).
- **Stopping the responder from inventing a rounder number than the data supports.** A
  regex-based grounding check with a repair retry and a verbatim-figures fallback (see
  `responder.py` / `test_responder.py`) — cheap enough to run on every answer, strict enough
  to reject a plausible-looking hallucination.
- **Compound questions without over-engineering the fan-out.** A simple
  `extract -> analyst -> extract` loop over a bounded queue, rather than a `Send` map-reduce
  graph — one edge to reason about, and overflow past the cap is disclosed rather than
  dropped.

## Deployment

Deployed to Streamlit Community Cloud from this repo's `main` branch, entrypoint `app.py`,
with `GOOGLE_API_KEY` set in the app's Secrets. No key is committed — `.env` and
`.streamlit/secrets.toml` are both gitignored; `.env.example` documents the required
variable.
