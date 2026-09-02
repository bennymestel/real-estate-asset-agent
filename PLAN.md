# Real Estate Asset Management — Multi-Agent System (LangGraph)

## Context

Interview take-home: a LangGraph multi-agent prototype that answers natural-language real-estate
asset-management questions over `data/cortex.parquet`. Deliverables are a public GitHub repo, a
deployed URL, and a README covering setup, architecture, the LangGraph workflow, and challenges.
The repo is currently empty apart from the brief and the dataset — everything below is new code.

The design is driven by what the data actually is (verified, see below): a **general-ledger fact
table**, not an asset register. Several example questions in the brief ("price of my asset at 123
Main St", "Last Appraisal Date") **cannot** be answered from it. Answering those honestly — naming
the gap and offering what *is* available — is the core of the "handles unexpected input" criterion,
and the reason the architecture puts a dedicated capability/grounding check between extraction and
calculation.

## Verified dataset facts (drive several design decisions)

3,924 rows × 12 columns.

| column | reality |
|---|---|
| `entity_name` | 1 value: `PropCo` |
| `property_name` | `Building 17, 120, 140, 160, 180`; **581 nulls** = entity-level overhead |
| `tenant_name` | `Tenant 1..18`; 759 nulls |
| `ledger_type` | `revenue` (3,135 rows) / `expenses` (789) |
| `ledger_group` | rental_income, sales_discounts, general_expenses, management_fees, taxes_and_insurances |
| `ledger_category` / `ledger_code` / `ledger_description` | 29 / 28 / 28 values, bilingual NL\|EN labels |
| `month` / `quarter` / `year` | `2024-M01` → **`2025-M03`** (2025 partial: Q1 only) |
| `profit` | signed float — the **only** measure |

- Net P&L = **1,533,331.87** (2024: 1,171,521.55 · 2025-Q1: 361,810.32). Revenue +2,887,684 / Expenses −1,354,352.
- Signs are messy: `sales_discounts` is negative revenue (−185,101.75); 16 expense categories net
  **positive** (`interest_mortgage` +442,392.87, `expense_return`, etc.). So "revenue = positive" is false —
  always aggregate the signed `profit` and split by `ledger_type`, never by sign.
- **Duplicate rows: verified, and deduping would be wrong.** 1,747 rows are exact repeats, but 1,348 rows
  carry `profit = 0` and the largest repeat groups (×96, ×52, ×48) are entirely zero-valued — inert to any
  sum. The 251 repeat groups that *do* carry value are **posting-and-reversal chains**: Building 180 /
  Tenant 14 / 2024-M06 is `+97,708.92 ×3` with `−97,708.92 ×2`, netting to exactly one month's rent;
  Building 140 / Tenant 3 is `+13,267.58 ×7` with `−13,267.58 ×6`, likewise netting to one. Deduping
  collapses each chain to `+1 and −1 = 0` and **deletes real revenue**. Signed summation over all rows is
  the correct read. (Naive dedupe would drop the headline total from 1,533,331.87 to 995,111.80, −35%.)
- **One real data defect:** `ledger_code 4650` is the only code mapped to two `ledger_category` values —
  `bank_charges` and `financial_expenses` — with the same description "Bankkosten | Bank charges", 121 rows
  and −3,627.54 each, matching month by month. A category fan-out double-counting one set of postings, worth
  −3,627.54 (0.24% of net). Surfaced by the anomaly tool and noted in the README; **not** silently patched.
- **Contra-posting pairs** exist (Tenant 7 / Building 120: +154,415.07 and −154,415.07 in the same month;
  also Tenant 14 ±101,128.27). Real reversals — the anomaly agent surfaces these.
- **No addresses, valuations, appraisal dates, sqm, occupancy, or lease terms.**

## Decisions (agreed)

- **LLM**: `gemini-3.1-flash-lite` for **every** node, via `langchain-google-genai`. Verified to exist and to
  support JSON-schema structured output, which is what the planner/extractor nodes depend on; it is the
  cheapest, lowest-latency Gemini, so the deployed demo stays free and snappy.
  `config.get_llm(tier)` still takes a tier argument, but both tiers resolve to this model by default via
  `LLM_MODEL_FAST` / `LLM_MODEL_SMART` env vars — so if step 3 shows weak compound-question decomposition
  or flat narration, bumping one env var to a larger Gemini fixes it with no code change. Two build-time
  checks in step 3: (a) the pinned `langchain-google-genai` version actually routes `gemini-3.1-flash-lite`,
  (b) whether the SDK exposes a thinking/reasoning level — if so, keep it minimal on classifier nodes.
- **Compute**: LLM emits a *validated Pydantic `QuerySpec`*; a hand-written pandas executor does 100% of the
  arithmetic. No LLM-generated code or SQL → deterministic, unit-testable, no hallucinated math.
- **Deploy**: Streamlit Community Cloud from the public GitHub repo; `GOOGLE_API_KEY` in the secrets UI.
- **Orchestration**: LangGraph `StateGraph` + `MemorySaver` checkpointer for follow-up turns.

## Architecture — the agent graph

**Four nodes.** A node exists only where the *unit of work* genuinely changes: decide → ground →
compute → speak. Everything deterministic (grounding checks, tool dispatch, numeric verification) is
a plain tested function inside the node that needs it, and every text-generation job — answer,
clarifying question, domain explainer — is one node with three prompt modes, not three nodes.

```
                       ┌────────────┐
   user turn ─────────▶│ supervisor │  guard + classify intent + split compound → sub-questions
                       └─────┬──────┘
                             │
              needs data     │     no data needed
             ┌───────────────┴──────────────────────┐
             ▼                                      │
       ┌───────────┐   QuerySpec + grounding        │
       │ extractor │   ─ unknown entity /           │
       └─────┬─────┘     unsupported field ────────▶│
             ▼                                      │
       ┌───────────┐   pandas: pnl / compare /      │
       │  analyst  │   top_n / timeseries /         │
       └─────┬─────┘   details / anomalies          │
             │                                      │
             │◀── loop to extractor while           │
             │    sub-questions remain              │
             ▼                                      ▼
                    ┌─────────────────────────────────┐
                    │           responder             │
                    │  mode: answer | clarify | know  │  → numeric grounding check → END
                    └─────────────────────────────────┘
```

### Node responsibilities — the plain version

Four people handling one request:

| node | in plain English |
|---|---|
| `supervisor` | **The receptionist.** Reads the message, decides what kind of question it is, and breaks "A and B?" into separate jobs. Never touches the data. |
| `extractor` | **The junior analyst.** Turns each job into a precise, filled-in query form — which buildings, which months, which number — and checks the data can actually answer it before anyone wastes time. |
| `analyst` | **The calculator.** Runs that form against the table with pandas. No LLM at all — just filtering and arithmetic, so the same question always gives the same number. |
| `responder` | **The writer.** Turns numbers into an English answer, and refuses to state any figure the calculator didn't produce. |

The point of the split: the LLM never does arithmetic, and the arithmetic never does English.

**Worked example — "Compare Building 120 and Building 180 this year, and flag anything unusual."**

1. `supervisor` → needs data; splits into (a) compare two buildings this year, (b) anomaly scan.
2. `extractor` (a) → `properties=[Building 120, Building 180]`, `period=2025 (Q1 only)`, `metric=net P&L`,
   `group_by=property`. Both resolve against the catalog → proceed.
3. `analyst` → filters, sums, returns `{B120: …, B180: …}` + trace (rows matched, subtotals).
4. Loop to `extractor` for (b) → `analyst` anomaly scan → finds the ±154,415.07 reversal pair.
5. `responder` → writes both answers, adds the "2025 is Q1 only" caveat, then verifies every number in
   its prose appears in the step 3/4 payloads.

### Node responsibilities — the detailed version

| node | job | LLM? |
|---|---|---|
| `supervisor` | Entry point and router. Guards junk/empty/off-domain input, classifies intent (`pnl_metric`, `comparison`, `ranking`, `entity_details`, `anomaly_scan`, `general_knowledge`, `unsupported`, `vague`), and splits a compound question into typed sub-questions (up to `MAX_SUB_QUESTIONS`, default 5). One structured-output call does all three. | yes (structured) |
| `extractor` | Builds a validated `QuerySpec` for the current sub-question — properties, tenants, ledger filters, `TimeRange`, metric, `group_by`, `top_n`, comparison baseline — then deterministically resolves free text to canonical values (rapidfuzz) **and grounds it**. Unknown entity, unsupported field (price/appraisal/sqm), or uncovered timeframe sets `clarification_reason` and short-circuits to `responder`. | yes (structured) + deterministic resolve/ground |
| `analyst` | Runs the spec through pandas, returning numbers **plus a calculation trace** (filters, rows matched, intermediate sums). Dispatches by intent to `pnl / aggregate / timeseries / top_n / compare / details / anomalies` — plain functions in `src/tools/`, unit-tested independently. | no |
| `responder` | The single voice of the system, in three modes. **answer**: direct answer, step-by-step derivation, table/chart payload, caveats. **clarify**: exactly one targeted question with concrete options from the live catalog ("I have Buildings 17, 120, 140, 160, 180 — which did you mean?"), or the honest "your ledger has no valuations, but here's what I *can* tell you". **knowledge**: general real-estate domain answer, explicitly labelled "not from your ledger". Then a deterministic post-check regex-extracts every number in the prose and asserts it exists in the results payload — one repair retry, else fall back to the grounded table. | yes + deterministic check |

### When the `supervisor → responder` short-circuit actually fires

Four real cases, all of which a reviewer will type into the demo:

| input | why no QuerySpec is possible or needed |
|---|---|
| "How is NOI calculated?" / "What's a good cap rate?" | `general_knowledge` — answerable from the model, never from the ledger. **Required by the brief**, which names "general knowledge" among the request types to detect and demands that "all types of questions must be handled, including questions that are vague, compound, or that the provided data may not fully support". Scoped tight: 2–4 sentences of real-estate domain knowledge, always labelled "general industry knowledge — not from your ledger", closing with a pointer to what the data *can* show. It is an asset-management assistant, not a general chatbot — "write me a poem" still gets the out-of-scope redirect. |
| "What data do you have?" / "Which properties do you cover?" | Answered from the **catalog** (a static schema card), not from an aggregation. No filtering, no arithmetic. |
| "What's the price of my asset at 123 Main St?" | The *metric* is unsupported — `price/valuation/appraisal/sqm/occupancy` aren't columns, and the schema card in the supervisor's prompt is enough to know that. No point resolving an address that can't be queried either way. |
| "How are things going?" / "asdfgh" / "write me a poem" | Vague or out-of-scope; nothing to extract. |

The split with the extractor path is principled: **unsupported *metric* → supervisor catches it**
(knowable from the schema alone); **unknown or missing *entity* → extractor catches it** (needs fuzzy
resolution against live catalog values — "Building 999", "the Oak building", or a P&L question that
never names a property). Both terminate in the responder's `clarify` mode, so the user experience is
identical; only the detection point differs. If in build step 4 the supervisor turns out to catch all
of these reliably enough that the edge is never exercised, the honest move is to delete the edge —
step 4's checkpoint scenario list covers all four, so it will be obvious either way.

**Why clarification isn't just a line in the supervisor's system prompt:** the supervisor can catch
*vague* input ("how are things going?") and does. But "Building 999 doesn't exist" and "your schema
has no valuations" are only knowable after the extractor fuzzy-resolves against the live catalog —
that's a deterministic check on real data, not a prompt instruction, and letting the supervisor guess
at it is exactly how you get a confidently invented $500,000 price. The *decision* is code in the
extractor; the *wording* is one more responder mode.

**Why `responder` isn't the supervisor's system prompt:** the supervisor runs *before* the numbers
exist — it sees a question, the responder sees results. That's two invocations at different points in
the graph, and in LangGraph a second invocation is a second node. They also need different bindings:
understanding wants `with_structured_output(QuerySpec)` at temperature 0, speaking wants free prose.
(Considered and rejected: collapsing the two structured calls — `supervisor` + `extractor` — into a
single planner call, and a 2-node ReAct `agent ⇄ tools` graph. The four-stage pipeline maps 1:1 to
the brief's processing stages and to the UI's trace panel, which is worth one extra box. README
records the rejected alternatives.)

**Compound questions** use a simple loop (`analyst → extractor` while the queue is non-empty) rather
than a `Send` map-reduce fan-out — one edge to explain, and it still demonstrates cyclic state in
LangGraph. The cap is a config constant, not a magic number: `MAX_SUB_QUESTIONS = 5`. It exists to
bound cost and latency (each sub-question is one extra LLM call plus one pandas pass, ~1–2s) and to
stop a runaway loop from hitting LangGraph's `recursion_limit` — worst case here is 2 + 2×5 + 1 ≈ 13
node visits, comfortably inside the default 25. Real asset-management questions top out around three
parts ("compare Q1 vs last year for Buildings 120 and 180, and flag anything odd"), so 5 is slack.

Crucially, **overflow is never silent**: if the supervisor identifies more parts than the cap, it
answers the first 5 and the responder states plainly which parts it did not cover and invites a
follow-up. A truncated answer that admits truncation is fine; one that quietly drops half the question
is the actual failure mode. Raising the cap is a one-line config change if the demo warrants it.

**Errors** are handled by a `@safe_node` decorator that wraps every node, writes `state.error`, and
routes to `responder` — no separate fallback node.

Deliberately *not* nodes, and the README says why: intake guard (in `supervisor`), validator (in
`extractor`), anomaly/details (in `analyst`'s dispatch), synthesis and numeric verification (in
`responder`), clarifier and general-knowledge (responder modes), fallback (the error decorator).
The graph still earns "multi-agent": conditional routing out of `supervisor`, a short-circuit edge
from `extractor`, and a real cycle for compound questions. Knowing what not to make a node is the
more senior signal — an interviewer who counts boxes is better answered by a diagram they can hold
in their head.

### State — the shared clipboard

Nodes never call each other. Each one receives this dict, returns only the fields it changed, and
LangGraph merges the update and routes onward. It is also what the conditional edges read to decide
where to go, and what the UI's Agent Trace panel renders.

| field | what's in it (Building 120/180 example) | written by |
|---|---|---|
| `question` | the raw user message | entry |
| `history` | earlier turns, so "and last year?" resolves | entry |
| `intent` | `comparison` | supervisor |
| `pending` | job queue `[compare-buildings, anomaly-scan]`; extractor pops one per pass, loop ends when empty | supervisor writes, extractor drains |
| `results` | what the calculator returned for each finished job | analyst appends |
| `clarification_reason` | `None`, or e.g. `unknown_entity: "Building 999"` — the flag that skips the analyst and puts the responder in clarify mode | extractor |
| `answer` | final English text | responder |
| `trace` | breadcrumbs: node visited, query form, rows matched, subtotals → the UI's trace panel | every node |
| `error` | set by `@safe_node` when a node throws, so the responder apologises instead of the app crashing | error decorator |

`TypedDict` = a plain dict with declared field types, so a typo like `state["quesiton"]` is caught by
the type checker rather than at runtime in front of the interviewer.

```python
class AgentState(TypedDict):
    question: str
    history: list[BaseMessage]
    intent: Intent | None
    pending: list[SubQuestion]        # drained by the extractor→analyst loop
    results: list[BranchResult]
    clarification_reason: ClarifyReason | None   # set by extractor grounding, read by responder
    answer: str
    trace: list[TraceStep]            # rendered in the UI's Agent Trace panel
    error: str | None
```

## Project structure

```
real-estate-asset-agent/
├── app.py                      # Streamlit chat UI
├── requirements.txt
├── README.md
├── PLAN.md                     # this plan, committed as the design record
├── .env.example  .gitignore  .streamlit/config.toml
├── data/cortex.parquet
├── src/
│   ├── config.py               # settings + get_llm(tier) factory (provider-swappable)
│   ├── schemas.py              # Intent, SubQuestion, QuerySpec, TimeRange, Finding, BranchResult
│   ├── formatting.py           # currency/percent, tables, trace rendering
│   ├── data/
│   │   ├── loader.py           # cached parquet load + month/quarter parsing to periods
│   │   ├── catalog.py          # distinct values, coverage window, compact "schema card" for prompts
│   │   └── resolver.py         # rapidfuzz canonical resolution + unresolved reporting
│   ├── tools/
│   │   ├── metrics.py          # pnl, aggregate, timeseries, top_n, compare
│   │   ├── details.py          # property / tenant detail cards
│   │   └── anomalies.py        # reversals, outliers, gaps, concentration
│   └── graph/
│       ├── state.py
│       ├── nodes.py            # supervisor, extractor, analyst, responder
│       │                       # (+ @safe_node error decorator)
│       ├── prompts.py          # incl. the responder's answer/clarify/knowledge modes
│       └── build.py            # StateGraph wiring, conditional edges, checkpointer
└── tests/
    ├── test_metrics.py         # golden numbers from the verified facts above
    ├── test_resolver.py
    └── test_graph_scenarios.py # 15 canned questions incl. the unanswerable ones
```

## Build order (incremental — one step per session, each independently runnable)

0. **Commit the plan** — write this document to `PLAN.md` at the project root so the design record ships
   with the code and the README can link to it.
1. **Data foundation** — `loader`, `catalog`, `resolver`, `requirements.txt`, repo skeleton + git init.
   Checkpoint: `python -m src.data.catalog` prints the schema card; resolver tests pass.
2. **Deterministic tools** — `metrics`, `details`, `anomalies` + unit tests asserting the golden
   numbers above (1,533,331.87 net; 2024 vs 2025-Q1; Tenant 7 top; the ±154,415.07 reversal pair).
   Checkpoint: `pytest` green, zero LLM calls so far.
3. **LLM layer** — `config.get_llm`, `schemas.py`, `supervisor` + `extractor` with
   `with_structured_output` on `gemini-3.1-flash-lite`. Checkpoint: script prints correct QuerySpecs for
   ~15 sample questions; if decomposition or extraction is shaky, raise `LLM_MODEL_SMART` and re-run.
4. **Graph** — `build.py`, conditional edges, the sub-question loop, `@safe_node`, MemorySaver.
   Checkpoint: CLI `python -m src.graph.run "…"` answers all 15, including the unanswerable ones.
5. **Responder** — the three prompt modes, derivation steps, numeric grounding post-check + repair retry.
6. **Streamlit UI** — chat, example-question chips, expandable **Agent Trace** (nodes visited, the
   QuerySpec, rows matched, intermediate sums), result table + chart, caveat banner, thread memory.
7. **Ship** — README (setup, architecture diagram, workflow, design rationale, challenges),
   push to GitHub, deploy to Streamlit Cloud, smoke-test the live URL.

## Assumptions (to be stated in the README)

1. **The workspace is pinned to an as-of reporting date: `REPORTING_AS_OF = 2025-03-31`** (the last
   period in the ledger). All relative dates resolve against it — "this year" → 2025 YTD (Jan–Mar),
   "this quarter" → 2025-Q1, "same period last year" → 2024-Q1. Rationale: today is Sept 2026 and the
   ledger ends Mar 2025, so a literal clock reading makes three of the brief's four example questions
   return zero rows. Real accounting tools work this way too — you view a closed book as of a date,
   not today. The Streamlit header displays **"Reporting period: as of 31 Mar 2025"** so nothing is
   hidden, every answer names the period it used, and the constant is the single thing to change when
   fresher data lands.
2. `profit` is the single signed measure; P&L = `sum(profit)`, revenue/expenses = the same sum split by
   `ledger_type`. No separate cost basis, valuation, or cash-flow column exists.
3. Rows with null `property_name` are entity-level costs; they are **excluded** from per-property figures
   and **included** in portfolio totals — every per-property answer says so.
4. **Duplicate rows are never deduped**, because they are re-postings and reversal chains that net
   correctly under signed summation — deduping them destroys real revenue (evidence above). The one
   genuine double-count, `ledger_code 4650` appearing under two categories, is reported as a finding
   rather than silently corrected, so the agent's numbers always tie back to the raw ledger.
5. "Price / value / appraisal / cap rate / occupancy / address" questions are structurally unsupported;
   the agent says so plainly and offers the nearest supported answer.
6. `Building 17` (not 170) is a real distinct property, not a typo — kept as-is.
7. Single entity (`PropCo`), so entity filtering is a no-op kept for future-proofing.
8. 2025 is a partial year (Q1 only) — any year-over-year comparison auto-switches to a like-for-like
   Q1-vs-Q1 basis and states the adjustment.

## Open questions

- **B.** How much should the UI show? Plan is chat + collapsible agent trace + one result table/chart.
  A full graph visualisation is possible but costs time better spent on edge cases.
- **C.** Should P&L exclude the entity-level overhead rows by default (property-level view) or include
  them (portfolio view)? Plan: include in portfolio totals, exclude from per-property, always disclosed.

## Verification

- `pytest tests/` — golden numbers, resolver behaviour, and a scenario suite that runs the full graph
  against 15 canned questions with mocked LLM output where determinism matters.
- Manual matrix through the Streamlit UI, one per pathway:
  - simple P&L — "What is the total P&L for all my properties this year?" → 361,810.32 for 2025-Q1 + partial-year caveat
  - comparison — "Compare Building 120 and Building 180"
  - period comparison — "How does this quarter compare to the same period last year?" → 2025-Q1 vs 2024-Q1
  - ranking + anomaly (compound) — "Who are my top tenants, and is anything unusual in the numbers?"
  - overflow (6+ parts in one message) — answers the first 5 and names the parts it skipped
  - unsupported metric (supervisor short-circuit) — "What is the price of my asset at 123 Main St?" → names the gap, offers alternatives
  - unknown entity (extractor grounding) — "How did Building 999 do?" → lists the five real properties
  - capability question (short-circuit) — "What data do you have?" → catalog + coverage window
  - general knowledge (short-circuit) — "How is NOI calculated?" → labelled as not-from-your-ledger
  - vague — "How are things going?" → one targeted clarifying question
  - out-of-scope — "Write me a poem" → polite redirect
  - follow-up memory — "…and what about last year?" after a prior turn
- Deployed URL smoke test: cold start, one question per pathway, confirm secrets are wired and no key is
  committed.
