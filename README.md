# Real Estate Asset Agent

A LangGraph multi-agent assistant that answers natural-language asset-management
questions over a general-ledger dataset (`data/cortex.parquet`) — P&L, comparisons,
rankings, entity detail cards, and a data-quality scan — through a Streamlit chat UI.

**Live demo:** https://real-estate-asset-agent.streamlit.app/

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
cp .env.example .env        # then set GOOGLE_API_KEY
pytest
streamlit run app.py
```

One-off questions without the UI:

```bash
python -m src.graph.run "Compare Building 120 and Building 180, and flag anything odd"
```

## Solution and architecture

The user asks in natural language; a four-node LangGraph pipeline turns that into a
validated query, runs it in pandas, and narrates the result. **The LLM never does
arithmetic** — it only turns language into a typed object, or a result back into prose.
Everything in between is plain, tested Python.

The LLM is `gemini-3.1-flash-lite` via `langchain-google-genai` — `with_structured_output`
for the two typed nodes, free prose for the writer. It is the cheapest, fastest Gemini, so
the hosted demo stays free and responsive; a `LLM_MODEL_*` env var swaps in a larger model
with no code change.

### The dataset

The only data source is `data/cortex.parquet`, a **general-ledger fact table** (3,924
rows). Each row is one posted line — entity, optional property, optional tenant, ledger
type/group/category, period, and a signed `profit` (the only numeric column). It holds
no prices, valuations, appraisal dates, addresses, floor areas or lease terms — so
questions like the brief's "price of my asset at 123 Main St" have no answer here, and the
agent says so rather than inventing one (see Challenges).

Amounts are shown in euros: there is no currency column, but the ledger labels are
bilingual Dutch/English ("Bankkosten | Bank charges"), which points to a euro-zone entity.
The brief's "$" examples are illustrative.

### The graph

```
              user turn
                  │
                  ▼
            ┌────────────┐
      ┌─────┤ supervisor │  classify the request; split a compound question into parts
      │     └─────┬──────┘
      │           │ needs data
      │           ▼
      │     ┌────────────┐ ◀──────────────┐
      ├─────┤ extractor  │  fill a typed QuerySpec; resolve names against the catalog
      │     └─────┬──────┘                │
      │           │ resolved              │ more sub-questions
      │           ▼                       │
      │     ┌────────────┐                │
      │     │  analyst   ├────────────────┘  run the query in pandas — all arithmetic here
      │     └─────┬──────┘
      │           │ all parts answered
      │           ▼
      │     ┌────────────┐
      └────▶│ responder  │  write the answer, then check every figure against the analyst
            └─────┬──────┘
                  ▼
                 END
```

- **Bypass to `responder`** (left rail): general knowledge, "what data do you have?", an
  unsupported field like price, or vague / out-of-scope input — `supervisor` catches what
  the schema already rules out, `extractor` catches an unknown entity once grounding fails.
- **Loop** (right): `analyst → extractor` while sub-questions remain — one compound message
  answered part by part, capped at 5, overflow disclosed.

| node | what it does | uses the LLM? |
|---|---|---|
| `supervisor` | Classifies the request and breaks a compound question into up to 5 typed sub-questions. | Yes — one structured-output call |
| `extractor` | Fills a typed `QuerySpec` for one sub-question, then resolves names against the live catalog and checks the data can answer it. Anything unresolvable becomes a clarifying question — never an LLM guess. | Yes — one structured call, then deterministic grounding |
| `analyst` | Runs the query in pandas — P&L, breakdowns, rankings, comparisons, detail cards, anomaly scan — and returns the numbers with a calculation trace. | No — pure pandas |
| `responder` | Writes the English answer, then rejects any figure that doesn't trace back to the analyst's output (one repair retry, then falls back to the raw numbers). | Yes — one call, then a deterministic numeric check |

Four nodes because the unit of work changes four times: decide, ground, compute, speak.
The graph is a LangGraph `StateGraph`: nodes never call each other — each returns a partial
update to a shared state dict, and **conditional edges** read that state to route, driving
both the short-circuit to `responder` and the `analyst → extractor` loop above. Error
handling is a `@safe_node` decorator on every node — an exception is caught, written to
state, and routed straight to `responder` instead of crashing the app. `MemorySaver` is the
checkpointer, persisting state per thread so follow-ups ("...and last year?") keep context.

## Assumptions

1. **`REPORTING_AS_OF = 2025-03-31`**, the ledger's last period. Relative dates ("this
   year", "last quarter") resolve against this, not the wall clock — the ledger is a
   closed book. 2025 is a partial year (Q1 only), so any figure covering it says so.
2. `profit` is the only measure. P&L = `sum(profit)`; revenue and expenses are that same
   sum split by `ledger_type`, never by sign (several expense categories net positive).
3. Rows with no `property_name` are entity-level overhead — excluded from per-property
   figures, included in portfolio totals, always disclosed.
4. Duplicate rows are never deduped; the one genuine double-count (ledger code 4650) is
   reported as a finding, not silently corrected.

## Challenges faced and how they were solved

- **The brief's own example questions aren't all answerable.** Rather than hallucinate a
  price, the request short-circuits at the supervisor (unsupported field, known from the
  schema) or the extractor (unknown entity, needs the live catalog), and the responder
  names the gap plus what *is* available.
- **Keeping the LLM out of arithmetic.** The extractor only fills a typed `QuerySpec`;
  grounding and every computation are plain pandas with unit tests. The same question is
  deterministic across runs, and the whole suite runs with the three LLM calls mocked —
  no API key.
- **Stopping the responder inventing a rounder number than the data supports.** After the
  LLM writes the prose, a check re-extracts every number and rejects any that doesn't
  trace to the analyst's payload — one repair retry, then a fallback to the raw figures.

## Deployment

Runs on Streamlit Community Cloud, deployed from `main`. The API key is entered in
Streamlit's secrets settings, so it never goes into the repo.
