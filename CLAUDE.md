# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this is

A LangGraph multi-agent prototype (take-home) that answers natural-language real-estate
asset-management questions over `data/cortex.parquet`, a general-ledger fact table. Full
design record: [PLAN.md](PLAN.md). Architecture, dataset facts, and rationale: [README.md](README.md).

## Ground rules

- **The LLM never does arithmetic.** `supervisor` and `extractor` only ever emit validated
  Pydantic objects (`SupervisorPlan`, `QuerySpec`); `analyst` is pure pandas; `responder`'s
  prose is fact-checked against the analyst's payload after the fact. Don't add an LLM call
  that computes or fabricates a number — add a pandas function in `src/tools/` instead, with
  a unit test.
- **Never dedupe ledger rows.** Verified: duplicates are posting/reversal chains that net
  correctly under signed summation; deduping deletes real revenue (see README's dataset
  section). If a task looks like "clean up the duplicates," stop and re-read that section
  first.
- **Golden numbers are regression tests, not approximations.** Net P&L 1,533,331.87,
  revenue 2,887,652.89, expenses −1,354,321.02, etc., are asserted in
  `tests/test_metrics.py`. If a code change moves one of these, that's a bug unless the
  test itself is being deliberately updated with justification.
- **Run `pytest` before considering a change done.** The full suite (86 tests) mocks all
  three LLM calls, so it requires no `GOOGLE_API_KEY` and makes zero network calls — there's
  no excuse to skip it.
- **Keep the four-node shape unless there's a real reason to change it.** `PLAN.md`
  documents what was deliberately *not* made a node (intake guard, validator, clarifier,
  fallback) and why. Adding a node should map to a genuine new unit of work, not a
  refactor-for-its-own-sake.

## Commands

```bash
pytest                                          # full suite, no API key needed
streamlit run app.py                            # UI
python -m src.graph.run "your question"         # CLI, needs GOOGLE_API_KEY
python -m src.data.catalog                       # print the live schema card
```
