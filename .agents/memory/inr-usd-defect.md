---
name: INR/USD cost display defect
description: Cost estimates are stored and displayed in INR only — no USD equivalent is shown, despite the app promising "vs the US" comparisons.
---

# INR / USD Cost Display Defect

## The rule
Any cost display in the app must show **both** INR and an approximate USD equivalent. Showing INR-only is a known defect.

**Why:** The sample question "What does cataract surgery cost in Bangalore vs the US?" implies USD context. The current data model only has `low_estimate_inr` / `high_estimate_inr` columns with no USD equivalent, and all demo cards render "INR 45,000 - INR 150,000" with no dollar figure alongside it.

## Where the defect surfaces
- `src/demo_mode.py` — `get_route_specific_result_cards()` at lines ~1659 and ~1700: hardcoded "INR 45,000 - INR 150,000" strings with no USD
- `src/sample_data.py` — CSV schema uses `low_estimate_inr` / `high_estimate_inr` columns only; no `low_estimate_usd` / `high_estimate_usd`
- `src/agent_tools.py` — cost tool scores on INR fields only
- `src/evaluation.py` — benchmark expected answers reference INR only

## How to fix
1. Add `low_estimate_usd` / `high_estimate_usd` columns to the procedure and travel cost CSVs in `src/sample_data.py` (use illustrative USD values; note exchange-rate disclaimer)
2. Update `get_route_specific_result_cards()` in `src/demo_mode.py` to display both currencies, e.g. "INR 45,000–150,000 (~USD 540–1,800)"
3. Update cost tool scoring and evaluation expected answers to reflect dual-currency display
4. Add a disclaimer that exchange rates are illustrative and change over time
