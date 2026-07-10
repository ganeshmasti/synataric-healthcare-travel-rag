# Command Center Light Theme And Route Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine Demo Mode into a light, compact healthcare command center with route-specific results for every supported scenario.

**Architecture:** Keep runtime behavior unchanged and limit implementation to `src/demo_mode.py`, `tests/test_demo_mode.py`, and `.streamlit/config.toml`. Add testable helper data for compact pipeline, metrics, MINT cards, and route-specific result cards; render these with Streamlit-native components.

**Tech Stack:** Python, Streamlit, pytest, Streamlit theme config.

---

### Task 1: Tests For Compact Command Center Contracts

**Files:**
- Modify: `tests/test_demo_mode.py`

- [ ] **Step 1: Add failing tests**

Add tests that expect the compact pipeline nodes, five KPI cards, route-specific result data for care plan, cost, provider, recovery, risk, safety, human clarification, out-of-scope, and coverage gap.

- [ ] **Step 2: Run tests red**

Run: `python -m pytest tests/test_demo_mode.py -q`
Expected: failures because compact helper data and route-specific result data are not complete yet.

### Task 2: Implement Light Theme And Data Helpers

**Files:**
- Create: `.streamlit/config.toml`
- Modify: `src/demo_mode.py`

- [ ] **Step 1: Add Streamlit theme config**

Create `.streamlit/config.toml` with the requested light medical theme tokens.

- [ ] **Step 2: Add `STYLE_BLOCK`**

Move CSS into one module-level `STYLE_BLOCK` and render it from `inject_demo_medical_css()`.

- [ ] **Step 3: Update helper data**

Make `get_architecture_pipeline_nodes()` return the compact seven-node pipeline and `get_metric_cards()` return the five requested KPI cards.

### Task 3: Route-Specific Result Rendering

**Files:**
- Modify: `src/demo_mode.py`

- [ ] **Step 1: Add route result data helper**

Add `get_route_specific_result_cards(route_key)` returning sanitized card data for cost, provider, recovery, risk, safety, human clarification, out-of-scope, coverage gap, and multi-step care plan.

- [ ] **Step 2: Render route-specific results**

Add `render_route_specific_result(result, scenario_key, question)` and route-specific wrappers. Keep all display text sanitized and use Streamlit-native components.

- [ ] **Step 3: Neutralize pre-run console**

Before a run, show only “MINT will select the lightest workflow that can satisfy the request.” Show actual routing decision only after run.

### Task 4: Verification

**Files:**
- Verify: `app.py`
- Verify: `src/demo_mode.py`
- Verify: `tests/test_demo_mode.py`

- [ ] **Step 1: Compile**

Run: `python -m py_compile app.py`
Run: `python -m py_compile src/demo_mode.py`
Expected: both pass.

- [ ] **Step 2: Tests**

Run: `python -m pytest tests/test_demo_mode.py -q`
Expected: all focused tests pass without live OpenAI or Pinecone calls.

- [ ] **Step 3: Streamlit**

Run or verify `streamlit run app.py`
Expected: app responds locally and Demo Mode shows the refined light command center.
