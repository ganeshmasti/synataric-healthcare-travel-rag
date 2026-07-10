# Synataric Command Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign Demo Mode into a compact, tabbed Synataric Command Center without changing runtime agent, RAG, router, safety, or evaluation behavior.

**Architecture:** Keep the change inside `src/demo_mode.py` by separating display data helpers from Streamlit-native render helpers. The Live Demo tab becomes the first-screen command center; architecture, evaluation, roadmap, and presenter notes move into secondary tabs. Tests validate helper data, metrics, sanitization, and no raw helper display text.

**Tech Stack:** Python, Streamlit native components, pytest.

---

### Task 1: Tests For Command Center Data Contracts

**Files:**
- Modify: `tests/test_demo_mode.py`

- [ ] **Step 1: Write failing tests**

Add imports and assertions for `get_architecture_pipeline_nodes`, `get_component_summary_rows`, `get_metric_cards`, `get_mint_ladder_cards`, `get_demo_tool_flow`, and `get_eval_delta_rows`.

- [ ] **Step 2: Verify red**

Run: `python -m pytest tests/test_demo_mode.py -q`
Expected: import errors or assertion failures because the new helpers are not implemented yet.

### Task 2: Command Center Helpers And Tabs

**Files:**
- Modify: `src/demo_mode.py`

- [ ] **Step 1: Add helper data functions**

Implement plain Python helpers for the architecture pipeline, component summary, KPI cards, compact MINT cards, ReAct tool flow, and evaluation delta rows.

- [ ] **Step 2: Add Streamlit-native render functions**

Implement `render_command_center_hero`, `render_architecture_pipeline`, `render_metrics_strip`, `render_mint_ladder`, `render_demo_console`, `render_react_reason_card`, `render_workflow_timeline`, `render_result_summary`, `render_architecture_details_tab`, `render_evaluation_details_tab`, `render_production_roadmap_tab`, and `render_presenter_notes_tab`.

- [ ] **Step 3: Wire tabbed page**

Update `render_demo_mode_page` to create Live Demo, Architecture Details, Evaluation Details, Production Roadmap, and Presenter Notes tabs. Keep the default Live Demo focused on the hero, pipeline, metrics, MINT ladder, demo console, and result area.

### Task 3: Verification

**Files:**
- Verify: `app.py`
- Verify: `src/demo_mode.py`
- Verify: `tests/test_demo_mode.py`

- [ ] **Step 1: Compile checks**

Run: `python -m py_compile app.py`
Run: `python -m py_compile src/demo_mode.py`
Expected: both exit 0.

- [ ] **Step 2: Focused tests**

Run: `python -m pytest tests/test_demo_mode.py -q`
Expected: all tests pass without live OpenAI or Pinecone calls.

- [ ] **Step 3: Streamlit manual review**

Run: `streamlit run app.py`
Expected: Demo Mode loads as the Synataric Command Center and preserves existing scenarios.
