# Demo Mode Executive Banner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact executive storytelling stack to the top of Synataric Demo Mode without changing backend runtime behavior.

**Architecture:** Keep all new behavior in `src/demo_mode.py` as plain data helpers plus Streamlit-native render functions. Preserve existing scenario runner, workflow timeline, care plan cards, evidence, benchmark details, and pitch script. Tests assert the new data contracts and guard against raw visible HTML and local path leakage.

**Tech Stack:** Python, Streamlit native components, pytest.

---

### Task 1: Tests For Executive Story Data

**Files:**
- Modify: `tests/test_demo_mode.py`

- [ ] **Step 1: Add failing tests**

Add imports for `build_architecture_snapshot_cards`, `build_agentic_callout`, `build_executive_metric_cards`, and `build_executive_story_content`. Add tests asserting card titles, metric values, ReAct tool names, no raw HTML fragments, and sanitized local path text.

- [ ] **Step 2: Run focused tests to verify failure**

Run: `python -m pytest tests/test_demo_mode.py -q`
Expected: FAIL with import errors for the new helpers.

### Task 2: UI-Only Helpers And Top Renderer

**Files:**
- Modify: `src/demo_mode.py`

- [ ] **Step 1: Implement plain data helpers**

Add helper functions returning dictionaries/lists for executive banner copy, Why Synataric cards, architecture cards, agentic callout, measured improvement cards, recording script, and production vision text.

- [ ] **Step 2: Implement Streamlit-native top renderer**

Add `render_executive_demo_story(metrics)` that uses `st.container(border=True)`, `st.columns`, `st.markdown`, `st.caption`, `st.metric`, and one non-nested `st.expander`. Do not modify agent files or runtime routing.

- [ ] **Step 3: Call renderer at top of Demo Mode**

In `render_demo_mode_page`, call `render_executive_demo_story(metrics)` immediately after CSS injection and before the existing header/scenario runner content.

### Task 3: Verification

**Files:**
- Verify: `app.py`
- Verify: `src/demo_mode.py`
- Verify: `tests/test_demo_mode.py`

- [ ] **Step 1: Run compile checks**

Run: `python -m py_compile app.py`
Run: `python -m py_compile src/demo_mode.py`
Expected: both commands exit 0.

- [ ] **Step 2: Run focused tests**

Run: `python -m pytest tests/test_demo_mode.py -q`
Expected: all tests pass without live OpenAI or Pinecone calls.

- [ ] **Step 3: Start Streamlit**

Run: `streamlit run app.py`
Expected: Demo Mode can be opened for manual acceptance.
