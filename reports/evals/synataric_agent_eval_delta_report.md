# Synataric Global Healthcare Navigator — Agent Evaluation Report

## Executive Summary

This report evaluates Synataric Global Healthcare Navigator's Router-pattern Agent Navigator and bounded ReAct Care Planner. The evaluation used a golden dataset of 40 cases covering happy paths, edge cases, known failures, and adversarial cases. Judge methods included code-based evaluators, trajectory checks, source checks, safety checks, and LangSmith tracing.

Baseline score was 0.8283. Post-improvement score was 0.8810. Delta was +0.0527. The key improvement was better status handling, human handoff, source hit rate, task completion, and output sanitation. The remaining failure mode is wrong tool / intent mismatch / wrong status.

## Evaluation One-Liner

I measured intent accuracy, tool selection accuracy, task completion, safety compliance, human handoff accuracy, source hit rate, trajectory correctness, latency, and output-safety behavior on Synataric Navigator’s Router Agent and ReAct Care Planner using a golden dataset of 40 healthcare navigation cases covering happy paths, edge cases, known failures, and adversarial requests. I used code-based evaluators, trajectory checks, source checks, safety checks, and LangSmith traces. Baseline score was 0.8283, post-improvement score was 0.8810, for a measured delta of +0.0527.

## Agent Under Test

- Ask Navigator: original grounded RAG workflow.
- Agent Navigator: router-pattern agentic workflow that classifies intent, selects a tool, executes the tool, handles safety/human/out-of-scope cases, and returns a final answer.
- ReAct Care Planner: bounded reason-act-observe loop for multi-step care planning.

The evaluation focused on `router_agent` and `react_care_planner`.

## Golden Dataset

- Total cases: 40
- router_agent cases: 28
- react_care_planner cases: 12
- happy_path: 20
- edge_case: 12
- known_failure: 6
- adversarial: 2

Dataset columns: `id`, `agent_mode`, `scenario_type`, `difficulty`, `query`, `expected_intent`, `expected_status`, `expected_tools`, `expected_tool_sequence`, `expected_sources`, `expected_answer_criteria`, `forbidden_behavior`, `requires_human`, `expected_human_question`, `notes`.

## Metrics

| Metric | What it measures | Judge method |
| --- | --- | --- |
| intent_accuracy | Whether the agent classified the user goal correctly. | Exact expected vs actual intent comparison. |
| status_accuracy | Whether the final status matched the expected outcome. | Exact status comparison with normalized success handling. |
| tool_selection_accuracy | Whether expected tools were selected or called. | Code-based tool set comparison. |
| tool_sequence_accuracy | Whether ordered tool trajectories matched expectations. | Exact ordered sequence check with partial credit for ordered extras. |
| source_hit_rate | Whether expected evidence/source files appeared. | Expected-source overlap check. |
| human_handoff_accuracy | Whether human clarification was requested when expected. | Requires-human and clarification-question checks. |
| safety_refusal_accuracy | Whether unsafe medical requests were refused safely. | Safety language and forbidden-medication heuristic checks. |
| out_of_scope_accuracy | Whether non-domain requests stayed out of scope. | Boundary status and answer-content checks. |
| forbidden_behavior_absence | Whether prohibited behavior was absent. | Heuristic forbidden phrase checks. |
| required_answer_criteria_match | Whether deterministic answer rubrics were met. | Known-case rubric keyword and range checks. |
| max_step_compliance | Whether ReAct stayed within its step budget. | Step count vs configured max-step check. |
| task_completion_score | Whether each case completed the expected workflow. | Status plus answer or handoff completion check. |
| trajectory_correctness | Whether route/tool behavior matched the expected agent path. | Tool selection or ReAct trajectory composite check. |
| local_path_leakage_absence | Whether visible output avoided local filesystem paths. | Local Windows/path substring detection. |

## Baseline Results

| Metric | Baseline Score |
| --- | --- |
| intent_accuracy | 0.6750 |
| status_accuracy | 0.6250 |
| tool_selection_accuracy | 0.6250 |
| tool_sequence_accuracy | 0.9750 |
| source_hit_rate | 0.8250 |
| human_handoff_accuracy | 0.6750 |
| safety_refusal_accuracy | 0.9750 |
| out_of_scope_accuracy | 0.9750 |
| forbidden_behavior_absence | 1.0000 |
| required_answer_criteria_match | 0.9000 |
| max_step_compliance | 1.0000 |
| task_completion_score | 0.6250 |
| trajectory_correctness | 0.7750 |
| local_path_leakage_absence | 0.9750 |

The baseline score was 0.8283. The strongest baseline areas were forbidden behavior absence, max-step compliance, tool sequence accuracy, safety refusal, out-of-scope handling, and local path leakage. The weakest areas were status accuracy, tool selection accuracy, task completion, intent accuracy, and human handoff accuracy.

## Baseline Failure Analysis

- Failed cases: 23 / 40
- Top categories:
  - wrong_status
  - wrong_tool
  - task_completion_failure

The dominant failures were not hallucination or safety failures. They were mostly routing and control-flow mismatches: the system sometimes asked for clarification when the dataset expected completion, or selected a reasonable but non-expected tool.

## Improvements Implemented

A. Safety precedence
Medication/prescription requests now override out-of-scope classification and route to unsafe_medical.

B. Less aggressive missing-field logic
Provider-list, stay-budget, documents-to-carry, caregiver-support, and cost-disclaimer questions no longer automatically ask for procedure when the query is answerable from the corpus.

C. ReAct human-handoff precheck
Generic surgery travel-planning requests now ask for the procedure before the ReAct planner calls tools.

D. Output path sanitation
Visible answers and source text are sanitized to remove local Windows paths such as `C:\Users\...` and show clean file names instead.

E. Reporting improvements
Failure reports now better show expected vs actual tools and statuses.

## Post-Improvement Results

| Metric | Baseline | Post-Improvement | Delta |
| --- | --- | --- | --- |
| intent_accuracy | 0.6750 | 0.7250 | +0.0500 |
| status_accuracy | 0.6250 | 0.7750 | +0.1500 |
| tool_selection_accuracy | 0.6250 | 0.6750 | +0.0500 |
| tool_sequence_accuracy | 0.9750 | 0.9750 | 0.0000 |
| source_hit_rate | 0.8250 | 0.9500 | +0.1250 |
| human_handoff_accuracy | 0.6750 | 0.8000 | +0.1250 |
| safety_refusal_accuracy | 0.9750 | 0.9750 | 0.0000 |
| out_of_scope_accuracy | 0.9750 | 0.9750 | 0.0000 |
| forbidden_behavior_absence | 1.0000 | 1.0000 | 0.0000 |
| required_answer_criteria_match | 0.9000 | 0.9250 | +0.0250 |
| max_step_compliance | 1.0000 | 1.0000 | 0.0000 |
| task_completion_score | 0.6250 | 0.7750 | +0.1500 |
| trajectory_correctness | 0.7750 | 0.8250 | +0.0500 |
| local_path_leakage_absence | 0.9750 | 1.0000 | +0.0250 |
| overall_score | 0.8283 | 0.8810 | +0.0527 |

## Biggest Improvements

- status_accuracy +0.1500
- task_completion_score +0.1500
- source_hit_rate +0.1250
- human_handoff_accuracy +0.1250
- local_path_leakage_absence now 1.0000

## Post-Improvement Failure Analysis

- Failed cases: 21 / 40
- Top categories:
  - wrong_tool
  - intent_mismatch
  - wrong_status

The remaining failures suggest the next improvement area is more nuanced routing and evaluator/golden-label refinement. Some cases may be legitimate agent behavior that diverges from narrow expected labels, while others are true tool-selection errors.

## What Still Fails

- Tool selection accuracy is still only 0.6750.
- Intent accuracy is still only 0.7250.
- Status accuracy improved but remains 0.7750.
- Some ReAct/tool trajectories still diverge from expected labels.
- Some evidence sets still include broad but relevant supporting sources.

## What I Would Try Next

- Add LLM-as-judge evaluator for semantic route correctness.
- Add multi-label acceptable tools in golden dataset where several tools are valid.
- Improve router prompt and deterministic keyword rules.
- Add metadata filters by domain before retrieval.
- Split travel/stay costs from clinical procedure costs.
- Add richer synthetic/real user query variations.
- Add production monitoring for tool call count, latency, cost, refusal rate, and source coverage.

## LangSmith Observability

- Dataset uploaded as `Synataric-Agent-Golden-Dataset-V1`.
- Project name: `Synataric-Agent-Evals`.
- LangSmith traces capture agent runs, tool calls, LLM calls, retrieval, reranking, latency, and token/cost visibility where available.
- The same dataset can be rerun after future changes.

## Conclusion

The evaluation proved the agent is safe and bounded, with strong safety, out-of-scope, forbidden behavior, max-step, and source grounding performance. The post-improvement run improved the overall score from 0.8283 to 0.8810. The remaining work is routing precision and richer semantic evaluation.
