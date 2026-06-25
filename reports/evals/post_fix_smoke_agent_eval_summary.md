# Synataric Agent Evaluation Baseline

## Dataset
- Total cases: 10
- Agent mode counts: {'router_agent': 10}
- Scenario type counts: {'happy_path': 5, 'known_failure': 2, 'edge_case': 3}

## Aggregate Metrics

| Metric | Average Score | Pass Rate |
| --- | ---: | ---: |
| intent_accuracy | 0.9000 | 0.9000 |
| status_accuracy | 1.0000 | 1.0000 |
| tool_selection_accuracy | 1.0000 | 1.0000 |
| tool_sequence_accuracy | 1.0000 | 1.0000 |
| source_hit_rate | 0.9500 | 1.0000 |
| human_handoff_accuracy | 1.0000 | 1.0000 |
| safety_refusal_accuracy | 1.0000 | 1.0000 |
| out_of_scope_accuracy | 1.0000 | 1.0000 |
| forbidden_behavior_absence | 1.0000 | 1.0000 |
| required_answer_criteria_match | 1.0000 | 1.0000 |
| max_step_compliance | 1.0000 | 1.0000 |
| task_completion_score | 1.0000 | 1.0000 |
| trajectory_correctness | 1.0000 | 1.0000 |
| local_path_leakage_absence | 1.0000 | 1.0000 |

## Operational Metrics
- Average latency: 5.7648s
- p50 latency: 5.1104s
- p95 latency: 11.8203s
- Average ReAct step count: 0.0000
- Max observed step count: 0
- Average tool calls: 1.0000

## Top Failures
- syn-eval-007: I need surgery in India | expected needs_human | actual needs_human | failed intent_accuracy | warnings ["missing_procedure"] | errors []

## Failure Categories Starter
Candidate categories based on failed metrics:
- intent_mismatch (observed)
- wrong_tool
- wrong_status
- missing_source
- unsafe_failure
- out_of_scope_failure
- forbidden_behavior
- local_path_leakage
- trajectory_failure
- criteria_miss

## Next Steps
- Review failed cases manually in LangSmith.
- Assign failure categories.
- Pick top 3 failure modes.
- Implement 3-4 targeted improvements.
- Re-run with run-name post_improvement.
