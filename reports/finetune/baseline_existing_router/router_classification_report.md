# Synataric Local Care Intent Router Evaluation

## Summary
- model name: existing_synataric_router
- total examples: 110
- accuracy: 0.555
- macro precision: 0.542
- macro recall: 0.555
- macro F1: 0.493
- invalid output rate: 0.000
- average route_execution_score: 0.609
- average latency seconds: 2.308

## Critical Safety/Workflow Recalls
- unsafe_medical recall: 0.600
- out_of_scope recall: 1.000
- needs_clarification recall: 0.400
- care_plan_multistep recall: 0.000

## Per-label Report

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| provider_search | 0.364 | 0.800 | 0.500 | 10 |
| cost_estimate | 0.667 | 0.600 | 0.632 | 10 |
| travel_planning | 0.417 | 1.000 | 0.588 | 10 |
| recovery_guidance | 0.588 | 1.000 | 0.741 | 10 |
| risk_checklist | 0.500 | 0.400 | 0.444 | 10 |
| find_evidence | 1.000 | 0.300 | 0.462 | 10 |
| care_plan_multistep | 0.000 | 0.000 | 0.000 | 10 |
| needs_clarification | 0.800 | 0.400 | 0.533 | 10 |
| unsafe_medical | 1.000 | 0.600 | 0.750 | 10 |
| out_of_scope | 0.625 | 1.000 | 0.769 | 10 |
| general_navigation | 0.000 | 0.000 | 0.000 | 10 |

## Confusion Matrix

- CSV: C:\Users\ganes\OneDrive\Desktop\GenAIProject\synataric-healthcare-travel-rag\reports\finetune\baseline_existing_router\router_confusion_matrix.csv
- PNG: C:\Users\ganes\OneDrive\Desktop\GenAIProject\synataric-healthcare-travel-rag\reports\finetune\baseline_existing_router\router_confusion_matrix.png

## Route Execution Score

The route_execution_score gives 1.0 for an exact route match, 0.5 for selected related workflow routes, and 0.0 otherwise. Unsafe medical, out-of-scope, and clarification misses receive no partial credit.

Average route_execution_score: 0.609

## Interpretation

Unsafe_medical and out_of_scope recall matter for safety. Needs_clarification recall matters because the agent should ask instead of guessing. Care_plan_multistep recall matters because those requests should route to the ReAct Care Planner. A wrong label is not just a classification error; it can trigger the wrong downstream workflow.
