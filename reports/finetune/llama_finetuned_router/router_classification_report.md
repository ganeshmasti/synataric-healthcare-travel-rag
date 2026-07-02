# Synataric Local Care Intent Router Evaluation

## Summary
- model name: llama_3_2_1b_lora_synataric_router
- total examples: 110
- accuracy: 1.000
- macro precision: 1.000
- macro recall: 1.000
- macro F1: 1.000
- invalid output rate: 0.000
- average route_execution_score: 1.000
- average latency seconds: 0.340

## Critical Safety/Workflow Recalls
- unsafe_medical recall: 1.000
- out_of_scope recall: 1.000
- needs_clarification recall: 1.000
- care_plan_multistep recall: 1.000

## Per-label Report

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| provider_search | 1.000 | 1.000 | 1.000 | 10 |
| cost_estimate | 1.000 | 1.000 | 1.000 | 10 |
| travel_planning | 1.000 | 1.000 | 1.000 | 10 |
| recovery_guidance | 1.000 | 1.000 | 1.000 | 10 |
| risk_checklist | 1.000 | 1.000 | 1.000 | 10 |
| find_evidence | 1.000 | 1.000 | 1.000 | 10 |
| care_plan_multistep | 1.000 | 1.000 | 1.000 | 10 |
| needs_clarification | 1.000 | 1.000 | 1.000 | 10 |
| unsafe_medical | 1.000 | 1.000 | 1.000 | 10 |
| out_of_scope | 1.000 | 1.000 | 1.000 | 10 |
| general_navigation | 1.000 | 1.000 | 1.000 | 10 |

## Confusion Matrix

- CSV: C:\Users\ganes\OneDrive\Desktop\GenAIProject\synataric-healthcare-travel-rag\reports\finetune\llama_finetuned_router\router_confusion_matrix.csv
- PNG: C:\Users\ganes\OneDrive\Desktop\GenAIProject\synataric-healthcare-travel-rag\reports\finetune\llama_finetuned_router\router_confusion_matrix.png

## Route Execution Score

The route_execution_score gives 1.0 for an exact route match, 0.5 for selected related workflow routes, and 0.0 otherwise. Unsafe medical, out-of-scope, and clarification misses receive no partial credit.

Average route_execution_score: 1.000

## Interpretation

Unsafe_medical and out_of_scope recall matter for safety. Needs_clarification recall matters because the agent should ask instead of guessing. Care_plan_multistep recall matters because those requests should route to the ReAct Care Planner. A wrong label is not just a classification error; it can trigger the wrong downstream workflow.
