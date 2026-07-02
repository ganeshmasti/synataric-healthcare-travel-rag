# Synataric Week 5 Fine-Tuned Local Care Intent Router Report

## 1. Executive Summary

This Week 5 project adapts the SFT intent-classification lab to Synataric Global Healthcare Navigator. The model is not a medical chatbot. It is a local care-intent router: one user message in, exactly one route label out.

The fine-tuned `meta-llama/Llama-3.2-1B-Instruct` LoRA router achieved 1.000 accuracy and 1.000 macro F1 on the balanced 110-example evaluation split, with 0.000 invalid output rate and 0.340 second average latency. The existing Synataric router baseline achieved 0.555 accuracy, 0.493 macro F1, 0.000 invalid output rate, and 2.308 second average latency on the same split.

Existing Synataric RAG, tools, and the ReAct Care Planner produce the downstream answer. This local model only selects the route.

## 2. Why This Matters for Synataric

Synataric needs predictable routing before retrieval or tool execution. A wrong route can send a user to the wrong downstream workflow, such as answering generally when the system should ask a clarifying question, route to the ReAct Care Planner, or trigger a medical safety response.

The startup value is practical: lower routing cost, lower latency, predictable tool selection, and fewer frontier model calls. A small local router can handle routine route selection while frontier models and Synataric RAG/tools remain focused on higher-value reasoning and answer generation.

## 3. Router Architecture

The router classifies one user message into exactly one label. It does not provide medical advice, diagnose, prescribe, rank providers, or answer healthcare questions.

The route label is consumed by the existing Synataric application layer:

- RAG retrieves educational healthcare travel content.
- Tools handle provider search, cost estimate, travel planning, recovery guidance, risk checklist, and evidence lookup.
- Safety/refusal logic handles unsafe medical and out-of-scope requests.
- The ReAct Care Planner handles broad multi-step planning requests.

## 4. Dataset

The dataset is synthetic, balanced, and designed for routing only. It contains 550 total examples across 11 labels, with 50 examples per label. The train/eval split uses 40 train examples and 10 eval examples per label.

Labels:

- `provider_search`
- `cost_estimate`
- `travel_planning`
- `recovery_guidance`
- `risk_checklist`
- `find_evidence`
- `care_plan_multistep`
- `needs_clarification`
- `unsafe_medical`
- `out_of_scope`
- `general_navigation`

Because the dataset is synthetic and balanced, production readiness requires fresh real-world holdout testing with naturally distributed user traffic, adversarial prompts, and edge cases not present in this dataset.

## 5. Training Setup

- Base model: `meta-llama/Llama-3.2-1B-Instruct`
- Fine-tuning method: LoRA / SFT
- Train rows: 440
- Eval rows: 110
- Labels: 11
- Examples per label: 50 total, 40 train, 10 eval

Training loss:

| Epoch | Train Loss | Validation Loss |
| ---: | ---: | ---: |
| 1 | 0.235017 | 0.036771 |
| 2 | 0.014036 | 0.004715 |
| 3 | 0.001264 | 0.000690 |

## 6. Existing Router Baseline

The baseline uses the existing Synataric router on the same 110-example evaluation split.

- Total examples: 110
- Accuracy: 0.555
- Macro precision: 0.542
- Macro recall: 0.555
- Macro F1: 0.493
- Invalid output rate: 0.000
- Average route_execution_score: 0.609
- Average latency seconds: 2.308
- Unsafe medical recall: 0.600
- Out-of-scope recall: 1.000
- Needs clarification recall: 0.400
- Care plan multistep recall: 0.000
- General navigation recall: 0.000

The baseline's 0.000 recall for `care_plan_multistep` is expected because that is a new Week 5 route that the existing classifier was not designed to support.

## 7. Fine-Tuned Router Results

The fine-tuned Llama 3.2-1B LoRA router was evaluated on the same 110-example split.

- Total examples: 110
- Accuracy: 1.000
- Macro precision: 1.000
- Macro recall: 1.000
- Macro F1: 1.000
- Invalid output rate: 0.000
- Average route_execution_score: 1.000
- Average latency seconds: 0.340
- Smoke test: 9 / 9

The model returned valid route labels and did not generate healthcare answers.

## 8. Baseline vs Fine-Tuned Comparison Table

| Metric | Existing Synataric Router | Fine-Tuned Llama 3.2-1B LoRA Router |
| --- | ---: | ---: |
| Total examples | 110 | 110 |
| Accuracy | 0.555 | 1.000 |
| Macro precision | 0.542 | 1.000 |
| Macro recall | 0.555 | 1.000 |
| Macro F1 | 0.493 | 1.000 |
| Invalid output rate | 0.000 | 0.000 |
| Average route_execution_score | 0.609 | 1.000 |
| Average latency seconds | 2.308 | 0.340 |
| Unsafe medical recall | 0.600 | 1.000 |
| Out-of-scope recall | 1.000 | 1.000 |
| Needs clarification recall | 0.400 | 1.000 |
| Care plan multistep recall | 0.000 | 1.000 |
| Smoke test | Not run | 9 / 9 |

## 9. Critical Safety and Workflow Labels

`unsafe_medical` recall matters because medication, diagnosis, prescription, dosage, and urgent clinical judgment requests should not be routed into normal healthcare navigation.

`out_of_scope` recall matters because non-healthcare or unsupported requests should not trigger healthcare workflows.

`needs_clarification` recall matters because the agent should ask for missing information instead of guessing.

`care_plan_multistep` recall matters because those requests should route to the ReAct Care Planner rather than a single narrow tool.

## 10. Latency and Cost Interpretation

The fine-tuned local router reduced average latency from 2.308 seconds to 0.340 seconds on the reported evaluation setup. It also shifts routing away from repeated frontier model calls, which can reduce cost and make tool selection more predictable.

This does not remove the need for frontier models or RAG. It narrows their use: the local router decides where the message should go, and downstream Synataric systems perform retrieval, planning, refusal, or answer generation.

## 11. Smoke Test Results

The fine-tuned router passed 9 out of 9 smoke tests. The smoke tests confirmed label-only behavior for representative cases and checked that the model acts as a router rather than a chatbot.

## 12. What This Proves

This proves that a small local LoRA/SFT model can learn the Synataric routing schema on a balanced synthetic dataset and produce exact route labels with low latency. It also shows that the new `care_plan_multistep` route can be learned as a separate workflow trigger.

The result supports a startup-friendly architecture: local intent routing for speed and cost control, with existing Synataric RAG/tools/ReAct planner handling downstream behavior.

## 13. What This Does Not Prove

This does not prove production readiness by itself. The dataset is synthetic and balanced, so the evaluation may be easier than real traffic. The result does not prove clinical correctness, provider quality, appointment readiness, PHI handling, or robustness against all adversarial prompts.

Before production use, Synataric needs fresh real-world holdout testing, monitoring for invalid or drifting labels, safety-focused red teaming, and evaluation on naturally imbalanced user traffic.

## 14. Production Next Steps

1. Build a real-world holdout set from de-identified user-like traffic.
2. Evaluate base model, existing router, and fine-tuned router on the same holdout set.
3. Add adversarial and safety-specific tests for `unsafe_medical`, `out_of_scope`, and prompt injection.
4. Calibrate fallback behavior when the router emits an invalid label or low-confidence output.
5. Add runtime monitoring for label distribution drift, latency, invalid outputs, and safety recalls.
6. Decide whether `care_plan_multistep` should be added to the production classifier schema before integration.
7. Keep downstream answer generation inside Synataric RAG/tools/ReAct planner, not inside the router.

## 15. Demo Script

1. "This is not a medical chatbot. It is a local care-intent router."
2. "The router takes one user message and returns exactly one Synataric route label."
3. "The existing Synataric RAG, tools, and ReAct planner produce the downstream answer."
4. "On the same 110-example validation set, the existing router reached 0.555 accuracy and 0.493 macro F1."
5. "The fine-tuned Llama 3.2-1B LoRA router reached 1.000 accuracy and 1.000 macro F1, with no invalid outputs."
6. "Latency improved from 2.308 seconds to 0.340 seconds in this evaluation setup."
7. "The biggest workflow gain is learning `care_plan_multistep`, which routes broad planning requests to the ReAct Care Planner."
8. "The production next step is real-world holdout testing because this dataset is synthetic and balanced."
