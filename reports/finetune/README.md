# Synataric Local Care Intent Router

## Business Goal

Build a small local intent router that classifies one user message into one Synataric route label. The router reduces the need to call a frontier model for every routing decision while preserving the existing Synataric RAG, tool, refusal, clarification, and ReAct care-plan flows.

## Why Fine-Tune A Router

A fine-tuned local router can lower cost, reduce latency, and make route selection more predictable. It is safer than fine-tuning a general healthcare answer generator because the model only returns labels and never produces medical answers.

## Labels

`provider_search`, `cost_estimate`, `travel_planning`, `recovery_guidance`, `risk_checklist`, `find_evidence`, `care_plan_multistep`, `needs_clarification`, `unsafe_medical`, `out_of_scope`, `general_navigation`

## Dataset

The dataset is `data/finetune/synataric_care_router_tickets.csv`. It contains 550 synthetic examples, with 50 examples per label. Each row includes `id`, `ticket`, `label`, `scenario_type`, `difficulty`, and `notes`. Examples cover cataract surgery, knee replacement, cardiac bypass, Bangalore, India, provider search, costs, travel, recovery, risks, evidence lookup, vague requests, unsafe medical requests, prompt injection, non-healthcare questions, and Synataric policy navigation.

## Commands

```bash
python scripts/validate_synataric_router_dataset.py
python scripts/build_llamafactory_synataric_router_dataset.py
```

## LLaMA Factory Workflow

1. Clone LLaMA Factory.
2. Copy train JSON into `LLaMA-Factory/data`.
3. Add the dataset info entry from `data/finetune/llamafactory/dataset_info_snippet.json`.
4. Open LLaMA Board.
5. Select `Qwen/Qwen3-1.7B-Base`.
6. Select `synataric_care_router`.
7. LoRA fine-tune.
8. Review loss curve.
9. Merge adapter.
10. Run classify smoke tests.
11. Evaluate validation set.

## Smoke Tests

Run `classify(ticket)` on known examples and confirm the output is exactly one allowed label with no punctuation, markdown, or explanation. Include at least one sample from each label and verify prompt-injection attempts still route to `unsafe_medical` or `out_of_scope` as appropriate.

## Evaluation Plan

Evaluate validation accuracy overall and by label. Review a confusion matrix for common pairs such as `travel_planning` versus `needs_clarification`, `cost_estimate` versus `general_navigation`, and `recovery_guidance` versus `risk_checklist`. Inspect unsafe medical false negatives manually because they are higher risk than ordinary routing misses.

## Expected Startup Value

The router should provide lower cost, lower latency, predictable routing, and no medical answer generation. It gives Synataric a practical local component while leaving clinical content generation and safety handling in the existing downstream system.

## Non-Goals

This phase does not create direct medical answers, live provider ranking, PHI handling, appointment booking, model weights, or training runs.
