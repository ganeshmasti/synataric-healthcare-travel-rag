# Synataric Week 5 Fine-Tune Router Plan

## 1. Objective

Map the Week 5 support-ticket router project to Synataric by fine-tuning a local classifier that returns exactly one care-route label for one user message.

## 2. Dataset

Use `data/finetune/synataric_care_router_tickets.csv`, a synthetic 550-row CSV with 50 rows per label. The dataset is for routing only and contains no PHI or model-generated medical answers.

## 3. Labels

The labels are `provider_search`, `cost_estimate`, `travel_planning`, `recovery_guidance`, `risk_checklist`, `find_evidence`, `care_plan_multistep`, `needs_clarification`, `unsafe_medical`, `out_of_scope`, and `general_navigation`.

## 4. LLaMA Factory Setup

Clone LLaMA Factory in Colab or a local GPU environment, install dependencies, and confirm LLaMA Board starts before copying Synataric data into the `data` folder.

## 5. Train/Validation Split

Run `python scripts/build_llamafactory_synataric_router_dataset.py` to create a deterministic stratified split with 440 training examples and 110 validation examples.

## 6. ShareGPT JSON Conversion

The conversion script emits ShareGPT examples where the human message contains the router instruction and ticket, and the assistant message is exactly the route label.

## 7. LLaMA Board Settings

Select `Qwen/Qwen3-1.7B-Base`, choose the `synataric_care_router` dataset, use LoRA fine-tuning, and keep generation output constrained during smoke tests.

## 8. Training Knobs To Understand

Track learning rate, epochs, LoRA rank, batch size, gradient accumulation, validation loss, and whether the model overfits simple label patterns.

## 9. Merge Adapter

After reviewing the loss curve and validation behavior, merge the LoRA adapter only if the router gives stable label-only outputs.

## 10. classify(ticket) Function

Create a thin inference wrapper that formats the same instruction used in training and returns the first valid label. The wrapper should reject explanations, markdown, punctuation, and labels outside the schema.

## 11. Smoke Tests

Test representative tickets from all 11 labels, including the required examples for cataract surgery, Bangalore travel, unsafe antibiotics requests, sports, Mars, and Synataric estimate disclaimers.

## 12. Validation Evaluation

Run the validation JSON through the merged or adapter-backed model and compute accuracy by label, macro accuracy, and exact-match label-only compliance.

## 13. Base Vs Fine-Tuned Comparison

Compare the base model and fine-tuned model on the same validation set. The fine-tuned model should improve exact-label accuracy and reduce verbose responses.

## 14. Confusion Matrix

Create a confusion matrix to inspect label-pair failures. Focus on `travel_planning` vs `needs_clarification`, `recovery_guidance` vs `risk_checklist`, `cost_estimate` vs `general_navigation`, and `provider_search` vs `care_plan_multistep`.

## 15. Startup Interpretation

Interpret the router as an operational cost and latency component, not as a medical authority. The startup value is predictable routing into existing Synataric systems.

## 16. Submission Checklist

Validate the CSV, build the LLaMA Factory JSON files, attach the label schema, include smoke-test examples, include evaluation notes, and confirm no model weights or runtime app files were modified.
