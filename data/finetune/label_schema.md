# Synataric Local Care Intent Router Label Schema

## Routing Principle

The Synataric Care Intent Router accepts one user message and returns exactly one route label. It does not answer healthcare questions, provide diagnoses, prescribe medication, rank providers, or generate care plans. Downstream Synataric RAG, tools, clarification flows, refusals, and ReAct planning handle the response after routing.

## Labels

| Label | Definition | Downstream Synataric Route | Examples |
| --- | --- | --- | --- |
| `provider_search` | Find hospitals, clinics, providers, doctors, or specialist centers. | Provider and hospital retrieval | "Where can I find good cataract surgery in India?" |
| `cost_estimate` | Procedure cost, stay cost, travel cost, pricing, budget, estimate, or cost disclaimer for a specific care topic. | Cost estimate retrieval | "What is the cost of cataract surgery in Bangalore?" |
| `travel_planning` | Travel logistics, stay duration, caregiver support, airport transfer, mobility, accommodation, or documents to carry. | Travel planning support | "Plan my travel for cataract surgery in Bangalore." |
| `recovery_guidance` | Recovery timeline, follow-up, post-op planning, wound care, rehab, or recovery support. | Recovery guidance retrieval | "What recovery guidance is available after cataract surgery?" |
| `risk_checklist` | Urgent symptoms, red flags, emergency guidance, risk checklist, or safety checklist. | Risk checklist and emergency safety flow | "What urgent symptoms require immediate care?" |
| `find_evidence` | User asks where something is explained, which source contains it, or asks to audit evidence. | Evidence locator | "Where is cataract recovery planning explained?" |
| `care_plan_multistep` | Broad multi-step care plan explicitly asking for multiple categories such as providers plus cost plus recovery plus risks plus travel. | ReAct care plan orchestration | "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks." |
| `needs_clarification` | Healthcare-related request missing required information such as procedure, destination, or care topic. | Clarifying question flow | "I need surgery in India." |
| `unsafe_medical` | Diagnosis, prescription, dosage, medication instructions, urgent clinical decision, or individualized treatment advice. | Medical safety refusal or urgent-care guidance | "Should I take antibiotics after surgery?" |
| `out_of_scope` | Non-healthcare, sports, politics, weather, stock market, coding, entertainment, or impossible unsupported destinations. | Out-of-scope refusal | "Who won the Super Bowl in 2024?" |
| `general_navigation` | General Synataric policy, educational disclaimer, whether estimates are final, how the system works, or what it can and cannot do. | General navigation or policy response | "Are Synataric cost estimates final prices?" |

## Safety Note

This router is a classifier only. It must return a label, not medical advice. Medication, prescription, diagnosis, dosage, and individualized treatment requests are routed to `unsafe_medical` even when the message also mentions travel, cost, or recovery.

## Ambiguity Rules

- If the user asks for multiple categories, use `care_plan_multistep`.
- If the user asks for medication, prescription, diagnosis, or dosage, use `unsafe_medical` even if the question mentions travel.
- If a healthcare request lacks a procedure and needs one to proceed, use `needs_clarification`.
- If clearly non-healthcare, use `out_of_scope`.
- If the question is about Synataric policies or estimate disclaimers, use `general_navigation`.

## Confusing Pairs

`provider_search` vs `care_plan_multistep`: choose `provider_search` when the user only asks for hospitals, clinics, doctors, or centers. Choose `care_plan_multistep` when the user asks for a broad plan spanning multiple routing categories.

`cost_estimate` vs `general_navigation`: choose `cost_estimate` for a cost estimate tied to a procedure, stay, travel, or budget. Choose `general_navigation` for questions about whether Synataric estimates are final, guaranteed, or official.

`travel_planning` vs `needs_clarification`: choose `travel_planning` when the message includes enough care context to plan logistics. Choose `needs_clarification` when the user asks for travel or surgery help but omits the procedure, destination, or care topic needed to proceed.

`recovery_guidance` vs `risk_checklist`: choose `recovery_guidance` for normal post-op timelines, follow-up, rehab, and support. Choose `risk_checklist` for urgent symptoms, red flags, emergency signs, or safety checklists.

`unsafe_medical` vs `out_of_scope`: choose `unsafe_medical` for healthcare requests that ask for diagnosis, treatment decisions, medication, prescription, dosage, or urgent clinical judgment. Choose `out_of_scope` for non-healthcare topics or impossible unsupported care scenarios.
