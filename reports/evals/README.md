# Synataric Agent Evaluation

I will measure intent accuracy, tool selection accuracy, task completion, safety compliance, human handoff accuracy, source hit rate, trajectory correctness, latency, and cost on Synataric Navigator's Router Agent and ReAct Care Planner using a golden dataset of 40 healthcare navigation cases covering happy paths, edge cases, known failures, and adversarial requests. I will use code-based evaluators, LLM-as-judge rubrics, RAG/source checks, and LangSmith traces. Pass bar: 85%+ task completion, 90%+ safety/out-of-scope correctness, 85%+ tool selection accuracy, 100% max-step compliance, and no unsafe medical advice.

## Dataset

The Phase 1 golden dataset lives at `data/evals/synataric_agent_golden_dataset.csv`.

It contains 40 labeled cases for:
- Router Agent
- ReAct Care Planner

Scenario mix:
- 20 happy_path
- 12 edge_case
- 6 known_failure
- 2 adversarial

Coverage includes provider search, cost estimates, recovery guidance, risk checklists, travel planning, evidence lookup, unsafe medical advice, out-of-scope requests, vague or missing information, and ReAct multi-step care planning.

## Validate

```bash
python -m src.agent_eval_dataset
```

## Upload To LangSmith

```bash
python scripts/upload_synataric_eval_dataset.py
```

Optional rebuild:

```bash
python scripts/upload_synataric_eval_dataset.py --recreate
```

LangSmith project: `Synataric-Agent-Evals`

Current phase: Phase 1: golden dataset and upload scaffolding.
