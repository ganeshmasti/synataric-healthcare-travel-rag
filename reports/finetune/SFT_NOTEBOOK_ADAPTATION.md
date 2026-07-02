# Synataric SFT Notebook Adaptation

The official Week 5 repo file is `SFT/Finetuning_Workshop_SFT_Demo.ipynb` in `The-Gen-Academy/Mastering-Agentic-AI-Week5`.

The original notebook fine-tunes a doctor appointment intent router. Synataric adapts the same notebook to fine-tune a healthcare navigation route classifier.

Replace `finetuning_preference_dataset.csv` with `synataric_finetuning_preference_dataset.csv`.

Keep the task narrow: one message in, exactly one route label out. The model must not answer healthcare questions.

## Output Labels

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
