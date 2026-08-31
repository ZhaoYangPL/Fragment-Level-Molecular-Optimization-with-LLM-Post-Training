# Four-Step Molecular Optimization Dataset

This directory contains the dataset construction pipeline and the final artifact of the
four-step (Analyse → Plan → Edit → Verify) molecular optimization dataset.

## Artifacts

| File | Description |
|---|---|
| `fourstep_qed_logp_mw_sft_train_v2_intent_4000.jsonl` | Final training split — 4,000 examples (~24 MB) |
| `fourstep_qed_logp_mw_sft_val_v2_intent_200.jsonl` | Validation split — 200 examples |
| `fourstep_qed_logp_mw_sft_train_v2_intent_50.jsonl` | 50-example preview for quick inspection |
| `manifest_v2_intent_4000_200_offset0.json` | Provenance manifest (offsets, source indices, checkpoints) |

Total: **4,200 examples** built from the `Alan123/qa_dataset_with_qed_logp_mw_grpo`
corpus (~163k prompts) and decomposed into four labeled steps.

## Construction pipeline

| Script | Role |
|---|---|
| `dataset_constuction.py` | Base builder: loads the source corpus, filters/parses `<QED>/<LogP>/<MW>` target tags, computes BRICS fragment lists, and generates the four-step chain with a Tinker-served LLM. Resumable checkpoints every 500 examples. |
| `dataset_constuction_v2.py` | v2 rebuild: structured PLAN schema (`Reasoning / Proposed edit / Expected effect`) plus hindsight-bias filtering so the plan cannot leak the gold answer. |
| `rewrite_plan_dataset.py` | Regenerates only the `<PLAN>` block of an existing dataset while keeping ANALYSE/EDIT/VERIFY untouched. |
| `visualize_fourstep_dataset.py` | Dataset diagnostics: overview, edit gallery, property-delta alignment, structure checks. |

Regenerate with `./run_v2_phase1_4000.fish` from the repo root (requires a Tinker
platform account; see the root README).

## JSONL schema

Each line has the following fields:

| Field | Type | Description |
|---|---|---|
| `prompt` | str | Fragment-level task prompt: current SMILES + fragment list + requested property changes |
| `completion` | str | Four labeled blocks `<ANALYSE>`, `<PLAN>`, `<EDIT>`, `<VERIFY>` |
| `messages` | list | Chat-format version of the example (system prompt + user + assistant) |
| `input_smiles` / `target_smiles` | str | SMILES before / after the gold single-step edit |
| `action_type` | str | `Replace` \| `Add` \| `Remove` |
| `fragments` | list[str] | BRICS fragments of the input molecule |
| `targets` | dict | Per-property target: `delta`, `direction`, `target_value` for QED/LogP/MW |
| `input_props` / `output_props` | dict | Computed properties of input / output molecule |
| `actual_prop_deltas` | dict | Realized property change of the gold edit |
| `target_met` | bool | Whether the gold edit satisfies all tolerances (QED 0.02, LogP 0.20, MW 5.0) |
| `plan_schema` | str | `reasoning_proposed_intent_expected_v2` |
| `plan_direction_labels` | dict | Expected property-direction sign of the proposed edit |
| `source_index` | int | Index of the source prompt in the original corpus |
| `dataset_version` | str | `v2_structured_plan` |
