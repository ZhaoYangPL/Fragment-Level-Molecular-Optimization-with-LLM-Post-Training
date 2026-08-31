#!/usr/bin/env python3
"""Generalization test: 100 target tuples × 10 trials, best-of-10 NTE.

Evaluates a trained GRPO model on uniformly-sampled (QED, LogP, MW) targets
that are unseen during training, following the protocol described in the README:

    "To test generalization, we uniformly sample 100 target tuples (QED, LogP, MW)
    across admissible ranges. For each tuple and each baseline we run 10 independent
    trials under the same compute budget and report the best-of-10."

Usage:
    python generalization_test.py \
        --checkpoint tinker://.../weights/... \
        --run-tag gen_test_baseline \
        --num-targets 100 \
        --trials-per-target 10
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

import dataset as dataset_mod
import model_utils
from metrics_io import slugify
from metrics_io import save_training_metrics

# ── project root ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RL_CORE_PATH = PROJECT_ROOT / "TURN2Bonus0.25KL_delta_abs_rl_core.py"

# ── admissible ranges for target properties ──────────────────────
QED_RANGE = (0.1, 0.9)
LOGP_RANGE = (-3.0, 6.0)
MW_RANGE = (100.0, 500.0)

# ── NTE buckets (mirrors rl_core._INITIAL_NTE_BUCKETS) ──────────
NTE_BUCKETS = (
    ("low",  None, 0.5),
    ("mid",  0.5, 2.0),
    ("high", 2.0, None),
)


# ═══════════════════════════════════════════════════════════════════
# RL core module loader
# ═══════════════════════════════════════════════════════════════════

def _load_rl_core():
    """Dynamically import the RL core module (mirrors train_grpo_delta_abs.py)."""
    if not RL_CORE_PATH.exists():
        raise FileNotFoundError(f"RL core file not found: {RL_CORE_PATH}")
    module_name = re.sub(r"[^A-Za-z0-9_]+", "_", RL_CORE_PATH.stem)
    spec = importlib.util.spec_from_file_location(module_name, RL_CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load RL core from {RL_CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════════════
# target sampling
# ═══════════════════════════════════════════════════════════════════

def sample_target_tuples(rng: np.random.Generator, num_targets: int = 100) -> list[dict[str, float]]:
    """Uniformly sample (QED, LogP, MW) tuples across admissible ranges."""
    targets = []
    for _ in range(num_targets):
        targets.append({
            "qed":  float(rng.uniform(*QED_RANGE)),
            "logp": float(rng.uniform(*LOGP_RANGE)),
            "mw":   float(rng.uniform(*MW_RANGE)),
        })
    return targets


# ═══════════════════════════════════════════════════════════════════
# starting molecule selection
# ═══════════════════════════════════════════════════════════════════

def _nte_bucket_label(nte: float | None) -> str:
    if nte is None:
        return "unknown"
    if nte <= 0.5:
        return "low"
    if nte <= 2.0:
        return "mid"
    return "high"


def select_starting_molecule(
    raw_data,
    rng: np.random.Generator,
    target: dict[str, float],
    *,
    core,
    min_nte: float = 0.05,
    max_attempts: int = 200,
) -> tuple[Chem.Mol, dict[str, float], str, str]:
    """Sample a molecule from *raw_data* whose initial NTE to *target* is ≥ *min_nte*.

    No upper bound is applied — following the paper's design: GRPO is
    ground-truth–free and reward-based, so performance is not tied to a
    particular training distribution.  The model is tested on arbitrary
    (start, target) pairs just as in the original protocol.

    Returns
    -------
    (mol, current_props, smiles, prompt) — ready to feed into the multi-turn loop.
    """
    n = len(raw_data)
    for _ in range(max_attempts):
        idx = int(rng.integers(0, n))
        example = raw_data[idx]
        # extract SMILES from the dataset prompt (the "input" column contains <SMILES>)
        dataset_prompt = str(example.get(core.PROMPT_COL, example[core.PROMPT_COL]))
        smiles = core.extract_smiles(dataset_prompt)
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        current = core._property_values(mol)
        # compute initial NTE between current molecule and target
        initial_nte = core.normalized_total_error(
            abs(current["qed"] - target["qed"]),
            abs(current["logp"] - target["logp"]),
            abs(current["mw"] - target["mw"]),
        )
        if math.isfinite(initial_nte) and initial_nte >= min_nte:
            prompt = build_test_prompt(mol, smiles, target, core=core)
            return mol, current, smiles, prompt

    raise RuntimeError(
        f"Could not find suitable starting molecule after {max_attempts} attempts. "
        f"Target: {target}"
    )


# ═══════════════════════════════════════════════════════════════════
# prompt construction
# ═══════════════════════════════════════════════════════════════════

def build_test_prompt(
    start_mol: Chem.Mol,
    start_smiles: str,
    target: dict[str, float],
    *,
    core,
) -> str:
    """Build a prompt in the EXACT format used during training.

    Format (from training data):
        Given the intermediate molecule SMILES <SMILES>...</SMILES>,
        which is composed of fragments [...]. Propose a single replace,
        add or remove step on fragment level that makes the new molecule's
        QED <QED>0.094</QED> higher, LogP <LogP>2.898</LogP> lower, and
        Molecular Weight <MW>101.104</MW> lower.
    """
    current = core._property_values(start_mol)

    # get fragments (same logic as _replace_prompt_smiles)
    try:
        fragment_list = core.get_clean_fragments(start_smiles)
    except Exception:
        fragment_list = []
    if not fragment_list:
        fragment_list = [start_smiles]

    # build property tags: <PROP>delta</PROP> direction
    specs = [
        ("QED", "qed"),
        ("LogP", "logp"),
        ("MW", "mw"),
    ]
    full_names = {
        "QED": "QED",
        "LogP": "LogP",
        "MW": "Molecular Weight",
    }
    tags = []
    for prop_tag, key in specs:
        delta = target[key] - current[key]
        direction = "higher" if delta >= 0 else "lower"
        tags.append(f"{full_names[prop_tag]} <{prop_tag}>{abs(delta):.3f}</{prop_tag}> {direction}")

    prompt = (
        f"Given the intermediate molecule SMILES <SMILES>{start_smiles}</SMILES>, "
        f"which is composed of fragments {fragment_list}. "
        f"Propose a single replace, add or remove step on fragment level that makes "
        f"the new molecule's {tags[0]}, {tags[1]}, and {tags[2]}."
    )
    return prompt


# ═══════════════════════════════════════════════════════════════════
# single-trial inference loop  (mirrors the inference path of run_mode_b)
# ═══════════════════════════════════════════════════════════════════

def run_single_trial(
    initial_prompt: str,
    *,
    sampling_client,
    tokenizer,
    renderer,
    core,
    max_turns: int = 3,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 256,
    debug: bool = False,
    debug_prefix: str = "",
) -> dict[str, Any]:
    """Run one multi-turn trajectory (up to *max_turns* turns).

    Mirrors the inference path of ``run_mode_b``:
      - same message construction (system → guidance → prompt → [feedback])
      - same sampling parameters
      - same accept/reject via NTE improvement
      - same rollback feedback on rejection
      - same ``_replace_prompt_smiles`` on acceptance

    Returns a dict with terminal NTE details + turn-by-turn records.
    """
    import tinker
    from tinker import types as tinker_types

    initial_smiles = core.extract_smiles(initial_prompt)
    initial_mol = Chem.MolFromSmiles(initial_smiles) if initial_smiles else None
    initial_nte = core._initial_nte_from_prompt(initial_prompt)
    prompt_guidance = core._prompt_guidance_for_initial_nte(initial_nte)

    state = {
        "current_prompt": initial_prompt,
        "current_valid_smiles": initial_smiles,
        "current_valid_canonical": Chem.MolToSmiles(initial_mol) if initial_mol else None,
        "has_valid_generated": False,
        "last_valid_generated_smiles": None,
        "last_valid_generated_canonical": None,
        "prev_env_feedback": "",
        "prev_guidance": "",
        "turns": [],
    }

    for turn_idx in range(max_turns):
        # ── build messages ────────────────────────────────────
        messages = [
            {"role": "system", "content": core.SYSTEM_PROMPT_RL},
            {"role": "user", "content": prompt_guidance},
            {"role": "user", "content": state["current_prompt"]},
        ]
        if turn_idx > 0 and (state["prev_env_feedback"] or state["prev_guidance"]):
            feedback_parts = [
                "Environment feedback from your previous attempt:",
                state["prev_env_feedback"],
                "",
                "Target-gap priorities from previous turn (must address first):",
                state["prev_guidance"],
                "",
                "Interpretation rules (must follow):",
                "- If a property says 'need ... more increase/decrease', move in that direction.",
                "- If a property says 'overshot by ...', reverse slightly toward target.",
                "- Keep already-met properties near target while fixing unmet ones.",
                "",
                "Use the current molecule shown in the prompt above as the only edit base.",
                "Do not restart from the original molecule.",
            ]
            messages.append({"role": "user", "content": "\n".join(feedback_parts).strip()})

        # ── sample ────────────────────────────────────────────
        prompt_input = renderer.build_generation_prompt(messages)
        result = sampling_client.sample(
            prompt=prompt_input,
            sampling_params=tinker_types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=renderer.get_stop_sequences(),
            ),
            num_samples=1,
        ).result()

        seq = result.sequences[0]
        raw_completion = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        completion_text = core._normalize_completion_text(raw_completion)

        # ── environment feedback ───────────────────────────────
        env_feedback, mol_out, all_met = core._run_env(
            completion_text, state["current_prompt"]
        )
        pred_smiles = core.extract_smiles(completion_text)

        # ── NTE comparison ─────────────────────────────────────
        turn_mol_in = Chem.MolFromSmiles(state["current_valid_smiles"]) if state["current_valid_smiles"] else None
        turn_valid = bool(pred_smiles and turn_mol_in is not None and mol_out is not None)

        new_nte = None
        old_nte = None
        improved = False

        if turn_valid:
            parsed = core._parse_property_targets(state["current_prompt"], turn_mol_in)
            if parsed is not None:
                target_props, _input_props = parsed
                new_nte, _np, _op = core.property_error(mol_out, target_props)
                old_nte, _op2, _ip = core.property_error(turn_mol_in, target_props)
                improved = core.is_nte_improvement(True, old_nte, new_nte)

        # ── accept or reject ──────────────────────────────────
        if improved:
            state["current_prompt"] = core._replace_prompt_smiles(
                state["current_prompt"], pred_smiles
            )
            state["current_valid_smiles"] = pred_smiles
            state["current_valid_canonical"] = Chem.MolToSmiles(mol_out)
            state["has_valid_generated"] = True
            state["last_valid_generated_smiles"] = pred_smiles
            state["last_valid_generated_canonical"] = state["current_valid_canonical"]

            # update guidance for next turn
            unmet_props, progress_lines = core._target_progress_lines(
                turn_mol_in, mol_out, state["current_prompt"]
            )
            unmet_text = ", ".join(unmet_props) if unmet_props else "none"
            progress_block = (
                "\n".join([f"- {line}" for line in progress_lines])
                if progress_lines
                else "- target-gap details unavailable."
            )
            state["prev_guidance"] = (
                f"all_met={all_met}; unmet_properties={unmet_text}.\n{progress_block}"
            )
            state["prev_env_feedback"] = core._compact_env_feedback(env_feedback)
        else:
            # rollback
            if not turn_valid:
                rejection_reason = "invalid_molecule"
            elif old_nte is None or new_nte is None:
                rejection_reason = "nte_unavailable"
            else:
                rejection_reason = "no_improvement"

            rollback_feedback, rollback_guidance, _ = core._rollback_state_feedback(
                state["current_prompt"],
                rejection_reason=str(rejection_reason),
                old_nte=old_nte,
                new_nte=new_nte,
            )
            state["prev_guidance"] = rollback_guidance
            state["prev_env_feedback"] = core._compact_env_feedback(rollback_feedback)

        state["turns"].append({
            "turn": int(turn_idx) + 1,
            "valid": bool(turn_valid),
            "improved": bool(improved),
            "accepted": bool(improved),
            "all_met": bool(all_met) if turn_valid else False,
            "new_nte": float(new_nte) if new_nte is not None else None,
            "old_nte": float(old_nte) if old_nte is not None else None,
            "output_text": completion_text,
            "pred_smiles": pred_smiles,
        })

        if debug:
            print(
                f"[DBG]{debug_prefix} turn={turn_idx} valid={turn_valid} "
                f"improved={improved} old_nte={old_nte} new_nte={new_nte} "
                f"all_met={all_met}"
            )

    # ── terminal NTE ───────────────────────────────────────────
    final_smiles = state["current_valid_smiles"] or initial_smiles
    final_details = core.compute_final_nte_details(final_smiles, initial_prompt)
    final_details["turns"] = state["turns"]
    final_details["initial_nte"] = initial_nte
    final_details["initial_nte_bucket"] = _nte_bucket_label(initial_nte)
    final_details["final_smiles"] = final_smiles
    final_details["initial_smiles"] = initial_smiles
    final_details["num_accepted_turns"] = sum(
        1 for t in state["turns"] if t["accepted"]
    )
    final_details["num_valid_turns"] = sum(
        1 for t in state["turns"] if t["valid"]
    )

    return final_details


# ═══════════════════════════════════════════════════════════════════
# best-of-N runner
# ═══════════════════════════════════════════════════════════════════

def run_best_of_n(
    initial_prompt: str,
    *,
    n: int = 10,
    **trial_kwargs,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run *n* independent trials, return (best_result, all_results)."""
    all_results = []
    for trial_idx in range(n):
        result = run_single_trial(initial_prompt, **trial_kwargs)
        result["trial_index"] = int(trial_idx)
        all_results.append(result)

    # best = lowest NTE among trials that produced a valid final molecule
    valid_results = [
        r for r in all_results
        if r.get("valid") and isinstance(r.get("nte"), (int, float))
    ]
    if valid_results:
        best = min(valid_results, key=lambda r: float(r["nte"]))
    else:
        # fallback: all trials invalid → pick the first one
        best = all_results[0]

    return best, all_results


# ═══════════════════════════════════════════════════════════════════
# aggregation
# ═══════════════════════════════════════════════════════════════════

def compute_aggregate_metrics(
    all_target_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate statistics across all target tuples."""
    best_records = [
        r["best"] for r in all_target_results
        if r["best"].get("valid") and isinstance(r["best"].get("nte"), (int, float))
    ]
    best_ntes = [float(b["nte"]) for b in best_records]

    # ── per-property errors from best-of-10 records ──────────
    qed_errs = [float(b["qed_error"]) for b in best_records if b.get("qed_error") is not None]
    logp_errs = [float(b["logp_error"]) for b in best_records if b.get("logp_error") is not None]
    mw_errs = [float(b["mw_error"]) for b in best_records if b.get("mw_error") is not None]

    # ── NTE bucket breakdown ────────────────────────────────
    bucket_stats: dict[str, Any] = {}
    for label, lo, hi in NTE_BUCKETS:
        bucket_results = [
            b for r in all_target_results
            if (b := r["best"]).get("initial_nte_bucket") == label
            and b.get("valid")
            and isinstance(b.get("nte"), (int, float))
        ]
        bucket_ntes = [float(b["nte"]) for b in bucket_results]
        bucket_stats[label] = {
            "count": len(bucket_results),
            "mean_best_nte": float(np.mean(bucket_ntes)) if bucket_ntes else None,
            "std_best_nte": float(np.std(bucket_ntes)) if bucket_ntes else None,
        }

    # ── acceptance / validity rates ──────────────────────────
    all_trials = []
    for r in all_target_results:
        all_trials.extend(r.get("all_trials", []))
    total_turns = sum(len(t.get("turns", [])) for t in all_trials)
    total_accepted = sum(1 for t in all_trials for turn in t.get("turns", []) if turn.get("accepted"))
    total_valid = sum(1 for t in all_trials for turn in t.get("turns", []) if turn.get("valid"))
    total_valid_trials = sum(1 for t in all_trials if t.get("valid"))

    return {
        "num_targets_evaluated": len(best_records),
        "num_targets_total": len(all_target_results),
        "num_trials_total": len(all_trials),
        "mean_best_nte": float(np.mean(best_ntes)) if best_ntes else None,
        "std_best_nte": float(np.std(best_ntes)) if best_ntes else None,
        "median_best_nte": float(np.median(best_ntes)) if best_ntes else None,
        "min_best_nte": float(np.min(best_ntes)) if best_ntes else None,
        "max_best_nte": float(np.max(best_ntes)) if best_ntes else None,
        "mean_best_qed_error": float(np.mean(qed_errs)) if qed_errs else None,
        "mean_best_logp_error": float(np.mean(logp_errs)) if logp_errs else None,
        "mean_best_mw_error": float(np.mean(mw_errs)) if mw_errs else None,
        "turn_acceptance_rate": float(total_accepted / total_turns) if total_turns > 0 else None,
        "turn_validity_rate": float(total_valid / total_turns) if total_turns > 0 else None,
        "trial_validity_rate": float(total_valid_trials / len(all_trials)) if all_trials else None,
        "nte_buckets": bucket_stats,
    }


# ═══════════════════════════════════════════════════════════════════
# output
# ═══════════════════════════════════════════════════════════════════

def _to_jsonable(value: Any) -> Any:
    """Convert numpy types to built-in Python types."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_generalization_results(
    all_target_results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    output_dir: str | Path,
    run_tag: str,
    config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write trial results, best-per-target summaries, and aggregate stats."""
    output_dir = Path(output_dir) / slugify(run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── config.json ──────────────────────────────────────────
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(_to_jsonable(config or {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── trial_results.jsonl ──────────────────────────────────
    trial_path = output_dir / "trial_results.jsonl"
    with trial_path.open("w", encoding="utf-8") as fh:
        for target_result in all_target_results:
            target_index = target_result["target_index"]
            target_props = target_result["target"]
            starting_smiles = target_result.get("starting_smiles")
            starting_props = target_result.get("starting_props", {})

            for trial in target_result.get("all_trials", []):
                row = {
                    "target_index": int(target_index),
                    "trial_index": int(trial.get("trial_index", 0)),
                    "qed_target": float(target_props["qed"]),
                    "logp_target": float(target_props["logp"]),
                    "mw_target": float(target_props["mw"]),
                    "starting_smiles": starting_smiles,
                    "starting_qed": float(starting_props.get("qed")) if starting_props else None,
                    "starting_logp": float(starting_props.get("logp")) if starting_props else None,
                    "starting_mw": float(starting_props.get("mw")) if starting_props else None,
                    "initial_nte": trial.get("initial_nte"),
                    "initial_nte_bucket": trial.get("initial_nte_bucket"),
                    "valid": trial.get("valid"),
                    "nte": trial.get("nte"),
                    "qed_error": trial.get("qed_error"),
                    "logp_error": trial.get("logp_error"),
                    "mw_error": trial.get("mw_error"),
                    "qed_out": trial.get("qed_out"),
                    "logp_out": trial.get("logp_out"),
                    "mw_out": trial.get("mw_out"),
                    "final_smiles": trial.get("final_smiles") if trial.get("valid") else None,
                    "num_accepted_turns": trial.get("num_accepted_turns"),
                    "num_valid_turns": trial.get("num_valid_turns"),
                    "turns": trial.get("turns", []),
                }
                fh.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")

    # ── best_per_target.jsonl ────────────────────────────────
    best_path = output_dir / "best_per_target.jsonl"
    with best_path.open("w", encoding="utf-8") as fh:
        for target_result in all_target_results:
            target_index = target_result["target_index"]
            target_props = target_result["target"]
            best = target_result["best"]
            all_trials = target_result.get("all_trials", [])

            valid_ntes = [
                float(t["nte"]) for t in all_trials
                if t.get("valid") and isinstance(t.get("nte"), (int, float))
            ]
            row = {
                "target_index": int(target_index),
                "qed_target": float(target_props["qed"]),
                "logp_target": float(target_props["logp"]),
                "mw_target": float(target_props["mw"]),
                "starting_smiles": target_result.get("starting_smiles"),
                "initial_nte": best.get("initial_nte"),
                "initial_nte_bucket": best.get("initial_nte_bucket"),
                "best_nte": best.get("nte"),
                "mean_nte": float(np.mean(valid_ntes)) if valid_ntes else None,
                "std_nte": float(np.std(valid_ntes)) if valid_ntes else None,
                "num_valid_trials": len(valid_ntes),
                "num_trials": len(all_trials),
                "best_qed_error": best.get("qed_error"),
                "best_logp_error": best.get("logp_error"),
                "best_mw_error": best.get("mw_error"),
                "best_qed_out": best.get("qed_out"),
                "best_logp_out": best.get("logp_out"),
                "best_mw_out": best.get("mw_out"),
                "best_final_smiles": best.get("final_smiles") if best.get("valid") else None,
                "best_trial_index": best.get("trial_index"),
            }
            fh.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")

    # ── summary.json ─────────────────────────────────────────
    summary_path = output_dir / "summary.json"
    summary_payload = {
        "run_tag": run_tag,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "config": _to_jsonable(config or {}),
        "aggregate": _to_jsonable(aggregate),
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "config": str(config_path),
        "trial_results": str(trial_path),
        "best_per_target": str(best_path),
        "summary": str(summary_path),
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generalization test: 100 target tuples × 10 trials, best-of-10 NTE.",
    )
    # ── required ──────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint", required=True,
        help="Tinker checkpoint path (training weights) to evaluate.",
    )
    parser.add_argument(
        "--run-tag", required=True,
        help="Name for this evaluation run (used in output paths).",
    )

    # ── model / tokenizer ─────────────────────────────────────
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--local-tokenizer-dir",
        help="Local tokenizer directory. Defaults to tokenizers/<base-model>.",
    )
    parser.add_argument(
        "--prefer-local-tokenizer", action="store_true", default=True,
        help="Prefer local tokenizer files (default).",
    )
    parser.add_argument(
        "--no-local-tokenizer", dest="prefer_local_tokenizer", action="store_false",
    )
    parser.add_argument(
        "--no-cache-tokenizer", dest="cache_tokenizer", action="store_false",
        help="Do not cache downloaded tokenizer locally.",
    )
    parser.set_defaults(cache_tokenizer=True)

    # ── evaluation ────────────────────────────────────────────
    parser.add_argument("--num-targets", type=int, default=100,
                        help="Number of target tuples to sample (default: 100).")
    parser.add_argument("--trials-per-target", type=int, default=10,
                        help="Number of independent trials per target (default: 10).")
    parser.add_argument("--max-turns", type=int, default=3,
                        help="Maximum edit turns per trial (default: 3).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rng-seed", type=int, default=42,
                        help="Seed for target tuple sampling and starting molecule selection.")
    parser.add_argument("--lora-rank", type=int, default=32)

    # ── data ──────────────────────────────────────────────────
    parser.add_argument("--dataset-name", default=dataset_mod.DEFAULT_DATASET_NAME)
    parser.add_argument("--data-split", default=dataset_mod.DEFAULT_DATA_SPLIT)
    parser.add_argument("--local-data-dir", default=str(dataset_mod.DEFAULT_LOCAL_DATA_DIR))
    parser.add_argument("--prompt-col", default=dataset_mod.DEFAULT_PROMPT_COL)
    parser.add_argument("--answer-col", default=dataset_mod.DEFAULT_ANSWER_COL)
    parser.add_argument("--proxy-host", default=dataset_mod.DEFAULT_PROXY_HOST)
    parser.add_argument("--proxy-port", type=int, default=dataset_mod.DEFAULT_PROXY_PORT)
    parser.add_argument("--no-proxy-probe", dest="use_proxy_probe", action="store_false")
    parser.set_defaults(use_proxy_probe=True, prefer_local_dataset=True, cache_dataset=True)
    parser.add_argument("--no-local-dataset", dest="prefer_local_dataset", action="store_false")
    parser.add_argument("--no-cache-dataset", dest="cache_dataset", action="store_false")

    # ── output ────────────────────────────────────────────────
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "generalization_test_results"))
    parser.add_argument("--debug", action="store_true", help="Enable per-turn debug output.")

    # ── Tinker auth ──────────────────────────────────────────
    parser.add_argument("--no-token-prompt", action="store_true",
                        help="Fail instead of prompting for TINKER_API_KEY.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_targets < 1:
        raise ValueError("--num-targets must be >= 1")
    if args.trials_per_target < 1:
        raise ValueError("--trials-per-target must be >= 1")
    if args.max_turns < 1:
        raise ValueError("--max-turns must be >= 1")

    rng = np.random.default_rng(args.rng_seed)

    # ── 1. load RL core module ───────────────────────────────────
    print("Loading RL core module...")
    core = _load_rl_core()
    core.configure_columns(args.prompt_col, args.answer_col)

    # ── 2. connect to Tinker ─────────────────────────────────────
    print("Connecting to Tinker...")
    service_client = model_utils.create_service_client(
        prompt_for_missing_tokens=not args.no_token_prompt
    )

    # ── 3. load tokenizer & renderer ─────────────────────────────
    local_tok_dir = (
        args.local_tokenizer_dir
        if args.local_tokenizer_dir
        else str(model_utils.local_tokenizer_dir_for_model(args.base_model))
    )
    tokenizer, renderer = model_utils.load_tokenizer_and_renderer(
        args.base_model,
        local_tokenizer_dir=local_tok_dir,
        prefer_local_tokenizer=args.prefer_local_tokenizer,
        cache_tokenizer=args.cache_tokenizer,
    )

    # ── 4. load trained LoRA weights → sampling client ───────────
    print(f"Loading checkpoint: {args.checkpoint}")
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
    )
    training_client.load_state(args.checkpoint)
    sampling_client = training_client.save_weights_and_get_sampling_client()
    print("Sampling client ready.")

    # ── 5. load dataset (for starting molecule pool) ─────────────
    raw_data = dataset_mod.load_raw_data(
        dataset_name=args.dataset_name,
        data_split=args.data_split,
        local_data_dir=args.local_data_dir,
        prefer_local=args.prefer_local_dataset,
        cache_local=args.cache_dataset,
        export_jsonl=True,
        use_proxy_probe=args.use_proxy_probe,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
    )
    dataset_mod.validate_columns(raw_data, prompt_col=args.prompt_col, answer_col=args.answer_col)
    print(f"Dataset loaded: {len(raw_data)} examples.")

    # ── 6. sample target tuples ──────────────────────────────────
    targets = sample_target_tuples(rng, args.num_targets)
    print(f"Sampled {len(targets)} target tuples.")

    # ── 7. evaluate each target ──────────────────────────────────
    trial_kwargs = dict(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        renderer=renderer,
        core=core,
        max_turns=args.max_turns,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    all_target_results: list[dict[str, Any]] = []
    t_start_all = time.monotonic()

    for t_idx, target in enumerate(targets):
        t_start = time.monotonic()

        # select starting molecule
        try:
            start_mol, start_props, start_smiles, prompt = select_starting_molecule(
                raw_data, rng, target, core=core,
            )
        except RuntimeError as exc:
            print(f"[target {t_idx:3d}/{args.num_targets}] SKIPPED: {exc}")
            continue

        debug_prefix = f"[target{t_idx}] " if args.debug else ""

        # run best-of-N
        best, all_trials = run_best_of_n(
            prompt, n=args.trials_per_target, debug=args.debug, debug_prefix=debug_prefix,
            **trial_kwargs,
        )

        elapsed = time.monotonic() - t_start

        best_nte = best.get("nte")
        init_nte = best.get("initial_nte")
        n_valid = sum(1 for t in all_trials if t.get("valid"))
        # per-property: start → target → best final
        sq, sl, sm = start_props["qed"], start_props["logp"], start_props["mw"]
        tq, tl, tm = target["qed"], target["logp"], target["mw"]
        bq = best.get("qed_out")
        bl = best.get("logp_out")
        bm = best.get("mw_out")

        if isinstance(best_nte, (int, float)) and isinstance(init_nte, (int, float)):
            rel_imp = (init_nte - best_nte) / max(float(init_nte), 1e-6)
            # per-property gap to target (absolute error)
            qe = abs(bq - tq) if isinstance(bq, (int, float)) else None
            le = abs(bl - tl) if isinstance(bl, (int, float)) else None
            me = abs(bm - tm) if isinstance(bm, (int, float)) else None
            def _ge(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "?"
            print(
                f"[target {t_idx:3d}/{args.num_targets}] "
                f"|QED_err|={_ge(qe)} | "
                f"|LogP_err|={_ge(le)} | "
                f"|MW_err|={_ge(me)} | "
                f"nte {init_nte:.2f}→{best_nte:.4f} ({rel_imp*100:.0f}%↓)"
                f" | {n_valid}/{args.trials_per_target} valid"
                f" | {elapsed:.1f}s"
            )
        else:
            print(
                f"[target {t_idx:3d}/{args.num_targets}] "
                f"best_nte=N/A | {elapsed:.1f}s"
            )

        all_target_results.append({
            "target_index": int(t_idx),
            "target": target,
            "starting_smiles": start_smiles,
            "starting_props": start_props,
            "best": best,
            "all_trials": all_trials,
        })

    total_elapsed = time.monotonic() - t_start_all
    print(f"Evaluation complete. Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # ── 8. aggregate ────────────────────────────────────────────
    aggregate = compute_aggregate_metrics(all_target_results)
    print("\nAggregate metrics:")
    print(f"  mean best-of-{args.trials_per_target} NTE:  {aggregate['mean_best_nte']:.4f}" if aggregate['mean_best_nte'] is not None else "  mean NTE: N/A")
    print(f"  std  best-of-{args.trials_per_target} NTE:  {aggregate['std_best_nte']:.4f}" if aggregate['std_best_nte'] is not None else "")
    print(f"  turn acceptance rate:  {aggregate['turn_acceptance_rate']:.3f}" if aggregate['turn_acceptance_rate'] is not None else "")
    print(f"  turn validity rate:    {aggregate['turn_validity_rate']:.3f}" if aggregate['turn_validity_rate'] is not None else "")
    print(f"  trial validity rate:   {aggregate['trial_validity_rate']:.3f}" if aggregate['trial_validity_rate'] is not None else "")
    for label, bs in aggregate.get("nte_buckets", {}).items():
        if bs["count"] > 0:
            print(f"  {label} bucket (n={bs['count']}): mean_best_nte={bs['mean_best_nte']:.4f}" if bs['mean_best_nte'] is not None else f"  {label} bucket (n={bs['count']}): N/A")

    # ── 9. save ──────────────────────────────────────────────────
    config = {
        "checkpoint": args.checkpoint,
        "base_model": args.base_model,
        "num_targets": args.num_targets,
        "trials_per_target": args.trials_per_target,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "rng_seed": args.rng_seed,
        "lora_rank": args.lora_rank,
        "dataset_name": args.dataset_name,
        "data_split": args.data_split,
    }
    artifacts = save_generalization_results(
        all_target_results,
        aggregate,
        output_dir=args.output_dir,
        run_tag=args.run_tag,
        config=config,
    )
    print(f"\nResults saved to: {artifacts['output_dir']}")
    print(f"  {artifacts['trial_results']}")
    print(f"  {artifacts['best_per_target']}")
    print(f"  {artifacts['summary']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
