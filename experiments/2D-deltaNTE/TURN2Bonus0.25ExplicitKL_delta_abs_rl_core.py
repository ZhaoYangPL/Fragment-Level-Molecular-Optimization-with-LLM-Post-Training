from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from typing import List
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import BRICS
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from tinker import types


RDLogger.DisableLog("rdApp.*")

PROMPT_COL = "input"
ANSWER_COL = "output"

FORMAT_PENALTY = -10.0
INVALIDITY_PENALTY = -10.0
REPETITION_PENALTY_STEP = 0.1
DELTA_NTE_WEIGHT = 2.0
ABS_NTE_WEIGHT = 0.5
DELTA_NTE_RELATIVE_EPS = 1e-6
DELTA_NTE_RELATIVE_MIN = -4.0
DELTA_NTE_RELATIVE_MAX = 1.0
ABS_NTE_CAP = 2.0
ABS_NTE_QUALITY_MAX = 1.0
EARLY_STOP_INITIAL_NTE_CUTOFF = 2.0
EARLY_STOP_LARGE_NTE_THRESHOLD = 0.5
EARLY_STOP_SMALL_NTE_RATIO = 0.25
EARLY_STOP_BONUS = 1.0
NTE_MILESTONE_BONUSES = (
    (1, 0.75, 0.25),
    (2, 0.50, 0.50),
    (3, 0.25, 0.75),
)
INITIAL_NTE_BUCKETS = (
    ("low", "low<=0.5", None, 0.5),
    ("mid", "mid(0.5,2]", 0.5, 2.0),
    ("high", "high>2", 2.0, None),
)

_REPLACE_FORMAT_RE = re.compile(
    r"^Replace\s+(.+?)\s+with\s+(.+?)\s+to\s+form\s+<SMILES>([^<>]+)</SMILES>\.$",
    re.DOTALL,
)
_ADD_FORMAT_RE = re.compile(
    r"^Add\s+(.+?)\s+to\s+form\s+<SMILES>([^<>]+)</SMILES>\.$",
    re.DOTALL,
)
_REMOVE_FORMAT_RE = re.compile(
    r"^Remove\s+(.+?)\s+to\s+form\s+<SMILES>([^<>]+)</SMILES>\.$",
    re.DOTALL,
)

TARGET_TOLERANCE = {
    "QED": 0.02,
    "LogP": 0.20,
    "MW": 5.0,
}

SYSTEM_PROMPT_RL = """
You are a helpful chemistry assistant for multi-turn molecule optimization.

Rules:
1) Output exactly one edit action sentence in one of these formats:
   - Replace <old_fragment> with <new_fragment> to form <SMILES>new_smiles_here</SMILES>.
   - Add <fragment> to form <SMILES>new_smiles_here</SMILES>.
   - Remove <fragment> to form <SMILES>new_smiles_here</SMILES>.
2) Do not output analysis/plan/verification/think blocks.
3) In turn >= 2, if the user provides a 'Current candidate molecule' SMILES,
   you must edit that candidate molecule, not the original input molecule.
"""


def configure_columns(prompt_col: str, answer_col: str) -> None:
    global PROMPT_COL, ANSWER_COL
    PROMPT_COL = prompt_col
    ANSWER_COL = answer_col


def _normalize_completion_text(text: str | None) -> str:
    if not text:
        return ""
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)
    return clean.replace("<|im_end|>", "").strip()


def extract_smiles(text: str | None) -> str | None:
    if not text:
        return None
    matches = re.findall(r"<SMILES>(.*?)</SMILES>", str(text), flags=re.DOTALL)
    if len(matches) != 1:
        return None
    smiles = matches[0].strip()
    if "<SMILES>" in smiles or "</SMILES>" in smiles:
        return None
    return smiles


def _has_strict_action_format(text: str) -> bool:
    if text.count("<SMILES>") != 1 or text.count("</SMILES>") != 1:
        return False
    return any(
        pattern.fullmatch(text)
        for pattern in (_REPLACE_FORMAT_RE, _ADD_FORMAT_RE, _REMOVE_FORMAT_RE)
    )


def strict_format_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for text in completions:
        normalized = _normalize_completion_text(text)
        rewards.append(1.0 if _has_strict_action_format(normalized) else FORMAT_PENALTY)
    return rewards


def format_r(completions, **kwargs):
    if isinstance(completions, str):
        return strict_format_reward([completions], **kwargs)[0]
    return strict_format_reward(list(completions), **kwargs)


def _property_values(mol) -> dict[str, float]:
    return {
        "qed": float(QED.qed(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "mw": float(Descriptors.ExactMolWt(mol)),
    }


def _mean_or_none(values: list[float]):
    return float(np.mean(values)) if values else None


def _std_or_none(values: list[float]):
    return float(np.std(values)) if values else None


def _bucket_label_for_initial_nte(initial_nte: float | None) -> str | None:
    if initial_nte is None:
        return None
    if initial_nte <= 0.5:
        return "low"
    if initial_nte <= 2.0:
        return "mid"
    return "high"


def _initial_nte_bucket_stats(records: list[dict]) -> dict[str, float | int | None]:
    stats: dict[str, float | int | None] = {}
    by_bucket = {label: [] for label, _display, _low, _high in INITIAL_NTE_BUCKETS}
    for record in records:
        initial_nte = record.get("initial_nte")
        if not isinstance(initial_nte, (int, float)):
            continue
        label = _bucket_label_for_initial_nte(float(initial_nte))
        if label is not None:
            by_bucket[label].append(record)

    for label, _display, _low, _high in INITIAL_NTE_BUCKETS:
        bucket_records = by_bucket[label]
        initial_values = [
            float(record["initial_nte"])
            for record in bucket_records
            if isinstance(record.get("initial_nte"), (int, float))
        ]
        terminal_values = [
            float(record["nte"])
            for record in bucket_records
            if isinstance(record.get("nte"), (int, float))
        ]
        rel_values = [
            float(record["nte_relative_improvement"])
            for record in bucket_records
            if isinstance(record.get("nte_relative_improvement"), (int, float))
        ]
        prefix = f"initial_nte_{label}"
        stats[f"{prefix}_count"] = len(bucket_records)
        stats[f"{prefix}_valid_count"] = len(rel_values)
        stats[f"{prefix}_mean"] = _mean_or_none(initial_values)
        stats[f"{prefix}_terminal_nte_mean"] = _mean_or_none(terminal_values)
        stats[f"{prefix}_relative_improvement_mean"] = _mean_or_none(rel_values)
        stats[f"{prefix}_relative_improvement_std"] = _std_or_none(rel_values)
    return stats


def _prompt_guidance_for_initial_nte(initial_nte: float | None) -> str:
    bucket = _bucket_label_for_initial_nte(initial_nte)
    if bucket == "low":
        return (
            "Optimization mode: conservative low-NTE refinement.\n"
            "The current task starts already close to the target. Prefer the smallest valid "
            "fragment-level edit.\n"
            "Avoid large scaffold, molecular-weight, LogP, or QED shifts.\n"
            "A tiny safe improvement is better than a risky large edit.\n"
            "Preserve already-good properties while making only a minimal correction."
        )
    if bucket == "high":
        return (
            "Optimization mode: aggressive target-gap reduction.\n"
            "The current task starts far from the target. Larger fragment edits are acceptable "
            "if they reduce the target gaps.\n"
            "Prioritize reducing the largest QED, LogP, or MW errors while keeping the output valid."
        )
    return (
        "Optimization mode: balanced refinement.\n"
        "Improve the remaining target gaps while keeping already-good properties near target.\n"
        "Prefer fragment edits that reduce NTE without causing large overshoot."
    )


def _parse_property_targets(prompt_text: str, mol_in):
    specs = {}
    for prop in ("QED", "LogP", "MW"):
        match = re.search(rf"<{prop}>(-?\d*\.?\d+)</{prop}>\s*(higher|lower)", prompt_text)
        if not match:
            return None
        delta = float(match.group(1))
        direction = 1.0 if match.group(2).lower() == "higher" else -1.0
        specs[prop.lower()] = (delta, direction)

    input_props = _property_values(mol_in)
    target = {
        "qed": max(0.01, min(0.99, input_props["qed"] + specs["qed"][1] * specs["qed"][0])),
        "logp": input_props["logp"] + specs["logp"][1] * specs["logp"][0],
        "mw": input_props["mw"] + specs["mw"][1] * specs["mw"][0],
    }
    return target, input_props


def property_error(mol, target: dict[str, float]) -> tuple[float, dict[str, float], dict[str, float]]:
    props = _property_values(mol)
    parts = {
        "qed_err": abs(props["qed"] - target["qed"]) / 0.5,
        "logp_err": abs(props["logp"] - target["logp"]) / 3.0,
        "mw_err": abs(props["mw"] - target["mw"]) / 50.0,
    }
    total = float(parts["qed_err"] + parts["logp_err"] + parts["mw_err"])
    return total, parts, props


def normalized_total_error(qed_error: float, logp_error: float, mw_error: float) -> float:
    return float(abs(qed_error) / 0.5 + abs(logp_error) / 3.0 + abs(mw_error) / 50.0)


def _initial_nte_from_prompt(prompt_text: str) -> float | None:
    in_smiles = extract_smiles(prompt_text)
    if not in_smiles:
        return None
    mol_in = Chem.MolFromSmiles(in_smiles)
    if mol_in is None:
        return None
    parsed = _parse_property_targets(prompt_text, mol_in)
    if parsed is None:
        return None
    target, _input_props = parsed
    initial_nte, _parts, _props = property_error(mol_in, target)
    return float(initial_nte)


def _sample_prompt_indices_with_initial_nte_filter(
    raw_data,
    rng,
    *,
    candidate_prompts_per_iter: int,
    max_kept_prompts_per_iter: int,
    prompt_initial_nte_min: float,
) -> tuple[list[int], dict[str, int | float]]:
    candidate_count = min(int(candidate_prompts_per_iter), len(raw_data))
    prompt_indices: list[int] = []
    skipped_low_nte = 0
    skipped_unparseable = 0

    if candidate_count > 0:
        candidate_indices = rng.choice(len(raw_data), size=candidate_count, replace=False)
    else:
        candidate_indices = []

    for idx in candidate_indices:
        int_idx = int(idx)
        example = raw_data[int_idx]
        prompt_text = (
            example.get(PROMPT_COL)
            if hasattr(example, "get")
            else example[PROMPT_COL]
        )
        initial_nte = _initial_nte_from_prompt(str(prompt_text))
        if initial_nte is None:
            skipped_unparseable += 1
            continue
        if initial_nte < prompt_initial_nte_min:
            skipped_low_nte += 1
            continue
        prompt_indices.append(int_idx)
        if len(prompt_indices) >= max_kept_prompts_per_iter:
            break

    stats = {
        "prompt_filter_candidates": int(candidate_count),
        "prompt_filter_kept": int(len(prompt_indices)),
        "prompt_filter_skipped_low_nte": int(skipped_low_nte),
        "prompt_filter_skipped_unparseable": int(skipped_unparseable),
        "prompt_initial_nte_min": float(prompt_initial_nte_min),
    }
    return prompt_indices, stats


def _early_stop_threshold(initial_nte: float | None) -> float | None:
    if initial_nte is None:
        return None
    if initial_nte > EARLY_STOP_INITIAL_NTE_CUTOFF:
        return float(EARLY_STOP_LARGE_NTE_THRESHOLD)
    return float(EARLY_STOP_SMALL_NTE_RATIO * initial_nte)


def _milestone_level_for_nte(initial_nte: float | None, current_nte: float | None) -> int:
    if initial_nte is None or current_nte is None or initial_nte <= 0.0:
        return 0
    level = 0
    for milestone_level, ratio, _bonus in NTE_MILESTONE_BONUSES:
        if current_nte <= ratio * initial_nte:
            level = int(milestone_level)
    return level


def _milestone_bonus_for_level_change(previous_level: int, current_level: int) -> float:
    if current_level <= previous_level:
        return 0.0
    bonus = 0.0
    for milestone_level, _ratio, milestone_bonus in NTE_MILESTONE_BONUSES:
        if previous_level < milestone_level <= current_level:
            bonus += float(milestone_bonus)
    return float(bonus)


def compute_final_nte_details(final_smiles: str | None, initial_prompt_text: str) -> dict:
    details = {
        "valid": False,
        "failure": None,
        "nte": None,
        "qed_error": None,
        "logp_error": None,
        "mw_error": None,
        "qed_target": None,
        "logp_target": None,
        "mw_target": None,
        "qed_out": None,
        "logp_out": None,
        "mw_out": None,
    }
    in_smiles = extract_smiles(initial_prompt_text)
    if not final_smiles:
        details["failure"] = "missing_final_smiles"
        return details
    if not in_smiles:
        details["failure"] = "missing_initial_smiles"
        return details

    mol_in = Chem.MolFromSmiles(in_smiles)
    mol_out = Chem.MolFromSmiles(final_smiles)
    if mol_in is None:
        details["failure"] = "invalid_initial_smiles"
        return details
    if mol_out is None:
        details["failure"] = "invalid_final_smiles"
        return details

    parsed = _parse_property_targets(initial_prompt_text, mol_in)
    if parsed is None:
        details["failure"] = "target_parse_failed"
        return details
    target, _input_props = parsed
    out_props = _property_values(mol_out)

    qed_error = abs(out_props["qed"] - target["qed"])
    logp_error = abs(out_props["logp"] - target["logp"])
    mw_error = abs(out_props["mw"] - target["mw"])
    details.update(
        {
            "valid": True,
            "failure": None,
            "nte": normalized_total_error(qed_error, logp_error, mw_error),
            "qed_error": float(qed_error),
            "logp_error": float(logp_error),
            "mw_error": float(mw_error),
            "qed_target": float(target["qed"]),
            "logp_target": float(target["logp"]),
            "mw_target": float(target["mw"]),
            "qed_out": float(out_props["qed"]),
            "logp_out": float(out_props["logp"]),
            "mw_out": float(out_props["mw"]),
        }
    )
    return details


def compute_property_reward_details(
    completion_text: str,
    prompt_text: str,
    initial_nte: float | None = None,
) -> dict:
    text = _normalize_completion_text(completion_text)
    in_smiles = extract_smiles(prompt_text)
    out_smiles = extract_smiles(text)
    bucket_initial_nte = (
        float(initial_nte)
        if isinstance(initial_nte, (int, float))
        else _initial_nte_from_prompt(prompt_text)
    )
    reward_bucket = _bucket_label_for_initial_nte(bucket_initial_nte) or "unknown"
    details = {
        "property_reward": float(INVALIDITY_PENALTY),
        "property_failure": None,
        "property_reward_bucket": reward_bucket,
        "property_reward_bucket_initial_nte": bucket_initial_nte,
        "property_reward_bucket_delta": None,
        "total_err": None,
        "old_total_err": None,
        "property_delta_nte": None,
        "property_delta_nte_raw": None,
        "property_delta_nte_clipped": None,
        "property_absolute_delta_nte": None,
        "property_relative_delta_nte": None,
        "property_relative_delta_nte_raw": None,
        "property_relative_delta_nte_clipped": None,
        "property_abs_nte": None,
        "property_abs_nte_quality": None,
        "property_delta_weight": float(DELTA_NTE_WEIGHT),
        "property_abs_weight": float(ABS_NTE_WEIGHT),
        "qed_err": None,
        "logp_err": None,
        "mw_err": None,
        "qed_target": None,
        "logp_target": None,
        "mw_target": None,
        "qed_out": None,
        "logp_out": None,
        "mw_out": None,
    }
    if not in_smiles or not out_smiles:
        details["property_failure"] = "missing_smiles"
        return details

    mol_in = Chem.MolFromSmiles(in_smiles)
    mol_out = Chem.MolFromSmiles(out_smiles)
    if mol_in is None or mol_out is None:
        details["property_failure"] = "invalid_smiles"
        return details

    parsed = _parse_property_targets(prompt_text, mol_in)
    if parsed is None:
        details["property_failure"] = "target_parse_failed"
        return details
    target, _input_props = parsed

    old_total_err, _old_parts, _old_props = property_error(mol_in, target)
    total_err, parts, output_props = property_error(mol_out, target)
    absolute_delta_nte = float(old_total_err - total_err)
    relative_delta_nte_raw = float(
        absolute_delta_nte / max(float(old_total_err), DELTA_NTE_RELATIVE_EPS)
    )
    relative_delta_nte_clipped = float(
        np.clip(
            relative_delta_nte_raw,
            DELTA_NTE_RELATIVE_MIN,
            DELTA_NTE_RELATIVE_MAX,
        )
    )
    abs_nte = float(total_err)
    abs_nte_quality = float(
        ABS_NTE_QUALITY_MAX
        - np.clip(abs_nte / ABS_NTE_CAP, 0.0, ABS_NTE_QUALITY_MAX)
    )
    if reward_bucket == "low":
        bucket_delta = float(np.clip(relative_delta_nte_raw, -2.0, 0.5))
        property_reward = float(0.5 * abs_nte_quality + 2.0 * bucket_delta)
    elif reward_bucket == "high":
        bucket_delta = relative_delta_nte_clipped
        property_reward = float(1.0 * relative_delta_nte_clipped + 1.0 * abs_nte_quality)
    else:
        bucket_delta = relative_delta_nte_clipped
        property_reward = float(2.0 * relative_delta_nte_clipped + 0.5 * abs_nte_quality)
    details.update(
        {
            "property_reward": property_reward,
            "property_failure": None,
            "property_reward_bucket": reward_bucket,
            "property_reward_bucket_initial_nte": bucket_initial_nte,
            "property_reward_bucket_delta": bucket_delta,
            "total_err": float(total_err),
            "old_total_err": float(old_total_err),
            "property_delta_nte": relative_delta_nte_raw,
            "property_delta_nte_raw": relative_delta_nte_raw,
            "property_delta_nte_clipped": relative_delta_nte_clipped,
            "property_absolute_delta_nte": absolute_delta_nte,
            "property_relative_delta_nte": relative_delta_nte_raw,
            "property_relative_delta_nte_raw": relative_delta_nte_raw,
            "property_relative_delta_nte_clipped": relative_delta_nte_clipped,
            "property_abs_nte": abs_nte,
            "property_abs_nte_quality": abs_nte_quality,
            "property_delta_weight": float(DELTA_NTE_WEIGHT),
            "property_abs_weight": float(ABS_NTE_WEIGHT),
            "qed_err": float(parts["qed_err"]),
            "logp_err": float(parts["logp_err"]),
            "mw_err": float(parts["mw_err"]),
            "qed_target": float(target["qed"]),
            "logp_target": float(target["logp"]),
            "mw_target": float(target["mw"]),
            "qed_out": float(output_props["qed"]),
            "logp_out": float(output_props["logp"]),
            "mw_out": float(output_props["mw"]),
        }
    )
    return details


def prop_r(completions: List[str], **kwargs) -> List[float]:
    prompts: List[str] = kwargs["prompts"]
    initial_ntes = kwargs.get("initial_ntes")
    if initial_ntes is None:
        initial_ntes = [None] * len(prompts)
    return [
        float(
            compute_property_reward_details(
                comp,
                prompt,
                initial_nte=initial_nte,
            )["property_reward"]
        )
        for comp, prompt, initial_nte in zip(completions, prompts, initial_ntes)
    ]


def multi_property_reward(completions: List[str], **kwargs) -> List[float]:
    return prop_r(completions, **kwargs)


def compute_reward_components(
    completion_text: str,
    prompt_text: str,
    ground_truth: str | None = None,
    *,
    repetition_penalty: float = 0.0,
    repetition_count: int = 0,
    initial_nte: float | None = None,
    **_unused,
) -> dict:
    text = _normalize_completion_text(completion_text)
    pred_smiles = extract_smiles(text)
    mol_pred = Chem.MolFromSmiles(pred_smiles) if pred_smiles else None

    format_score = float(strict_format_reward([text])[0])
    prop_details = compute_property_reward_details(
        text,
        prompt_text,
        initial_nte=initial_nte,
    )
    property_score = float(prop_details["property_reward"])
    repetition_score = float(repetition_penalty)
    total = float(format_score + property_score + repetition_score)

    out = {
        "strict": format_score,
        "format": format_score,
        "property": property_score,
        "repetition": repetition_score,
        "repetition_count": int(repetition_count),
        "total": total,
        "has_smiles": bool(pred_smiles),
        "valid_smiles": mol_pred is not None,
        "valid_format": format_score > 0.0,
    }
    out.update(prop_details)
    return out


def compute_reward(
    completion_text: str,
    prompt_text: str,
    ground_truth: str | None = None,
    **kwargs,
) -> float:
    return float(
        compute_reward_components(
            completion_text,
            prompt_text,
            ground_truth,
            **kwargs,
        )["total"]
    )


def compute_group_advantages(rewards: list[float]) -> list[float]:
    arr = np.array(rewards, dtype=np.float64)
    std = arr.std()
    if std < 1e-8:
        return [0.0] * len(rewards)
    return ((arr - arr.mean()) / std).tolist()


def _mol_from_prompt(prompt_text: str) -> Optional[Chem.Mol]:
    match = re.search(r"<SMILES>(.*?)</SMILES>", prompt_text)
    if not match:
        return None
    return Chem.MolFromSmiles(match.group(1))


def _target_progress_lines(mol_in: Chem.Mol, mol_out: Chem.Mol, prompt_text: str) -> tuple[list[str], list[str]]:
    prop_specs = [
        ("QED", QED.qed, "{:.3f}"),
        ("LogP", Crippen.MolLogP, "{:.3f}"),
        ("MW", Descriptors.ExactMolWt, "{:.3f}"),
    ]

    misses = []
    lines = []

    for prop, fn, fmt in prop_specs:
        match = re.search(rf"<{prop}>(-?\d*\.?\d+)</{prop}>\s*(higher|lower)", prompt_text)
        if not match:
            continue

        delta = float(match.group(1))
        direction = match.group(2).lower()

        base_val = float(fn(mol_in))
        current_val = float(fn(mol_out))
        sign = 1.0 if direction == "higher" else -1.0
        target_val = base_val + sign * delta

        if prop == "QED":
            target_val = max(0.01, min(0.99, target_val))

        if direction == "higher":
            signed_need = target_val - current_val
            action = "increase"
        else:
            signed_need = current_val - target_val
            action = "decrease"

        still_needed = abs(signed_need)
        tol = TARGET_TOLERANCE.get(prop, 0.0)
        eps = 1e-12
        if direction == "higher":
            direction_met = (current_val - base_val) >= -eps
        else:
            direction_met = (base_val - current_val) >= -eps
        met = still_needed <= tol

        if signed_need > 0:
            progress_desc = f"need {fmt.format(still_needed)} more {action}"
        elif signed_need < 0:
            progress_desc = f"overshot by {fmt.format(still_needed)} in {action} direction"
        else:
            progress_desc = "exactly on target"

        if not met:
            misses.append(prop)

        lines.append(
            f"{prop}: base={fmt.format(base_val)}, current={fmt.format(current_val)}, "
            f"target={fmt.format(target_val)} ({direction} by {delta:.3f}), "
            f"still_needed={fmt.format(still_needed)} ({progress_desc}), "
            f"tol={fmt.format(tol)}, direction_met={direction_met}, met={met}"
        )

    return misses, lines


def _run_env(completion_text: str, prompt_text: str) -> tuple[str, Optional[Chem.Mol], bool]:
    text = _normalize_completion_text(completion_text)
    pred_smiles = extract_smiles(text)
    if not pred_smiles:
        return "Your output did not contain a valid <SMILES>...</SMILES> block. Please revise.", None, False

    mol = Chem.MolFromSmiles(pred_smiles)
    if mol is None:
        return "The SMILES you provided is chemically invalid. Please revise.", None, False

    qed_val = QED.qed(mol)
    logp_val = Crippen.MolLogP(mol)
    mw_val = Descriptors.ExactMolWt(mol)

    mol_in = _mol_from_prompt(prompt_text)
    if mol_in is None:
        feedback = (
            f"Environment feedback: QED={qed_val:.3f}, LogP={logp_val:.2f}, MW={mw_val:.1f}. "
            "Input molecule is missing/invalid in prompt, so target gap is unavailable."
        )
        return feedback, mol, False

    misses, progress_lines = _target_progress_lines(mol_in, mol, prompt_text)
    all_met = len(progress_lines) > 0 and len(misses) == 0

    if progress_lines:
        progress_block = "\n- " + "\n- ".join(progress_lines)
    else:
        progress_block = "\n- No parseable target tags (<QED>/<LogP>/<MW>) found in prompt."

    if all_met:
        summary = "All targets met within tolerance. Keep output concise and chemically valid."
    else:
        unmet = ", ".join(misses) if misses else "unknown"
        summary = f"Targets not met (by target gap tolerance): {unmet}. Please refine toward target values."

    feedback = (
        f"Environment feedback: QED={qed_val:.3f}, LogP={logp_val:.2f}, MW={mw_val:.1f}."
        f"\nTarget progress:{progress_block}"
        f"\n{summary}"
    )
    return feedback, mol, all_met


class EMABaseline:
    def __init__(self, alpha: float = 0.05, init: float = 0.0):
        self.alpha = alpha
        self.value = init
        self._warmed = False

    def update_and_get(self, G: float) -> float:
        current = self.value
        if not self._warmed:
            self.value = G
            self._warmed = True
        else:
            self.value = (1 - self.alpha) * self.value + self.alpha * G
        return current


def get_clean_fragments(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    try:
        mol_broken = BRICS.BreakBRICSBonds(mol)
        fragments = Chem.GetMolFrags(mol_broken, asMols=True, sanitizeFrags=True)
        clean_frags = []
        for frag in fragments:
            editable = Chem.EditableMol(frag)
            to_remove = [
                atom.GetIdx()
                for atom in frag.GetAtoms()
                if atom.GetSymbol() == "*" and atom.GetAtomicNum() == 0
            ]
            for idx in sorted(to_remove, reverse=True):
                editable.RemoveAtom(idx)
            cleaned = editable.GetMol()
            Chem.SanitizeMol(cleaned)
            clean_frags.append(Chem.MolToSmiles(cleaned, isomericSmiles=True))
        return clean_frags
    except Exception as exc:
        print(f"Error in fragmenting molecule: {exc}")
        return []


def run_mode_b(
    example: dict,
    renderer,
    sampling_client,
    tokenizer,
    ema_baseline: EMABaseline,
    num_trajectories: int = 8,
    max_turns: int = 3,
    gamma: float = 0.85,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    debug: bool = False,
    debug_prefix: str = "",
) -> tuple[list[types.Datum], list[list[float]], list[float], dict]:
    prompt_text = str(example[PROMPT_COL])
    ground_truth = str(example[ANSWER_COL])

    def _replace_prompt_smiles(base_prompt: str, current_smiles: str) -> str:
        new_prompt = re.sub(
            r"<SMILES>.*?</SMILES>",
            lambda _m: f"<SMILES>{current_smiles}</SMILES>",
            base_prompt,
            count=1,
            flags=re.DOTALL,
        )

        try:
            fragment_list = get_clean_fragments(current_smiles)
        except Exception:
            fragment_list = []
        if not fragment_list:
            fragment_list = [current_smiles]

        fragment_repr = repr(fragment_list)
        new_prompt = re.sub(
            r"(<SMILES>.*?</SMILES>)(?:,\s*which is composed of fragments\s+.*?\.)?",
            lambda match: f"{match.group(1)}, which is composed of fragments {fragment_repr}.",
            new_prompt,
            count=1,
            flags=re.DOTALL,
        )

        mol_old = _mol_from_prompt(base_prompt)
        mol_new = Chem.MolFromSmiles(current_smiles)
        if mol_old is None or mol_new is None:
            return new_prompt

        prop_specs = [
            ("QED", QED.qed, True),
            ("LogP", Crippen.MolLogP, False),
            ("MW", Descriptors.ExactMolWt, False),
        ]
        for prop, fn, is_qed in prop_specs:
            match = re.search(rf"<{prop}>(-?\d*\.?\d+)</{prop}>\s*(higher|lower)", new_prompt)
            if not match:
                continue
            old_delta = float(match.group(1))
            old_dir = match.group(2).lower()
            old_base = float(fn(mol_old))
            target_abs = old_base + (old_delta if old_dir == "higher" else -old_delta)
            if is_qed:
                target_abs = max(0.01, min(0.99, target_abs))

            new_base = float(fn(mol_new))
            new_dir = "higher" if target_abs >= new_base else "lower"
            new_delta = abs(target_abs - new_base)
            new_prompt = re.sub(
                rf"<{prop}>-?\d*\.?\d+</{prop}>\s*(higher|lower)",
                f"<{prop}>{new_delta:.3f}</{prop}> {new_dir}",
                new_prompt,
                count=1,
            )
        return new_prompt

    def _compact_env_feedback(feedback_text: str) -> str:
        if not feedback_text:
            return ""
        lines = [line.strip() for line in feedback_text.splitlines() if line.strip()]
        if not lines:
            return feedback_text
        if lines[0].startswith("Environment feedback:"):
            if len(lines) == 1:
                return lines[0]
            tail = lines[-1]
            if tail == lines[0]:
                return lines[0]
            return lines[0] + "\n" + tail
        return feedback_text

    def _rollback_state_feedback(current_prompt: str) -> tuple[str, str, bool]:
        rollback_mol = _mol_from_prompt(current_prompt)
        if rollback_mol is None:
            feedback = "Environment feedback: rollback molecule is missing or invalid in the current prompt."
            guidance = (
                "all_met=False; unmet_properties=unknown.\n"
                "- target-gap details unavailable because the rollback molecule is invalid."
            )
            return feedback, guidance, False

        rollback_smiles = Chem.MolToSmiles(rollback_mol, isomericSmiles=True)
        qed_val = QED.qed(rollback_mol)
        logp_val = Crippen.MolLogP(rollback_mol)
        mw_val = Descriptors.ExactMolWt(rollback_mol)
        misses, progress_lines = _target_progress_lines(rollback_mol, rollback_mol, current_prompt)
        all_met_state = bool(progress_lines) and len(misses) == 0
        unmet_text = ", ".join(misses) if misses else ("none" if progress_lines else "unknown")
        guidance_block = (
            "\n".join([f"- {line}" for line in progress_lines])
            if progress_lines
            else "- target-gap details unavailable."
        )
        progress_block = (
            "\n- " + "\n- ".join(progress_lines)
            if progress_lines
            else "\n- No parseable target tags (<QED>/<LogP>/<MW>) found in prompt."
        )
        if all_met_state:
            summary = "Rollback molecule already meets all targets within tolerance."
        else:
            summary = f"Rollback molecule targets not met: {unmet_text}. Continue refining from this valid molecule."
        feedback = (
            "Environment feedback: previous attempt invalid/missing; rolled back to previous valid "
            f"molecule <SMILES>{rollback_smiles}</SMILES> with "
            f"QED={qed_val:.3f}, LogP={logp_val:.2f}, MW={mw_val:.1f}. "
            "This rollback molecule is the only edit base for the next turn."
            f"\nTarget progress:{progress_block}"
            f"\n{summary}"
        )
        guidance = f"all_met={all_met_state}; unmet_properties={unmet_text}.\n{guidance_block}"
        return feedback, guidance, all_met_state

    initial_mol = _mol_from_prompt(prompt_text)
    initial_smiles = extract_smiles(prompt_text)
    initial_canonical = Chem.MolToSmiles(initial_mol) if initial_mol is not None else None
    initial_nte = _initial_nte_from_prompt(prompt_text)
    early_stop_threshold_value = _early_stop_threshold(initial_nte)
    prompt_guidance = _prompt_guidance_for_initial_nte(initial_nte)

    K = int(num_trajectories)
    traj_states = [
        {
            "turns": [],
            "current_prompt": prompt_text,
            "prev_env_feedback": "",
            "prev_guidance": "",
            "done": False,
            "current_valid_smiles": initial_smiles if initial_canonical else None,
            "current_valid_canonical": initial_canonical,
            "current_valid_output_text": None,
            "has_valid_generated": False,
            "last_valid_generated_smiles": None,
            "last_valid_generated_canonical": None,
            "last_valid_generated_output_text": None,
            "best_milestone_level": 0,
        }
        for _ in range(K)
    ]

    for turn_idx in range(max_turns):
        active_indices = [k for k in range(K) if not traj_states[k]["done"]]
        if not active_indices:
            break

        prompt_inputs = {}
        prompt_tokens = {}
        for k in active_indices:
            st = traj_states[k]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RL},
                {"role": "user", "content": prompt_guidance},
                {"role": "user", "content": st["current_prompt"]},
            ]
            if turn_idx > 0 and (st["prev_env_feedback"] or st["prev_guidance"]):
                feedback_parts = [
                    "Environment feedback from your previous attempt:",
                    st["prev_env_feedback"],
                    "",
                    "Target-gap priorities from previous turn (must address first):",
                    st["prev_guidance"],
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

            prompt_input = renderer.build_generation_prompt(messages)
            prompt_inputs[k] = prompt_input
            prompt_tokens[k] = prompt_input.to_ints()

        futures = {
            k: sampling_client.sample(
                prompt=prompt_inputs[k],
                sampling_params=types.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=renderer.get_stop_sequences(),
                ),
                num_samples=1,
            )
            for k in active_indices
        }

        for k in active_indices:
            result = futures[k].result()
            seq = result.sequences[0]
            raw_completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            completion_text = _normalize_completion_text(raw_completion_text)

            st = traj_states[k]
            reward_components = compute_reward_components(
                completion_text,
                st["current_prompt"],
                ground_truth,
                initial_nte=initial_nte,
            )
            turn_reward = float(reward_components["total"])
            env_feedback, mol_out, all_met = _run_env(completion_text, st["current_prompt"])
            pred_smiles = extract_smiles(completion_text)
            valid_output_canonical = Chem.MolToSmiles(mol_out) if mol_out is not None else None
            turn_mol_in = _mol_from_prompt(st["current_prompt"])
            turn_input_smiles = extract_smiles(st["current_prompt"])
            turn_input_canonical = Chem.MolToSmiles(turn_mol_in) if turn_mol_in is not None else None
            valid_turn = bool(pred_smiles and turn_mol_in is not None and mol_out is not None)
            current_nte = (
                float(reward_components["total_err"])
                if isinstance(reward_components.get("total_err"), (int, float))
                else None
            )
            previous_milestone_level = int(st.get("best_milestone_level", 0))
            milestone_level = 0
            milestone_bonus = 0.0
            early_stop_met = False
            early_stop_bonus = 0.0

            if valid_turn:
                milestone_level = _milestone_level_for_nte(initial_nte, current_nte)
                milestone_bonus = _milestone_bonus_for_level_change(
                    previous_milestone_level,
                    milestone_level,
                )
                if milestone_level > previous_milestone_level:
                    st["best_milestone_level"] = milestone_level

                early_stop_met = (
                    early_stop_threshold_value is not None
                    and current_nte is not None
                    and current_nte <= early_stop_threshold_value
                )
                early_stop_bonus = float(EARLY_STOP_BONUS if early_stop_met else 0.0)

            bonus_total = float(milestone_bonus + early_stop_bonus)
            reward_components["base_total_before_bonus"] = float(reward_components["total"])
            reward_components["milestone_bonus"] = float(milestone_bonus)
            reward_components["milestone_level"] = int(milestone_level)
            reward_components["best_milestone_level"] = int(st.get("best_milestone_level", 0))
            reward_components["previous_milestone_level"] = int(previous_milestone_level)
            reward_components["early_stop_bonus"] = float(early_stop_bonus)
            reward_components["early_stop_met"] = bool(early_stop_met)
            reward_components["early_stop_threshold"] = early_stop_threshold_value
            reward_components["initial_nte"] = initial_nte
            reward_components["current_nte"] = current_nte
            reward_components["total"] = float(reward_components["total"] + bonus_total)
            turn_reward = float(turn_reward + bonus_total)

            st["turns"].append(
                {
                    "turn_index": int(turn_idx),
                    "prompt_tokens": prompt_tokens[k],
                    "comp_tokens": list(seq.tokens),
                    "comp_logprobs": list(seq.logprobs),
                    "turn_reward": turn_reward,
                    "reward_components": reward_components,
                    "input_smiles": turn_input_smiles,
                    "input_canonical_smiles": turn_input_canonical,
                    "turn_valid_canonical_smiles": valid_output_canonical,
                    "completion_text": completion_text,
                }
            )

            if debug:
                print(
                    f"[DBG]{debug_prefix} traj={k} turn={turn_idx} "
                    f"reward={turn_reward:.4f} all_met={all_met} "
                    f"fmt={reward_components['strict']:.3f} "
                    f"prop={reward_components['property']:.3f} "
                    f"rep={reward_components['repetition']:.3f} "
                    f"milestone={milestone_bonus:.3f} early={early_stop_bonus:.3f} "
                    f"valid_mol={mol_out is not None}"
                )
                print(f"[DBG]{debug_prefix} traj={k} turn={turn_idx} raw_output: {raw_completion_text!r}")
                print(f"[DBG]{debug_prefix} traj={k} turn={turn_idx} normalized_output: {completion_text}")

            if valid_turn:
                unmet_props, progress_lines = _target_progress_lines(
                    turn_mol_in, mol_out, st["current_prompt"]
                )
                unmet_text = ", ".join(unmet_props) if unmet_props else "none"
                progress_block = (
                    "\n".join([f"- {line}" for line in progress_lines])
                    if progress_lines
                    else "- target-gap details unavailable."
                )
                st["prev_guidance"] = (
                    f"all_met={all_met}; unmet_properties={unmet_text}.\n{progress_block}"
                )
                st["current_valid_smiles"] = pred_smiles
                st["current_valid_canonical"] = valid_output_canonical
                st["current_valid_output_text"] = completion_text
                st["has_valid_generated"] = True
                st["last_valid_generated_smiles"] = pred_smiles
                st["last_valid_generated_canonical"] = valid_output_canonical
                st["last_valid_generated_output_text"] = completion_text
                st["current_prompt"] = _replace_prompt_smiles(st["current_prompt"], pred_smiles)
                st["prev_env_feedback"] = _compact_env_feedback(env_feedback)
            else:
                rollback_feedback, rollback_guidance, _rollback_all_met = _rollback_state_feedback(
                    st["current_prompt"]
                )
                st["prev_guidance"] = rollback_guidance
                st["prev_env_feedback"] = _compact_env_feedback(rollback_feedback)
                if debug:
                    print(f"[DBG]{debug_prefix} traj={k} turn={turn_idx} rollback_state_feedback:")
                    print(rollback_feedback)

            if (turn_idx == max_turns - 1) or early_stop_met:
                st["done"] = True

    def _final_result_canonical(st: dict) -> str | None:
        if st.get("has_valid_generated"):
            return st.get("current_valid_canonical") or st.get("last_valid_generated_canonical")
        return None

    final_canonical_by_traj = [_final_result_canonical(st) for st in traj_states]

    def _monitor_terminal_record(traj_idx: int, st: dict) -> dict:
        turns = st.get("turns", [])
        last_turn = turns[-1] if turns else {}
        last_turn_canonical = last_turn.get("turn_valid_canonical_smiles")
        last_output_text = last_turn.get("completion_text")
        if last_turn_canonical:
            monitor_canonical = last_turn_canonical
            terminal_output_text = last_output_text
            source = "last_turn_valid"
        elif st.get("last_valid_generated_canonical"):
            monitor_canonical = st.get("last_valid_generated_canonical")
            terminal_output_text = st.get("last_valid_generated_output_text")
            source = "rollback_generated"
        elif st.get("current_valid_canonical"):
            monitor_canonical = st.get("current_valid_canonical")
            terminal_output_text = st.get("current_valid_output_text")
            source = "rollback_initial"
        else:
            monitor_canonical = None
            terminal_output_text = None
            source = "missing"
        return {
            "trajectory_index": int(traj_idx),
            "last_turn": int(last_turn.get("turn_index", -1)) + 1 if last_turn else None,
            "last_turn_valid": bool(last_turn_canonical),
            "source": source,
            "canonical_smiles": monitor_canonical,
            "terminal_output_text": terminal_output_text,
            "last_model_output_text": last_output_text,
        }

    terminal_molecule_records = [
        _monitor_terminal_record(traj_idx, st)
        for traj_idx, st in enumerate(traj_states)
    ]
    terminal_state_canonical_by_traj = [
        record.get("canonical_smiles") for record in terminal_molecule_records
    ]

    final_canonicals = [canonical for canonical in final_canonical_by_traj if canonical]
    final_counts = Counter(final_canonicals)
    for st, final_can in zip(traj_states, final_canonical_by_traj):
        repetition_count = max(int(final_counts.get(final_can, 0)) - 1, 0) if final_can else 0
        repetition_penalty = -REPETITION_PENALTY_STEP * repetition_count
        st["final_repetition_count"] = repetition_count
        st["final_repetition_penalty"] = float(repetition_penalty)
        if st["turns"]:
            last_turn = st["turns"][-1]
            components = last_turn["reward_components"]
            components["repetition"] = float(repetition_penalty)
            components["repetition_count"] = int(repetition_count)
            components["total"] = float(components["total"] + repetition_penalty)
            last_turn["turn_reward"] = float(last_turn["turn_reward"] + repetition_penalty)

    terminal_nte_records = [
        compute_final_nte_details(terminal_can, prompt_text)
        for terminal_can in terminal_state_canonical_by_traj
    ]
    initial_nte_value = (
        float(initial_nte) if isinstance(initial_nte, (int, float)) else None
    )
    for record, nte_record in zip(terminal_molecule_records, terminal_nte_records):
        record["nte"] = nte_record.get("nte")
        record["qed_error"] = nte_record.get("qed_error")
        record["logp_error"] = nte_record.get("logp_error")
        record["mw_error"] = nte_record.get("mw_error")
        record["initial_nte"] = initial_nte_value
        if initial_nte_value is not None and isinstance(nte_record.get("nte"), (int, float)):
            record["nte_relative_improvement"] = float(
                (initial_nte_value - float(nte_record["nte"]))
                / max(initial_nte_value, DELTA_NTE_RELATIVE_EPS)
            )
        else:
            record["nte_relative_improvement"] = None

    all_trajectories = []
    for st in traj_states:
        turns = st["turns"]
        n = len(turns)
        returns = [0.0] * n
        running = 0.0
        for i in reversed(range(n)):
            running = turns[i]["turn_reward"] + gamma * running
            returns[i] = running
        G_k = returns[0] if returns else 0.0
        all_trajectories.append((turns, returns, G_k))

        if debug:
            turn_rewards_dbg = [round(turn["turn_reward"], 4) for turn in turns]
            returns_dbg = [round(value, 4) for value in returns]
            traj_idx = traj_states.index(st)
            final_can_dbg = final_canonical_by_traj[traj_idx]
            print(
                f"[DBG]{debug_prefix} traj={traj_idx} "
                f"turn_rewards={turn_rewards_dbg} returns={returns_dbg} G_k={G_k:.4f} "
                f"final={final_can_dbg} "
                f"rep_count={st.get('final_repetition_count', 0)}"
            )

    G_values = [trajectory[2] for trajectory in all_trajectories]
    G_arr = np.array(G_values, dtype=np.float64)
    G_mean = G_arr.mean() if len(G_arr) else 0.0
    G_std = G_arr.std() if len(G_arr) else 0.0

    if G_std < 1e-8:
        relative_advs = [G - ema_baseline.value for G in G_values]
        for G in G_values:
            ema_baseline.update_and_get(G)
    else:
        relative_advs = ((G_arr - G_mean) / G_std).tolist()
        for G in G_values:
            ema_baseline.update_and_get(G)

    for idx, record in enumerate(terminal_molecule_records):
        record["trajectory_return"] = float(G_values[idx]) if idx < len(G_values) else None
        record["advantage"] = float(relative_advs[idx]) if idx < len(relative_advs) else None
        if idx < len(traj_states):
            record["final_repetition_count"] = int(traj_states[idx].get("final_repetition_count", 0))
            record["final_repetition_penalty"] = float(
                traj_states[idx].get("final_repetition_penalty", 0.0)
            )

    def _component_or_none(components: dict, name: str):
        value = components.get(name)
        return float(value) if isinstance(value, (int, float)) else None

    turn_output_records = []
    for traj_idx, ((turns, returns, G_k), adv) in enumerate(zip(all_trajectories, relative_advs)):
        for turn, turn_return in zip(turns, returns):
            components = turn.get("reward_components", {})
            completion_text = turn.get("completion_text")
            turn_output_records.append(
                {
                    "trajectory_index": int(traj_idx),
                    "turn": int(turn.get("turn_index", -1)) + 1,
                    "valid": bool(turn.get("turn_valid_canonical_smiles")),
                    "input_smiles": turn.get("input_smiles"),
                    "input_canonical_smiles": turn.get("input_canonical_smiles"),
                    "canonical_smiles": turn.get("turn_valid_canonical_smiles"),
                    "raw_smiles": extract_smiles(completion_text),
                    "output_text": completion_text,
                    "turn_reward": float(turn.get("turn_reward", 0.0)),
                    "return": float(turn_return),
                    "trajectory_return": float(G_k),
                    "advantage": float(adv),
                    "base_total_before_bonus": _component_or_none(
                        components, "base_total_before_bonus"
                    ),
                    "reward_total": _component_or_none(components, "total"),
                    "reward_format": _component_or_none(components, "strict"),
                    "reward_property": _component_or_none(components, "property"),
                    "reward_repetition": _component_or_none(components, "repetition"),
                    "milestone_bonus": _component_or_none(components, "milestone_bonus"),
                    "milestone_level": components.get("milestone_level"),
                    "best_milestone_level": components.get("best_milestone_level"),
                    "previous_milestone_level": components.get("previous_milestone_level"),
                    "early_stop_bonus": _component_or_none(components, "early_stop_bonus"),
                    "early_stop_met": bool(components.get("early_stop_met", False)),
                    "early_stop_threshold": _component_or_none(
                        components, "early_stop_threshold"
                    ),
                    "initial_nte": _component_or_none(components, "initial_nte"),
                    "current_nte": _component_or_none(components, "current_nte"),
                    "property_delta_nte": _component_or_none(components, "property_delta_nte_raw"),
                    "property_delta_nte_clipped": _component_or_none(
                        components, "property_delta_nte_clipped"
                    ),
                    "property_absolute_delta_nte": _component_or_none(
                        components, "property_absolute_delta_nte"
                    ),
                    "property_relative_delta_nte": _component_or_none(
                        components, "property_relative_delta_nte_raw"
                    ),
                    "property_relative_delta_nte_clipped": _component_or_none(
                        components, "property_relative_delta_nte_clipped"
                    ),
                    "property_reward_bucket": components.get("property_reward_bucket"),
                    "property_reward_bucket_delta": _component_or_none(
                        components, "property_reward_bucket_delta"
                    ),
                    "property_abs_nte": _component_or_none(components, "property_abs_nte"),
                    "property_abs_nte_quality": _component_or_none(
                        components, "property_abs_nte_quality"
                    ),
                    "total_err": _component_or_none(components, "total_err"),
                    "old_total_err": _component_or_none(components, "old_total_err"),
                }
            )

    datums = []
    kl_masks = []
    traj_returns = []
    dropped_zero_adv = 0
    for idx, ((turns, returns, G_k), adv) in enumerate(zip(all_trajectories, relative_advs)):
        traj_returns.append(float(G_k))
        if abs(adv) < 1e-12:
            dropped_zero_adv += 1
            if debug:
                print(f"[DBG]{debug_prefix} drop traj={idx} because adv is near zero ({adv:.4e})")
            continue
        for turn in turns:
            n_p = len(turn["prompt_tokens"])
            n_c = len(turn["comp_tokens"])
            full = turn["prompt_tokens"] + turn["comp_tokens"]
            min_len = min(
                len(full) - 1,
                n_p - 1 + n_c,
                n_p - 1 + len(turn["comp_logprobs"]),
            )
            kl_mask = ([0.0] * (n_p - 1) + [1.0] * n_c)[:min_len]
            datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=full[:-1]),
                    loss_fn_inputs=dict(
                        target_tokens=full[1:][:min_len],
                        advantages=([0.0] * (n_p - 1) + [adv] * n_c)[:min_len],
                        logprobs=([0.0] * (n_p - 1) + turn["comp_logprobs"])[:min_len],
                    ),
                )
            )
            kl_masks.append(kl_mask)

    adv_arr = np.array(relative_advs, dtype=np.float64) if relative_advs else np.array([], dtype=np.float64)
    turn_components = [
        turn["reward_components"]
        for traj_turns, _, _ in all_trajectories
        for turn in traj_turns
        if "reward_components" in turn
    ]
    turn_records = [
        turn
        for traj_turns, _, _ in all_trajectories
        for turn in traj_turns
    ]
    num_turns_total = len(turn_records)
    num_turn_valid = sum(1 for turn in turn_records if turn.get("turn_valid_canonical_smiles"))
    turn_valid_rate = float(num_turn_valid / num_turns_total) if num_turns_total else None
    turn_total_counts: Counter[int] = Counter()
    turn_valid_counts: Counter[int] = Counter()
    for turn in turn_records:
        turn_index = turn.get("turn_index")
        if not isinstance(turn_index, (int, float)):
            continue
        turn_no = int(turn_index) + 1
        turn_total_counts[turn_no] += 1
        if turn.get("turn_valid_canonical_smiles"):
            turn_valid_counts[turn_no] += 1

    def _component_values(name: str) -> list[float]:
        return [
            float(component[name])
            for component in turn_components
            if isinstance(component.get(name), (int, float))
        ]

    def _mean_component(name: str) -> float:
        values = _component_values(name)
        return float(np.mean(values)) if values else 0.0

    def _min_component(name: str):
        values = _component_values(name)
        return min(values) if values else None

    def _max_component(name: str):
        values = _component_values(name)
        return max(values) if values else None

    reward_bucket_counts = Counter(
        component.get("property_reward_bucket")
        for component in turn_components
        if component.get("property_reward_bucket") in {"low", "mid", "high"}
    )
    valid_nte_records = [record for record in terminal_nte_records if record.get("valid")]

    def _values(name: str) -> list[float]:
        return [float(record[name]) for record in valid_nte_records if record.get(name) is not None]

    nte_values = _values("nte")
    nte_relative_improvements = [
        float(record["nte_relative_improvement"])
        for record in terminal_molecule_records
        if isinstance(record.get("nte_relative_improvement"), (int, float))
    ]
    final_qed_errors = _values("qed_error")
    final_logp_errors = _values("logp_error")
    final_mw_errors = _values("mw_error")
    nte_bucket_stats = _initial_nte_bucket_stats(terminal_molecule_records)
    terminal_total = len(terminal_nte_records)
    terminal_valid_rate = (
        float(len(valid_nte_records) / terminal_total) if terminal_total else None
    )

    def _mean_or_none(values: list[float]):
        return float(np.mean(values)) if values else None

    def _std_or_none(values: list[float]):
        return float(np.std(values)) if values else None

    details = {
        "mean_G": float(G_mean),
        "std_G": float(G_std),
        "mean_adv": float(np.mean(adv_arr)) if adv_arr.size else 0.0,
        "mean_abs_adv": float(np.mean(np.abs(adv_arr))) if adv_arr.size else 0.0,
        "std_adv": float(np.std(adv_arr)) if adv_arr.size else 0.0,
        "pos_adv_ratio": float(np.mean(adv_arr > 0.0)) if adv_arr.size else 0.0,
        "num_traj": len(all_trajectories),
        "num_turns_total": num_turns_total,
        "num_turn_total": num_turns_total,
        "num_turn_valid": int(num_turn_valid),
        "turn_valid_rate": turn_valid_rate,
        "dropped_zero_adv": dropped_zero_adv,
        "mean_reward_total": _mean_component("total"),
        "mean_reward_format": _mean_component("strict"),
        "mean_reward_property": _mean_component("property"),
        "min_reward_property": _min_component("property"),
        "max_reward_property": _max_component("property"),
        "mean_property_delta_nte": _mean_component("property_delta_nte_raw"),
        "mean_property_delta_nte_clipped": _mean_component("property_delta_nte_clipped"),
        "mean_property_relative_delta_nte": _mean_component("property_relative_delta_nte_raw"),
        "mean_property_relative_delta_nte_clipped": _mean_component(
            "property_relative_delta_nte_clipped"
        ),
        "mean_property_absolute_delta_nte": _mean_component("property_absolute_delta_nte"),
        "mean_property_abs_nte": _mean_component("property_abs_nte"),
        "mean_property_abs_nte_quality": _mean_component("property_abs_nte_quality"),
        "property_reward_bucket": _bucket_label_for_initial_nte(initial_nte) or "unknown",
        "property_reward_bucket_low_count": int(reward_bucket_counts.get("low", 0)),
        "property_reward_bucket_mid_count": int(reward_bucket_counts.get("mid", 0)),
        "property_reward_bucket_high_count": int(reward_bucket_counts.get("high", 0)),
        "mean_reward_milestone_bonus": _mean_component("milestone_bonus"),
        "mean_reward_early_stop_bonus": _mean_component("early_stop_bonus"),
        "num_early_stop_met": int(
            sum(1 for component in turn_components if component.get("early_stop_met"))
        ),
        "initial_nte": initial_nte,
        "early_stop_threshold": early_stop_threshold_value,
        "property_delta_weight": float(DELTA_NTE_WEIGHT),
        "property_abs_weight": float(ABS_NTE_WEIGHT),
        "mean_reward_repetition": _mean_component("repetition"),
        "mean_repetition_count": _mean_component("repetition_count"),
        "num_final_valid": len(valid_nte_records),
        "num_final_missing": len(terminal_nte_records) - len(valid_nte_records),
        "num_terminal_total": terminal_total,
        "terminal_valid_rate": terminal_valid_rate,
        "mean_nte": _mean_or_none(nte_values),
        "std_nte": _std_or_none(nte_values),
        "mean_nte_relative_improvement": _mean_or_none(nte_relative_improvements),
        "std_nte_relative_improvement": _std_or_none(nte_relative_improvements),
        **nte_bucket_stats,
        "mean_final_qed_error": _mean_or_none(final_qed_errors),
        "mean_final_logp_error": _mean_or_none(final_logp_errors),
        "mean_final_mw_error": _mean_or_none(final_mw_errors),
        "nte_values": nte_values,
        "nte_relative_improvements": nte_relative_improvements,
        "final_qed_errors": final_qed_errors,
        "final_logp_errors": final_logp_errors,
          "final_mw_errors": final_mw_errors,
          "final_canonical_smiles": final_canonicals,
          "final_canonical_smiles_per_trajectory": final_canonical_by_traj,
          "terminal_canonical_smiles_per_trajectory": terminal_state_canonical_by_traj,
          "terminal_molecule_records": terminal_molecule_records,
          "turn_output_records": turn_output_records,
      }
    for turn_no in range(1, int(max_turns) + 1):
        turn_total = int(turn_total_counts.get(turn_no, 0))
        turn_valid = int(turn_valid_counts.get(turn_no, 0))
        details[f"turn{turn_no}_total_count"] = turn_total
        details[f"turn{turn_no}_valid_count"] = turn_valid
        details[f"turn{turn_no}_valid_rate"] = (
            float(turn_valid / turn_total) if turn_total else None
        )
    return datums, kl_masks, traj_returns, details


def _run_async_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "KL penalty logprob computation was called from an active asyncio loop. "
        "Run this training loop from a synchronous context or make the caller async."
    )


async def _compute_reference_logprobs(kl_reference_client, sequence_inputs):
    return await asyncio.gather(
        *[
            kl_reference_client.compute_logprobs_async(sequence_input)
            for sequence_input in sequence_inputs
        ]
    )


def _empty_kl_metrics(kl_penalty_coef: float) -> dict:
    return {
        "kl_penalty_coef": float(kl_penalty_coef),
        "kl_policy_reference": None,
        "explicit_forward_kl": None,
        "kl_tokens": 0,
        "ppo_loss": None,
        "ppo_kl_total_loss": None,
        "ppo_clip_fraction": None,
        "importance_ratio_mean": None,
    }


def _validate_loss_sequence_lengths(
    *,
    sequence_index: int,
    new_logprobs,
    old_logprobs,
    reference_logprobs,
    advantages,
    action_mask,
) -> int:
    lengths = {
        "new_logprobs": len(new_logprobs),
        "old_logprobs": len(old_logprobs),
        "reference_logprobs": len(reference_logprobs),
        "advantages": len(advantages),
        "action_mask": len(action_mask),
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            f"Explicit PPO + KL sequence {sequence_index} length mismatch: {lengths}"
        )
    sequence_len = next(iter(unique_lengths))
    if sequence_len <= 0:
        raise ValueError(
            f"Explicit PPO + KL sequence {sequence_index} has no target tokens"
        )
    return int(sequence_len)


def compute_ppo_forward_kl_loss(
    new_logprobs_list,
    *,
    old_logprobs_list,
    reference_logprobs_list,
    advantages_list,
    action_masks,
    kl_penalty_coef: float,
    clip_low_threshold: float = 0.9,
    clip_high_threshold: float = 1.1,
):
    if kl_penalty_coef < 0:
        raise ValueError("kl_penalty_coef must be >= 0")
    if clip_low_threshold <= 0 or clip_low_threshold >= clip_high_threshold:
        raise ValueError(
            "PPO clip thresholds must satisfy 0 < clip_low_threshold < "
            "clip_high_threshold"
        )

    batch_lengths = {
        len(new_logprobs_list),
        len(old_logprobs_list),
        len(reference_logprobs_list),
        len(advantages_list),
        len(action_masks),
    }
    if len(batch_lengths) != 1:
        raise ValueError(
            "Explicit PPO + KL batch list lengths must match: "
            f"new={len(new_logprobs_list)}, old={len(old_logprobs_list)}, "
            f"reference={len(reference_logprobs_list)}, "
            f"advantages={len(advantages_list)}, masks={len(action_masks)}"
        )

    ppo_terms = []
    forward_kl_terms = []
    importance_ratios = []
    clip_indicators = []

    for sequence_index, (
        new_logprobs,
        old_logprobs,
        reference_logprobs,
        advantages,
        action_mask,
    ) in enumerate(
        zip(
            new_logprobs_list,
            old_logprobs_list,
            reference_logprobs_list,
            advantages_list,
            action_masks,
            strict=True,
        )
    ):
        _validate_loss_sequence_lengths(
            sequence_index=sequence_index,
            new_logprobs=new_logprobs,
            old_logprobs=old_logprobs,
            reference_logprobs=reference_logprobs,
            advantages=advantages,
            action_mask=action_mask,
        )
        device = new_logprobs.device
        dtype = new_logprobs.dtype
        old = torch.as_tensor(old_logprobs, dtype=dtype, device=device)
        reference = torch.as_tensor(reference_logprobs, dtype=dtype, device=device)
        advantages_tensor = torch.as_tensor(advantages, dtype=dtype, device=device)
        mask_values = torch.as_tensor(action_mask, dtype=dtype, device=device)
        if not torch.all((mask_values == 0) | (mask_values == 1)):
            raise ValueError(
                f"Explicit PPO + KL sequence {sequence_index} action mask must be binary"
            )
        mask = mask_values.to(dtype=torch.bool)
        if not torch.any(mask):
            continue

        new_action = new_logprobs[mask]
        old_action = old[mask]
        reference_action = reference[mask]
        action_advantages = advantages_tensor[mask]

        importance_ratio = torch.exp(new_action - old_action)
        unclipped = importance_ratio * action_advantages
        clipped_ratio = torch.clamp(
            importance_ratio,
            min=float(clip_low_threshold),
            max=float(clip_high_threshold),
        )
        clipped = clipped_ratio * action_advantages
        ppo_terms.append(torch.minimum(unclipped, clipped))

        policy_reference_logratio = new_action - reference_action
        nonnegative_kl = (
            torch.expm1(-policy_reference_logratio) + policy_reference_logratio
        )
        forward_kl_terms.append(
            importance_ratio * torch.clamp(nonnegative_kl, min=0.0)
        )
        importance_ratios.append(importance_ratio)
        clip_indicators.append(
            (importance_ratio < clip_low_threshold)
            | (importance_ratio > clip_high_threshold)
        )

    if not ppo_terms:
        raise ValueError("No completion tokens available for explicit PPO + KL loss")

    flat_ppo_terms = torch.cat(ppo_terms)
    flat_forward_kl = torch.cat(forward_kl_terms)
    flat_importance_ratios = torch.cat(importance_ratios)
    flat_clip_indicators = torch.cat(clip_indicators)

    ppo_loss = -flat_ppo_terms.mean()
    forward_kl = flat_forward_kl.mean()
    total_loss = ppo_loss + float(kl_penalty_coef) * forward_kl
    if not torch.isfinite(total_loss):
        raise ValueError("Explicit PPO + KL produced a non-finite total loss")
    metrics = {
        "ppo_loss": float(ppo_loss.detach().item()),
        "explicit_forward_kl": float(forward_kl.detach().item()),
        "ppo_kl_total_loss": float(total_loss.detach().item()),
        "ppo_clip_fraction": float(
            flat_clip_indicators.float().mean().detach().item()
        ),
        "importance_ratio_mean": float(
            flat_importance_ratios.mean().detach().item()
        ),
    }
    return total_loss, metrics


def _prepare_explicit_ppo_kl(
    datums: list[types.Datum],
    kl_masks: list[list[float]],
    kl_reference_client,
    kl_penalty_coef: float,
) -> tuple[list[types.Datum], object, int]:
    if kl_penalty_coef <= 0:
        raise ValueError("kl_penalty_coef must be > 0 for explicit PPO + KL")
    if kl_reference_client is None:
        raise ValueError("kl_reference_client is required for explicit PPO + KL")
    if not datums:
        raise ValueError("Explicit PPO + KL requires at least one datum")
    if len(datums) != len(kl_masks):
        raise ValueError(
            "Explicit PPO + KL datum/mask count mismatch: "
            f"datums={len(datums)}, masks={len(kl_masks)}"
        )

    sequence_inputs = []
    old_logprobs = []
    advantages = []
    for datum_index, (datum, kl_mask) in enumerate(
        zip(datums, kl_masks, strict=True)
    ):
        target_tokens = list(datum.loss_fn_inputs["target_tokens"].data)
        if not target_tokens:
            raise ValueError(
                f"Explicit PPO + KL datum {datum_index} has no target tokens"
            )
        sequence_inputs.append(datum.model_input.append_int(int(target_tokens[-1])))
        datum_old_logprobs = list(datum.loss_fn_inputs["logprobs"].data)
        datum_advantages = list(datum.loss_fn_inputs["advantages"].data)
        lengths = {
            "target_tokens": len(target_tokens),
            "old_logprobs": len(datum_old_logprobs),
            "advantages": len(datum_advantages),
            "action_mask": len(kl_mask),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Explicit PPO + KL datum {datum_index} length mismatch: {lengths}"
            )
        if any(value not in (0, 0.0, 1, 1.0) for value in kl_mask):
            raise ValueError(
                f"Explicit PPO + KL datum {datum_index} action mask must be binary"
            )
        old_logprobs.append(datum_old_logprobs)
        advantages.append(datum_advantages)

    reference_logprobs = _run_async_sync(
        _compute_reference_logprobs(kl_reference_client, sequence_inputs)
    )
    if len(reference_logprobs) != len(datums):
        raise ValueError(
            "Explicit PPO + KL reference response count mismatch: "
            f"expected={len(datums)}, received={len(reference_logprobs)}"
        )

    reference_action_logprobs = []
    for datum_index, (datum, ref_logprobs) in enumerate(
        zip(datums, reference_logprobs, strict=True)
    ):
        target_len = len(datum.loss_fn_inputs["target_tokens"].data)
        ref_action_logprobs = list(ref_logprobs[1:])
        if len(ref_action_logprobs) != target_len:
            raise ValueError(
                f"Explicit PPO + KL datum {datum_index} reference logprob length "
                f"mismatch: expected={target_len}, received={len(ref_action_logprobs)}"
            )
        reference_action_logprobs.append(ref_action_logprobs)

    custom_datums = [
        types.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": datum.loss_fn_inputs["target_tokens"],
                "weights": types.TensorData.from_numpy(
                    np.asarray(mask, dtype=np.float32)
                ),
            },
        )
        for datum, mask in zip(datums, kl_masks, strict=True)
    ]
    kl_tokens = int(
        sum(sum(1 for value in mask if value > 0) for mask in kl_masks)
    )
    if kl_tokens <= 0:
        raise ValueError("Explicit PPO + KL found no completion tokens in action masks")

    def explicit_ppo_kl_loss(_data, new_logprobs_list):
        return compute_ppo_forward_kl_loss(
            new_logprobs_list,
            old_logprobs_list=old_logprobs,
            reference_logprobs_list=reference_action_logprobs,
            advantages_list=advantages,
            action_masks=kl_masks,
            kl_penalty_coef=kl_penalty_coef,
        )

    return custom_datums, explicit_ppo_kl_loss, kl_tokens


def _datums_for_training(datums: list[types.Datum]) -> list[types.Datum]:
    return [
        types.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                key: value
                for key, value in datum.loss_fn_inputs.items()
                if key != "mask"
            },
        )
        for datum in datums
    ]


def run_multistep_rl_training_fixed(
    raw_data,
    renderer,
    training_client,
    tokenizer,
    num_iterations: int = 50,
    prompts_per_iter: int = 4,
    candidate_prompts_per_iter: int = 6,
    max_kept_prompts_per_iter: int = 4,
    prompt_initial_nte_min: float = 0.5,
    learning_rate: float = 3e-6,
    trajectories_per_prompt: int = 8,
    max_turns: int = 3,
    gamma: float = 0.85,
    ema_alpha: float = 0.05,
    rng_seed: int = 42,
    debug_adv: bool = False,
    debug_prompts_per_iter: int = 0,
    debug_iterations: int = 0,
    keep_skipped_metrics: bool = True,
    refresh_sampling_every: int = 6,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    kl_reference_client=None,
    kl_penalty_coef: float = 0.03,
):
    if refresh_sampling_every < 1:
        raise ValueError("refresh_sampling_every must be >= 1")
    if candidate_prompts_per_iter < 1:
        raise ValueError("candidate_prompts_per_iter must be >= 1")
    if max_kept_prompts_per_iter < 1:
        raise ValueError("max_kept_prompts_per_iter must be >= 1")
    if kl_penalty_coef < 0:
        raise ValueError("kl_penalty_coef must be >= 0")
    if kl_penalty_coef > 0 and kl_reference_client is None:
        raise ValueError("kl_reference_client is required when kl_penalty_coef > 0")

    rng = np.random.default_rng(rng_seed)
    ema = EMABaseline(alpha=ema_alpha)
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="single-stage-init"
    )

    all_metrics = []
    print("Starting single-stage multi-turn GRPO training")
    print("reward = format_r + bucketed_relative_delta_abs_prop_r + prompt-local repetition_penalty")
    print(
        f"sampling temperature={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}, "
        f"trajectories_per_prompt={trajectories_per_prompt}, refresh_sampling_every={refresh_sampling_every}"
    )
    print(
        "prompt filter: "
        f"candidates={candidate_prompts_per_iter}, max_kept={max_kept_prompts_per_iter}, "
        f"initial_nte_min={prompt_initial_nte_min:.3f}, legacy_prompts_per_iter={prompts_per_iter}"
    )
    if kl_penalty_coef > 0:
        print(
            "Explicit PPO + forward-KL loss enabled: "
            f"beta={kl_penalty_coef:.4f}, PPO clip=[0.9, 1.1]"
        )
    header = f"{'Iter':>4} | {'Mean G':>8} | {'std_G':>7} | {'|adv|':>7} | {'std_adv':>7} | {'+adv%':>6} | {'#data':>6} | {'t(s)':>6}"
    print(header)
    print("-" * len(header))

    def _flatten_numeric(details_list, key: str) -> list[float]:
        values = []
        for details in details_list:
            for value in details.get(key, []) or []:
                if isinstance(value, (int, float)):
                    values.append(float(value))
        return values

    def _mean_or_none(values: list[float]):
        return float(np.mean(values)) if values else None

    def _std_or_none(values: list[float]):
        return float(np.std(values)) if values else None

    def _fmt_optional(value, digits: int = 4) -> str:
        return "nan" if value is None else f"{value:.{digits}f}"

    def _terminal_records_from_details(details_list: list[dict]) -> list[dict]:
        records = []
        for details in details_list:
            records.extend(details.get("terminal_molecule_records", []) or [])
        return records

    def _fmt_initial_nte_bucket_stats(stats: dict[str, float | int | None]) -> str:
        parts = []
        for label, display, _low, _high in INITIAL_NTE_BUCKETS:
            prefix = f"initial_nte_{label}"
            count = int(stats.get(f"{prefix}_count") or 0)
            valid_count = int(stats.get(f"{prefix}_valid_count") or 0)
            parts.append(
                f"{display}: n={valid_count}/{count} "
                f"init={_fmt_optional(stats.get(f'{prefix}_mean'), digits=3)} "
                f"term={_fmt_optional(stats.get(f'{prefix}_terminal_nte_mean'), digits=3)} "
                f"rel={_fmt_optional(stats.get(f'{prefix}_relative_improvement_mean'), digits=3)}"
            )
        return " | ".join(parts)

    def _property_reward_range(details_list: list[dict]):
        mins = [
            float(details["min_reward_property"])
            for details in details_list
            if isinstance(details.get("min_reward_property"), (int, float))
        ]
        maxs = [
            float(details["max_reward_property"])
            for details in details_list
            if isinstance(details.get("max_reward_property"), (int, float))
        ]
        return (min(mins) if mins else None, max(maxs) if maxs else None)

    def _mean_numeric_detail(details_list: list[dict], key: str):
        values = [
            float(details[key])
            for details in details_list
            if isinstance(details.get(key), (int, float))
        ]
        return float(np.mean(values)) if values else None

    def _print_terminal_molecule_monitor(iteration: int, details_list: list[dict]) -> None:
        print(f"      terminal molecules | iteration={iteration}")
        any_records = False
        for prompt_batch_idx, details in enumerate(details_list):
            prompt_index = details.get("prompt_index", "unknown")
            records = details.get("terminal_molecule_records", []) or []
            for record in records:
                any_records = True
                source = record.get("source", "missing")
                canonical = record.get("canonical_smiles") or "None"
                last_turn = record.get("last_turn")
                last_turn_text = "none" if last_turn is None else str(last_turn)
                traj_idx = record.get("trajectory_index")
                terminal_output = record.get("terminal_output_text") or "None"
                last_model_output = record.get("last_model_output_text") or "None"
                if source == "last_turn_valid":
                    status = "last_valid"
                elif source == "rollback_generated":
                    status = "rollback_to_recent_valid"
                elif source == "rollback_initial":
                    status = "rollback_to_initial"
                else:
                    status = "no_valid_molecule"
                print(
                    f"        prompt_batch={prompt_batch_idx} traj={traj_idx} "
                    f"prompt_idx={prompt_index} last_turn={last_turn_text} "
                    f"status={status} smiles={canonical}"
                )
                print(f"          output={terminal_output}")
                if last_model_output != terminal_output:
                    print(f"          last_model_output={last_model_output}")
        if not any_records:
            print("        no trajectory records")

    def _terminal_records_by_traj(records: list[dict]) -> dict[int, dict]:
        out = {}
        for record in records:
            traj_idx = record.get("trajectory_index")
            if isinstance(traj_idx, int):
                out[traj_idx] = record
        return out

    def _turn_records_by_traj(records: list[dict]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        for record in records:
            traj_idx = record.get("trajectory_index")
            if isinstance(traj_idx, int):
                out.setdefault(traj_idx, []).append(record)
        for turns in out.values():
            turns.sort(key=lambda item: int(item.get("turn") or 0))
        return out

    def _print_iteration_trace_example(iteration: int, details_list: list[dict]) -> None:
        if not details_list:
            print(f"      trace all | iteration={iteration} no prompt details")
            return

        def _terminal_nte_values(details: dict) -> list[float]:
            return [
                float(value)
                for value in details.get("nte_values", []) or []
                if isinstance(value, (int, float))
            ]

        print(
            f"      trace all | iteration={iteration} prompts={len(details_list)} "
            f"(each prompt prints all terminal trajectories and turns)"
        )
        ordered_details = sorted(
            details_list,
            key=lambda item: int(item.get("prompt_batch_index", 0) or 0),
        )
        for prompt_no, details in enumerate(ordered_details, start=1):
            selected_nte_values = _terminal_nte_values(details)
            selected_nte_mean = _mean_or_none(selected_nte_values)
            selected_nte_std = _std_or_none(selected_nte_values)
            prompt_batch = details.get("prompt_batch_index", 0)
            prompt_index = details.get("prompt_index", "unknown")
            prompt_text = details.get("prompt_text") or "None"
            ground_truth = details.get("ground_truth")
            terminal_records = details.get("terminal_molecule_records", []) or []
            turn_records = details.get("turn_output_records", []) or []
            terminals_by_traj = _terminal_records_by_traj(terminal_records)
            turns_by_traj = _turn_records_by_traj(turn_records)
            traj_ids = sorted(set(terminals_by_traj) | set(turns_by_traj))

            print(
                f"        prompt {prompt_no}/{len(ordered_details)} | "
                f"prompt_batch={prompt_batch} prompt_idx={prompt_index} | "
                f"terminal_nte_mean={_fmt_optional(selected_nte_mean)} "
                f"std={_fmt_optional(selected_nte_std)} n={len(selected_nte_values)} "
                f"missing={details.get('num_final_missing', 0)}"
            )
            print("          question:")
            for line in str(prompt_text).splitlines() or ["None"]:
                print(f"            {line}")
            if ground_truth:
                print("          ground_truth:")
                print(f"            {ground_truth}")

            if not traj_ids:
                print("          no trajectory turns recorded")
                continue

            for traj_idx in traj_ids:
                terminal = terminals_by_traj.get(traj_idx, {})
                print(
                    f"          traj={traj_idx} terminal | "
                    f"status={terminal.get('source', 'unknown')} | "
                    f"smiles={terminal.get('canonical_smiles') or 'None'} | "
                    f"nte={_fmt_optional(terminal.get('nte'))} | "
                    f"G={_fmt_optional(terminal.get('trajectory_return'))} | "
                    f"adv={_fmt_optional(terminal.get('advantage'))}"
                )
                terminal_output = terminal.get("terminal_output_text")
                if terminal_output:
                    print(f"            terminal_output={terminal_output}")
                last_model_output = terminal.get("last_model_output_text")
                if last_model_output and last_model_output != terminal_output:
                    print(f"            last_model_output={last_model_output}")

                for turn in turns_by_traj.get(traj_idx, []):
                    print(
                        f"            turn={turn.get('turn')} valid={turn.get('valid')} | "
                        f"reward={_fmt_optional(turn.get('turn_reward'))} | "
                        f"return={_fmt_optional(turn.get('return'))} | "
                        f"format={_fmt_optional(turn.get('reward_format'), digits=3)} | "
                        f"property={_fmt_optional(turn.get('reward_property'), digits=3)} | "
                        f"rep={_fmt_optional(turn.get('reward_repetition'), digits=3)} | "
                        f"milestone_bonus={_fmt_optional(turn.get('milestone_bonus'), digits=3)} | "
                        f"early_bonus={_fmt_optional(turn.get('early_stop_bonus'), digits=3)}"
                    )
                    print(f"              output={turn.get('output_text') or 'None'}")
                    print(
                        f"              input base raw={turn.get('input_smiles') or 'None'} | "
                        f"canonical={turn.get('input_canonical_smiles') or 'None'}"
                    )
                    print(
                        f"              smiles raw={turn.get('raw_smiles') or 'None'} | "
                        f"canonical={turn.get('canonical_smiles') or 'None'}"
                    )
                    print(
                        f"              nte old={_fmt_optional(turn.get('old_total_err'))} | "
                        f"new={_fmt_optional(turn.get('total_err'))} | "
                        f"bucket={turn.get('property_reward_bucket') or 'unknown'} | "
                        f"rel_delta={_fmt_optional(turn.get('property_relative_delta_nte'))} | "
                        f"rel_clip={_fmt_optional(turn.get('property_relative_delta_nte_clipped'))} | "
                        f"bucket_delta={_fmt_optional(turn.get('property_reward_bucket_delta'))} | "
                        f"abs_delta={_fmt_optional(turn.get('property_absolute_delta_nte'))} | "
                        f"abs_q={_fmt_optional(turn.get('property_abs_nte_quality'))}"
                    )
                    print(
                        f"              bonus initial_nte={_fmt_optional(turn.get('initial_nte'))} | "
                        f"current_nte={_fmt_optional(turn.get('current_nte'))} | "
                        f"early_threshold={_fmt_optional(turn.get('early_stop_threshold'))} | "
                        f"early_met={turn.get('early_stop_met')} | "
                        f"milestone_level={turn.get('milestone_level')} | "
                        f"best_level={turn.get('best_milestone_level')}"
                    )

    for iteration in range(num_iterations):
        t0 = time.time()

        prompt_indices, prompt_filter_stats = _sample_prompt_indices_with_initial_nte_filter(
            raw_data,
            rng,
            candidate_prompts_per_iter=candidate_prompts_per_iter,
            max_kept_prompts_per_iter=max_kept_prompts_per_iter,
            prompt_initial_nte_min=prompt_initial_nte_min,
        )
        print(
            "      prompt filter | "
            f"candidates={prompt_filter_stats['prompt_filter_candidates']} "
            f"kept={prompt_filter_stats['prompt_filter_kept']} "
            f"skipped_low_nte={prompt_filter_stats['prompt_filter_skipped_low_nte']} "
            f"skipped_unparseable={prompt_filter_stats['prompt_filter_skipped_unparseable']}"
        )
        if not prompt_indices:
            print(
                "      prompt filter | no eligible prompt "
                f"(initial_nte_min={prompt_initial_nte_min:.3f})"
            )
        all_datums = []
        all_kl_masks = []
        all_rewards = []
        all_details = []

        for local_idx, prompt_idx in enumerate(prompt_indices):
            example = raw_data[int(prompt_idx)]
            try:
                do_debug = (
                    debug_adv
                    and iteration < debug_iterations
                    and local_idx < debug_prompts_per_iter
                )
                datums, kl_masks, rewards, details = run_mode_b(
                    example,
                    renderer,
                    sampling_client,
                    tokenizer,
                    ema_baseline=ema,
                    num_trajectories=trajectories_per_prompt,
                    max_turns=max_turns,
                    gamma=gamma,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    debug=do_debug,
                    debug_prefix=f" it={iteration} p={int(prompt_idx)} ",
                )
            except Exception as exc:
                print(f"  [skip] prompt {prompt_idx}: {exc}")
                continue

            details["prompt_index"] = int(prompt_idx)
            details["prompt_batch_index"] = int(local_idx)
            details["prompt_text"] = example.get(PROMPT_COL)
            details["ground_truth"] = example.get(ANSWER_COL)
            all_datums.extend(datums)
            all_kl_masks.extend(kl_masks)
            all_rewards.extend(rewards)
            all_details.append(details)

        elapsed = time.time() - t0

        nte_values = _flatten_numeric(all_details, "nte_values")
        nte_relative_improvements = _flatten_numeric(all_details, "nte_relative_improvements")
        final_qed_errors = _flatten_numeric(all_details, "final_qed_errors")
        final_logp_errors = _flatten_numeric(all_details, "final_logp_errors")
        final_mw_errors = _flatten_numeric(all_details, "final_mw_errors")
        num_final_valid = int(np.sum([details.get("num_final_valid", 0) for details in all_details]))
        num_final_missing = int(np.sum([details.get("num_final_missing", 0) for details in all_details]))
        num_terminal_total = num_final_valid + num_final_missing
        terminal_valid_rate = (
            float(num_final_valid / num_terminal_total) if num_terminal_total else None
        )
        num_turn_valid = int(np.sum([details.get("num_turn_valid", 0) for details in all_details]))
        num_turn_total = int(np.sum([details.get("num_turn_total", 0) for details in all_details]))
        turn_valid_rate = float(num_turn_valid / num_turn_total) if num_turn_total else None
        turn_metrics: dict[str, float | int | None] = {}
        for turn_no in range(1, int(max_turns) + 1):
            turn_total = int(np.sum([details.get(f"turn{turn_no}_total_count", 0) for details in all_details]))
            turn_valid = int(np.sum([details.get(f"turn{turn_no}_valid_count", 0) for details in all_details]))
            turn_metrics[f"turn{turn_no}_total_count"] = turn_total
            turn_metrics[f"turn{turn_no}_valid_count"] = turn_valid
            turn_metrics[f"turn{turn_no}_valid_rate"] = (
                float(turn_valid / turn_total) if turn_total else None
            )
        prompt_trace_records = [
            {
                "prompt_batch_index": details.get("prompt_batch_index"),
                "prompt_index": details.get("prompt_index"),
                "prompt_text": details.get("prompt_text"),
                "ground_truth": details.get("ground_truth"),
                "terminal_molecule_records": details.get("terminal_molecule_records", []) or [],
                "turn_output_records": details.get("turn_output_records", []) or [],
            }
            for details in all_details
        ]
        nte_bucket_stats = _initial_nte_bucket_stats(
            _terminal_records_from_details(all_details)
        )
        property_reward_bucket_low_count = int(
            np.sum([details.get("property_reward_bucket_low_count", 0) for details in all_details])
        )
        property_reward_bucket_mid_count = int(
            np.sum([details.get("property_reward_bucket_mid_count", 0) for details in all_details])
        )
        property_reward_bucket_high_count = int(
            np.sum([details.get("property_reward_bucket_high_count", 0) for details in all_details])
        )
        _print_iteration_trace_example(iteration, all_details)
        mean_nte = _mean_or_none(nte_values)
        std_nte = _std_or_none(nte_values)
        mean_nte_relative_improvement = _mean_or_none(nte_relative_improvements)
        std_nte_relative_improvement = _std_or_none(nte_relative_improvements)
        mean_final_qed_error = _mean_or_none(final_qed_errors)
        mean_final_logp_error = _mean_or_none(final_logp_errors)
        mean_final_mw_error = _mean_or_none(final_mw_errors)
        kl_metrics = _empty_kl_metrics(kl_penalty_coef)

        if not all_datums:
            if all_details:
                mean_G = float(np.mean([details.get("mean_G", 0.0) for details in all_details]))
                std_G = float(np.mean([details.get("std_G", 0.0) for details in all_details]))
                mean_adv = float(np.mean([details.get("mean_adv", 0.0) for details in all_details]))
                mean_abs_adv = float(np.mean([details.get("mean_abs_adv", 0.0) for details in all_details]))
                std_adv = float(np.mean([details.get("std_adv", 0.0) for details in all_details]))
                pos_adv_ratio = float(np.mean([details.get("pos_adv_ratio", 0.0) for details in all_details]))
                dropped = int(np.sum([details.get("dropped_zero_adv", 0) for details in all_details]))
                mean_reward_total = float(np.mean([details.get("mean_reward_total", 0.0) for details in all_details]))
                mean_reward_format = float(np.mean([details.get("mean_reward_format", 0.0) for details in all_details]))
                mean_reward_property = float(np.mean([details.get("mean_reward_property", 0.0) for details in all_details]))
                min_reward_property, max_reward_property = _property_reward_range(all_details)
                mean_property_delta_nte = _mean_numeric_detail(all_details, "mean_property_delta_nte")
                mean_property_delta_nte_clipped = _mean_numeric_detail(all_details, "mean_property_delta_nte_clipped")
                mean_property_relative_delta_nte = _mean_numeric_detail(all_details, "mean_property_relative_delta_nte")
                mean_property_relative_delta_nte_clipped = _mean_numeric_detail(all_details, "mean_property_relative_delta_nte_clipped")
                mean_property_absolute_delta_nte = _mean_numeric_detail(all_details, "mean_property_absolute_delta_nte")
                mean_property_abs_nte = _mean_numeric_detail(all_details, "mean_property_abs_nte")
                mean_property_abs_nte_quality = _mean_numeric_detail(all_details, "mean_property_abs_nte_quality")
                mean_reward_milestone_bonus = float(np.mean([details.get("mean_reward_milestone_bonus", 0.0) for details in all_details]))
                mean_reward_early_stop_bonus = float(np.mean([details.get("mean_reward_early_stop_bonus", 0.0) for details in all_details]))
                num_early_stop_met = int(np.sum([details.get("num_early_stop_met", 0) for details in all_details]))
                mean_reward_repetition = float(np.mean([details.get("mean_reward_repetition", 0.0) for details in all_details]))

                print(
                    f"{iteration:>4} | no valid data, skipped "
                    f"(mean_G={mean_G:.4f}, std_G={std_G:.4f}, |adv|={mean_abs_adv:.4f}, "
                    f"std_adv={std_adv:.4f}, +adv%={100.0 * pos_adv_ratio:.1f}, dropped={dropped})"
                )
                print(
                    "      rewards | "
                    f"total={mean_reward_total:+.3f} | "
                    f"format={mean_reward_format:+.3f} | "
                    f"property={mean_reward_property:+.3f} "
                    f"[{_fmt_optional(min_reward_property, digits=3)}, {_fmt_optional(max_reward_property, digits=3)}] | "
                    f"repetition={mean_reward_repetition:+.3f}"
                )
                print(
                    "      turn property parts | "
                    f"buckets low/mid/high={property_reward_bucket_low_count}/"
                    f"{property_reward_bucket_mid_count}/{property_reward_bucket_high_count} | "
                    f"rel_delta={_fmt_optional(mean_property_relative_delta_nte, digits=3)} | "
                    f"rel_clip={_fmt_optional(mean_property_relative_delta_nte_clipped, digits=3)} | "
                    f"abs_delta={_fmt_optional(mean_property_absolute_delta_nte, digits=3)} | "
                    f"abs_nte={_fmt_optional(mean_property_abs_nte, digits=3)} | "
                    f"abs_q={_fmt_optional(mean_property_abs_nte_quality, digits=3)}"
                )
                print(
                    "      bonuses | "
                    f"milestone={mean_reward_milestone_bonus:+.3f} | "
                    f"early_stop={mean_reward_early_stop_bonus:+.3f} | "
                    f"early_stop_count={num_early_stop_met}"
                )
                print(
                    "      terminal nte | "
                    f"mean={_fmt_optional(mean_nte)} std={_fmt_optional(std_nte)} "
                    f"valid={num_final_valid} missing={num_final_missing} | "
                    f"|dQED|={_fmt_optional(mean_final_qed_error)} | "
                    f"|dLogP|={_fmt_optional(mean_final_logp_error)} | "
                    f"|dMW|={_fmt_optional(mean_final_mw_error, digits=2)}"
                )
                print(
                    "      nte relative improvement | "
                    f"mean={_fmt_optional(mean_nte_relative_improvement)} "
                    f"std={_fmt_optional(std_nte_relative_improvement)}"
                )
                print(
                    "      initial nte buckets | "
                    f"{_fmt_initial_nte_bucket_stats(nte_bucket_stats)}"
                )
                print(
                    "      validity | "
                    f"terminal={num_final_valid}/{num_terminal_total} "
                    f"({100.0 * terminal_valid_rate if terminal_valid_rate is not None else float('nan'):.1f}%) | "
                    f"turns={num_turn_valid}/{num_turn_total} "
                    f"({100.0 * turn_valid_rate if turn_valid_rate is not None else float('nan'):.1f}%)"
                )

                if keep_skipped_metrics:
                    all_metrics.append(
                        {
                            "iteration": iteration,
                            "mean_G": mean_G,
                            "std_G": std_G,
                            "mean_adv": mean_adv,
                            "mean_abs_adv": mean_abs_adv,
                            "std_adv": std_adv,
                            "pos_adv_ratio": pos_adv_ratio,
                            "num_data": 0,
                            "skipped": True,
                            **prompt_filter_stats,
                            "prompt_trace_records": prompt_trace_records,
                            "dropped_zero_adv": dropped,
                            "mean_reward_total": mean_reward_total,
                            "mean_reward_format": mean_reward_format,
                            "mean_reward_property": mean_reward_property,
                            "min_reward_property": min_reward_property,
                            "max_reward_property": max_reward_property,
                            "mean_property_delta_nte": mean_property_delta_nte,
                            "mean_property_delta_nte_clipped": mean_property_delta_nte_clipped,
                            "mean_property_relative_delta_nte": mean_property_relative_delta_nte,
                            "mean_property_relative_delta_nte_clipped": mean_property_relative_delta_nte_clipped,
                            "mean_property_absolute_delta_nte": mean_property_absolute_delta_nte,
                            "mean_property_abs_nte": mean_property_abs_nte,
                            "mean_property_abs_nte_quality": mean_property_abs_nte_quality,
                            "property_reward_bucket_low_count": property_reward_bucket_low_count,
                            "property_reward_bucket_mid_count": property_reward_bucket_mid_count,
                            "property_reward_bucket_high_count": property_reward_bucket_high_count,
                            "mean_property_reward_bucket_low_count": property_reward_bucket_low_count,
                            "mean_property_reward_bucket_mid_count": property_reward_bucket_mid_count,
                            "mean_property_reward_bucket_high_count": property_reward_bucket_high_count,
                            "mean_reward_milestone_bonus": mean_reward_milestone_bonus,
                            "mean_reward_early_stop_bonus": mean_reward_early_stop_bonus,
                            "num_early_stop_met": num_early_stop_met,
                            "property_delta_weight": float(DELTA_NTE_WEIGHT),
                            "property_abs_weight": float(ABS_NTE_WEIGHT),
                            "mean_reward_repetition": mean_reward_repetition,
                            "mean_nte": mean_nte,
                            "std_nte": std_nte,
                            "num_final_valid": num_final_valid,
                            "num_final_missing": num_final_missing,
                            "num_terminal_total": num_terminal_total,
                            "terminal_valid_rate": terminal_valid_rate,
                            "mean_nte_relative_improvement": mean_nte_relative_improvement,
                            "std_nte_relative_improvement": std_nte_relative_improvement,
                            **nte_bucket_stats,
                            "num_turn_valid": num_turn_valid,
                            "num_turn_total": num_turn_total,
                            "turn_valid_rate": turn_valid_rate,
                            **turn_metrics,
                            "mean_final_qed_error": mean_final_qed_error,
                            "mean_final_logp_error": mean_final_logp_error,
                            "mean_final_mw_error": mean_final_mw_error,
                            "sampling_temperature": float(temperature),
                            "sampling_top_p": float(top_p),
                            "sampling_top_k": int(top_k),
                            "trajectories_per_prompt": int(trajectories_per_prompt),
                            "refresh_sampling_every": int(refresh_sampling_every),
                            **kl_metrics,
                            "time": elapsed,
                        }
                    )
            else:
                print(f"{iteration:>4} | no valid data, skipped")
                if keep_skipped_metrics:
                    all_metrics.append(
                        {
                            "iteration": iteration,
                            "num_data": 0,
                            "skipped": True,
                            **prompt_filter_stats,
                            "prompt_trace_records": prompt_trace_records,
                            "sampling_temperature": float(temperature),
                            "sampling_top_p": float(top_p),
                            "sampling_top_k": int(top_k),
                            "trajectories_per_prompt": int(trajectories_per_prompt),
                            "refresh_sampling_every": int(refresh_sampling_every),
                            **kl_metrics,
                            "time": elapsed,
                        }
                    )
            continue

        kl_metrics = _empty_kl_metrics(kl_penalty_coef)
        if kl_penalty_coef > 0:
            custom_datums, explicit_loss_fn, kl_tokens = _prepare_explicit_ppo_kl(
                all_datums,
                all_kl_masks,
                kl_reference_client,
                kl_penalty_coef,
            )
            fwdbwd_result = training_client.forward_backward_custom(
                custom_datums,
                explicit_loss_fn,
            ).result()
            kl_metrics["kl_tokens"] = int(kl_tokens)
            for key in (
                "ppo_loss",
                "explicit_forward_kl",
                "ppo_kl_total_loss",
                "ppo_clip_fraction",
                "importance_ratio_mean",
            ):
                value = fwdbwd_result.metrics.get(key)
                if isinstance(value, (int, float)):
                    kl_metrics[key] = float(value)
            kl_metrics["kl_policy_reference"] = kl_metrics["explicit_forward_kl"]
        else:
            training_client.forward_backward(
                _datums_for_training(all_datums),
                loss_fn="ppo",
                loss_fn_config={
                    "clip_low_threshold": 0.9,
                    "clip_high_threshold": 1.1,
                },
            ).result()

        training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate)
        ).result()

        should_refresh = ((iteration + 1) % refresh_sampling_every == 0) or (iteration == num_iterations - 1)
        if should_refresh:
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"single-stage-iter-{iteration:04d}"
            )

        mean_G = float(np.mean([details.get("mean_G", 0.0) for details in all_details]))
        std_G = float(np.mean([details.get("std_G", 0.0) for details in all_details]))
        mean_adv = float(np.mean([details.get("mean_adv", 0.0) for details in all_details]))
        mean_abs_adv = float(np.mean([details.get("mean_abs_adv", 0.0) for details in all_details]))
        std_adv = float(np.mean([details.get("std_adv", 0.0) for details in all_details]))
        pos_adv_ratio = float(np.mean([details.get("pos_adv_ratio", 0.0) for details in all_details]))
        dropped = int(np.sum([details.get("dropped_zero_adv", 0) for details in all_details]))
        mean_reward_total = float(np.mean([details.get("mean_reward_total", 0.0) for details in all_details]))
        mean_reward_format = float(np.mean([details.get("mean_reward_format", 0.0) for details in all_details]))
        mean_reward_property = float(np.mean([details.get("mean_reward_property", 0.0) for details in all_details]))
        min_reward_property, max_reward_property = _property_reward_range(all_details)
        mean_property_delta_nte = _mean_numeric_detail(all_details, "mean_property_delta_nte")
        mean_property_delta_nte_clipped = _mean_numeric_detail(all_details, "mean_property_delta_nte_clipped")
        mean_property_relative_delta_nte = _mean_numeric_detail(all_details, "mean_property_relative_delta_nte")
        mean_property_relative_delta_nte_clipped = _mean_numeric_detail(all_details, "mean_property_relative_delta_nte_clipped")
        mean_property_absolute_delta_nte = _mean_numeric_detail(all_details, "mean_property_absolute_delta_nte")
        mean_property_abs_nte = _mean_numeric_detail(all_details, "mean_property_abs_nte")
        mean_property_abs_nte_quality = _mean_numeric_detail(all_details, "mean_property_abs_nte_quality")
        mean_reward_milestone_bonus = float(np.mean([details.get("mean_reward_milestone_bonus", 0.0) for details in all_details]))
        mean_reward_early_stop_bonus = float(np.mean([details.get("mean_reward_early_stop_bonus", 0.0) for details in all_details]))
        num_early_stop_met = int(np.sum([details.get("num_early_stop_met", 0) for details in all_details]))
        mean_reward_repetition = float(np.mean([details.get("mean_reward_repetition", 0.0) for details in all_details]))

        print(
            f"{iteration:>4} | {mean_G:>8.4f} | {std_G:>7.4f} | {mean_abs_adv:>7.4f} | "
            f"{std_adv:>7.4f} | {100.0 * pos_adv_ratio:>5.1f}% | {len(all_datums):>6} | {elapsed:>6.1f}"
        )
        print(
            "      rewards | "
            f"total={mean_reward_total:+.3f} | "
            f"format={mean_reward_format:+.3f} | "
            f"property={mean_reward_property:+.3f} "
            f"[{_fmt_optional(min_reward_property, digits=3)}, {_fmt_optional(max_reward_property, digits=3)}] | "
            f"repetition={mean_reward_repetition:+.3f}"
        )
        print(
            "      turn property parts | "
            f"buckets low/mid/high={property_reward_bucket_low_count}/"
            f"{property_reward_bucket_mid_count}/{property_reward_bucket_high_count} | "
            f"rel_delta={_fmt_optional(mean_property_relative_delta_nte, digits=3)} | "
            f"rel_clip={_fmt_optional(mean_property_relative_delta_nte_clipped, digits=3)} | "
            f"abs_delta={_fmt_optional(mean_property_absolute_delta_nte, digits=3)} | "
            f"abs_nte={_fmt_optional(mean_property_abs_nte, digits=3)} | "
            f"abs_q={_fmt_optional(mean_property_abs_nte_quality, digits=3)}"
        )
        print(
            "      bonuses | "
            f"milestone={mean_reward_milestone_bonus:+.3f} | "
            f"early_stop={mean_reward_early_stop_bonus:+.3f} | "
            f"early_stop_count={num_early_stop_met}"
        )
        print(
            "      terminal nte | "
            f"mean={_fmt_optional(mean_nte)} std={_fmt_optional(std_nte)} "
            f"valid={num_final_valid} missing={num_final_missing} | "
            f"|dQED|={_fmt_optional(mean_final_qed_error)} | "
            f"|dLogP|={_fmt_optional(mean_final_logp_error)} | "
            f"|dMW|={_fmt_optional(mean_final_mw_error, digits=2)}"
        )
        print(
            "      nte relative improvement | "
            f"mean={_fmt_optional(mean_nte_relative_improvement)} "
            f"std={_fmt_optional(std_nte_relative_improvement)}"
        )
        print(
            "      initial nte buckets | "
            f"{_fmt_initial_nte_bucket_stats(nte_bucket_stats)}"
        )
        print(
            "      validity | "
            f"terminal={num_final_valid}/{num_terminal_total} "
            f"({100.0 * terminal_valid_rate if terminal_valid_rate is not None else float('nan'):.1f}%) | "
            f"turns={num_turn_valid}/{num_turn_total} "
            f"({100.0 * turn_valid_rate if turn_valid_rate is not None else float('nan'):.1f}%)"
        )
        if kl_metrics["kl_tokens"]:
            print(
                "      explicit kl | "
                f"beta={kl_metrics['kl_penalty_coef']:.4f} | "
                f"forward={kl_metrics['explicit_forward_kl']:.6f} | "
                f"ppo={kl_metrics['ppo_loss']:.6f} | "
                f"total={kl_metrics['ppo_kl_total_loss']:.6f} | "
                f"clip_fraction={kl_metrics['ppo_clip_fraction']:.4f} | "
                f"ratio_mean={kl_metrics['importance_ratio_mean']:.4f} | "
                f"tokens={kl_metrics['kl_tokens']}"
            )

        metric = dict(
            iteration=iteration,
            mean_G=mean_G,
            std_G=std_G,
            mean_adv=mean_adv,
            mean_abs_adv=mean_abs_adv,
            std_adv=std_adv,
            pos_adv_ratio=pos_adv_ratio,
            num_data=len(all_datums),
            **prompt_filter_stats,
            prompt_trace_records=prompt_trace_records,
            dropped_zero_adv=dropped,
            mean_reward_total=mean_reward_total,
            mean_reward_format=mean_reward_format,
            mean_reward_property=mean_reward_property,
            min_reward_property=min_reward_property,
            max_reward_property=max_reward_property,
            mean_property_delta_nte=mean_property_delta_nte,
            mean_property_delta_nte_clipped=mean_property_delta_nte_clipped,
            mean_property_relative_delta_nte=mean_property_relative_delta_nte,
            mean_property_relative_delta_nte_clipped=mean_property_relative_delta_nte_clipped,
            mean_property_absolute_delta_nte=mean_property_absolute_delta_nte,
            mean_property_abs_nte=mean_property_abs_nte,
            mean_property_abs_nte_quality=mean_property_abs_nte_quality,
            property_reward_bucket_low_count=property_reward_bucket_low_count,
            property_reward_bucket_mid_count=property_reward_bucket_mid_count,
            property_reward_bucket_high_count=property_reward_bucket_high_count,
            mean_property_reward_bucket_low_count=property_reward_bucket_low_count,
            mean_property_reward_bucket_mid_count=property_reward_bucket_mid_count,
            mean_property_reward_bucket_high_count=property_reward_bucket_high_count,
            mean_reward_milestone_bonus=mean_reward_milestone_bonus,
            mean_reward_early_stop_bonus=mean_reward_early_stop_bonus,
            num_early_stop_met=num_early_stop_met,
            property_delta_weight=float(DELTA_NTE_WEIGHT),
            property_abs_weight=float(ABS_NTE_WEIGHT),
            mean_reward_repetition=mean_reward_repetition,
            mean_nte=mean_nte,
            std_nte=std_nte,
            mean_nte_relative_improvement=mean_nte_relative_improvement,
            std_nte_relative_improvement=std_nte_relative_improvement,
            **nte_bucket_stats,
            num_final_valid=num_final_valid,
            num_final_missing=num_final_missing,
            num_terminal_total=num_terminal_total,
            terminal_valid_rate=terminal_valid_rate,
            num_turn_valid=num_turn_valid,
            num_turn_total=num_turn_total,
            turn_valid_rate=turn_valid_rate,
            **turn_metrics,
            mean_final_qed_error=mean_final_qed_error,
            mean_final_logp_error=mean_final_logp_error,
            mean_final_mw_error=mean_final_mw_error,
            sampling_temperature=float(temperature),
            sampling_top_p=float(top_p),
            sampling_top_k=int(top_k),
            trajectories_per_prompt=int(trajectories_per_prompt),
            refresh_sampling_every=int(refresh_sampling_every),
            **kl_metrics,
            ema_baseline=ema.value,
            time=elapsed,
            skipped=False,
            refreshed_sampling_client=bool(should_refresh),
        )
        all_metrics.append(metric)

    print(f"\nTraining complete ({num_iterations} iterations)")
    return all_metrics
