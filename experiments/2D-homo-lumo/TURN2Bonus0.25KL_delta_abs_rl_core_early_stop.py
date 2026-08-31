from __future__ import annotations

import asyncio
import math
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
DELTA_NTE_WEIGHT = 1.0
ABS_NTE_WEIGHT = 1.0
DELTA_NTE_RELATIVE_EPS = 1e-6
DELTA_NTE_RELATIVE_MIN = -4.0
DELTA_NTE_RELATIVE_MAX = 1.0
ABS_NTE_CAP = 2.0
ABS_NTE_QUALITY_MAX = 1.0
EARLY_STOP_NTE_RATIO = 0.25
EARLY_STOP_BONUS = 2.0
# Used only for metrics analysis, not for reward computation
_INITIAL_NTE_BUCKETS = (
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
    "QED": 0.05,
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
    by_bucket = {label: [] for label, _display, _low, _high in _INITIAL_NTE_BUCKETS}
    for record in records:
        initial_nte = record.get("initial_nte")
        if not isinstance(initial_nte, (int, float)):
            continue
        label = _bucket_label_for_initial_nte(float(initial_nte))
        if label is not None:
            by_bucket[label].append(record)

    for label, _display, _low, _high in _INITIAL_NTE_BUCKETS:
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
    return (
        "Improve the target property gaps while keeping already-good properties near target.\n"
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


def _sample_prompt_indices(
    raw_data,
    rng,
    *,
    candidate_prompts_per_iter: int,
    max_kept_prompts_per_iter: int,
) -> tuple[list[int], dict[str, int | float]]:
    candidate_count = min(int(candidate_prompts_per_iter), len(raw_data))
    prompt_indices: list[int] = []
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
        prompt_indices.append(int_idx)
        if len(prompt_indices) >= max_kept_prompts_per_iter:
            break

    stats = {
        "prompt_sampling_candidates": int(candidate_count),
        "prompt_sampling_kept": int(len(prompt_indices)),
        "prompt_sampling_skipped_unparseable": int(skipped_unparseable),
    }
    return prompt_indices, stats


def is_nte_improvement(
    valid_turn: bool,
    old_nte: float | None,
    new_nte: float | None,
) -> bool:
    return bool(
        valid_turn
        and isinstance(old_nte, (int, float))
        and isinstance(new_nte, (int, float))
        and float(new_nte) < float(old_nte)
    )


def compute_scheduled_learning_rate(
    global_iteration: int,
    *,
    warmup_start_lr: float = 5e-6,
    max_lr: float = 1e-5,
    min_lr: float = 5e-7,
    warmup_iterations: int = 1000,
    hold_iterations: int = 5000,
    total_iterations: int = 25000,
) -> float:
    if global_iteration < 0:
        raise ValueError("global_iteration must be >= 0")
    if warmup_iterations < 0 or hold_iterations < 0:
        raise ValueError("warmup_iterations and hold_iterations must be >= 0")
    decay_start = warmup_iterations + hold_iterations
    if total_iterations <= decay_start:
        raise ValueError("total_iterations must be greater than warmup_iterations + hold_iterations")
    if min(warmup_start_lr, max_lr, min_lr) <= 0:
        raise ValueError("learning rates must be positive")
    if not (warmup_start_lr <= max_lr and min_lr <= max_lr):
        raise ValueError("warmup_start_lr and min_lr must be <= max_lr")

    step = int(global_iteration)
    if warmup_iterations > 0 and step < warmup_iterations:
        progress = step / warmup_iterations
        return float(warmup_start_lr + progress * (max_lr - warmup_start_lr))
    if step <= decay_start:
        return float(max_lr)
    if step >= total_iterations:
        return float(min_lr)

    progress = (step - decay_start) / (total_iterations - decay_start)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (max_lr - min_lr) * cosine)


def _early_stop_threshold(initial_nte: float | None) -> float | None:
    if initial_nte is None:
        return None
    return float(EARLY_STOP_NTE_RATIO * initial_nte)


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
    """Compute property reward using a unified formula: r = 2.0 * Δ_rel_clipped + 0.5 * Q."""
    text = _normalize_completion_text(completion_text)
    in_smiles = extract_smiles(prompt_text)
    out_smiles = extract_smiles(text)
    _initial_nte = (
        float(initial_nte)
        if isinstance(initial_nte, (int, float))
        else _initial_nte_from_prompt(prompt_text)
    )
    details = {
        "property_reward": float(INVALIDITY_PENALTY),
        "property_failure": None,
        "improved": False,
        "accepted": False,
        "rejection_reason": "invalid_molecule",
        "initial_nte": _initial_nte,
        "total_err": None,
        "old_total_err": None,
        "property_delta_nte_raw": None,
        "property_delta_nte_clipped": None,
        "property_absolute_delta_nte": None,
        "property_relative_delta_nte_raw": None,
        "property_relative_delta_nte_clipped": None,
        "property_abs_nte": None,
        "property_abs_nte_quality": None,
        "qed_err": None, "logp_err": None, "mw_err": None,
        "qed_target": None, "logp_target": None, "mw_target": None,
        "qed_out": None, "logp_out": None, "mw_out": None,
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
        details["rejection_reason"] = "nte_unavailable"
        return details
    target, _input_props = parsed

    old_total_err, _old_parts, _old_props = property_error(mol_in, target)
    total_err, parts, output_props = property_error(mol_out, target)
    absolute_delta_nte = float(old_total_err - total_err)
    relative_delta_nte_raw = float(
        absolute_delta_nte / max(float(old_total_err), DELTA_NTE_RELATIVE_EPS)
    )
    relative_delta_nte_clipped = float(
        np.clip(relative_delta_nte_raw, DELTA_NTE_RELATIVE_MIN, DELTA_NTE_RELATIVE_MAX)
    )
    abs_nte_quality = float(
        ABS_NTE_QUALITY_MAX - np.clip(float(total_err) / ABS_NTE_CAP, 0.0, ABS_NTE_QUALITY_MAX)
    )
    improved = is_nte_improvement(True, old_total_err, total_err)
    q_term = ABS_NTE_WEIGHT * abs_nte_quality if improved else 0.0
    property_reward = float(DELTA_NTE_WEIGHT * relative_delta_nte_clipped + q_term)
    details.update(
        {
            "property_reward": property_reward,
            "property_failure": None,
            "improved": bool(improved),
            "accepted": bool(improved),
            "rejection_reason": None if improved else "no_improvement",
            "total_err": float(total_err),
            "old_total_err": float(old_total_err),
            "property_delta_nte_raw": relative_delta_nte_raw,
            "property_delta_nte_clipped": relative_delta_nte_clipped,
            "property_absolute_delta_nte": absolute_delta_nte,
            "property_relative_delta_nte_raw": relative_delta_nte_raw,
            "property_relative_delta_nte_clipped": relative_delta_nte_clipped,
            "property_abs_nte": float(total_err),
            "property_abs_nte_quality": abs_nte_quality,
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


def _replace_prompt_smiles(base_prompt: str, current_smiles: str) -> str:
    """Update prompt SMILES and property targets to reflect the new current molecule."""
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
    """Compress environment feedback to the first and last meaningful lines."""
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


def _rollback_state_feedback(
    current_prompt: str,
    *,
    rejection_reason: str,
    old_nte: float | None = None,
    new_nte: float | None = None,
) -> tuple[str, str, bool]:
    """Generate feedback when a turn is rejected, rolling back to the previous molecule."""
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
    if rejection_reason == "no_improvement":
        rejection_summary = (
            "Previous attempt was chemically valid but rejected because it did not improve NTE "
            f"(old={old_nte:.12g}, new={new_nte:.12g}; acceptance requires new NTE < old NTE)."
        )
    elif rejection_reason == "nte_unavailable":
        rejection_summary = (
            "Previous attempt was rejected because comparable old/new NTE values were unavailable."
        )
    else:
        rejection_summary = "Previous attempt was invalid or missing and was rejected."
    feedback = (
        f"Environment feedback: {rejection_summary} Rolled back to the previous accepted "
        f"molecule <SMILES>{rollback_smiles}</SMILES> with "
        f"QED={qed_val:.3f}, LogP={logp_val:.2f}, MW={mw_val:.1f}. "
        "This rollback molecule is the only edit base for the next turn."
        f"\nTarget progress:{progress_block}"
        f"\n{summary}"
    )
    guidance = f"all_met={all_met_state}; unmet_properties={unmet_text}.\n{guidance_block}"
    return feedback, guidance, all_met_state


def run_mode_b(
    example: dict,
    renderer,
    sampling_client,
    tokenizer,
    ema_baseline: EMABaseline,
    num_trajectories: int = 8,
    max_turns: int = 3,
    gamma: float = 1.1,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    debug: bool = False,
    debug_prefix: str = "",
) -> tuple[list[types.Datum], list[list[float]], list[float], dict]:
    prompt_text = str(example[PROMPT_COL])
    ground_truth = str(example[ANSWER_COL])

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
            old_nte = (
                float(reward_components["old_total_err"])
                if isinstance(reward_components.get("old_total_err"), (int, float))
                else None
            )
            improved = is_nte_improvement(valid_turn, old_nte, current_nte)
            if improved:
                rejection_reason = None
            elif not valid_turn:
                rejection_reason = "invalid_molecule"
            elif old_nte is None or current_nte is None:
                rejection_reason = "nte_unavailable"
            else:
                rejection_reason = "no_improvement"
            reward_components["improved"] = bool(improved)
            reward_components["accepted"] = bool(improved)
            reward_components["rejection_reason"] = rejection_reason
            early_stop_met = False
            early_stop_bonus = 0.0

            if improved:
                early_stop_met = (
                    early_stop_threshold_value is not None
                    and current_nte is not None
                    and current_nte <= early_stop_threshold_value
                )
                early_stop_bonus = float(EARLY_STOP_BONUS if early_stop_met else 0.0)

            reward_components["base_total_before_bonus"] = float(reward_components["total"])
            reward_components["early_stop_bonus"] = float(early_stop_bonus)
            reward_components["early_stop_met"] = bool(early_stop_met)
            reward_components["early_stop_threshold"] = early_stop_threshold_value
            reward_components["initial_nte"] = initial_nte
            reward_components["current_nte"] = current_nte
            reward_components["total"] = float(reward_components["total"] + early_stop_bonus)
            turn_reward = float(turn_reward + early_stop_bonus)

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
                    "turn_accepted_canonical_smiles": (
                        valid_output_canonical if improved else None
                    ),
                    "improved": bool(improved),
                    "accepted": bool(improved),
                    "rejection_reason": rejection_reason,
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
                    f"early={early_stop_bonus:.3f} "
                    f"valid_mol={mol_out is not None} improved={improved} "
                    f"rejection={rejection_reason}"
                )
                print(f"[DBG]{debug_prefix} traj={k} turn={turn_idx} raw_output: {raw_completion_text!r}")
                print(f"[DBG]{debug_prefix} traj={k} turn={turn_idx} normalized_output: {completion_text}")

            if improved:
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
                    st["current_prompt"],
                    rejection_reason=str(rejection_reason),
                    old_nte=old_nte,
                    new_nte=current_nte,
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
        last_turn_accepted = bool(last_turn.get("accepted", False))
        last_output_text = last_turn.get("completion_text")
        if st.get("has_valid_generated") and st.get("current_valid_canonical"):
            monitor_canonical = st.get("current_valid_canonical")
            terminal_output_text = st.get("current_valid_output_text")
            source = "accepted_generated"
        elif st.get("current_valid_canonical"):
            monitor_canonical = st.get("current_valid_canonical")
            terminal_output_text = st.get("current_valid_output_text")
            source = "accepted_initial"
        else:
            monitor_canonical = None
            terminal_output_text = None
            source = "missing"
        return {
            "trajectory_index": int(traj_idx),
            "last_turn": int(last_turn.get("turn_index", -1)) + 1 if last_turn else None,
            "last_turn_valid": bool(last_turn_canonical),
            "last_turn_accepted": last_turn_accepted,
            "last_turn_rejection_reason": last_turn.get("rejection_reason"),
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
        # All trajectories have identical returns → no meaningful group-relative signal
        relative_advs = [0.0] * len(G_values)
    else:
        relative_advs = ((G_arr - G_mean) / G_std).tolist()
    for G in G_values:
        ema_baseline.update_and_get(G)

    # === Turn-level GRPO advantages ===
    # Group turn rewards by turn_index, normalize within each group independently.
    # This solves the temporal credit assignment problem: a good Turn 1 is no longer
    # penalized by a bad Turn 2, and vice versa.
    MIN_TURN_GROUP_SIZE = 4
    turn_reward_groups: dict[int, list[float]] = {}
    turn_refs: dict[int, list[tuple[int, int]]] = {}

    for traj_idx, (turns, _returns, _G_k) in enumerate(all_trajectories):
        for turn_idx_in_traj, turn in enumerate(turns):
            t_idx = int(turn["turn_index"])
            turn_reward_groups.setdefault(t_idx, []).append(turn["turn_reward"])
            turn_refs.setdefault(t_idx, []).append((traj_idx, turn_idx_in_traj))

    turn_adv_lookup: dict[tuple[int, int], float] = {}
    dropped_zero_turn_adv = 0

    for t_idx in sorted(turn_reward_groups):
        rewards = turn_reward_groups[t_idx]
        refs = turn_refs[t_idx]
        arr = np.array(rewards, dtype=np.float64)

        if len(rewards) >= MIN_TURN_GROUP_SIZE:
            mean = arr.mean()
            std = arr.std()
            if std < 1e-8:
                advs = [0.0] * len(rewards)
            else:
                advs = ((arr - mean) / std).tolist()
        else:
            # Fallback: group too small, use trajectory-level advantage
            advs = [relative_advs[traj_idx] for traj_idx, _ in refs]

        for (traj_idx, turn_idx_in_traj), turn_adv in zip(refs, advs):
            key = (traj_idx, turn_idx_in_traj)
            turn_adv_lookup[key] = float(turn_adv)
            if abs(float(turn_adv)) < 1e-12:
                dropped_zero_turn_adv += 1

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
        for turn_idx_in_traj, (turn, turn_return) in enumerate(zip(turns, returns)):
            components = turn.get("reward_components", {})
            completion_text = turn.get("completion_text")
            turn_adv = turn_adv_lookup.get((traj_idx, turn_idx_in_traj), adv)
            turn_output_records.append(
                {
                    "trajectory_index": int(traj_idx),
                    "turn": int(turn.get("turn_index", -1)) + 1,
                    "valid": bool(turn.get("turn_valid_canonical_smiles")),
                    "improved": bool(turn.get("improved", False)),
                    "accepted": bool(turn.get("accepted", False)),
                    "rejection_reason": turn.get("rejection_reason"),
                    "input_smiles": turn.get("input_smiles"),
                    "input_canonical_smiles": turn.get("input_canonical_smiles"),
                    "canonical_smiles": turn.get("turn_valid_canonical_smiles"),
                    "raw_smiles": extract_smiles(completion_text),
                    "output_text": completion_text,
                    "turn_reward": float(turn.get("turn_reward", 0.0)),
                    "return": float(turn_return),
                    "trajectory_return": float(G_k),
                    "advantage": float(adv),
                    "turn_advantage": float(turn_adv),
                    "base_total_before_bonus": _component_or_none(
                        components, "base_total_before_bonus"
                    ),
                    "reward_total": _component_or_none(components, "total"),
                    "reward_format": _component_or_none(components, "strict"),
                    "reward_property": _component_or_none(components, "property"),
                    "reward_repetition": _component_or_none(components, "repetition"),
                    "early_stop_bonus": _component_or_none(components, "early_stop_bonus"),
                    "early_stop_met": bool(components.get("early_stop_met", False)),
                    "initial_nte": _component_or_none(components, "initial_nte"),
                    "current_nte": _component_or_none(components, "current_nte"),
                    "property_delta_nte_raw": _component_or_none(components, "property_delta_nte_raw"),
                    "property_delta_nte_clipped": _component_or_none(
                        components, "property_delta_nte_clipped"
                    ),
                    "property_absolute_delta_nte": _component_or_none(
                        components, "property_absolute_delta_nte"
                    ),
                    "property_relative_delta_nte_raw": _component_or_none(
                        components, "property_relative_delta_nte_raw"
                    ),
                    "property_relative_delta_nte_clipped": _component_or_none(
                        components, "property_relative_delta_nte_clipped"
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
    dropped_zero_adv = 0  # Now counts dropped TURNS (turn-level GRPO)
    for idx, ((turns, returns, G_k), adv) in enumerate(zip(all_trajectories, relative_advs)):
        traj_returns.append(float(G_k))
        for turn_idx_in_traj, turn in enumerate(turns):
            turn_adv = turn_adv_lookup.get((idx, turn_idx_in_traj), adv)
            if abs(turn_adv) < 1e-12:
                dropped_zero_adv += 1
                if debug:
                    print(
                        f"[DBG]{debug_prefix} drop traj={idx} turn={turn['turn_index']} "
                        f"because turn_adv is near zero ({turn_adv:.4e})"
                    )
                continue
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
                        advantages=([0.0] * (n_p - 1) + [turn_adv] * n_c)[:min_len],
                        logprobs=([0.0] * (n_p - 1) + turn["comp_logprobs"])[:min_len],
                    ),
                )
            )
            kl_masks.append(kl_mask)

    adv_arr = np.array(relative_advs, dtype=np.float64) if relative_advs else np.array([], dtype=np.float64)
    turn_adv_values = list(turn_adv_lookup.values())
    turn_adv_arr = np.array(turn_adv_values, dtype=np.float64) if turn_adv_values else np.array([], dtype=np.float64)
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
    num_turn_improved = sum(1 for turn in turn_records if turn.get("improved"))
    num_turn_accepted = sum(1 for turn in turn_records if turn.get("accepted"))
    num_turn_valid_rejected = sum(
        1
        for turn in turn_records
        if turn.get("turn_valid_canonical_smiles") and not turn.get("accepted")
    )
    turn_valid_rate = float(num_turn_valid / num_turns_total) if num_turns_total else None
    turn_improved_rate = (
        float(num_turn_improved / num_turns_total) if num_turns_total else None
    )
    turn_accepted_rate = (
        float(num_turn_accepted / num_turns_total) if num_turns_total else None
    )
    turn_total_counts: Counter[int] = Counter()
    turn_valid_counts: Counter[int] = Counter()
    turn_accepted_counts: Counter[int] = Counter()
    for turn in turn_records:
        turn_index = turn.get("turn_index")
        if not isinstance(turn_index, (int, float)):
            continue
        turn_no = int(turn_index) + 1
        turn_total_counts[turn_no] += 1
        if turn.get("turn_valid_canonical_smiles"):
            turn_valid_counts[turn_no] += 1
        if turn.get("accepted"):
            turn_accepted_counts[turn_no] += 1

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

    # Turn-group statistics for logging
    turn_group_stats = {}
    for t_idx in sorted(turn_reward_groups):
        rewards = turn_reward_groups[t_idx]
        refs = turn_refs[t_idx]
        r_arr = np.array(rewards, dtype=np.float64)
        advs_for_group = [turn_adv_lookup.get(ref, 0.0) for ref in refs]
        a_arr = np.array(advs_for_group, dtype=np.float64)
        turn_group_stats[f"turn{t_idx+1}_count"] = len(rewards)
        turn_group_stats[f"turn{t_idx+1}_mean_r"] = float(r_arr.mean()) if r_arr.size else 0.0
        turn_group_stats[f"turn{t_idx+1}_std_r"] = float(r_arr.std()) if r_arr.size else 0.0
        turn_group_stats[f"turn{t_idx+1}_mean_adv"] = float(a_arr.mean()) if a_arr.size else 0.0
        turn_group_stats[f"turn{t_idx+1}_std_adv"] = float(a_arr.std()) if a_arr.size else 0.0

    details = {
        "mean_G": float(G_mean),
        "std_G": float(G_std),
        "mean_adv": float(np.mean(adv_arr)) if adv_arr.size else 0.0,
        "mean_abs_adv": float(np.mean(np.abs(adv_arr))) if adv_arr.size else 0.0,
        "std_adv": float(np.std(adv_arr)) if adv_arr.size else 0.0,
        "pos_adv_ratio": float(np.mean(adv_arr > 0.0)) if adv_arr.size else 0.0,
        "mean_turn_adv": float(np.mean(turn_adv_arr)) if turn_adv_arr.size else 0.0,
        "mean_abs_turn_adv": float(np.mean(np.abs(turn_adv_arr))) if turn_adv_arr.size else 0.0,
        "std_turn_adv": float(np.std(turn_adv_arr)) if turn_adv_arr.size else 0.0,
        "pos_turn_adv_ratio": float(np.mean(turn_adv_arr > 0.0)) if turn_adv_arr.size else 0.0,
        "num_traj": len(all_trajectories),
        "num_turns_total": num_turns_total,
        "num_turn_total": num_turns_total,
        "num_turn_valid": int(num_turn_valid),
        "turn_valid_rate": turn_valid_rate,
        "num_turn_improved": int(num_turn_improved),
        "turn_improved_rate": turn_improved_rate,
        "num_turn_accepted": int(num_turn_accepted),
        "turn_accepted_rate": turn_accepted_rate,
        "num_turn_valid_rejected": int(num_turn_valid_rejected),
        "dropped_zero_adv": dropped_zero_adv,
        **turn_group_stats,
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
        turn_accepted = int(turn_accepted_counts.get(turn_no, 0))
        details[f"turn{turn_no}_total_count"] = turn_total
        details[f"turn{turn_no}_valid_count"] = turn_valid
        details[f"turn{turn_no}_accepted_count"] = turn_accepted
        details[f"turn{turn_no}_valid_rate"] = (
            float(turn_valid / turn_total) if turn_total else None
        )
        details[f"turn{turn_no}_accepted_rate"] = (
            float(turn_accepted / turn_total) if turn_total else None
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
        "explicit_forward_kl": None,
        "kl_tokens": 0,
        "ppo_loss": None,
        "ppo_kl_total_loss": None,
        "ppo_clip_fraction": None,
        "importance_ratio_mean": None,
    }


def compute_ppo_forward_kl_loss(
    new_logprobs_list,
    *,
    old_logprobs_list,
    reference_logprobs_list,
    advantages_list,
    action_masks,
    kl_penalty_coef: float,
    clip_low_threshold: float = 0.8,
    clip_high_threshold: float = 1.2,
):
    ppo_terms = []
    forward_kl_terms = []
    importance_ratios = []
    clip_indicators = []

    for new_logprobs, old_logprobs, reference_logprobs, advantages, action_mask in zip(
        new_logprobs_list,
        old_logprobs_list,
        reference_logprobs_list,
        advantages_list,
        action_masks,
        strict=True,
    ):
        device = new_logprobs.device
        dtype = new_logprobs.dtype
        old = torch.as_tensor(old_logprobs, dtype=dtype, device=device)
        reference = torch.as_tensor(reference_logprobs, dtype=dtype, device=device)
        advantages_tensor = torch.as_tensor(advantages, dtype=dtype, device=device)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
        usable_len = min(
            len(new_logprobs),
            len(old),
            len(reference),
            len(advantages_tensor),
            len(mask),
        )
        if usable_len <= 0:
            continue
        mask = mask[:usable_len]
        if not torch.any(mask):
            continue

        new_action = new_logprobs[:usable_len][mask]
        old_action = old[:usable_len][mask]
        reference_action = reference[:usable_len][mask]
        action_advantages = advantages_tensor[:usable_len][mask]

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
        nonnegative_kl = torch.expm1(-policy_reference_logratio) + policy_reference_logratio
        forward_kl_terms.append(importance_ratio * torch.clamp(nonnegative_kl, min=0.0))
        importance_ratios.append(importance_ratio)
        clip_indicators.append(
            (importance_ratio < clip_low_threshold) | (importance_ratio > clip_high_threshold)
        )

    if not ppo_terms:
        raise ValueError("No action tokens available for explicit PPO + KL loss")

    flat_ppo_terms = torch.cat(ppo_terms)
    flat_forward_kl = torch.cat(forward_kl_terms)
    flat_importance_ratios = torch.cat(importance_ratios)
    flat_clip_indicators = torch.cat(clip_indicators)

    ppo_loss = -flat_ppo_terms.mean()
    forward_kl = flat_forward_kl.mean()
    total_loss = ppo_loss + float(kl_penalty_coef) * forward_kl
    metrics = {
        "ppo_loss": float(ppo_loss.detach().item()),
        "explicit_forward_kl": float(forward_kl.detach().item()),
        "ppo_kl_total_loss": float(total_loss.detach().item()),
        "ppo_clip_fraction": float(flat_clip_indicators.float().mean().detach().item()),
        "importance_ratio_mean": float(flat_importance_ratios.mean().detach().item()),
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
    sequence_inputs = []
    for datum, kl_mask in zip(datums, kl_masks):
        target_tokens = list(datum.loss_fn_inputs["target_tokens"].data)
        if not target_tokens:
            raise ValueError("Explicit PPO + KL received a datum with no target tokens")
        if len(kl_mask) != len(target_tokens):
            raise ValueError("KL action mask length does not match target token length")
        sequence_inputs.append(datum.model_input.append_int(int(target_tokens[-1])))

    reference_logprobs = _run_async_sync(
        _compute_reference_logprobs(kl_reference_client, sequence_inputs)
    )
    reference_action_logprobs = []
    for datum, ref_logprobs in zip(datums, reference_logprobs, strict=True):
        target_len = len(datum.loss_fn_inputs["target_tokens"].data)
        ref_action_logprobs = list(ref_logprobs[1:])
        if len(ref_action_logprobs) < target_len:
            raise ValueError(
                "Reference logprob response is shorter than the target token sequence"
            )
        reference_action_logprobs.append(ref_action_logprobs[:target_len])

    old_logprobs = [
        list(datum.loss_fn_inputs["logprobs"].data)
        for datum in datums
    ]
    advantages = [
        list(datum.loss_fn_inputs["advantages"].data)
        for datum in datums
    ]
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
    start_iteration: int = 0,
    prompts_per_iter: int = 4,
    candidate_prompts_per_iter: int = 6,
    max_kept_prompts_per_iter: int = 4,
    lr_warmup_start: float = 5e-6,
    lr_max: float = 5e-5,
    lr_min: float = 1e-6,
    lr_warmup_iters: int = 1000,
    lr_hold_iters: int = 5000,
    lr_total_iters: int = 25000,
    trajectories_per_prompt: int = 8,
    max_turns: int = 3,
    gamma: float = 1.1,
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
    if start_iteration < 0:
        raise ValueError("start_iteration must be >= 0")
    if kl_penalty_coef < 0:
        raise ValueError("kl_penalty_coef must be >= 0")
    if kl_penalty_coef > 0 and kl_reference_client is None:
        raise ValueError("kl_reference_client is required when kl_penalty_coef > 0")
    compute_scheduled_learning_rate(
        start_iteration,
        warmup_start_lr=lr_warmup_start,
        max_lr=lr_max,
        min_lr=lr_min,
        warmup_iterations=lr_warmup_iters,
        hold_iterations=lr_hold_iters,
        total_iterations=lr_total_iters,
    )

    rng = np.random.default_rng(rng_seed)
    ema = EMABaseline(alpha=ema_alpha)
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"single-stage-global-{start_iteration:05d}-init"
    )

    all_metrics = []
    print("Starting single-stage multi-turn GRPO training")
    print(
        "reward = strict_format_reward + accepted_path_property_r "
        "(invalid=-5, valid uses unified prop formula) + prompt-local repetition_penalty"
    )
    print(
        f"sampling temperature={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}, "
        f"trajectories_per_prompt={trajectories_per_prompt}, refresh_sampling_every={refresh_sampling_every}"
    )
    print(
        "prompt sampling: "
        f"candidates={candidate_prompts_per_iter}, max_kept={max_kept_prompts_per_iter}, "
        f"all parseable NTE buckets eligible, legacy_prompts_per_iter={prompts_per_iter}"
    )
    print(
        "learning rate schedule: "
        f"global_start={start_iteration}, warmup={lr_warmup_iters}, hold={lr_hold_iters}, "
        f"total={lr_total_iters}, lr={lr_warmup_start:g}->{lr_max:g}->{lr_min:g}"
    )
    if kl_penalty_coef > 0:
        print(f"Explicit PPO + forward-KL loss enabled: beta={kl_penalty_coef:.4f}")
    header = f"{'Global':>7} | {'Mean G':>8} | {'std_G':>7} | {'|adv|':>7} | {'std_adv':>7} | {'+adv%':>6} | {'#data':>6} | {'LR':>9} | {'t(s)':>6}"
    print(header)
    print("-" * len(header))

    def _flatten_numeric(details_list, key: str) -> list[float]:
        values = []
        for details in details_list:
            for value in details.get(key, []) or []:
                if isinstance(value, (int, float)):
                    values.append(float(value))
        return values

    def _fmt_optional(value, digits: int = 4) -> str:
        return "nan" if value is None else f"{value:.{digits}f}"

    def _terminal_records_from_details(details_list: list[dict]) -> list[dict]:
        records = []
        for details in details_list:
            records.extend(details.get("terminal_molecule_records", []) or [])
        return records

    def _fmt_initial_nte_bucket_stats(stats: dict[str, float | int | None]) -> str:
        parts = []
        for label, display, _low, _high in _INITIAL_NTE_BUCKETS:
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
        print(f"      terminal summary | iteration={iteration}")
        for prompt_batch_idx, details in enumerate(details_list):
            records = details.get("terminal_molecule_records", []) or []
            accepted = sum(1 for r in records if r.get("source") == "accepted_generated")
            rolled = sum(1 for r in records if r.get("source") == "accepted_initial")
            missing = sum(1 for r in records if r.get("source") == "missing")
            prompt_idx = details.get("prompt_index", "?")
            nte_vals = [r.get("nte") for r in records if isinstance(r.get("nte"), (int, float))]
            avg_nte = f"{sum(nte_vals)/len(nte_vals):.3f}" if nte_vals else "nan"
            print(
                f"        prompt={prompt_idx} | "
                f"terminal_nte_mean={avg_nte} | "
                f"accepted={accepted} rolled_back={rolled} missing={missing}"
            )

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
            print(f"      trace | iteration={iteration} no prompt details")
            return

        print(f"      trace | iteration={iteration} prompts={len(details_list)}")
        ordered_details = sorted(
            details_list,
            key=lambda item: int(item.get("prompt_batch_index", 0) or 0),
        )
        for prompt_no, details in enumerate(ordered_details, start=1):
            prompt_idx = details.get("prompt_index", "?")
            terminal_records = details.get("terminal_molecule_records", []) or []
            turn_records = details.get("turn_output_records", []) or []
            terminals_by_traj = _terminal_records_by_traj(terminal_records)
            turns_by_traj = _turn_records_by_traj(turn_records)
            traj_ids = sorted(set(terminals_by_traj) | set(turns_by_traj))

            # Prompt header line
            nte_vals = [r.get("nte") for r in terminal_records if isinstance(r.get("nte"), (int, float))]
            avg_nte = f"{sum(nte_vals)/len(nte_vals):.3f}" if nte_vals else "nan"
            accepted = sum(1 for r in terminal_records if r.get("source") == "accepted_generated")
            print(
                f"      ├─ prompt {prompt_no}/{len(ordered_details)} idx={prompt_idx} | "
                f"terminal_nte_mean={avg_nte} | accepted={accepted}/{len(terminal_records)}"
            )

            if not traj_ids:
                print("      │  (no trajectories)")
                continue

            for traj_idx in traj_ids:
                terminal = terminals_by_traj.get(traj_idx, {})
                status = terminal.get("source", "?")
                smiles = terminal.get("canonical_smiles") or "None"
                nte = _fmt_optional(terminal.get("nte"))
                g_val = _fmt_optional(terminal.get("trajectory_return"))
                adv = _fmt_optional(terminal.get("advantage"))
                print(
                    f"      │  traj={traj_idx} | {status} | "
                    f"nte={nte} G={g_val} adv={adv} | smiles={smiles}"
                )

                for turn in turns_by_traj.get(traj_idx, []):
                    t = turn.get("turn")
                    v, imp, acc = turn.get("valid"), turn.get("improved"), turn.get("accepted")
                    rej = turn.get("rejection_reason")
                    rwd = _fmt_optional(turn.get("turn_reward"))
                    ret = _fmt_optional(turn.get("return"))
                    fmt_r = _fmt_optional(turn.get("reward_format"), digits=3)
                    prop_r = _fmt_optional(turn.get("reward_property"), digits=3)
                    rep_r = _fmt_optional(turn.get("reward_repetition"), digits=3)
                    early = _fmt_optional(turn.get("early_stop_bonus"), digits=3)
                    old_nte = _fmt_optional(turn.get("old_total_err"))
                    new_nte = _fmt_optional(turn.get("total_err"))
                    rel_d = _fmt_optional(turn.get("property_relative_delta_nte_raw"))
                    init_nte = _fmt_optional(turn.get("initial_nte"))
                    e_met = turn.get("early_stop_met")
                    print(
                        f"      │  │ turn={t} | "
                        f"valid={v} improved={imp} accepted={acc} rejection={rej} | "
                        f"reward={rwd} return={ret} | "
                        f"fmt={fmt_r} prop={prop_r} rep={rep_r} early={early}"
                    )
                    print(
                        f"      │  │  nte: {old_nte} → {new_nte} "
                        f"(rel_delta={rel_d}) | initial_nte={init_nte} early_met={e_met}"
                    )
                    out_text = turn.get("output_text") or "None"
                    # Truncate very long outputs
                    if len(out_text) > 120:
                        out_text = out_text[:120] + "..."
                    print(f"      │  │  output: {out_text}")

            print(f"      │")

    for local_iteration in range(num_iterations):
        global_iteration = int(start_iteration + local_iteration)
        iteration = global_iteration
        learning_rate = compute_scheduled_learning_rate(
            global_iteration,
            warmup_start_lr=lr_warmup_start,
            max_lr=lr_max,
            min_lr=lr_min,
            warmup_iterations=lr_warmup_iters,
            hold_iterations=lr_hold_iters,
            total_iterations=lr_total_iters,
        )
        t0 = time.time()

        prompt_indices, prompt_sampling_stats = _sample_prompt_indices(
            raw_data,
            rng,
            candidate_prompts_per_iter=candidate_prompts_per_iter,
            max_kept_prompts_per_iter=max_kept_prompts_per_iter,
        )
        print(
            "      prompt sampling | "
            f"candidates={prompt_sampling_stats['prompt_sampling_candidates']} "
            f"kept={prompt_sampling_stats['prompt_sampling_kept']} "
            f"skipped_unparseable={prompt_sampling_stats['prompt_sampling_skipped_unparseable']}"
        )
        if not prompt_indices:
            print("      prompt sampling | no parseable prompt selected")
        all_datums = []
        all_kl_masks = []
        all_rewards = []
        all_details = []

        for local_idx, prompt_idx in enumerate(prompt_indices):
            example = raw_data[int(prompt_idx)]
            try:
                do_debug = (
                    debug_adv
                    and local_iteration < debug_iterations
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
                    debug_prefix=(
                        f" global_it={global_iteration} local_it={local_iteration} "
                        f"p={int(prompt_idx)} "
                    ),
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
        num_turn_improved = int(
            np.sum([details.get("num_turn_improved", 0) for details in all_details])
        )
        num_turn_accepted = int(
            np.sum([details.get("num_turn_accepted", 0) for details in all_details])
        )
        num_turn_valid_rejected = int(
            np.sum([details.get("num_turn_valid_rejected", 0) for details in all_details])
        )
        turn_valid_rate = float(num_turn_valid / num_turn_total) if num_turn_total else None
        turn_improved_rate = (
            float(num_turn_improved / num_turn_total) if num_turn_total else None
        )
        turn_accepted_rate = (
            float(num_turn_accepted / num_turn_total) if num_turn_total else None
        )
        turn_metrics: dict[str, float | int | None] = {}
        for turn_no in range(1, int(max_turns) + 1):
            turn_total = int(np.sum([details.get(f"turn{turn_no}_total_count", 0) for details in all_details]))
            turn_valid = int(np.sum([details.get(f"turn{turn_no}_valid_count", 0) for details in all_details]))
            turn_accepted = int(
                np.sum(
                    [
                        details.get(f"turn{turn_no}_accepted_count", 0)
                        for details in all_details
                    ]
                )
            )
            turn_metrics[f"turn{turn_no}_total_count"] = turn_total
            turn_metrics[f"turn{turn_no}_valid_count"] = turn_valid
            turn_metrics[f"turn{turn_no}_accepted_count"] = turn_accepted
            turn_metrics[f"turn{turn_no}_valid_rate"] = (
                float(turn_valid / turn_total) if turn_total else None
            )
            turn_metrics[f"turn{turn_no}_accepted_rate"] = (
                float(turn_accepted / turn_total) if turn_total else None
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
        _print_iteration_trace_example(iteration, all_details)
        mean_nte = _mean_or_none(nte_values)
        std_nte = _std_or_none(nte_values)
        mean_nte_relative_improvement = _mean_or_none(nte_relative_improvements)
        std_nte_relative_improvement = _std_or_none(nte_relative_improvements)
        mean_final_qed_error = _mean_or_none(final_qed_errors)
        mean_final_logp_error = _mean_or_none(final_logp_errors)
        mean_final_mw_error = _mean_or_none(final_mw_errors)
        kl_metrics = _empty_kl_metrics(kl_penalty_coef)
        iteration_fields = {
            "iteration": global_iteration,
            "global_iteration": global_iteration,
            "local_iteration": local_iteration,
            "learning_rate": float(learning_rate),
        }

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
                mean_reward_early_stop_bonus = float(np.mean([details.get("mean_reward_early_stop_bonus", 0.0) for details in all_details]))
                num_early_stop_met = int(np.sum([details.get("num_early_stop_met", 0) for details in all_details]))
                mean_reward_repetition = float(np.mean([details.get("mean_reward_repetition", 0.0) for details in all_details]))

                print(
                    f"{global_iteration:>7} | no valid data, skipped | lr={learning_rate:.3e} "
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
                    "      bonuses | "
                    f"early_stop={mean_reward_early_stop_bonus:+.3f} | "
                    f"early_stop_count={num_early_stop_met}"
                )
                print(
                    "      turn property parts | "
                    f"rel_delta={_fmt_optional(mean_property_relative_delta_nte, digits=3)} | "
                    f"rel_clip={_fmt_optional(mean_property_relative_delta_nte_clipped, digits=3)} | "
                    f"abs_delta={_fmt_optional(mean_property_absolute_delta_nte, digits=3)} | "
                    f"abs_nte={_fmt_optional(mean_property_abs_nte, digits=3)} | "
                    f"abs_q={_fmt_optional(mean_property_abs_nte_quality, digits=3)}"
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
                    f"({100.0 * turn_valid_rate if turn_valid_rate is not None else float('nan'):.1f}%) | "
                    f"accepted={num_turn_accepted}/{num_turn_total} "
                    f"({100.0 * turn_accepted_rate if turn_accepted_rate is not None else float('nan'):.1f}%) | "
                    f"valid_rejected={num_turn_valid_rejected}"
                )

                if keep_skipped_metrics:
                    all_metrics.append(
                        {
                            **iteration_fields,
                            "mean_G": mean_G,
                            "std_G": std_G,
                            "mean_adv": mean_adv,
                            "mean_abs_adv": mean_abs_adv,
                            "std_adv": std_adv,
                            "pos_adv_ratio": pos_adv_ratio,
                            "num_data": 0,
                            "skipped": True,
                            **prompt_sampling_stats,
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
                            "num_turn_improved": num_turn_improved,
                            "turn_improved_rate": turn_improved_rate,
                            "num_turn_accepted": num_turn_accepted,
                            "turn_accepted_rate": turn_accepted_rate,
                            "num_turn_valid_rejected": num_turn_valid_rejected,
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
                print(
                    f"{global_iteration:>7} | no valid data, skipped | "
                    f"lr={learning_rate:.3e}"
                )
                if keep_skipped_metrics:
                    all_metrics.append(
                        {
                            **iteration_fields,
                            "num_data": 0,
                            "skipped": True,
                            "num_turn_valid": num_turn_valid,
                            "num_turn_total": num_turn_total,
                            "turn_valid_rate": turn_valid_rate,
                            "num_turn_improved": num_turn_improved,
                            "turn_improved_rate": turn_improved_rate,
                            "num_turn_accepted": num_turn_accepted,
                            "turn_accepted_rate": turn_accepted_rate,
                            "num_turn_valid_rejected": num_turn_valid_rejected,
                            **turn_metrics,
                            **prompt_sampling_stats,
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
        else:
            fwdbwd = training_client.forward_backward(
                _datums_for_training(all_datums),
                loss_fn="ppo",
                loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
            )
            fwdbwd.result()

        training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate, grad_clip_norm=1.0)
        ).result()

        should_refresh = (
            ((local_iteration + 1) % refresh_sampling_every == 0)
            or (local_iteration == num_iterations - 1)
        )
        if should_refresh:
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"single-stage-global-{global_iteration:05d}"
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
        mean_reward_early_stop_bonus = float(np.mean([details.get("mean_reward_early_stop_bonus", 0.0) for details in all_details]))
        num_early_stop_met = int(np.sum([details.get("num_early_stop_met", 0) for details in all_details]))
        mean_reward_repetition = float(np.mean([details.get("mean_reward_repetition", 0.0) for details in all_details]))

        print(
            f"{global_iteration:>7} | {mean_G:>8.4f} | {std_G:>7.4f} | {mean_abs_adv:>7.4f} | "
            f"{std_adv:>7.4f} | {100.0 * pos_adv_ratio:>5.1f}% | {len(all_datums):>6} | "
            f"{learning_rate:>9.3e} | {elapsed:>6.1f}"
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
            "      bonuses | "
            f"early_stop={mean_reward_early_stop_bonus:+.3f} | "
            f"early_stop_count={num_early_stop_met}"
        )
        print(
            "      turn property parts | "
            f"rel_delta={_fmt_optional(mean_property_relative_delta_nte, digits=3)} | "
            f"rel_clip={_fmt_optional(mean_property_relative_delta_nte_clipped, digits=3)} | "
            f"abs_delta={_fmt_optional(mean_property_absolute_delta_nte, digits=3)} | "
            f"abs_nte={_fmt_optional(mean_property_abs_nte, digits=3)} | "
            f"abs_q={_fmt_optional(mean_property_abs_nte_quality, digits=3)}"
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
            f"({100.0 * turn_valid_rate if turn_valid_rate is not None else float('nan'):.1f}%) | "
            f"accepted={num_turn_accepted}/{num_turn_total} "
            f"({100.0 * turn_accepted_rate if turn_accepted_rate is not None else float('nan'):.1f}%) | "
            f"valid_rejected={num_turn_valid_rejected}"
        )
        if kl_metrics["kl_tokens"]:
            print(
                "      kl | "
                f"coef={kl_metrics['kl_penalty_coef']:.4f} | "
                f"forward={kl_metrics['explicit_forward_kl']:.6f} | "
                f"ppo={kl_metrics['ppo_loss']:.6f} | "
                f"total={kl_metrics['ppo_kl_total_loss']:.6f} | "
                f"tokens={kl_metrics['kl_tokens']} | "
                f"clip%={kl_metrics.get('ppo_clip_fraction', 0):.3f} | "
                f"ρ_mean={kl_metrics.get('importance_ratio_mean', 0):.3f}"
            )

        # Turn-group GRPO stats (aggregated across prompts)
        _tg_parts = []
        for _ti in range(1, int(max_turns) + 1):
            _counts = [d.get(f"turn{_ti}_count", 0) for d in all_details]
            _mean_rs = [d.get(f"turn{_ti}_mean_r") for d in all_details if isinstance(d.get(f"turn{_ti}_mean_r"), (int, float))]
            _std_rs = [d.get(f"turn{_ti}_std_r") for d in all_details if isinstance(d.get(f"turn{_ti}_std_r"), (int, float))]
            _total_n = sum(int(c) for c in _counts if isinstance(c, (int, float)))
            _avg_r = float(np.mean(_mean_rs)) if _mean_rs else None
            _avg_std = float(np.mean(_std_rs)) if _std_rs else None
            if _total_n > 0:
                _tg_parts.append(
                    f"T{_ti}:n={_total_n} r={_avg_r:+.2f}" if _avg_r is not None else f"T{_ti}:n={_total_n}"
                    + (f"±{_avg_std:.2f}" if _avg_std is not None else "")
                )
        if _tg_parts:
            print("      turn groups | " + " | ".join(_tg_parts))

        metric = dict(
            **iteration_fields,
            mean_G=mean_G,
            std_G=std_G,
            mean_adv=mean_adv,
            mean_abs_adv=mean_abs_adv,
            std_adv=std_adv,
            pos_adv_ratio=pos_adv_ratio,
            num_data=len(all_datums),
            **prompt_sampling_stats,
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
            num_turn_improved=num_turn_improved,
            turn_improved_rate=turn_improved_rate,
            num_turn_accepted=num_turn_accepted,
            turn_accepted_rate=turn_accepted_rate,
            num_turn_valid_rejected=num_turn_valid_rejected,
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

    print(
        f"\nTraining complete ({num_iterations} iterations, "
        f"global {start_iteration}..{start_iteration + num_iterations - 1})"
    )
    return all_metrics
