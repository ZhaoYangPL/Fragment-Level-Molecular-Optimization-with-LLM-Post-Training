from __future__ import annotations

import re
import time
from collections import Counter
from typing import List
from typing import Optional

import numpy as np
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

_FORMAT_RE = re.compile(
    r"^(Replace|Add|Remove)\s+(.+?)(?:\s+with\s+(.+?))?\s+to\s+form\s+<SMILES>(.+)</SMILES>\.$",
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
    match = re.search(r"<SMILES>(.*?)</SMILES>", str(text), flags=re.DOTALL)
    return match.group(1).strip() if match else None


def strict_format_reward(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    for text in completions:
        normalized = _normalize_completion_text(text)
        rewards.append(1.0 if _FORMAT_RE.fullmatch(normalized) else FORMAT_PENALTY)
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


def compute_property_reward_details(completion_text: str, prompt_text: str) -> dict:
    text = _normalize_completion_text(completion_text)
    in_smiles = extract_smiles(prompt_text)
    out_smiles = extract_smiles(text)
    details = {
        "property_reward": float(INVALIDITY_PENALTY),
        "property_failure": None,
        "total_err": None,
        "old_total_err": None,
        "property_delta_nte": None,
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
    property_reward = float(old_total_err - total_err)
    details.update(
        {
            "property_reward": property_reward,
            "property_failure": None,
            "total_err": float(total_err),
            "old_total_err": float(old_total_err),
            "property_delta_nte": property_reward,
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
    return [
        float(compute_property_reward_details(comp, prompt)["property_reward"])
        for comp, prompt in zip(completions, prompts)
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
    **_unused,
) -> dict:
    text = _normalize_completion_text(completion_text)
    pred_smiles = extract_smiles(text)
    mol_pred = Chem.MolFromSmiles(pred_smiles) if pred_smiles else None

    format_score = float(strict_format_reward([text])[0])
    prop_details = compute_property_reward_details(text, prompt_text)
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
) -> tuple[list[types.Datum], list[float], dict]:
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
            "Environment feedback: previous attempt invalid/missing; using previous valid molecule "
            f"with QED={qed_val:.3f}, LogP={logp_val:.2f}, MW={mw_val:.1f}."
            f"\nTarget progress:{progress_block}"
            f"\n{summary}"
        )
        guidance = f"all_met={all_met_state}; unmet_properties={unmet_text}.\n{guidance_block}"
        return feedback, guidance, all_met_state

    initial_mol = _mol_from_prompt(prompt_text)
    initial_smiles = extract_smiles(prompt_text)
    initial_canonical = Chem.MolToSmiles(initial_mol) if initial_mol is not None else None

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
            "has_valid_generated": False,
            "last_valid_generated_smiles": None,
            "last_valid_generated_canonical": None,
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
            )
            turn_reward = float(reward_components["total"])
            env_feedback, mol_out, all_met = _run_env(completion_text, st["current_prompt"])
            pred_smiles = extract_smiles(completion_text)
            valid_output_canonical = Chem.MolToSmiles(mol_out) if mol_out is not None else None

            st["turns"].append(
                {
                    "turn_index": int(turn_idx),
                    "prompt_tokens": prompt_tokens[k],
                    "comp_tokens": list(seq.tokens),
                    "comp_logprobs": list(seq.logprobs),
                    "turn_reward": turn_reward,
                    "reward_components": reward_components,
                    "turn_valid_canonical_smiles": valid_output_canonical,
                }
            )

            if debug:
                print(
                    f"[DBG]{debug_prefix} traj={k} turn={turn_idx} "
                    f"reward={turn_reward:.4f} all_met={all_met} "
                    f"fmt={reward_components['strict']:.3f} "
                    f"prop={reward_components['property']:.3f} "
                    f"rep={reward_components['repetition']:.3f} "
                    f"valid_mol={mol_out is not None}"
                )
                print(f"[DBG]{debug_prefix} traj={k} turn={turn_idx} raw_output: {raw_completion_text!r}")
                print(f"[DBG]{debug_prefix} traj={k} turn={turn_idx} normalized_output: {completion_text}")

            turn_mol_in = _mol_from_prompt(st["current_prompt"])
            if pred_smiles and turn_mol_in is not None and mol_out is not None:
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
                st["has_valid_generated"] = True
                st["last_valid_generated_smiles"] = pred_smiles
                st["last_valid_generated_canonical"] = valid_output_canonical
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

            if (turn_idx == max_turns - 1) or all_met:
                st["done"] = True

    def _final_result_canonical(st: dict) -> str | None:
        if st.get("has_valid_generated"):
            return st.get("current_valid_canonical") or st.get("last_valid_generated_canonical")
        return None

    final_canonical_by_traj = [_final_result_canonical(st) for st in traj_states]

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

    final_nte_records = [
        compute_final_nte_details(final_can, prompt_text)
        for final_can in final_canonical_by_traj
    ]

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

    datums = []
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

    valid_nte_records = [record for record in final_nte_records if record.get("valid")]

    def _values(name: str) -> list[float]:
        return [float(record[name]) for record in valid_nte_records if record.get(name) is not None]

    nte_values = _values("nte")
    final_qed_errors = _values("qed_error")
    final_logp_errors = _values("logp_error")
    final_mw_errors = _values("mw_error")
    terminal_total = len(final_nte_records)
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
        "mean_reward_repetition": _mean_component("repetition"),
        "mean_repetition_count": _mean_component("repetition_count"),
        "num_final_valid": len(valid_nte_records),
        "num_final_missing": len(final_nte_records) - len(valid_nte_records),
        "num_terminal_total": terminal_total,
        "terminal_valid_rate": terminal_valid_rate,
        "mean_nte": _mean_or_none(nte_values),
        "std_nte": _std_or_none(nte_values),
        "mean_final_qed_error": _mean_or_none(final_qed_errors),
        "mean_final_logp_error": _mean_or_none(final_logp_errors),
        "mean_final_mw_error": _mean_or_none(final_mw_errors),
        "nte_values": nte_values,
        "final_qed_errors": final_qed_errors,
        "final_logp_errors": final_logp_errors,
        "final_mw_errors": final_mw_errors,
        "final_canonical_smiles": final_canonicals,
        "final_canonical_smiles_per_trajectory": final_canonical_by_traj,
    }
    for turn_no in range(1, int(max_turns) + 1):
        turn_total = int(turn_total_counts.get(turn_no, 0))
        turn_valid = int(turn_valid_counts.get(turn_no, 0))
        details[f"turn{turn_no}_total_count"] = turn_total
        details[f"turn{turn_no}_valid_count"] = turn_valid
        details[f"turn{turn_no}_valid_rate"] = (
            float(turn_valid / turn_total) if turn_total else None
        )
    return datums, traj_returns, details


def run_multistep_rl_training_fixed(
    raw_data,
    renderer,
    training_client,
    tokenizer,
    num_iterations: int = 50,
    prompts_per_iter: int = 4,
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
):
    if refresh_sampling_every < 1:
        raise ValueError("refresh_sampling_every must be >= 1")

    rng = np.random.default_rng(rng_seed)
    ema = EMABaseline(alpha=ema_alpha)
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="single-stage-init"
    )

    all_metrics = []
    print("Starting single-stage multi-turn GRPO training")
    print("reward = format_r + prop_r + prompt-local repetition_penalty")
    print(
        f"sampling temperature={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}, "
        f"trajectories_per_prompt={trajectories_per_prompt}, refresh_sampling_every={refresh_sampling_every}"
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

    for iteration in range(num_iterations):
        t0 = time.time()

        prompt_indices = rng.choice(len(raw_data), size=prompts_per_iter, replace=False)
        all_datums = []
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
                datums, rewards, details = run_mode_b(
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

            all_datums.extend(datums)
            all_rewards.extend(rewards)
            all_details.append(details)

        elapsed = time.time() - t0

        nte_values = _flatten_numeric(all_details, "nte_values")
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
        mean_nte = _mean_or_none(nte_values)
        std_nte = _std_or_none(nte_values)
        mean_final_qed_error = _mean_or_none(final_qed_errors)
        mean_final_logp_error = _mean_or_none(final_logp_errors)
        mean_final_mw_error = _mean_or_none(final_mw_errors)

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
                    "      nte | "
                    f"mean={_fmt_optional(mean_nte)} std={_fmt_optional(std_nte)} "
                    f"valid={num_final_valid} missing={num_final_missing} | "
                    f"|dQED|={_fmt_optional(mean_final_qed_error)} | "
                    f"|dLogP|={_fmt_optional(mean_final_logp_error)} | "
                    f"|dMW|={_fmt_optional(mean_final_mw_error, digits=2)}"
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
                            "dropped_zero_adv": dropped,
                            "mean_reward_total": mean_reward_total,
                            "mean_reward_format": mean_reward_format,
                            "mean_reward_property": mean_reward_property,
                            "min_reward_property": min_reward_property,
                            "max_reward_property": max_reward_property,
                            "mean_reward_repetition": mean_reward_repetition,
                            "mean_nte": mean_nte,
                            "std_nte": std_nte,
                            "num_final_valid": num_final_valid,
                            "num_final_missing": num_final_missing,
                            "num_terminal_total": num_terminal_total,
                            "terminal_valid_rate": terminal_valid_rate,
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
                            "sampling_temperature": float(temperature),
                            "sampling_top_p": float(top_p),
                            "sampling_top_k": int(top_k),
                            "trajectories_per_prompt": int(trajectories_per_prompt),
                            "refresh_sampling_every": int(refresh_sampling_every),
                            "time": elapsed,
                        }
                    )
            continue

        fwdbwd = training_client.forward_backward(
            all_datums,
            loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
        )
        optim = training_client.optim_step(types.AdamParams(learning_rate=learning_rate))
        fwdbwd.result()
        optim.result()

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
            "      nte | "
            f"mean={_fmt_optional(mean_nte)} std={_fmt_optional(std_nte)} "
            f"valid={num_final_valid} missing={num_final_missing} | "
            f"|dQED|={_fmt_optional(mean_final_qed_error)} | "
            f"|dLogP|={_fmt_optional(mean_final_logp_error)} | "
            f"|dMW|={_fmt_optional(mean_final_mw_error, digits=2)}"
        )
        print(
            "      validity | "
            f"terminal={num_final_valid}/{num_terminal_total} "
            f"({100.0 * terminal_valid_rate if terminal_valid_rate is not None else float('nan'):.1f}%) | "
            f"turns={num_turn_valid}/{num_turn_total} "
            f"({100.0 * turn_valid_rate if turn_valid_rate is not None else float('nan'):.1f}%)"
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
            dropped_zero_adv=dropped,
            mean_reward_total=mean_reward_total,
            mean_reward_format=mean_reward_format,
            mean_reward_property=mean_reward_property,
            min_reward_property=min_reward_property,
            max_reward_property=max_reward_property,
            mean_reward_repetition=mean_reward_repetition,
            mean_nte=mean_nte,
            std_nte=std_nte,
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
            ema_baseline=ema.value,
            time=elapsed,
            skipped=False,
            refreshed_sampling_client=bool(should_refresh),
        )
        all_metrics.append(metric)

    print(f"\nTraining complete ({num_iterations} iterations)")
    return all_metrics
