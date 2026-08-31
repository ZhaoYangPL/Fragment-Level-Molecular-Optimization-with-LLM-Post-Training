#!/usr/bin/env python3
"""
Build the v2 four-step SFT dataset with a structured PLAN schema.

This script intentionally does not reuse or rewrite the old 4,200-row dataset.
It starts again from Alan123/qa_dataset_with_qed_logp_mw_grpo via the original
dataset_constuction.py helpers, but writes to datasets_v2/ by default.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors

import dataset_constuction as base


V2_DATASET_VERSION = "v2_structured_plan"
V2_PLAN_SCHEMA = "reasoning_proposed_intent_expected_v2"

PREVIEW_OUTPUT_TRAIN = (
    "datasets_v2/preview_50/"
    "fourstep_qed_logp_mw_sft_train_v2_preview50.jsonl"
)
PREVIEW_OUTPUT_VAL = (
    "datasets_v2/preview_50/"
    "fourstep_qed_logp_mw_sft_val_v2_preview50.jsonl"
)
V2_SPLIT_OFFSET = 0
V2_MANIFEST_PATH: str | None = None
V2_LAST_TRAIN_INDICES: list[int] = []
V2_LAST_VAL_INDICES: list[int] = []
V2_PROGRESS_INTERVAL = 5

HINDSIGHT_PATTERNS = (
    r"\bgold edit\b",
    r"\bgold answer\b",
    r"\bknown answer\b",
    r"\bknown correct\b",
    r"\bfinal molecule\b",
    r"\bfinal properties\b",
    r"\bobserved shift\b",
    r"\bobserved property\b",
    r"\bmeasured property shifts?\b",
    r"\bafter the edit\b",
    r"\bthis correct edit\b",
    r"\breference edit\b",
    r"\brationali[sz]e the answer\b",
)

SMILES_LIKE_PROPOSED_PATTERN = re.compile(r"[=#@\[\]\\/]")

PLAN_SYSTEM_PROMPT_V2 = """You are an expert medicinal chemistry reasoning assistant.
You will be given an optimization request plus internal answer-conditioned
information used only to keep the generated plan consistent with a verified edit.

Your output MUST use exactly these three lines and no extra text:
Reasoning: [2-3 concise sentences of chemical mechanism, written as pre-edit reasoning]
Proposed edit: [one sentence naming the fragment-level edit]
Expected effect: QED will [increase/decrease], LogP will [increase/decrease], MW will [increase/decrease]

Requirements:
- Do not include <PLAN> tags; the caller will add them.
- Do not use bullets, numbering, markdown, or section headings other than the exact three labels.
- Reasoning must sound like deciding what to do before writing EDIT, not explaining an answer afterward.
- Proposed edit should describe the intended fragment-level operation, affected fragment role, and structural motif, but must not copy full SMILES.
- Never mention gold edit, known answer, final molecule, final properties, observed shifts, measured shifts, reference edit, or after-the-edit evidence.
- Use concrete chemistry when relevant: polarity, heteroatoms, halogens, aromaticity, scaffold bulk, ring rigidity, hydrogen bonding, lipophilicity, or charge.
"""


def _collapse_spaces(text: str) -> str:
    return " ".join(str(text or "").replace("\n", " ").split()).strip()


def _sentence_count(text: str) -> int:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return len(parts)


def _trim_to_three_sentences(text: str) -> str:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    if len(parts) <= 3:
        return text.strip()
    return " ".join(parts[:3])


def _contains_hindsight(text: str) -> bool:
    lowered = _collapse_spaces(text).lower()
    return any(re.search(pattern, lowered) for pattern in HINDSIGHT_PATTERNS)


def _direction_for(context: base.ExampleContext, prop: str) -> str:
    delta = float(context.actual_deltas[prop])
    if abs(delta) < 1e-12:
        return "increase" if context.parsed_prompt.targets[prop].direction == "higher" else "decrease"
    return "increase" if delta > 0 else "decrease"


def expected_effect_line(context: base.ExampleContext) -> str:
    return (
        "Expected effect: "
        f"QED will {_direction_for(context, 'QED')}, "
        f"LogP will {_direction_for(context, 'LogP')}, "
        f"MW will {_direction_for(context, 'MW')}"
    )


def _has_smiles_like_proposed_text(text: str) -> bool:
    return bool(SMILES_LIKE_PROPOSED_PATTERN.search(text or ""))


def _canonical_fragment(fragment: str | None) -> str | None:
    if not fragment:
        return None
    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _starting_fragment_label(context: base.ExampleContext, fragment: str | None) -> str:
    if not fragment:
        return "the selected starting fragment"
    fragments = context.parsed_prompt.fragments
    for idx, candidate in enumerate(fragments, start=1):
        if candidate == fragment:
            return f"starting fragment {idx}"

    target_can = _canonical_fragment(fragment)
    if target_can:
        for idx, candidate in enumerate(fragments, start=1):
            if _canonical_fragment(candidate) == target_can:
                return f"starting fragment {idx}"
    return "the selected starting fragment"


def _fragment_feature_phrase(fragment: str | None) -> str:
    """Describe a fragment without copying full SMILES syntax into PLAN."""
    if not fragment:
        return "fragment level"

    mol = Chem.MolFromSmiles(fragment)
    if mol is None:
        return "fragment level"

    heavy_atoms = mol.GetNumHeavyAtoms()
    heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    rings = Lipinski.RingCount(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    halogens = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in {9, 17, 35, 53})
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    features: list[str] = []
    if heavy_atoms <= 3:
        features.append("small")
    elif heavy_atoms >= 16:
        features.append("very bulky")
    elif heavy_atoms >= 10:
        features.append("bulky")
    if aromatic_rings:
        features.append("aromatic")
    elif rings:
        features.append("ring containing")
    if hba or hbd:
        features.append("hydrogen bonding")
    if heteroatoms:
        features.append("heteroatom rich" if heteroatoms >= 2 else "heteroatom containing")
    if halogens:
        features.append("halogenated")
    if formal_charge:
        features.append("charged")

    if not features:
        features.append("carbon rich")
    return " ".join(dict.fromkeys(features[:5]))


def _fragment_descriptor(fragment: str | None, role: str) -> str:
    return f"{role} {_fragment_feature_phrase(fragment)} fragment"


def _target_direction_phrase(context: base.ExampleContext) -> str:
    parts = []
    for prop in ["QED", "LogP", "MW"]:
        direction = context.parsed_prompt.targets[prop].direction
        word = "higher" if direction == "higher" else "lower"
        parts.append(f"{word} {prop}")
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _descriptive_proposed_edit_line(context: base.ExampleContext) -> str:
    edit = context.parsed_edit
    if edit.action_type == "Replace":
        source_label = _starting_fragment_label(context, edit.old_fragment)
        old_features = _fragment_feature_phrase(edit.old_fragment)
        new_features = _fragment_feature_phrase(edit.new_fragment)
        sentence = (
            f"Replace {source_label}, the {old_features} motif, with a "
            f"{new_features} replacement motif to target {_target_direction_phrase(context)}."
        )
    elif edit.action_type == "Add":
        frag_features = _fragment_feature_phrase(edit.fragment)
        sentence = (
            f"Add a {frag_features} motif to target "
            f"{_target_direction_phrase(context)}."
        )
    elif edit.action_type == "Remove":
        source_label = _starting_fragment_label(context, edit.fragment)
        frag_features = _fragment_feature_phrase(edit.fragment)
        sentence = (
            f"Remove {source_label}, the {frag_features} motif, to target "
            f"{_target_direction_phrase(context)}."
        )
    else:
        sentence = f"Apply one {edit.action_type.lower()} edit at fragment level."
    return f"Proposed edit: {sentence}"


def _raw_proposed_edit_line(context: base.ExampleContext) -> str:
    edit = context.parsed_edit
    if edit.action_type == "Replace" and edit.old_fragment and edit.new_fragment:
        sentence = f"Replace {edit.old_fragment} with {edit.new_fragment}."
    elif edit.action_type == "Add" and edit.fragment:
        sentence = f"Add {edit.fragment}."
    elif edit.action_type == "Remove" and edit.fragment:
        sentence = f"Remove {edit.fragment}."
    else:
        sentence = f"Apply one {edit.action_type.lower()} edit at fragment level."
    return f"Proposed edit: {sentence}"


def proposed_edit_line(context: base.ExampleContext) -> str:
    raw_line = _raw_proposed_edit_line(context)
    if _has_smiles_like_proposed_text(raw_line):
        return _descriptive_proposed_edit_line(context)
    return raw_line


def _fragment_action_phrase(context: base.ExampleContext) -> str:
    edit = context.parsed_edit
    if edit.action_type == "Replace" and edit.old_fragment and edit.new_fragment:
        return (
            f"Replacing the {_fragment_descriptor(edit.old_fragment, 'source')} "
            f"with the {_fragment_descriptor(edit.new_fragment, 'replacement')}"
        )
    if edit.action_type == "Add" and edit.fragment:
        return f"Adding the {_fragment_descriptor(edit.fragment, 'new')}"
    if edit.action_type == "Remove" and edit.fragment:
        return f"Removing the {_fragment_descriptor(edit.fragment, 'selected')}"
    return f"The {edit.action_type.lower()} operation"


def _feature_change_summary(context: base.ExampleContext) -> str:
    mol_in = Chem.MolFromSmiles(context.parsed_prompt.input_smiles)
    mol_out = Chem.MolFromSmiles(context.parsed_edit.target_smiles)
    if mol_in is None or mol_out is None:
        return "size, polarity, and scaffold character"

    def summarize(mol: Chem.Mol) -> dict[str, float]:
        return {
            "heteroatoms": float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "hba": float(Lipinski.NumHAcceptors(mol)),
            "hbd": float(Lipinski.NumHDonors(mol)),
            "rings": float(Lipinski.RingCount(mol)),
            "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "halogens": float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in {9, 17, 35, 53})),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        }

    before = summarize(mol_in)
    after = summarize(mol_out)
    clauses: list[str] = []

    def add_if_changed(key: str, up: str, down: str, threshold: float = 0.0) -> None:
        delta = after[key] - before[key]
        if delta > threshold:
            clauses.append(up)
        elif delta < -threshold:
            clauses.append(down)

    add_if_changed("heteroatoms", "heteroatom density", "heteroatom density")
    add_if_changed("hba", "hydrogen-bond acceptor capacity", "hydrogen-bond acceptor capacity")
    add_if_changed("hbd", "hydrogen-bond donor capacity", "hydrogen-bond donor capacity")
    add_if_changed("halogens", "halogen substitution", "halogen substitution")
    add_if_changed("aromatic_rings", "aromatic surface", "aromatic surface")
    add_if_changed("rings", "ring rigidity", "ring rigidity")
    add_if_changed("fraction_csp3", "saturation", "saturation", threshold=0.10)

    if not clauses:
        return "size, polarity, and scaffold character"
    return ", ".join(dict.fromkeys(clauses[:3]))


def fallback_reasoning_line(context: base.ExampleContext) -> str:
    action_phrase = _fragment_action_phrase(context)
    features = _feature_change_summary(context)
    return (
        "Reasoning: "
        "To satisfy the requested QED, LogP, and MW directions, I would focus on the "
        "fragment-level change that most directly controls molecular size, polarity, "
        "and lipophilicity. "
        f"{action_phrase} should adjust {features}, giving a plausible chemical basis "
        "for the expected movement of all three properties."
    )


def build_fallback_plan(context: base.ExampleContext) -> str:
    return "\n".join(
        [
            fallback_reasoning_line(context),
            proposed_edit_line(context),
            expected_effect_line(context),
        ]
    )


def _extract_label_value(text: str, label: str) -> str | None:
    match = re.search(
        rf"{re.escape(label)}\s*:\s*(.+?)(?=\n(?:Reasoning|Proposed edit|Expected effect)\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    value = _collapse_spaces(match.group(1))
    return value or None


def _clean_reasoning(raw_text: str, context: base.ExampleContext) -> str:
    cleaned = base.strip_wrapping_artifacts(raw_text)
    reasoning = _extract_label_value(cleaned, "Reasoning")
    if reasoning is None:
        # Do not accept unlabelled model prose in v2; fallback is safer and consistent.
        return fallback_reasoning_line(context)

    reasoning = _collapse_spaces(reasoning)
    reasoning = re.sub(r"^(Reasoning\s*:\s*)+", "", reasoning, flags=re.IGNORECASE).strip()
    if (
        not reasoning
        or len(reasoning) < 60
        or _contains_hindsight(reasoning)
        or _has_smiles_like_proposed_text(reasoning)
    ):
        return fallback_reasoning_line(context)

    reasoning = _trim_to_three_sentences(reasoning)
    if _sentence_count(reasoning) < 2:
        return fallback_reasoning_line(context)
    return f"Reasoning: {reasoning}"


def normalize_plan_text(raw_text: str, context: base.ExampleContext) -> str:
    plan = "\n".join(
        [
            _clean_reasoning(raw_text, context),
            proposed_edit_line(context),
            expected_effect_line(context),
        ]
    )
    errors = validate_structured_plan(plan, context)
    if errors:
        raise ValueError(f"Internal v2 PLAN validation failed: {errors}")
    return plan


def build_plan_user_prompt(context: base.ExampleContext) -> str:
    input_props = ", ".join(
        f"{prop}={base.format_value(prop, context.input_props[prop])}"
        for prop in ["QED", "LogP", "MW"]
    )
    output_props = ", ".join(
        f"{prop}={base.format_value(prop, context.output_props[prop])}"
        for prop in ["QED", "LogP", "MW"]
    )
    target_summary = "; ".join(
        (
            f"{prop}: {target.direction} by {base.format_delta(prop, target.delta)} "
            f"(target {base.format_value(prop, target.target_value)})"
        )
        for prop, target in context.parsed_prompt.targets.items()
    )
    actual_shift_summary = "; ".join(
        f"{prop}: {_direction_for(context, prop)}"
        for prop in ["QED", "LogP", "MW"]
    )
    return "\n".join(
        [
            "Optimization request:",
            context.parsed_prompt.input_text,
            "",
            "Internal consistency context; do not mention this context in the output:",
            f"Starting molecule: {context.parsed_prompt.input_smiles}",
            f"Starting fragments: {context.parsed_prompt.fragments}",
            f"Starting properties: {input_props}",
            f"Edit to plan toward: {context.parsed_edit.output_text}",
            f"Resulting molecule for property computation: {context.parsed_edit.target_smiles}",
            f"Resulting properties for direction computation: {output_props}",
            f"Requested targets: {target_summary}",
            f"Computed effect directions: {actual_shift_summary}",
            "",
            "Write the PLAN as exactly three lines in this order:",
            "Reasoning: ...",
            "Proposed edit: ...",
            "Expected effect: ...",
        ]
    )


def _extract_plan_from_completion(completion: str) -> str:
    match = re.search(r"<PLAN>\s*(.*?)\s*</PLAN>", completion or "", flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def validate_structured_plan(plan_text: str, context: base.ExampleContext) -> list[str]:
    errors: list[str] = []
    lines = [line.strip() for line in (plan_text or "").splitlines() if line.strip()]
    expected_prefixes = ["Reasoning:", "Proposed edit:", "Expected effect:"]
    if len(lines) != 3:
        errors.append(f"expected exactly 3 non-empty PLAN lines, got {len(lines)}")
        return errors
    for line, prefix in zip(lines, expected_prefixes, strict=True):
        if not line.lower().startswith(prefix.lower()):
            errors.append(f"line must start with {prefix}")

    if _contains_hindsight(lines[0]):
        errors.append("reasoning contains hindsight leakage")
    if _has_smiles_like_proposed_text(lines[0]):
        errors.append("reasoning contains SMILES-like syntax")
    if _sentence_count(lines[0].split(":", 1)[1].strip()) < 2:
        errors.append("reasoning must contain at least two sentences")

    proposed = lines[1]
    edit = context.parsed_edit
    if _has_smiles_like_proposed_text(proposed):
        errors.append("proposed edit contains SMILES-like syntax")
    if edit.action_type.lower() not in proposed.lower():
        errors.append("proposed edit action does not match EDIT action")
    if edit.action_type == "Replace":
        has_raw_fragments = (
            bool(edit.old_fragment and edit.old_fragment in proposed)
            and bool(edit.new_fragment and edit.new_fragment in proposed)
        )
        has_descriptive_fragments = (
            bool(
                re.search(
                    r"\bstarting fragment\b|\bselected starting fragment\b|\bsource\b|\bselected\b",
                    proposed,
                    flags=re.IGNORECASE,
                )
            )
            and bool(re.search(r"\breplacement\b|\btarget\b|\bmotif\b", proposed, flags=re.IGNORECASE))
        )
        if not has_raw_fragments and not has_descriptive_fragments:
            errors.append("proposed edit missing replace fragment description")
    elif edit.action_type in {"Add", "Remove"} and edit.fragment and edit.fragment not in proposed:
        has_descriptive_fragment = bool(
            re.search(r"\bfragment\b|\bmotif\b", proposed, flags=re.IGNORECASE)
        )
        if not has_descriptive_fragment:
            errors.append("proposed edit missing fragment description")

    expected = lines[2].lower()
    for prop in ["QED", "LogP", "MW"]:
        direction = _direction_for(context, prop)
        pattern = rf"{prop.lower()}\s+will\s+{direction}"
        if not re.search(pattern, expected):
            errors.append(f"expected effect missing {prop} {direction}")
    return errors


_ORIGINAL_BUILD_RECORD = base.build_record


def build_record(context: base.ExampleContext, plan_text: str) -> dict[str, Any]:
    structured_plan = normalize_plan_text(plan_text, context)
    record = _ORIGINAL_BUILD_RECORD(context, structured_plan)
    record["dataset_version"] = V2_DATASET_VERSION
    record["plan_schema"] = V2_PLAN_SCHEMA
    record["plan_direction_labels"] = {
        prop: _direction_for(context, prop)
        for prop in ["QED", "LogP", "MW"]
    }
    errors = validate_structured_plan(_extract_plan_from_completion(record["completion"]), context)
    if errors:
        raise ValueError(f"Invalid v2 PLAN record: {errors}")
    return record


def initialize_output(path: Path, resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not resume and path.stat().st_size > 0:
        raise FileExistsError(
            f"Refusing to overwrite existing v2 output: {path}. "
            "Use --resume or choose a new output path."
        )
    path.touch(exist_ok=True)


def parse_v2_extra_args() -> None:
    global V2_SPLIT_OFFSET, V2_MANIFEST_PATH

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--split-offset",
        type=int,
        default=0,
        help="Offset into the shuffled source-index list. Use 4200 for the next shard after 4000+200.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional JSON manifest path for selected source indices.",
    )
    extra_args, remaining = parser.parse_known_args()
    if extra_args.split_offset < 0:
        raise ValueError("--split-offset must be non-negative.")

    V2_SPLIT_OFFSET = int(extra_args.split_offset)
    V2_MANIFEST_PATH = extra_args.manifest_path
    sys.argv = [sys.argv[0], *remaining]


def choose_split_indices(dataset_len: int, num_train: int, num_val: int, seed: int) -> tuple[list[int], list[int]]:
    global V2_LAST_TRAIN_INDICES, V2_LAST_VAL_INDICES

    total_requested = num_train + num_val
    end = V2_SPLIT_OFFSET + total_requested
    if end > dataset_len:
        raise ValueError(
            f"Requested offset={V2_SPLIT_OFFSET} plus {total_requested} samples, "
            f"but dataset only has {dataset_len} rows."
        )

    indices = list(range(dataset_len))
    import random

    random.Random(seed).shuffle(indices)
    shard = indices[V2_SPLIT_OFFSET:end]
    train_indices = shard[:num_train]
    val_indices = shard[num_train:]
    V2_LAST_TRAIN_INDICES = list(train_indices)
    V2_LAST_VAL_INDICES = list(val_indices)
    return train_indices, val_indices


def write_manifest_if_requested() -> None:
    if not V2_MANIFEST_PATH:
        return
    path = Path(V2_MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_version": V2_DATASET_VERSION,
        "plan_schema": V2_PLAN_SCHEMA,
        "split_offset": V2_SPLIT_OFFSET,
        "num_train": len(V2_LAST_TRAIN_INDICES),
        "num_val": len(V2_LAST_VAL_INDICES),
        "train_source_indices": V2_LAST_TRAIN_INDICES,
        "val_source_indices": V2_LAST_VAL_INDICES,
        "next_split_offset": V2_SPLIT_OFFSET + len(V2_LAST_TRAIN_INDICES) + len(V2_LAST_VAL_INDICES),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[v2] manifest saved to {path}")


def install_v2_overrides() -> None:
    base.DEFAULT_OUTPUT_TRAIN = PREVIEW_OUTPUT_TRAIN
    base.DEFAULT_OUTPUT_VAL = PREVIEW_OUTPUT_VAL
    base.DEFAULT_NUM_TRAIN = 50
    base.DEFAULT_NUM_VAL = 0
    base.DEFAULT_CHECKPOINT_EVERY = 50
    base.DEFAULT_MAX_TOKENS = 512
    base.PROGRESS_INTERVAL = V2_PROGRESS_INTERVAL
    base.PLAN_SYSTEM_PROMPT = PLAN_SYSTEM_PROMPT_V2
    base.LOW_SIGNAL_PLAN_PATTERNS = tuple(base.LOW_SIGNAL_PLAN_PATTERNS) + HINDSIGHT_PATTERNS
    base.build_fallback_plan = build_fallback_plan
    base.normalize_plan_text = normalize_plan_text
    base.build_plan_user_prompt = build_plan_user_prompt
    base.build_record = build_record
    base.initialize_output = initialize_output
    base.choose_split_indices = choose_split_indices


def main() -> None:
    parse_v2_extra_args()
    install_v2_overrides()
    base.main()
    write_manifest_if_requested()


if __name__ == "__main__":
    main()
