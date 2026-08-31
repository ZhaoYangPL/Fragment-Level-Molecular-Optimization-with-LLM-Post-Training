#!/usr/bin/env python3
"""
Build a four-step SFT dataset from Alan123/qa_dataset_with_qed_logp_mw_grpo.

The resulting JSONL records contain:
  - prompt/completion for prompt-completion style training
  - messages for chat-style SFT
  - rich metadata for auditability and resumable generation

Usage example:
  ./tinker_env/bin/python dataset_constuction.py --num-train 100 --num-val 20
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
import socket
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


DATASET_NAME = "Alan123/qa_dataset_with_qed_logp_mw_grpo"
DEFAULT_LOCAL_ARROW = (
    "/home/weekend/.cache/huggingface/datasets/"
    "Alan123___qa_dataset_with_qed_logp_mw_grpo/default/0.0.0/"
    "43d5cfb67d027752b62fce04d5f2406824b9dc4f/"
    "qa_dataset_with_qed_logp_mw_grpo-train.arrow"
)
DEFAULT_OUTPUT_TRAIN = "fourstep_qed_logp_mw_sft_train.jsonl"
DEFAULT_OUTPUT_VAL = "fourstep_qed_logp_mw_sft_val.jsonl"
DEFAULT_ONLINE_CACHE_DIR = "/tmp/hf-cache-ai4materials"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 192
DEFAULT_NUM_TRAIN = 4000
DEFAULT_NUM_VAL = 200
PROGRESS_INTERVAL = 25
DEFAULT_CHECKPOINT_EVERY = 500
DEFAULT_PROXY_HOST = "172.18.48.1"
DEFAULT_PROXY_PORT = 7897
DEFAULT_PLAN_INIT_TIMEOUT_SECONDS = 60.0
DEFAULT_PLAN_RESULT_TIMEOUT_SECONDS = 60.0

TARGET_TOLERANCE = {
    "QED": 0.02,
    "LogP": 0.20,
    "MW": 5.0,
}

PROPERTY_FORMATS = {
    "QED": "{:.4f}",
    "LogP": "{:.4f}",
    "MW": "{:.2f}",
}

DELTA_FORMATS = {
    "QED": "{:.4f}",
    "LogP": "{:.4f}",
    "MW": "{:.3f}",
}

SMILES_RE = re.compile(r"<SMILES>(.*?)</SMILES>", re.DOTALL)
FRAGMENTS_RE = re.compile(
    r"which is composed of fragments\s+(\[.*?\])\.\s*Propose",
    re.DOTALL,
)

TARGET_PATTERNS = {
    "QED": re.compile(r"<QED>\s*([-+]?\d*\.?\d+)\s*</QED>\s*(higher|lower)", re.IGNORECASE),
    "LogP": re.compile(
        r"<LogP>\s*([-+]?\d*\.?\d+)\s*</LogP>\s*(higher|lower)",
        re.IGNORECASE,
    ),
    "MW": re.compile(r"<MW>\s*([-+]?\d*\.?\d+)\s*</MW>\s*(higher|lower)", re.IGNORECASE),
}

ACTION_TYPE_RE = re.compile(r"^\s*(Replace|Add|Remove)\b", re.IGNORECASE)
REPLACE_RE = re.compile(
    r"^\s*Replace\s+(.*?)\s+with\s+(.*?)\s+to\s+form\s+<SMILES>",
    re.IGNORECASE | re.DOTALL,
)
ADD_RE = re.compile(r"^\s*Add\s+(.*?)\s+to\s+form\s+<SMILES>", re.IGNORECASE | re.DOTALL)
REMOVE_RE = re.compile(
    r"^\s*Remove\s+(.*?)\s+to\s+form\s+<SMILES>",
    re.IGNORECASE | re.DOTALL,
)

FOURSTEP_SYSTEM_PROMPT = """You are a chemistry assistant for multi-turn molecule optimization.
At each turn, output EXACTLY the following four labeled blocks in order:

<ANALYSE>
Current molecule: <SMILES>...</SMILES>
Fragments: [...]
Current properties: QED=..., LogP=..., MW=...
Requested changes: QED ..., LogP ..., MW ...
</ANALYSE>

<PLAN>
Proposed edit: ...
Expected effect: QED will ..., LogP will ..., MW will ...
Reasoning: ...
</PLAN>

<EDIT>
Replace/Add/Remove ... to form <SMILES>...</SMILES>.
</EDIT>

<VERIFY>
New molecule: <SMILES>...</SMILES>
New properties: QED=..., LogP=..., MW=...
Remaining gaps: ...
All targets met: Yes/No
</VERIFY>"""

PLAN_SYSTEM_PROMPT = """You are an expert medicinal chemistry reasoning assistant.
You will be given:
- an optimization request,
- the starting molecule and its fragments,
- the known gold edit answer,
- the measured properties before and after the edit.

Your job is to reconstruct a plausible, chemically grounded PLAN that could have led
to this known correct edit. Let the reasoning be natural and concise rather than
template-like.

Requirements:
- Return plain text only. Do not include the <PLAN> or </PLAN> tags; the caller will add them.
- Use 2 to 4 sentences of free-form reasoning.
- Refer to the actual fragment change from the gold edit.
- Explain likely structural causes behind the QED, LogP, and MW movement using
  concrete concepts when relevant: polarity, heteroatoms, halogens, aromaticity,
  ring size/rigidity, scaffold bulk, hydrogen-bonding, lipophilicity, or charge.
- Mention the direction of QED, LogP, and MW explicitly.
- Do not simply restate that the edit matches the target; explain why it might.
- Do not discuss the prompt, task, constraints, or formatting instructions.
- Do not include markdown fences, bullets, numbering, or section headings such as
  'Thinking Process' or 'Analysis'."""

LOW_SIGNAL_PLAN_PATTERNS = (
    r"moves? the measured properties toward the requested directions",
    r"moves? the properties toward the requested directions",
    r"helps satisfy the requested .* change",
    r"this source edit is kept because",
    r"toward the requested directions",
    r"thinking process",
    r"analyze the request",
    r"\bconstraints:\b",
    r"\btask:\b",
    r"\binput:\b",
    r"do not include .*tags",
    r"write only .* block",
)


@dataclass(frozen=True)
class PropertyTarget:
    delta: float
    direction: str
    target_value: float


@dataclass(frozen=True)
class ParsedPrompt:
    input_text: str
    input_smiles: str
    fragments: list[str]
    targets: dict[str, PropertyTarget]


@dataclass(frozen=True)
class ParsedEdit:
    output_text: str
    action_type: str
    target_smiles: str
    old_fragment: str | None = None
    new_fragment: str | None = None
    fragment: str | None = None


@dataclass(frozen=True)
class ExampleContext:
    source_index: int
    parsed_prompt: ParsedPrompt
    parsed_edit: ParsedEdit
    input_props: dict[str, float]
    output_props: dict[str, float]
    actual_deltas: dict[str, float]
    met_flags: dict[str, bool]
    remaining_gaps: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct a four-step SFT dataset for QED/LogP/MW optimization.",
    )
    parser.add_argument("--local-arrow", default=DEFAULT_LOCAL_ARROW)
    parser.add_argument("--output-train", default=DEFAULT_OUTPUT_TRAIN)
    parser.add_argument("--output-val", default=DEFAULT_OUTPUT_VAL)
    parser.add_argument("--num-train", type=int, default=DEFAULT_NUM_TRAIN)
    parser.add_argument("--num-val", type=int, default=DEFAULT_NUM_VAL)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Write a numbered checkpoint copy of the JSONL after every N saved records. Set to 0 to disable.",
    )
    parser.add_argument("--proxy-host", default=DEFAULT_PROXY_HOST)
    parser.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    parser.add_argument(
        "--disable-proxy-probe",
        action="store_true",
        help="Do not probe and auto-configure http_proxy/https_proxy.",
    )
    parser.add_argument(
        "--plan-init-timeout",
        type=float,
        default=DEFAULT_PLAN_INIT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--plan-result-timeout",
        type=float,
        default=DEFAULT_PLAN_RESULT_TIMEOUT_SECONDS,
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_train < 0 or args.num_val < 0:
        raise ValueError("--num-train and --num-val must be non-negative.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be non-negative.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative.")
    if args.proxy_port <= 0:
        raise ValueError("--proxy-port must be positive.")
    if args.plan_init_timeout <= 0:
        raise ValueError("--plan-init-timeout must be positive.")
    if args.plan_result_timeout <= 0:
        raise ValueError("--plan-result-timeout must be positive.")


def test_tcp_port(host: str, port: int, timeout_seconds: float = 2.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_seconds)
        return sock.connect_ex((host, port)) == 0


def configure_proxy_env(
    proxy_host: str,
    proxy_port: int,
    disable_probe: bool,
) -> None:
    existing_http = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
    existing_https = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    if existing_http or existing_https:
        print(
            "[proxy] Using existing proxy environment variables "
            f"(http_proxy={'set' if existing_http else 'missing'}, "
            f"https_proxy={'set' if existing_https else 'missing'})."
        )
        return

    if disable_probe:
        print("[proxy] Proxy probe disabled; leaving proxy environment unchanged.")
        return

    proxy_url = f"http://{proxy_host}:{proxy_port}"
    if test_tcp_port(proxy_host, proxy_port):
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        print(f"[proxy] Auto-configured http_proxy/https_proxy to {proxy_url}")
    else:
        print(
            "[proxy] No reachable proxy detected at "
            f"{proxy_host}:{proxy_port}; continuing without proxy."
        )


def load_raw_dataset(local_arrow: str) -> Dataset:
    arrow_path = Path(local_arrow)
    if arrow_path.exists():
        print(f"[load] Reading dataset from local arrow: {arrow_path}")
        return Dataset.from_file(str(arrow_path))

    cache_dir = Path(DEFAULT_ONLINE_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[load] Local arrow not found. Falling back to load_dataset "
        f"with cache_dir={cache_dir}"
    )
    return load_dataset(DATASET_NAME, split="train", cache_dir=str(cache_dir))


def extract_smiles(text: str) -> str | None:
    if not text:
        return None
    match = SMILES_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def parse_fragments(prompt_text: str) -> list[str]:
    match = FRAGMENTS_RE.search(prompt_text)
    if not match:
        return []
    try:
        fragments = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(fragments, list):
        return []
    return [str(fragment).strip() for fragment in fragments]


def compute_properties(mol: Chem.Mol) -> dict[str, float]:
    return {
        "QED": round(float(QED.qed(mol)), 4),
        "LogP": round(float(Crippen.MolLogP(mol)), 4),
        "MW": round(float(Descriptors.ExactMolWt(mol)), 2),
    }


def parse_prompt(prompt_text: str) -> ParsedPrompt:
    input_smiles = extract_smiles(prompt_text)
    if input_smiles is None:
        raise ValueError("Prompt is missing an input <SMILES> tag.")

    fragments = parse_fragments(prompt_text)
    targets: dict[str, PropertyTarget] = {}
    mol_in = Chem.MolFromSmiles(input_smiles)
    if mol_in is None:
        raise ValueError("Prompt input SMILES is chemically invalid.")

    props_in = compute_properties(mol_in)

    for prop_name, pattern in TARGET_PATTERNS.items():
        match = pattern.search(prompt_text)
        if not match:
            raise ValueError(f"Prompt is missing target spec for {prop_name}.")
        delta = float(match.group(1))
        direction = match.group(2).lower()
        sign = 1.0 if direction == "higher" else -1.0
        target_value = props_in[prop_name] + sign * delta
        if prop_name == "QED":
            target_value = min(0.99, max(0.01, target_value))
        targets[prop_name] = PropertyTarget(
            delta=delta,
            direction=direction,
            target_value=round(target_value, 4 if prop_name != "MW" else 2),
        )

    return ParsedPrompt(
        input_text=prompt_text,
        input_smiles=input_smiles,
        fragments=fragments,
        targets=targets,
    )


def parse_edit(output_text: str) -> ParsedEdit:
    action_match = ACTION_TYPE_RE.match(output_text or "")
    if not action_match:
        raise ValueError("Output does not start with Replace/Add/Remove.")
    action_type = action_match.group(1).capitalize()

    target_smiles = extract_smiles(output_text)
    if target_smiles is None:
        raise ValueError("Output is missing a target <SMILES> tag.")

    replace_match = REPLACE_RE.match(output_text)
    add_match = ADD_RE.match(output_text)
    remove_match = REMOVE_RE.match(output_text)

    old_fragment: str | None = None
    new_fragment: str | None = None
    fragment: str | None = None
    if replace_match:
        old_fragment = replace_match.group(1).strip()
        new_fragment = replace_match.group(2).strip()
    elif add_match:
        fragment = add_match.group(1).strip()
    elif remove_match:
        fragment = remove_match.group(1).strip()

    return ParsedEdit(
        output_text=output_text.strip(),
        action_type=action_type,
        target_smiles=target_smiles,
        old_fragment=old_fragment,
        new_fragment=new_fragment,
        fragment=fragment,
    )


def compute_example_context(source_index: int, input_text: str, output_text: str) -> ExampleContext:
    parsed_prompt = parse_prompt(input_text)
    parsed_edit = parse_edit(output_text)

    mol_in = Chem.MolFromSmiles(parsed_prompt.input_smiles)
    mol_out = Chem.MolFromSmiles(parsed_edit.target_smiles)
    if mol_in is None:
        raise ValueError("Input molecule is invalid after parsing.")
    if mol_out is None:
        raise ValueError("Output molecule is invalid after parsing.")

    input_props = compute_properties(mol_in)
    output_props = compute_properties(mol_out)

    actual_deltas: dict[str, float] = {}
    remaining_gaps: dict[str, float] = {}
    met_flags: dict[str, bool] = {}
    for prop_name, target in parsed_prompt.targets.items():
        actual_delta = output_props[prop_name] - input_props[prop_name]
        remaining_gap = target.target_value - output_props[prop_name]
        actual_deltas[prop_name] = round(actual_delta, 4 if prop_name != "MW" else 2)
        remaining_gaps[prop_name] = round(
            remaining_gap,
            4 if prop_name != "MW" else 2,
        )
        met_flags[prop_name] = abs(remaining_gap) <= TARGET_TOLERANCE[prop_name]

    return ExampleContext(
        source_index=source_index,
        parsed_prompt=parsed_prompt,
        parsed_edit=parsed_edit,
        input_props=input_props,
        output_props=output_props,
        actual_deltas=actual_deltas,
        met_flags=met_flags,
        remaining_gaps=remaining_gaps,
    )


def format_value(prop_name: str, value: float) -> str:
    return PROPERTY_FORMATS[prop_name].format(value)


def format_delta(prop_name: str, value: float) -> str:
    return DELTA_FORMATS[prop_name].format(abs(value))


def build_analyse_block(context: ExampleContext) -> str:
    fragments_repr = repr(context.parsed_prompt.fragments)
    current_props = ", ".join(
        f"{prop}={format_value(prop, context.input_props[prop])}"
        for prop in ["QED", "LogP", "MW"]
    )
    requested_changes = "; ".join(
        (
            f"{prop} {target.direction} by {format_delta(prop, target.delta)} "
            f"to target {format_value(prop, target.target_value)}"
        )
        for prop, target in context.parsed_prompt.targets.items()
    )
    return "\n".join(
        [
            "<ANALYSE>",
            f"Current molecule: <SMILES>{context.parsed_prompt.input_smiles}</SMILES>",
            f"Fragments: {fragments_repr}",
            f"Current properties: {current_props}",
            f"Requested changes: {requested_changes}",
            "</ANALYSE>",
        ]
    )


def describe_expected_effect(context: ExampleContext) -> str:
    parts = []
    for prop in ["QED", "LogP", "MW"]:
        delta = context.actual_deltas[prop]
        direction = "increase" if delta >= 0 else "decrease"
        parts.append(f"{prop} will {direction} by ~{format_delta(prop, delta)}")
    return ", ".join(parts)


def fallback_proposed_edit(context: ExampleContext) -> str:
    edit = context.parsed_edit
    if edit.action_type == "Replace" and edit.old_fragment and edit.new_fragment:
        return f"Replace {edit.old_fragment} with {edit.new_fragment}."
    if edit.action_type == "Add" and edit.fragment:
        return f"Add {edit.fragment} to the current molecule."
    if edit.action_type == "Remove" and edit.fragment:
        return f"Remove {edit.fragment} from the current molecule."
    return f"Apply the source {edit.action_type.lower()} edit at fragment level."


def fallback_reasoning(context: ExampleContext) -> str:
    mol_in = Chem.MolFromSmiles(context.parsed_prompt.input_smiles)
    mol_out = Chem.MolFromSmiles(context.parsed_edit.target_smiles)
    if mol_in is None or mol_out is None:
        return (
            "The fragment swap is chosen because the new motif changes the molecule's "
            "polarity, bulk, and scaffold character in a way that is consistent with "
            "the observed property shifts."
        )

    def summarize_mol(mol: Chem.Mol) -> dict[str, float]:
        return {
            "heavy_atoms": float(Lipinski.HeavyAtomCount(mol)),
            "heteroatoms": float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "hba": float(Lipinski.NumHAcceptors(mol)),
            "hbd": float(Lipinski.NumHDonors(mol)),
            "rings": float(Lipinski.RingCount(mol)),
            "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "halogens": float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in {9, 17, 35, 53})),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        }

    desc_in = summarize_mol(mol_in)
    desc_out = summarize_mol(mol_out)
    change_clauses: list[str] = []

    def maybe_add(name: str, positive_text: str, negative_text: str, threshold: float = 0.0) -> None:
        delta = desc_out[name] - desc_in[name]
        if delta > threshold:
            change_clauses.append(positive_text)
        elif delta < -threshold:
            change_clauses.append(negative_text)

    maybe_add("heteroatoms", "introduces a more heteroatom-rich environment", "removes heteroatom density")
    maybe_add("hba", "adds hydrogen-bond acceptors", "reduces hydrogen-bond acceptors")
    maybe_add("hbd", "adds hydrogen-bond donors", "reduces hydrogen-bond donors")
    maybe_add("halogens", "adds halogen substitution", "removes halogen substitution")
    maybe_add("aromatic_rings", "increases aromatic surface", "reduces aromatic surface")
    maybe_add("rings", "makes the scaffold more ring-rich and rigid", "makes the scaffold less ring-dense")
    maybe_add("fraction_csp3", "makes the scaffold more saturated", "makes the scaffold less saturated", threshold=0.10)

    size_delta = context.output_props["MW"] - context.input_props["MW"]
    lipophilicity_delta = context.output_props["LogP"] - context.input_props["LogP"]

    size_phrase = (
        "a larger, heavier scaffold"
        if size_delta > 0
        else "a smaller, lighter scaffold"
    )
    lipophilicity_phrase = (
        "a more lipophilic product"
        if lipophilicity_delta > 0
        else "a less lipophilic product"
    )

    detail_phrase = ""
    if change_clauses:
        detail_phrase = " In particular, it " + ", ".join(change_clauses[:3]) + "."

    return (
        f"The edit shifts the molecule toward {size_phrase} and {lipophilicity_phrase}, "
        "which provides a concrete structural basis for the observed MW and LogP change."
        f"{detail_phrase} That same change in scaffold balance is also consistent with the "
        "observed QED movement."
    )


def build_fallback_plan(context: ExampleContext) -> str:
    edit_sentence = fallback_proposed_edit(context)
    effect_sentence = describe_expected_effect(context) + "."
    reasoning_sentence = fallback_reasoning(context)
    return f"{edit_sentence} {effect_sentence} {reasoning_sentence}"


def strip_wrapping_artifacts(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"```(?:text|markdown)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r"</?PLAN>", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned


def extract_named_line(text: str, label: str) -> str | None:
    match = re.search(
        rf"{re.escape(label)}\s*:\s*(.+?)(?=\n[A-Za-z][A-Za-z ]*\s*:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    value = " ".join(match.group(1).strip().split())
    if not value:
        return None
    return f"{label}: {value}"


def normalize_plan_text(raw_text: str, context: ExampleContext) -> str:
    cleaned = strip_wrapping_artifacts(raw_text)
    fallback = build_fallback_plan(context)

    if not cleaned:
        return fallback

    normalized = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = "\n".join(line.strip(" \t") for line in normalized.splitlines()).strip()
    if not normalized:
        return fallback

    collapsed = " ".join(normalized.lower().split())
    if len(collapsed) < 48:
        return fallback
    if any(re.search(pattern, collapsed) for pattern in LOW_SIGNAL_PLAN_PATTERNS):
        return fallback
    return normalized


def build_plan_user_prompt(context: ExampleContext) -> str:
    input_props = ", ".join(
        f"{prop}={format_value(prop, context.input_props[prop])}"
        for prop in ["QED", "LogP", "MW"]
    )
    output_props = ", ".join(
        f"{prop}={format_value(prop, context.output_props[prop])}"
        for prop in ["QED", "LogP", "MW"]
    )
    target_summary = "; ".join(
        (
            f"{prop}: {target.direction} by {format_delta(prop, target.delta)} "
            f"(target {format_value(prop, target.target_value)})"
        )
        for prop, target in context.parsed_prompt.targets.items()
    )
    actual_shift_summary = "; ".join(
        f"{prop}: {('increase' if context.actual_deltas[prop] >= 0 else 'decrease')} by "
        f"{format_delta(prop, context.actual_deltas[prop])}"
        for prop in ["QED", "LogP", "MW"]
    )
    return "\n".join(
        [
            "Optimization request:",
            context.parsed_prompt.input_text,
            "",
            f"Starting molecule: {context.parsed_prompt.input_smiles}",
            f"Starting fragments: {context.parsed_prompt.fragments}",
            f"Starting properties: {input_props}",
            f"Gold edit answer: {context.parsed_edit.output_text}",
            f"Final molecule: {context.parsed_edit.target_smiles}",
            f"Final properties: {output_props}",
            f"Requested targets: {target_summary}",
            f"Measured property shifts after the gold edit: {actual_shift_summary}",
            (
                "Write a compact but substantive PLAN that explains the causal reasoning "
                "behind this edit. Treat the final edit as correct and reconstruct the "
                "intermediate thought process a strong chemist might use before executing it."
            ),
            (
                "Avoid generic statements like 'this moves the properties in the right "
                "direction'. Instead, explain which structural features of the replaced "
                "and introduced fragments plausibly drive the observed QED, LogP, and MW shifts."
            ),
        ]
    )


class TinkerPlanGenerator:
    def __init__(
        self,
        base_model: str,
        temperature: float,
        max_tokens: int,
        init_timeout_seconds: float,
        result_timeout_seconds: float,
    ) -> None:
        self.base_model = base_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.init_timeout_seconds = init_timeout_seconds
        self.result_timeout_seconds = result_timeout_seconds
        self._ready = False
        self._disabled_reason: str | None = None
        self._sampling_client: Any | None = None
        self._renderer: Any | None = None
        self._tokenizer: Any | None = None
        self._types: Any | None = None

    @staticmethod
    def _run_with_timeout(func: Any, timeout_seconds: float) -> Any:
        result: dict[str, Any] = {}

        def worker() -> None:
            try:
                result["value"] = func()
            except Exception as exc:  # pragma: no cover - environment dependent
                result["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout_seconds)
        if thread.is_alive():
            raise TimeoutError(f"operation exceeded {timeout_seconds:.1f}s")
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def _ensure_ready(self) -> None:
        if self._ready or self._disabled_reason is not None:
            return
        try:
            self._run_with_timeout(self._initialize_clients, self.init_timeout_seconds)
            self._ready = True
        except Exception as exc:  # pragma: no cover - environment dependent
            self._disabled_reason = f"Tinker initialization failure: {exc}"

    def _initialize_clients(self) -> None:
        import tinker
        from tinker import types
        from tinker_cookbook import model_info, renderers, tokenizer_utils

        service_client = tinker.ServiceClient()
        self._sampling_client = service_client.create_sampling_client(
            base_model=self.base_model,
        )
        self._tokenizer = tokenizer_utils.get_tokenizer(self.base_model)
        renderer_candidates = model_info.get_recommended_renderer_names(self.base_model)
        renderer_name = renderer_candidates[0]
        for candidate in renderer_candidates:
            if "disable_thinking" in candidate or candidate.endswith("_instruct"):
                renderer_name = candidate
                break
        self._renderer = renderers.get_renderer(renderer_name, self._tokenizer)
        self._types = types

    def generate_plan(self, context: ExampleContext) -> tuple[str, bool, str | None]:
        self._ensure_ready()
        if not self._ready:
            return build_fallback_plan(context), False, self._disabled_reason

        messages = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": build_plan_user_prompt(context)},
        ]
        try:
            prompt_input = self._renderer.build_generation_prompt(messages)
            result = self._sampling_client.sample(
                prompt=prompt_input,
                sampling_params=self._types.SamplingParams(
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=self._renderer.get_stop_sequences(),
                ),
                num_samples=1,
            ).result(timeout=self.result_timeout_seconds)
            sequence = result.sequences[0]
            raw_text = self._tokenizer.decode(
                sequence.tokens,
                skip_special_tokens=True,
            )
            return normalize_plan_text(raw_text, context), True, None
        except Exception as exc:  # pragma: no cover - environment dependent
            return build_fallback_plan(context), False, str(exc)


def build_verify_block(context: ExampleContext) -> str:
    new_props = ", ".join(
        f"{prop}={format_value(prop, context.output_props[prop])}"
        for prop in ["QED", "LogP", "MW"]
    )
    gap_parts = []
    for prop in ["QED", "LogP", "MW"]:
        gap = context.remaining_gaps[prop]
        tol = TARGET_TOLERANCE[prop]
        tol_str = f"{tol:.2f}"
        if abs(gap) <= tol:
            status = "on target"
        elif gap > 0:
            status = f"need {format_delta(prop, gap)} higher"
        else:
            status = f"need {format_delta(prop, gap)} lower"
        gap_parts.append(
            (
                f"{prop} {status} "
                f"(target={format_value(prop, context.parsed_prompt.targets[prop].target_value)}, "
                f"tol={tol_str})"
            )
        )
    all_met = "Yes" if all(context.met_flags.values()) else "No"
    return "\n".join(
        [
            "<VERIFY>",
            f"New molecule: <SMILES>{context.parsed_edit.target_smiles}</SMILES>",
            f"New properties: {new_props}",
            f"Remaining gaps: {'; '.join(gap_parts)}",
            f"All targets met: {all_met}",
            "</VERIFY>",
        ]
    )


def serialize_targets(targets: dict[str, PropertyTarget]) -> dict[str, dict[str, Any]]:
    return {
        prop: {
            "delta": target.delta,
            "direction": target.direction,
            "target_value": target.target_value,
        }
        for prop, target in targets.items()
    }


def build_record(context: ExampleContext, plan_text: str) -> dict[str, Any]:
    completion = "\n\n".join(
        [
            build_analyse_block(context),
            "<PLAN>\n" + plan_text + "\n</PLAN>",
            "<EDIT>\n" + context.parsed_edit.output_text + "\n</EDIT>",
            build_verify_block(context),
        ]
    )
    messages = [
        {"role": "system", "content": FOURSTEP_SYSTEM_PROMPT},
        {"role": "user", "content": context.parsed_prompt.input_text},
        {"role": "assistant", "content": completion},
    ]
    return {
        "prompt": context.parsed_prompt.input_text,
        "completion": completion,
        "messages": messages,
        "source_index": context.source_index,
        "source_input": context.parsed_prompt.input_text,
        "source_output": context.parsed_edit.output_text,
        "input_smiles": context.parsed_prompt.input_smiles,
        "target_smiles": context.parsed_edit.target_smiles,
        "action_type": context.parsed_edit.action_type,
        "fragments": context.parsed_prompt.fragments,
        "targets": serialize_targets(context.parsed_prompt.targets),
        "input_props": context.input_props,
        "output_props": context.output_props,
        "actual_prop_deltas": context.actual_deltas,
        "target_met": all(context.met_flags.values()),
    }


def load_existing_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indices: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL line in {path} at line {line_number}: {exc}"
                ) from exc
            source_index = payload.get("source_index")
            if isinstance(source_index, int):
                indices.add(source_index)
    return indices


def choose_split_indices(dataset_len: int, num_train: int, num_val: int, seed: int) -> tuple[list[int], list[int]]:
    total_requested = num_train + num_val
    if total_requested > dataset_len:
        raise ValueError(
            f"Requested {total_requested} samples but dataset only has {dataset_len} rows."
        )
    indices = list(range(dataset_len))
    random.Random(seed).shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train : num_train + num_val]
    return train_indices, val_indices


def initialize_output(path: Path, resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not resume:
        path.write_text("", encoding="utf-8")
    elif not path.exists():
        path.touch()


def emit_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_checkpoint_path(path: Path, record_count: int) -> Path:
    checkpoint_dir = path.parent / f"{path.stem}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{path.stem}.{record_count:05d}{path.suffix}"


def write_checkpoint_snapshot(path: Path, record_count: int) -> Path:
    checkpoint_path = build_checkpoint_path(path, record_count)
    shutil.copy2(path, checkpoint_path)
    return checkpoint_path


def summarize_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def build_split(
    *,
    split_name: str,
    indices: list[int],
    dataset: Dataset,
    output_path: Path,
    resume: bool,
    plan_generator: TinkerPlanGenerator,
    sleep_seconds: float,
    checkpoint_every: int,
) -> dict[str, Any]:
    initialize_output(output_path, resume=resume)
    completed_indices = load_existing_indices(output_path) if resume else set()
    persisted_records = len(completed_indices)
    stats: dict[str, Any] = {
        "split": split_name,
        "requested": len(indices),
        "already_present": 0,
        "written": 0,
        "skipped": 0,
        "target_met_count": 0,
        "skip_reasons": Counter(),
        "actions": Counter(),
        "plan_generation": Counter(),
        "plan_fallback_reasons": Counter(),
        "checkpoint_count": 0,
    }

    for position, source_index in enumerate(indices, start=1):
        if source_index in completed_indices:
            stats["already_present"] += 1
            continue

        example = dataset[source_index]
        try:
            context = compute_example_context(
                source_index=source_index,
                input_text=str(example["input"]),
                output_text=str(example["output"]),
            )
        except Exception as exc:
            stats["skipped"] += 1
            stats["skip_reasons"][type(exc).__name__] += 1
            continue

        plan_text, used_tinker, error_text = plan_generator.generate_plan(context)
        if used_tinker:
            stats["plan_generation"]["tinker"] += 1
        else:
            stats["plan_generation"]["fallback"] += 1
            if error_text:
                normalized_error = " ".join(error_text.strip().split())
                stats["plan_fallback_reasons"][normalized_error] += 1

        record = build_record(context, plan_text)
        emit_record(output_path, record)
        stats["written"] += 1
        persisted_records += 1
        stats["actions"][context.parsed_edit.action_type] += 1
        if record["target_met"]:
            stats["target_met_count"] += 1

        if checkpoint_every > 0 and persisted_records % checkpoint_every == 0:
            checkpoint_path = write_checkpoint_snapshot(output_path, persisted_records)
            stats["checkpoint_count"] += 1
            print(
                f"[{split_name}] checkpoint saved at {persisted_records} records -> "
                f"{checkpoint_path}"
            )

        if stats["written"] % PROGRESS_INTERVAL == 0 or position == len(indices):
            print(
                f"[{split_name}] processed={position}/{len(indices)} "
                f"written={stats['written']} skipped={stats['skipped']} "
                f"resume_hits={stats['already_present']}"
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return stats


def print_summary(stats: dict[str, Any], output_path: Path) -> None:
    met_rate = (
        f"{(stats['target_met_count'] / stats['written']):.1%}"
        if stats["written"] > 0
        else "n/a"
    )
    print(f"\n[{stats['split']}] Summary")
    print(f"  output_path: {output_path}")
    print(f"  requested: {stats['requested']}")
    print(f"  already_present: {stats['already_present']}")
    print(f"  written: {stats['written']}")
    print(f"  skipped: {stats['skipped']}")
    print(f"  target_met_rate: {met_rate}")
    print(f"  action_distribution: {summarize_counter(stats['actions'])}")
    print(f"  plan_generation: {summarize_counter(stats['plan_generation'])}")
    print(f"  checkpoints_written: {stats['checkpoint_count']}")
    print(f"  plan_fallback_reasons: {summarize_counter(stats['plan_fallback_reasons'])}")
    print(f"  skip_reasons: {summarize_counter(stats['skip_reasons'])}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_proxy_env(
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        disable_probe=args.disable_proxy_probe,
    )

    dataset = load_raw_dataset(args.local_arrow)
    train_indices, val_indices = choose_split_indices(
        dataset_len=len(dataset),
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
    )
    print(
        f"[split] Dataset rows={len(dataset)} train={len(train_indices)} val={len(val_indices)} "
        f"seed={args.seed}"
    )

    plan_generator = TinkerPlanGenerator(
        base_model=args.base_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        init_timeout_seconds=args.plan_init_timeout,
        result_timeout_seconds=args.plan_result_timeout,
    )

    train_output = Path(args.output_train)
    val_output = Path(args.output_val)

    train_stats = build_split(
        split_name="train",
        indices=train_indices,
        dataset=dataset,
        output_path=train_output,
        resume=args.resume,
        plan_generator=plan_generator,
        sleep_seconds=args.sleep_seconds,
        checkpoint_every=args.checkpoint_every,
    )
    val_stats = build_split(
        split_name="val",
        indices=val_indices,
        dataset=dataset,
        output_path=val_output,
        resume=args.resume,
        plan_generator=plan_generator,
        sleep_seconds=args.sleep_seconds,
        checkpoint_every=args.checkpoint_every,
    )

    print_summary(train_stats, train_output)
    print_summary(val_stats, val_output)


if __name__ == "__main__":
    main()
