#!/usr/bin/env python3
"""
Rewrite only the <PLAN> block of an existing four-step SFT dataset.

This script reads an existing JSONL dataset whose ANALYSE / EDIT / VERIFY blocks
are already trusted, regenerates the PLAN block with Tinker, and writes the
updated samples to a new JSONL file. The source dataset is never overwritten.

Example:
  ./tinker_env/bin/python rewrite_plan_dataset.py \
      --input fourstep_qed_logp_mw_sft_train.jsonl \
      --output fourstep_qed_logp_mw_sft_train_plan_v2.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_constuction import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PLAN_INIT_TIMEOUT_SECONDS,
    DEFAULT_PLAN_RESULT_TIMEOUT_SECONDS,
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
    DEFAULT_TEMPERATURE,
    PROGRESS_INTERVAL,
    TinkerPlanGenerator,
    compute_example_context,
    configure_proxy_env,
)


PLAN_BLOCK_RE = re.compile(r"<PLAN>\s*(.*?)\s*</PLAN>", re.DOTALL | re.IGNORECASE)
EDIT_BLOCK_RE = re.compile(r"<EDIT>\s*(.*?)\s*</EDIT>", re.DOTALL | re.IGNORECASE)


def derive_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}_plan_v2{input_path.suffix}")
    return input_path.with_name(f"{input_path.name}_plan_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate PLAN blocks for an existing four-step SFT dataset with Tinker.",
    )
    parser.add_argument(
        "--input",
        default="fourstep_qed_logp_mw_sft_train.jsonl",
        help="Existing four-step JSONL dataset to rewrite.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="New JSONL path. Defaults to '<input>_plan_v2.jsonl'.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of input rows to process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file and skip already-written source_index rows.",
    )
    parser.add_argument(
        "--fail-on-plan-error",
        action="store_true",
        help="Abort immediately if a row cannot get a Tinker-generated PLAN.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Output path must be different from the source dataset path; refusing to overwrite input.",
        )
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be non-negative.")
    if args.proxy_port <= 0:
        raise ValueError("--proxy-port must be positive.")
    if args.plan_init_timeout <= 0:
        raise ValueError("--plan-init-timeout must be positive.")
    if args.plan_result_timeout <= 0:
        raise ValueError("--plan-result-timeout must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Choose a new path or rerun with --resume.",
        )


def initialize_output(path: Path, resume: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        path.touch(exist_ok=True)


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


def extract_plan_text(completion: str) -> str:
    match = PLAN_BLOCK_RE.search(completion or "")
    if not match:
        raise ValueError("Completion is missing a <PLAN> block.")
    return match.group(1).strip()


def extract_edit_text(completion: str) -> str:
    match = EDIT_BLOCK_RE.search(completion or "")
    if not match:
        raise ValueError("Completion is missing an <EDIT> block.")
    return match.group(1).strip()


def replace_plan_block(completion: str, new_plan_text: str) -> str:
    if not PLAN_BLOCK_RE.search(completion or ""):
        raise ValueError("Completion is missing a <PLAN> block.")
    replacement = "<PLAN>\n" + new_plan_text + "\n</PLAN>"
    return PLAN_BLOCK_RE.sub(lambda _: replacement, completion, count=1)


def update_assistant_message(messages: Any, new_completion: str) -> Any:
    if not isinstance(messages, list):
        return messages

    updated_messages = copy.deepcopy(messages)
    for message in reversed(updated_messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            message["content"] = new_completion
            return updated_messages
    return updated_messages


def resolve_example_context(record: dict[str, Any], line_number: int) -> Any:
    input_text = record.get("source_input") or record.get("prompt")
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError("Record is missing source_input/prompt.")

    output_text = record.get("source_output")
    if not isinstance(output_text, str) or not output_text.strip():
        completion = record.get("completion")
        if not isinstance(completion, str):
            raise ValueError("Record is missing source_output and completion.")
        output_text = extract_edit_text(completion)

    source_index = record.get("source_index")
    if not isinstance(source_index, int):
        source_index = line_number - 1

    return compute_example_context(
        source_index=source_index,
        input_text=input_text,
        output_text=output_text,
    )


def rewrite_record(
    record: dict[str, Any],
    *,
    line_number: int,
    plan_generator: TinkerPlanGenerator,
    base_model: str,
) -> tuple[dict[str, Any], bool]:
    completion = record.get("completion")
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("Record is missing completion.")

    context = resolve_example_context(record, line_number)
    original_plan = extract_plan_text(completion)
    new_plan, used_tinker, error_text = plan_generator.generate_plan(context)
    if not used_tinker:
        raise RuntimeError(
            f"Tinker plan generation failed: {error_text or 'unknown error'}"
        )

    new_completion = replace_plan_block(completion, new_plan)
    rewritten = dict(record)
    rewritten["completion"] = new_completion
    if "messages" in record:
        rewritten["messages"] = update_assistant_message(record["messages"], new_completion)
    rewritten["original_plan"] = original_plan
    rewritten["plan_rewrite_source"] = "tinker"
    rewritten["plan_rewrite_model"] = base_model
    rewritten["plan_rewritten"] = True
    return rewritten, original_plan != new_plan


def emit_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


def print_summary(stats: dict[str, Any], output_path: Path) -> None:
    print("\n[rewrite] Summary")
    print(f"  output_path: {output_path}")
    print(f"  seen: {stats['seen']}")
    print(f"  already_present: {stats['already_present']}")
    print(f"  written: {stats['written']}")
    print(f"  changed: {stats['changed']}")
    print(f"  skipped: {stats['skipped']}")
    print(f"  action_distribution: {summarize_counter(stats['actions'])}")
    print(f"  skip_reasons: {summarize_counter(stats['skip_reasons'])}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else derive_output_path(input_path)
    validate_args(args, input_path, output_path)

    configure_proxy_env(
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        disable_probe=args.disable_proxy_probe,
    )

    plan_generator = TinkerPlanGenerator(
        base_model=args.base_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        init_timeout_seconds=args.plan_init_timeout,
        result_timeout_seconds=args.plan_result_timeout,
    )

    initialize_output(output_path, resume=args.resume)
    completed_indices = load_existing_indices(output_path) if args.resume else set()
    stats: dict[str, Any] = {
        "seen": 0,
        "already_present": 0,
        "written": 0,
        "changed": 0,
        "skipped": 0,
        "actions": Counter(),
        "skip_reasons": Counter(),
    }

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if args.limit is not None and stats["seen"] >= args.limit:
                break
            if not line.strip():
                continue

            stats["seen"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["skipped"] += 1
                stats["skip_reasons"]["JSONDecodeError"] += 1
                if args.fail_on_plan_error:
                    raise ValueError(
                        f"Invalid JSON in {input_path} at line {line_number}: {exc}"
                    ) from exc
                continue

            source_index = record.get("source_index")
            if isinstance(source_index, int) and source_index in completed_indices:
                stats["already_present"] += 1
                continue

            try:
                rewritten_record, changed = rewrite_record(
                    record,
                    line_number=line_number,
                    plan_generator=plan_generator,
                    base_model=args.base_model,
                )
            except Exception as exc:
                stats["skipped"] += 1
                stats["skip_reasons"][type(exc).__name__] += 1
                if args.fail_on_plan_error:
                    raise
                continue

            action_type = rewritten_record.get("action_type")
            if isinstance(action_type, str):
                stats["actions"][action_type] += 1

            emit_record(output_path, rewritten_record)
            stats["written"] += 1
            if changed:
                stats["changed"] += 1
            if isinstance(source_index, int):
                completed_indices.add(source_index)

            if stats["written"] % PROGRESS_INTERVAL == 0:
                print(
                    f"[rewrite] seen={stats['seen']} written={stats['written']} "
                    f"skipped={stats['skipped']} resume_hits={stats['already_present']}"
                )

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print_summary(stats, output_path)


if __name__ == "__main__":
    main()
