#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import dataset as dataset_mod
from metrics_io import slugify
from model_utils import create_grpo_training_client
from model_utils import create_service_client
from model_utils import load_tokenizer_and_renderer
from model_utils import local_tokenizer_dir_for_model
from plotting import save_training_outputs


OUTPUT_ROOT = Path(__file__).resolve().parent
METHOD_VARIANT = "accepted_path_explicit_kl_delta_abs_turn2bonus0.25"
RL_CORE_FILENAME = "TURN2Bonus0.25KL_delta_abs_rl_core.py"
DEFAULT_KL_REFERENCE_CHECKPOINT = (
    "tinker://da3b0ce4-fbcc-5619-8304-7752622fdc4e:train:2/weights/general-sft-1000"
)
TINKER_LABEL_ALLOWED_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_tinker_label(label: str) -> str:
    sanitized = TINKER_LABEL_ALLOWED_CHARS.sub("", label)
    if not sanitized:
        raise ValueError(
            "--save-name must contain at least one alphanumeric, hyphen, underscore, or dot."
        )
    return sanitized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 2D single-stage multi-turn GRPO training pipeline."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Initial Tinker checkpoint path to load before GRPO training.",
    )
    parser.add_argument(
        "--num-iterations",
        required=True,
        type=int,
        help="Number of GRPO training iterations to run.",
    )
    parser.add_argument(
        "--start-iteration",
        required=True,
        type=int,
        help="Number of global training iterations already completed before this segment.",
    )
    parser.add_argument(
        "--save-name",
        required=True,
        help="Name passed to grpo_training_client.save_state(name=...).",
    )

    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--local-tokenizer-dir",
        help="Local tokenizer directory. Defaults to tokenizers/<base-model-with-__>.",
    )
    parser.add_argument(
        "--no-local-tokenizer",
        dest="prefer_local_tokenizer",
        action="store_false",
        help="Ignore local tokenizer files and load tokenizer by model id.",
    )
    parser.add_argument(
        "--no-cache-tokenizer",
        dest="cache_tokenizer",
        action="store_false",
        help="Do not save the tokenizer locally when loading by model id.",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--prompts-per-iter", type=int, default=2)
    parser.add_argument("--candidate-prompts-per-iter", type=int, default=6)
    parser.add_argument("--max-kept-prompts-per-iter", type=int, default=4)
    parser.add_argument("--lr-warmup-start", type=float, default=5e-6)
    parser.add_argument("--lr-max", type=float, default=1e-5)
    parser.add_argument("--lr-min", type=float, default=5e-7)
    parser.add_argument("--lr-warmup-iters", type=int, default=1000)
    parser.add_argument("--lr-hold-iters", type=int, default=5000)
    parser.add_argument("--lr-total-iters", type=int, default=25000)
    parser.add_argument("--trajectories-per-prompt", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=1.1)
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--refresh-sampling-every", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--kl-penalty-coef",
        type=float,
        default=0.03,
        help="Explicit forward-KL beta. Use 0 to keep the built-in PPO fast path.",
    )
    parser.add_argument(
        "--kl-target",
        type=float,
        default=0.08,
        help="Target forward KL for adaptive β. 0 disables adaptive β.",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help="Number of PPO epochs per data collection (standard: 4-8).",
    )
    parser.add_argument(
        "--ppo-mini-batch-size",
        type=int,
        default=16,
        help="Mini-batch size for PPO gradient accumulation.",
    )
    parser.add_argument(
        "--kl-reference-checkpoint",
        default=DEFAULT_KL_REFERENCE_CHECKPOINT,
        help="Reference checkpoint used for KL regularization.",
    )
    parser.add_argument(
        "--run-tag",
        help="Optional output tag. Use this to avoid collisions between repeated runs.",
    )

    parser.add_argument("--dataset-name", default=dataset_mod.DEFAULT_DATASET_NAME)
    parser.add_argument("--data-split", default=dataset_mod.DEFAULT_DATA_SPLIT)
    parser.add_argument(
        "--local-data-dir",
        default=str(dataset_mod.DEFAULT_LOCAL_DATA_DIR),
        help="Local save_to_disk dataset directory. Training reads this first by default.",
    )
    parser.add_argument(
        "--no-local-dataset",
        dest="prefer_local_dataset",
        action="store_false",
        help="Ignore the local dataset directory and load from HuggingFace.",
    )
    parser.add_argument(
        "--no-cache-dataset",
        dest="cache_dataset",
        action="store_false",
        help="Do not save a HuggingFace-loaded dataset into --local-data-dir.",
    )
    parser.add_argument(
        "--no-jsonl-dataset-export",
        dest="export_dataset_jsonl",
        action="store_false",
        help="Do not export local dataset splits as JSONL when caching.",
    )
    parser.add_argument("--prompt-col", default=dataset_mod.DEFAULT_PROMPT_COL)
    parser.add_argument("--answer-col", default=dataset_mod.DEFAULT_ANSWER_COL)
    parser.add_argument("--proxy-host", default=dataset_mod.DEFAULT_PROXY_HOST)
    parser.add_argument("--proxy-port", type=int, default=dataset_mod.DEFAULT_PROXY_PORT)
    parser.add_argument(
        "--no-proxy-probe",
        dest="use_proxy_probe",
        action="store_false",
        help="Do not probe/set the HTTP proxy before loading the HuggingFace dataset.",
    )
    parser.set_defaults(
        use_proxy_probe=True,
        prefer_local_dataset=True,
        cache_dataset=True,
        export_dataset_jsonl=True,
        prefer_local_tokenizer=True,
        cache_tokenizer=True,
    )

    parser.add_argument(
        "--no-token-prompt",
        action="store_true",
        help="Fail instead of prompting if TINKER_API_KEY is missing.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting existing local artifacts with the same run tag.",
    )
    parser.add_argument("--debug-adv", action="store_true")
    parser.add_argument("--debug-prompts-per-iter", type=int, default=0)
    parser.add_argument("--debug-iterations", type=int, default=0)
    parser.add_argument(
        "--drop-skipped-metrics",
        dest="keep_skipped_metrics",
        action="store_false",
        help="Do not keep metric rows for iterations that produce no PPO data.",
    )
    parser.set_defaults(keep_skipped_metrics=True)
    return parser.parse_args()


_TRAINING_CONFIG_KEYS = [
    "num_iterations", "start_iteration",
    "prompts_per_iter", "candidate_prompts_per_iter", "max_kept_prompts_per_iter",
    "lr_warmup_start", "lr_max", "lr_min",
    "lr_warmup_iters", "lr_hold_iters", "lr_total_iters",
    "trajectories_per_prompt", "max_turns",
    "gamma", "ema_alpha", "rng_seed",
    "debug_adv", "debug_prompts_per_iter", "debug_iterations",
    "keep_skipped_metrics", "refresh_sampling_every",
    "temperature", "top_p", "top_k", "kl_penalty_coef", "kl_target",
    "ppo_epochs", "ppo_mini_batch_size",
]


def build_training_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in _TRAINING_CONFIG_KEYS}


def build_run_tag(args: argparse.Namespace, raw_data_len: int) -> str:
    if args.run_tag:
        return slugify(args.run_tag)
    end_iteration = args.start_iteration + args.num_iterations
    return slugify(
        f"{METHOD_VARIANT}_global{args.start_iteration}-{end_iteration}_"
        f"prompts{raw_data_len}_samples{args.trajectories_per_prompt}_"
        f"temp{args.temperature}_top_p{args.top_p}_top_k{args.top_k}_"
        f"kl{args.kl_penalty_coef}"
    )


def create_kl_reference_client(service_client, args: argparse.Namespace):
    if args.kl_penalty_coef <= 0:
        return None, None
    if not args.kl_reference_checkpoint:
        raise ValueError("--kl-reference-checkpoint is required when --kl-penalty-coef > 0")

    checkpoint = args.kl_reference_checkpoint
    print(f"Creating KL reference sampling client from checkpoint: {checkpoint}")
    if "/sampler_weights/" in checkpoint:
        return (
            service_client.create_sampling_client(
                base_model=args.base_model,
                model_path=checkpoint,
            ),
            None,
        )

    if "/weights/" in checkpoint:
        print(
            "KL reference checkpoint is a training weights checkpoint; "
            "loading it into a separate reference training client first."
        )
        reference_training_client = service_client.create_lora_training_client(
            base_model=args.base_model,
            rank=args.lora_rank,
        )
        reference_training_client.load_state(checkpoint)
        reference_sampling_client = reference_training_client.save_weights_and_get_sampling_client()
        return reference_sampling_client, reference_training_client

    return (
        service_client.create_sampling_client(
            base_model=args.base_model,
            model_path=checkpoint,
        ),
        None,
    )


def load_rl_core_module():
    core_path = OUTPUT_ROOT / RL_CORE_FILENAME
    if not core_path.exists():
        raise FileNotFoundError(f"RL core file not found: {core_path}")
    module_name = re.sub(r"[^A-Za-z0-9_]+", "_", core_path.stem)
    spec = importlib.util.spec_from_file_location(module_name, core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load RL core module from {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any], *, allow_overwrite: bool = True) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. Change --run-tag or pass --allow-overwrite."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_run_config(
    *,
    output_root: Path,
    run_tag: str,
    metadata: dict[str, Any],
    training_config: dict[str, Any],
    allow_overwrite: bool,
) -> Path:
    path = output_root / "out" / "configs" / f"run_config_{run_tag}.json"
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata,
        "training_config": training_config,
    }
    _write_json(path, payload, allow_overwrite=allow_overwrite)
    print(f"Saved run config to: {path}")
    return path


def save_checkpoint_record(
    *,
    output_root: Path,
    run_tag: str,
    metadata: dict[str, Any],
    final_checkpoint_path: str,
) -> dict[str, str]:
    checkpoint_dir = output_root / "out" / "checkpoints"
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "run_tag": run_tag,
        "save_name": metadata["final_save_name"],
        "initial_checkpoint": metadata["initial_checkpoint"],
        "final_checkpoint_path": final_checkpoint_path,
        "base_model": metadata["base_model"],
        "lora_rank": metadata["lora_rank"],
        "method_variant": METHOD_VARIANT,
    }
    run_path = checkpoint_dir / f"final_checkpoint_{run_tag}.json"
    latest_path = checkpoint_dir / "final_checkpoint.json"
    _write_json(run_path, payload, allow_overwrite=True)
    _write_json(latest_path, payload, allow_overwrite=True)
    print(f"Saved checkpoint record to: {run_path}")
    print(f"Updated latest checkpoint record: {latest_path}")
    return {"run_record": str(run_path), "latest_record": str(latest_path)}


def main() -> int:
    args = parse_args()
    if args.num_iterations < 1:
        raise ValueError("--num-iterations must be >= 1")
    if args.start_iteration < 0:
        raise ValueError("--start-iteration must be >= 0")
    sanitized_save_name = sanitize_tinker_label(args.save_name)
    if sanitized_save_name != args.save_name:
        print(
            "Sanitized final checkpoint name for Tinker: "
            f"{args.save_name!r} -> {sanitized_save_name!r}"
        )
        args.save_name = sanitized_save_name

    RL_core = load_rl_core_module()

    RL_core.configure_columns(args.prompt_col, args.answer_col)
    RL_core.compute_scheduled_learning_rate(
        args.start_iteration,
        warmup_start_lr=args.lr_warmup_start,
        max_lr=args.lr_max,
        min_lr=args.lr_min,
        warmup_iterations=args.lr_warmup_iters,
        hold_iterations=args.lr_hold_iters,
        total_iterations=args.lr_total_iters,
    )
    if args.kl_penalty_coef < 0:
        raise ValueError("--kl-penalty-coef must be >= 0")

    service_client = create_service_client(
        prompt_for_missing_tokens=not args.no_token_prompt
    )

    raw_data = dataset_mod.load_raw_data(
        dataset_name=args.dataset_name,
        data_split=args.data_split,
        local_data_dir=args.local_data_dir,
        prefer_local=args.prefer_local_dataset,
        cache_local=args.cache_dataset,
        export_jsonl=args.export_dataset_jsonl,
        use_proxy_probe=args.use_proxy_probe,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
    )
    dataset_mod.validate_columns(raw_data, prompt_col=args.prompt_col, answer_col=args.answer_col)

    local_tokenizer_dir = (
        args.local_tokenizer_dir
        if args.local_tokenizer_dir
        else str(local_tokenizer_dir_for_model(args.base_model))
    )
    tokenizer, renderer = load_tokenizer_and_renderer(
        args.base_model,
        local_tokenizer_dir=local_tokenizer_dir,
        prefer_local_tokenizer=args.prefer_local_tokenizer,
        cache_tokenizer=args.cache_tokenizer,
    )
    training_client = create_grpo_training_client(
        service_client,
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        checkpoint=args.checkpoint,
    )
    kl_reference_client, _kl_reference_training_client = create_kl_reference_client(
        service_client,
        args,
    )

    training_config = build_training_config(args)
    run_tag = build_run_tag(args, raw_data_len=len(raw_data))
    metadata = {
        "source_notebook": "2D-nonsubprocess.ipynb",
        "method_variant": METHOD_VARIANT,
        "dataset_name": args.dataset_name,
        "data_split": args.data_split,
        "local_data_dir": args.local_data_dir,
        "prefer_local_dataset": args.prefer_local_dataset,
        "cache_dataset": args.cache_dataset,
        "prompt_col": args.prompt_col,
        "answer_col": args.answer_col,
        "base_model": args.base_model,
        "local_tokenizer_dir": local_tokenizer_dir,
        "prefer_local_tokenizer": args.prefer_local_tokenizer,
        "cache_tokenizer": args.cache_tokenizer,
        "lora_rank": args.lora_rank,
        "initial_checkpoint": args.checkpoint,
        "kl_reference_checkpoint": args.kl_reference_checkpoint,
        "final_save_name": args.save_name,
        "training_config": training_config,
    }

    run_config_path = save_run_config(
        output_root=OUTPUT_ROOT,
        run_tag=run_tag,
        metadata=metadata,
        training_config=training_config,
        allow_overwrite=args.allow_overwrite,
    )
    metadata["run_config_path"] = str(run_config_path)

    print(f"Running method={METHOD_VARIANT}")
    print("Single-stage GRPO config:")
    print(json.dumps(training_config, ensure_ascii=False, indent=2))
    print(f"Run tag: {run_tag}")

    all_metrics = RL_core.run_multistep_rl_training_fixed(
        raw_data=raw_data,
        renderer=renderer,
        training_client=training_client,
        tokenizer=tokenizer,
        kl_reference_client=kl_reference_client,
        **training_config,
    )

    print(f"Saving final GRPO checkpoint with name={args.save_name!r}")
    grpo_save_result = training_client.save_state(name=args.save_name).result()
    grpo_checkpoint_path = grpo_save_result.path
    print(f"GRPO checkpoint saved at: {grpo_checkpoint_path}")

    metadata["grpo_checkpoint_path"] = grpo_checkpoint_path
    checkpoint_records = save_checkpoint_record(
        output_root=OUTPUT_ROOT,
        run_tag=run_tag,
        metadata=metadata,
        final_checkpoint_path=grpo_checkpoint_path,
    )

    artifacts = save_training_outputs(
        all_metrics,
        output_root=OUTPUT_ROOT,
        run_tag=run_tag,
        raw_data_len=len(raw_data),
        metadata=metadata,
        allow_overwrite=args.allow_overwrite,
    )
    artifacts["checkpoint_records"] = checkpoint_records
    artifacts["final_checkpoint_path"] = grpo_checkpoint_path

    print("Run artifacts:")
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
