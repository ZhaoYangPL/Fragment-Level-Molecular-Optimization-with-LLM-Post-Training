#!/usr/bin/env python3
"""
Utilities for saving and recovering notebook RL training metrics.

Primary use cases:
1. From a live notebook kernel, save in-memory ``all_metrics`` to ``out/``.
2. As a fallback, recover printed per-iteration metrics from a saved ``.ipynb``.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ITER_LINE_RE = re.compile(
    r"^\s*(?P<iteration>\d+)\s*\|\s*"
    r"(?P<mean_G>[-+]?\d*\.?\d+)\s*\|\s*"
    r"(?P<std_G>[-+]?\d*\.?\d+)\s*\|\s*"
    r"(?P<mean_abs_adv>[-+]?\d*\.?\d+)\s*\|\s*"
    r"(?P<std_adv>[-+]?\d*\.?\d+)\s*\|\s*"
    r"(?P<pos_adv_ratio>[-+]?\d*\.?\d+)%\s*\|\s*"
    r"(?P<num_datums>\d+)\s*\|\s*"
    r"(?P<elapsed_s>[-+]?\d*\.?\d+)\s*$"
)

REWARD_LINE_RE = re.compile(
    r"rewards \| total=(?P<mean_reward_total>[-+]?\d*\.?\d+)\s+"
    r"(?:\(weighted=(?P<mean_reward_weighted>[-+]?\d*\.?\d+)\)\s+)?\|\s+"
    r"format=(?P<mean_reward_format>[-+]?\d*\.?\d+)\s+\|\s+"
    r"property=(?P<mean_reward_property>[-+]?\d*\.?\d+)\s+\|\s+"
    r"(?:similarity=(?P<mean_reward_similarity>[-+]?\d*\.?\d+)\s+\|\s+)?"
    r"repetition=(?P<mean_reward_repetition>[-+]?\d*\.?\d+)"
)

OPTIONAL_FLOAT_RE = r"[-+]?(?:\d*\.?\d+|nan)"

NTE_LINE_RE = re.compile(
    rf"nte \| mean=(?P<mean_nte>{OPTIONAL_FLOAT_RE})\s+"
    rf"std=(?P<std_nte>{OPTIONAL_FLOAT_RE})\s+"
    r"valid=(?P<num_nte_valid>\d+)\s+"
    r"missing=(?P<num_nte_missing>\d+)\s+\|\s+"
    rf"\|dQED\|=(?P<mean_final_qed_error>{OPTIONAL_FLOAT_RE})\s+\|\s+"
    rf"\|dLogP\|=(?P<mean_final_logp_error>{OPTIONAL_FLOAT_RE})\s+\|\s+"
    rf"\|dMW\|=(?P<mean_final_mw_error>{OPTIONAL_FLOAT_RE})",
    re.IGNORECASE,
)

VALIDITY_LINE_RE = re.compile(
    rf"validity \| terminal_valid=(?P<terminal_valid_rate>{OPTIONAL_FLOAT_RE})%?\s+"
    r"\((?P<num_terminal_valid>\d+)/(?P<num_terminal_total>\d+)\)\s+\|\s+"
    rf"rollback_used=(?P<rollback_used_rate>{OPTIONAL_FLOAT_RE})%?\s+"
    r"\((?P<num_rollback_used>\d+)/(?P<num_rollback_total>\d+)\)",
    re.IGNORECASE,
)


def _parse_optional_float(text: str) -> float | None:
    if text.lower() == "nan":
        return None
    return float(text)


def _slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "run"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_fresh_run_dir(run_dir: Path, *, allow_overwrite: bool = False) -> None:
    if allow_overwrite:
        run_dir.mkdir(parents=True, exist_ok=True)
        return

    if run_dir.exists():
        try:
            has_contents = any(run_dir.iterdir())
        except OSError:
            has_contents = True
        if has_contents:
            raise FileExistsError(
                f"Refusing to overwrite existing training metrics directory: {run_dir}. "
                "Choose a new run_name / run_tag, or pass allow_overwrite=True if you "
                "really intend to replace it."
            )

    run_dir.mkdir(parents=True, exist_ok=True)


def _infer_run_name(
    metrics: list[dict[str, Any]],
    *,
    run_name: str | None = None,
    experiment_name: str | None = None,
    raw_data_len: int | None = None,
) -> str:
    if run_name:
        return _slugify(run_name)

    parts: list[str] = []
    if experiment_name:
        parts.append(_slugify(experiment_name))
    parts.append(f"iters{len(metrics)}")
    if raw_data_len is not None:
        parts.append(f"prompts{raw_data_len}")
    parts.append(_timestamp())
    return "_".join(parts)


def _summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_points": len(metrics),
        "metric_keys": sorted({key for row in metrics for key in row.keys()}),
    }
    if not metrics:
        return summary

    summary["first_iteration"] = metrics[0].get("iteration")
    summary["last_iteration"] = metrics[-1].get("iteration")

    numeric_keys = [
        "mean_G",
        "std_G",
        "mean_abs_adv",
        "std_adv",
        "pos_adv_ratio",
        "mean_reward_total",
        "mean_reward_weighted",
        "mean_reward_format",
        "mean_reward_property",
        "mean_reward_similarity",
        "mean_reward_repetition",
        "mean_nte",
        "std_nte",
        "mean_final_qed_error",
        "mean_final_logp_error",
        "mean_final_mw_error",
        "num_nte_valid",
        "num_nte_missing",
        "num_final_valid",
        "num_final_missing",
        "num_terminal_total",
        "num_terminal_valid",
        "num_rollback_used",
        "terminal_valid_rate",
        "rollback_used_rate",
        "effective_repetition_weight",
        "repetition_window_iterations",
        "sampling_temperature",
        "sampling_top_p",
        "sampling_top_k",
        "trajectories_per_prompt",
        "refresh_sampling_every",
        "num_3d_desc_success",
        "num_3d_desc_fallback",
        "num_datums",
        "elapsed_s",
    ]
    for key in numeric_keys:
        values = [row.get(key) for row in metrics if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        summary[key] = {
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
    return summary


def save_training_metrics(
    metrics: list[dict[str, Any]],
    *,
    output_dir: str | Path = "out/training_metrics",
    run_name: str | None = None,
    experiment_name: str | None = None,
    raw_data_len: int | None = None,
    metadata: dict[str, Any] | None = None,
    allow_overwrite: bool = False,
) -> dict[str, str]:
    clean_metrics = [_to_builtin(row) for row in metrics]
    output_root = Path(output_dir)
    final_run_name = _infer_run_name(
        clean_metrics,
        run_name=run_name,
        experiment_name=experiment_name,
        raw_data_len=raw_data_len,
    )
    run_dir = output_root / final_run_name
    _ensure_fresh_run_dir(run_dir, allow_overwrite=allow_overwrite)

    bundle = {
        "run_name": final_run_name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": _to_builtin(metadata or {}),
        "summary": _summarize_metrics(clean_metrics),
        "metrics": clean_metrics,
    }

    json_path = run_dir / "metrics_bundle.json"
    jsonl_path = run_dir / "metrics.jsonl"
    csv_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"

    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(bundle["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in clean_metrics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = sorted({key for row in clean_metrics for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in clean_metrics:
            writer.writerow(row)

    return {
        "run_dir": str(run_dir),
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
    }


def save_training_metrics_from_globals(
    global_ns: dict[str, Any],
    *,
    output_dir: str | Path = "out/training_metrics",
    run_name: str | None = None,
    allow_overwrite: bool = False,
) -> dict[str, str]:
    if "all_metrics" not in global_ns:
        raise KeyError("all_metrics not found in globals().")
    metrics = global_ns["all_metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("all_metrics is empty or not a list.")

    raw_data = global_ns.get("raw_data")
    raw_data_len = len(raw_data) if raw_data is not None else None
    experiment_name = global_ns.get("RUN_EXPERIMENT")
    metadata = {
        "source": "live_notebook_globals",
        "experiment_name": experiment_name,
        "run_stage": global_ns.get("RUN_STAGE"),
        "method_variant": global_ns.get("METHOD_VARIANT"),
        "raw_data_len": raw_data_len,
        "grpo_checkpoint_path": global_ns.get("grpo_checkpoint_path"),
        "base_reward_weights": global_ns.get("BASE_REWARD_WEIGHTS"),
        "fixed_three_stage_boundaries": global_ns.get("FIXED_THREE_STAGE_BOUNDARIES"),
        "use_repetition_reward": global_ns.get("USE_REPETITION_REWARD"),
        "effective_repetition_weight": global_ns.get("EFFECTIVE_REPETITION_WEIGHT"),
        "repetition_window_iterations": global_ns.get("REPETITION_WINDOW_ITERATIONS_ACTIVE"),
        "repetition_penalty_step": global_ns.get("REPETITION_PENALTY_STEP_ACTIVE"),
        "repetition_penalty_floor": global_ns.get("REPETITION_PENALTY_FLOOR_ACTIVE"),
        "sampling_temperature": global_ns.get("SAMPLING_TEMPERATURE_ACTIVE"),
        "sampling_max_tokens": global_ns.get("SAMPLING_MAX_TOKENS_ACTIVE"),
        "trajectories_per_prompt": global_ns.get("TRAJECTORIES_PER_PROMPT_ACTIVE"),
        "refresh_sampling_every": global_ns.get("REFRESH_SAMPLING_EVERY_ACTIVE"),
        "use_3d_desc": global_ns.get("USE_3D_DESC_ACTIVE"),
        "desc3d_seed": global_ns.get("DESC3D_SEED_ACTIVE"),
        "include_shape_label": global_ns.get("INCLUDE_SHAPE_LABEL_ACTIVE"),
    }
    return save_training_metrics(
        metrics,
        output_dir=output_dir,
        run_name=run_name,
        experiment_name=experiment_name,
        raw_data_len=raw_data_len,
        metadata=metadata,
        allow_overwrite=allow_overwrite,
    )


def recover_metrics_from_notebook(
    notebook_path: str | Path,
    *,
    output_dir: str | Path = "out/recovered_training_metrics",
    run_name: str | None = None,
) -> dict[str, str]:
    notebook_path = Path(notebook_path)
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))

    stream_text_chunks: list[str] = []
    recovered: list[dict[str, Any]] = []
    pending_row: dict[str, Any] | None = None

    for cell in payload.get("cells", []):
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            if not text:
                continue
            stream_text_chunks.append(text)
            for line in text.splitlines():
                iter_match = ITER_LINE_RE.match(line)
                if iter_match:
                    pending_row = {
                        "iteration": int(iter_match.group("iteration")),
                        "mean_G": float(iter_match.group("mean_G")),
                        "std_G": float(iter_match.group("std_G")),
                        "mean_abs_adv": float(iter_match.group("mean_abs_adv")),
                        "std_adv": float(iter_match.group("std_adv")),
                        "pos_adv_ratio": float(iter_match.group("pos_adv_ratio")) / 100.0,
                        "num_datums": int(iter_match.group("num_datums")),
                        "elapsed_s": float(iter_match.group("elapsed_s")),
                    }
                    recovered.append(pending_row)
                    continue

                reward_match = REWARD_LINE_RE.search(line)
                if reward_match and pending_row is not None:
                    weighted = reward_match.group("mean_reward_weighted")
                    similarity = reward_match.group("mean_reward_similarity")
                    pending_row.update(
                        {
                            "mean_reward_total": float(reward_match.group("mean_reward_total")),
                            "mean_reward_format": float(reward_match.group("mean_reward_format")),
                            "mean_reward_property": float(reward_match.group("mean_reward_property")),
                            "mean_reward_repetition": float(reward_match.group("mean_reward_repetition")),
                        }
                    )
                    if weighted is not None:
                        pending_row["mean_reward_weighted"] = float(weighted)
                    if similarity is not None:
                        pending_row["mean_reward_similarity"] = float(similarity)
                    continue

                nte_match = NTE_LINE_RE.search(line)
                if nte_match and pending_row is not None:
                    pending_row.update(
                        {
                            "mean_nte": _parse_optional_float(nte_match.group("mean_nte")),
                            "std_nte": _parse_optional_float(nte_match.group("std_nte")),
                            "mean_final_qed_error": _parse_optional_float(
                                nte_match.group("mean_final_qed_error")
                            ),
                            "mean_final_logp_error": _parse_optional_float(
                                nte_match.group("mean_final_logp_error")
                            ),
                            "mean_final_mw_error": _parse_optional_float(
                                nte_match.group("mean_final_mw_error")
                            ),
                            "num_nte_valid": int(nte_match.group("num_nte_valid")),
                            "num_nte_missing": int(nte_match.group("num_nte_missing")),
                            "num_final_valid": int(nte_match.group("num_nte_valid")),
                            "num_final_missing": int(nte_match.group("num_nte_missing")),
                        }
                    )

                validity_match = VALIDITY_LINE_RE.search(line)
                if validity_match and pending_row is not None:
                    terminal_rate = _parse_optional_float(validity_match.group("terminal_valid_rate"))
                    rollback_rate = _parse_optional_float(validity_match.group("rollback_used_rate"))
                    pending_row.update(
                        {
                            "num_terminal_total": int(validity_match.group("num_terminal_total")),
                            "num_terminal_valid": int(validity_match.group("num_terminal_valid")),
                            "num_rollback_used": int(validity_match.group("num_rollback_used")),
                            "terminal_valid_rate": (
                                None if terminal_rate is None else terminal_rate / 100.0
                            ),
                            "rollback_used_rate": (
                                None if rollback_rate is None else rollback_rate / 100.0
                            ),
                        }
                    )

    metadata = {
        "source": "saved_notebook_outputs",
        "notebook_path": str(notebook_path),
        "recovered_points": len(recovered),
    }
    artifacts = save_training_metrics(
        recovered,
        output_dir=output_dir,
        run_name=run_name or f"{notebook_path.stem}_recovered",
        metadata=metadata,
    )

    raw_stream_path = Path(artifacts["run_dir"]) / "raw_stream.txt"
    raw_stream_path.write_text("\n".join(stream_text_chunks), encoding="utf-8")
    artifacts["raw_stream"] = str(raw_stream_path)
    return artifacts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recover or save notebook RL training metrics.")
    parser.add_argument("--notebook", help="Recover metrics from a saved .ipynb file.")
    parser.add_argument("--output-dir", default="out/recovered_training_metrics")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    if not args.notebook:
        raise SystemExit("Pass --notebook to recover metrics from a saved notebook.")

    artifacts = recover_metrics_from_notebook(
        args.notebook,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
