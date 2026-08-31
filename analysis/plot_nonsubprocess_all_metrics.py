#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ITER_RE = re.compile(r"iters?(\d+)", re.IGNORECASE)

CORE_METRICS = [
    ("mean_G", "Mean G", "#1f77b4"),
    ("std_G", "Std G", "#17becf"),
    ("mean_reward_total", "Total Reward", "#d62728"),
    ("mean_nte", "NTE", "#e377c2"),
    ("num_final_valid", "Final Valid Count", "#2ca02c"),
    ("terminal_valid_rate", "Terminal Valid Rate", "#1b9e77"),
    ("rollback_used_rate", "Rollback Used Rate", "#7570b3"),
]

REWARD_METRICS = [
    ("mean_reward_format", "Format Reward", "#ff7f0e"),
    ("mean_reward_property", "Property Reward", "#2ca02c"),
    ("mean_reward_repetition", "Repetition Reward", "#7f7f7f"),
    ("mean_reward_total", "Total Reward", "#d62728"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect nonsubprocess training metrics under out/training_metrics, "
            "ignore combined* aggregate folders, sort by itersXXX, and plot all key metrics."
        )
    )
    parser.add_argument("--metrics-root", default="out/training_metrics")
    parser.add_argument(
        "--output-dir",
        default="out/training_metrics/nonsubprocess_all_by_iter",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Odd/even moving-average window for the smoothed trend line.",
    )
    return parser.parse_args()


def infer_end_iteration(run_dir: Path) -> int | None:
    match = ITER_RE.search(run_dir.name)
    if not match:
        return None
    return int(match.group(1))


def discover_metric_runs(metrics_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_dir in metrics_root.iterdir():
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("combined"):
            continue
        metrics_csv = run_dir / "metrics.csv"
        if not metrics_csv.exists():
            continue
        end_iteration = infer_end_iteration(run_dir)
        if end_iteration is None:
            continue
        runs.append(
            {
                "run_dir": run_dir,
                "metrics_csv": metrics_csv,
                "end_iteration": end_iteration,
            }
        )
    runs.sort(key=lambda spec: (spec["end_iteration"], spec["run_dir"].name))
    return runs


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return ""
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    try:
        if re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
        return float(stripped)
    except ValueError:
        return stripped


def enrich_rows(run_specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combined_rows: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    for spec in run_specs:
        rows = read_csv_rows(spec["metrics_csv"])
        end_iteration = int(spec["end_iteration"])
        start_iteration = end_iteration - len(rows)
        run_name = spec["run_dir"].name

        for offset, row in enumerate(rows):
            enriched = {key: coerce_scalar(value) for key, value in row.items()}
            enriched["source_run"] = run_name
            enriched["source_metrics_csv"] = str(spec["metrics_csv"])
            enriched["source_end_iteration"] = end_iteration
            enriched["local_iteration"] = enriched.get("iteration", offset)
            enriched["global_iteration"] = start_iteration + offset
            combined_rows.append(enriched)

        segments.append(
            {
                "run_name": run_name,
                "metrics_csv": str(spec["metrics_csv"]),
                "start_iteration": start_iteration,
                "end_iteration_exclusive": end_iteration,
                "end_iteration_inclusive": end_iteration - 1,
                "num_points": len(rows),
            }
        )

    combined_rows.sort(key=lambda row: row["global_iteration"])
    return combined_rows, segments


def ensure_output_dir(output_dir: Path, *, allow_overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}. "
            "Pass --allow-overwrite or choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def save_outputs(rows: list[dict[str, Any]], segments: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    jsonl_path = output_dir / "nonsubprocess_all_metrics.jsonl"
    csv_path = output_dir / "nonsubprocess_all_metrics.csv"
    summary_path = output_dir / "nonsubprocess_all_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    gaps: list[dict[str, int]] = []
    for left, right in zip(segments, segments[1:]):
        expected_next = int(left["end_iteration_exclusive"])
        actual_next = int(right["start_iteration"])
        if expected_next != actual_next:
            gaps.append({"expected_next": expected_next, "actual_next": actual_next})

    summary = {
        "num_points": len(rows),
        "first_global_iteration": rows[0]["global_iteration"] if rows else None,
        "last_global_iteration": rows[-1]["global_iteration"] if rows else None,
        "segments": segments,
        "gaps_or_overlaps": gaps,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"jsonl": jsonl_path, "csv": csv_path, "summary": summary_path}


def values_for(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            x_values.append(float(row["global_iteration"]))
            y_values.append(float(value))
    return np.array(x_values, dtype=float), np.array(y_values, dtype=float)


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 3:
        return y.copy()
    window = min(window, len(y))
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def plot_metric_grid(
    *,
    rows: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    metrics: list[tuple[str, str, str]],
    output_path: Path,
    title: str,
    smooth_window: int,
    ncols: int,
) -> None:
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.8 * nrows), sharex=True)
    axes_flat = np.array(axes).reshape(-1)
    boundaries = [segment["start_iteration"] for segment in segments[1:] if segment["num_points"] > 0]

    for ax, (key, label, color) in zip(axes_flat, metrics):
        x, y = values_for(rows, key)
        if len(x) == 0:
            ax.set_title(f"{label} (missing)")
            ax.axis("off")
            continue

        ax.plot(x, y, color=color, alpha=0.18, linewidth=1.0, label="raw")
        ax.plot(x, smooth(y, smooth_window), color=color, linewidth=2.4, label=f"MA({smooth_window})")
        for boundary in boundaries:
            ax.axvline(boundary, color="#555555", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.24)
        ax.legend(loc="best", fontsize=8)

    for ax in axes_flat[len(metrics):]:
        ax.axis("off")
    for ax in axes_flat[-ncols:]:
        ax.set_xlabel("Global Iteration")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_root = Path(args.metrics_root)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, allow_overwrite=args.allow_overwrite)

    run_specs = discover_metric_runs(metrics_root)
    if not run_specs:
        raise RuntimeError(f"No metric runs found under {metrics_root}")

    rows, segments = enrich_rows(run_specs)
    saved = save_outputs(rows, segments, output_dir)

    first_iter = rows[0]["global_iteration"] if rows else None
    last_iter = rows[-1]["global_iteration"] if rows else None
    plot_metric_grid(
        rows=rows,
        segments=segments,
        metrics=CORE_METRICS,
        output_path=output_dir / "nonsubprocess_core_metrics.png",
        title=f"Nonsubprocess Core Metrics ({first_iter}-{last_iter})",
        smooth_window=args.smooth_window,
        ncols=2,
    )
    plot_metric_grid(
        rows=rows,
        segments=segments,
        metrics=REWARD_METRICS,
        output_path=output_dir / "nonsubprocess_reward_components.png",
        title=f"Nonsubprocess Reward Components ({first_iter}-{last_iter})",
        smooth_window=args.smooth_window,
        ncols=2,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "combined_csv": str(saved["csv"]),
                "combined_jsonl": str(saved["jsonl"]),
                "summary": str(saved["summary"]),
                "core_plot": str(output_dir / "nonsubprocess_core_metrics.png"),
                "reward_plot": str(output_dir / "nonsubprocess_reward_components.png"),
                "num_points": len(rows),
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
