#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PREFIX_RE = re.compile(r"^(\d+)")
ITERS_RE = re.compile(r"iters?(\d+)", re.IGNORECASE)

METHOD_SPECS = {
    "2D": {
        "root": Path("experiments/2D-qed-logp-mw/out/training_metrics"),
        "color": "#2563eb",
    },
    "3D descriptors": {
        "root": Path("experiments/3DDESC-qed-logp-mw/out/training_metrics"),
        "color": "#e11d48",
    },
}

CORE_METRICS = [
    ("mean_reward_total", "Total reward", "#d62728"),
    ("mean_reward_property", "Property reward", "#2ca02c"),
    ("mean_G", "Mean G", "#1f77b4"),
    ("mean_nte", "NTE", "#9467bd"),
    ("terminal_valid_rate", "Terminal valid rate", "#1b9e77"),
    ("pos_adv_ratio", "Positive advantage ratio", "#ff7f0e"),
    ("mean_final_qed_error", "Final QED error", "#8c564b"),
    ("mean_final_mw_error", "Final MW error", "#17becf"),
    ("mean_final_logp_error", "Final LogP error", "#7f7f7f"),
]

REWARD_METRICS = [
    ("mean_reward_format", "Format reward", "#ff7f0e"),
    ("mean_reward_property", "Property reward", "#2ca02c"),
    ("mean_reward_repetition", "Repetition reward", "#7f7f7f"),
    ("mean_reward_total", "Total reward", "#d62728"),
]

PROPERTY_METRICS = [
    ("mean_final_qed_error", "Final QED error", "#8c564b"),
    ("mean_final_mw_error", "Final MW error", "#17becf"),
    ("mean_final_logp_error", "Final LogP error", "#7f7f7f"),
    ("mean_nte", "NTE", "#9467bd"),
]

VALIDITY_METRICS = [
    ("num_final_valid", "Terminal valid count", "#1b9e77"),
    ("num_final_missing", "Terminal missing count", "#dc2626"),
    ("terminal_valid_rate", "Terminal valid rate", "#047857"),
    ("terminal_valid_per_datum_rate", "Terminal valid / train datum", "#65a30d"),
    ("num_turn_valid", "All-turn valid count", "#0891b2"),
    ("turn_valid_rate", "All-turn valid rate", "#0e7490"),
    ("turn1_valid_count", "Turn 1 valid count", "#2563eb"),
    ("turn2_valid_count", "Turn 2 valid count", "#7c3aed"),
    ("turn3_valid_count", "Turn 3 valid count", "#c026d3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate staged 2D and 3D-descriptor training metrics, infer global "
            "iterations from folder prefixes, and draw individual plus comparison plots."
        )
    )
    parser.add_argument(
        "--two-d-root",
        default=str(METHOD_SPECS["2D"]["root"]),
        help="Root directory containing 2D staged metrics folders.",
    )
    parser.add_argument(
        "--three-d-root",
        default=str(METHOD_SPECS["3D descriptors"]["root"]),
        help="Root directory containing 3D-descriptor staged metrics folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/training_plots/2d_vs_3d_descriptor_summary",
        help="Directory for combined CSV/JSON files and PNG plots.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Moving-average window used for trend lines within each contiguous segment.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also draw faint raw per-iteration curves behind the smoothed trends.",
    )
    return parser.parse_args()


def infer_segment_end(run_dir: Path) -> int:
    match = PREFIX_RE.search(run_dir.name)
    if not match:
        raise ValueError(f"Cannot infer segment end from folder prefix: {run_dir}")
    return int(match.group(1))


def infer_segment_length(run_dir: Path) -> int:
    match = ITERS_RE.search(run_dir.name)
    if not match:
        raise ValueError(f"Cannot infer segment length from itersN in folder name: {run_dir}")
    return int(match.group(1))


def coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return ""
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
        return float(stripped)
    except ValueError:
        return stripped


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: coerce_scalar(value) for key, value in row.items()} for row in reader]


def safe_rate(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def attach_derived_metrics(row: dict[str, Any]) -> None:
    final_valid = row.get("num_final_valid")
    final_missing = row.get("num_final_missing")
    if isinstance(final_valid, (int, float)) and isinstance(final_missing, (int, float)):
        terminal_total = final_valid + final_missing
        row["num_terminal_total"] = terminal_total
        if terminal_total:
            row["terminal_valid_rate"] = float(final_valid) / float(terminal_total)

    per_datum_rate = safe_rate(row.get("num_final_valid"), row.get("num_data"))
    if per_datum_rate is not None:
        row["terminal_valid_per_datum_rate"] = per_datum_rate

    desc_success = row.get("num_3d_desc_success")
    desc_fallback = row.get("num_3d_desc_fallback")
    if isinstance(desc_success, (int, float)) and isinstance(desc_fallback, (int, float)):
        total = desc_success + desc_fallback
        if total:
            row["desc_success_rate"] = float(desc_success) / float(total)


def discover_runs(metrics_root: Path) -> list[dict[str, Any]]:
    if not metrics_root.exists():
        raise FileNotFoundError(f"Metrics root does not exist: {metrics_root}")

    runs: list[dict[str, Any]] = []
    for run_dir in metrics_root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("combined"):
            continue
        metrics_csv = run_dir / "metrics.csv"
        if not metrics_csv.exists():
            continue

        segment_end = infer_segment_end(run_dir)
        segment_length = infer_segment_length(run_dir)
        runs.append(
            {
                "run_dir": run_dir,
                "metrics_csv": metrics_csv,
                "segment_start": segment_end - segment_length,
                "segment_end_exclusive": segment_end,
                "segment_length_from_name": segment_length,
            }
        )

    runs.sort(key=lambda spec: (spec["segment_start"], spec["segment_end_exclusive"], spec["run_dir"].name))
    return runs


def load_method(method: str, metrics_root: Path) -> dict[str, Any]:
    run_specs = discover_runs(metrics_root)
    if not run_specs:
        raise RuntimeError(f"No metrics.csv files found under {metrics_root}")

    rows: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    for spec in run_specs:
        csv_rows = read_csv_rows(spec["metrics_csv"])
        start = int(spec["segment_start"])
        end_exclusive = int(spec["segment_end_exclusive"])
        expected_len = int(spec["segment_length_from_name"])
        segment_rows: list[dict[str, Any]] = []

        for offset, row in enumerate(csv_rows):
            local_iteration = row.get("iteration", offset)
            if not isinstance(local_iteration, (int, float)):
                local_iteration = offset

            enriched = dict(row)
            enriched["method"] = method
            enriched["source_run"] = spec["run_dir"].name
            enriched["source_metrics_csv"] = str(spec["metrics_csv"])
            enriched["segment_start"] = start
            enriched["segment_end_exclusive"] = end_exclusive
            enriched["segment_length_from_name"] = expected_len
            enriched["local_iteration"] = int(local_iteration)
            enriched["global_iteration"] = start + int(local_iteration)
            attach_derived_metrics(enriched)
            segment_rows.append(enriched)
            rows.append(enriched)

        actual_start = min((row["global_iteration"] for row in segment_rows), default=start)
        actual_end = max((row["global_iteration"] for row in segment_rows), default=start - 1) + 1
        segments.append(
            {
                "method": method,
                "run_name": spec["run_dir"].name,
                "metrics_csv": str(spec["metrics_csv"]),
                "start_iteration": start,
                "end_iteration_exclusive": end_exclusive,
                "end_iteration_inclusive": end_exclusive - 1,
                "actual_start_iteration": actual_start,
                "actual_end_iteration_exclusive": actual_end,
                "num_points": len(csv_rows),
                "segment_length_from_name": expected_len,
                "length_matches_name": len(csv_rows) == expected_len,
            }
        )

    rows.sort(key=lambda row: (row["global_iteration"], row["source_run"]))
    return {
        "method": method,
        "root": metrics_root,
        "rows": rows,
        "segments": segments,
    }


def compute_gaps_or_overlaps(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    ordered = sorted(segments, key=lambda segment: segment["start_iteration"])
    for left, right in zip(ordered, ordered[1:]):
        expected_next = int(left["end_iteration_exclusive"])
        actual_next = int(right["start_iteration"])
        if actual_next > expected_next:
            events.append(
                {
                    "type": "gap",
                    "start_iteration": expected_next,
                    "end_iteration_exclusive": actual_next,
                    "num_missing_iterations": actual_next - expected_next,
                }
            )
        elif actual_next < expected_next:
            events.append(
                {
                    "type": "overlap",
                    "left_end_iteration_exclusive": expected_next,
                    "right_start_iteration": actual_next,
                    "num_overlapping_iterations": expected_next - actual_next,
                }
            )
    return events


def fieldnames_for(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "method",
        "global_iteration",
        "local_iteration",
        "source_run",
        "segment_start",
        "segment_end_exclusive",
    ]
    fields = sorted({key for row in rows for key in row.keys()})
    return [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]


def save_method_outputs(dataset: dict[str, Any], output_dir: Path, slug: str) -> dict[str, Path]:
    rows = dataset["rows"]
    segments = dataset["segments"]
    jsonl_path = output_dir / f"combined_{slug}_metrics.jsonl"
    csv_path = output_dir / f"combined_{slug}_metrics.csv"
    summary_path = output_dir / f"combined_{slug}_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_for(rows))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": dataset["method"],
        "metrics_root": str(dataset["root"]),
        "num_points": len(rows),
        "first_global_iteration": rows[0]["global_iteration"] if rows else None,
        "last_global_iteration": rows[-1]["global_iteration"] if rows else None,
        "metric_keys": sorted({key for row in rows for key in row.keys()}),
        "segments": segments,
        "gaps_or_overlaps": compute_gaps_or_overlaps(segments),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"jsonl": jsonl_path, "csv": csv_path, "summary": summary_path}


def save_both_outputs(datasets: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    rows = [row for dataset in datasets for row in dataset["rows"]]
    rows.sort(key=lambda row: (row["method"], row["global_iteration"], row["source_run"]))
    csv_path = output_dir / "combined_2d_3d_metrics.csv"
    jsonl_path = output_dir / "combined_2d_3d_metrics.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames_for(rows))
        writer.writeheader()
        writer.writerows(rows)

    return {"csv": csv_path, "jsonl": jsonl_path}


def smooth_values(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 3:
        return y.copy()
    window = min(window, len(y))
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def xy_for_rows(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            x_values.append(float(row["global_iteration"]))
            y_values.append(float(value))
    return np.array(x_values, dtype=float), np.array(y_values, dtype=float)


def rows_for_segment(rows: list[dict[str, Any]], segment: dict[str, Any]) -> list[dict[str, Any]]:
    run_name = segment["run_name"]
    return [row for row in rows if row["source_run"] == run_name]


def draw_gaps(ax: plt.Axes, segments: list[dict[str, Any]]) -> None:
    for event in compute_gaps_or_overlaps(segments):
        if event["type"] == "gap":
            ax.axvspan(
                event["start_iteration"],
                event["end_iteration_exclusive"],
                color="#d4d4d8",
                alpha=0.22,
                linewidth=0,
            )


def draw_boundaries(ax: plt.Axes, segments: list[dict[str, Any]]) -> None:
    for segment in sorted(segments, key=lambda item: item["start_iteration"])[1:]:
        ax.axvline(
            segment["start_iteration"],
            color="#52525b",
            linestyle="--",
            linewidth=0.8,
            alpha=0.42,
        )


def plot_dataset_grid(
    *,
    dataset: dict[str, Any],
    metrics: list[tuple[str, str, str]],
    output_path: Path,
    title: str,
    smooth_window: int,
    ncols: int,
    show_raw: bool,
) -> None:
    rows = dataset["rows"]
    segments = dataset["segments"]
    nrows = int(math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.65 * nrows), sharex=True)
    axes_flat = np.array(axes).reshape(-1)

    for ax, (key, label, color) in zip(axes_flat, metrics):
        plotted = False
        draw_gaps(ax, segments)
        draw_boundaries(ax, segments)

        for segment in segments:
            segment_rows = rows_for_segment(rows, segment)
            x, y = xy_for_rows(segment_rows, key)
            if len(x) == 0:
                continue
            plotted = True
            if show_raw:
                ax.plot(x, y, color=color, alpha=0.14, linewidth=0.9)
            ax.plot(x, smooth_values(y, smooth_window), color=color, linewidth=2.1)

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.24)
        if not plotted:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)

    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")
    for ax in axes_flat[-ncols:]:
        ax.set_xlabel("Global iteration")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_grid(
    *,
    datasets: list[dict[str, Any]],
    metrics: list[tuple[str, str, str]],
    output_path: Path,
    title: str,
    smooth_window: int,
    ncols: int,
    show_raw: bool,
) -> None:
    nrows = int(math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.65 * nrows), sharex=True)
    axes_flat = np.array(axes).reshape(-1)

    for ax, (key, label, _metric_color) in zip(axes_flat, metrics):
        any_plotted = False
        for dataset in datasets:
            method = dataset["method"]
            method_color = METHOD_SPECS[method]["color"]
            method_plotted = False
            draw_gaps(ax, dataset["segments"])

            for segment in dataset["segments"]:
                segment_rows = rows_for_segment(dataset["rows"], segment)
                x, y = xy_for_rows(segment_rows, key)
                if len(x) == 0:
                    continue
                any_plotted = True
                if show_raw:
                    ax.plot(x, y, color=method_color, alpha=0.10, linewidth=0.8)
                ax.plot(
                    x,
                    smooth_values(y, smooth_window),
                    color=method_color,
                    linewidth=2.25,
                    label=method if not method_plotted else None,
                )
                method_plotted = True

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.24)
        if any_plotted:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)

    for ax in axes_flat[len(metrics) :]:
        ax.axis("off")
    for ax in axes_flat[-ncols:]:
        ax.set_xlabel("Global iteration")

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_run_summary(output_dir: Path, saved: dict[str, Any], datasets: list[dict[str, Any]]) -> None:
    payload = {
        "output_dir": str(output_dir),
        "saved": {key: str(value) for key, value in saved.items()},
        "methods": [],
    }
    for dataset in datasets:
        rows = dataset["rows"]
        payload["methods"].append(
            {
                "method": dataset["method"],
                "num_points": len(rows),
                "first_global_iteration": rows[0]["global_iteration"] if rows else None,
                "last_global_iteration": rows[-1]["global_iteration"] if rows else None,
                "segments": [
                    {
                        "run_name": segment["run_name"],
                        "start": segment["start_iteration"],
                        "end_exclusive": segment["end_iteration_exclusive"],
                        "num_points": segment["num_points"],
                    }
                    for segment in dataset["segments"]
                ],
                "gaps_or_overlaps": compute_gaps_or_overlaps(dataset["segments"]),
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    METHOD_SPECS["2D"]["root"] = Path(args.two_d_root)
    METHOD_SPECS["3D descriptors"]["root"] = Path(args.three_d_root)

    datasets = [
        load_method("2D", METHOD_SPECS["2D"]["root"]),
        load_method("3D descriptors", METHOD_SPECS["3D descriptors"]["root"]),
    ]

    saved: dict[str, Any] = {}
    saved.update({f"2d_{key}": path for key, path in save_method_outputs(datasets[0], output_dir, "2d").items()})
    saved.update({f"3d_{key}": path for key, path in save_method_outputs(datasets[1], output_dir, "3d").items()})
    saved.update({f"both_{key}": path for key, path in save_both_outputs(datasets, output_dir).items()})

    for dataset, slug in [(datasets[0], "2d"), (datasets[1], "3d")]:
        first_iter = dataset["rows"][0]["global_iteration"]
        last_iter = dataset["rows"][-1]["global_iteration"]
        method = dataset["method"]
        plot_dataset_grid(
            dataset=dataset,
            metrics=CORE_METRICS,
            output_path=output_dir / f"{slug}_core_metrics.png",
            title=f"{method} core metrics ({first_iter}-{last_iter})",
            smooth_window=args.smooth_window,
            ncols=3,
            show_raw=args.show_raw,
        )
        plot_dataset_grid(
            dataset=dataset,
            metrics=REWARD_METRICS,
            output_path=output_dir / f"{slug}_reward_components.png",
            title=f"{method} reward components ({first_iter}-{last_iter})",
            smooth_window=args.smooth_window,
            ncols=2,
            show_raw=args.show_raw,
        )
        plot_dataset_grid(
            dataset=dataset,
            metrics=PROPERTY_METRICS,
            output_path=output_dir / f"{slug}_property_errors.png",
            title=f"{method} property errors ({first_iter}-{last_iter})",
            smooth_window=args.smooth_window,
            ncols=2,
            show_raw=args.show_raw,
        )
        plot_dataset_grid(
            dataset=dataset,
            metrics=VALIDITY_METRICS,
            output_path=output_dir / f"{slug}_validity_diagnostics.png",
            title=f"{method} validity diagnostics ({first_iter}-{last_iter})",
            smooth_window=args.smooth_window,
            ncols=3,
            show_raw=args.show_raw,
        )

    plot_comparison_grid(
        datasets=datasets,
        metrics=CORE_METRICS,
        output_path=output_dir / "2d_vs_3d_core_metrics.png",
        title="2D vs 3D descriptors: core metrics",
        smooth_window=args.smooth_window,
        ncols=3,
        show_raw=args.show_raw,
    )
    plot_comparison_grid(
        datasets=datasets,
        metrics=REWARD_METRICS,
        output_path=output_dir / "2d_vs_3d_reward_components.png",
        title="2D vs 3D descriptors: reward components",
        smooth_window=args.smooth_window,
        ncols=2,
        show_raw=args.show_raw,
    )
    plot_comparison_grid(
        datasets=datasets,
        metrics=PROPERTY_METRICS,
        output_path=output_dir / "2d_vs_3d_property_errors.png",
        title="2D vs 3D descriptors: property errors",
        smooth_window=args.smooth_window,
        ncols=2,
        show_raw=args.show_raw,
    )
    plot_comparison_grid(
        datasets=datasets,
        metrics=VALIDITY_METRICS,
        output_path=output_dir / "2d_vs_3d_validity_diagnostics.png",
        title="2D vs 3D descriptors: validity diagnostics",
        smooth_window=args.smooth_window,
        ncols=3,
        show_raw=args.show_raw,
    )

    saved["2d_core_plot"] = output_dir / "2d_core_metrics.png"
    saved["2d_reward_plot"] = output_dir / "2d_reward_components.png"
    saved["2d_property_plot"] = output_dir / "2d_property_errors.png"
    saved["2d_validity_plot"] = output_dir / "2d_validity_diagnostics.png"
    saved["3d_core_plot"] = output_dir / "3d_core_metrics.png"
    saved["3d_reward_plot"] = output_dir / "3d_reward_components.png"
    saved["3d_property_plot"] = output_dir / "3d_property_errors.png"
    saved["3d_validity_plot"] = output_dir / "3d_validity_diagnostics.png"
    saved["comparison_core_plot"] = output_dir / "2d_vs_3d_core_metrics.png"
    saved["comparison_reward_plot"] = output_dir / "2d_vs_3d_reward_components.png"
    saved["comparison_property_plot"] = output_dir / "2d_vs_3d_property_errors.png"
    saved["comparison_validity_plot"] = output_dir / "2d_vs_3d_validity_diagnostics.png"

    print_run_summary(output_dir, saved, datasets)


if __name__ == "__main__":
    main()
