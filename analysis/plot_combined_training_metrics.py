#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


METRIC_SPECS = [
    ("mean_G", "Mean G", "#1f77b4"),
    ("mean_reward_total", "Total Reward", "#d62728"),
    ("mean_reward_property", "Property Reward", "#2ca02c"),
    ("mean_reward_similarity", "Similarity Reward", "#9467bd"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine one or more training metric runs into one smoothed plot.",
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        nargs=2,
        metavar=("JSONL", "START"),
        help=(
            "Repeat this argument to add any number of runs. "
            "Example: --run out/runA/metrics.jsonl 150 --run out/runB/metrics.jsonl 200"
        ),
    )
    parser.add_argument("--first-jsonl")
    parser.add_argument("--second-jsonl")
    parser.add_argument("--first-start", type=int)
    parser.add_argument("--second-start", type=int)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where combined csv/jsonl/plot will be written.",
    )
    parser.add_argument(
        "--smoothing-scale",
        type=float,
        default=1.0,
        help="Relative smoothing strength for spline fitting.",
    )
    parser.add_argument(
        "--pre-smooth-window",
        type=int,
        default=9,
        help="Window size for edge-padded moving-average smoothing before spline fitting.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title. Defaults to an auto-generated iteration range.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def attach_global_iterations(
    rows: list[dict[str, Any]],
    *,
    start_iteration: int,
    run_name: str,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        enriched_row = dict(row)
        enriched_row["local_iteration"] = row.get("iteration", index)
        enriched_row["global_iteration"] = start_iteration + index
        enriched_row["source_run"] = run_name
        enriched.append(enriched_row)
    return enriched


def ensure_fresh_output_dir(output_dir: Path, *, allow_overwrite: bool) -> None:
    if allow_overwrite:
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    if output_dir.exists():
        try:
            has_contents = any(output_dir.iterdir())
        except OSError:
            has_contents = True
        if has_contents:
            raise FileExistsError(
                f"Refusing to overwrite existing combined output directory: {output_dir}. "
                "Choose a new --output-dir or pass --allow-overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=True)


def collect_run_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.runs:
        if any(
            value is not None
            for value in (args.first_jsonl, args.second_jsonl, args.first_start, args.second_start)
        ):
            raise ValueError(
                "Use either repeatable --run arguments or the legacy --first/--second arguments, not both."
            )
        run_specs = [
            {
                "path": Path(jsonl_path),
                "start_iteration": int(start_iteration),
            }
            for jsonl_path, start_iteration in args.runs
        ]
    else:
        legacy_values = (args.first_jsonl, args.second_jsonl, args.first_start, args.second_start)
        if any(value is not None for value in legacy_values):
            if any(value is None for value in legacy_values):
                raise ValueError(
                    "Legacy mode requires --first-jsonl, --second-jsonl, --first-start, and --second-start together."
                )
            run_specs = [
                {"path": Path(args.first_jsonl), "start_iteration": int(args.first_start)},
                {"path": Path(args.second_jsonl), "start_iteration": int(args.second_start)},
            ]
        else:
            raise ValueError("Provide at least one run via --run, or use the legacy --first/--second arguments.")

    run_specs.sort(key=lambda spec: spec["start_iteration"])
    return run_specs


def enrich_runs(run_specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combined_rows: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []

    for spec in run_specs:
        path = spec["path"]
        start_iteration = spec["start_iteration"]
        rows = load_jsonl(path)
        run_name = path.parent.name
        enriched_rows = attach_global_iterations(
            rows,
            start_iteration=start_iteration,
            run_name=run_name,
        )
        combined_rows.extend(enriched_rows)

        segments.append(
            {
                "run_name": run_name,
                "jsonl_path": str(path),
                "start_iteration": start_iteration,
                "end_iteration": start_iteration + len(rows) - 1 if rows else start_iteration - 1,
                "num_points": len(rows),
            }
        )

    combined_rows.sort(key=lambda row: row["global_iteration"])
    return combined_rows, segments


def save_combined_metrics(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    segments: list[dict[str, Any]],
) -> dict[str, Path]:
    jsonl_path = output_dir / "combined_metrics.jsonl"
    csv_path = output_dir / "combined_metrics.csv"
    summary_path = output_dir / "combined_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "num_points": len(rows),
        "first_global_iteration": rows[0]["global_iteration"] if rows else None,
        "last_global_iteration": rows[-1]["global_iteration"] if rows else None,
        "source_runs": [segment["run_name"] for segment in segments],
        "segments": segments,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"jsonl": jsonl_path, "csv": csv_path, "summary": summary_path}


def moving_average_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(y, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def fit_spline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    smoothing_scale: float,
    pre_smooth_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < 4:
        return x, y
    y_for_fit = moving_average_smooth(y, pre_smooth_window)
    variance = float(np.var(y_for_fit))
    if variance <= 1e-12:
        dense_x = np.linspace(x.min(), x.max(), 300)
        dense_y = np.interp(dense_x, x, y_for_fit)
        return dense_x, dense_y

    smoothing = len(x) * variance * smoothing_scale
    spline = UnivariateSpline(x, y_for_fit, s=smoothing)
    dense_x = np.linspace(x.min(), x.max(), 400)
    dense_y = spline(dense_x)
    return dense_x, dense_y


def plot_metrics(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    segments: list[dict[str, Any]],
    smoothing_scale: float,
    pre_smooth_window: int,
    figure_title: str | None,
) -> Path:
    x = np.array([row["global_iteration"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes = axes.ravel()

    boundary_xs = [segment["start_iteration"] for segment in segments[1:] if segment["num_points"] > 0]

    for ax, (key, metric_title, color) in zip(axes, METRIC_SPECS):
        y = np.array([row.get(key, np.nan) for row in rows], dtype=float)
        valid = ~np.isnan(y)
        xv = x[valid]
        yv = y[valid]

        ax.plot(
            xv,
            yv,
            color=color,
            alpha=0.14,
            linewidth=1.0,
            label="raw",
        )
        smooth_x, smooth_y = fit_spline(
            xv,
            yv,
            smoothing_scale=smoothing_scale,
            pre_smooth_window=pre_smooth_window,
        )
        ax.plot(smooth_x, smooth_y, color=color, linewidth=3.0, label="moving-average + spline")

        for boundary_x in boundary_xs:
            ax.axvline(boundary_x, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)

        ax.set_title(metric_title)
        ax.set_ylabel(metric_title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    for ax in axes[-2:]:
        ax.set_xlabel("Global Iteration")

    auto_title = "Combined Training Trends"
    if rows:
        auto_title = (
            f"Combined Training Trends: Iterations "
            f"{rows[0]['global_iteration']}-{rows[-1]['global_iteration']}"
        )
    fig.suptitle(figure_title or auto_title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    plot_path = output_dir / "combined_training_trends_smoothed.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    ensure_fresh_output_dir(output_dir, allow_overwrite=args.allow_overwrite)
    run_specs = collect_run_specs(args)
    combined_rows, segments = enrich_runs(run_specs)

    saved = save_combined_metrics(combined_rows, output_dir, segments=segments)
    plot_path = plot_metrics(
        combined_rows,
        output_dir,
        segments=segments,
        smoothing_scale=args.smoothing_scale,
        pre_smooth_window=args.pre_smooth_window,
        figure_title=args.title,
    )

    print(json.dumps(
        {
            "output_dir": str(output_dir),
            "plot": str(plot_path),
            "combined_csv": str(saved["csv"]),
            "combined_jsonl": str(saved["jsonl"]),
            "summary": str(saved["summary"]),
            "num_points": len(combined_rows),
            "segments": segments,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
