#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

import dataset as dataset_mod


NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TARGET_PATTERN = re.compile(
    rf"HOMO\s+value\s*<HOMO>\s*(?P<homo>{NUMBER_PATTERN})\s*</HOMO>\s*"
    rf"(?P<homo_direction>higher|lower)\s*,?\s*and\s+"
    rf"LUMO\s+value\s*<LUMO>\s*(?P<lumo>{NUMBER_PATTERN})\s*</LUMO>\s*"
    rf"(?P<lumo_direction>higher|lower)",
    re.IGNORECASE,
)
QUANTILES = (
    ("p0", 0.0),
    ("p0.1", 0.001),
    ("p1", 0.01),
    ("p5", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p99.9", 0.999),
    ("p100", 1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze HOMO/LUMO target changes in the local GRPO dataset."
    )
    parser.add_argument(
        "--local-data-dir",
        default=str(dataset_mod.DEFAULT_LOCAL_DATA_DIR),
        help="Directory containing HuggingFace save_to_disk output.",
    )
    parser.add_argument("--data-split", default=dataset_mod.DEFAULT_DATA_SPLIT)
    parser.add_argument("--prompt-col", default=dataset_mod.DEFAULT_PROMPT_COL)
    parser.add_argument("--answer-col", default=dataset_mod.DEFAULT_ANSWER_COL)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Analysis output directory. Defaults to <local-data-dir>/analysis.",
    )
    parser.add_argument(
        "--histogram-width",
        type=float,
        default=0.5,
        help="Fixed histogram bin width for all distributions.",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Resolved HuggingFace revision used to create the local dataset.",
    )
    parser.add_argument(
        "--source-status",
        default="local save_to_disk dataset",
        help="Short provenance note included in the report.",
    )
    return parser.parse_args()


def parse_target(prompt: str) -> dict[str, Any] | None:
    match = TARGET_PATTERN.search(prompt)
    if match is None:
        return None

    homo_magnitude = float(match.group("homo"))
    lumo_magnitude = float(match.group("lumo"))
    homo_direction = match.group("homo_direction").lower()
    lumo_direction = match.group("lumo_direction").lower()
    return {
        "homo_magnitude": homo_magnitude,
        "homo_direction": homo_direction,
        "homo_signed_change": homo_magnitude if homo_direction == "higher" else -homo_magnitude,
        "lumo_magnitude": lumo_magnitude,
        "lumo_direction": lumo_direction,
        "lumo_signed_change": lumo_magnitude if lumo_direction == "higher" else -lumo_magnitude,
    }


def metric_statistics(values: np.ndarray) -> dict[str, Any]:
    quantile_values = np.quantile(values, [value for _, value in QUANTILES])
    return {
        "count": int(values.size),
        "finite_count": int(np.isfinite(values).sum()),
        "zero_count": int(np.count_nonzero(values == 0.0)),
        "negative_count": int(np.count_nonzero(values < 0.0)),
        "positive_count": int(np.count_nonzero(values > 0.0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std_population": float(values.std(ddof=0)),
        "quantiles": {
            label: float(value)
            for (label, _), value in zip(QUANTILES, quantile_values, strict=True)
        },
    }


def histogram_edges(values: np.ndarray, width: float, *, nonnegative: bool) -> np.ndarray:
    if width <= 0:
        raise ValueError(f"Histogram width must be positive, got {width}.")

    lower = 0.0 if nonnegative else math.floor(float(values.min()) / width) * width
    upper = math.ceil(float(values.max()) / width) * width
    if upper <= lower:
        upper = lower + width

    bin_count = int(round((upper - lower) / width))
    return np.linspace(lower, upper, bin_count + 1)


def write_targets_csv(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "row_index",
        "homo_magnitude",
        "homo_direction",
        "homo_signed_change",
        "lumo_magnitude",
        "lumo_direction",
        "lumo_signed_change",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_histograms_csv(
    distributions: list[tuple[str, str, np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    fieldnames = [
        "property",
        "representation",
        "bin_left",
        "bin_right",
        "right_edge_inclusive",
        "count",
        "fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for property_name, representation, values, edges in distributions:
            counts, _ = np.histogram(values, bins=edges)
            for index, count in enumerate(counts):
                writer.writerow(
                    {
                        "property": property_name,
                        "representation": representation,
                        "bin_left": float(edges[index]),
                        "bin_right": float(edges[index + 1]),
                        "right_edge_inclusive": index == len(counts) - 1,
                        "count": int(count),
                        "fraction": float(count / values.size),
                    }
                )


def save_distribution_plot(
    distributions: list[tuple[str, str, np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        ("HOMO", "magnitude"): "#386cb0",
        ("LUMO", "magnitude"): "#7fc97f",
        ("HOMO", "signed_change"): "#f0027f",
        ("LUMO", "signed_change"): "#fdc086",
    }

    for axis, (property_name, representation, values, edges) in zip(
        axes.flat, distributions, strict=True
    ):
        axis.hist(values, bins=edges, color=colors[(property_name, representation)], alpha=0.85)
        axis.axvline(float(np.median(values)), color="black", linestyle="--", linewidth=1.2)
        if representation == "signed_change":
            axis.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0)
        axis.set_title(f"{property_name} {representation.replace('_', ' ')}")
        axis.set_xlabel("Target change value")
        axis.set_ylabel("Sample count")
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("HOMO-LUMO target-change distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def format_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def markdown_metric_row(name: str, stats: dict[str, Any]) -> str:
    quantiles = stats["quantiles"]
    return (
        f"| {name} | {stats['count']} | {format_number(stats['min'])} | "
        f"{format_number(quantiles['p25'])} | {format_number(quantiles['p50'])} | "
        f"{format_number(quantiles['p75'])} | {format_number(quantiles['p95'])} | "
        f"{format_number(stats['max'])} | {format_number(stats['mean'])} | "
        f"{format_number(stats['std_population'])} |"
    )


def write_markdown_report(statistics: dict[str, Any], path: Path) -> None:
    metrics = statistics["metrics"]
    combinations = statistics["directions"]["combinations"]
    source_revision = statistics["source"].get("resolved_revision") or "未提供"

    lines = [
        "# HOMO-LUMO 数据分布分析",
        "",
        "## 数据与口径",
        "",
        f"- 数据集：`{statistics['dataset']['name']}`",
        f"- split：`{statistics['dataset']['split']}`",
        f"- 样本数：{statistics['dataset']['row_count']}",
        f"- 本地目录：`{statistics['dataset']['local_data_dir']}`",
        f"- Hugging Face revision：`{source_revision}`",
        f"- 来源状态：{statistics['source']['status']}",
        "- `<HOMO>`、`<LUMO>` 被解释为 prompt 要求的变化幅度，不是绝对轨道能量。",
        "- 带符号变化量定义为：`higher = 正值`，`lower = 负值`。",
        "",
        "## 解析验证",
        "",
        f"- 成功解析：{statistics['parsing']['parsed_count']}",
        f"- 解析失败：{statistics['parsing']['failed_count']}",
        f"- 非有限数值：{statistics['parsing']['nonfinite_value_count']}",
        f"- 负幅度：{statistics['parsing']['negative_magnitude_count']}",
        "",
        "## 核心统计",
        "",
        "| 指标 | 数量 | 最小值 | P25 | 中位数 | P75 | P95 | 最大值 | 均值 | 总体标准差 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        markdown_metric_row("HOMO 幅度", metrics["homo_magnitude"]),
        markdown_metric_row("LUMO 幅度", metrics["lumo_magnitude"]),
        markdown_metric_row("HOMO 带符号变化", metrics["homo_signed_change"]),
        markdown_metric_row("LUMO 带符号变化", metrics["lumo_signed_change"]),
        "",
        "## 方向分布",
        "",
        f"- HOMO higher：{statistics['directions']['homo']['higher']}",
        f"- HOMO lower：{statistics['directions']['homo']['lower']}",
        f"- LUMO higher：{statistics['directions']['lumo']['higher']}",
        f"- LUMO lower：{statistics['directions']['lumo']['lower']}",
        "",
        "| HOMO 方向 | LUMO 方向 | 数量 | 比例 |",
        "|---|---|---:|---:|",
    ]
    for key in ("higher|higher", "higher|lower", "lower|higher", "lower|lower"):
        homo_direction, lumo_direction = key.split("|")
        count = combinations.get(key, 0)
        fraction = count / statistics["dataset"]["row_count"]
        lines.append(f"| {homo_direction} | {lumo_direction} | {count} | {fraction:.4%} |")

    lines.extend(
        [
            "",
            "## 分布补充",
            "",
            f"- HOMO/LUMO 幅度相关系数：{statistics['correlations']['magnitude_pearson']:.6f}",
            f"- HOMO/LUMO 带符号变化相关系数：{statistics['correlations']['signed_change_pearson']:.6f}",
            f"- 唯一 prompt：{statistics['duplicates']['unique_prompts']}",
            f"- 唯一 output：{statistics['duplicates']['unique_outputs']}",
            f"- 固定直方图宽度：{statistics['histogram']['bin_width']}",
            "",
            "完整分位数、直方图区间计数和逐样本解析结果分别保存在 "
            "`homo_lumo_statistics.json`、`homo_lumo_histograms.csv` 和 "
            "`homo_lumo_targets.csv`；可视化保存在 `homo_lumo_distributions.png`。",
            "",
            "本报告仅描述数据分布，不包含 reward 公式或分桶建议。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    local_data_dir = Path(args.local_data_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else local_data_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(local_data_dir))
    split = dataset_mod._get_split(dataset, args.data_split)
    dataset_mod.validate_columns(split, prompt_col=args.prompt_col, answer_col=args.answer_col)

    records: list[dict[str, Any]] = []
    failed_rows: list[int] = []
    for row_index, prompt in enumerate(split[args.prompt_col]):
        parsed = parse_target(prompt)
        if parsed is None:
            failed_rows.append(row_index)
            continue
        records.append({"row_index": row_index, **parsed})

    if failed_rows:
        preview = ", ".join(str(index) for index in failed_rows[:20])
        raise ValueError(
            f"Failed to parse {len(failed_rows)} prompts. First row indices: {preview}"
        )

    homo_magnitude = np.asarray([row["homo_magnitude"] for row in records], dtype=float)
    lumo_magnitude = np.asarray([row["lumo_magnitude"] for row in records], dtype=float)
    homo_signed = np.asarray([row["homo_signed_change"] for row in records], dtype=float)
    lumo_signed = np.asarray([row["lumo_signed_change"] for row in records], dtype=float)
    all_values = np.column_stack(
        [homo_magnitude, lumo_magnitude, homo_signed, lumo_signed]
    )

    nonfinite_count = int(np.count_nonzero(~np.isfinite(all_values)))
    negative_magnitude_count = int(
        np.count_nonzero(homo_magnitude < 0.0) + np.count_nonzero(lumo_magnitude < 0.0)
    )
    if len(records) != len(split) or nonfinite_count or negative_magnitude_count:
        raise ValueError(
            "Dataset validation failed: "
            f"rows={len(split)}, parsed={len(records)}, nonfinite={nonfinite_count}, "
            f"negative_magnitudes={negative_magnitude_count}."
        )

    homo_directions = Counter(row["homo_direction"] for row in records)
    lumo_directions = Counter(row["lumo_direction"] for row in records)
    direction_combinations = Counter(
        f"{row['homo_direction']}|{row['lumo_direction']}" for row in records
    )
    statistics = {
        "dataset": {
            "name": dataset_mod.DEFAULT_DATASET_NAME,
            "split": args.data_split,
            "row_count": len(split),
            "columns": list(split.column_names),
            "local_data_dir": str(local_data_dir),
        },
        "source": {
            "requested_revision": "main",
            "resolved_revision": args.source_revision,
            "status": args.source_status,
        },
        "interpretation": {
            "tag_values": "target change magnitudes, not absolute orbital energies",
            "signed_change": {"higher": "positive", "lower": "negative"},
        },
        "parsing": {
            "parsed_count": len(records),
            "failed_count": len(failed_rows),
            "failed_row_indices": failed_rows,
            "nonfinite_value_count": nonfinite_count,
            "negative_magnitude_count": negative_magnitude_count,
        },
        "metrics": {
            "homo_magnitude": metric_statistics(homo_magnitude),
            "lumo_magnitude": metric_statistics(lumo_magnitude),
            "homo_signed_change": metric_statistics(homo_signed),
            "lumo_signed_change": metric_statistics(lumo_signed),
        },
        "directions": {
            "homo": dict(sorted(homo_directions.items())),
            "lumo": dict(sorted(lumo_directions.items())),
            "combinations": dict(sorted(direction_combinations.items())),
        },
        "correlations": {
            "magnitude_pearson": float(np.corrcoef(homo_magnitude, lumo_magnitude)[0, 1]),
            "signed_change_pearson": float(np.corrcoef(homo_signed, lumo_signed)[0, 1]),
        },
        "duplicates": {
            "unique_prompts": len(set(split[args.prompt_col])),
            "duplicate_prompt_rows": len(split) - len(set(split[args.prompt_col])),
            "unique_outputs": len(set(split[args.answer_col])),
            "duplicate_output_rows": len(split) - len(set(split[args.answer_col])),
        },
        "histogram": {
            "bin_width": args.histogram_width,
            "interval_convention": "[left, right), except the final bin includes its right edge",
        },
    }

    distributions = [
        (
            "HOMO",
            "magnitude",
            homo_magnitude,
            histogram_edges(homo_magnitude, args.histogram_width, nonnegative=True),
        ),
        (
            "LUMO",
            "magnitude",
            lumo_magnitude,
            histogram_edges(lumo_magnitude, args.histogram_width, nonnegative=True),
        ),
        (
            "HOMO",
            "signed_change",
            homo_signed,
            histogram_edges(homo_signed, args.histogram_width, nonnegative=False),
        ),
        (
            "LUMO",
            "signed_change",
            lumo_signed,
            histogram_edges(lumo_signed, args.histogram_width, nonnegative=False),
        ),
    ]

    targets_path = output_dir / "homo_lumo_targets.csv"
    statistics_path = output_dir / "homo_lumo_statistics.json"
    histograms_path = output_dir / "homo_lumo_histograms.csv"
    plot_path = output_dir / "homo_lumo_distributions.png"
    report_path = output_dir / "homo_lumo_analysis.md"

    write_targets_csv(records, targets_path)
    statistics_path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_histograms_csv(distributions, histograms_path)
    save_distribution_plot(distributions, plot_path)
    write_markdown_report(statistics, report_path)

    print(f"Analyzed {len(records)} rows with 0 parse failures.")
    for artifact in (
        targets_path,
        statistics_path,
        histograms_path,
        plot_path,
        report_path,
    ):
        print(f"Saved: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
