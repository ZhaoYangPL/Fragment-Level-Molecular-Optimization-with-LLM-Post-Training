#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ai4materials")
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except ImportError:  # pragma: no cover - handled at runtime for non-RDKit envs
    Chem = None
    Draw = None


PROPERTIES = ("QED", "LogP", "MW")
BLOCKS = ("ANALYSE", "PLAN", "EDIT", "VERIFY")
PROPERTY_LABELS = {
    "QED": "类药性",
    "LogP": "脂水分配系数",
    "MW": "分子量",
}
ACTION_LABELS = {
    "Replace": "替换",
    "Add": "添加",
    "Remove": "移除",
    "Unknown": "未知",
}
SPLIT_LABELS = {
    "train": "训练集",
    "val": "验证集",
}
BLOCK_LABELS = {
    "ANALYSE": "分析",
    "PLAN": "规划",
    "EDIT": "编辑",
    "VERIFY": "验证",
}
ACTION_COLORS = {
    "Replace": "#2563eb",
    "Add": "#059669",
    "Remove": "#dc2626",
}
PROPERTY_COLORS = {
    "QED": "#2563eb",
    "LogP": "#f97316",
    "MW": "#059669",
}
CHINESE_FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
CHINESE_FONT = (
    font_manager.FontProperties(fname=str(CHINESE_FONT_PATH))
    if CHINESE_FONT_PATH.exists()
    else font_manager.FontProperties(family="sans-serif")
)


def configure_chinese_fonts() -> None:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def cn_text_kwargs(**kwargs: Any) -> dict[str, Any]:
    return {"fontproperties": CHINESE_FONT, **kwargs}


def set_cn_title(ax: plt.Axes, text: str, **kwargs: Any) -> None:
    ax.set_title(text, **cn_text_kwargs(**kwargs))


def set_cn_xlabel(ax: plt.Axes, text: str, **kwargs: Any) -> None:
    ax.set_xlabel(text, **cn_text_kwargs(**kwargs))


def set_cn_ylabel(ax: plt.Axes, text: str, **kwargs: Any) -> None:
    ax.set_ylabel(text, **cn_text_kwargs(**kwargs))


def apply_cn_tick_font(ax: plt.Axes, *, x: bool = False, y: bool = False) -> None:
    if x:
        for label in ax.get_xticklabels():
            label.set_fontproperties(CHINESE_FONT)
    if y:
        for label in ax.get_yticklabels():
            label.set_fontproperties(CHINESE_FONT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the four-step QED/LogP/MW SFT dataset.",
    )
    parser.add_argument(
        "--train",
        default="fourstep_qed_logp_mw_sft_train_checkpoints/fourstep_qed_logp_mw_sft_train.jsonl",
        help="Training JSONL path.",
    )
    parser.add_argument(
        "--val",
        default="fourstep_qed_logp_mw_sft_train_checkpoints/fourstep_qed_logp_mw_sft_val.jsonl",
        help="Validation JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default="fourstep_qed_logp_mw_sft_train_checkpoints/visualizations",
        help="Directory where plots and summary files will be written.",
    )
    parser.add_argument(
        "--gallery-size",
        type=int,
        default=10,
        help="Number of examples to include in the molecule gallery.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed used when selecting gallery examples.",
    )
    return parser.parse_args()


def load_jsonl(path: Path, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_split"] = split
            record["_line_number"] = line_number
            rows.append(record)
    return rows


def signed_target_delta(target: dict[str, Any] | None) -> float:
    if not isinstance(target, dict):
        return math.nan
    delta = target.get("delta")
    direction = target.get("direction")
    if not isinstance(delta, (int, float)):
        return math.nan
    if direction == "lower":
        return -float(delta)
    if direction == "higher":
        return float(delta)
    return math.nan


def get_number(mapping: dict[str, Any] | None, key: str) -> float:
    if not isinstance(mapping, dict):
        return math.nan
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) else math.nan


def block_present(completion: str, block: str) -> bool:
    pattern = rf"<{block}>\s*.*?\s*</{block}>"
    return bool(re.search(pattern, completion or "", flags=re.DOTALL | re.IGNORECASE))


def direction_combo(row: dict[str, Any]) -> str:
    parts: list[str] = []
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    for prop in PROPERTIES:
        target = targets.get(prop) if isinstance(targets, dict) else None
        direction = target.get("direction") if isinstance(target, dict) else "?"
        if direction == "higher":
            parts.append(f"{prop}+")
        elif direction == "lower":
            parts.append(f"{prop}-")
        else:
            parts.append(f"{prop}?")
    return " ".join(parts)


def translate_direction_combo(combo: str) -> str:
    labels: list[str] = []
    for token in combo.split():
        if token.endswith("+"):
            prop = token[:-1]
            labels.append(f"{PROPERTY_LABELS.get(prop, prop)}升高")
        elif token.endswith("-"):
            prop = token[:-1]
            labels.append(f"{PROPERTY_LABELS.get(prop, prop)}降低")
        else:
            labels.append(PROPERTY_LABELS.get(token, token))
    return "，".join(labels)


def flatten_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        completion = row.get("completion") or ""
        prompt = row.get("prompt") or ""
        fragments = row.get("fragments") if isinstance(row.get("fragments"), list) else []
        flat: dict[str, Any] = {
            "split": row.get("_split"),
            "line_number": row.get("_line_number"),
            "source_index": row.get("source_index"),
            "action_type": row.get("action_type") or "Unknown",
            "target_met": bool(row.get("target_met")),
            "input_smiles": row.get("input_smiles"),
            "target_smiles": row.get("target_smiles"),
            "fragment_count": len(fragments),
            "input_smiles_len": len(row.get("input_smiles") or ""),
            "target_smiles_len": len(row.get("target_smiles") or ""),
            "prompt_chars": len(prompt),
            "completion_chars": len(completion),
            "direction_combo": direction_combo(row),
        }
        for block in BLOCKS:
            flat[f"has_{block.lower()}"] = block_present(completion, block)

        targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
        actual_deltas = (
            row.get("actual_prop_deltas")
            if isinstance(row.get("actual_prop_deltas"), dict)
            else {}
        )
        input_props = row.get("input_props") if isinstance(row.get("input_props"), dict) else {}
        output_props = row.get("output_props") if isinstance(row.get("output_props"), dict) else {}
        for prop in PROPERTIES:
            target = targets.get(prop) if isinstance(targets, dict) else None
            flat[f"{prop}_target_delta"] = signed_target_delta(target)
            flat[f"{prop}_target_abs_delta"] = abs(flat[f"{prop}_target_delta"])
            flat[f"{prop}_target_value"] = get_number(target, "target_value")
            flat[f"{prop}_actual_delta"] = get_number(actual_deltas, prop)
            flat[f"{prop}_input"] = get_number(input_props, prop)
            flat[f"{prop}_output"] = get_number(output_props, prop)
            if not math.isnan(flat[f"{prop}_target_delta"]) and not math.isnan(
                flat[f"{prop}_actual_delta"]
            ):
                flat[f"{prop}_delta_error"] = (
                    flat[f"{prop}_actual_delta"] - flat[f"{prop}_target_delta"]
                )
            else:
                flat[f"{prop}_delta_error"] = math.nan
        flat_rows.append(flat)
    return flat_rows


def values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = [row.get(key) for row in rows]
    return np.asarray([float(v) for v in vals if isinstance(v, (int, float)) and not math.isnan(v)])


def annotate_bars(ax: plt.Axes, bars: Any, *, fmt: str = "{:.0f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_overview(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("四步监督微调数据集总览", **cn_text_kwargs(fontsize=18, fontweight="bold"))

    split_counts = Counter(row["split"] for row in rows)
    ax = axes[0, 0]
    splits = ["train", "val"]
    bars = ax.bar(
        [SPLIT_LABELS.get(split, split) for split in splits],
        [split_counts.get(split, 0) for split in splits],
        color=["#1d4ed8", "#16a34a"],
    )
    annotate_bars(ax, bars)
    set_cn_title(ax, "数据集规模")
    set_cn_ylabel(ax, "样本数")
    apply_cn_tick_font(ax, x=True)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    action_counts = Counter(row["action_type"] for row in rows)
    actions = [name for name, _ in action_counts.most_common()]
    bars = ax.bar(
        [ACTION_LABELS.get(name, name) for name in actions],
        [action_counts[name] for name in actions],
        color=[ACTION_COLORS.get(name, "#64748b") for name in actions],
    )
    annotate_bars(ax, bars)
    ax.set_yscale("symlog", linthresh=10)
    set_cn_title(ax, "片段编辑动作")
    set_cn_ylabel(ax, "样本数")
    apply_cn_tick_font(ax, x=True)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 2]
    combo_counts = Counter(row["direction_combo"] for row in rows)
    combo_items = combo_counts.most_common(8)
    labels = [translate_direction_combo(item[0]) for item in combo_items]
    counts = [item[1] for item in combo_items]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, counts, color="#7c3aed")
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    set_cn_title(ax, "目标变化方向组合")
    set_cn_xlabel(ax, "样本数")
    apply_cn_tick_font(ax, y=True)
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1, 0]
    box_data = [values(rows, f"{prop}_target_abs_delta") for prop in PROPERTIES]
    ax.boxplot(
        box_data,
        tick_labels=[PROPERTY_LABELS[prop] for prop in PROPERTIES],
        showfliers=False,
        patch_artist=True,
    )
    for patch, prop in zip(ax.artists, PROPERTIES):
        patch.set_facecolor(PROPERTY_COLORS[prop])
        patch.set_alpha(0.35)
    set_cn_title(ax, "目标属性变化幅度")
    set_cn_ylabel(ax, "绝对变化量（对称对数坐标）")
    ax.set_yscale("symlog", linthresh=1)
    apply_cn_tick_font(ax, x=True)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    x = values(rows, "QED_actual_delta")
    y = values(rows, "LogP_actual_delta")
    c = values(rows, "MW_actual_delta")
    if len(x) == len(y) == len(c) and len(x):
        scatter = ax.scatter(x, y, c=c, s=15, cmap="viridis", alpha=0.65, linewidths=0)
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("分子量实际变化量", **cn_text_kwargs())
    ax.axhline(0, color="#94a3b8", linewidth=1)
    ax.axvline(0, color="#94a3b8", linewidth=1)
    set_cn_title(ax, "实际变化覆盖范围")
    set_cn_xlabel(ax, "类药性实际变化量")
    set_cn_ylabel(ax, "脂水分配系数实际变化量")
    ax.grid(alpha=0.2)

    ax = axes[1, 2]
    quality_labels = ["target_met"] + [f"has_{block.lower()}" for block in BLOCKS]
    quality_values = [
        100.0 * sum(bool(row.get(label)) for row in rows) / max(len(rows), 1)
        for label in quality_labels
    ]
    display_labels = ["目标达成"] + [BLOCK_LABELS.get(block, block) for block in BLOCKS]
    bars = ax.bar(display_labels, quality_values, color="#0f766e")
    annotate_bars(ax, bars, fmt="{:.1f}%")
    ax.set_ylim(0, 105)
    set_cn_title(ax, "格式完整率")
    set_cn_ylabel(ax, "百分比")
    ax.tick_params(axis="x", rotation=25)
    apply_cn_tick_font(ax, x=True)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = output_dir / "fourstep_dataset_overview.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_property_alignment(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle("目标属性变化量与实际变化量对齐情况", **cn_text_kwargs(fontsize=17, fontweight="bold"))

    for ax, prop in zip(axes, PROPERTIES):
        target = values(rows, f"{prop}_target_delta")
        actual = values(rows, f"{prop}_actual_delta")
        if len(target) and len(target) == len(actual):
            color = PROPERTY_COLORS[prop]
            ax.scatter(target, actual, s=14, alpha=0.45, color=color, linewidths=0)
            low = min(float(np.nanmin(target)), float(np.nanmin(actual)))
            high = max(float(np.nanmax(target)), float(np.nanmax(actual)))
            margin = (high - low) * 0.08 if high > low else 0.1
            low -= margin
            high += margin
            ax.plot([low, high], [low, high], color="#111827", linewidth=1.2, label="完全一致")
            errors = actual - target
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors * errors)))
            ax.text(
                0.04,
                0.95,
                "平均绝对误差\n均方根误差",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                fontproperties=CHINESE_FONT,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cbd5e1"},
            )
            ax.text(
                0.27,
                0.95,
                f"{mae:.4g}\n{rmse:.4g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
        ax.axhline(0, color="#cbd5e1", linewidth=0.8)
        ax.axvline(0, color="#cbd5e1", linewidth=0.8)
        set_cn_title(ax, PROPERTY_LABELS[prop])
        set_cn_xlabel(ax, "目标有符号变化量")
        set_cn_ylabel(ax, "实际有符号变化量")
        ax.grid(alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = output_dir / "fourstep_property_delta_alignment.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_structure_diagnostics(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("四步数据集结构诊断", **cn_text_kwargs(fontsize=17, fontweight="bold"))

    ax = axes[0, 0]
    fragment_counts = [int(row["fragment_count"]) for row in rows]
    bins = np.arange(min(fragment_counts), max(fragment_counts) + 2) - 0.5 if fragment_counts else 10
    ax.hist(fragment_counts, bins=bins, color="#2563eb", alpha=0.8)
    set_cn_title(ax, "每个输入分子的片段数量")
    set_cn_xlabel(ax, "片段数量")
    set_cn_ylabel(ax, "样本数")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    input_len = values(rows, "input_smiles_len")
    target_len = values(rows, "target_smiles_len")
    bins = np.linspace(
        min(np.min(input_len), np.min(target_len)),
        max(np.max(input_len), np.max(target_len)),
        35,
    )
    ax.hist(input_len, bins=bins, alpha=0.55, label="输入分子", color="#2563eb")
    ax.hist(target_len, bins=bins, alpha=0.55, label="目标分子", color="#f97316")
    set_cn_title(ax, "分子字符串长度")
    set_cn_xlabel(ax, "字符数")
    set_cn_ylabel(ax, "样本数")
    ax.legend(prop=CHINESE_FONT)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    prompt_chars = values(rows, "prompt_chars")
    completion_chars = values(rows, "completion_chars")
    ax.scatter(prompt_chars, completion_chars, s=12, alpha=0.4, color="#0f766e", linewidths=0)
    set_cn_title(ax, "提示词长度与回答长度")
    set_cn_xlabel(ax, "提示词字符数")
    set_cn_ylabel(ax, "回答字符数")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    error_data = [values(rows, f"{prop}_delta_error") for prop in PROPERTIES]
    ax.boxplot(
        error_data,
        tick_labels=[PROPERTY_LABELS[prop] for prop in PROPERTIES],
        showfliers=False,
    )
    ax.axhline(0, color="#111827", linewidth=1)
    set_cn_title(ax, "实际变化量减目标变化量")
    set_cn_ylabel(ax, "误差")
    apply_cn_tick_font(ax, x=True)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "fourstep_structure_diagnostics.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def choose_gallery_examples(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("split") == "train" and row.get("input_smiles") and row.get("target_smiles")
    ]
    by_action: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_action.setdefault(str(row.get("action_type")), []).append(row)

    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for action in sorted(by_action):
        candidates = by_action[action]
        if not candidates:
            continue
        selected.append(candidates[int(rng.integers(0, len(candidates)))])

    remaining = [row for row in eligible if row not in selected]
    if remaining and len(selected) < n:
        indices = rng.choice(len(remaining), size=min(n - len(selected), len(remaining)), replace=False)
        selected.extend(remaining[int(index)] for index in np.atleast_1d(indices))

    return selected[:n]


def save_molecule_gallery(rows: list[dict[str, Any]], output_dir: Path, n: int, seed: int) -> Path | None:
    if Chem is None or Draw is None:
        return None

    examples = choose_gallery_examples(rows, n, seed)
    panels: list[dict[str, Any]] = []
    for index, row in enumerate(examples, start=1):
        input_mol = Chem.MolFromSmiles(str(row.get("input_smiles")))
        target_mol = Chem.MolFromSmiles(str(row.get("target_smiles")))
        if input_mol is None or target_mol is None:
            continue
        action = row.get("action_type")
        q_delta = row.get("QED_actual_delta")
        logp_delta = row.get("LogP_actual_delta")
        mw_delta = row.get("MW_actual_delta")
        panels.append(
            {
                "index": index,
                "action": ACTION_LABELS.get(str(action), str(action)),
                "input_image": Draw.MolToImage(input_mol, size=(420, 260)),
                "target_image": Draw.MolToImage(target_mol, size=(420, 260)),
                "q_delta": q_delta,
                "logp_delta": logp_delta,
                "mw_delta": mw_delta,
            }
        )

    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), 2, figsize=(8.8, max(2.8 * len(panels), 3.2)))
    if len(panels) == 1:
        axes = np.asarray([axes])
    fig.suptitle("分子编辑示例", **cn_text_kwargs(fontsize=18, fontweight="bold"))

    for row_axes, panel in zip(axes, panels):
        input_ax, target_ax = row_axes
        input_ax.imshow(np.asarray(panel["input_image"]))
        target_ax.imshow(np.asarray(panel["target_image"]))
        for ax in row_axes:
            ax.axis("off")

        input_ax.set_title(
            f"输入分子：{panel['action']}",
            **cn_text_kwargs(fontsize=11),
        )
        target_ax.set_title("目标分子", **cn_text_kwargs(fontsize=11))
        target_ax.text(
            0.02,
            -0.02,
            "类药性变化\n脂水变化\n分子量变化",
            transform=target_ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontproperties=CHINESE_FONT,
        )
        target_ax.text(
            0.36,
            -0.02,
            f"{panel['q_delta']:+.3f}\n{panel['logp_delta']:+.2f}\n{panel['mw_delta']:+.1f}",
            transform=target_ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=2.0)
    path = output_dir / "fourstep_molecule_edit_gallery.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "fourstep_dataset_records.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_examples": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "action_counts": dict(Counter(row["action_type"] for row in rows)),
        "direction_combo_counts": dict(Counter(row["direction_combo"] for row in rows)),
        "target_met_rate": sum(bool(row["target_met"]) for row in rows) / max(len(rows), 1),
        "block_presence_rates": {
            block: sum(bool(row.get(f"has_{block.lower()}")) for row in rows) / max(len(rows), 1)
            for block in BLOCKS
        },
        "property_stats": {},
    }
    for prop in PROPERTIES:
        target = values(rows, f"{prop}_target_delta")
        actual = values(rows, f"{prop}_actual_delta")
        error = actual - target if len(target) == len(actual) else np.asarray([])
        summary["property_stats"][prop] = {
            "target_delta_min": float(np.min(target)) if len(target) else None,
            "target_delta_max": float(np.max(target)) if len(target) else None,
            "actual_delta_min": float(np.min(actual)) if len(actual) else None,
            "actual_delta_max": float(np.max(actual)) if len(actual) else None,
            "delta_error_mae": float(np.mean(np.abs(error))) if len(error) else None,
            "delta_error_rmse": float(np.sqrt(np.mean(error * error))) if len(error) else None,
        }
    return summary


def save_summary(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "fourstep_dataset_summary.json"
    path.write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    configure_chinese_fonts()
    args = parse_args()
    train_path = Path(args.train)
    val_path = Path(args.val)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_jsonl(train_path, "train") + load_jsonl(val_path, "val")
    rows = flatten_records(raw_rows)

    outputs: list[Path] = [
        save_overview(rows, output_dir),
        save_property_alignment(rows, output_dir),
        save_structure_diagnostics(rows, output_dir),
        save_csv(rows, output_dir),
        save_summary(rows, output_dir),
    ]
    gallery = save_molecule_gallery(rows, output_dir, args.gallery_size, args.seed)
    if gallery is not None:
        outputs.append(gallery)

    print(f"Loaded {len(rows)} examples from {train_path} and {val_path}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
