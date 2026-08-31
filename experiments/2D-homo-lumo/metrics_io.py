from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "run"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(val) for val in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


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
                "Choose a new --run-tag or pass --allow-overwrite."
            )

    run_dir.mkdir(parents=True, exist_ok=True)


def _summarize_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_points": len(metrics),
        "metric_keys": sorted({key for row in metrics for key in row.keys()}),
    }
    if not metrics:
        return summary

    summary["first_iteration"] = metrics[0].get("iteration")
    summary["last_iteration"] = metrics[-1].get("iteration")

    numeric_keys = sorted(
        {
            key
            for row in metrics
            for key, value in row.items()
            if isinstance(value, (int, float))
        }
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in metrics if isinstance(row.get(key), (int, float))]
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
    output_dir: str | Path,
    run_name: str,
    metadata: dict[str, Any] | None = None,
    allow_overwrite: bool = False,
) -> dict[str, str]:
    clean_metrics = [_to_builtin(row) for row in metrics]
    output_root = Path(output_dir)
    final_run_name = slugify(run_name)
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
        writer.writerows(clean_metrics)

    return {
        "run_dir": str(run_dir),
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
    }
