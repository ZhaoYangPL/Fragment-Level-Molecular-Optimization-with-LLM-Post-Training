#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


csv.field_size_limit(sys.maxsize)

ROOT_DIR = Path(__file__).resolve().parent
METRICS_ROOT = ROOT_DIR / "out" / "training_metrics"
OUTPUT_PATH = ROOT_DIR / "out" / "explicit_kl_pilot_selection.json"
LAST_N = 50
NTE_TOLERANCE = 0.03
VALIDITY_TIE_TOLERANCE = 0.002
PILOTS = (
    (0.0, "pilot11000_explicitkl_beta0"),
    (0.01, "pilot11000_explicitkl_beta001"),
    (0.03, "pilot11000_explicitkl_beta003"),
    (0.05, "pilot11000_explicitkl_beta005"),
)


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _weighted_rate(rows: list[dict[str, str]], numerator: str, denominator: str) -> float:
    total_numerator = sum(_float(row, numerator) or 0.0 for row in rows)
    total_denominator = sum(_float(row, denominator) or 0.0 for row in rows)
    if total_denominator <= 0:
        raise ValueError(f"No denominator values available for {denominator}")
    return total_numerator / total_denominator


def _weighted_mean(rows: list[dict[str, str]], value_key: str, weight_key: str) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        value = _float(row, value_key)
        weight = _float(row, weight_key)
        if value is None or weight is None or weight <= 0:
            continue
        weighted_sum += value * weight
        total_weight += weight
    if total_weight <= 0:
        raise ValueError(f"No weighted values available for {value_key}")
    return weighted_sum / total_weight


def _summarize(beta: float, run_tag: str) -> dict[str, float | int | str]:
    path = METRICS_ROOT / f"multiturn_{run_tag}" / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pilot metrics: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < LAST_N:
        raise ValueError(f"Pilot {run_tag} has {len(rows)} rows; expected at least {LAST_N}")
    tail = rows[-LAST_N:]
    return {
        "beta": beta,
        "run_tag": run_tag,
        "metrics_path": str(path),
        "rows_total": len(rows),
        "turn_valid_rate_last50": _weighted_rate(
            tail, "num_turn_valid", "num_turn_total"
        ),
        "terminal_valid_rate_last50": _weighted_rate(
            tail, "num_final_valid", "num_terminal_total"
        ),
        "mean_nte_last50": _weighted_mean(
            tail, "mean_nte", "num_final_valid"
        ),
    }


def main() -> int:
    summaries = [_summarize(beta, run_tag) for beta, run_tag in PILOTS]
    baseline = next(item for item in summaries if item["beta"] == 0.0)
    baseline_nte = float(baseline["mean_nte_last50"])
    maximum_eligible_nte = baseline_nte * (1.0 + NTE_TOLERANCE)
    eligible = [
        item
        for item in summaries
        if float(item["mean_nte_last50"]) <= maximum_eligible_nte
    ]
    if not eligible:
        raise RuntimeError("No KL pilot satisfies the terminal NTE tolerance")

    best_validity = max(float(item["turn_valid_rate_last50"]) for item in eligible)
    near_best = [
        item
        for item in eligible
        if best_validity - float(item["turn_valid_rate_last50"])
        <= VALIDITY_TIE_TOLERANCE
    ]
    selected = min(near_best, key=lambda item: float(item["beta"]))
    payload = {
        "selection_rule": {
            "last_iterations": LAST_N,
            "terminal_nte_max_relative_to_beta0": NTE_TOLERANCE,
            "turn_validity_tie_tolerance": VALIDITY_TIE_TOLERANCE,
            "tie_breaker": "smaller_beta",
        },
        "baseline_mean_nte_last50": baseline_nte,
        "maximum_eligible_mean_nte_last50": maximum_eligible_nte,
        "pilots": summaries,
        "selected": selected,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved pilot selection to: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
