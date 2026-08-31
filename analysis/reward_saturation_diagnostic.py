#!/usr/bin/env python3
"""
Reward saturation diagnostics for the nonsubprocess GRPO notebook.

Typical notebook usage after the reward/model setup cells have run:

    import importlib
    import reward_saturation_diagnostic as rsd

    rsd = importlib.reload(rsd)
    records, summary, artifacts = rsd.run_reward_saturation_diagnostic(
        globals(),
        checkpoint="800",          # or "700"
        num_prompts=32,
        samples_per_prompt=12,
        temperature=0.9,
        max_tokens=128,
    )

This module does not train. It only samples and records reward components.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


DEFAULT_CHECKPOINT_PATHS = {
    "700": (
        "tinker://e98c277f-2a86-5ecf-a47d-a9d437ee5678:train:0/"
        "weights/MultistepRL-grpo-Weighted-NoRep.weekend.Qwen8B.0700"
    ),
    "800": (
        "tinker://e98c277f-2a86-5ecf-a47d-a9d437ee5678:train:1/"
        "weights/Stage3MultistepRL-grpo-Weighted-NoRep.weekend.Qwen8B.0800"
    ),
}

DEFAULT_STAGE23_REWARD_WEIGHTS = {
    "format": 0.05,
    "property": 0.75,
    "similarity": 0.20,
    "repetition": 0.00,
}

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def normalize_completion_text(text: str | None) -> str:
    if not text:
        return ""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return clean.replace("<|im_end|>", "").strip()


def extract_smiles(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"<SMILES>(.*?)</SMILES>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def strict_format_reward_one(text: str) -> float:
    pat = re.compile(
        r"^(Replace|Add|Remove)\s+(.+?)(?:\s+with\s+(.+?))?\s+to\s+form\s+<SMILES>(.+)</SMILES>\.$",
        re.DOTALL,
    )
    return 1.0 if pat.fullmatch(text.strip()) else -4.0


def similarity_reward(mol_pred: Any, mol_target: Any) -> float:
    if mol_pred is None or mol_target is None:
        return 0.0
    fp_pred = MORGAN_GEN.GetFingerprint(mol_pred)
    fp_target = MORGAN_GEN.GetFingerprint(mol_target)
    return float(DataStructs.TanimotoSimilarity(fp_pred, fp_target))


def _parse_target_delta(prompt_text: str, prop: str) -> tuple[float, int] | None:
    match = re.search(
        rf"<{prop}>\s*({FLOAT_RE})\s*</{prop}>\s*(higher|lower)",
        prompt_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    delta = float(match.group(1))
    direction = 1 if match.group(2).lower() == "higher" else -1
    return delta, direction


def _round_or_none(value: Any, ndigits: int = 8) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return round(float(value), ndigits)
    return None


def compute_reward_saturation_row(
    completion_text: str,
    prompt_text: str,
    ground_truth: str,
    *,
    reward_weights: dict[str, float] | None = None,
    err_cap: float = 3.0,
    direction_penalty: float = 4.0,
    soft_format_floor: float = -4.0,
    soft_property_floor: float = -2.5,
    total_reward_floor: float = -2.0,
) -> dict[str, Any]:
    """
    Recompute the reward internals used by the current nonsubprocess reward.

    The main diagnostic fields requested by the experiment are:
    qed_err, logp_err, mw_err, dir_pen_qed, dir_pen_logp, dir_pen_mw,
    total_err, bounded_err, property_r, weighted_total, final_total.
    """
    reward_weights = dict(reward_weights or DEFAULT_STAGE23_REWARD_WEIGHTS)
    text = normalize_completion_text(completion_text)
    pred_smiles = extract_smiles(text)
    target_smiles = extract_smiles(ground_truth) or str(ground_truth).strip()

    base_row: dict[str, Any] = {
        "completion": text,
        "pred_smiles": pred_smiles,
        "target_smiles": target_smiles,
        "has_smiles": bool(pred_smiles),
        "valid_smiles": False,
        "qed_err": None,
        "logp_err": None,
        "mw_err": None,
        "dir_pen_qed": None,
        "dir_pen_logp": None,
        "dir_pen_mw": None,
        "total_err": None,
        "bounded_err": None,
        "property_r": None,
        "format_r": None,
        "similarity_r": 0.0,
        "weighted_total": None,
        "final_total": None,
        "failure_reason": None,
    }

    if not pred_smiles:
        base_row.update(
            {
                "property_r": -4.0,
                "format_r": -4.0,
                "weighted_total": -4.0,
                "final_total": -4.0,
                "failure_reason": "missing_pred_smiles",
            }
        )
        return base_row

    mol_pred = Chem.MolFromSmiles(pred_smiles)
    if mol_pred is None:
        base_row.update(
            {
                "property_r": -4.0,
                "format_r": -4.0,
                "weighted_total": -4.0,
                "final_total": -4.0,
                "failure_reason": "invalid_pred_smiles",
            }
        )
        return base_row

    base_row["valid_smiles"] = True

    mol_target = Chem.MolFromSmiles(target_smiles) if target_smiles else None
    similarity_r = similarity_reward(mol_pred, mol_target)
    format_r = max(strict_format_reward_one(text), soft_format_floor)

    in_smiles = extract_smiles(prompt_text)
    mol_in = Chem.MolFromSmiles(in_smiles) if in_smiles else None
    target_specs = {
        "qed": _parse_target_delta(prompt_text, "QED"),
        "logp": _parse_target_delta(prompt_text, "LogP"),
        "mw": _parse_target_delta(prompt_text, "MW"),
    }

    if mol_in is None or any(spec is None for spec in target_specs.values()):
        # This matches compute_reward_components for a valid molecule whose
        # property reward parser fails: multi_property_reward returns -4.0,
        # then SOFT_PROPERTY_FLOOR lifts it to -2.5 before weighting.
        property_r = max(-4.0, soft_property_floor)
        weighted_total = (
            reward_weights.get("property", 0.0) * property_r
            + reward_weights.get("format", 0.0) * format_r
            + reward_weights.get("similarity", 0.0) * similarity_r
        )
        base_row.update(
            {
                "property_r": float(property_r),
                "format_r": float(format_r),
                "similarity_r": float(similarity_r),
                "weighted_total": float(weighted_total),
                "final_total": float(max(weighted_total, total_reward_floor)),
                "failure_reason": "property_parse_failed",
            }
        )
        return base_row

    qed_in = QED.qed(mol_in)
    logp_in = Crippen.MolLogP(mol_in)
    mw_in = Descriptors.ExactMolWt(mol_in)

    qed_out = QED.qed(mol_pred)
    logp_out = Crippen.MolLogP(mol_pred)
    mw_out = Descriptors.ExactMolWt(mol_pred)

    qed_delta, d_qed = target_specs["qed"]  # type: ignore[misc]
    logp_delta, d_logp = target_specs["logp"]  # type: ignore[misc]
    mw_delta, d_mw = target_specs["mw"]  # type: ignore[misc]

    qed_target = max(0.01, min(0.99, qed_in + d_qed * qed_delta))
    logp_target = logp_in + d_logp * logp_delta
    mw_target = mw_in + d_mw * mw_delta

    qed_err = abs(qed_out - qed_target) / 0.5
    logp_err = abs(logp_out - logp_target) / 3.0
    mw_err = abs(mw_out - mw_target) / 50.0

    dir_pen_qed = direction_penalty if (qed_out - qed_in) * d_qed < 0 else 1.0
    dir_pen_logp = direction_penalty if (logp_out - logp_in) * d_logp < 0 else 1.0
    dir_pen_mw = direction_penalty if (mw_out - mw_in) * d_mw < 0 else 1.0

    total_err = (
        qed_err * dir_pen_qed
        + logp_err * dir_pen_logp
        + mw_err * dir_pen_mw
    )
    bounded_err = min(max(total_err, 0.0), err_cap)
    property_r = 1.0 - bounded_err

    weighted_total = (
        reward_weights.get("property", 0.0) * property_r
        + reward_weights.get("format", 0.0) * format_r
        + reward_weights.get("similarity", 0.0) * similarity_r
        + reward_weights.get("repetition", 0.0) * 0.0
    )
    final_total = max(weighted_total, total_reward_floor)

    base_row.update(
        {
            "input_smiles": in_smiles,
            "qed_in": _round_or_none(qed_in),
            "logp_in": _round_or_none(logp_in),
            "mw_in": _round_or_none(mw_in),
            "qed_out": _round_or_none(qed_out),
            "logp_out": _round_or_none(logp_out),
            "mw_out": _round_or_none(mw_out),
            "qed_target": _round_or_none(qed_target),
            "logp_target": _round_or_none(logp_target),
            "mw_target": _round_or_none(mw_target),
            "qed_err": float(qed_err),
            "logp_err": float(logp_err),
            "mw_err": float(mw_err),
            "dir_pen_qed": float(dir_pen_qed),
            "dir_pen_logp": float(dir_pen_logp),
            "dir_pen_mw": float(dir_pen_mw),
            "total_err": float(total_err),
            "bounded_err": float(bounded_err),
            "property_r": float(property_r),
            "format_r": float(format_r),
            "similarity_r": float(similarity_r),
            "weighted_total": float(weighted_total),
            "final_total": float(final_total),
            "failure_reason": None,
        }
    )
    return base_row


def _get_global(global_ns: dict[str, Any], name: str) -> Any:
    if name not in global_ns:
        raise KeyError(f"Notebook global {name!r} is required for this diagnostic.")
    return global_ns[name]


def _maybe_result(value: Any) -> Any:
    return value.result() if hasattr(value, "result") else value


def make_sampling_client_for_checkpoint(
    global_ns: dict[str, Any],
    *,
    checkpoint: str = "800",
    checkpoint_path: str | None = None,
    sampling_name: str | None = None,
) -> Any:
    if checkpoint == "current":
        return _get_global(global_ns, "grpo_sampling_client")

    path = checkpoint_path or DEFAULT_CHECKPOINT_PATHS.get(str(checkpoint))
    if not path:
        raise ValueError(
            f"Unknown checkpoint {checkpoint!r}. Pass checkpoint_path explicitly."
        )

    service_client = _get_global(global_ns, "service_client")
    base_model = _get_global(global_ns, "BASE_MODEL")
    lora_rank = _get_global(global_ns, "LORA_RANK")

    training_client = service_client.create_lora_training_client(
        base_model=base_model,
        rank=lora_rank,
    )
    _maybe_result(training_client.load_state(path))
    return training_client.save_weights_and_get_sampling_client(
        name=sampling_name or f"reward-saturation-{checkpoint}-{int(time.time())}"
    )


def _infer_validation_data(global_ns: dict[str, Any]) -> tuple[Any, str]:
    for name in ("validation_data", "val_data", "sft_val_raw", "val_raw"):
        if name in global_ns:
            return global_ns[name], name

    dataset = global_ns.get("dataset")
    if hasattr(dataset, "keys") and "validation" in dataset:
        return dataset["validation"], "dataset['validation']"
    if hasattr(dataset, "keys") and "test" in dataset:
        return dataset["test"], "dataset['test']"

    return _get_global(global_ns, "raw_data"), "raw_data_fallback"


def _example_at(data: Any, idx: int) -> dict[str, Any]:
    row = data[int(idx)]
    if not isinstance(row, dict):
        raise TypeError(f"Expected dict-like dataset row, got {type(row)!r}")
    return row


def select_prompt_indices(data: Any, *, num_prompts: int, seed: int) -> list[int]:
    n = len(data)
    if n <= 0:
        raise ValueError("Validation data is empty.")
    k = min(int(num_prompts), n)
    return sorted(random.Random(seed).sample(range(n), k))


def _build_generation_prompt(
    renderer: Any,
    system_prompt: str,
    prompt_text: str,
) -> Any:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ]
    return renderer.build_generation_prompt(messages)


def sample_completions_for_prompt(
    *,
    sampling_client: Any,
    renderer: Any,
    tokenizer: Any,
    types_mod: Any,
    system_prompt: str,
    prompt_text: str,
    samples_per_prompt: int = 12,
    temperature: float = 0.9,
    max_tokens: int = 128,
) -> list[str]:
    prompt_input = _build_generation_prompt(renderer, system_prompt, prompt_text)
    params_kwargs: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if hasattr(renderer, "get_stop_sequences"):
        params_kwargs["stop"] = renderer.get_stop_sequences()
    sampling_params = types_mod.SamplingParams(**params_kwargs)

    futures = [
        sampling_client.sample(
            prompt=prompt_input,
            sampling_params=sampling_params,
            num_samples=1,
        )
        for _ in range(int(samples_per_prompt))
    ]

    completions: list[str] = []
    for future in futures:
        result = future.result()
        seq = result.sequences[0]
        completions.append(
            normalize_completion_text(
                tokenizer.decode(seq.tokens, skip_special_tokens=True)
            )
        )
    return completions


def summarize_saturation(
    records: list[dict[str, Any]],
    *,
    prompt_key: str = "prompt_group_id",
    unique_round_digits: int = 6,
) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {"num_samples": 0}

    def is_close(row: dict[str, Any], key: str, target: float) -> bool:
        value = row.get(key)
        return isinstance(value, (int, float)) and math.isclose(
            float(value), target, abs_tol=1e-9
        )

    def ge(row: dict[str, Any], key: str, threshold: float) -> bool:
        value = row.get(key)
        return isinstance(value, (int, float)) and float(value) >= threshold

    def le(row: dict[str, Any], key: str, threshold: float) -> bool:
        value = row.get(key)
        return isinstance(value, (int, float)) and float(value) <= threshold

    by_prompt: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_prompt[row.get(prompt_key)].append(row)

    prompt_unique_counts: dict[str, int] = {}
    prompt_property_floor_ratios: dict[str, float] = {}
    for prompt_id, rows in by_prompt.items():
        values = []
        for row in rows:
            value = row.get("property_r")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(round(float(value), unique_round_digits))
            else:
                values.append(None)
        prompt_unique_counts[str(prompt_id)] = len(set(values))
        prompt_property_floor_ratios[str(prompt_id)] = (
            sum(is_close(row, "property_r", -2.0) for row in rows) / len(rows)
        )

    unique_hist = Counter(prompt_unique_counts.values())
    failure_hist = Counter(str(row.get("failure_reason") or "ok") for row in records)
    valid_count = sum(bool(row.get("valid_smiles")) for row in records)
    valid_records = [row for row in records if row.get("valid_smiles")]
    valid_total = len(valid_records)

    summary = {
        "num_prompts": len(by_prompt),
        "num_samples": total,
        "valid_smiles_ratio": valid_count / total,
        "property_r_eq_neg2_count": sum(is_close(row, "property_r", -2.0) for row in records),
        "property_r_eq_neg2_ratio": sum(is_close(row, "property_r", -2.0) for row in records) / total,
        "property_r_le_neg2_count": sum(le(row, "property_r", -2.0) for row in records),
        "property_r_le_neg2_ratio": sum(le(row, "property_r", -2.0) for row in records) / total,
        "total_err_ge_3_count": sum(ge(row, "total_err", 3.0) for row in records),
        "total_err_ge_3_ratio": sum(ge(row, "total_err", 3.0) for row in records) / total,
        "final_total_eq_neg2_count": sum(is_close(row, "final_total", -2.0) for row in records),
        "final_total_eq_neg2_ratio": sum(is_close(row, "final_total", -2.0) for row in records) / total,
        "final_total_le_neg2_count": sum(le(row, "final_total", -2.0) for row in records),
        "final_total_le_neg2_ratio": sum(le(row, "final_total", -2.0) for row in records) / total,
        "property_r_unique_count_by_prompt": prompt_unique_counts,
        "property_r_unique_count_histogram": {str(k): v for k, v in sorted(unique_hist.items())},
        "property_r_floor_ratio_by_prompt": prompt_property_floor_ratios,
        "failure_reason_histogram": dict(failure_hist),
    }

    if valid_total:
        summary.update(
            {
                "valid_only_property_r_eq_neg2_ratio": (
                    sum(is_close(row, "property_r", -2.0) for row in valid_records)
                    / valid_total
                ),
                "valid_only_total_err_ge_3_ratio": (
                    sum(ge(row, "total_err", 3.0) for row in valid_records)
                    / valid_total
                ),
                "valid_only_final_total_eq_neg2_ratio": (
                    sum(is_close(row, "final_total", -2.0) for row in valid_records)
                    / valid_total
                ),
            }
        )

    flags: list[str] = []
    if summary["property_r_eq_neg2_ratio"] >= 0.50:
        flags.append("property_r_floor_saturation")
    if summary["final_total_eq_neg2_ratio"] >= 0.50:
        flags.append("final_total_floor_saturation")
    if summary["total_err_ge_3_ratio"] >= 0.50:
        flags.append("total_err_cap_saturation")
    if sum(1 for v in prompt_unique_counts.values() if v <= 1) / len(prompt_unique_counts) >= 0.50:
        flags.append("low_within_prompt_property_diversity")
    summary["diagnostic_flags"] = flags
    return summary


def write_diagnostic_outputs(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    output_dir: str | Path = "out/reward_saturation_diagnostics",
    run_name: str | None = None,
) -> dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    final_name = run_name or f"reward_saturation_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / final_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = run_dir / "samples.jsonl"
    csv_path = run_dir / "samples.csv"
    summary_path = run_dir / "summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = sorted({key for row in records for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "run_dir": str(run_dir),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "summary": str(summary_path),
    }


def run_reward_saturation_diagnostic(
    global_ns: dict[str, Any],
    *,
    checkpoint: str = "800",
    checkpoint_path: str | None = None,
    validation_data: Any | None = None,
    num_prompts: int = 32,
    samples_per_prompt: int = 12,
    temperature: float = 0.9,
    max_tokens: int = 128,
    seed: int = 20260509,
    reward_weights: dict[str, float] | None = None,
    output_dir: str | Path = "out/reward_saturation_diagnostics",
    run_name: str | None = None,
    use_existing_sampling_client: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """
    Sample prompts and save saturation diagnostics.

    Pass use_existing_sampling_client=True if the notebook already loaded the
    desired checkpoint into grpo_sampling_client.
    """
    renderer = _get_global(global_ns, "renderer")
    tokenizer = _get_global(global_ns, "tokenizer")
    types_mod = _get_global(global_ns, "types")
    system_prompt = _get_global(global_ns, "SYSTEM_PROMPT_RL")
    prompt_col = global_ns.get("PROMPT_COL", "input")
    answer_col = global_ns.get("ANSWER_COL", "output")
    weights = dict(
        reward_weights
        or global_ns.get("BASE_REWARD_WEIGHTS")
        or DEFAULT_STAGE23_REWARD_WEIGHTS
    )

    data_source_name = "explicit_validation_data"
    if validation_data is None:
        validation_data, data_source_name = _infer_validation_data(global_ns)

    prompt_indices = select_prompt_indices(
        validation_data,
        num_prompts=num_prompts,
        seed=seed,
    )

    if use_existing_sampling_client:
        sampling_client = _get_global(global_ns, "grpo_sampling_client")
        checkpoint_label = "current"
    else:
        sampling_client = make_sampling_client_for_checkpoint(
            global_ns,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        )
        checkpoint_label = str(checkpoint)

    records: list[dict[str, Any]] = []
    for prompt_group_id, prompt_idx in enumerate(prompt_indices):
        example = _example_at(validation_data, prompt_idx)
        prompt_text = str(example[prompt_col])
        ground_truth = str(example[answer_col])

        completions = sample_completions_for_prompt(
            sampling_client=sampling_client,
            renderer=renderer,
            tokenizer=tokenizer,
            types_mod=types_mod,
            system_prompt=system_prompt,
            prompt_text=prompt_text,
            samples_per_prompt=samples_per_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        for sample_idx, completion_text in enumerate(completions):
            row = compute_reward_saturation_row(
                completion_text,
                prompt_text,
                ground_truth,
                reward_weights=weights,
            )
            row.update(
                {
                    "checkpoint": checkpoint_label,
                    "data_source": data_source_name,
                    "prompt_group_id": prompt_group_id,
                    "prompt_idx": int(prompt_idx),
                    "sample_idx": int(sample_idx),
                    "prompt_text": prompt_text,
                    "ground_truth": ground_truth,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "samples_per_prompt": int(samples_per_prompt),
                    "w_format": float(weights.get("format", 0.0)),
                    "w_property": float(weights.get("property", 0.0)),
                    "w_similarity": float(weights.get("similarity", 0.0)),
                    "w_repetition": float(weights.get("repetition", 0.0)),
                }
            )
            records.append(row)

        print(
            f"[rewarddiag] prompt {prompt_group_id + 1}/{len(prompt_indices)} "
            f"idx={prompt_idx} done"
        )

    summary = summarize_saturation(records)
    summary.update(
        {
            "checkpoint": checkpoint_label,
            "data_source": data_source_name,
            "prompt_indices": prompt_indices,
            "reward_weights": weights,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "samples_per_prompt": int(samples_per_prompt),
            "seed": int(seed),
        }
    )

    artifacts = write_diagnostic_outputs(
        records,
        summary,
        output_dir=output_dir,
        run_name=run_name
        or f"ckpt{checkpoint_label}_prompts{len(prompt_indices)}_G{samples_per_prompt}_{time.strftime('%Y%m%d_%H%M%S')}",
    )

    print(json.dumps({
        "property_r_eq_neg2_ratio": summary.get("property_r_eq_neg2_ratio"),
        "total_err_ge_3_ratio": summary.get("total_err_ge_3_ratio"),
        "final_total_eq_neg2_ratio": summary.get("final_total_eq_neg2_ratio"),
        "property_r_unique_count_histogram": summary.get("property_r_unique_count_histogram"),
        "diagnostic_flags": summary.get("diagnostic_flags"),
        "artifacts": artifacts,
    }, ensure_ascii=False, indent=2))
    return records, summary, artifacts
