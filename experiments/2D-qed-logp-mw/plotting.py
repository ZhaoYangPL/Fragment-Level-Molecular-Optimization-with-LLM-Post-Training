from __future__ import annotations

from pathlib import Path
from typing import Any

from metrics_io import save_training_metrics


def _smooth_series(values, window: int = 21):
    import numpy as np

    arr = np.asarray([np.nan if value is None else value for value in values], dtype=np.float64)
    if arr.size == 0 or window <= 1 or arr.size < window:
        return arr.tolist()
    kernel = np.ones(window, dtype=np.float64) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    valid = np.isfinite(arr).astype(np.float64)
    filled = np.where(np.isfinite(arr), arr, 0.0)
    padded_values = np.pad(filled, (pad_left, pad_right), mode="edge")
    padded_valid = np.pad(valid, (pad_left, pad_right), mode="edge")
    numerator = np.convolve(padded_values, kernel, mode="valid")
    denominator = np.convolve(padded_valid, kernel, mode="valid")
    smooth = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    return smooth.tolist()


def _numeric_values(all_metrics: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(value)
        for metric in all_metrics
        if isinstance((value := metric.get(key)), (int, float))
    ]


def _count_series(all_metrics: list[dict[str, Any]], key: str) -> list[float]:
    return [
        float(value) if isinstance((value := metric.get(key)), (int, float)) else 0.0
        for metric in all_metrics
    ]


def _assert_fresh_output_path(path: Path, *, kind: str, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing {kind}: {path}. Change --run-tag or pass --allow-overwrite."
        )


def save_training_outputs(
    all_metrics: list[dict[str, Any]],
    *,
    output_root: str | Path,
    run_tag: str,
    raw_data_len: int,
    metadata: dict[str, Any],
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")

    from matplotlib import pyplot as plt

    if all_metrics is None:
        raise ValueError("all_metrics is undefined.")
    if not all_metrics:
        raise ValueError("all_metrics is empty. Please run training first.")

    output_root = Path(output_root)
    save_dir = output_root / "output" / "training_plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    iters = [metric["iteration"] for metric in all_metrics]

    artifacts: dict[str, Any] = {"run_tag": run_tag}
    property_reward_mins = _numeric_values(all_metrics, "min_reward_property")
    property_reward_maxs = _numeric_values(all_metrics, "max_reward_property")
    property_reward_range_source = "turn_level"
    if not property_reward_mins or not property_reward_maxs:
        mean_property_rewards = _numeric_values(all_metrics, "mean_reward_property")
        property_reward_mins = mean_property_rewards
        property_reward_maxs = mean_property_rewards
        property_reward_range_source = "iteration_mean"
    property_reward_range = None
    if property_reward_mins and property_reward_maxs:
        property_reward_range = {
            "min": min(property_reward_mins),
            "max": max(property_reward_maxs),
            "source": property_reward_range_source,
        }
        print(
            "Property reward range "
            f"({property_reward_range_source}): "
            f"[{property_reward_range['min']:.6f}, {property_reward_range['max']:.6f}]"
        )
        artifacts["property_reward_range"] = property_reward_range

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.flatten()
    main_metrics = [
        ("mean_G", "Mean Trajectory Return (G)", "Return"),
        ("std_G", "Std of Trajectory Return", "Std"),
        ("mean_abs_adv", "Mean |Advantage|", "|Adv|"),
        ("mean_nte", "Final Molecule NTE", "NTE"),
    ]
    for ax, (key, title, ylabel) in zip(axes, main_metrics):
        values = [metric.get(key) for metric in all_metrics]
        ax.plot(iters, values, "o-", markersize=3, alpha=0.45, label="raw")
        ax.plot(iters, _smooth_series(values), linewidth=2.0, label="smooth")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    plt.suptitle("Single-stage Multi-turn GRPO Training Progress", fontsize=14)
    plt.tight_layout()
    main_plot_path = save_dir / f"training_progress_{run_tag}.png"
    _assert_fresh_output_path(main_plot_path, kind="training plot", allow_overwrite=allow_overwrite)
    fig.savefig(main_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved main training plot to: {main_plot_path}")
    artifacts["main_plot"] = str(main_plot_path)

    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes2 = axes2.flatten()
    reward_component_keys = [
        ("mean_reward_total", "Total Reward"),
        ("mean_reward_format", "Format Reward"),
        ("mean_reward_property", "Property Reward"),
        ("mean_reward_repetition", "Prompt-local Repetition Penalty"),
    ]
    for ax, (metric_key, title) in zip(axes2, reward_component_keys):
        values = [metric.get(metric_key, 0.0) for metric in all_metrics]
        ax.plot(iters, values, color="#9ecae1", linewidth=1.0, alpha=0.35, label="raw")
        ax.plot(iters, _smooth_series(values), color="#08519c", linewidth=2.0, label="smooth")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        if metric_key == "mean_reward_property" and property_reward_range is not None:
            ax.text(
                0.02,
                0.04,
                f"range: [{property_reward_range['min']:.3f}, {property_reward_range['max']:.3f}]",
                transform=ax.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
    plt.suptitle("Reward Decomposition by Iteration", fontsize=14)
    plt.tight_layout()
    reward_plot_path = save_dir / f"reward_decomposition_{run_tag}.png"
    _assert_fresh_output_path(
        reward_plot_path,
        kind="reward decomposition plot",
        allow_overwrite=allow_overwrite,
    )
    fig2.savefig(reward_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved reward decomposition plot to: {reward_plot_path}")
    artifacts["reward_plot"] = str(reward_plot_path)

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    error_metrics = [
        ("mean_final_qed_error", "Final |dQED|"),
        ("mean_final_logp_error", "Final |dLogP|"),
        ("mean_final_mw_error", "Final |dMW|"),
    ]
    for ax, (metric_key, title) in zip(axes3, error_metrics):
        values = [metric.get(metric_key) for metric in all_metrics]
        ax.plot(iters, values, "o-", markersize=3, alpha=0.45, label="raw")
        ax.plot(iters, _smooth_series(values), linewidth=2.0, label="smooth")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    plt.suptitle("Final Molecule Property Errors", fontsize=14)
    plt.tight_layout()
    error_plot_path = save_dir / f"final_property_errors_{run_tag}.png"
    _assert_fresh_output_path(error_plot_path, kind="final property error plot", allow_overwrite=allow_overwrite)
    fig3.savefig(error_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved final property error plot to: {error_plot_path}")
    artifacts["error_plot"] = str(error_plot_path)

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
    validity_specs = [
        (
            axes4[0],
            [
                ("num_final_valid", "valid", "#238b45"),
                ("num_final_missing", "missing", "#cb181d"),
                ("num_terminal_total", "total", "#636363"),
            ],
            "Terminal Molecule Counts",
        ),
        (
            axes4[1],
            [
                ("num_turn_valid", "valid", "#238b45"),
                ("num_turn_accepted", "accepted", "#756bb1"),
                ("num_turn_total", "total", "#636363"),
            ],
            "Turn-level Valid and Accepted Counts",
        ),
    ]
    for ax, series_specs, title in validity_specs:
        for key, label, color in series_specs:
            values = _count_series(all_metrics, key)
            ax.plot(iters, values, "o-", markersize=3, linewidth=1.4, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    plt.suptitle("Valid and Accepted Molecule Counts by Iteration", fontsize=14)
    plt.tight_layout()
    valid_count_plot_path = save_dir / f"valid_molecule_counts_{run_tag}.png"
    _assert_fresh_output_path(
        valid_count_plot_path,
        kind="valid molecule count plot",
        allow_overwrite=allow_overwrite,
    )
    fig4.savefig(valid_count_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig4)
    print(f"Saved valid molecule count plot to: {valid_count_plot_path}")
    artifacts["valid_count_plot"] = str(valid_count_plot_path)

    fig5, ax5 = plt.subplots(figsize=(12, 4.5), sharex=True)
    improvement_specs = [
        (
            "mean_nte_relative_improvement",
            "Terminal relative improvement",
            "#238b45",
        ),
        (
            "mean_property_relative_delta_nte_clipped",
            "Turn relative delta used by reward",
            "#08519c",
        ),
    ]
    for metric_key, label, color in improvement_specs:
        values = [metric.get(metric_key) for metric in all_metrics]
        ax5.plot(
            iters,
            values,
            "o-",
            markersize=3,
            linewidth=1.0,
            alpha=0.35,
            color=color,
            label=f"{label} raw",
        )
        ax5.plot(
            iters,
            _smooth_series(values),
            linewidth=2.0,
            color=color,
            label=f"{label} smooth",
        )
    ax5.axhline(0.0, color="#636363", linewidth=1.0, linestyle="--", alpha=0.6)
    ax5.set_title("NTE Relative Improvement")
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Relative improvement")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="best", fontsize=8)
    plt.tight_layout()
    improvement_plot_path = save_dir / f"nte_relative_improvement_{run_tag}.png"
    _assert_fresh_output_path(
        improvement_plot_path,
        kind="NTE relative improvement plot",
        allow_overwrite=allow_overwrite,
    )
    fig5.savefig(improvement_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig5)
    print(f"Saved NTE relative improvement plot to: {improvement_plot_path}")
    artifacts["nte_relative_improvement_plot"] = str(improvement_plot_path)

    fig6, ax6 = plt.subplots(figsize=(12, 4.5), sharex=True)
    bucket_specs = [
        ("initial_nte_low_relative_improvement_mean", "initial NTE <= 0.5", "#756bb1"),
        ("initial_nte_mid_relative_improvement_mean", "0.5 < initial NTE <= 2.0", "#238b45"),
        ("initial_nte_high_relative_improvement_mean", "initial NTE > 2.0", "#de2d26"),
    ]
    for metric_key, label, color in bucket_specs:
        values = [metric.get(metric_key) for metric in all_metrics]
        ax6.plot(
            iters,
            values,
            "o-",
            markersize=3,
            linewidth=1.0,
            alpha=0.35,
            color=color,
            label=f"{label} raw",
        )
        ax6.plot(
            iters,
            _smooth_series(values),
            linewidth=2.0,
            color=color,
            label=f"{label} smooth",
        )
    ax6.axhline(0.0, color="#636363", linewidth=1.0, linestyle="--", alpha=0.6)
    ax6.set_title("Terminal Relative Improvement by Initial NTE Bucket")
    ax6.set_xlabel("Iteration")
    ax6.set_ylabel("Relative improvement")
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc="best", fontsize=8)
    plt.tight_layout()
    bucket_plot_path = save_dir / f"nte_bucket_relative_improvement_{run_tag}.png"
    _assert_fresh_output_path(
        bucket_plot_path,
        kind="NTE bucket relative improvement plot",
        allow_overwrite=allow_overwrite,
    )
    fig6.savefig(bucket_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig6)
    print(f"Saved NTE bucket relative improvement plot to: {bucket_plot_path}")
    artifacts["nte_bucket_relative_improvement_plot"] = str(bucket_plot_path)

    metrics_artifacts = save_training_metrics(
        all_metrics,
        output_dir=output_root / "out" / "training_metrics",
        run_name=f"multiturn_{run_tag}",
        metadata={
            **metadata,
            "raw_data_len": raw_data_len,
            "run_tag": run_tag,
        },
        allow_overwrite=allow_overwrite,
    )
    print(f"Saved metrics bundle to: {metrics_artifacts['run_dir']}")
    artifacts["metrics"] = metrics_artifacts
    return artifacts
