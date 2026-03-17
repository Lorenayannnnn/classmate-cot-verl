import os
import json
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mpl_plt


# ------------------------------------------------------------------ #
# Display names and colours per method
# ------------------------------------------------------------------ #
_METHOD_DISPLAY = {
    "baseline_all_tokens": "Baseline",
    "baseline_cot_only":   "Baseline (CoT only)",
    "baseline_output_only":"Baseline (output only)",
    "OURS_self":           "OURS (self)",
    "OURS_llama":          "OURS (llama)",
}

_COLOR_MAP = {
    "Baseline":               "#F28C28",
    "Baseline (CoT only)":    "#F5A623",
    "Baseline (output only)": "#F7C06B",
    "OURS (self)":            "#D94F3D",
    "OURS (llama)":           "#3A9E5F",
}


def _metrics_path(result_dir, method, step_idx, seed, metrics_filename):
    """Return the path to a metrics file.

    Layout (matches inference output):
      regular step: {result_dir}/{method}/step_{step_idx}/{seed}/{metrics_filename}
      base step:    {result_dir}/{method}/step_base/{metrics_filename}  (seed-independent)
    """
    if step_idx == "base":
        return os.path.join(result_dir, method, "step_base", metrics_filename)
    return os.path.join(result_dir, method, f"step_{step_idx}", seed, metrics_filename)


def plot_monitor_metrics_across_steps(
    methods_list,
    seeds_list,
    all_step_idx_list,
    result_dir,         # outputs_eval/{task}/{base_model}
    dataset_name,
    metrics_filename,
    behavior_key=None,  # key to read from the nested metrics dict; defaults to dataset_name
    monitor_model_name=None,
    judge_model_name=None,
):
    """
    For each method in methods_list, load metrics across steps and seeds,
    then plot mean ± std across seeds as a line with shaded error band.

    Directory layout expected:
        regular: {result_dir}/{method}/step_{step_idx}/{seed}/{metrics_filename}
        base:    {result_dir}/{method}/step_base/{metrics_filename}

    Metrics file format (new nested format):
        { behavior_key: { "mse": ..., "prediction_mean": ..., ... }, ... }

    behavior_key selects which top-level key to read. Defaults to dataset_name.
    For general_reward's actual scoring key use behavior_key="general_reward".
    """
    bk = behavior_key or dataset_name

    def _load_behavior_metrics(metrics_fn):
        """Load the metrics dict for bk from a file, returning {} if missing."""
        if not os.path.exists(metrics_fn):
            return {}
        with open(metrics_fn) as f:
            data = json.load(f)
        return data.get(bk, {})

    def _detect_metric_mode():
        """Return 'non_binary', 'general_reward_score', or 'binary'."""
        if bk == "general_reward":
            return "general_reward_score"
        for method in methods_list:
            for step_idx in all_step_idx_list:
                for seed in seeds_list:
                    metrics_fn = _metrics_path(result_dir, method, step_idx, seed, metrics_filename)
                    bk_data = _load_behavior_metrics(metrics_fn)
                    if bk_data:
                        return "non_binary" if "mse" in bk_data else "binary"
        return "binary"

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

    display_names = [_METHOD_DISPLAY.get(m, m) for m in methods_list]
    hue_order = display_names
    model_palette = {name: _COLOR_MAP.get(name, "#888888") for name in hue_order}

    float_metrics = {"precision", "recall", "f1", "mse", "prediction_mean", "ground_truth_mean", "mean_score"}

    metric_mode = _detect_metric_mode()
    count_metrics = ["total_monitored_entries", "total_valid_output_entries", "total_valid_CoT_entries", "total_entries"]
    if metric_mode == "general_reward_score":
        metrics = ["mean_score", "total_valid_entries", "total_entries"]
    elif metric_mode == "non_binary":
        metrics = ["mse", "prediction_mean", "ground_truth_mean"] + count_metrics
    else:
        metrics = ["tp", "tn", "fp", "fn", "precision", "recall", "f1"] + count_metrics

    def _to_num(s):
        return 0 if s == "base" else int(s)

    n_cols = 3
    n_rows = int(np.ceil(len(metrics) / n_cols))  # type: ignore[arg-type]
    fig_height = max(10, n_rows * 3.5)
    fig, axes = mpl_plt.subplots(
        n_rows, n_cols,
        figsize=(max(18, len(all_step_idx_list) * 2.7), fig_height),
    )
    axes = axes.flatten()

    legend_handles = legend_labels = None

    for metric_idx, metric in enumerate(metrics):
        data_rows = []

        for method, display_name in zip(methods_list, display_names):
            for step_idx in all_step_idx_list:
                for seed in seeds_list:
                    metrics_fn = _metrics_path(result_dir, method, step_idx, seed, metrics_filename)
                    bk_data = _load_behavior_metrics(metrics_fn)
                    metric_value = bk_data.get(metric, np.nan)

                    data_rows.append({
                        "RL Steps": _to_num(step_idx),
                        "Value":    metric_value,
                        "Method":   display_name,
                        "Seed":     seed,
                    })

        if not data_rows:
            continue

        dataframe = pd.DataFrame(data_rows)
        ax = axes[metric_idx]

        sns.lineplot(
            data=dataframe,
            x="RL Steps",
            y="Value",
            hue="Method",
            hue_order=hue_order,
            palette=model_palette,
            estimator="mean",
            errorbar="sd",
            marker="o",
            markersize=9,
            linewidth=3.0,
            ax=ax,
        )

        # Annotate mean values at each point
        mean_df = dataframe.groupby(["Method", "RL Steps"])["Value"].mean().reset_index()
        for _, row in mean_df.iterrows():
            if pd.isna(row["Value"]):
                continue
            value_str = f"{row['Value']:.2f}" if metric in float_metrics else f"{int(round(row['Value']))}"
            ax.text(
                row["RL Steps"], row["Value"],
                value_str,
                ha="center", va="bottom",
                fontsize=9, color="#555555",
            )

        metric_label = metric.replace("_", " ").title()
        ax.set_title(metric_label, fontsize=14, fontweight="bold", pad=6)
        ax.set_xlabel("RL Steps", fontsize=13)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
        sns.despine(ax=ax)

        if metric_idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])

    behavior_title = bk.replace("_", " ").title()
    dataset_title = dataset_name.replace("_", " ").title()
    base_model = os.path.basename(result_dir)
    title_lines = [f"Monitor Metrics — {dataset_title} / {behavior_title} / {base_model} (mean ± std across seeds)"]
    if monitor_model_name or judge_model_name:
        subtitle_parts = []
        if monitor_model_name:
            subtitle_parts.append(f"Monitor: {monitor_model_name}")
        if judge_model_name:
            subtitle_parts.append(f"Judge: {judge_model_name}")
        title_lines.append("  |  ".join(subtitle_parts))
    fig.suptitle("\n".join(title_lines), fontsize=14, fontweight="bold")
    fig.tight_layout()

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(legend_labels),
            frameon=True, framealpha=0.9,
            edgecolor="#cccccc",
            fontsize=10, title="Method", title_fontsize=11,
        )

    metrics_stem = os.path.splitext(metrics_filename)[0]
    vis_dir = os.path.join(result_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    output_fn = os.path.join(vis_dir, f"{dataset_name}_{bk}_{metrics_stem}_comparison.png")
    fig.savefig(output_fn, bbox_inches="tight", dpi=150)
    mpl_plt.close(fig)
    print(f"[saved] {output_fn}")



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Visualize monitor/judge metrics across training steps.")
    arg_parser.add_argument("--max_step_num", type=int, required=True)
    arg_parser.add_argument("--step_size", type=int, required=True)
    arg_parser.add_argument("--viz_offset", type=int, default=0)
    arg_parser.add_argument("--result_dir", type=str, required=True,
                            help="Base dir at the {task}/{base_model} level, "
                                 "e.g. outputs_eval/inference_main_model/confidence/Qwen3-0.6B")
    arg_parser.add_argument("--dataset_name", type=str, required=True,
                            help="Dataset name for display, e.g. confidence, sycophancy, longer_response")
    arg_parser.add_argument("--methods", type=str, default="baseline_all_tokens,OURS_self",
                            help="Comma-separated methods to compare")
    arg_parser.add_argument("--seeds", type=str, default="seed_0,seed_1,seed_2",
                            help="Comma-separated seed dirs")
    arg_parser.add_argument("--metrics_filename", type=str, required=True,
                            help="Metrics JSON filename, e.g. gpt-4o_monitor-gpt-4.1_llm_judge_metrics.json")
    arg_parser.add_argument("--behavior_key", type=str, default=None,
                            help="Top-level key in metrics JSON to plot (defaults to dataset_name). "
                                 "Use 'general_reward' to plot the general reward score sub-dict.")
    arg_parser.add_argument("--monitor_model_name", type=str, default=None)
    arg_parser.add_argument("--judge_model_name", type=str, default=None)

    args = arg_parser.parse_args()

    methods_list = args.methods.split(",")
    seeds_list = args.seeds.split(",")
    all_step_idx = ["base"] + [
        str(args.viz_offset + (i + 1) * args.step_size)
        for i in range(args.max_step_num)
    ]

    plot_monitor_metrics_across_steps(
        methods_list=methods_list,
        seeds_list=seeds_list,
        all_step_idx_list=all_step_idx,
        result_dir=args.result_dir,
        dataset_name=args.dataset_name,
        metrics_filename=args.metrics_filename,
        behavior_key=args.behavior_key,
        monitor_model_name=args.monitor_model_name,
        judge_model_name=args.judge_model_name,
    )

#bash my_eval/scripts/visualize_results_across_models.sh
