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
    "baseline_all_tokens": "Baseline (All Tokens)",
    "baseline_cot_only":   "Baseline (CoT only)",
    "baseline_all_tokens_w_kl": "Baseline (All Tokens) w/ KL",
    "baseline_output_only":"Baseline (output only)",
    "OURS_self":           "OURS (self as classmate)",
    "OURS_llama":          "OURS (LLaMA as classmate)",
}

_COLOR_MAP = {
    "Baseline (All Tokens)":         "#5DBFA4",
    "Baseline (All Tokens) w/ KL":   "#185FA5",
    "Baseline (CoT only)":           "#38B0D8",
    "OURS (self as classmate)":      "#D85A30",
    "OURS (LLaMA as classmate)":     "#EF9F27",
}



def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def _build_metrics_filename(monitor_model_name: str, judge_model_name: str, shared: bool = False) -> str:
    base = (
        f"{_sanitize_model_name(monitor_model_name)}_monitor"
        f"-{_sanitize_model_name(judge_model_name)}_llm_judge_metrics.json"
    )
    return f"intersection_{base}" if shared else base


def _metrics_path(result_dir, method, step_idx, seed, metrics_filename):
    """Return the path to a metrics file.

    Layout (matches inference output):
      regular step: {result_dir}/{method}/step_{step_idx}/{seed}/main/{metrics_filename}
      base step:    {result_dir}/{method}/step_base/{seed}/main/{metrics_filename}
                    or (seed-independent tasks):
                    {result_dir}/{method}/step_base/main/{metrics_filename}
    """
    if step_idx == "base":
        with_seed = os.path.join(result_dir, method, "step_base", seed, "main", metrics_filename)
        without_seed = os.path.join(result_dir, method, "step_base", "main", metrics_filename)
        return with_seed if os.path.exists(with_seed) else without_seed
    return os.path.join(result_dir, method, f"step_{step_idx}", seed, "main", metrics_filename)


def plot_monitor_metrics_across_steps(
    methods_list,
    seeds_list,
    all_step_idx_list,
    result_dir,           # outputs_eval/{task}/{base_model}
    dataset_name,
    behavior_key=None,    # key to read from the nested metrics dict; defaults to dataset_name
    monitor_model_name=None,
    judge_model_name=None,
    figure_suffix=None,   # optional suffix appended to the output filename
):
    """
    For each method in methods_list, load metrics across steps and seeds,
    then plot mean ± std across seeds as a line with shaded error band.

    Directory layout expected:
        regular: {result_dir}/{method}/step_{step_idx}/{seed}/main/{metrics_filename}
        base:    {result_dir}/{method}/step_base/{seed}/main/{metrics_filename}
                 or (seed-independent):
                 {result_dir}/{method}/step_base/main/{metrics_filename}

    Metrics file format (new nested format):
        { behavior_key: { "rmse": ..., "prediction_mean": ..., ... }, ... }

    behavior_key selects which top-level key to read. Defaults to dataset_name.
    For general_reward's actual scoring key use behavior_key="general_reward".
    Perf metrics (RMSE, prediction_mean, ground_truth_mean, mean_score) always read from
    intersection_ files (shared valid entries across all methods/steps/seeds).
    Count metrics (total_monitored_entries, total_valid_*) read from raw (non-intersection) files.
    """
    assert monitor_model_name and judge_model_name, (
        "monitor_model_name and judge_model_name are required"
    )
    # Perf metrics (RMSE, prediction_mean, etc.) always come from the intersection file so that
    # all methods are evaluated on the exact same entries.  Count metrics (total_monitored_entries,
    # total_valid_*) always come from the raw (non-intersection) file to show actual per-method counts.
    intersection_metrics_filename = _build_metrics_filename(monitor_model_name, judge_model_name, shared=True)
    raw_metrics_filename          = _build_metrics_filename(monitor_model_name, judge_model_name, shared=False)
    metrics_filename = intersection_metrics_filename  # default used by _load_behavior_metrics / _mean_shared_entries
    bk = behavior_key or dataset_name

    # For the general_reward dataset the top-level "general_reward" figure
    # (bk == dataset_name) is superseded by the per-behavior figures and the
    # main figure; skip it to avoid generating a redundant/misleading plot.
    if dataset_name == "general_reward" and bk == dataset_name:
        print(f"[skip] general_reward / general_reward top-level figure — use visualize_main_figure.py instead.")
        return

    def _load_behavior_metrics(metrics_fn, bk_override=None):
        """Load the metrics dict for bk (or bk_override) from a file, returning {} if missing."""
        bk_use = bk_override or bk

        def _extract_from_data(data):
            return data.get(bk_use)

        if not os.path.exists(metrics_fn):
            return {}
        with open(metrics_fn) as f:
            data = json.load(f)
        result = _extract_from_data(data)
        return result if result is not None else {}

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
                        return "non_binary" if "rmse" in bk_data else "binary"
        return "binary"

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

    display_names = [_METHOD_DISPLAY.get(m, m) for m in methods_list]
    hue_order = display_names
    model_palette = {name: _COLOR_MAP.get(name, "#888888") for name in hue_order}

    float_metrics = {"precision", "recall", "f1", "rmse", "prediction_mean", "ground_truth_mean", "mean_score"}

    metric_mode = _detect_metric_mode()
    if metric_mode == "general_reward_score":
        perf_metrics  = ["mean_score"]
        count_metrics = ["total_valid_entries"]
    elif metric_mode == "non_binary":
        perf_metrics  = ["rmse", "prediction_mean", "ground_truth_mean"]
        count_metrics = ["total_monitored_entries", "total_valid_output_entries", "total_valid_CoT_entries"]
    else:
        perf_metrics  = ["tp", "tn", "fp", "fn", "precision", "recall", "f1"]
        count_metrics = ["total_monitored_entries", "total_valid_output_entries", "total_valid_CoT_entries"]

    def _to_num(s):
        return 0 if s == "base" else int(s)

    def _build_metric_rows(metric, bk_load=None, alt_metrics_filename=None):
        fn = alt_metrics_filename or metrics_filename
        rows = []
        for method, display_name in zip(methods_list, display_names):
            for step_idx in all_step_idx_list:
                for seed in seeds_list:
                    metrics_fn = _metrics_path(result_dir, method, step_idx, seed, fn)
                    bk_data = _load_behavior_metrics(metrics_fn, bk_load)
                    rows.append({
                        "RL Steps": _to_num(step_idx),
                        "Value":    bk_data.get(metric, np.nan),
                        "Method":   display_name,
                        "Seed":     seed,
                    })
        return rows

    def _mean_shared_entries():
        """Return mean total_monitored_entries across all (method, step, seed) from intersection file.

        Since the intersection is the same for all methods at a given (step, seed), this is just
        the average number of shared valid entries per evaluation point.
        """
        vals = []
        for method in methods_list:
            for step_idx in all_step_idx_list:
                for seed in seeds_list:
                    metrics_fn = _metrics_path(result_dir, method, step_idx, seed, metrics_filename)
                    bk_data = _load_behavior_metrics(metrics_fn)
                    v = bk_data.get("total_monitored_entries", None)
                    if v is not None:
                        vals.append(v)
        return int(round(np.mean(vals))) if vals else None

    def _plot_metric_ax(ax, metric, bk_load=None, label_override=None, alt_metrics_filename=None):
        """Plot one metric on ax. Returns (legend_handles, legend_labels) or (None, None)."""
        data_rows = _build_metric_rows(metric, bk_load, alt_metrics_filename=alt_metrics_filename)
        if not data_rows:
            return None, None
        df = pd.DataFrame(data_rows)
        sns.lineplot(
            data=df, x="RL Steps", y="Value",
            hue="Method", hue_order=hue_order, palette=model_palette,
            estimator="mean", errorbar="sd",
            marker="o", markersize=9, linewidth=3.0,
            ax=ax,
        )
        x_ticks = sorted(df["RL Steps"].dropna().unique())
        ax.set_xticks(x_ticks)
        mean_df = df.groupby(["Method", "RL Steps"])["Value"].mean().reset_index()
        for _, row in mean_df.iterrows():
            if pd.isna(row["Value"]):
                continue
            value_str = f"{row['Value']:.2f}" if metric in float_metrics else f"{int(round(row['Value']))}"
            ax.text(row["RL Steps"], row["Value"], value_str,
                    ha="center", va="bottom", fontsize=9, color="#555555")
        metric_label = label_override or metric.replace("_", " ").title()
        ax.set_title(metric_label, fontsize=14, fontweight="bold", pad=6)
        ax.set_xlabel("RL Steps", fontsize=13)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
        sns.despine(ax=ax)
        return ax.get_legend_handles_labels()

    # Build plot_specs: list of (metric, bk_load, label_override, alt_metrics_filename)
    # Perf metrics use the intersection file (metrics_filename); count metrics use the raw file.
    perf_specs  = [(m, None, None, None)                      for m in perf_metrics]
    count_specs = [(m, None, None, raw_metrics_filename)      for m in count_metrics]
    if dataset_name == "general_reward" and metric_mode != "general_reward_score":
        # Insert reward score subplot (from intersection file) between perf and count metrics
        perf_specs.append(("mean_score", "general_reward", "Reward Score", None))
    plot_specs = perf_specs + count_specs

    fig_width = max(18, len(all_step_idx_list) * 2.7)
    n_cols = 4 if dataset_name == "general_reward" else 3
    n_rows = int(np.ceil(len(plot_specs) / n_cols))  # type: ignore[arg-type]
    fig_height = max(10, n_rows * 3.5)
    fig, axes = mpl_plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
    )
    axes = axes.flatten()

    legend_handles = legend_labels = None

    for spec_idx, (metric, bk_load, label, alt_fn) in enumerate(plot_specs):
        ax = axes[spec_idx]
        h, l = _plot_metric_ax(ax, metric, bk_load=bk_load, label_override=label, alt_metrics_filename=alt_fn)
        if spec_idx == 0 and h:
            legend_handles, legend_labels = h, l
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # For general_reward the grid is 2×4 with 7 specs — axes[7] is the empty cell
    # below "Reward Score".  Reserve it for the legend; delete all other surplus cells.
    legend_cell_idx = len(plot_specs) if dataset_name == "general_reward" else None
    for idx in range(len(plot_specs), len(axes)):
        if idx == legend_cell_idx:
            continue
        fig.delaxes(axes[idx])

    behavior_title = bk.replace("_", " ").title()
    dataset_title = dataset_name.replace("_", " ").title()
    base_model = os.path.basename(result_dir)
    if bk == dataset_name:
        topic_title = dataset_title
        topic_fn = dataset_name
    else:
        topic_title = f"{dataset_title} / {behavior_title}"
        topic_fn = f"{dataset_name}_{bk}"

    # Perf metrics always use shared intersection; annotate the shared entry count in the title.
    # title_lines = [f"{topic_title} / {base_model} (mean ± std across seeds) [perf: shared intersection]"]
    title_lines = [f"{topic_title} / {base_model}"]
    subtitle_parts = []
    if monitor_model_name:
        subtitle_parts.append(f"Monitor: {monitor_model_name}")
    if bk == "longer_response":
        subtitle_parts.append("Ground Truth: Token Count (no judge)")
    elif judge_model_name:
        subtitle_parts.append(f"Judge: {judge_model_name}")
    mean_n = _mean_shared_entries()
    if mean_n is not None:
        subtitle_parts.append(f"Shared valid entries: ~{mean_n}/500 (avg across steps & seeds)")
    if subtitle_parts:
        title_lines.append("  |  ".join(subtitle_parts))

    fig.suptitle("\n".join(title_lines), fontsize=14, fontweight="bold")
    fig.tight_layout()

    if legend_handles and legend_labels:
        if legend_cell_idx is not None:
            # Place legend in the empty cell below "Reward Score" (general_reward layout)
            legend_ax = axes[legend_cell_idx]
            legend_ax.axis("off")
            legend_ax.legend(
                legend_handles, legend_labels,
                loc="center", ncol=1,
                frameon=True, framealpha=0.9,
                edgecolor="#cccccc",
                fontsize=13, title="Method", title_fontsize=14,
            )
        else:
            fig.legend(
                legend_handles, legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=len(legend_labels),
                frameon=True, framealpha=0.9,
                edgecolor="#cccccc",
                fontsize=13, title="Method", title_fontsize=14,
            )

    metrics_stem = os.path.splitext(metrics_filename)[0]
    monitor_slug = (monitor_model_name or "unknown").replace("/", "_")
    judge_slug = (judge_model_name or "unknown").replace("/", "_")
    monitor_judge_dir = f"{monitor_slug}_monitor-{judge_slug}_llm_judge"
    task_group = "general_reward" if dataset_name == "general_reward" else "specific_behaviors"
    vis_dir = os.path.join(
        os.path.dirname(os.path.dirname(result_dir)), "visualization", "full_figures", task_group, monitor_judge_dir, base_model
    )
    os.makedirs(vis_dir, exist_ok=True)
    suffix_str = f"_{figure_suffix}" if figure_suffix else ""
    output_fn = os.path.join(vis_dir, f"{base_model}_{topic_fn}_{metrics_stem}_comparison{suffix_str}.pdf")
    fig.savefig(output_fn, bbox_inches="tight")
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
    arg_parser.add_argument("--behavior_key", type=str, default=None,
                            help="Top-level key in metrics JSON to plot (defaults to dataset_name). "
                                 "Use 'general_reward' to plot the general reward score sub-dict.")
    arg_parser.add_argument("--monitor_model_name", type=str, required=True)
    arg_parser.add_argument("--judge_model_name",   type=str, required=True)
    arg_parser.add_argument("--figure_suffix", type=str, default=None,
                            help="Optional suffix appended to the output PNG filename to distinguish figures.")

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
        behavior_key=args.behavior_key,
        monitor_model_name=args.monitor_model_name,
        judge_model_name=args.judge_model_name,
        figure_suffix=args.figure_suffix,
    )

#bash my_eval/scripts/visualize_results_across_models.sh
