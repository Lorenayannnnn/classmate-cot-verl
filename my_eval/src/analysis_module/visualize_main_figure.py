"""
visualize_main_figure.py

2×4 main-section figure for the paper, always loaded from intersection_ files.

Layout — 2 rows × 4 columns:

  Group 1 (general_reward task, behavior_keys has 3 entries + include_reward_score):
    Row 0 (RMSE):          col0  col1  col2  col3
                           RMSE  RMSE  RMSE  Reward Score
    Row 1 (Ground Truth):  GT    GT    GT    [empty]
    Behaviors in order: longer_response, confidence, sycophancy

  Group 2 (specific-behavior task, behavior_keys has 4 entries):
    Row 0 (RMSE):          col0  col1  col2  col3
                           RMSE  RMSE  RMSE  RMSE
    Row 1 (Ground Truth):  GT    GT    GT    GT
    Behaviors in order: longer_response, confidence, sycophancy, unsafe_compliance

Each behavior column title shows the behavior display name.

Usage:
  # General reward (3 behaviors + reward score):
  python visualize_main_figure.py \\
    --result_dir outputs_eval/general_reward/Qwen3-0.6B \\
    --dataset_name general_reward \\
    --behavior_keys longer_response,confidence,sycophancy \\
    --include_reward_score \\
    --methods baseline_all_tokens,OURS_self

  # Specific behavior (4 behaviors, one column per trained model):
  python visualize_main_figure.py \\
    --dataset_name specific_behaviors \\
    --behavior_keys longer_response,confidence,sycophancy,unsafe_compliance \\
    --methods baseline_all_tokens,OURS_self \\
    --per_behavior_result_dirs longer_response:outputs_eval/longer_response/Qwen3-0.6B ... \\
    --per_behavior_step_configs longer_response:6:20:60 ...
"""

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
    "baseline_all_tokens":       "Baseline (All Tokens)",
    "baseline_cot_only":         "Baseline (CoT only)",
    "baseline_all_tokens_w_kl":  "Baseline (All Tokens) w/ KL",
    "baseline_output_only":      "Baseline (output only)",
    "OURS_self":                 "OURS (self as classmate)",
    "OURS_llama":                "OURS (LLaMA as classmate)",
}

_COLOR_MAP = {
    "Baseline (All Tokens)":        "#5DBFA4",
    "Baseline (All Tokens) w/ KL":  "#185FA5",
    "Baseline (CoT only)":          "#38B0D8",
    "OURS (self as classmate)":     "#D85A30",
    "OURS (LLaMA as classmate)":    "#EF9F27",
}

_BEHAVIOR_DISPLAY = {
    "sycophancy":       "Sycophancy",
    "confidence":       "Confidence",
    "longer_response":  "Longer Response",
    "unsafe_compliance":"Unsafe Compliance",
    "general_reward":   "General Reward",
}


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def _build_metrics_filename(
    monitor_model_name: str,
    judge_model_name: str,
    shared: bool = False,
    use_dynamic_icl: bool = False,
    no_explanation: bool = False,
) -> str:
    suffix = f"{'_use_dynamic_icl' if use_dynamic_icl else ''}{'_no_expl' if no_explanation else ''}"
    base = (
        f"{_sanitize_model_name(monitor_model_name)}_monitor"
        f"-{_sanitize_model_name(judge_model_name)}_llm_judge_metrics{suffix}.json"
    )
    return f"intersection_{base}" if shared else base


def _metrics_path(result_dir, method, step_idx, seed, metrics_filename, dataset_split_name="test"):
    """Return the path to a metrics file.

    Layout:
      regular step: {result_dir}/{method}/step_{step_idx}/{seed}/{split}/main/{metrics_filename}
      base step:    {result_dir}/{method}/step_base/{seed}/{split}/main/{metrics_filename}
                    or (seed-independent):
                    {result_dir}/{method}/step_base/{split}/main/{metrics_filename}
    """
    if step_idx == "base":
        with_seed    = os.path.join(result_dir, method, "step_base", seed, dataset_split_name, "main", metrics_filename)
        without_seed = os.path.join(result_dir, method, "step_base", dataset_split_name, "main", metrics_filename)
        return with_seed if os.path.exists(with_seed) else without_seed
    return os.path.join(result_dir, method, f"step_{step_idx}", seed, dataset_split_name, "main", metrics_filename)


def plot_main_figure(
    methods_list,
    seeds_list,
    all_step_idx_list,     # used when step_config_per_bk is not set
    result_dir,            # outputs_eval/{task}/{base_model}; used for output path and fallback
    dataset_name,
    behavior_keys,         # ordered list of behaviors; 3 for general_reward, 4 for specific
    monitor_model_name,
    judge_model_name,
    include_reward_score=False,   # True for general_reward: add reward score in col 3 row 0
    figure_suffix=None,
    result_dir_per_bk=None,       # dict {bk: result_dir}; overrides result_dir per behavior column
    step_config_per_bk=None,      # dict {bk: (max_step_num, step_size, viz_offset)}
    dataset_split_name="test",
    use_dynamic_icl=False,
    no_explanation=False,
):
    """
    2×4 main-section figure.  Always loads from intersection_ files.

    Row 0: RMSE for each behavior column; col 3 is reward score (general_reward)
           or 4th behavior RMSE (specific behavior).
    Row 1: Ground truth mean for each behavior column; col 3 is empty
           (general_reward) or 4th behavior GT (specific behavior).

    For specific-behavior figures, pass result_dir_per_bk and step_config_per_bk
    so each column reads from the model trained on that specific behavior.
    """
    assert monitor_model_name and judge_model_name, (
        "monitor_model_name and judge_model_name are required"
    )
    metrics_filename = _build_metrics_filename(monitor_model_name, judge_model_name, shared=True,
                                               use_dynamic_icl=use_dynamic_icl, no_explanation=no_explanation)

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #
    def _extract_from_data(data, bk):
        if bk in data:
            return data[bk]
        else:
            raise ValueError(f"bk {bk} not in data {data}")
        # for key, value in data.items():
        #     if bk in key:
        #         return value
        # return None

    def _load_bk(metrics_fn, bk):
        if not os.path.exists(metrics_fn):
            return {}
        with open(metrics_fn) as f:
            data = json.load(f)
        result = _extract_from_data(data, bk)
        return result if result is not None else {}

    def _to_num(s):
        return 0 if s == "base" else int(s)

    def _rdir(bk):
        """Return the result_dir to use for a given behavior key."""
        return (result_dir_per_bk or {}).get(bk, result_dir)

    def _step_list(bk):
        """Return the step index list for a given behavior key."""
        if step_config_per_bk and bk in step_config_per_bk:
            max_step_num, step_size, viz_offset = step_config_per_bk[bk]
            return ["base"] + [
                str(viz_offset + (i + 1) * step_size) for i in range(max_step_num)
            ]
        return all_step_idx_list

    def _build_rows(metric, bk):
        rows = []
        for method, display_name in zip(methods_list, display_names):
            for step_idx in _step_list(bk):
                for seed in seeds_list:
                    fn = _metrics_path(_rdir(bk), method, step_idx, seed, metrics_filename, dataset_split_name)
                    bk_data = _load_bk(fn, bk)
                    rows.append({
                        "RL Steps": _to_num(step_idx),
                        "Value":    bk_data.get(metric, np.nan),
                        "Method":   display_name,
                        "Seed":     seed,
                    })
        return rows

    def _plot_ax(ax, metric, bk):
        """Plot one subplot. Returns (handles, labels) from the first call."""
        rows = _build_rows(metric, bk)
        if not rows:
            return None, None
        df = pd.DataFrame(rows)
        sns.lineplot(
            data=df, x="RL Steps", y="Value",
            hue="Method", hue_order=hue_order, palette=model_palette,
            estimator="mean", errorbar="sd",
            marker="o", markersize=9, linewidth=3.0,
            ax=ax,
        )
        ax.set_xticks(sorted(df["RL Steps"].dropna().unique()))
        mean_df = df.groupby(["Method", "RL Steps"])["Value"].mean().reset_index()
        for _, row in mean_df.iterrows():
            if pd.isna(row["Value"]):
                continue
            ax.text(row["RL Steps"], row["Value"], f"{row['Value']:.2f}",
                    ha="center", va="bottom", fontsize=9, color="#555555")
        ax.set_xlabel("RL Steps", fontsize=13)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
        sns.despine(ax=ax)
        h, l = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        return h, l

    # ---------------------------------------------------------------- #
    # Layout: always 2 rows × 4 cols
    # ---------------------------------------------------------------- #
    display_names = [_METHOD_DISPLAY.get(m, m) for m in methods_list]
    hue_order     = display_names
    model_palette = {name: _COLOR_MAP.get(name, "#888888") for name in hue_order}

    n_cols = 4
    n_rows = 2
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    # Use the longest per-behavior step list for figure width sizing
    _max_steps = max(len(_step_list(bk)) for bk in behavior_keys)
    fig_width  = max(5 * n_cols, _max_steps * 2.5)
    fig_height = 7.5
    fig, axes  = mpl_plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    def _col_title(bk):
        name = _BEHAVIOR_DISPLAY.get(bk, bk.replace("_", " ").title())
        if bk == "longer_response":
            name += " (Token Count)"
        return name

    legend_handles = legend_labels = None

    # ── Row 0: RMSE (behaviors) + reward score or 4th behavior ─── #
    for col, bk in enumerate(behavior_keys):
        ax = axes[0, col]
        ax.set_title(_col_title(bk), fontsize=13, fontweight="bold", pad=6)
        h, l = _plot_ax(ax, "rmse", bk)
        if legend_handles is None and h:
            legend_handles, legend_labels = h, l

    if include_reward_score:
        ax = axes[0, 3]
        ax.set_title("Reward Score", fontsize=13, fontweight="bold", pad=6)
        h, l = _plot_ax(ax, "mean_score", "general_reward")
        if legend_handles is None and h:
            legend_handles, legend_labels = h, l

    # ── Row 1: Ground truth mean (behaviors) + empty for general_reward ── #
    for col, bk in enumerate(behavior_keys):
        ax = axes[1, col]
        ax.set_title(_col_title(bk), fontsize=13, fontweight="bold", pad=6)
        _plot_ax(ax, "ground_truth_mean", bk)

    # Row labels on the leftmost column
    axes[0, 0].set_ylabel("RMSE", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Ground Truth Mean", fontsize=13, fontweight="bold")

    # ---------------------------------------------------------------- #
    # Figure title
    # ---------------------------------------------------------------- #
    # For output paths, use result_dir if set; otherwise fall back to first per-bk dir.
    _any_result_dir = result_dir or next(iter((result_dir_per_bk or {}).values()))
    base_model   = os.path.basename(_any_result_dir)
    dataset_disp = dataset_name.replace("_", " ").title()
    # title_lines  = [f"{dataset_disp} / {base_model} (mean ± std across seeds) [shared intersection]"]
    title_lines  = [f"{dataset_disp} / {base_model}"]
    title_lines.append(f"Monitor: {monitor_model_name}  |  Judge: {judge_model_name}")
    fig.suptitle("\n".join(title_lines), fontsize=13, fontweight="bold")
    fig.tight_layout()

    if legend_handles and legend_labels:
        if include_reward_score:
            # Place legend in the empty [row 1, col 3] cell (below Reward Score)
            legend_ax = axes[1, 3]
            legend_ax.axis("off")
            legend_ax.legend(
                legend_handles, legend_labels,
                loc="center",
                ncol=1,
                frameon=True, framealpha=0.9,
                edgecolor="#cccccc",
                fontsize=13, title="Method", title_fontsize=14,
            )
        else:
            fig.legend(
                legend_handles, legend_labels,
                loc="upper right",
                bbox_to_anchor=(1.0, 1.0),
                ncol=1,
                frameon=True, framealpha=0.9,
                edgecolor="#cccccc",
                fontsize=13, title="Method", title_fontsize=14,
            )

    # ---------------------------------------------------------------- #
    # Save
    # ---------------------------------------------------------------- #
    metrics_stem      = os.path.splitext(metrics_filename)[0]
    _key_suffix       = f"{'_use_dynamic_icl' if use_dynamic_icl else ''}{'_no_expl' if no_explanation else ''}"
    monitor_slug      = _sanitize_model_name(monitor_model_name)
    judge_slug        = _sanitize_model_name(judge_model_name)
    monitor_judge_dir = f"{monitor_slug}_monitor-{judge_slug}_judge{_key_suffix}"
    vis_dir = os.path.join(
        os.path.dirname(os.path.dirname(_any_result_dir)),
        "visualization", "main_figures", dataset_name, monitor_judge_dir, base_model,
    )
    os.makedirs(vis_dir, exist_ok=True)
    suffix_str = f"_{figure_suffix}" if figure_suffix else ""
    output_fn  = os.path.join(
        vis_dir,
        f"{base_model}_{dataset_name}_{metrics_stem}_main{suffix_str}.pdf",
    )
    fig.savefig(output_fn, bbox_inches="tight")
    mpl_plt.close(fig)
    print(f"[saved] {output_fn}")


_BEHAVIOR_COLOR = {
    "sycophancy":        "#5DBFA4",
    "confidence":        "#D85A30",
    "longer_response":   "#185FA5",
    "unsafe_compliance": "#EF9F27",
}


def plot_shared_valid_entry_counts(
    methods_list,
    seeds_list,
    all_step_idx_list,     # used when step_config_per_bk is not set
    result_dir,
    dataset_name,
    behavior_keys,
    monitor_model_name,
    judge_model_name,
    result_dir_per_bk=None,
    step_config_per_bk=None,
    dataset_split_name="test",
    use_dynamic_icl=False,
    no_explanation=False,
):
    """
    Single plot: one line per behavior showing total_monitored_entries
    (shared across all methods & seeds at each step) vs. training steps.
    Reads from intersection_ files, using the first method/seed (all are identical
    for total_monitored_entries by definition of the intersection).
    """
    assert monitor_model_name and judge_model_name
    metrics_filename = _build_metrics_filename(monitor_model_name, judge_model_name, shared=True,
                                               use_dynamic_icl=use_dynamic_icl, no_explanation=no_explanation)

    def _rdir(bk):
        return (result_dir_per_bk or {}).get(bk, result_dir)

    def _step_list(bk):
        if step_config_per_bk and bk in step_config_per_bk:
            max_step_num, step_size, viz_offset = step_config_per_bk[bk]
            return ["base"] + [
                str(viz_offset + (i + 1) * step_size) for i in range(max_step_num)
            ]
        return all_step_idx_list

    def _to_num(s):
        return 0 if s == "base" else int(s)

    method0 = methods_list[0]
    seed0   = seeds_list[0]

    n_bk = len(behavior_keys)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    fig, axes = mpl_plt.subplots(1, n_bk, figsize=(5 * n_bk, 4))
    if n_bk == 1:
        axes = [axes]

    for ax, bk in zip(axes, behavior_keys):
        steps, counts = [], []
        for step_idx in _step_list(bk):
            fn = _metrics_path(_rdir(bk), method0, step_idx, seed0, metrics_filename, dataset_split_name)
            if not os.path.exists(fn):
                continue
            with open(fn) as f:
                data = json.load(f)
            bk_data = data.get(bk, {})
            n = bk_data.get("total_monitored_entries")
            if n is not None:
                steps.append(_to_num(step_idx))
                counts.append(n)
        color = _BEHAVIOR_COLOR.get(bk, "#888888")
        label = _BEHAVIOR_DISPLAY.get(bk, bk.replace("_", " ").title())
        ax.plot(steps, counts, marker="o", linewidth=2.5, markersize=7, color=color, label=label)
        for x, y in zip(steps, counts):
            ax.text(x, y, str(y), ha="center", va="bottom", fontsize=9, color="#555555")
        ax.set_xlabel("RL Steps", fontsize=13)
        ax.set_ylabel("Shared Valid Entries", fontsize=13)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks(steps)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
        sns.despine(ax=ax)

    fig.suptitle(
        f"Shared Valid Entry Count per Behavior ({dataset_name.replace('_', ' ').title()})\n"
        f"Monitor: {monitor_model_name}  |  Judge: {judge_model_name}",
        fontsize=12,
    )
    fig.tight_layout()

    _any_result_dir   = result_dir or next(iter((result_dir_per_bk or {}).values()))
    base_model        = os.path.basename(_any_result_dir)
    _key_suffix       = f"{'_use_dynamic_icl' if use_dynamic_icl else ''}{'_no_expl' if no_explanation else ''}"
    monitor_slug      = _sanitize_model_name(monitor_model_name)
    judge_slug        = _sanitize_model_name(judge_model_name)
    monitor_judge_dir = f"{monitor_slug}_monitor-{judge_slug}_judge{_key_suffix}"
    vis_dir = os.path.join(
        os.path.dirname(os.path.dirname(_any_result_dir)),
        "visualization", "main_figures", dataset_name, monitor_judge_dir, base_model,
    )
    os.makedirs(vis_dir, exist_ok=True)
    output_fn = os.path.join(vis_dir, "shared_valid_entry_num.pdf")
    fig.savefig(output_fn, bbox_inches="tight")
    mpl_plt.close(fig)
    print(f"[saved] {output_fn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate compact main-section figure (RMSE per behavior).")
    ap.add_argument("--max_step_num",   type=int, default=None,
                    help="Number of eval checkpoints. Required unless --per_behavior_step_configs is set.")
    ap.add_argument("--step_size",      type=int, default=None)
    ap.add_argument("--viz_offset",     type=int, default=0)
    ap.add_argument("--result_dir",     type=str, default=None,
                    help="Base dir at the {task}/{base_model} level. "
                         "Required unless --per_behavior_result_dirs is set.")
    ap.add_argument("--dataset_name",   type=str, required=True,
                    help="Dataset name, e.g. general_reward, specific_behaviors")
    ap.add_argument("--behavior_keys",  type=str, required=True,
                    help="Comma-separated behavior keys, e.g. longer_response,confidence,sycophancy")
    ap.add_argument("--include_reward_score", action="store_true",
                    help="Append a reward-score subplot (from the general_reward sub-dict).")
    ap.add_argument("--methods",        type=str, default="baseline_all_tokens,OURS_self")
    ap.add_argument("--seeds",          type=str, default="seed_0,seed_1,seed_2")
    ap.add_argument("--monitor_model_name", type=str, required=True)
    ap.add_argument("--judge_model_name",   type=str, required=True)
    ap.add_argument("--figure_suffix",  type=str, default=None)
    ap.add_argument("--per_behavior_result_dirs", type=str, nargs="+", default=None,
                    help="Per-behavior result dirs as 'bk:path' pairs, "
                         "e.g. confidence:outputs_eval/confidence/Qwen3-0.6B. "
                         "Use for specific-behavior figures where each column is a different model.")
    ap.add_argument("--per_behavior_step_configs", type=str, nargs="+", default=None,
                    help="Per-behavior step configs as 'bk:max_step_num:step_size:viz_offset', "
                         "e.g. confidence:6:20:60. Required when --per_behavior_result_dirs is set.")
    ap.add_argument("--dataset_split_name", type=str, default="test",
                    help="Dataset split to read metrics from (e.g. test, dev).")
    ap.add_argument("--use_dynamic_icl", action="store_true",
                    help="Read metrics files generated with dynamic ICL examples (adds _use_dynamic_icl suffix).")
    ap.add_argument("--no_explanation",  action="store_true",
                    help="Read metrics files generated with no-explanation monitor (adds _no_expl suffix).")

    args = ap.parse_args()

    # Build global step list (used when per_behavior_step_configs is not set)
    if args.max_step_num is not None and args.step_size is not None:
        all_step_idx = ["base"] + [
            str(args.viz_offset + (i + 1) * args.step_size)
            for i in range(args.max_step_num)
        ]
    else:
        all_step_idx = None

    # Parse per-behavior overrides
    result_dir_per_bk = None
    if args.per_behavior_result_dirs:
        result_dir_per_bk = dict(s.split(":", 1) for s in args.per_behavior_result_dirs)

    step_config_per_bk = None
    if args.per_behavior_step_configs:
        step_config_per_bk = {}
        for s in args.per_behavior_step_configs:
            bk, max_s, step_s, offset_s = s.split(":")
            step_config_per_bk[bk] = (int(max_s), int(step_s), int(offset_s))

    bk_list = args.behavior_keys.split(",")

    plot_main_figure(
        methods_list         = args.methods.split(","),
        seeds_list           = args.seeds.split(","),
        all_step_idx_list    = all_step_idx,
        result_dir           = args.result_dir,
        dataset_name         = args.dataset_name,
        behavior_keys        = bk_list,
        monitor_model_name   = args.monitor_model_name,
        judge_model_name     = args.judge_model_name,
        include_reward_score = args.include_reward_score,
        figure_suffix        = args.figure_suffix,
        result_dir_per_bk    = result_dir_per_bk,
        step_config_per_bk   = step_config_per_bk,
        dataset_split_name   = args.dataset_split_name,
        use_dynamic_icl      = args.use_dynamic_icl,
        no_explanation       = args.no_explanation,
    )

    plot_shared_valid_entry_counts(
        methods_list       = args.methods.split(","),
        seeds_list         = args.seeds.split(","),
        all_step_idx_list  = all_step_idx,
        result_dir         = args.result_dir,
        dataset_name       = args.dataset_name,
        behavior_keys      = bk_list,
        monitor_model_name = args.monitor_model_name,
        judge_model_name   = args.judge_model_name,
        result_dir_per_bk  = result_dir_per_bk,
        step_config_per_bk = step_config_per_bk,
        dataset_split_name = args.dataset_split_name,
        use_dynamic_icl    = args.use_dynamic_icl,
        no_explanation     = args.no_explanation,
    )

#bash my_eval/scripts/visualize_main_figure.sh
