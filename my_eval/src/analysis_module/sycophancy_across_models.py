import os
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mpl_plt
from tqdm import tqdm
from openai import OpenAI

from my_eval.src.data_module.dataset_configs import DATASET_NAME_TO_CLASS, DATASET_NAME_TO_SUBSET_NAME_LIST

def plot_monitor_metrics_across_steps(
    model_name_or_path_list,
    dataset_list,
    all_step_idx_list,
    output_dir_path,
    metrics_filename: str = "monitor_metrics.json",
    monitor_model_name: str | None = None,
    judge_model_name: str | None = None,
):
    def _detect_metric_mode(dataset_name):
        for main_model_name_or_path in model_name_or_path_list:
            for step_idx in all_step_idx_list:
                main_model_dir = f"{output_dir_path}/{main_model_name_or_path}"
                result_dir = f"{main_model_dir}/step_{step_idx}/{dataset_name}/main"
                metrics_fn = os.path.join(result_dir, metrics_filename)
                preds_fn = os.path.join(result_dir, "preds.jsonl")
                with open(preds_fn, "r") as f:
                    preds_data = [json.loads(line) for line in f]
                    assert len(preds_data) == 1000, f"Expected 1000 entries in preds.jsonl for mode detection, got {len(preds_data)}"

                if os.path.exists(metrics_fn):
                    with open(metrics_fn, "r") as f:
                        metric_data = json.load(f)
                    if "mse" in metric_data:
                        return "non_binary"
                    if "tp" in metric_data:
                        return "binary"
        return "binary"

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)

    _COLOR_MAP = {
        "Baseline": "#F28C28",
        "Main + sycophantic Qwen-0.6B classmate": "#D94F3D",
        "Main + unsycophantic Qwen-0.6B classmate": "#3A9E5F",
        "Main + unsycophantic Qwen-0.6B classmate (no cl IS)": "#3A78B8",
    }

    # Build display name map once
    model_name_map = {}
    for p in model_name_or_path_list:
        # TODO haha change legend name temporarily
        if "baseline" in p:
            model_name_map[p] = "Baseline"
        elif "OURS_cl_SELF" in p:
            model_name_map[p] = "Main + sycophantic Qwen-0.6B classmate"
        elif "no_cl_IS" in p:
            model_name_map[p] = "Main + unsycophantic Qwen-0.6B classmate (no cl IS)"
        elif "OURS_gdpo" in p:
            model_name_map[p] = "Main + unsycophantic Qwen-0.6B classmate"
        else:
            model_name_map[p] = p
    _LEGEND_ORDER = {
        "Baseline": 0,
        "Main + sycophantic Qwen-0.6B classmate": 1,
        "Main + unsycophantic Qwen-0.6B classmate": 2,
        "Main + unsycophantic Qwen-0.6B classmate (no cl IS)": 3,
    }
    hue_order = sorted(
        [model_name_map[p] for p in model_name_or_path_list],
        key=lambda name: _LEGEND_ORDER.get(name, 99),
    )
    model_palette = {name: _COLOR_MAP.get(name, "#888888") for name in hue_order}

    # float_metrics = {"precision", "recall", "f1", "mse", "prediction_mean", "ground_truth_mean", "mae"}
    float_metrics = {"precision", "recall", "f1", "mse", "prediction_mean", "ground_truth_mean"}


    for dataset_name in dataset_list:
        metric_mode = _detect_metric_mode(dataset_name)
        count_metrics = ["total_monitored_entries", "total_output_valid_entries", "total_monitor_valid_entries"]
        if metric_mode == "non_binary":
            # metrics = ["mse", "prediction_mean", "ground_truth_mean", "mae"] + count_metrics
            metrics = ["mse", "prediction_mean", "ground_truth_mean"] + count_metrics
        else:
            metrics = ["tp", "tn", "fp", "fn", "precision", "recall", "f1"] + count_metrics

        n_cols = 3
        n_rows = int(np.ceil(len(metrics) / n_cols))
        fig_height = max(10, n_rows * 3.5)
        fig, axes = mpl_plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(18, len(all_step_idx_list) * 2.7), fig_height),
        )
        axes = axes.flatten()

        legend_handles = None
        legend_labels = None

        for metric_idx, metric in enumerate(metrics):
            data_rows = []

            for main_model_name_or_path in model_name_or_path_list:
                model_name = model_name_map[main_model_name_or_path]

                for step_idx in all_step_idx_list:
                    main_model_dir = f"{output_dir_path}/{main_model_name_or_path}"
                    result_dir = f"{main_model_dir}/step_{step_idx}/{dataset_name}/main"
                    metrics_fn = os.path.join(result_dir, metrics_filename)

                    if os.path.exists(metrics_fn):
                        with open(metrics_fn, "r") as f:
                            metric_data = json.load(f)
                        metric_value = metric_data.get(metric, np.nan)
                    else:
                        metric_value = np.nan

                    data_rows.append({
                        "RL Steps": step_idx,
                        "Metric": metric,
                        "Value": metric_value,
                        "Model": model_name
                    })

            if len(data_rows) == 0:
                continue

            dataframe = pd.DataFrame(data_rows)
            ax = axes[metric_idx]
            sns.lineplot(
                data=dataframe,
                x="RL Steps",
                y="Value",
                hue="Model",
                hue_order=hue_order,
                palette=model_palette,
                marker="o",
                markersize=7,
                linewidth=2.0,
                ax=ax,
            )

            metric_label = metric.replace("_", " ").title()
            ax.set_title(metric_label, fontsize=12, fontweight="bold", pad=6)
            ax.set_xlabel("RL Steps", fontsize=11)
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
            sns.despine(ax=ax)

            step_to_x = {step: idx for idx, step in enumerate(all_step_idx_list)}
            for _, row in dataframe.iterrows():
                x = step_to_x.get(row["RL Steps"], None)
                if x is None or pd.isna(row["Value"]):
                    continue
                value_str = f"{row['Value']:.2f}" if metric in float_metrics else f"{int(round(row['Value']))}"
                ax.text(
                    x,
                    row["Value"],
                    value_str,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#555555",
                )

            if metric_idx == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

        dataset_title = dataset_name.replace("_", " ").title()
        title_lines = [f"Monitor Metrics — {dataset_title} (across RL Steps)"]
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
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(legend_labels), 3),
                frameon=True,
                framealpha=0.9,
                edgecolor="#cccccc",
                fontsize=10,
                title="Model",
                title_fontsize=11,
            )

        metrics_stem = os.path.splitext(metrics_filename)[0]
        combined_output_dir = f"{output_dir_path}/combined_comparison"
        os.makedirs(combined_output_dir, exist_ok=True)
        output_fn = f"{combined_output_dir}/{dataset_name}_{metrics_stem}_comparison.png"
        fig.savefig(output_fn, bbox_inches="tight", dpi=150)
        mpl_plt.close(fig)


def calculate_metric_vals(model_name_or_path_list, dataset_list, all_step_idx_list, output_dir_path):
    metric_vals_dict = {model_name_or_path: [] for model_name_or_path in model_name_or_path_list}

    for dataset_name in dataset_list:
        verifier = DATASET_NAME_TO_CLASS[dataset_name]()
        for main_model_name_or_path in tqdm(model_name_or_path_list):
            for step_idx in all_step_idx_list:
                main_model_dir = f"{output_dir_path}/{main_model_name_or_path}"
                result_dir = f"{main_model_dir}/step_{step_idx}/{dataset_name}/main"
                metrics_fn = os.path.join(result_dir, "monitor_metrics.json")
                preds_fn = os.path.join(result_dir, "preds.jsonl")

                with open(preds_fn, "r") as f:
                    preds_data = [json.loads(line) for line in f]

                monitor_scores = [entry["monitor_score"] for entry in preds_data]
                main_scores = [entry["main_score"] for entry in preds_data]

                monitor_metrics = verifier.compute_metrics(
                    predictions=monitor_scores,
                    ground_truths=main_scores,
                )

                monitor_metrics["total_monitored_entries"] = int(len(main_scores))

                with open(metrics_fn, "w") as f:
                    json.dump(monitor_metrics, f, indent=4)

    return metric_vals_dict


if __name__ == "__main__":
    # main_model_name_or_path_list = [
    #     "20260203-Qwen3-0.6B_mmlu_sycophancy_new_baseline_627984_episodes_seed_42",
    #     "20260203-Qwen3-0.6B_mmlu_sycophan_new_vanilla_always_cl_partial_deepseek_627984_ep_seed_42",
    # ]
    #
    # max_step_num = 7  # start form 0
    # step_size = 48

    # main_model_name_or_path_list = [
    #     "20260211-Qwen3-0.6B_helpful_instructions_baseline_448000_episodes_seed_42",
    #     "20260211-Qwen3-0.6B_helpful_instructions_cl_correction_448000_episodes_seed_42",
    # ]

    main_model_name_or_path_list = [
        "20260217-Qwen3-0.6B_grpo_sycophancy_warmup_baseline_192000_episodes_seed_42",
        "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_gdpo_192000_episodes_seed_42",
        "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_cl_SELF_gdpo_192000_episodes_seed_42",
        "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_gdpo_192000_episodes_seed_42_no_cl_IS"
    ]

    max_step_num = 7  # start form 0
    step_size = 20

    output_dir = "outputs_eval/inference_main_model"
    # dataset_name_list = ["mmlu_sycophancy"]
    dataset_name_list = ["helpful_instructions"]
    # dataset_name_list = ["anthropic_hh_rlhf"]
    max_y, min_y = None, None

    skip_indices = []

    # mmlu sycophancy
    all_step_idx = ["base"] + [str((i+1) * step_size) for i in range(0, max_step_num) if str((i + 1) * step_size) not in skip_indices]
    # all_step_idx = ["base"] + [str(10 + i * step_size) for i in range(0, (max_step_num+1)) if str(i * step_size) not in skip_indices]
    # calculate_metric_vals(main_model_name_or_path_list, dataset_name_list, all_step_idx, output_dir)

    print("Start!")
    plot_monitor_metrics_across_steps(
        main_model_name_or_path_list, dataset_name_list, all_step_idx, output_dir,
        # metrics_filename="monitor_metrics.json",
        # monitor_model_name="Qwen3-4B-Instruct-2507",
        # judge_model_name="Qwen3-4B-Instruct-2507",
        metrics_filename="gpt-4o-mini_monitor-gpt-4.1-mini_llm_judge_metrics.json",
        monitor_model_name="gpt-4o-mini",
        judge_model_name="gpt-4.1-mini",
    )

#bash my_eval/scripts/run_sycophancy_across_models.sh