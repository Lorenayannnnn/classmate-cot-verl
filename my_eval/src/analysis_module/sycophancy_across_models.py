import os
import re
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mpl_plt
from tqdm import tqdm
from openai import OpenAI

from my_eval.src.analysis_module.utils import dataset_to_max_sample_num
from my_eval.src.data_module.dataset_configs import DATASET_NAME_TO_CLASS, DATASET_NAME_TO_SUBSET_NAME_LIST


def visualize_avg_std_across_steps(avg_vals_dict, std_vals_dict, x_labels, x_label_name, y_label, title, output_fn, max_y=None, min_y=None):
    """Visualize multiple models on the same plot.
    
    Args:
        avg_vals_dict: Dict mapping model names to list of average values
        std_vals_dict: Dict mapping model names to list of std values
        x_labels: List of x-axis labels
        x_label_name: Name for x-axis
        y_label: Name for y-axis
        title: Plot title
        output_fn: Output filename
        max_y: Maximum y value
        min_y: Minimum y value
    """
    # Create dataframe with all models
    data_rows = []
    for model_name, avg_vals in avg_vals_dict.items():

        # TODO haha temporarily change legend name
        if "baseline" in model_name:
            model_name = "baseline"
        elif "always_m_cl" in model_name:
            model_name = "main + (always) classmate reward"

        for i, (x_label, avg_val) in enumerate(zip(x_labels, avg_vals)):
            data_rows.append({
                x_label_name: x_label,
                y_label: avg_val,
                'Model': model_name
            })
    
    dataframe = pd.DataFrame(data_rows)
    sns.set_theme(style="whitegrid", rc={"figure.figsize": (max(10, len(x_labels)), 6)})

    plt = sns.lineplot(data=dataframe, x=x_label_name, y=y_label, hue='Model', marker='o', markersize=8)

    # Add value annotations at each datapoint
    for model_name, avg_vals in avg_vals_dict.items():
        for i, (x_label, avg_val) in enumerate(zip(x_labels, avg_vals)):
            plt.text(i, avg_val, f'{avg_val:.1f}', ha='center', va='bottom', fontsize=8)

    # Get the largest integer that is smaller than all vals in avg_vals
    all_vals = [v for vals in avg_vals_dict.values() for v in vals]
    if min_y is None:
        min_y = int(min(all_vals)) - 3
    if max_y is None:
        max_y = int(max(all_vals)) + 3
    plt.set_ybound(lower=min_y, upper=max_y)
    plt.set_title(title)
    plt.set_xlabel(x_label_name)
    plt.legend(title='Model')
    fig = plt.get_figure()
    output_dir = os.path.dirname(output_fn)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(output_fn, bbox_inches='tight')
    plt.clear()


def plot_monitor_metrics_across_steps(model_name_or_path_list, dataset_list, all_step_idx_list, output_dir_path):
    metrics = ["tp", "tn", "fp", "fn", "precision", "recall", "f1"]

    for dataset_name in dataset_list:
        fig, axes = mpl_plt.subplots(4, 2, figsize=(max(10, len(all_step_idx_list)) * 1.4, 12), sharex=True)
        axes = axes.flatten()

        legend_handles = None
        legend_labels = None

        for metric_idx, metric in enumerate(metrics):
            data_rows = []

            for main_model_name_or_path in model_name_or_path_list:
                # Keep model naming consistent with other plots
                model_name = main_model_name_or_path
                if "baseline" in model_name:
                    model_name = "baseline"
                elif "always_m_cl" in model_name:
                    model_name = "main + (always) classmate reward"

                for step_idx in all_step_idx_list:
                    main_model_dir = f"{output_dir_path}/inference_main_model/{main_model_name_or_path}"
                    result_dir = f"{main_model_dir}/step_{step_idx}/{dataset_name}/main"
                    metrics_fn = os.path.join(result_dir, "monitor_metrics.json")

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
                marker="o",
                markersize=6,
                ax=ax,
            )
            ax.set_title(metric)
            ax.set_xlabel("RL Steps")
            ax.set_ylabel(metric)

            step_to_x = {step: idx for idx, step in enumerate(all_step_idx_list)}
            for _, row in dataframe.iterrows():
                x = step_to_x.get(row["RL Steps"], None)
                if x is None or pd.isna(row["Value"]):
                    continue
                if metric in {"precision", "recall", "f1"}:
                    value_str = f"{row['Value']:.2f}"
                else:
                    value_str = f"{int(round(row['Value']))}"
                ax.text(
                    x,
                    row["Value"],
                    value_str,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

            if metric_idx == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

        if legend_handles and legend_labels:
            fig.legend(legend_handles, legend_labels, loc="lower center", ncol=2, frameon=False)

        fig.suptitle(f"Monitor Metrics across RL Steps on {dataset_name}")
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        combined_output_dir = f"{output_dir_path}/inference_main_model/combined_comparison"
        os.makedirs(combined_output_dir, exist_ok=True)
        output_fn = f"{combined_output_dir}/{dataset_name}_monitor_metrics_comparison.png"
        fig.savefig(output_fn, bbox_inches="tight")
        mpl_plt.close(fig)


if __name__ == "__main__":
    main_model_name_or_path_list = [
        "20260203-Qwen3-0.6B_mmlu_sycophancy_new_baseline_627984_episodes_seed_42",
        "20260203-Qwen3-0.6B_mmlu_sycophan_new_vanilla_always_cl_partial_deepseek_627984_ep_seed_42",
    ]
    max_step_num = 6  # start form 0
    step_size = 48

    output_dir = "outputs_eval"
    dataset_name_list = ["mmlu_sycophancy"]
    max_y, min_y = None, None

    skip_indices = []

    # mmlu sycophancy
    all_step_idx = ["base"] + [str((i+1) * step_size) for i in range(0, (max_step_num)) if str((i+1) * step_size) not in skip_indices]
    plot_monitor_metrics_across_steps(main_model_name_or_path_list, dataset_name_list, all_step_idx, output_dir)

#bash my_eval/scripts/run_sycophancy_across_models.sh