import os

import pandas as pd
import seaborn as sns


dataset_to_max_sample_num = {
    # "gsm8k": 200,
    "gsm8k": 1319,
    "hendrycks_math": 100,
    "aimo-validation-aime": 90,
}


def visualize_avg_std_across_steps(avg_vals, std_vals, x_labels, x_label_name, y_label, title, output_fn, max_y=None, min_y=None):
    dataframe = pd.DataFrame({
        x_label_name: x_labels,
        y_label: avg_vals,
        # "std": std_vals
    })
    sns.set_theme(style="whitegrid", rc={"figure.figsize": (max(8, len(x_labels)), 6)})

    plt = sns.lineplot(dataframe, x=x_label_name, y=y_label, marker='o', markersize=8)

    # Mark the y value on each point
    for i, v in enumerate(avg_vals):
        plt.text(i, v + 0.5, f"{v:.1f}", ha='center', va='bottom')
    # plt.errorbar(x_labels, avg_vals, yerr=std_vals, fmt='-o', capsize=5)
    # Get the largest integer that is smaller than all vals in avg_vals
    if min_y is None:
        min_y = int(min(avg_vals)) - 3
    if max_y is None:
        max_y = int(max(avg_vals)) + 3
    plt.set_ybound(lower=min_y, upper=max_y)
    plt.set_title(title)
    plt.set_xlabel(x_label_name)
    fig = plt.get_figure()
    output_dir = os.path.dirname(output_fn)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(output_fn)
    plt.clear()
