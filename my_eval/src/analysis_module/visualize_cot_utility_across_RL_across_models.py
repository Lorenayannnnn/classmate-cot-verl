import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from my_eval.src.analysis_module.utils import dataset_to_max_sample_num
from my_eval.src.analysis_module.utils_across_models import visualize_avg_std_across_steps
from my_eval.src.data_module.dataset_configs import DATASET_NAME_TO_SUBSET_NAME_LIST, DATASET_NAME_TO_CLASS


def keep_number_digits_only(s):
    return re.sub(r"\D", "", s)

def visualize_cot_utility_across_RL(main_model_name_or_path_list):
    """Visualize CoT utility averaged across all classmates and datasets for multiple main models."""
    all_models_avg_metrics = {}
    all_models_std_metrics = {}
    
    for main_model_name_or_path in main_model_name_or_path_list:
        print(f"\n{'='*80}\nProcessing model: {main_model_name_or_path}\n{'='*80}")
        
        step_to_utility_avg = []
        step_to_utility_std = []
        for step_idx in tqdm(all_step_idx, desc=f"{main_model_name_or_path}"):
            step_str = "wo_cot" if step_idx == "wo_cot" else f"step_{step_idx}"
            step_to_classmate_avg_acc = []
            for classmate in all_classmate_model_names:
                total_classmate_utility = 0
                for dataset_name in all_dataset_names:
                    subset_name_list = DATASET_NAME_TO_SUBSET_NAME_LIST[dataset_name]
                    total_subset_classmate_utility = 0
                    for subset_name in subset_name_list:
                        if "mmlu" in dataset_name:
                            dataset_object = DATASET_NAME_TO_CLASS[dataset_name](subject=subset_name)
                        else:
                            dataset_object = DATASET_NAME_TO_CLASS[dataset_name]()
                        result_jsonl_fn = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/{step_str}/{classmate}/{dataset_name}{subset_name}/preds.jsonl"
                        total_utility = 0
                        all_correct_lines = []
                        all_incorrect_lines = []

                        with open(result_jsonl_fn, "r") as result_jsonl:
                            # ['question', 'main_CoT', 'main_cot_exclude_last_step', 'main_pred', 'classmate_pred', 'gt']
                            lines = result_jsonl.readlines()
                            # if len(lines) > dataset_to_max_sample_num[dataset_name]:
                            #     print(f"Warning: {result_jsonl_fn} has more samples ({len(lines)}) than expected ({dataset_to_max_sample_num[dataset_name]}). Truncating.")
                            #     lines = lines[:dataset_to_max_sample_num[dataset_name]]
                            assert len(lines) == dataset_to_max_sample_num[dataset_name], f"Expected {dataset_to_max_sample_num[dataset_name]} samples for {dataset_name}, but got {len(lines)}"
                            sample_num = len(lines)
                            for line in lines:
                                data = json.loads(line)
                                if dataset_object.compare_pred_and_gt(data["classmate_continuation"], data["gt"]):
                                    total_utility += 1
                                    all_correct_lines.append(line)
                                else:
                                    all_incorrect_lines.append(line)

                        incorrect_output_jsonl_fn = result_jsonl_fn.replace("preds.jsonl", "incorrect_preds.jsonl")
                        correct_output_jsonl_fn = result_jsonl_fn.replace("preds.jsonl", "correct_preds.jsonl")
                        if os.path.exists(incorrect_output_jsonl_fn):
                            os.remove(incorrect_output_jsonl_fn)
                        if os.path.exists(correct_output_jsonl_fn):
                            os.remove(correct_output_jsonl_fn)
                        with open(incorrect_output_jsonl_fn, "w") as incorrect_output_jsonl:
                            incorrect_output_jsonl.writelines(all_incorrect_lines)
                        with open(correct_output_jsonl_fn, "w") as correct_output_jsonl:
                            correct_output_jsonl.writelines(all_correct_lines)
                        total_subset_classmate_utility += total_utility / sample_num * 100
                    total_classmate_utility += total_subset_classmate_utility / len(subset_name_list)
                step_to_classmate_avg_acc.append(total_classmate_utility / len(all_dataset_names))
            step_to_utility_avg.append(np.mean(step_to_classmate_avg_acc))
            step_to_utility_std.append(np.std(step_to_classmate_avg_acc))
        
        # Store results for this model
        model_short_name = main_model_name_or_path.replace('grpo_olmo_1B_hendrycks_math_', '').replace('_3298240_episodes', '').replace('20251122-', '')
        all_models_avg_metrics[model_short_name] = step_to_utility_avg
        all_models_std_metrics[model_short_name] = step_to_utility_std

    # Visualize all models together
    combined_output_dir = "outputs/cot_utility_to_classmate/combined_comparison"
    os.makedirs(combined_output_dir, exist_ok=True)
    
    visualize_avg_std_across_steps(
        avg_vals_dict=all_models_avg_metrics,
        std_vals_dict=all_models_std_metrics,
        x_labels=all_step_idx,
        x_label_name="RL Steps",
        y_label="CoT Utility to Classmate (%)",
        title=f"CoT Utility to Classmate Across RL Steps - Model Comparison",
        output_fn=f"{combined_output_dir}/cot_utility_across_RL_steps_comparison.png",
        max_y=max_y, min_y=min_y
    )



def visualize_cot_utility_across_RL_model_separate(main_model_name_or_path_list):
    """Visualize CoT utility for each classmate model separately, comparing main models."""
    for classmate in all_classmate_model_names:
        all_models_avg_metrics = {}
        
        for main_model_name_or_path in main_model_name_or_path_list:
            step_to_utility_avg = []
            for step_idx in tqdm(all_step_idx, desc=f"{main_model_name_or_path} - {classmate}"):
                step_str = "wo_cot" if step_idx == "wo_cot" else f"step_{step_idx}"
                total_classmate_utility = 0
                for dataset_name in all_dataset_names:
                    subset_name_list = DATASET_NAME_TO_SUBSET_NAME_LIST[dataset_name]
                    total_subset_classmate_utility = 0
                    for subset_name in subset_name_list:
                        if "mmlu" in dataset_name:
                            dataset_object = DATASET_NAME_TO_CLASS[dataset_name](subject=subset_name)
                        else:
                            dataset_object = DATASET_NAME_TO_CLASS[dataset_name]()
                        result_jsonl_fn = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/{step_str}/{classmate}/{dataset_name}{subset_name}/preds.jsonl"
                        total_utility = 0
                        all_correct_lines = []
                        all_incorrect_lines = []
                        with open(result_jsonl_fn, "r") as result_jsonl:
                            # ['question', 'main_CoT', 'main_cot_exclude_last_step', 'main_pred', 'classmate_pred', 'gt']
                            lines = result_jsonl.readlines()
                            sample_num = len(lines)
                            for line in lines:
                                data = json.loads(line)
                                if dataset_object.compare_pred_and_gt(data["classmate_continuation"], data["gt"]):
                                    total_utility += 1
                                    all_correct_lines.append(line)
                                else:
                                    all_incorrect_lines.append(line)
                        incorrect_output_jsonl_fn = result_jsonl_fn.replace("preds.jsonl", "incorrect_preds.jsonl")
                        correct_output_jsonl_fn = result_jsonl_fn.replace("preds.jsonl", "correct_preds.jsonl")
                        if os.path.exists(incorrect_output_jsonl_fn):
                            os.remove(incorrect_output_jsonl_fn)
                        if os.path.exists(correct_output_jsonl_fn):
                            os.remove(correct_output_jsonl_fn)
                        with open(incorrect_output_jsonl_fn, "w") as incorrect_output_jsonl:
                            incorrect_output_jsonl.writelines(all_incorrect_lines)
                        with open(correct_output_jsonl_fn, "w") as correct_output_jsonl:
                            correct_output_jsonl.writelines(all_correct_lines)
                        total_subset_classmate_utility += total_utility / sample_num * 100
                    total_classmate_utility += total_subset_classmate_utility / len(subset_name_list)
                step_to_utility_avg.append(total_classmate_utility / len(all_dataset_names))
            
            # Store results for this model
            model_short_name = main_model_name_or_path.replace('grpo_olmo_1B_hendrycks_math_', '').replace('_3298240_episodes', '').replace('20251122-', '')
            all_models_avg_metrics[model_short_name] = step_to_utility_avg
        
        # Visualize all models together for this classmate
        combined_output_dir = f"outputs/cot_utility_to_classmate/combined_comparison/{classmate}"
        os.makedirs(combined_output_dir, exist_ok=True)
        
        visualize_avg_std_across_steps(
            avg_vals_dict=all_models_avg_metrics,
            std_vals_dict={k: None for k in all_models_avg_metrics.keys()},
            x_labels=all_step_idx,
            x_label_name="RL Steps",
            y_label=f"CoT Utility to {classmate} (%)",
            title=f"CoT Utility to {classmate} Across RL Steps - Model Comparison",
            output_fn=f"{combined_output_dir}/cot_utility_across_RL_steps_comparison.png",
            max_y=max_y, min_y=min_y
        )


def visualize_step_to_correct_dist(model_name, step_to_correct_dist, output_fn, x_labels):
    """
    A list of
    step_correct_dist_dict = {
        "main_correct": {
            "classmate_correct_percent": 0.0,
            "classmate_incorrect_percent": 0.0
        },
        "main_incorrect": {
            "classmate_correct_percent": 0.0,
            "classmate_incorrect_percent": 0.0
        }
    }
    x_labels: step indices
    """
    import matplotlib.pyplot as plt

    # Extract data for plotting
    main_correct_classmate_correct = [d["main_correct"]["classmate_correct_percent"] for d in step_to_correct_dist]
    main_correct_classmate_incorrect = [d["main_correct"]["classmate_incorrect_percent"] for d in step_to_correct_dist]
    main_incorrect_classmate_correct = [d["main_incorrect"]["classmate_correct_percent"] for d in step_to_correct_dist]
    main_incorrect_classmate_incorrect = [d["main_incorrect"]["classmate_incorrect_percent"] for d in step_to_correct_dist]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Define colors for classmate correct/incorrect (same across both bar groups)
    color_classmate_correct = '#2ecc71'  # Green
    color_classmate_incorrect = '#e74c3c'  # Red

    # Create x-axis positions
    x = np.arange(len(x_labels))
    width = 0.35  # Width of bars

    # Plot stacked bars for main_correct
    ax.bar(x - width/2, main_correct_classmate_correct, width,
           label='Classmate Correct', color=color_classmate_correct, edgecolor='white')
    ax.bar(x - width/2, main_correct_classmate_incorrect, width,
           bottom=main_correct_classmate_correct,
           label='Classmate Incorrect', color=color_classmate_incorrect, edgecolor='white')

    # Plot stacked bars for main_incorrect
    ax.bar(x + width/2, main_incorrect_classmate_correct, width,
           color=color_classmate_correct, edgecolor='white')
    ax.bar(x + width/2, main_incorrect_classmate_incorrect, width,
           bottom=main_incorrect_classmate_correct,
           color=color_classmate_incorrect, edgecolor='white')

    # Add labels and title
    ax.set_xlabel('RL Steps', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'{model_name}: Main Model Correct/Incorrect Distribution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add text labels on bars to indicate main_correct vs main_incorrect
    for i, x_pos in enumerate(x):
        # Label for main_correct bar
        ax.text(x_pos - width/2, -5, 'Main\nCorrect',
               ha='center', va='top', fontsize=8, fontweight='bold')
        # Label for main_incorrect bar
        ax.text(x_pos + width/2, -5, 'Main\nIncorrect',
               ha='center', va='top', fontsize=8, fontweight='bold')
        
        # Add percentage labels on top of green bars (classmate correct)
        # For main_correct bar
        main_correct_total = main_correct_classmate_correct[i] + main_correct_classmate_incorrect[i]
        if main_correct_total > 0:
            main_correct_pct = (main_correct_classmate_correct[i] / main_correct_total) * 100
            ax.text(x_pos - width/2, main_correct_total + 5, f'{main_correct_pct:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2ecc71')
        
        # For main_incorrect bar
        main_incorrect_total = main_incorrect_classmate_correct[i] + main_incorrect_classmate_incorrect[i]
        if main_incorrect_total > 0:
            main_incorrect_pct = (main_incorrect_classmate_correct[i] / main_incorrect_total) * 100
            ax.text(x_pos + width/2, main_incorrect_total + 5, f'{main_incorrect_pct:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2ecc71')

    plt.tight_layout()

    # Save the figure
    output_dir = os.path.dirname(output_fn)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to {output_fn}")




def visualize_cot_utility_across_RL_for_each_model_dataset(main_model_name_or_path_list):
    """Visualize CoT utility for each classmate-dataset combination, comparing main models."""
    for classmate in all_classmate_model_names:
        for dataset_name in all_dataset_names:
            all_models_avg_metrics = {}
            subset_name_list = DATASET_NAME_TO_SUBSET_NAME_LIST[dataset_name]

            for main_model_name_or_path in main_model_name_or_path_list:
                step_to_utility_avg = []
                # step_to_correct_dist = []
                for step_idx in tqdm(all_step_idx, desc=f"{main_model_name_or_path} - {classmate} - {dataset_name}"):
                    step_str = "wo_cot" if step_idx == "wo_cot" else f"step_{step_idx}"
                    total_subset_classmate_utility = 0

                    # step_correct_dist_dict = {
                    #     "main_correct": {
                    #         "classmate_correct_percent": 0.0,
                    #         "classmate_incorrect_percent": 0.0
                    #     },
                    #     "main_incorrect": {
                    #         "classmate_correct_percent": 0.0,
                    #         "classmate_incorrect_percent": 0.0
                    #     }
                    # }

                    for subset_name in subset_name_list:
                        if "mmlu" in dataset_name:
                            dataset_object = DATASET_NAME_TO_CLASS[dataset_name](subject=subset_name)
                        else:
                            dataset_object = DATASET_NAME_TO_CLASS[dataset_name]()
                        result_jsonl_fn = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/{step_str}/{classmate}/{dataset_name}{subset_name}/preds.jsonl"
                        total_utility = 0
                        # subset_correct_dist_dict = {
                        #     "main_correct": {
                        #         "classmate_correct": 0.0,
                        #         "classmate_incorrect": 0.0
                        #     },
                        #     "main_incorrect": {
                        #         "classmate_correct": 0.0,
                        #         "classmate_incorrect": 0.0
                        #     }
                        # }
                        with open(result_jsonl_fn, "r") as result_jsonl:
                            lines = result_jsonl.readlines()
                            for line in lines:
                                data = json.loads(line)

                                # TODO haha
                                if dataset_object.compare_pred_and_gt(data["classmate_continuation"], data["gt"]):
                                    total_utility += 1
                                #     if step_idx != "wo_cot":
                                #         if dataset_object.compare_pred_and_gt(data["main_CoT"], data["gt"]):
                                #             subset_correct_dist_dict["main_correct"]["classmate_correct"] += 1
                                #         else:
                                #             subset_correct_dist_dict["main_incorrect"]["classmate_correct"] += 1
                                # else:
                                #     if step_idx != "wo_cot":
                                #         if dataset_object.compare_pred_and_gt(data["main_CoT"], data["classmate_continuation"]):
                                #             subset_correct_dist_dict["main_correct"]["classmate_incorrect"] += 1
                                #         else:
                                #             subset_correct_dist_dict["main_incorrect"]["classmate_incorrect"] += 1

                                # if step_idx != "wo_cot":
                                #     if dataset_object.compare_pred_and_gt(data["main_CoT"], data["gt"]):
                                #         if dataset_object.compare_pred_and_gt(data["classmate_continuation"], data["gt"]):
                                #             subset_correct_dist_dict["main_correct"]["classmate_correct"] += 1
                                #         else:
                                #             subset_correct_dist_dict["main_correct"]["classmate_incorrect"] += 1
                                #     else:
                                #         if dataset_object.compare_pred_and_gt(data["classmate_continuation"], data["gt"]):
                                #         # if dataset_object.compare_pred_and_gt(data["classmate_continuation"], data["main_CoT"]):
                                #             subset_correct_dist_dict["main_incorrect"]["classmate_correct"] += 1
                                #         else:
                                #             subset_correct_dist_dict["main_incorrect"]["classmate_incorrect"] += 1

                        sample_num = len(lines)
                        total_subset_classmate_utility += total_utility / sample_num * 100

                        # main_correct_cnt = subset_correct_dist_dict["main_correct"]["classmate_correct"] + subset_correct_dist_dict["main_correct"]["classmate_incorrect"]
                        # main_incorrect_cnt = subset_correct_dist_dict["main_incorrect"]["classmate_correct"] + subset_correct_dist_dict["main_incorrect"]["classmate_incorrect"]

                        # step_correct_dist_dict["main_correct"]["classmate_correct_percent"] += subset_correct_dist_dict["main_correct"]["classmate_correct"] / main_correct_cnt * 100 if main_correct_cnt > 0 else 0.0
                        # step_correct_dist_dict["main_correct"]["classmate_incorrect_percent"] += subset_correct_dist_dict["main_correct"]["classmate_incorrect"] / main_correct_cnt * 100 if main_correct_cnt > 0 else 0.0
                        # step_correct_dist_dict["main_incorrect"]["classmate_correct_percent"] += subset_correct_dist_dict["main_incorrect"]["classmate_correct"] / main_incorrect_cnt * 100 if main_incorrect_cnt > 0 else 0.0
                        # step_correct_dist_dict["main_incorrect"]["classmate_incorrect_percent"] += subset_correct_dist_dict["main_incorrect"]["classmate_incorrect"] / main_incorrect_cnt * 100 if main_incorrect_cnt > 0 else 0.0

                    step_to_utility_avg.append(total_subset_classmate_utility / len(subset_name_list))

                    # step_correct_dist_dict["main_correct"]["classmate_correct_percent"] /= len(subset_name_list)
                    # step_correct_dist_dict["main_correct"]["classmate_incorrect_percent"] /= len(subset_name_list)
                    # step_correct_dist_dict["main_incorrect"]["classmate_correct_percent"] /= len(subset_name_list)
                    # step_correct_dist_dict["main_incorrect"]["classmate_incorrect_percent"] /= len(subset_name_list)
                    # step_to_correct_dist.append(step_correct_dist_dict)
                
                # Store results for this model
                model_short_name = main_model_name_or_path.replace('grpo_olmo_1B_hendrycks_math_', '').replace('_3298240_episodes', '').replace('20251122-', '')
                all_models_avg_metrics[model_short_name] = step_to_utility_avg

                # Visualize step_to_correct_dist for this model
                # dist_output_dir = f"outputs/cot_utility_to_classmate/combined_comparison/{classmate}/{dataset_name}"
                # os.makedirs(dist_output_dir, exist_ok=True)
                # dist_output_fn = f"{dist_output_dir}/{model_short_name}_correct_distribution.png"
                # visualize_step_to_correct_dist(model_short_name, step_to_correct_dist, dist_output_fn, all_step_idx)


            # Visualize all models together for this classmate-dataset combination
            combined_output_dir = f"outputs/cot_utility_to_classmate/combined_comparison/{classmate}/{dataset_name}"
            os.makedirs(combined_output_dir, exist_ok=True)

            # Save all_models_avg_metrics
            with open(f"{combined_output_dir}/all_models_avg_metrics.json", "w") as f:
                json.dump(all_models_avg_metrics, f, indent=4)
            
            visualize_avg_std_across_steps(
                avg_vals_dict=all_models_avg_metrics,
                std_vals_dict={k: None for k in all_models_avg_metrics.keys()},
                x_labels=all_step_idx,
                x_label_name="RL Steps",
                y_label=f"CoT Utility to {classmate} (%)",
                title=f"CoT Utility to {classmate} on {dataset_name} - Model Comparison",
                output_fn=f"{combined_output_dir}/cot_utility_across_RL_steps_comparison.png",
                max_y=max_y, min_y=min_y
            )





# def characterize_cot():
#     for classmate in all_classmate_model_names:
#         for dataset_name in all_dataset_names:
#             subset_name_list = DATASET_NAME_TO_SUBSET_NAME_LIST[dataset_name]
#             for subset_name in subset_name_list:
#                 if "mmlu" in dataset_name:
#                     dataset_object = DATASET_NAME_TO_CLASS[dataset_name](subject=subset_name)
#                 else:
#                     dataset_object = DATASET_NAME_TO_CLASS[dataset_name]()
#                 identified_lines = []
#                 result_jsonl_fn_120 = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/step_{three_steps_for_characterizing_cot[0]}/{classmate}/{dataset_name}{subset_name}/preds.jsonl"
#                 with open(result_jsonl_fn_120, "r") as result_jsonl:
#                     # ['question', 'main_CoT', 'main_cot_exclude_last_step', 'main_pred', 'classmate_pred', 'gt']
#                     lines = result_jsonl.readlines()
#                     for line in lines:
#                         data = json.loads(line)
#                         if not dataset_object.compare_pred_and_gt(data["classmate_continuation"], data["gt"]):
#                             result_jsonl_fn_360 = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/step_{three_steps_for_characterizing_cot[1]}/{classmate}/{dataset_name}{subset_name}/preds.jsonl"
#                             with open(result_jsonl_fn_360, "r") as result_jsonl_360:
#                                 # ['question', 'main_CoT', 'main_cot_exclude_last_step', 'main_pred', 'classmate_pred', 'gt']
#                                 lines_360 = result_jsonl_360.readlines()
#                                 for line_360 in lines_360:
#                                     data_360 = json.loads(line_360)
#                                     if data["question"] == data_360["question"]:
#                                         if dataset_object.compare_pred_and_gt(data_360["classmate_continuation"], data_360["gt"]):
#                                             result_jsonl_fn_720 = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/step_{three_steps_for_characterizing_cot[2]}/{classmate}/{dataset_name}{subset_name}/preds.jsonl"
#                                             with open(result_jsonl_fn_720, "r") as result_jsonl_720:
#                                                 # ['question', 'main_CoT', 'main_cot_exclude_last_step', 'main_pred', 'classmate_pred', 'gt']
#                                                 lines_720 = result_jsonl_720.readlines()
#                                                 for line_720 in lines_720:
#                                                     data_720 = json.loads(line_720)
#                                                     if data["question"] == data_720["question"]:
#                                                         if not dataset_object.compare_pred_and_gt(data_720["classmate_pred"], data_720["gt"]):
#                                                             identified_lines.append({
#                                                                 "question": data["question"],
#                                                                 f"main_CoT_{three_steps_for_characterizing_cot[0]}_exclude": data["main_cot_exclude_last_step"],
#                                                                 f"classmate_{three_steps_for_characterizing_cot[0]}_pred": data["classmate_pred"],
#                                                                 f"main_CoT_{three_steps_for_characterizing_cot[1]}_exclude": data_360["main_cot_exclude_last_step"],
#                                                                 f"classmate_{three_steps_for_characterizing_cot[1]}_pred": data_360["classmate_pred"],
#                                                                 f"main_CoT_{three_steps_for_characterizing_cot[2]}_exclude": data_720["main_cot_exclude_last_step"],
#                                                                 f"classmate_{three_steps_for_characterizing_cot[2]}_pred": data_720["classmate_pred"],
#                                                                 "gt": data["gt"]
#                                                             })
#                 output_fn = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/cot_case_studies/{classmate}/{dataset_name}{subset_name}_{three_step_str}_characterize.jsonl"
#                 output_dir = os.path.dirname(output_fn)
#                 if not os.path.exists(output_dir):
#                     os.makedirs(output_dir)
#                 with open(output_fn, "w") as output_jsonl:
#                     for item in identified_lines:
#                         output_jsonl.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    # TODO: Datasets & Classmate models
    all_dataset_names = ["hendrycks_math"]
    # all_dataset_names = ["gsm8k", "hendrycks_math", "aimo-validation-aime"]
    # all_classmate_model_names = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Ministral-8B-Instruct-2410", "Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "OLMo-2-0425-1B-Instruct"]
    # all_classmate_model_names = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Ministral-8B-Instruct-2410", "OLMo-2-0425-1B-Instruct", "Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "OLMo-2-0425-1B-SFT"]
    # all_classmate_model_names = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Ministral-8B-Instruct-2410", "Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "OLMo-2-0425-1B-SFT"]
    # all_classmate_model_names = ["Qwen2.5-7B-Instruct", "Llama-3.2-1B-Instruct", "OLMo-2-0425-1B-Instruct", "Llama-3.1-8B-Instruct"]
    # all_classmate_model_names = ["Qwen2.5-7B-Instruct", "Llama-3.2-3B-Instruct", "OLMo-2-0425-1B-Instruct"]

    # all_classmate_model_names = ["Llama-3.2-1B-Instruct", "Qwen2.5-7B-Instruct", "OLMo-2-0425-1B-Instruct"]
    all_classmate_model_names = ["Llama-3.2-1B-Instruct"]

    # TODO: Main Model and ckpt steps
    # main_model_name_or_path = "OLMo-2-1124-13B-Instruct"
    # main_model_name_or_path = "olmo2_rlvr_1b_20251020_checkpoints_gsm8k_w_question"
    # main_model_name_or_path = "grpo_olmo_1B_gsm8k_baseline"
    # main_model_name_or_path = "grpo_olmo_1B_gsm8k_baseline_400000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_with_classmate_llama_400000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_gsm8k_baseline_800000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_with_classmate_llama_800000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_with_classmate_llama_reward_weight_1_400000_episodes"

    # main_model_name_or_path = "20251122-grpo_olmo_1B_hendrycks_math_baseline_3298240_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_hendrycks_math_with_classmate_llama_reward_weight_1_3298240_episodes"

    # main_model_name_or_path_list = ["20251122-grpo_olmo_1B_hendrycks_math_baseline_3298240_episodes", "grpo_olmo_1B_hendrycks_math_with_classmate_llama_reward_weight_1_3298240_episodes"]
    # main_model_name_or_path_list = [
    #     "20251210-OLMo-2-1124-7B-DPO_RLVR-GSM-MATH-IF-Mixed-Constraints_baseline_237392_episodes",
    #     "20251210-OLMo-2-7B-DPO-Mixed-Constraints_with_classmate_reward_llama_237400_episodes"]
    # main_model_name_or_path_list = ["20251217-Qwen3-4B-Base_DeepScaleR_baseline_322512_episodes",
    #                                 "20251217-Qwen3-4B-Base_DeepScaleR_w_classmate_llama0.5_322512_episodes",
    #                                 "20251217-Qwen3-4B-Base_DeepScaleR_w_classmate_llama_322512_episodes"]

    # main_model_name_or_path_list = ["20251224-Qwen3-1.7B_GSM_MATH_baseline_357024_episodes", "20251224-Qwen3-1.7B_GSM_MATH_w_1_classmate_llama_357024_episodes"]
    # main_model_name_or_path_list = ["20251224-Qwen3-0.6B_GSM_MATH_baseline_357024_episodes", "20251224-Qwen3-0.6B_GSM_MATH_w_1_classmate_llama_357024_episodes_seed_42"]
    # main_model_name_or_path_list = [
    #     "20251228-Qwen3-0.6B_gsm8k_minimal_answer_box_prompt_baseline_717408_episodes_seed_42",
    #     "20251228-Qwen3-0.6B_gsm8k_minimal_answer_box_w_1_classmate_llama_717408_episodes_seed_42"
    # ]
    # main_model_name_or_path_list = [
    #     "20251229-Qwen3-0.6B_math_minimal_answer_box_baseline_719328_episodes_seed_42",
    #     "20251229-Qwen3-0.6B_math_minimal_answer_box_w_1_classmate_llama_719328_episodes_seed_42"]

    # main_model_name_or_path_list = [
    #     "20260105-Qwen3-0.6B_gsm8k_minimal_answer_box_baseline_1434816_episodes_seed_42",
    #     "20260105-Qwen3-0.6B_gsm8k_minimal_answer_box_w_1_classmate_llama_1434816_episodes_seed_42",
    #     "20260105-Qwen3-0.6B_gsm8k_no_classmate_main_incorrect_1_llama_1434816_episodes_seed_42",
    #     "20260112-Qwen3-0.6B_gsm8k_main_cl_separate_no_cl_main_incorrect_1_llama_1434816_episodes_seed_42",
        # "20260113-Qwen3-0.6B_gsm8k_dgpo_no_cl_main_wrong_cl_partial_1_llama_1434816_episodes_seed_42",
        # "20260113-Qwen3-0.6B_gsm8k-m_cl_sep_norm-no_cl_m_wrong-non_neg-cl_partial_1_llama_1434816_episode_seed_42"
        # "20260115-Qwen3-0.6B_gsm8k_m_cl_sep_norm_always_cl_partial_llama_1434816_episodes_seed_42"
    # ]

    # gsm8k
    # main_model_name_or_path_list = [
    #     "20260117-Qwen3-1.7B-Base_gsm8k_minimal_answer_box_prompt_baseline_1195680_episodes_seed_42",
    #     "20260118-Qwen3-1.7B-Base_gsm8k_m_cl_sep_no_cl_m_wrong_partial_llama_1195680_episodes_seed_42"
    # ]
    #
    # max_step_num = 6  # start form 0
    # step_size = 84

    # MATH
    # main_model_name_or_path_list = [
    #     "20260117-Qwen3-1.7B-Base_math_answer_box_baseline_719328_episodes_seed_42",
    #     "20260118-Qwen3-1.7B-Base_MATH_m_cl_sep_no_cl_m_wrong_partial_llama_719328_episodes_seed_42",
    #     "20260120-Qwen3-1.7Base_MATH_m_cl_sep_keep_0.5_no_cl_m_inc_partial_llama_719328_episodes_seed_42"
    # ]
    # max_step_num = 7  # start form 0
    # step_size = 50

    main_model_name_or_path_list = [
        "20260126-Qwen3-1.7B-Base_MATH_700_heldout_baseline_652224_episodes_seed_42",
        "20260126-Qwen3-1.7Base_MATH700heldout_vanilla_no_cl_m_wrong_consis_partial_llama_652224_ep_s_42",
        "20260129-Qwen3-1.7Base_MATH700heldout_vanilla_always_cl_consis_partial_llama_652224_ep_s_42"
    ]
    max_step_num = 7  # start form 0
    step_size = 46


    max_y = None
    min_y = None

    # three_steps_for_characterizing_cot = ["19", "171", "361"]
    # three_step_str = "_".join(three_steps_for_characterizing_cot)

    # gsm8k
    # all_step_idx = ["wo_cot", "base"] + [str(i * step_size) for i in range(1, (max_step_num+1) + 1)]

    #MATH
    all_step_idx = ["wo_cot", "base"] + [str(69 + i * step_size) for i in range(0, (max_step_num+1))]
    # all_step_idx = ["base"]
    skip_indices = []
    all_step_idx = [idx for idx in all_step_idx if idx not in skip_indices]

    # print("Utility macro-avg across all models and datasets")
    # visualize_cot_utility_across_RL(main_model_name_or_path_list)
    # print("Utility macro-avg across all datasets for each model")
    # visualize_cot_utility_across_RL_model_separate(main_model_name_or_path_list)
    print("Utility for each model on each dataset")
    visualize_cot_utility_across_RL_for_each_model_dataset(main_model_name_or_path_list)

    # print("Characterize CoT")
    # characterize_cot()

#bash scripts/visualize_cot_utility_across_RL_across_models.sh