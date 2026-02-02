import json
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from my_eval.src.analysis_module.utils import DATASET_NAME_TO_CLASS, visualize_avg_std_across_steps
from my_eval.src.data_module.dataset_configs import DATASET_NAME_TO_SUBSET_NAME_LIST


def keep_number_digits_only(s):
    return re.sub(r"\D", "", s)


def visualize_cot_utility_across_RL():
    step_to_utility_avg = []
    step_to_utility_std = []
    for step_idx in tqdm(all_step_idx):
        step_str = "wo_cot" if step_idx == "wo_cot" else f"step_{step_idx}"
        step_to_classmate_avg_acc = []
        for classmate in all_classmate_model_names:
            print(f"Processing step {step_str}, classmate {classmate}...")
            total_classmate_utility = 0
            for dataset_name in all_dataset_names:
                print(f"Processing step {step_str}, dataset {dataset_name}...")
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
                        # assert len(lines) == 1319, f"{result_jsonl_fn} has {len(lines)} lines, expected 1319 for GSM8k."
                        assert len(lines) == 200, f"{result_jsonl_fn} has {len(lines)} lines, expected 200 for hendrycks math."
                        sample_num = len(lines)
                        for line in lines:
                            data = json.loads(line)
                            if dataset_object.compare_pred_and_gt(data["classmate_pred"], data["gt"]):
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

    # avg_vals, std_vals, x_labels, x_label_name, y_label, title, output_fn
    visualize_avg_std_across_steps(
        avg_vals=step_to_utility_avg,
        std_vals=step_to_utility_std,
        x_labels=all_step_idx,
        x_label_name="RL Steps",
        y_label="CoT Utility to Classmate (%)",
        title=f"CoT Utility to Classmate Across RL Steps ({main_model_name_or_path})",
        output_fn=f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/figures/cot_utility_across_RL_steps.png",
        # max_y=max_y,
        # min_y=min_y
    )



def visualize_cot_utility_across_RL_model_separate(max_y, min_y):
    for classmate in all_classmate_model_names:
        step_to_utility_avg = []
        for step_idx in tqdm(all_step_idx):
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
                            if dataset_object.compare_pred_and_gt(data["classmate_pred"], data["gt"]):
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
        # avg_vals, std_vals, x_labels, x_label_name, y_label, title, output_fn
        visualize_avg_std_across_steps(
            avg_vals=step_to_utility_avg,
            std_vals=None,
            x_labels=all_step_idx,
            x_label_name="RL Steps",
            y_label=f"CoT Utility to {classmate} (%)",
            title=f"CoT Utility to {classmate} Across RL Steps ({main_model_name_or_path})",
            output_fn=f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/figures/{classmate}_cot_utility_across_RL_steps.png",
            max_y=max_y, min_y=min_y
        )


def visualize_cot_utility_across_RL_for_each_model_dataset():
    for classmate in all_classmate_model_names:
        for dataset_name in all_dataset_names:
            subset_name_list = DATASET_NAME_TO_SUBSET_NAME_LIST[dataset_name]
            step_to_utility_avg = []
            for step_idx in tqdm(all_step_idx):
                step_str = "wo_cot" if step_idx == "wo_cot" else f"step_{step_idx}"
                total_subset_classmate_utility = 0
                for subset_name in subset_name_list:
                    if "mmlu" in dataset_name:
                        dataset_object = DATASET_NAME_TO_CLASS[dataset_name](subject=subset_name)
                    else:
                        dataset_object = DATASET_NAME_TO_CLASS[dataset_name]()
                    result_jsonl_fn = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/{step_str}/{classmate}/{dataset_name}{subset_name}/preds.jsonl"
                    total_utility = 0
                    with open(result_jsonl_fn, "r") as result_jsonl:
                        # ['question', 'main_CoT', 'main_cot_exclude_last_step', 'main_pred', 'classmate_pred', 'gt']
                        lines = result_jsonl.readlines()
                        for line in lines:
                            data = json.loads(line)
                            if dataset_object.compare_pred_and_gt(data["classmate_pred"], data["gt"]):
                                total_utility += 1
                            # else:
                            #     print(data["classmate_pred"])
                            #     print(data["gt"])
                            #     breakpoint()
                    sample_num = len(lines)
                    total_subset_classmate_utility += total_utility / sample_num * 100
                step_to_utility_avg.append(total_subset_classmate_utility / len(subset_name_list))

            # Visualize
            visualize_avg_std_across_steps(
                avg_vals=step_to_utility_avg,
                std_vals=None,
                x_labels=all_step_idx,
                x_label_name="RL Steps",
                y_label=f"CoT Utility to {classmate} (%)",
                title=f"CoT Utility to {classmate} Across RL Steps on {dataset_name} ({main_model_name_or_path})",
                output_fn=f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/figures/{dataset_name}/{classmate}_cot_utility_across_RL_steps.png",
                # max_y=100, min_y=35
            )





def characterize_cot():
    for classmate in all_classmate_model_names:
        for dataset_name in all_dataset_names:
            subset_name_list = DATASET_NAME_TO_SUBSET_NAME_LIST[dataset_name]
            for subset_name in subset_name_list:
                if "mmlu" in dataset_name:
                    dataset_object = DATASET_NAME_TO_CLASS[dataset_name](subject=subset_name)
                else:
                    dataset_object = DATASET_NAME_TO_CLASS[dataset_name]()
                identified_lines = []
                result_jsonl_fn_120 = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/step_{three_steps_for_characterizing_cot[0]}/{classmate}/{dataset_name}{subset_name}/preds.jsonl"
                with open(result_jsonl_fn_120, "r") as result_jsonl:
                    # ['question', 'main_CoT', 'main_cot_exclude_last_step', 'main_pred', 'classmate_pred', 'gt']
                    lines = result_jsonl.readlines()
                    for line in lines:
                        data = json.loads(line)
                        if not dataset_object.compare_pred_and_gt(data["classmate_pred"], data["gt"]):
                            result_jsonl_fn_360 = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/step_{three_steps_for_characterizing_cot[1]}/{classmate}/{dataset_name}{subset_name}/preds.jsonl"
                            with open(result_jsonl_fn_360, "r") as result_jsonl_360:
                                # ['question', 'main_CoT', 'main_cot_exclude_last_step', 'main_pred', 'classmate_pred', 'gt']
                                lines_360 = result_jsonl_360.readlines()
                                for line_360 in lines_360:
                                    data_360 = json.loads(line_360)
                                    if data["question"] == data_360["question"]:
                                        if dataset_object.compare_pred_and_gt(data_360["classmate_pred"], data_360["gt"]):
                                            result_jsonl_fn_720 = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/step_{three_steps_for_characterizing_cot[2]}/{classmate}/{dataset_name}{subset_name}/preds.jsonl"
                                            with open(result_jsonl_fn_720, "r") as result_jsonl_720:
                                                # ['question', 'main_CoT', 'main_cot_exclude_last_step', 'main_pred', 'classmate_pred', 'gt']
                                                lines_720 = result_jsonl_720.readlines()
                                                for line_720 in lines_720:
                                                    data_720 = json.loads(line_720)
                                                    if data["question"] == data_720["question"]:
                                                        if not dataset_object.compare_pred_and_gt(data_720["classmate_pred"], data_720["gt"]):
                                                            identified_lines.append({
                                                                "question": data["question"],
                                                                f"main_CoT_{three_steps_for_characterizing_cot[0]}_exclude": data["main_cot_exclude_last_step"],
                                                                f"classmate_{three_steps_for_characterizing_cot[0]}_pred": data["classmate_pred"],
                                                                f"main_CoT_{three_steps_for_characterizing_cot[1]}_exclude": data_360["main_cot_exclude_last_step"],
                                                                f"classmate_{three_steps_for_characterizing_cot[1]}_pred": data_360["classmate_pred"],
                                                                f"main_CoT_{three_steps_for_characterizing_cot[2]}_exclude": data_720["main_cot_exclude_last_step"],
                                                                f"classmate_{three_steps_for_characterizing_cot[2]}_pred": data_720["classmate_pred"],
                                                                "gt": data["gt"]
                                                            })
                output_fn = f"outputs/cot_utility_to_classmate/{main_model_name_or_path}/cot_case_studies/{classmate}/{dataset_name}{subset_name}_{three_step_str}_characterize.jsonl"
                output_dir = os.path.dirname(output_fn)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(output_fn, "w") as output_jsonl:
                    for item in identified_lines:
                        output_jsonl.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    # TODO: Datasets & Classmate models
    all_dataset_names = ["gsm8k", "hendrycks_math", "aime"]
    # all_dataset_names = ["hendrycks_math"]
    # all_classmate_model_names = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Ministral-8B-Instruct-2410", "Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "OLMo-2-0425-1B-Instruct"]
    # all_classmate_model_names = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Ministral-8B-Instruct-2410", "OLMo-2-0425-1B-Instruct", "Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "OLMo-2-0425-1B-SFT"]
    # all_classmate_model_names = ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Ministral-8B-Instruct-2410", "Qwen2.5-1.5B-Instruct", "Llama-3.2-1B-Instruct", "OLMo-2-0425-1B-SFT"]
    # all_classmate_model_names = ["Qwen2.5-7B-Instruct", "Llama-3.2-1B-Instruct", "OLMo-2-0425-1B-Instruct", "Llama-3.1-8B-Instruct"]
    all_classmate_model_names = ["Qwen2.5-7B-Instruct", "Llama-3.2-3B-Instruct", "OLMo-2-0425-1B-Instruct"]

    # TODO: Main Model and ckpt steps
    # main_model_name_or_path = "OLMo-2-1124-13B-Instruct"
    # main_model_name_or_path = "olmo2_rlvr_1b_20251020_checkpoints_gsm8k_w_question"
    # main_model_name_or_path = "grpo_olmo_1B_gsm8k_baseline"
    # main_model_name_or_path = "grpo_olmo_1B_gsm8k_baseline_400000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_with_classmate_llama_400000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_gsm8k_baseline_800000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_with_classmate_llama_800000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_with_classmate_llama_reward_weight_1_400000_episodes"

    main_model_name_or_path = "20251122-grpo_olmo_1B_hendrycks_math_baseline_3298240_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_hendrycks_math_with_classmate_llama_reward_weight_1_3298240_episodes"
    max_step_num = 39  # start form 0
    step_size = 19
    stride = 2

    # three_steps_for_characterizing_cot = ["19", "171", "361"]
    # three_step_str = "_".join(three_steps_for_characterizing_cot)

    # all_step_idx = ["wo_cot", "base"] + [str(i * step_size) for i in range(1, max_step_num + 1, stride)]
    all_step_idx = ["wo_cot", "base"] + [19, 209, 247, 285, 399, 437, 475, 589, 627, 665]
    skip_indices = []
    all_step_idx = [idx for idx in all_step_idx if idx not in skip_indices]
    print("Utility macro-avg across all models and datasets")
    visualize_cot_utility_across_RL()
    # print("Utility macro-avg across all datasets for each model")
    # visualize_cot_utility_across_RL_model_separate()
    print("Utility for each model on each dataset")
    visualize_cot_utility_across_RL_for_each_model_dataset()
    # print("Characterize CoT")
    # characterize_cot()

#bash scripts/visualize_cot_utility_across_RL.sh