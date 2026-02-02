import os
import re
import json

import numpy as np
import pandas as pd
import seaborn as sns
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
        elif "always_cl_consis_partial_llama" in model_name:
            model_name = "separate main cl norm + always classmate + consistency + classmate non-truncated"
        elif "no_cl_m_wrong_consis_partial" in model_name:
            model_name = "separate main cl norm + no classmate when main incorrect + consistency + classmate non-truncated"
        elif "m_cl_sep_keep_0.5_no_cl_m_inc_partial" in model_name:
            model_name = "separate main cl norm + no classmate when main incorrect + classmate non-truncated + keep 0.5"
        elif "main_cl_separate_no_cl_main_incorrect" in model_name or "m_cl_sep_no_cl_m_wrong_partial" in model_name:
            model_name = "separate main cl norm + no classmate when main incorrect + classmate non-truncated"
        elif "w_1_classmate" in model_name:
            model_name = "always have classmate reward + classmate all"
        elif "no_classmate_main_incorrect" in model_name:
            model_name = "no classmate reward when main incorrect + classmate all"
        elif "dgpo_no_cl_main_wrong_cl_partial" in model_name:
            model_name = "DGPO + no classmate when main incorrect + classmate non-truncated"
        elif "m_cl_sep_norm_always_cl_partial" in model_name:
            model_name = "separate main cl norm + always classmate + classmate non-truncated"

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
        # Apply the same legend name transformations
        # display_name = model_name
        # if "baseline" in model_name:
        #     display_name = "baseline"
        # elif "main_cl_separate_no_cl_main_incorrect" in model_name or "m_cl_sep_no_cl_m_wrong_partial" in model_name:
        #     display_name = "separate main cl norm + no classmate when main incorrect + classmate non-truncated"
        # elif "w_1_classmate" in model_name:
        #     display_name = "always have classmate reward + classmate all"
        # elif "no_classmate_main_incorrect" in model_name:
        #     display_name = "no classmate reward when main incorrect + classmate all"
        # elif "dgpo_no_cl_main_wrong_cl_partial" in model_name:
        #     display_name = "DGPO + no classmate when main incorrect + classmate non-truncated"
        # elif "m_cl_sep_norm_always_cl_partial" in model_name:
        #     display_name = "separate main cl norm + always classmate + classmate non-truncated"
        
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


def keep_number_digits_only(s):
    return re.sub(r"\D", "", s)

def use_gpt_to_check_intermediate_steps(question: str, main_model_CoT: str, model: str = "gpt-5.2-nano", temperature: float = 0.0):
    """Use GPT to check if every intermediate step in the Chain of Thought is correct.
    
    Args:
        question: The original question
        main_model_CoT: The Chain of Thought reasoning from the main model
        model: The GPT model to use (default: gpt-4)
        temperature: Temperature for sampling (default: 0.0 for deterministic)
    
    Returns:
        dict: A dictionary with keys:
            - all_correct (bool): True if all steps are correct, False otherwise
            - num_of_errors (int): Number of errors found
            - reason (str): Explanation of the errors or confirmation of correctness
    """
    
    zero_shot_prompt_template = """Analyze this Chain of Thought reasoning step-by-step:

Question: {question}

Chain of Thought: {cot}

Check each step for arithmetic errors, logical inconsistencies, invalid assumptions, and incorrect conclusions.

Respond in JSON format:
{{
    "all_correct": True or False,
    "num_of_errors": number of errors found (0 if all correct),
    "reason": "brief explanation of errors or confirmation if correct"
}}"""

    # Initialize OpenAI client
    client = OpenAI()  # This will use the OPENAI_API_KEY environment variable
    
    try:
        # Format the prompt
        response = client.responses.create(
            model=model,
            # reasoning={"effort": "low"},
            instructions="You are a mathematical reasoning expert. Your task is to carefully check every intermediate step in a Chain of Thought (CoT).",
            input=zero_shot_prompt_template.format(question=question, cot=main_model_CoT),
        )
        # Parse the response
        result = json.loads(response.output_text)
        
        # Validate the response has required fields
        if "all_correct" not in result or "num_of_errors" not in result or "reason" not in result:
            return {
                "all_correct": False,
                "num_of_errors": -1,
                "reason": "Invalid response format from GPT"
            }
        
        return result
        
    except Exception as e:
        # Handle any errors
        return {
            "all_correct": False,
            "num_of_errors": -1,
            "reason": f"Error calling OpenAI API: {str(e)}"
        }



def calculate_main_model_acc_and_filter_out_incorrect_cot(main_model_name_or_path_list, max_y=None, min_y=None):
    """Calculate accuracy for multiple models and plot them together.
    
    Args:
        main_model_name_or_path_list: List of model names/paths to process
        max_y: Maximum y value for plot
        min_y: Minimum y value for plot
    """
    import json

    for dataset_name in dataset_name_list:
        all_models_avg_metrics = {}
        all_models_std_metrics = {}

        for main_model_name_or_path in main_model_name_or_path_list:
            print(f"\n{'='*80}\nProcessing model: {main_model_name_or_path}\n{'='*80}")

            main_model_dir = f"{output_dir}/inference_main_model/{main_model_name_or_path}{template_str}"
            main_model_dataset_dir = f"{main_model_dir}/parsed_results/{dataset_name}"
            if os.path.exists(main_model_dataset_dir):
                os.system(f"rm -rf {main_model_dataset_dir}")
            os.makedirs(main_model_dataset_dir)

            all_steps_avg_metric = []
            all_steps_metric_std = []
            for step_idx in all_step_idx:
                all_subset_avg_utility = []
                for subset_name in tqdm(DATASET_NAME_TO_SUBSET_NAME_LIST[dataset_name], desc=f"Step {step_idx}"):
                    subset_acc_output_txt_fn = f"{main_model_dataset_dir}/{subset_name}_acc_cot_utility.txt"

                    if dataset_name == "mmlu":
                        dataset_object = DATASET_NAME_TO_CLASS[dataset_name](subset_name)
                    else:
                        dataset_object = DATASET_NAME_TO_CLASS[dataset_name]()
                    result_dir = f"{main_model_dir}/step_{step_idx}/{dataset_name}{subset_name}"
                    pred_jsonl_fn = f"{result_dir}/preds.jsonl"
                    all_lines = []
                    all_correct_lines = []
                    all_incorrect_lines = []
                    with open(pred_jsonl_fn, "r") as pred_jsonl:
                        lines = pred_jsonl.readlines()

                        total = len(lines)
                        if total > dataset_to_max_sample_num[dataset_name]:
                            lines = lines[:dataset_to_max_sample_num[dataset_name]]
                        assert len(lines) == dataset_to_max_sample_num[dataset_name], f"Expected {dataset_to_max_sample_num[dataset_name]} samples for {dataset_name}, but got {len(lines)}"
                        correct_cot_count = 0
                        for line in lines:
                            data = json.loads(line)
                            data["main_cot_exclude_last_step"] = dataset_object.remove_last_step(data)
                            if dataset_object.compare_pred_and_gt(data["main_CoT"], data["gt"]):
                                correct_cot_count += 1
                                data["main_is_correct"] = True
                                all_correct_lines.append(json.dumps(data) + "\n")
                            else:
                                data["main_is_correct"] = False
                                all_incorrect_lines.append(json.dumps(data) + "\n")
                            all_lines.append(json.dumps(data) + "\n")

                        accuracy = correct_cot_count / total * 100
                        all_subset_avg_utility.append(accuracy)
                        acc_str = f"Step {step_idx} {subset_name} Accuracy: {accuracy:.1f}% ({correct_cot_count}/{total})"
                        with open(subset_acc_output_txt_fn, "a") as acc_output_txt:
                            acc_output_txt.write(acc_str + "\n")
                        print(subset_name, acc_str)
                    correct_pred_jsonl_fn = f"{result_dir}/correct_preds.jsonl"
                    incorrect_pred_jsonl_fn = f"{result_dir}/incorrect_preds.jsonl"
                    if os.path.exists(correct_pred_jsonl_fn):
                        os.remove(correct_pred_jsonl_fn)
                    if os.path.exists(incorrect_pred_jsonl_fn):
                        os.remove(incorrect_pred_jsonl_fn)
                    os.remove(pred_jsonl_fn)
                    with open(correct_pred_jsonl_fn, "w") as correct_pred_jsonl:
                        correct_pred_jsonl.writelines(all_correct_lines)
                    with open(incorrect_pred_jsonl_fn, "w") as incorrect_pred_jsonl:
                        incorrect_pred_jsonl.writelines(all_incorrect_lines)
                    with open(pred_jsonl_fn, "w") as pred_jsonl:
                        pred_jsonl.writelines(all_lines)
                all_steps_avg_metric.append(np.mean(all_subset_avg_utility).item())
                all_steps_metric_std.append(np.std(all_subset_avg_utility).item())

            # Store results for this model
            model_short_name = main_model_name_or_path.replace('grpo_olmo_1B_hendrycks_math_', '').replace('_3298240_episodes', '').replace('20251122-', '')
            all_models_avg_metrics[model_short_name] = all_steps_avg_metric
            all_models_std_metrics[model_short_name] = all_steps_metric_std

        # Visualize all models together
        combined_output_dir = f"{output_dir}/inference_main_model/combined_comparison"
        os.makedirs(combined_output_dir, exist_ok=True)

        visualize_avg_std_across_steps(
            avg_vals_dict=all_models_avg_metrics,
            std_vals_dict=all_models_std_metrics,
            x_labels=all_step_idx,
            x_label_name="RL Steps",
            y_label="Main Model Accuracy (%)",
            title=f"Model Comparison across RL Steps on {dataset_name}",
            output_fn=f"{combined_output_dir}/{dataset_name}_comparison.png",
            max_y=max_y,
            min_y=min_y
        )


if __name__ == "__main__":
    template_str = ""

    # TODO change the following
    # main_model_name_or_path = "OLMo-2-1124-13B-Instruct"
    # max_step_num = 6
    # step_size = 120

    # main_model_name_or_path = "olmo2_rlvr_1b_20251020_checkpoints"
    # main_model_name_or_path = "grpo_olmo_1B_gsm8k_baseline"
    # main_model_name_or_path = "grpo_olmo_1B_with_classmate_llama"
    # main_model_name_or_path = "grpo_olmo_1B_gsm8k_baseline_400000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_with_classmate_llama_400000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_gsm8k_baseline_800000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_with_classmate_llama_800000_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_with_classmate_llama_reward_weight_1_400000_episodes"

    # main_model_name_or_path = "20251122-grpo_olmo_1B_hendrycks_math_baseline_3298240_episodes"
    # main_model_name_or_path = "grpo_olmo_1B_hendrycks_math_with_classmate_llama_reward_weight_1_3298240_episodes"

    # main_model_name_or_path_list = ["20251217-Qwen3-4B-Base_DeepScaleR_baseline_322512_episodes", "20251217-Qwen3-4B-Base_DeepScaleR_w_classmate_llama0.5_322512_episodes", "20251217-Qwen3-4B-Base_DeepScaleR_w_classmate_llama_322512_episodes"]
    # main_model_name_or_path_list = ["20251224-Qwen3-1.7B_GSM_MATH_baseline_357024_episodes", "20251224-Qwen3-1.7B_GSM_MATH_w_1_classmate_llama_357024_episodes"]
    # main_model_name_or_path_list = ["20251224-Qwen3-0.6B_GSM_MATH_baseline_357024_episodes", "20251224-Qwen3-0.6B_GSM_MATH_w_1_classmate_llama_357024_episodes_seed_42"]
    # main_model_name_or_path_list = [
    #     "20251226-Qwen3-0.6B_gsm8k_think_prompt_baseline_717408_episodes_seed_42",
    #     "20251226-Qwen3-0.6B_gsm8k_think_prompt_w_1_classmate_llama_358704_episodes_seed_42"]
    # main_model_name_or_path_list = [
    #     "20251228-Qwen3-0.6B_gsm8k_minimal_answer_box_prompt_baseline_717408_episodes_seed_42",
    #     "20251228-Qwen3-0.6B_gsm8k_minimal_answer_box_w_1_classmate_llama_717408_episodes_seed_42"]
    # main_model_name_or_path_list = [
    #     "20251229-Qwen3-0.6B_math_minimal_answer_box_baseline_719328_episodes_seed_42",
    #     "20251229-Qwen3-0.6B_math_minimal_answer_box_w_1_classmate_llama_719328_episodes_seed_42"]

    # main_model_name_or_path_list = [
        # "20260105-Qwen3-0.6B_gsm8k_minimal_answer_box_baseline_1434816_episodes_seed_42",
        # "20260105-Qwen3-0.6B_gsm8k_minimal_answer_box_w_1_classmate_llama_1434816_episodes_seed_42",
        # "20260105-Qwen3-0.6B_gsm8k_no_classmate_main_incorrect_1_llama_1434816_episodes_seed_42",
        # "20260112-Qwen3-0.6B_gsm8k_main_cl_separate_no_cl_main_incorrect_1_llama_1434816_episodes_seed_42",
        # "20260113-Qwen3-0.6B_gsm8k_dgpo_no_cl_main_wrong_cl_partial_1_llama_1434816_episodes_seed_42",
        # "20260115-Qwen3-0.6B_gsm8k_m_cl_sep_norm_always_cl_partial_llama_1434816_episodes_seed_42"
    # ]

    #gsm8k
    # main_model_name_or_path_list = [
    #     "20260117-Qwen3-1.7B-Base_gsm8k_minimal_answer_box_prompt_baseline_1195680_episodes_seed_42",
    #     "20260118-Qwen3-1.7B-Base_gsm8k_m_cl_sep_no_cl_m_wrong_partial_llama_1195680_episodes_seed_42"
    # ]
    # max_step_num = 6      # start form 0
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

    output_dir = "outputs"
    # dataset_name_list = ["gsm8k", "hendrycks_math", "aimo-validation-aime"]
    dataset_name_list = ["hendrycks_math"]
    # dataset_name = "gsm8k"
    # dataset_name = "hendrycks_math"
    # dataset_name = "aimo-validation-aime"
    max_y, min_y = None, None
    # max_y, min_y = 75, 65

    skip_indices = []
    # gsm8k
    # all_step_idx = ["base"] + [str(i * step_size) for i in range(1, (max_step_num+1) + 1) if str(i * step_size) not in skip_indices]

    # MATH
    all_step_idx = ["base"] + [str(69 + i * step_size) for i in range(0, (max_step_num+1)) if str(i * step_size) not in skip_indices]
    calculate_main_model_acc_and_filter_out_incorrect_cot(main_model_name_or_path_list, max_y=max_y, min_y=min_y)

#bash scripts/run_utils_across_models.sh