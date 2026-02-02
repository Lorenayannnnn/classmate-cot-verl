import json

from tqdm import tqdm
from math_verify import parse, verify

from my_eval.src.data_module.dataset_configs import DATASET_NAME_TO_CLASS, DATASET_NAME_TO_SUBSET_NAME_LIST

def main_correctness_and_consistency():
    main_correct_classmate_correct_m_cl_same_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }
    main_correct_classmate_correct_m_cl_diff_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }

    main_incorrect_classmate_correct_m_cl_same_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }
    main_incorrect_classmate_correct_m_cl_diff_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }
    
    main_correct_classmate_incorrect_m_cl_same_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }
    main_correct_classmate_incorrect_m_cl_diff_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }

    main_incorrect_classmate_incorrect_m_cl_same_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }
    main_incorrect_classmate_incorrect_m_cl_diff_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }

    for step in tqdm(step_list):
        for model_name, model_fn_base in model_fn_list.items():
            subset_name_list = DATASET_NAME_TO_SUBSET_NAME_LIST[dataset_name]
            # TODO assuming subsets are balanced
            main_correct_classmate_correct_same_step_cnt = 0
            main_correct_classmate_correct_diff_step_cnt = 0
            main_incorrect_classmate_correct_same_step_cnt = 0
            main_incorrect_classmate_correct_diff_step_cnt = 0
            main_correct_classmate_incorrect_same_step_cnt = 0
            main_correct_classmate_incorrect_diff_step_cnt = 0
            main_incorrect_classmate_incorrect_same_step_cnt = 0
            main_incorrect_classmate_incorrect_diff_step_cnt = 0
            total_cnt = 0

            for subset_name in subset_name_list:
                model_fn = f"{model_fn_base}/step_{step}/Llama-3.2-1B-Instruct/{dataset_name}{subset_name}/preds.jsonl"
                # print(f"================= Step {step} mode {model_name} =================")
                with open(model_fn) as f:
                    preds = [line.strip() for line in f]

                for entry in preds:
                    entry = json.loads(entry)
                    gt = entry['gt']
                    classmate_is_correct = dataset_obj.compare_pred_and_gt(entry["classmate_continuation"], gt)
                    main_is_correct = dataset_obj.compare_pred_and_gt(entry["main_CoT"], gt)
                    
                    # Check if main and classmate have the same final answer
                    main_answer = parse(entry["main_CoT"])
                    classmate_answer = parse(entry["classmate_continuation"])
                    is_same_answer = verify(main_answer, classmate_answer)
                    # verify(entry["classmate_continuation"], gt)
                    # verify(classmate_answer, test_gt)
                    # verify(test_gt, classmate_answer)
                    # test_gt = parse(gt)

                    if main_is_correct and classmate_is_correct:
                        if is_same_answer:
                            main_correct_classmate_correct_same_step_cnt += 1
                        else:
                            main_correct_classmate_correct_diff_step_cnt += 1
                    elif not main_is_correct and classmate_is_correct:
                        if is_same_answer:
                            main_incorrect_classmate_correct_same_step_cnt += 1
                            print("Found case where both main and classmate are correct but have the same answer:")
                            breakpoint()
                        else:
                            main_incorrect_classmate_correct_diff_step_cnt += 1
                    elif main_is_correct and not classmate_is_correct:
                        if is_same_answer:
                            main_correct_classmate_incorrect_same_step_cnt += 1
                            print("Found case where main is correct and classmate is incorrect but have the same answer:")
                            breakpoint()
                        else:
                            main_correct_classmate_incorrect_diff_step_cnt += 1
                    elif not main_is_correct and not classmate_is_correct:
                        if is_same_answer:
                            main_incorrect_classmate_incorrect_same_step_cnt += 1
                        else:
                            main_incorrect_classmate_incorrect_diff_step_cnt += 1

                total_cnt += len(preds)

            main_correct_classmate_correct_m_cl_same_step_percentage[model_name].append(main_correct_classmate_correct_same_step_cnt / total_cnt * 100)
            main_correct_classmate_correct_m_cl_diff_step_percentage[model_name].append(main_correct_classmate_correct_diff_step_cnt / total_cnt * 100)
            main_incorrect_classmate_correct_m_cl_same_step_percentage[model_name].append(main_incorrect_classmate_correct_same_step_cnt / total_cnt * 100)
            main_incorrect_classmate_correct_m_cl_diff_step_percentage[model_name].append(main_incorrect_classmate_correct_diff_step_cnt / total_cnt * 100)
            main_correct_classmate_incorrect_m_cl_same_step_percentage[model_name].append(main_correct_classmate_incorrect_same_step_cnt / total_cnt * 100)
            main_correct_classmate_incorrect_m_cl_diff_step_percentage[model_name].append(main_correct_classmate_incorrect_diff_step_cnt / total_cnt * 100)
            main_incorrect_classmate_incorrect_m_cl_same_step_percentage[model_name].append(main_incorrect_classmate_incorrect_same_step_cnt / total_cnt * 100)
            main_incorrect_classmate_incorrect_m_cl_diff_step_percentage[model_name].append(main_incorrect_classmate_incorrect_diff_step_cnt / total_cnt * 100)

    # Plot in eight subplots in the same image
    import matplotlib.pyplot as plt

    plot_configs = [
        (main_correct_classmate_correct_m_cl_same_step_percentage, 'Main correct AND classmate correct (same answer) ↑', (0, 0)),
        (main_correct_classmate_correct_m_cl_diff_step_percentage, 'Main correct AND classmate correct (diff answer) ↓', (0, 1)),
        (main_incorrect_classmate_correct_m_cl_same_step_percentage, 'Main incorrect AND classmate correct (same answer) ↓', (0, 2)),
        (main_incorrect_classmate_correct_m_cl_diff_step_percentage, 'Main incorrect AND classmate correct (diff answer) ↓', (0, 3)),
        (main_correct_classmate_incorrect_m_cl_same_step_percentage, 'Main correct AND classmate incorrect (same answer) ↓', (1, 0)),
        (main_correct_classmate_incorrect_m_cl_diff_step_percentage, 'Main correct AND classmate incorrect (diff answer) ↓', (1, 1)),
        (main_incorrect_classmate_incorrect_m_cl_same_step_percentage, 'Main incorrect AND classmate incorrect (same answer)', (1, 2)),
        (main_incorrect_classmate_incorrect_m_cl_diff_step_percentage, 'Main incorrect AND classmate incorrect (diff answer)', (1, 3))
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for data, title, (i, j) in plot_configs:
        for model_name in model_fn_list:
            axes[i, j].plot(step_list, data[model_name], label=labels[model_name], marker=markers[model_name])
        axes[i, j].set_xticks(step_list)
        axes[i, j].set_xlabel('Training Steps')
        axes[i, j].set_ylabel('Percentage (%)')
        axes[i, j].set_title(title)
        axes[i, j].legend()
        axes[i, j].grid()

    plt.tight_layout()
    plt.savefig(f"{result_base_dir}/{dataset_name}_main_classmate_acc_consistency_analysis.png")
    plt.close()

def main():
    main_correct_classmate_correct_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }
    main_incorrect_classmate_correct_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }
    main_correct_classmate_incorrect_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }
    main_incorrect_classmate_incorrect_step_percentage = {
        model_fn: [] for model_fn in model_fn_list
    }

    for step in tqdm(step_list):
        for model_name, model_fn_base in model_fn_list.items():
            subset_name_list = DATASET_NAME_TO_SUBSET_NAME_LIST[dataset_name]
            # TODO assuming subsets are balanced
            main_correct_classmate_correct_cnt = 0
            main_incorrect_classmate_correct_cnt = 0
            main_correct_classmate_incorrect_cnt = 0
            main_incorrect_classmate_incorrect_cnt = 0
            total_cnt = 0

            for subset_name in subset_name_list:
                model_fn = f"{model_fn_base}/step_{step}/Llama-3.2-1B-Instruct/{dataset_name}/{subset_name}/preds.jsonl"
                # print(f"================= Step {step} mode {model_name} =================")
                with open(model_fn) as f:
                    preds = [line.strip() for line in f]

                for entry in preds:
                    entry = json.loads(entry)
                    gt = entry['gt']
                    classmate_is_correct = dataset_obj.compare_pred_and_gt(entry["classmate_continuation"], gt)
                    main_is_correct = dataset_obj.compare_pred_and_gt(entry["main_CoT"], gt)
                    if main_is_correct and classmate_is_correct:
                        main_correct_classmate_correct_cnt += 1
                    elif not main_is_correct and classmate_is_correct:
                        main_incorrect_classmate_correct_cnt += 1
                    elif main_is_correct and not classmate_is_correct:
                        main_correct_classmate_incorrect_cnt += 1
                    elif not main_is_correct and not classmate_is_correct:
                        main_incorrect_classmate_incorrect_cnt += 1

                total_cnt += len(preds)

            # total_cnt = len(preds)
            main_correct_classmate_correct_percentage = main_correct_classmate_correct_cnt / total_cnt * 100
            main_incorrect_classmate_correct_percentage = main_incorrect_classmate_correct_cnt / total_cnt * 100
            main_correct_classmate_incorrect_percentage = main_correct_classmate_incorrect_cnt / total_cnt * 100
            main_incorrect_classmate_incorrect_percentage = main_incorrect_classmate_incorrect_cnt / total_cnt * 100

            main_correct_classmate_correct_step_percentage[model_name].append(main_correct_classmate_correct_percentage)
            main_incorrect_classmate_correct_step_percentage[model_name].append(
                main_incorrect_classmate_correct_percentage)
            main_correct_classmate_incorrect_step_percentage[model_name].append(
                main_correct_classmate_incorrect_percentage)
            main_incorrect_classmate_incorrect_step_percentage[model_name].append(
                main_incorrect_classmate_incorrect_percentage)

    # Plot in four subplots in the same image
    import matplotlib.pyplot as plt

    plot_configs = [
        (main_correct_classmate_correct_step_percentage, 'Main correct AND classmate correct / 1319 ↑', (0, 0)),
        (main_incorrect_classmate_correct_step_percentage, 'Main incorrect AND classmate correct / 1319 ↓', (0, 1)),
        (main_correct_classmate_incorrect_step_percentage, 'Main correct AND classmate incorrect / 1319 ↓', (1, 0)),
        (main_incorrect_classmate_incorrect_step_percentage, 'Main incorrect AND classmate incorrect / 1319 ↓', (1, 1))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for data, title, (i, j) in plot_configs:
        for model_name in model_fn_list:
            axes[i, j].plot(step_list, data[model_name], label=labels[model_name], marker=markers[model_name])
        axes[i, j].set_xticks(step_list)
        axes[i, j].set_xlabel('Training Steps')
        axes[i, j].set_ylabel('Percentage (%)')
        axes[i, j].set_title(title)
        axes[i, j].legend()
        axes[i, j].grid()

    plt.tight_layout()
    plt.savefig(f"{result_base_dir}/{dataset_name}_main_classmate_correctness_analysis.png")
    plt.close()

if __name__ == "__main__":
    # all_dataset_names = ["gsm8k", "hendrycks_math", "aimo-validation-aime"]
    all_dataset_names = ["hendrycks_math"]

    for dataset_name in all_dataset_names:
        # dataset_name = "gsm8k"
        # dataset_name = "hendrycks_math"
        # step_list = [75, 175, 275, 375, 475, 575, 675]
        max_step_num = 7  # start form 0
        step_size = 46

        max_y = None
        min_y = None

        # MATH
        step_list =[str(69 + i * step_size) for i in range(0, (max_step_num + 1))]

        result_base_dir = "outputs"

        # Qwen3-0.6B_gsm8k_m_cl_sep_norm_always_cl_reward_cl_partial_llama_1434816_episodes_seed_42

        # model_fn_list = {
        #     "baseline": f"{result_base_dir}/cot_utility_to_classmate/20260105-Qwen3-0.6B_gsm8k_minimal_answer_box_baseline_1434816_episodes_seed_42",
        #     "always_has_classmate_reward": f"{result_base_dir}/cot_utility_to_classmate/20260105-Qwen3-0.6B_gsm8k_minimal_answer_box_w_1_classmate_llama_1434816_episodes_seed_42",
        #     "no_classmate_reward_when_main_incorrect": f"{result_base_dir}/cot_utility_to_classmate/20260105-Qwen3-0.6B_gsm8k_no_classmate_main_incorrect_1_llama_1434816_episodes_seed_42",
        #     "separate_main_cl_norm_no_classmate_reward_when_main_incorrect": f"{result_base_dir}/cot_utility_to_classmate/20260112-Qwen3-0.6B_gsm8k_main_cl_separate_no_cl_main_incorrect_1_llama_1434816_episodes_seed_42",
        #     # "dgpo_no_cl_main_wrong_cl_partial": f"{result_base_dir}/cot_utility_to_classmate/20260113-Qwen3-0.6B_gsm8k_dgpo_no_cl_main_wrong_cl_partial_1_llama_1434816_episodes_seed_42",
        #     # "m_cl_sep_norm-no_cl_m_wrong-non_neg-cl_partial": f"{result_base_dir}/cot_utility_to_classmate/20260113-Qwen3-0.6B_gsm8k-m_cl_sep_norm-no_cl_m_wrong-non_neg-cl_partial_1_llama_1434816_episode_seed_42",
        #     "m_cl_sep_norm-always_cl_partial": f"{result_base_dir}/cot_utility_to_classmate/20260115-Qwen3-0.6B_gsm8k_m_cl_sep_norm_always_cl_partial_llama_1434816_episodes_seed_42",
        # }
        #
        # markers = {
        #     'baseline': 'o',
        #     'always_has_classmate_reward': 's',
        #     'no_classmate_reward_when_main_incorrect': '^',
        #     "separate_main_cl_norm_no_classmate_reward_when_main_incorrect": 'd',
        #     # "dgpo_no_cl_main_wrong_cl_partial": 'v',
        #     # "m_cl_sep_norm-no_cl_m_wrong-non_neg-cl_partial": 'P',
        #     "m_cl_sep_norm-always_cl_partial": 'P',
        # }
        #
        # labels = {
        #     'baseline': 'Baseline',
        #     'always_has_classmate_reward': 'Always has classmate + classmate all',
        #     'no_classmate_reward_when_main_incorrect': 'No classmate when main incorrect + classmate all',
        #     'separate_main_cl_norm_no_classmate_reward_when_main_incorrect': 'separate main cl norm + no classmate when main incorrect + classmate non-truncated',
        #     # 'dgpo_no_cl_main_wrong_cl_partial': 'DGPO + no classmate when main incorrect + classmate non-truncated',
        #     # "m_cl_sep_norm-no_cl_m_wrong-non_neg-cl_partial": 'separate main cl norm + no classmate when main incorrect + non-neg classmate non-truncated',
        #     "m_cl_sep_norm-always_cl_partial": 'separate main cl norm + always classmate + classmate non-truncated',
        # }

        # gsm8k
        # model_fn_list = {
        #     "baseline": f"{result_base_dir}/cot_utility_to_classmate/20260117-Qwen3-1.7B-Base_gsm8k_minimal_answer_box_prompt_baseline_1195680_episodes_seed_42",
        #     "separate_main_cl_norm_no_classmate_reward_when_main_incorrect": f"{result_base_dir}/cot_utility_to_classmate/20260118-Qwen3-1.7B-Base_gsm8k_m_cl_sep_no_cl_m_wrong_partial_llama_1195680_episodes_seed_42",
        # }

        # MATH
        # model_fn_list = {
        #     "baseline": f"{result_base_dir}/cot_utility_to_classmate/20260117-Qwen3-1.7B-Base_math_answer_box_baseline_719328_episodes_seed_42",
        #     "separate_main_cl_norm_no_classmate_reward_when_main_incorrect": f"{result_base_dir}/cot_utility_to_classmate/20260118-Qwen3-1.7B-Base_MATH_m_cl_sep_no_cl_m_wrong_partial_llama_719328_episodes_seed_42",
        #     # "separate_main_cl_norm_no_classmate_reward_when_main_incorrect_keep_0.5": f"{result_base_dir}/cot_utility_to_classmate/20260120-Qwen3-1.7Base_MATH_m_cl_sep_keep_0.5_no_cl_m_inc_partial_llama_719328_episodes_seed_42",
        # }
        model_fn_list = {
            "baseline": f"{result_base_dir}/cot_utility_to_classmate/20260126-Qwen3-1.7B-Base_MATH_700_heldout_baseline_652224_episodes_seed_42",
            "separate_main_cl_norm_no_cl_m_wrong_consistency_partial": f"{result_base_dir}/cot_utility_to_classmate/20260126-Qwen3-1.7Base_MATH700heldout_vanilla_no_cl_m_wrong_consis_partial_llama_652224_ep_s_42",
            "separate_main_cl_norm_always_cl_consis_partial": f"{result_base_dir}/cot_utility_to_classmate/20260129-Qwen3-1.7Base_MATH700heldout_vanilla_always_cl_consis_partial_llama_652224_ep_s_42"
            # "separate_main_cl_norm_no_classmate_reward_when_main_incorrect_keep_0.5": f"{result_base_dir}/cot_utility_to_classmate/20260120-Qwen3-1.7Base_MATH_m_cl_sep_keep_0.5_no_cl_m_inc_partial_llama_719328_episodes_seed_42",
        }

        # main_model_name_or_path_list = [
        #         "20260126-Qwen3-1.7B-Base_MATH_700_heldout_baseline_652224_episodes_seed_42",
        #         "20260126-Qwen3-1.7Base_MATH700heldout_vanilla_no_cl_m_wrong_consis_partial_llama_652224_ep_s_42"
        #     ]

        markers = {
            'baseline': 'o',
            # 'separate_main_cl_norm_no_classmate_reward_when_main_incorrect': 'd',
            # 'separate_main_cl_norm_no_classmate_reward_when_main_incorrect_keep_0.5': 'v',
            # 'separate_main_cl_norm_no_classmate_reward_when_main_incorrect_keep_0.5': 'v',
            'separate_main_cl_norm_always_cl_consis_partial': 'd',
            'separate_main_cl_norm_no_cl_m_wrong_consistency_partial': 'v',
        }

        labels = {
            'baseline': 'Baseline',
            # 'separate_main_cl_norm_no_classmate_reward_when_main_incorrect': 'separate main cl norm + no classmate when main incorrect + classmate non-truncated',
            # 'separate_main_cl_norm_no_classmate_reward_when_main_incorrect_keep_0.5': 'separate main cl norm + no classmate when main incorrect + keep 0.5 classmate non-truncated',
            'separate_main_cl_norm_no_cl_m_wrong_consistency_partial': 'separate main cl norm + no classmate when main incorrect + classmate non-truncated + consistency',
            "separate_main_cl_norm_always_cl_consis_partial": 'separate main cl norm + always classmate + classmate non-truncated + consistency',
        }

        dataset_obj = DATASET_NAME_TO_CLASS[dataset_name]()
        # main()
        main_correctness_and_consistency()

#bash scripts/visualize_main_classmate_correctness.sh