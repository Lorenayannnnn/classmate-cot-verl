
import argparse

from verl.utils.reward_score.BaseVerifier import get_verifier
from monitor_judge_backends import estimate_openai_cost, run_different_monitor_and_judge


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run different monitor and judge models on saved predictions.")
    arg_parser.add_argument("--main_model_name_or_path", type=str)
    arg_parser.add_argument("--max_new_tokens", type=int)
    arg_parser.add_argument("--do_base", action="store_true", help="Whether to run the base model without any training as well.")
    arg_parser.add_argument("--estimate_cost", action="store_true", help="Estimate OpenAI API cost using 10 samples then exit.")
    args = arg_parser.parse_args()

    repo_prefix = "LorenaYannnnn"

    # monitor_model_name = "Qwen/Qwen3-4B-Instruct-2507"
    # judge_model_name = "Qwen/Qwen3-4B-Instruct-2507"
    # monitor_source = "tinker"  # "openai" | "tinker" | "vllm"
    # judge_source = "tinker"  # "openai" | "tinker" | "vllm"

    # monitor_model_name = "gpt-4o-mini"
    # judge_model_name = "gpt-4.1-mini"
    # monitor_source = "openai"  # "openai" | "tinker" | "vllm"
    # judge_source = "openai"  # "openai" | "tinker" | "vllm"

    monitor_model_name = "gpt-4o"
    judge_model_name = "gpt-4.1"
    monitor_source = "openai"  # "openai" | "tinker" | "vllm"
    judge_source = "openai"  # "openai" | "tinker" | "vllm"

    # monitor_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # judge_model_name = "gpt-4.1-mini"
    # monitor_source = "vllm"  # "openai" | "tinker" | "vllm"
    # judge_source = "openai"  # "openai" | "tinker" | "vllm"

    # monitor_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    # judge_model_name = "gpt-4.1-mini"
    # monitor_source = "vllm"  # "openai" | "tinker" | "vllm"
    # judge_source = "openai"  # "openai" | "tinker" | "vllm"

    # do_base = [True, False, False]

    # main_model_name_or_path_list = [
    #     "20260217-Qwen3-0.6B_grpo_sycophancy_warmup_baseline_192000_episodes_seed_42",
    #     "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_gdpo_192000_episodes_seed_42",
    #     "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_cl_SELF_gdpo_192000_episodes_seed_42"
    # ]
    main_model_name_or_path_list = [
        args.main_model_name_or_path,
    ]

    max_step_num = 6  # start from 0
    step_size = 40

    dataset_name = "longer_response"
    output_dir = f"outputs_eval/inference_main_model_20260309_longer_response/{dataset_name}"
    dataset_name_list = [dataset_name]

    skip_indices = ['150']

    # all_step_idx = ["base"] + [
    #     str((i + 1) * step_size)
    #     for i in range(0, max_step_num)
    #     if str((i + 1) * step_size) not in skip_indices
    # ]

    all_step_idx = ["base"] + [str(30 + i * step_size) for i in range(0, (max_step_num + 1)) if
                               str(30 + i * step_size) not in skip_indices]

    if args.estimate_cost:
        verifier = get_verifier(data_source=dataset_name_list[0], max_new_tokens=args.max_new_tokens)
        estimate_openai_cost(
            dataset_name_list=dataset_name_list,
            step_idx_list=all_step_idx,
            output_dir=output_dir,
            monitor_model_name=monitor_model_name,
            judge_model_name=judge_model_name,
            verifier=verifier,
            main_model_name_or_path=args.main_model_name_or_path,
        )
    else:
        run_different_monitor_and_judge(
            main_model_name_or_path_list,
            dataset_name_list,
            all_step_idx,
            output_dir,
            monitor_model_name=monitor_model_name,
            judge_model_name=judge_model_name,
            do_base=args.do_base,
            monitor_source=monitor_source,
            judge_source=judge_source,
            repo_prefix=repo_prefix,
            max_new_tokens=args.max_new_tokens,
        )

#export PYTHONPATH=:${PYTHONPATH} export VLLM_WORKER_MULTIPROC_METHOD=spawn
#python my_eval/src/analysis_module/different_monitor_longer_responses_across_models.py --main_model_name_or_path "20260308-length_only-Qwen3-0.6B_grpo_baseline_192000_episodes_seed_42" --max_new_tokens 3072 --do_base
#python my_eval/src/analysis_module/different_monitor_longer_responses_across_models.py --main_model_name_or_path "20260308-length_only-Qwen3-0.6B_OURS_cl_self_partial_192000_episodes_seed_42" --max_new_tokens 3072
