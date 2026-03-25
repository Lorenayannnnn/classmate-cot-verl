
import argparse

from verl.utils.reward_score.BaseVerifier import get_verifier
from monitor_judge_backends import estimate_openai_cost, run_different_monitor_and_judge


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run different monitor and judge models on saved predictions.")
    arg_parser.add_argument("--max_step_num", type=int)
    arg_parser.add_argument("--step_size", type=int)
    arg_parser.add_argument("--viz_offset", type=int, default=0)
    arg_parser.add_argument("--monitor_model_name", type=str)
    arg_parser.add_argument("--monitor_backend_type", type=str)
    arg_parser.add_argument("--judge_model_name", type=str)
    arg_parser.add_argument("--llm_judge_backend_type", type=str)

    arg_parser.add_argument("--result_dir", type=str)       # outputs_eval/confidence/Qwen3-0.6B/baseline_all_tokens/seed_0
    arg_parser.add_argument("--dataset_name", type=str)       # "sycophancy", "confidence"...

    arg_parser.add_argument("--max_new_tokens", type=int, default=3072, help="max_new_tokens of the policy model")
    arg_parser.add_argument("--do_base", type=bool)
    arg_parser.add_argument("--estimate_cost", action="store_true", help="Estimate OpenAI API cost using 10 samples then exit.")
    arg_parser.add_argument("--use_ICL_demo", action="store_true", help="Prepend ICL demonstrations from data/{behavior}/monitor_ICL_examples.json to each monitor prompt.")


    args = arg_parser.parse_args()

    # monitor_model_name = "Qwen/Qwen3-4B-Instruct-2507"
    # judge_model_name = "Qwen/Qwen3-4B-Instruct-2507"
    # monitor_source = "tinker"  # "openai" | "tinker" | "vllm"
    # judge_source = "tinker"  # "openai" | "tinker" | "vllm"

    # monitor_model_name = "gpt-4o-mini"
    # judge_model_name = "gpt-4.1-mini"
    # monitor_source = "openai"  # "openai" | "tinker" | "vllm"
    # judge_source = "openai"  # "openai" | "tinker" | "vllm"

    # monitor_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # judge_model_name = "gpt-4.1-mini"
    # monitor_source = "vllm"  # "openai" | "tinker" | "vllm"
    # judge_source = "openai"  # "openai" | "tinker" | "vllm"

    # monitor_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    # judge_model_name = "gpt-4.1-mini"
    # monitor_source = "vllm"  # "openai" | "tinker" | "vllm"
    # judge_source = "openai"  # "openai" | "tinker" | "vllm"

    monitor_model_name = args.monitor_model_name
    judge_model_name = args.judge_model_name
    monitor_source = args.monitor_backend_type
    judge_source = args.llm_judge_backend_type

    max_step_num = args.max_step_num
    step_size = args.step_size
    viz_offset = args.viz_offset

    output_dir = args.result_dir
    dataset_name_list = [args.dataset_name]

    skip_indices = []

    all_step_idx = ["base"] + [
        str(viz_offset + (i + 1) * step_size)
        for i in range(0, max_step_num)
        if str(viz_offset + (i + 1) * step_size) not in skip_indices
    ]

    if args.estimate_cost:
        verifier = get_verifier(data_source=dataset_name_list[0], max_new_tokens=args.max_new_tokens)
        estimate_openai_cost(
            dataset_name_list=dataset_name_list,
            step_idx_list=all_step_idx,
            output_dir=output_dir,
            monitor_model_name=monitor_model_name,
            judge_model_name=judge_model_name,
            verifier=verifier,
            main_model_name_or_path=output_dir,
        )
    else:
        run_different_monitor_and_judge(
            dataset_name_list,
            all_step_idx,
            output_dir,
            monitor_model_name=monitor_model_name,
            judge_model_name=judge_model_name,
            do_base=args.do_base,
            monitor_source=monitor_source,
            judge_source=judge_source,
            max_new_tokens=args.max_new_tokens,
            use_ICL_demo=args.use_ICL_demo,
        )

#export PYTHONPATH=:${PYTHONPATH} export VLLM_WORKER_MULTIPROC_METHOD=spawn
#python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_grpo_sycophancy_warmup_baseline_192000_episodes_seed_42" --do_base True
#python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_gdpo_192000_episodes_seed_42" --do_base False
#python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_cl_SELF_gdpo_192000_episodes_seed_42" --do_base False
#python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_gdpo_192000_episodes_seed_42_no_cl_IS" --do_base False
