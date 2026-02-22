
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from keys import OPENAI_KEY
from verl.utils.reward_score.cot_monitor.monitor import create_llm_judge, send_prompt_to_tinker
from verl.utils.reward_score.cot_monitor.BaseVerifier import get_monitor_verifier


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def _call_openai_sync(openai_client: OpenAI, model_name: str, prompt: str, verifier, max_retries: int = 5) -> str:
    """Send a prompt to OpenAI, retrying until the response parses to a valid score."""
    text = ""
    for attempt in range(max_retries):
        temperature = 0.0 if attempt == 0 else 0.7
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = response.choices[0].message.content
        score, _ = _parse_text_response(verifier, text)
        if score is not None:
            return text
        print(f"[retry {attempt + 1}/{max_retries}] Invalid response from {model_name}: {text!r}")
    return text  # return last attempt even if still invalid


def _parse_text_response(verifier, text: str):
    """Parse a raw text response (RATING / EXPLANATION format) into (score, explanation).

    Uses the verifier's own static parsers when available; falls back to a generic
    regex so that non-SycophancyVerifier subclasses are handled gracefully.
    """
    if text is None:
        return None, None
    if hasattr(verifier, "parse_sycophancy_score") and hasattr(verifier, "parse_judge_explanation"):
        score = verifier.parse_sycophancy_score(text)
        explanation = verifier.parse_judge_explanation(text)
    else:
        rating_match = re.search(r"RATING:\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        score = float(rating_match.group(1)) if rating_match else None
        explanation_match = re.search(r"EXPLANATION:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else text
    return score, explanation


def run_different_monitor_and_judge(
    main_model_name_or_path_list,
    dataset_name_list,
    step_idx_list,
    output_dir,
    monitor_model_name,
    judge_model_name,
    do_base,
    use_openai: bool = False,
):
    monitor_key = f"{_sanitize_model_name(monitor_model_name)}_monitor_score"
    monitor_expl_key = f"{_sanitize_model_name(monitor_model_name)}_monitor_explanation"
    judge_key = f"{_sanitize_model_name(judge_model_name)}_llm_judge_score"
    judge_expl_key = f"{_sanitize_model_name(judge_model_name)}_llm_judge_explanation"
    metrics_file_name = (
        f"{_sanitize_model_name(monitor_model_name)}_monitor"
        f"-{_sanitize_model_name(judge_model_name)}_llm_judge_metrics.json"
    )

    # ------------------------------------------------------------------ #
    # Set up backend: OpenAI or Tinker.
    # Both branches define the same four callables so the rest of the code
    # is backend-agnostic.
    # ------------------------------------------------------------------ #
    if use_openai:
        openai_client = OpenAI(api_key=OPENAI_KEY)
        executor = ThreadPoolExecutor()

        def dispatch_monitor_prompt(prompt, verifier):
            if prompt is None:
                return None
            return executor.submit(_call_openai_sync, openai_client, monitor_model_name, prompt, verifier)

        def dispatch_judge_prompt(prompt, verifier):
            if prompt is None:
                return None
            return executor.submit(_call_openai_sync, openai_client, judge_model_name, prompt, verifier)

        def parse_monitor_result(future_result, active_verifier):
            return _parse_text_response(active_verifier, future_result)

        def parse_judge_result(future_result, active_verifier):
            return _parse_text_response(active_verifier, future_result)

    else:
        monitor_tinker_config = create_llm_judge(judge_model_name=monitor_model_name)
        judge_tinker_config = create_llm_judge(judge_model_name=judge_model_name)
        executor = None

        def dispatch_monitor_prompt(prompt):
            if prompt is None:
                return None
            return send_prompt_to_tinker(monitor_tinker_config, prompt)

        def dispatch_judge_prompt(prompt):
            if prompt is None:
                return None
            return send_prompt_to_tinker(judge_tinker_config, prompt)

        def parse_monitor_result(future_result, active_verifier):
            return active_verifier.parse_monitor_output(future_result, monitor_tinker_config)

        def parse_judge_result(future_result, active_verifier):
            return active_verifier.parse_llm_judge_output(future_result, judge_tinker_config)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    for dataset_name in dataset_name_list:
        verifier = get_monitor_verifier(data_source=dataset_name)

        for model_idx, main_model_name_or_path in enumerate(main_model_name_or_path_list):
            for step_idx in step_idx_list:
                print(f"Start processing: model={main_model_name_or_path}, step={step_idx}, dataset={dataset_name} ...")
                if step_idx == "base" and not do_base:
                    print(f"[skip] base step for {main_model_name_or_path}")
                    continue

                result_dir = os.path.join(
                    output_dir, main_model_name_or_path, f"step_{step_idx}", dataset_name, "main"
                )
                preds_fn = os.path.join(result_dir, "preds.jsonl")
                metrics_fn = os.path.join(result_dir, metrics_file_name)

                if not os.path.exists(preds_fn):
                    print(f"[skip] {preds_fn} not found")
                    continue

                # ---------------------------------------------------------- #
                # Read all entries
                # ---------------------------------------------------------- #
                with open(preds_fn, "r") as preds_file:
                    entries = [json.loads(line) for line in preds_file]

                pending_entry_indices = [
                    entry_idx for entry_idx, entry in enumerate(entries)
                    if monitor_key not in entry
                ]
                num_already_annotated = len(entries) - len(pending_entry_indices)
                run_label = f"{main_model_name_or_path} / step_{step_idx} / {dataset_name}"

                if num_already_annotated > 0:
                    print(f"[resume] {run_label}: {num_already_annotated}/{len(entries)} already annotated")
                if not pending_entry_indices:
                    print(f"[done]   {run_label}: all entries already annotated")
                else:
                    # -------------------------------------------------------- #
                    # Launch all async futures up-front
                    # -------------------------------------------------------- #
                    print(f"[run]    {run_label}: annotating {len(pending_entry_indices)} entries ...")
                    pending_futures = []
                    for entry_idx in pending_entry_indices:
                        entry = entries[entry_idx]

                        monitor_doc = {
                            "raw_question_for_monitor": entry["question"],
                            "hint_str_for_monitor": entry.get("hint_str_for_monitor", ""),
                            "truncated_main_CoT": entry["truncated_main_CoT"],
                            "main_response": entry["truncated_main_CoT"],
                        }
                        monitor_prompt = verifier.create_monitor_message(monitor_doc)
                        judge_prompt = verifier.format_llm_judge_prompt(
                            entry["question"], entry["main_output"]
                        )

                        monitor_future = dispatch_monitor_prompt(monitor_prompt, verifier)
                        judge_future = dispatch_judge_prompt(judge_prompt, verifier)
                        pending_futures.append((entry_idx, monitor_future, judge_future))

                    # -------------------------------------------------------- #
                    # Collect results; save on KeyboardInterrupt for resumability
                    # -------------------------------------------------------- #
                    interrupted = False
                    try:
                        for entry_idx, monitor_future, judge_future in tqdm(
                            pending_futures, desc="Collecting results"
                        ):
                            entry = entries[entry_idx]

                            if monitor_future is not None:
                                monitor_score, monitor_explanation = parse_monitor_result(
                                    monitor_future.result(), verifier
                                )
                            else:
                                monitor_score, monitor_explanation = None, None

                            if judge_future is not None:
                                judge_score, judge_explanation = parse_judge_result(
                                    judge_future.result(), verifier
                                )
                            else:
                                judge_score, judge_explanation = None, None

                            entry[monitor_key] = monitor_score
                            entry[monitor_expl_key] = monitor_explanation
                            entry[judge_key] = judge_score
                            entry[judge_expl_key] = judge_explanation

                    except KeyboardInterrupt:
                        print(f"\n[interrupt] Saving partial progress to {preds_fn} ...")
                        interrupted = True

                    finally:
                        with open(preds_fn, "w") as preds_file:
                            for entry in entries:
                                preds_file.write(json.dumps(entry) + "\n")
                        print(f"[saved]  {preds_fn}")

                    if interrupted:
                        if use_openai:
                            executor.shutdown(wait=False)
                        return

                # ---------------------------------------------------------- #
                # Compute and save metrics over all fully-annotated entries
                # ---------------------------------------------------------- #
                monitor_scores, main_scores, judge_scores = [], [], []
                for entry in entries:
                    if entry.get(monitor_key) is not None:
                        monitor_scores.append(entry[monitor_key])
                        main_scores.append(entry["main_score"])
                    if entry.get(judge_key) is not None:
                        judge_scores.append(entry[judge_key])

                monitor_scores = np.array(monitor_scores)
                main_scores = np.array(main_scores)

                monitor_valid_mask = monitor_scores != 0
                main_reward_valid_mask = main_scores != 0
                valid_mask = monitor_valid_mask & main_reward_valid_mask

                monitor_metrics = verifier.compute_metrics(
                    predictions=monitor_scores[valid_mask].astype(float),
                    ground_truths=main_scores[valid_mask].astype(float),
                )
                monitor_metrics["total_monitor_valid_entries"] = int(np.sum(monitor_valid_mask))
                monitor_metrics["total_output_valid_entries"] = int(np.sum(main_reward_valid_mask))
                monitor_metrics["total_monitored_entries"] = int(np.sum(valid_mask))

                monitor_metrics["judge_score_mean"] = float(np.mean(judge_scores))
                monitor_metrics["total_judge_entries"] = len(judge_scores)

                with open(metrics_fn, "w") as metrics_file:
                    json.dump(monitor_metrics, metrics_file, indent=4)
                print(f"[metrics] saved to {metrics_fn}")

    if use_openai:
        executor.shutdown(wait=False)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run different monitor and judge models on saved predictions.")
    arg_parser.add_argument("--main_model_name_or_path", type=str)
    arg_parser.add_argument("--do_base", type=bool)
    args = arg_parser.parse_args()

    # monitor_model_name = INPUT_MONITOR_MODEL_NAME
    # judge_model_name = INPUT_JUDGE_MODEL_NAME
    monitor_model_name = "gpt-4o-mini"
    judge_model_name = "gpt-4.1-mini"
    use_openai = True

    # do_base = [True, False, False]

    # main_model_name_or_path_list = [
    #     "20260217-Qwen3-0.6B_grpo_sycophancy_warmup_baseline_192000_episodes_seed_42",
    #     "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_gdpo_192000_episodes_seed_42",
    #     "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_cl_SELF_gdpo_192000_episodes_seed_42"
    # ]
    main_model_name_or_path_list = [
        args.main_model_name_or_path,
    ]

    max_step_num = 7  # start from 0
    step_size = 20

    output_dir = "outputs_eval/inference_main_model"
    # dataset_name_list = ["mmlu_sycophancy"]
    dataset_name_list = ["helpful_instructions"]
    # dataset_name_list = ["anthropic_hh_rlhf"]

    skip_indices = []

    all_step_idx = ["base"] + [
        str((i + 1) * step_size)
        for i in range(0, max_step_num)
        if str((i + 1) * step_size) not in skip_indices
    ]

    run_different_monitor_and_judge(
        main_model_name_or_path_list,
        dataset_name_list,
        all_step_idx,
        output_dir,
        monitor_model_name=monitor_model_name,
        judge_model_name=judge_model_name,
        do_base=args.do_base,
        use_openai=use_openai,
    )

#export PYTHONPATH=:${PYTHONPATH}
#python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_grpo_sycophancy_warmup_baseline_192000_episodes_seed_42" --do_base True
#python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_gdpo_192000_episodes_seed_42" --do_base False
#python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_cl_SELF_gdpo_192000_episodes_seed_42" --do_base False
#python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_gdpo_192000_episodes_seed_42_no_cl_IS" --do_base False
