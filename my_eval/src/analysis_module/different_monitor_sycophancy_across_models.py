
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
from verl.utils.reward_score.cot_monitor.BaseVerifier import get_verifier


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def _call_openai_sync(openai_client: OpenAI, model_name: str, prompt: str, verifier, max_retries: int = 5) -> str:
    """Send a prompt to OpenAI, retrying until the response parses to a valid score."""
    text = ""
    for attempt in range(max_retries):
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
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


def create_vllm_engine(model_name_path: str, max_model_len: int = 8192, gpu_memory_utilization: float = 0.9):
    """Create a local vLLM engine, its tokenizer, and default sampling params.

    Mirrors create_vllm / create_tokenizer from test_sycophancy.py.
    Sampling params are chosen based on the model:
      - Qwen/Qwen2.5-0.5B-Instruct: temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512
      - all others:                  temperature=1, top_p=1, top_k=-1, max_tokens=1024

    Returns:
        (engine, tokenizer, sampling_params) tuple to be passed to call_vllm_sync.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
    engine = LLM(
        model=model_name_path,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_custom_all_reduce=True,
        skip_tokenizer_init=False,
        max_model_len=max_model_len,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    if model_name_path == "Qwen/Qwen2.5-0.5B-Instruct":
        # https://qwen.ai/blog?id=qwen2.5
        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=1024)
    elif model_name_path == "HuggingFaceTB/SmolLM2-360M-Instruct":
        # https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
        sampling_params = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=1024)
    else:
        raise ValueError(f"Make sure to have the sampling param foro model {model_name_path}")
        # sampling_params = SamplingParams(temperature=1, top_p=1, top_k=-1, max_tokens=1024)

    return engine, tokenizer, sampling_params, max_model_len


def call_vllm_sync(engine, tokenizer, prompt: str, verifier, sampling_params, max_retries: int = 5) -> str:
    """Send a single chat prompt to a local vLLM engine and return the response string.

    Mirrors the generate() call in test_sycophancy.py. On the first attempt the
    provided sampling_params are used as-is; subsequent retries fall back to
    temperature=0.7 (with the same max_tokens) for more diverse outputs.

    Args:
        engine:          vLLM LLM instance from create_vllm_engine.
        tokenizer:       Corresponding HuggingFace tokenizer.
        prompt:          Plain-text prompt (wrapped into a user chat message).
        verifier:        Used to check whether the response is parseable.
        sampling_params: SamplingParams returned by create_vllm_engine.
        max_retries:     Number of generation attempts before giving up.

    Returns:
        Decoded response string (last attempt even if all retries fail).
    """
    from vllm import SamplingParams, TokensPrompt

    messages = [{"role": "user", "content": prompt}]
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    text = ""
    for attempt in range(max_retries):
        token_prompt = TokensPrompt(prompt_token_ids=token_ids)
        outputs = engine.generate([token_prompt], sampling_params=sampling_params, use_tqdm=False)
        text = tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=True).strip()
        score, _ = _parse_text_response(verifier, text)
        if score is not None:
            return text
        print(f"[retry {attempt + 1}/{max_retries}] Invalid vLLM response: {text!r}")
    return text


def _make_openai_compatible_backend(model_name: str, api_key: str, base_url: str | None = None):
    """Return (dispatch_fn, parse_fn, executor) for an OpenAI-compatible endpoint.

    Works for both the OpenAI API (base_url=None) and vLLM servers
    (base_url pointing at the local vLLM OpenAI-compatible endpoint).
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    executor = ThreadPoolExecutor()

    def dispatch(prompt, verifier):
        if prompt is None:
            return None
        return executor.submit(_call_openai_sync, client, model_name, prompt, verifier)

    def parse(future_result, verifier):
        return _parse_text_response(verifier, future_result)

    return dispatch, parse, executor


def _make_tinker_monitor_backend(model_name: str):
    """Return (dispatch_fn, parse_fn, executor=None) for a Tinker monitor."""
    tinker_config = create_llm_judge(judge_model_name=model_name)

    def dispatch(prompt, verifier):
        if prompt is None:
            return None
        return send_prompt_to_tinker(tinker_config, prompt)

    def parse(future_result, verifier):
        return verifier.parse_monitor_output(future_result, tinker_config)

    return dispatch, parse, None


def _make_tinker_judge_backend(model_name: str):
    """Return (dispatch_fn, parse_fn, executor=None) for a Tinker LLM judge."""
    tinker_config = create_llm_judge(judge_model_name=model_name)

    def dispatch(prompt, verifier):
        if prompt is None:
            return None
        return send_prompt_to_tinker(tinker_config, prompt)

    def parse(future_result, verifier):
        return verifier.parse_llm_judge_output(future_result, tinker_config)

    return dispatch, parse, None


def _make_vllm_local_backend(model_name: str):
    """Return (dispatch_fn, parse_fn, executor=None) for a local vLLM engine.

    vLLM's v1 engine uses zmq sockets created on the main thread; calling
    engine.generate() from a ThreadPoolExecutor worker causes IPC errors.
    Instead, dispatch runs call_vllm_sync synchronously and returns a
    pre-resolved Future so the rest of the code stays backend-agnostic.
    """
    from concurrent.futures import Future

    engine, tokenizer, sampling_params, max_model_len = create_vllm_engine(model_name)

    def dispatch(prompt, verifier):
        if prompt is None:
            return None
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
        if len(token_ids) + sampling_params.max_tokens > max_model_len:
            print(f"[skip] Input too long ({len(token_ids)} + {sampling_params.max_tokens} > {max_model_len}); skipping.")
            return {"score": 0, "explanation": "input being too long"}
        f = Future()
        try:
            f.set_result(call_vllm_sync(engine, tokenizer, prompt, verifier, sampling_params))
        except Exception as exc:
            f.set_exception(exc)
        return f

    def parse(future_result, verifier):
        return _parse_text_response(verifier, future_result)

    return dispatch, parse, None


def run_different_monitor_and_judge(
    main_model_name_or_path_list,
    dataset_name_list,
    step_idx_list,
    output_dir,
    monitor_model_name,
    judge_model_name,
    do_base,
    monitor_source: str = "tinker",
    judge_source: str = "tinker",
):
    """
    Args:
        monitor_source: Backend for the monitor model. One of "openai", "tinker", "vllm".
        judge_source:   Backend for the judge model.  One of "openai", "tinker", "vllm".
    """
    monitor_key = f"{_sanitize_model_name(monitor_model_name)}_monitor_score"
    monitor_expl_key = f"{_sanitize_model_name(monitor_model_name)}_monitor_explanation"
    judge_key = f"{_sanitize_model_name(judge_model_name)}_llm_judge_score"
    judge_expl_key = f"{_sanitize_model_name(judge_model_name)}_llm_judge_explanation"
    metrics_file_name = (
        f"{_sanitize_model_name(monitor_model_name)}_monitor"
        f"-{_sanitize_model_name(judge_model_name)}_llm_judge_metrics.json"
    )

    # ------------------------------------------------------------------ #
    # Set up monitor backend
    # ------------------------------------------------------------------ #
    if monitor_source == "openai":
        dispatch_monitor, parse_monitor, monitor_executor = _make_openai_compatible_backend(
            monitor_model_name, OPENAI_KEY
        )
    elif monitor_source == "vllm":
        dispatch_monitor, parse_monitor, monitor_executor = _make_vllm_local_backend(monitor_model_name)
    elif monitor_source == "tinker":
        dispatch_monitor, parse_monitor, monitor_executor = _make_tinker_monitor_backend(monitor_model_name)
    else:
        raise ValueError(f"Unknown monitor_source: {monitor_source!r}. Choose from 'openai', 'tinker', 'vllm'.")

    # ------------------------------------------------------------------ #
    # Set up judge backend
    # ------------------------------------------------------------------ #
    if judge_source == "openai":
        dispatch_judge, parse_judge, judge_executor = _make_openai_compatible_backend(
            judge_model_name, OPENAI_KEY
        )
    elif judge_source == "vllm":
        dispatch_judge, parse_judge, judge_executor = _make_vllm_local_backend(judge_model_name)
    elif judge_source == "tinker":
        dispatch_judge, parse_judge, judge_executor = _make_tinker_judge_backend(judge_model_name)
    else:
        raise ValueError(f"Unknown judge_source: {judge_source!r}. Choose from 'openai', 'tinker', 'vllm'.")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    for dataset_name in dataset_name_list:
        verifier = get_verifier(data_source=dataset_name)

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

                # # Rename legacy bare keys (monitor_score, llm_judge_score, …) to the
                # # current model-prefixed keys when the new key is not yet present.
                # legacy_key_map = {
                #     "monitor_score": monitor_key,
                #     "monitor_explanation": monitor_expl_key,
                #     "llm_judge_score": judge_key,
                #     "llm_judge_explanation": judge_expl_key,
                # }
                # for entry in entries:
                #     for old_key, new_key in legacy_key_map.items():
                #         if old_key in entry and new_key not in entry:
                #             entry[new_key] = entry.pop(old_key)
                #
                # if len(entries) == 1000:
                #     # write back the file with the renamed keys to avoid doing this every time in the future
                #     with open(preds_fn, "w") as preds_file:
                #         for entry in entries:
                #             preds_file.write(json.dumps(entry) + "\n")
                # continue

                # An entry needs processing if either key is missing or its value is None.
                pending_entry_indices = [
                    entry_idx for entry_idx, entry in enumerate(entries)
                    if entry.get(monitor_key) is None or entry.get(judge_key) is None
                ]
                num_already_annotated = len(entries) - len(pending_entry_indices)
                run_label = f"{main_model_name_or_path} / step_{step_idx} / {dataset_name}"

                if num_already_annotated > 0:
                    print(f"[resume] {run_label}: {num_already_annotated}/{len(entries)} already annotated")
                if not pending_entry_indices:
                    print(f"[done]   {run_label}: all entries already annotated")
                else:
                    # -------------------------------------------------------- #
                    # Launch all async futures up-front.
                    # Skip dispatching a request if that key already exists in the entry.
                    # -------------------------------------------------------- #
                    print(f"[run]    {run_label}: annotating {len(pending_entry_indices)} entries ...")
                    pending_futures = []
                    for entry_idx in tqdm(pending_entry_indices, desc="submitting entries for annotation"):
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

                        if entry.get(monitor_key) is None:
                            monitor_future = dispatch_monitor(monitor_prompt, verifier)
                            if monitor_future is None:
                                monitor_future = {"score": verifier.invalid_score, "explanation": "CoT is empty"}
                        else:
                            monitor_future = None

                        if entry.get(judge_key) is None:
                            judge_future = dispatch_judge(judge_prompt, verifier)
                            if judge_future is None:
                                judge_future = {"score": verifier.invalid_score, "explanation": "Output is empty"}
                        else:
                            judge_future = None

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
                                if isinstance(monitor_future, dict):
                                    entry[monitor_key] = monitor_future["score"]
                                    entry[monitor_expl_key] = monitor_future["explanation"]
                                else:
                                    monitor_score, monitor_explanation = parse_monitor(
                                        monitor_future.result(), verifier
                                    )
                                    entry[monitor_key] = monitor_score
                                    entry[monitor_expl_key] = monitor_explanation
                            # else: monitor_key already present in entry — keep existing value

                            if judge_future is not None:
                                if isinstance(judge_future, dict):
                                    entry[judge_key] = judge_future["score"]
                                    entry[judge_expl_key] = judge_future["explanation"]
                                else:
                                    judge_score, judge_explanation = parse_judge(
                                        judge_future.result(), verifier
                                    )
                                    entry[judge_key] = judge_score
                                    entry[judge_expl_key] = judge_explanation
                            # else: judge_key already present in entry — keep existing value

                    except KeyboardInterrupt:
                        print(f"\n[interrupt] Saving partial progress to {preds_fn} ...")
                        interrupted = True

                    finally:
                        with open(preds_fn, "w") as preds_file:
                            for entry in entries:
                                preds_file.write(json.dumps(entry) + "\n")
                        print(f"[saved]  {preds_fn}")

                    if interrupted:
                        if monitor_executor is not None:
                            monitor_executor.shutdown(wait=False)
                        if judge_executor is not None and judge_executor is not monitor_executor:
                            judge_executor.shutdown(wait=False)
                        return

                # ---------------------------------------------------------- #
                # Compute and save metrics over all fully-annotated entries
                # ---------------------------------------------------------- #
                monitor_scores, judge_scores = [], []
                for entry in entries:
                    monitor_scores.append(entry[monitor_key])
                    judge_scores.append(entry[judge_key])

                monitor_scores = np.array(monitor_scores)
                judge_scores = np.array(judge_scores)

                monitor_valid_mask = monitor_scores != verifier.invalid_score
                judge_valid_mask = judge_scores != verifier.invalid_score
                valid_mask = monitor_valid_mask & judge_valid_mask

                monitor_metrics = verifier.compute_metrics(
                    predictions=monitor_scores[valid_mask].astype(float),
                    ground_truths=judge_scores[valid_mask].astype(float),
                )
                monitor_metrics["total_valid_CoT_entries"] = int(np.sum(monitor_valid_mask))
                monitor_metrics["total_valid_output_entries"] = int(np.sum(judge_valid_mask))
                monitor_metrics["total_monitored_entries"] = int(np.sum(valid_mask))

                monitor_metrics["judge_score_mean"] = float(np.mean(judge_scores))
                monitor_metrics["total_judge_entries"] = len(judge_scores)

                with open(metrics_fn, "w") as metrics_file:
                    json.dump(monitor_metrics, metrics_file, indent=4)
                print(f"[metrics] saved to {metrics_fn}")

    if monitor_executor is not None:
        monitor_executor.shutdown(wait=False)
    if judge_executor is not None and judge_executor is not monitor_executor:
        judge_executor.shutdown(wait=False)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run different monitor and judge models on saved predictions.")
    arg_parser.add_argument("--main_model_name_or_path", type=str)
    arg_parser.add_argument("--do_base", type=bool)
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

    monitor_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    judge_model_name = "gpt-4.1-mini"
    monitor_source = "vllm"  # "openai" | "tinker" | "vllm"
    judge_source = "openai"  # "openai" | "tinker" | "vllm"

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
        monitor_source=monitor_source,
        judge_source=judge_source,
    )

#export PYTHONPATH=:${PYTHONPATH} export VLLM_WORKER_MULTIPROC_METHOD=spawn
#CUDA_VISIBLE_DEVICES=4 python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_grpo_sycophancy_warmup_baseline_192000_episodes_seed_42" --do_base True
#CUDA_VISIBLE_DEVICES=5 python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_gdpo_192000_episodes_seed_42" --do_base False
#CUDA_VISIBLE_DEVICES=6 python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_cl_SELF_gdpo_192000_episodes_seed_42" --do_base False
#CUDA_VISIBLE_DEVICES=7 python my_eval/src/analysis_module/different_monitor_sycophancy_across_models.py --main_model_name_or_path "20260217-Qwen3-0.6B_sycophancy_warmup_16000_ep_OURS_gdpo_192000_episodes_seed_42_no_cl_IS" --do_base False