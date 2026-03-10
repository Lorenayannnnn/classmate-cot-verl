"""
Shared backend helpers for different_monitor_*_across_models.py scripts.

Provides:
  - Backend factory functions (_make_openai_compatible_monitor_backend, etc.)
  - run_different_monitor_and_judge: main annotation + metrics loop
  - estimate_openai_cost: extrapolate OpenAI API cost from n sample calls
  - OPENAI_PRICING: prices per 1M tokens by model
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from keys import OPENAI_KEY
from verl.utils.reward_score.monitor import create_llm_judge, send_prompt_to_tinker
from verl.utils.reward_score.BaseVerifier import get_verifier

# Prices in USD per 1M tokens: (input, output)
OPENAI_PRICING = {
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o":       (2.50, 10.00),
    "gpt-4.1":      (2.00,  8.00),
}


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def _call_openai_sync(openai_client: OpenAI, model_name: str, prompt: str, verifier, validate_fn, max_retries: int = 5) -> str:
    """Send a prompt to OpenAI, retrying until the response parses to a valid score."""
    text = ""
    for attempt in range(max_retries):
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        score, _ = validate_fn(text, verifier)
        if score is not None:
            return text
        print(f"[retry {attempt + 1}/{max_retries}] Invalid response from {model_name}: {text!r}")
    return text  # return last attempt even if still invalid


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


def call_vllm_sync(engine, tokenizer, prompt: str, verifier, validate_fn, sampling_params, max_retries: int = 5) -> str:
    """Send a single chat prompt to a local vLLM engine and return the response string.

    Mirrors the generate() call in test_sycophancy.py. On the first attempt the
    provided sampling_params are used as-is; subsequent retries fall back to
    temperature=0.7 (with the same max_tokens) for more diverse outputs.

    Args:
        engine:          vLLM LLM instance from create_vllm_engine.
        tokenizer:       Corresponding HuggingFace tokenizer.
        prompt:          Plain-text prompt (wrapped into a user chat message).
        verifier:        Used to check whether the response is parseable.
        validate_fn:     Function (text, verifier) -> (score, explanation) for retry validation.
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
        score, _ = validate_fn(text, verifier)
        if score is not None:
            return text
        print(f"[retry {attempt + 1}/{max_retries}] Invalid vLLM response: {text!r}")
    return text


def _make_openai_compatible_monitor_backend(model_name: str, api_key: str, base_url: str | None = None):
    """Return (dispatch_fn, parse_fn, executor) for an OpenAI-compatible monitor endpoint."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    executor = ThreadPoolExecutor()

    def _validate(text, verifier):
        return verifier.parse_monitor_output(text, None)

    def dispatch(prompt, verifier):
        if prompt is None:
            return None
        return executor.submit(_call_openai_sync, client, model_name, prompt, verifier, _validate)

    def parse(future_result, verifier):
        return verifier.parse_monitor_output(future_result, None)

    return dispatch, parse, executor


def _make_openai_compatible_judge_backend(model_name: str, api_key: str, base_url: str | None = None):
    """Return (dispatch_fn, parse_fn, executor) for an OpenAI-compatible judge endpoint."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    executor = ThreadPoolExecutor()

    def _validate(text, verifier):
        return verifier.parse_llm_judge_output(text, None)

    def dispatch(prompt, verifier):
        if prompt is None:
            return None
        return executor.submit(_call_openai_sync, client, model_name, prompt, verifier, _validate)

    def parse(future_result, verifier, continuation=None, continuation_token_ids=None, gt=None):
        return verifier.parse_llm_judge_output(future_result, None, continuation=continuation, continuation_token_ids=continuation_token_ids, gt=gt)

    return dispatch, parse, executor


def _make_tinker_monitor_backend(model_name: str):
    """Return (dispatch_fn, parse_fn, executor=None) for a Tinker monitor."""
    tinker_config = create_llm_judge(judge_model_name=model_name)

    def dispatch(prompt, verifier):
        if prompt is None:
            return None
        return send_prompt_to_tinker(tinker_config, prompt)

    def parse(future_result, verifier, continuation=None, gt=None):
        return verifier.parse_monitor_output(future_result, tinker_config, continuation=continuation, gt=gt)

    return dispatch, parse, None


def _make_tinker_judge_backend(model_name: str):
    """Return (dispatch_fn, parse_fn, executor=None) for a Tinker LLM judge."""
    tinker_config = create_llm_judge(judge_model_name=model_name)

    def dispatch(prompt, verifier):
        if prompt is None:
            return None
        return send_prompt_to_tinker(tinker_config, prompt)

    def parse(future_result, verifier, continuation=None, continuation_token_ids=None, gt=None):
        return verifier.parse_llm_judge_output(future_result, tinker_config, continuation=continuation, continuation_token_ids=continuation_token_ids, gt=gt)

    return dispatch, parse, None


def _make_vllm_local_monitor_backend(model_name: str):
    """Return (dispatch_fn, parse_fn, executor=None) for a local vLLM monitor engine."""
    from concurrent.futures import Future

    engine, tokenizer, sampling_params, max_model_len = create_vllm_engine(model_name)

    def _validate(text, verifier):
        return verifier.parse_monitor_output(text, None)

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
            f.set_result(call_vllm_sync(engine, tokenizer, prompt, verifier, _validate, sampling_params))
        except Exception as exc:
            f.set_exception(exc)
        return f

    def parse(future_result, verifier):
        return verifier.parse_monitor_output(future_result, None)

    return dispatch, parse, None


def _make_vllm_local_judge_backend(model_name: str):
    """Return (dispatch_fn, parse_fn, executor=None) for a local vLLM judge engine."""
    from concurrent.futures import Future

    engine, tokenizer, sampling_params, max_model_len = create_vllm_engine(model_name)

    def _validate(text, verifier):
        return verifier.parse_llm_judge_output(text, None)

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
            f.set_result(call_vllm_sync(engine, tokenizer, prompt, verifier, _validate, sampling_params))
        except Exception as exc:
            f.set_exception(exc)
        return f

    def parse(future_result, verifier, continuation=None, continuation_token_ids=None, gt=None):
        return verifier.parse_llm_judge_output(future_result, None, continuation=continuation, continuation_token_ids=continuation_token_ids, gt=gt)

    return dispatch, parse, None


def estimate_openai_cost(
    dataset_name_list,
    step_idx_list,
    output_dir,
    monitor_model_name,
    judge_model_name,
    verifier,
    main_model_name_or_path,
    n_samples: int = 10,
):
    """Call OpenAI with n_samples prompts, measure actual token usage, and extrapolate total cost."""
    client = OpenAI(api_key=OPENAI_KEY)

    monitor_input_tokens = 0
    monitor_output_tokens = 0
    judge_input_tokens = 0
    judge_output_tokens = 0
    sampled = 0

    for dataset_name in dataset_name_list:
        for step_idx in step_idx_list:
            preds_fn = os.path.join(output_dir, main_model_name_or_path, f"step_{step_idx}", "main", "preds.jsonl")
            if not os.path.exists(preds_fn):
                continue
            with open(preds_fn, "r") as f:
                entries = [json.loads(line) for line in f]
            for entry in entries[:n_samples]:
                monitor_doc = {
                    "raw_question_for_monitor": entry["question"],
                    "hint_str_for_monitor": entry.get("hint_str_for_monitor", ""),
                    "truncated_main_CoT": entry["truncated_main_CoT"],
                    "main_response": entry["truncated_main_CoT"],
                }
                monitor_prompt = verifier.create_monitor_message(monitor_doc)
                judge_prompt = verifier.format_llm_judge_prompt(entry["question"], entry["main_output"])

                if monitor_prompt is not None:
                    resp = client.chat.completions.create(
                        model=monitor_model_name,
                        messages=[{"role": "user", "content": monitor_prompt}],
                    )
                    monitor_input_tokens += resp.usage.prompt_tokens
                    monitor_output_tokens += resp.usage.completion_tokens

                if judge_prompt is not None:
                    resp = client.chat.completions.create(
                        model=judge_model_name,
                        messages=[{"role": "user", "content": judge_prompt}],
                    )
                    judge_input_tokens += resp.usage.prompt_tokens
                    judge_output_tokens += resp.usage.completion_tokens

                sampled += 1
                if sampled >= n_samples:
                    break
            if sampled >= n_samples:
                break
        if sampled >= n_samples:
            break

    if sampled == 0:
        print("[estimate] No entries found — check output_dir and step_idx_list.")
        return

    # Count total entries that need annotation across all files
    total_entries = 0
    for dataset_name in dataset_name_list:
        for step_idx in step_idx_list:
            preds_fn = os.path.join(output_dir, main_model_name_or_path, f"step_{step_idx}", "main", "preds.jsonl")
            if os.path.exists(preds_fn):
                with open(preds_fn, "r") as f:
                    total_entries += sum(1 for _ in f)

    scale = total_entries / sampled

    total_monitor_input  = monitor_input_tokens  * scale
    total_monitor_output = monitor_output_tokens * scale
    total_judge_input    = judge_input_tokens    * scale
    total_judge_output   = judge_output_tokens   * scale

    def _cost(model, inp, out):
        price_in, price_out = OPENAI_PRICING.get(model, (0, 0))
        return inp / 1e6 * price_in + out / 1e6 * price_out

    monitor_cost = _cost(monitor_model_name, total_monitor_input, total_monitor_output)
    judge_cost   = _cost(judge_model_name,   total_judge_input,   total_judge_output)

    print(f"\n{'='*60}")
    print(f"Cost estimate (based on {sampled} samples → {total_entries} total entries)")
    print(f"{'='*60}")
    print(f"Monitor ({monitor_model_name}):")
    print(f"  input  tokens: {total_monitor_input:,.0f}  →  ${monitor_cost - _cost(monitor_model_name, 0, total_monitor_output):.4f}")
    print(f"  output tokens: {total_monitor_output:,.0f}  →  ${_cost(monitor_model_name, 0, total_monitor_output):.4f}")
    print(f"  subtotal: ${monitor_cost:.4f}")
    print(f"Judge ({judge_model_name}):")
    print(f"  input  tokens: {total_judge_input:,.0f}  →  ${judge_cost - _cost(judge_model_name, 0, total_judge_output):.4f}")
    print(f"  output tokens: {total_judge_output:,.0f}  →  ${_cost(judge_model_name, 0, total_judge_output):.4f}")
    print(f"  subtotal: ${judge_cost:.4f}")
    print(f"{'='*60}")
    print(f"TOTAL ESTIMATED COST: ${monitor_cost + judge_cost:.4f}")
    print(f"{'='*60}\n")
    if monitor_model_name not in OPENAI_PRICING or judge_model_name not in OPENAI_PRICING:
        print(f"[warning] Pricing not found for one or more models — add to OPENAI_PRICING dict.")


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
    repo_prefix: str | None = None,
    max_new_tokens: int | None = None,
):
    """
    Args:
        monitor_source: Backend for the monitor model. One of "openai", "tinker", "vllm".
        judge_source:   Backend for the judge model.  One of "openai", "tinker", "vllm".
        repo_prefix:    HuggingFace repo prefix (e.g. "LorenaYannnnn") used to load the
                        main model tokenizer for fallback main_output_ids encoding.
                        Pass None (default) to skip tokenizer loading.
        max_new_tokens: Passed to get_verifier() for verifiers that need it (e.g. length_only).
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
        dispatch_monitor, parse_monitor, monitor_executor = _make_openai_compatible_monitor_backend(
            monitor_model_name, OPENAI_KEY
        )
    elif monitor_source == "vllm":
        dispatch_monitor, parse_monitor, monitor_executor = _make_vllm_local_monitor_backend(monitor_model_name)
    elif monitor_source == "tinker":
        dispatch_monitor, parse_monitor, monitor_executor = _make_tinker_monitor_backend(monitor_model_name)
    else:
        raise ValueError(f"Unknown monitor_source: {monitor_source!r}. Choose from 'openai', 'tinker', 'vllm'.")

    # ------------------------------------------------------------------ #
    # Set up judge backend
    # ------------------------------------------------------------------ #
    if judge_source == "openai":
        dispatch_judge, parse_judge, judge_executor = _make_openai_compatible_judge_backend(
            judge_model_name, OPENAI_KEY
        )
    elif judge_source == "vllm":
        dispatch_judge, parse_judge, judge_executor = _make_vllm_local_judge_backend(judge_model_name)
    elif judge_source == "tinker":
        dispatch_judge, parse_judge, judge_executor = _make_tinker_judge_backend(judge_model_name)
    else:
        raise ValueError(f"Unknown judge_source: {judge_source!r}. Choose from 'openai', 'tinker', 'vllm'.")

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    for dataset_name in dataset_name_list:
        verifier = get_verifier(data_source=dataset_name, max_new_tokens=max_new_tokens)

        for model_idx, main_model_name_or_path in enumerate(main_model_name_or_path_list):
            # Load main model tokenizer only when repo_prefix is provided (used as
            # a fallback to encode main_output_ids when they are missing from saved entries).
            if repo_prefix is not None:
                from transformers import AutoTokenizer
                main_model_tokenizer = AutoTokenizer.from_pretrained(
                    f"{repo_prefix}/{main_model_name_or_path}", trust_remote_code=True
                )
            else:
                main_model_tokenizer = None

            for step_idx in step_idx_list:
                print(f"Start processing: model={main_model_name_or_path}, step={step_idx}, dataset={dataset_name} ...")
                if step_idx == "base" and not do_base:
                    print(f"[skip] base step for {main_model_name_or_path}")
                    continue

                result_dir = os.path.join(
                    # output_dir, main_model_name_or_path, f"step_{step_idx}", dataset_name, "main"
                    output_dir, main_model_name_or_path, f"step_{step_idx}", "main"
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

                # An entry needs processing if either key is missing or its value is None.
                pending_entry_indices = [
                    entry_idx for entry_idx, entry in enumerate(entries)
                    # TODO haha
                    # if entry.get(monitor_key) is None or entry.get(judge_key) is None
                ]
                num_already_annotated = len(entries) - len(pending_entry_indices)
                # run_label = f"{main_model_name_or_path} / step_{step_idx} / {dataset_name}"
                run_label = f"{main_model_name_or_path} / step_{step_idx}"

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
                                if judge_prompt is None and entry.get("main_output") is not None:
                                    # Verifier computes score directly from continuation (e.g. LengthOnlyVerifier)
                                    judge_score, judge_explanation = verifier.parse_llm_judge_output(
                                        None, None,
                                        continuation=entry["main_output"],
                                        continuation_token_ids=entry.get("main_output_ids"),
                                        gt=entry.get("ground_truth_for_llm_judge"),
                                    )
                                    judge_future = {"score": judge_score, "explanation": judge_explanation}
                                else:
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
                                    if main_model_tokenizer is not None and "main_output_ids" not in entry:
                                        entry["main_output_ids"] = main_model_tokenizer.encode(
                                            entry["main_output"], add_special_tokens=False
                                        )

                                    judge_score, judge_explanation = parse_judge(
                                        judge_future.result(), verifier,
                                        continuation=entry.get("main_output"),
                                        continuation_token_ids=entry.get("main_output_ids"),
                                        gt=entry.get("ground_truth_for_llm_judge")
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
