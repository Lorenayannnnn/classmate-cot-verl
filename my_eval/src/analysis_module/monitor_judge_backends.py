"""
Shared backend helpers for different_monitor_*_across_models.py scripts.

Provides:
  - make_backend: factory returning a BaseBackend for a given source and model name.
  - run_different_monitor_and_judge: main annotation + metrics loop
  - estimate_openai_cost: extrapolate OpenAI API cost from n sample calls
  - OPENAI_PRICING: prices per 1M tokens by model
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from keys import OPENAI_KEY
from verl.utils.reward_score.BaseVerifier import get_verifier
from verl.utils.reward_score.monitor import create_llm_judge
from verl.workers.reward_model.judge_backend import BaseBackend
from verl.workers.reward_model.openai_backend import OpenAICompatibleBackend
from verl.workers.reward_model.tinker_backend import TinkerJudgeBackend
from verl.workers.reward_model.vllm_generative_backend import VLLMGenerativeJudgeBackend

# Prices in USD per 1M tokens: (input, output)
OPENAI_PRICING = {
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o":       (2.50, 10.00),
    "gpt-4.1":      (2.00,  8.00),
}


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def _vllm_generation_config(model_name: str) -> dict:
    """Return model-specific vLLM sampling params."""
    if model_name == "Qwen/Qwen2.5-0.5B-Instruct":
        # https://qwen.ai/blog?id=qwen2.5
        return {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.05, "max_tokens": 1024}
    elif model_name == "HuggingFaceTB/SmolLM2-360M-Instruct":
        # https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
        return {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1024}
    else:
        raise ValueError(f"Make sure to have the sampling param for model {model_name}")


def make_backend(source: str, model_name: str) -> BaseBackend:
    """Construct a BaseBackend for the given source and model.

    Args:
        source: One of "tinker", "vllm", "openai".
        model_name: Model name or path.
    """
    if source == "tinker":
        return TinkerJudgeBackend(create_llm_judge(judge_model_name=model_name))
    elif source == "vllm":
        return VLLMGenerativeJudgeBackend(
            model_name_or_path=model_name,
            max_model_len=8192,
            generation_config=_vllm_generation_config(model_name),
            gpu_memory_utilization=0.9,
        )
    elif source == "openai":
        return OpenAICompatibleBackend(model_name=model_name, api_key=OPENAI_KEY)
    else:
        raise ValueError(f"Unknown source: {source!r}. Choose from 'tinker', 'vllm', 'openai'.")


def _prepare_inputs_for_behavior(
    pending_entry_indices,
    entries,
    b_monitor_key,
    b_judge_key,
    behavior_key,
    behavior_verifier,
    judge_backend,
):
    """Collect monitor and judge inputs for all pending entries for one behavior.

    Returns:
        monitor_inputs: list of prompt strings (or None for already-annotated).
        judge_inputs: list of backend inputs (or None).
        judge_use_continuation: parallel bool list — True when score comes from
            token count rather than an LLM call (e.g. LongerResponseVerifier).
    """
    monitor_inputs = []
    judge_inputs = []
    judge_use_continuation = []

    for entry_idx in tqdm(pending_entry_indices, desc=f"preparing inputs ({behavior_key})"):
        entry = entries[entry_idx]

        # Monitor input
        if entry.get(b_monitor_key) is not None:
            monitor_inputs.append(None)  # already annotated — skip
        else:
            monitor_doc = {
                "raw_question_for_monitor": entry["question"],
                "hint_str_for_monitor": entry.get("hint_str_for_monitor", ""),
                "truncated_main_CoT": entry["truncated_main_CoT"],
                "main_response": entry["truncated_main_CoT"],
            }
            monitor_inputs.append(behavior_verifier.create_monitor_message(monitor_doc))

        # Judge input
        if entry.get(b_judge_key) is not None:
            judge_inputs.append(None)  # already annotated — skip
            judge_use_continuation.append(False)
        elif entry.get("main_output") is None:
            judge_inputs.append(None)
            judge_use_continuation.append(False)
        else:
            prompt = judge_backend.prepare_input(behavior_verifier, entry["question"], entry["main_output"])
            if prompt is None:
                # Verifier computes score directly from continuation (e.g. LengthOnlyVerifier)
                judge_inputs.append(None)
                judge_use_continuation.append(True)
            else:
                judge_inputs.append(prompt)
                judge_use_continuation.append(False)

    return monitor_inputs, judge_inputs, judge_use_continuation


def _parse_outputs_for_behavior(
    pending_entry_indices,
    entries,
    b_monitor_key,
    b_monitor_expl_key,
    b_judge_key,
    b_judge_expl_key,
    behavior_key,
    behavior_verifier,
    monitor_outputs,
    judge_outputs,
    judge_use_continuation,
    main_model_tokenizer,
):
    """Parse raw backend outputs and write scores/explanations into each entry dict."""
    for i, entry_idx in enumerate(tqdm(pending_entry_indices, desc=f"parsing results ({behavior_key})")):
        entry = entries[entry_idx]

        # Parse monitor output
        if entry.get(b_monitor_key) is None:
            raw = monitor_outputs[i]
            if raw is None:
                entry[b_monitor_key] = behavior_verifier.invalid_score
                entry[b_monitor_expl_key] = "CoT is empty"
            else:
                score, explanation = behavior_verifier.parse_monitor_output(raw)
                entry[b_monitor_key] = score
                entry[b_monitor_expl_key] = explanation

        # Parse judge output
        if entry.get(b_judge_key) is None:
            if judge_use_continuation[i]:
                # Verifier computes score directly from continuation (e.g. LengthOnlyVerifier)
                score, explanation = behavior_verifier.parse_llm_judge_output(
                    None,
                    continuation=entry["main_output"],
                    continuation_token_ids=entry.get("main_output_ids"),
                    gt=entry.get("ground_truth_for_llm_judge"),
                )
            else:
                raw = judge_outputs[i]
                if isinstance(raw, dict):
                    # HFScoringJudgeBackend returns ready-to-use dicts
                    score = raw["score"]
                    explanation = raw.get("judge_explanation", "hf_scoring")
                elif raw is None:
                    score, explanation = behavior_verifier.invalid_score, "Output is empty"
                else:
                    if main_model_tokenizer is not None and "main_output_ids" not in entry:
                        entry["main_output_ids"] = main_model_tokenizer.encode(
                            entry["main_output"], add_special_tokens=False
                        )
                    score, explanation = behavior_verifier.parse_llm_judge_output(
                        raw,
                        continuation=entry.get("main_output"),
                        continuation_token_ids=entry.get("main_output_ids"),
                        gt=entry.get("ground_truth_for_llm_judge"),
                    )
            entry[b_judge_key] = score
            entry[b_judge_expl_key] = explanation


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
    from openai import OpenAI
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
    dataset_name_list,
    step_idx_list,
    output_dir,
    monitor_model_name,
    judge_model_name,
    do_base,
    monitor_source: str = "tinker",
    judge_source: str = "tinker",
    max_new_tokens: int | None = None,
):
    """
    Args:
        monitor_source: Backend for the monitor model. One of "openai", "tinker", "vllm".
        judge_source:   Backend for the judge model.  One of "openai", "tinker", "vllm".
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

    monitor_backend = make_backend(monitor_source, monitor_model_name)
    judge_backend = make_backend(judge_source, judge_model_name)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    for dataset_name in dataset_name_list:
        is_general_reward = dataset_name == "general_reward"
        behavior_keys = (
            ["sycophancy", "confidence", "longer_response"]
            if is_general_reward
            else [dataset_name]
        )

        for step_idx in step_idx_list:
            print(f"Start processing: step={step_idx}, dataset={dataset_name} ...")
            if step_idx == "base" and not do_base:
                print(f"[skip] base step")
                continue

            # output_dir is the seed-level dir: {method_dir}/{seed}
            # Inference writes to: {method_dir}/step_{step_idx}/{seed}/  (for base: {method_dir}/step_base/)
            method_dir = os.path.dirname(output_dir)
            seed_name = os.path.basename(output_dir)
            if step_idx == "base":
                result_dir = os.path.join(method_dir, "step_base")
            else:
                result_dir = os.path.join(method_dir, f"step_{step_idx}", seed_name)
            preds_fn = os.path.join(result_dir, "preds.jsonl")

            if not os.path.exists(preds_fn):
                print(f"[skip] {preds_fn} not found")
                continue

            # ---------------------------------------------------------- #
            # Read all entries
            # ---------------------------------------------------------- #
            with open(preds_fn, "r") as preds_file:
                entries = [json.loads(line) for line in preds_file]

            assert len(entries) == 500, f"Expected 500 entries in {preds_fn}, got {len(entries)}"

            all_entry_indices = list(range(len(entries)))

            # ---------------------------------------------------------- #
            # For each behavior, collect monitor and judge inputs,
            # run backends, and parse outputs. Save on KeyboardInterrupt.
            # Already-annotated entries are skipped inside the helpers.
            # ---------------------------------------------------------- #
            print(f"[run]    {output_dir} / step_{step_idx}: annotating {len(entries)} entries ...")
            interrupted = False
            try:
                for behavior_key in behavior_keys:
                    b_monitor_key      = f"{behavior_key}_{monitor_key}"
                    b_monitor_expl_key = f"{behavior_key}_{monitor_expl_key}"
                    b_judge_key        = f"{behavior_key}_{judge_key}"
                    b_judge_expl_key   = f"{behavior_key}_{judge_expl_key}"
                    behavior_verifier = get_verifier(data_source=behavior_key, max_new_tokens=max_new_tokens)

                    monitor_inputs, judge_inputs, judge_use_continuation = _prepare_inputs_for_behavior(
                        all_entry_indices, entries,
                        b_monitor_key, b_judge_key,
                        behavior_key, behavior_verifier, judge_backend,
                    )

                    monitor_outputs = monitor_backend.run_batch(monitor_inputs)
                    judge_outputs   = judge_backend.run_batch(judge_inputs)

                    _parse_outputs_for_behavior(
                        all_entry_indices, entries,
                        b_monitor_key, b_monitor_expl_key,
                        b_judge_key, b_judge_expl_key,
                        behavior_key, behavior_verifier,
                        monitor_outputs, judge_outputs, judge_use_continuation,
                        None,  # main_model_tokenizer not used here
                    )

            except KeyboardInterrupt:
                print(f"\n[interrupt] Saving partial progress to {preds_fn} ...")
                interrupted = True

            finally:
                with open(preds_fn, "w") as preds_file:
                    for entry in entries:
                        preds_file.write(json.dumps(entry) + "\n")
                print(f"[saved]  {preds_fn}")

            if interrupted:
                monitor_backend.close()
                judge_backend.close()
                return

            # ---------------------------------------------------------- #
            # Compute and save metrics
            # Combined into one file: { behavior_key: metrics_dict, ... }
            # ---------------------------------------------------------- #
            all_metrics = {}
            for behavior_key in behavior_keys:
                b_monitor_key = f"{behavior_key}_{monitor_key}"
                b_judge_key   = f"{behavior_key}_{judge_key}"
                behavior_verifier = get_verifier(data_source=behavior_key, max_new_tokens=max_new_tokens)

                monitor_scores = np.array([entry.get(b_monitor_key, behavior_verifier.invalid_score) for entry in entries])
                judge_scores   = np.array([entry.get(b_judge_key,   behavior_verifier.invalid_score) for entry in entries])

                monitor_valid_mask = monitor_scores != behavior_verifier.invalid_score
                judge_valid_mask   = judge_scores   != behavior_verifier.invalid_score
                both_valid_mask    = monitor_valid_mask & judge_valid_mask

                b_metrics = behavior_verifier.compute_metrics(
                    predictions=monitor_scores[both_valid_mask].astype(float),
                    ground_truths=judge_scores[both_valid_mask].astype(float),
                )
                b_metrics["total_valid_CoT_entries"]    = int(np.sum(monitor_valid_mask))
                b_metrics["total_valid_output_entries"] = int(np.sum(judge_valid_mask))
                b_metrics["total_monitored_entries"]    = int(np.sum(both_valid_mask))
                b_metrics["total_entries"]              = len(entries)
                all_metrics[behavior_key] = b_metrics

            metrics_output_path = os.path.join(result_dir, metrics_file_name)
            with open(metrics_output_path, "w") as metrics_file:
                json.dump(all_metrics, metrics_file, indent=4)
            print(f"[metrics] saved to {metrics_output_path}")

    monitor_backend.close()
    judge_backend.close()
