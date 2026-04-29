
from concurrent.futures import Future
from typing import Any, Dict, List

from openai import (
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
import torch
import tinker
from keys import TINKER_API_KEY
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from verl.utils.reward_score.BaseVerifier import get_verifier

USE_OPENAI = False
USE_TINKER = True


# ---------------------------------------------------------------------------
# Tinker judge client
# ---------------------------------------------------------------------------

def create_llm_judge(
    judge_model_name: str,
    base_url: str | None = None,
) -> Dict[str, Any]:
    """
    Create an LLM judge using Tinker framework with sampling client.

    Returns a config dict with keys: judge_model_name, sampling_client,
    renderer, tokenizer, max_tokens.
    """
    judge_tokenizer = get_tokenizer(judge_model_name)
    judge_renderer_name = model_info.get_recommended_renderer_name(judge_model_name)
    judge_renderer = renderers.get_renderer(judge_renderer_name, judge_tokenizer)

    service_client = tinker.ServiceClient(base_url=base_url, api_key=TINKER_API_KEY)
    judge_sampling_client = service_client.create_sampling_client(base_model=judge_model_name)

    return {
        "judge_model_name": judge_model_name,
        "sampling_client": judge_sampling_client,
        "renderer": judge_renderer,
        "tokenizer": judge_tokenizer,
        "max_tokens": 4096,
    }

def send_prompt_to_tinker(
    input_judge_client_config: Dict[str, Any],
    prompt: str,
    temperature: float = 0.0,
    icl_pairs: list[tuple[str, str]] | None = None,
) -> Future:
    """
    Send a single prompt to the Tinker judge model asynchronously.

    Returns a Future whose result() is the raw Tinker SampleResult.
    Parse it with verifier.parse_llm_judge_output(result, judge_client_config).

    Args:
        icl_pairs: Optional list of (user_msg, assistant_msg) few-shot demo
            pairs prepended as alternating user/assistant turns before the
            final user prompt.
    """
    sampling_client = input_judge_client_config["sampling_client"]
    renderer = input_judge_client_config["renderer"]

    sampling_params = tinker.types.SamplingParams(
        max_tokens=input_judge_client_config["max_tokens"],
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
    )

    messages = []
    if icl_pairs:
        for user_msg, assistant_msg in icl_pairs:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": prompt})
    model_input = renderer.build_generation_prompt(messages)

    return sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )


def monitor_cot_wrapper_w_tinker(
    monitor_config: Dict[str, Any],
    monitor_prompts: List[str],
    verifier=None,
    temperature: float = 0.0,
    max_retries: int = 5,
):
    """
    Handle batch-wise requests to tinker
    """
    if USE_TINKER:
        if verifier is None:
            verifier = get_verifier()

        tinker_futures = []

        for monitor_prompt in monitor_prompts:
            if monitor_prompt is None:
                tinker_futures.append(None)
            else:
                tinker_futures.append(send_prompt_to_tinker(monitor_config, monitor_prompt, temperature))

        # Execute monitor_cot calls in parallel
        all_monitor_scores = []
        all_monitor_explanations = []

        for monitor_prompt, sample_future in zip(monitor_prompts, tinker_futures):
            if sample_future is None:
                all_monitor_scores.append(verifier.invalid_score)
                all_monitor_explanations.append("No monitor prompt provided.")
                continue
            sample_result = sample_future.result()
            monitor_score, monitor_explanation = verifier.parse_monitor_output(sample_result, monitor_config)
            # print("🐛🐛🐛 monitor_prompt", monitor_prompt)
            # print("🐛🐛🐛 monitor_score", monitor_score)
            # print("🐛🐛🐛 monitor_explanation", monitor_explanation)
            for attempt in range(1, max_retries):
                if monitor_score != verifier.invalid_score:
                    break
                print(f"[retry {attempt}/{max_retries - 1}] Invalid monitor response, retrying...")
                retry_future = send_prompt_to_tinker(monitor_config, monitor_prompt, temperature)
                sample_result = retry_future.result()
                monitor_score, monitor_explanation = verifier.parse_monitor_output(sample_result, monitor_config)
            all_monitor_scores.append(monitor_score)
            all_monitor_explanations.append(monitor_explanation)

        return all_monitor_scores, all_monitor_explanations
    else:
        raise NotImplementedError("monitor_cot_wrapper_new is only implemented for Tinker judge. Set USE_TINKER=True and USE_OPENAI=False.")


def _find_subsequence(token_ids_tensor, subseq_list):
    """Return sorted list of start positions where subseq_list appears in token_ids_tensor."""
    n = token_ids_tensor.size(0)
    m = len(subseq_list)
    if m == 0 or n < m:
        return []
    subseq_tensor = torch.tensor(subseq_list, dtype=token_ids_tensor.dtype, device=token_ids_tensor.device)
    positions = []
    for i in range(n - m + 1):
        if torch.equal(token_ids_tensor[i:i + m], subseq_tensor):
            positions.append(i)
    return positions


def parse_out_main_cot_output(
    main_pred,
    main_pred_token_ids,
    tokenizer,
    think_start_str,
    think_end_str,
):
    """Parse CoT and final output from a model response using token-level boundaries.

    Encodes ``think_start_str`` and ``think_end_str`` into token IDs and locates
    them directly in ``main_pred_token_ids`` so that the returned ``cot_start`` /
    ``cot_end`` indices are correctly aligned with ``data.batch["responses"]``.
    ``main_cot`` is decoded from those exact token positions (no re-encoding
    round-trip).

    Supports single-token markers (e.g. ``think_start_str`` / ``think_end_str``) and
    multi-token markers (e.g. ``### Reasoning`` / ``### Test Case``).

    Note: only called when enable_thinking=True.

    Parameters
    ----------
    main_pred            : str   – decoded response string (used as fallback text
                                   for main_cot when the closing marker is absent)
    main_pred_token_ids  : Tensor or list – full response token IDs
    tokenizer            : tokenizer instance
    think_start_str      : str   – opening marker string (e.g. ``<think>`` or
                                   ``### Reasoning``)
    think_end_str        : str   – closing marker string (e.g. ``</think>`` or
                                   ``### Test Case``)

    Returns
    -------
    main_cot         : str   – CoT text
    final_output     : str | None – text after the closing marker, or None if not found
    cot_start        : int   – index of first CoT token in the response tensor
                               (i.e. right after the opening marker)
    cot_end          : int | None – exclusive end index of CoT content
                               (== start index of closing marker), or None when not found;
                               caller should clamp with the valid response length
    open_think_pos   : int | None – start index of the opening marker itself, or None
                               when the opening marker was not found
    close_think_pos  : int | None – start index of the closing marker itself (== cot_end),
                               or None when the closing marker was not found
    open_marker_len  : int – number of tokens in the opening marker
    close_marker_len : int – number of tokens in the closing marker
    """
    if not isinstance(main_pred_token_ids, torch.Tensor):
        main_pred_token_ids = torch.tensor(main_pred_token_ids)

    think_open_ids  = tokenizer.encode(think_start_str, add_special_tokens=False)
    think_close_ids = tokenizer.encode(think_end_str, add_special_tokens=False)

    open_positions  = _find_subsequence(main_pred_token_ids, think_open_ids)
    close_positions = _find_subsequence(main_pred_token_ids, think_close_ids)

    open_marker_len  = len(think_open_ids)
    close_marker_len = len(think_close_ids)

    if len(open_positions) > 0:
        first_open = open_positions[0]
        open_think_pos = first_open
        cot_start = first_open + open_marker_len  # CoT begins right after opening marker

        # First closing marker that appears after the opening marker
        close_after_open = [p for p in close_positions if p > first_open]
        if len(close_after_open) > 0:
            cot_end = close_after_open[0]  # CoT ends before closing marker (exclusive)
            close_think_pos = cot_end
            final_output = tokenizer.decode(
                main_pred_token_ids[cot_end + close_marker_len:].tolist(), skip_special_tokens=True
            )
            main_cot = tokenizer.decode(
                main_pred_token_ids[cot_start:cot_end].tolist(), skip_special_tokens=True
            )
        else:
            # No closing marker found after opening marker.  cot_end=None signals to
            # the caller to clamp using the valid response length from response_mask.
            # main_cot falls back to text splitting because we cannot determine the
            # valid end boundary (vs. padding) from token IDs alone here.
            cot_end = None
            close_think_pos = None
            final_output = None
            main_cot = main_pred.split(think_start_str, 1)[1] if think_start_str in main_pred else main_pred
    else:
        # Opening marker not found in response (e.g. OLMo-3 where <think> is forced
        # into the prompt, so the response begins mid-CoT).  CoT starts at position 0.
        open_think_pos = None
        cot_start = 0

        if len(close_positions) > 0:
            # Closing marker found — use it to delimit CoT and final output.
            cot_end = close_positions[0]
            close_think_pos = cot_end
            final_output = tokenizer.decode(
                main_pred_token_ids[cot_end + close_marker_len:].tolist(), skip_special_tokens=True
            )
            main_cot = tokenizer.decode(
                main_pred_token_ids[cot_start:cot_end].tolist(), skip_special_tokens=True
            )
        else:
            cot_end = None
            close_think_pos = None
            final_output = None
            main_cot = main_pred

    return main_cot, final_output, cot_start, cot_end, open_think_pos, close_think_pos, open_marker_len, close_marker_len



def _parse_keep_spec(keep_ratio):
    """Parse keep_ratio into (mode, value).

    Accepts:
        1 or 1.0 or "1"           → ("all",         None)  keep all CoT tokens
        "first-100-tokens"         → ("first_tokens", 100)  keep first 100 CoT tokens
        "last-100-tokens"          → ("last_tokens",  100)  keep last  100 CoT tokens
        0.5 or "0.5"              → ("ratio",        0.5)  keep first 50 % of CoT tokens
    """
    if isinstance(keep_ratio, str):
        if keep_ratio.startswith("last-") and keep_ratio.endswith("-tokens"):
            return "last_tokens", int(keep_ratio[len("last-") : -len("-tokens")])
        if keep_ratio.startswith("first-") and keep_ratio.endswith("-tokens"):
            return "first_tokens", int(keep_ratio[len("first-") : -len("-tokens")])
        r = float(keep_ratio)
    else:
        r = float(keep_ratio)
    return ("all", None) if r == 1.0 else ("ratio", r)


def process_main_cot_helper(tokenizer, enable_thinking, data_source, main_response, main_pred_token_ids, main_response_mask, keep_ratio, think_start_str, think_end_str):
    """
    Split CoT from main model by all whitespace characters (e.g., whitespace, tab, newline), and keep a portion of tokens
    """
    if enable_thinking:
        # main_cot, final_output, start_with_think, end_with_think, cot_start, cot_end = parse_out_main_cot_output(
        main_cot, final_output, cot_start, cot_end, _, _, _, _ = parse_out_main_cot_output(
            main_response, main_pred_token_ids, tokenizer, think_start_str, think_end_str
        )

        if main_response_mask is not None:
            max_response_len = main_response_mask.size(0)

            # Resolve cot_end when </think> was not found
            if cot_end is None:
                cot_end = int(main_response_mask.sum().item())
            cot_end = min(cot_end, max_response_len)

            # Build classmate_input_mask anchored to actual token positions in `responses`
            classmate_input_mask = (
                [0] * cot_start
                + [1] * max(0, cot_end - cot_start)
                + [0] * (max_response_len - cot_end)
            )
            main_response_mask = classmate_input_mask

            # Extract exact CoT token IDs from the response tensor (no re-encoding)
            main_pred_token_ids = main_pred_token_ids[cot_start:cot_end].tolist()[:max_response_len]

            # if len(main_pred_token_ids) > 0:
            #     # Trim front and back until it doesn't start or end with \n
            #     idx = 0
            #     while tokenizer.decode(main_pred_token_ids[0]).replace("\n", "") == "":
            #         main_pred_token_ids = main_pred_token_ids[1:]
            #         main_response_mask[idx] = 0
            #         idx += 1
            #
            #     idx = 0
            #     while tokenizer.decode(main_pred_token_ids[-1]).replace("\n", "") == "":
            #         main_pred_token_ids = main_pred_token_ids[:-1]
            #         main_response_mask[-1 - idx] = 0
            #
            #     main_cot = tokenizer.decode(main_pred_token_ids)
        else:
            # Inference path (no mask): slice to CoT content so keep_ratio counts
            # only tokens between <think> and </think>, not markers or final output.
            if cot_end is not None:
                main_pred_token_ids = list(main_pred_token_ids[cot_start:cot_end])
    else:
        main_cot = main_response
        final_output = main_response
        cot_end = None
        # end_with_think = False

    # with open("debug_main_cot.txt", "w") as f:
    #     f.write(f"main_cot: {main_cot}\n")
    #     f.write(f"final_output: {final_output}\n")
    #     f.write(f"main_pred_token_ids: {main_pred_token_ids}\n (len: {len(main_pred_token_ids)})\n")
    #     f.write(f"main_pred: {tokenizer.decode(main_pred_token_ids)}\n")
    #     f.write(f"main_response_mask: {main_response_mask.tolist()}\n")
    #     f.write(f"start_with_think: {start_with_think}\n")
    #     f.write(f"end_with_think: {end_with_think}\n")

    mode, keep_val = _parse_keep_spec(keep_ratio)

    if mode == "all":
        return main_cot, main_pred_token_ids, final_output, main_response_mask if main_response_mask is not None else None, cot_end

    # ── Compute number of CoT tokens to keep and select them ─────────────────
    n_cot = len(main_pred_token_ids) if main_pred_token_ids is not None else 0
    if mode in ("first_tokens", "last_tokens"):
        num_to_keep = min(keep_val, n_cot)
    else:  # "ratio"
        num_to_keep = max(1, int(n_cot * keep_val)) if n_cot > 0 else 0

    if mode == "last_tokens":
        truncated_token_ids = list(main_pred_token_ids[n_cot - num_to_keep:]) if num_to_keep > 0 else []
    else:  # "first_tokens" or "ratio"
        truncated_token_ids = list(main_pred_token_ids[:num_to_keep])
    truncated_text = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)

    # ── Update response mask to match the selected CoT slice ─────────────────
    # The 1s in main_response_mask form a contiguous block [cot_start, cot_end),
    # so we can reconstruct directly without scanning.
    if main_response_mask is not None:
        mask_len = len(main_response_mask)
        if mode == "last_tokens":
            keep_start = cot_end - num_to_keep
            classmate_response_mask = [0] * keep_start + [1] * num_to_keep + [0] * (mask_len - cot_end)
        else:  # first_tokens or ratio
            classmate_response_mask = [0] * cot_start + [1] * num_to_keep + [0] * (mask_len - cot_start - num_to_keep)
    else:
        classmate_response_mask = None

    return truncated_text, truncated_token_ids, final_output, classmate_response_mask, cot_end

