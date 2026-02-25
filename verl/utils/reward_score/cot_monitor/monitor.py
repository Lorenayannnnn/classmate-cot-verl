
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
from keys import TINKER_API_KEY, INPUT_MONITOR_MODEL_NAME
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from verl.utils.reward_score.cot_monitor.BaseVerifier import get_verifier

USE_OPENAI = False
USE_TINKER = True


# ---------------------------------------------------------------------------
# Tinker judge client
# ---------------------------------------------------------------------------

def create_llm_judge(
    judge_model_name: str = INPUT_MONITOR_MODEL_NAME,
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
        "max_tokens": 1024,
    }

def send_prompt_to_tinker(
    input_judge_client_config: Dict[str, Any],
    prompt: str,
    temperature: float = 0.0,
) -> Future:
    """
    Send a single prompt to the Tinker judge model asynchronously.

    Returns a Future whose result() is the raw Tinker SampleResult.
    Parse it with verifier.parse_llm_judge_output(result, judge_client_config).
    """
    sampling_client = input_judge_client_config["sampling_client"]
    renderer = input_judge_client_config["renderer"]

    sampling_params = tinker.types.SamplingParams(
        max_tokens=input_judge_client_config["max_tokens"],
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
    )

    messages = [{"role": "user", "content": prompt}]
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


def parse_out_main_cot_output(
    main_pred,
    main_pred_token_ids,
    tokenizer,
    think_open_id,
    think_close_id,
):
    """Parse CoT and final output from a model response using token-level boundaries.

    Locates <think> and </think> directly in ``main_pred_token_ids`` so that
    the returned ``cot_start`` / ``cot_end`` indices are correctly aligned with
    ``data.batch["responses"]``.  ``main_cot`` is decoded from those exact token
    positions (no re-encoding round-trip).

    Note: only called when enable_thinking=True.

    Parameters
    ----------
    main_pred            : str   – decoded response string (used as fallback text
                                   for main_cot when </think> is absent)
    main_pred_token_ids  : Tensor or list – full response token IDs
    tokenizer            : tokenizer instance
    think_open_id        : int – token ID of <think>
    think_close_id       : int – token ID of </think>

    Returns
    -------
    main_cot         : str   – CoT text
    final_output     : str | None – text after </think>, or None if not found
    # start_with_think : bool
    # end_with_think   : bool
    cot_start        : int   – index of first CoT token in the response tensor
    cot_end          : int | None – exclusive end index (== index of </think>),
                                    or None when </think> was not found;
                                    caller should clamp with the valid response length
    """
    if not isinstance(main_pred_token_ids, torch.Tensor):
        main_pred_token_ids = torch.tensor(main_pred_token_ids)

    open_positions  = (main_pred_token_ids == think_open_id).nonzero(as_tuple=True)[0]
    close_positions = (main_pred_token_ids == think_close_id).nonzero(as_tuple=True)[0]

    if len(open_positions) > 0:
        first_open = open_positions[0].item()
        # haha include <think> in CoT?
        cot_start = first_open + 1  # CoT begins right after <think>
        # cot_start = first_open  # CoT begins AT <think>
        # start_with_think = True

        # First </think> that appears after the opening tag
        close_after_open = close_positions[close_positions > first_open]
        if len(close_after_open) > 0:
            # haha include <think> in CoT?
            cot_end = close_after_open[0].item()  # CoT ends right before </think>
            # cot_end = close_after_open[0].item() + 1    # CoT ends right AT </think>
            # end_with_think = True
            final_output = tokenizer.decode(
                main_pred_token_ids[cot_end + 1:].tolist(), skip_special_tokens=True
            )
            main_cot = tokenizer.decode(
                main_pred_token_ids[cot_start:cot_end].tolist(), skip_special_tokens=True
            )
        else:
            # No </think> found after <think>.  cot_end=None signals to the caller
            # to clamp using the valid response length from response_mask.
            # main_cot falls back to text splitting because we cannot determine the
            # valid end boundary (vs. padding) from token IDs alone here.
            cot_end = None
            # end_with_think = False
            final_output = None
            main_cot = main_pred.split("<think>", 1)[1] if "<think>" in main_pred else main_pred
    else:
        cot_start = 0
        # start_with_think = False
        cot_end = None
        # end_with_think = False
        final_output = None
        main_cot = main_pred

    # return main_cot, final_output, start_with_think, end_with_think, cot_start, cot_end
    return main_cot, final_output, cot_start, cot_end



def process_main_cot_helper(tokenizer, enable_thinking, data_source, main_response, main_pred_token_ids, main_response_mask, keep_ratio):
    """
    Split CoT from main model by all whitespace characters (e.g., whitespace, tab, newline), and keep a portion of tokens
    """
    # haha double check for code_contests_modify_code
    if data_source == "code_contests_modify_code":
        raise NotImplementedError("code_contests_modify_code truncation not supported now.")

    # start_with_think = False

    if enable_thinking:
        think_open_id  = tokenizer.convert_tokens_to_ids("<think>")
        think_close_id = tokenizer.convert_tokens_to_ids("</think>")

        # main_cot, final_output, start_with_think, end_with_think, cot_start, cot_end = parse_out_main_cot_output(
        main_cot, final_output, cot_start, cot_end = parse_out_main_cot_output(
            main_response, main_pred_token_ids, tokenizer, think_open_id, think_close_id
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
        main_cot = main_response
        final_output = main_response
        # end_with_think = False

    # with open("debug_main_cot.txt", "w") as f:
    #     f.write(f"main_cot: {main_cot}\n")
    #     f.write(f"final_output: {final_output}\n")
    #     f.write(f"main_pred_token_ids: {main_pred_token_ids}\n (len: {len(main_pred_token_ids)})\n")
    #     f.write(f"main_pred: {tokenizer.decode(main_pred_token_ids)}\n")
    #     f.write(f"main_response_mask: {main_response_mask.tolist()}\n")
    #     f.write(f"start_with_think: {start_with_think}\n")
    #     f.write(f"end_with_think: {end_with_think}\n")

    if keep_ratio == 1:
        # count num of new lines at the front and end
        # num_newlines_front = len(main_cot) - len(main_cot.lstrip("\n"))
        # num_newlines_end = len(main_cot) - len(main_cot.rstrip("\n"))
        return main_cot, main_pred_token_ids, final_output, main_response_mask if main_response_mask is not None else None

    raise NotImplementedError("Only keep_ratio=1 is supported now. Implementing other keep_ratio requires careful handling of response mask and start/end with think cases.")

    # num_to_keep = int(len(main_pred_token_ids) * keep_ratio)
    #
    # if num_to_keep <= 0:
    #     truncated_token_ids = main_pred_token_ids
    # else:
    #     truncated_token_ids = main_pred_token_ids[:num_to_keep]
    #
    # truncated_text = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
    # main_original_truncated_token_ids = tokenizer.encode("<think>" + truncated_text) if (
    #             enable_thinking and start_with_think) else truncated_token_ids
    # actual_kept_len = len(main_original_truncated_token_ids)
    #
    # # print("🐛🐛🐛 truncated_text", truncated_text)
    # # if type(truncated_token_ids) == torch.Tensor:
    # #     truncated_token_ids = truncated_token_ids.tolist()
    # # if type(main_original_truncated_token_ids) == torch.Tensor:
    # #     main_original_truncated_token_ids = main_original_truncated_token_ids.tolist()
    # # print("🐛🐛🐛 truncated_token_ids", truncated_token_ids)
    # # print("🐛🐛🐛 main_original_truncated_token_ids", main_original_truncated_token_ids)
    #
    # if main_response_mask is not None:
    #     classmate_response_mask = main_response_mask[:actual_kept_len].tolist() + [0] * (len(main_response_mask) - actual_kept_len)
    # else:
    #     classmate_response_mask = None
    #
    # return truncated_text, final_output, classmate_response_mask, end_with_think, start_with_think

