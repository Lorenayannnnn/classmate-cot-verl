
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

from verl.utils.reward_score.cot_monitor.BaseVerifier import get_monitor_verifier

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
):
    """
    Handle batch-wise requests to tinker
    """
    if USE_TINKER:
        if verifier is None:
            verifier = get_monitor_verifier()

        tinker_futures = []

        for monitor_prompt in monitor_prompts:
            if monitor_prompt is None:
                tinker_futures.append(None)
            else:
                tinker_futures.append(send_prompt_to_tinker(monitor_config, monitor_prompt, temperature))

        # Execute monitor_cot calls in parallel
        all_monitor_scores = []
        all_monitor_explanations = []

        for sample_future in tinker_futures:
            if sample_future is None:
                all_monitor_scores.append(0)
                all_monitor_explanations.append("No monitor prompt provided.")
                continue
            sample_result = sample_future.result()
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
        # TODO haha include <think> in CoT?
        cot_start = first_open + 1  # CoT begins right after <think>
        # cot_start = first_open  # CoT begins AT <think>
        start_with_think = True

        # First </think> that appears after the opening tag
        close_after_open = close_positions[close_positions > first_open]
        if len(close_after_open) > 0:
            # TODO haha include <think> in CoT?
            cot_end = close_after_open[0].item()  # CoT ends right before </think>
            # cot_end = close_after_open[0].item() + 1    # CoT ends right AT </think>
            end_with_think = True
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
            end_with_think = False
            final_output = None
            main_cot = main_pred.split("<think>", 1)[1] if "<think>" in main_pred else main_pred
    else:
        cot_start = 0
        start_with_think = False
        cot_end = None
        end_with_think = False
        final_output = None
        main_cot = main_pred

    # return main_cot, final_output, start_with_think, end_with_think, cot_start, cot_end
    return main_cot, final_output, cot_start, cot_end



def process_main_cot_helper(tokenizer, enable_thinking, data_source, main_response, main_pred_token_ids, main_response_mask, keep_ratio):
    """
    Split CoT from main model by all whitespace characters (e.g., whitespace, tab, newline), and keep a portion of tokens
    """
    # TODO haha double check for code_contests_modify_code
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

# def send_one_api_call(
#     openai_client,
#     model_name: str,
#     messages: List[Dict[str, str]],
#     temperature: float = 0.0,
#     response_format: Optional[Dict[str, str]] = None,
#     seed: Optional[int] = None,
#     max_retries: int = 3,
#     required_fields: Optional[List[str]] = None,
# ) -> Dict[str, Any]:
#     """
#     Send a single API call to OpenAI with retry logic and error handling.
#
#     Args:
#         openai_client: OpenAI client instance
#         model_name: Name of the model to use (e.g., "gpt-4", "gpt-3.5-turbo")
#         messages: List of message dictionaries with "role" and "content"
#         temperature: Temperature for sampling (default: 0.0 for deterministic)
#         response_format: Response format specification (e.g., {"type": "json_object"})
#         seed: Seed for reproducibility
#         max_retries: Maximum number of retries for transient errors (default: 3)
#         required_fields: List of required fields in JSON response (only used if response_format is json)
#
#     Returns:
#         Dict with keys:
#             - "success": bool indicating if the call succeeded
#             - "content": str with the response content (or error message)
#             - "parsed": parsed JSON (if response_format is json and parsing succeeded)
#             - "error_type": str indicating type of error (if failed)
#     """
#
#     for attempt in range(max_retries):
#         if USE_TINKER:
#             """
#             Send a prompt to the judge model and receive the response.
#
#             Args:
#                 prompt: The prompt to send to the judge model
#                 max_tokens: Maximum tokens in the response
#                 temperature: Temperature for sampling (0.0 for deterministic)
#
#             Returns:
#                 The parsed response text from the judge model
#             """
#             if judge_client_config is None:
#                 raise ValueError("Tinker judge not initialized. Set use_tinker_judge=True in __init__")
#
#             sampling_client = judge_client_config['sampling_client']
#             renderer = judge_client_config['renderer']
#
#             sampling_params = tinker.types.SamplingParams(
#                 max_tokens=judge_client_config['max_tokens'],
#                 stop=renderer.get_stop_sequences(),
#                 temperature=temperature,
#             )
#
#             # if judge_client_config["judge_model_name"] == "openai/gpt-oss-20b":
#             #     messages.append({"role": "system", "content": "Reasoning: low\nAnswer with only the final answer."})        # Let the model to return the answer immediately
#                 # messages.append({"role": "thinking", "content": ""})
#
#             model_input = renderer.build_generation_prompt(messages)
#
#             # Send prompt to judge model
#             sample_future = sampling_client.sample(
#                 prompt=model_input,
#                 num_samples=1,
#                 sampling_params=sampling_params,
#             )
#
#             sample_result = sample_future.result()
#
#             # Parse response
#             judge_tokens = sample_result.sequences[0].tokens
#             parsed_judge, _ = renderer.parse_response(judge_tokens)
#             judge_text = renderers.get_text_content(parsed_judge)
#
#             # try:
#             #     # Parse out from ```json{response}```
#             #     json_regex = r"```json\s*(\{.*?\})\s*```"
#             #     match = re.search(json_regex, judge_text, re.DOTALL)
#             #     if not match:
#             #         print("🐛🐛🐛 json output not found in judge_text", judge_text)
#             #         print("🐛🐛🐛 messages", messages)
#             #         raise JSONDecodeError("JSON response not found in judge output", judge_text, 0)
#             #     parsed_output = json.loads(match.group(1))
#             # except json.JSONDecodeError as e:
#             #     # Malformed JSON response; retry
#             #     if attempt < max_retries - 1:
#             #         wait_time = 2 ** attempt
#             #         print(
#             #             f"JSON decode error (attempt {attempt + 1}/{max_retries}): {e}. "
#             #             f"Retrying in {wait_time}s..."
#             #         )
#             #         time.sleep(wait_time)
#             #         continue
#             #
#             #     print(f"JSON decode error after {max_retries} attempts: {e}")
#             #     return {
#             #         "success": False,
#             #         "content": f"JSON decode error: {str(e)}",
#             #         "parsed": None,
#             #         "error_type": "json_decode",
#             #     }
#
#             judge_score, judge_explanation = parse_sycophancy_score(judge_text), parse_judge_explanation(judge_text)
#
#             return {
#                 "success": True,
#                 "content": judge_text,
#                 "parsed": {
#                     "monitor_score": judge_score,
#                     "explanation": judge_explanation,
#                 },
#                 "error_type": None,
#             }
#         else:
#             try:
#                 # Build request kwargs
#                 request_kwargs = {
#                     "model": model_name,
#                     "messages": messages,
#                     "temperature": temperature,
#                 }
#
#                 if response_format is not None:
#                     request_kwargs["response_format"] = response_format
#
#                 if seed is not None:
#                     request_kwargs["seed"] = seed
#
#                 # Make the API call
#                 response = openai_client.chat.completions.create(**request_kwargs)
#
#                 # Validate response
#                 if not response.choices:
#                     raise ValueError("Empty response from OpenAI API")
#
#                 content = response.choices[0].message.content
#                 if not content:
#                     raise ValueError("Empty content in OpenAI response")
#
#                 # Parse JSON if requested
#                 parsed_output = None
#                 if response_format and response_format.get("type") == "json_object":
#                     parsed_output = json.loads(content)
#
#                     # Validate required fields
#                     if required_fields:
#                         missing_fields = [f for f in required_fields if f not in parsed_output]
#                         if missing_fields:
#                             raise ValueError(f"Missing required fields in response: {missing_fields}")
#
#                 return {
#                     "success": True,
#                     "content": content,
#                     "parsed": parsed_output,
#                     "error_type": None,
#                 }
#
#             except (RateLimitError, APITimeoutError, InternalServerError, APIError) as e:
#                 # Retryable: 429, 500, 503, timeouts, and generic API errors
#                 if attempt < max_retries - 1:
#                     wait_time = 2 ** attempt
#                     print(
#                         f"Retryable error calling OpenAI API (attempt {attempt + 1}/{max_retries}): {e}. "
#                         f"Retrying in {wait_time}s..."
#                     )
#                     time.sleep(wait_time)
#                     continue
#
#                 print(f"Error calling OpenAI API after {max_retries} attempts: {e}")
#                 return {
#                     "success": False,
#                     "content": f"Error after {max_retries} retries: {str(e)}",
#                     "parsed": None,
#                     "error_type": "retryable_exhausted",
#                 }
#
#             except (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError) as e:
#                 # Non-retryable: 400, 401, 403, 404
#                 print(f"Non-retryable OpenAI API error: {e}")
#                 return {
#                     "success": False,
#                     "content": f"Non-retryable error: {str(e)}",
#                     "parsed": None,
#                     "error_type": "non_retryable",
#                 }
#
#             except json.JSONDecodeError as e:
#                 # Malformed JSON response; retry
#                 if attempt < max_retries - 1:
#                     wait_time = 2 ** attempt
#                     print(
#                         f"JSON decode error (attempt {attempt + 1}/{max_retries}): {e}. "
#                         f"Retrying in {wait_time}s..."
#                     )
#                     time.sleep(wait_time)
#                     continue
#
#                 print(f"JSON decode error after {max_retries} attempts: {e}")
#                 return {
#                     "success": False,
#                     "content": f"JSON decode error: {str(e)}",
#                     "parsed": None,
#                     "error_type": "json_decode",
#                 }
#
#             except Exception as e:
#                 # Generic exceptions
#                 print(f"Error calling OpenAI API: {e}")
#                 return {
#                     "success": False,
#                     "content": f"Error: {str(e)}",
#                     "parsed": None,
#                     "error_type": "unknown",
#                 }
#
#     # Should not reach here, but just in case
#     return {
#         "success": False,
#         "content": "Max retries exceeded",
#         "parsed": None,
#         "error_type": "max_retries_exceeded",
#     }
