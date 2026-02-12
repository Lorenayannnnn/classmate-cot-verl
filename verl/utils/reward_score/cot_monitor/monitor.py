import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

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

import tinker
from keys import TINKER_API_KEY
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from verl.utils.my_utils import parse_out_main_cot_output

# TODO haha change tinker llm judge model name
USE_OPENAI = False
USE_TINKER = True
TINKER_MODEL_NAME = "openai/gpt-oss-20b"
# JUDGE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# JUDGE_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

def create_llm_judge(
    judge_model_name: str = TINKER_MODEL_NAME,
    base_url: str | None = None,
) -> Dict[str, Any]:
    """
    Create an LLM judge using Tinker framework with sampling client.

    Args:
        judge_model_name: The model to use for judging (default: Llama-3.1-8B-Instruct)
        base_url: Optional base URL for the Tinker service

    Returns:
        A dictionary containing:
            - 'sampling_client': The Tinker sampling client
            - 'renderer': The renderer for parsing responses
            - 'tokenizer': The tokenizer for the model
    """
    # Get tokenizer and renderer for judge model
    judge_tokenizer = get_tokenizer(judge_model_name)
    judge_renderer_name = model_info.get_recommended_renderer_name(judge_model_name)
    judge_renderer = renderers.get_renderer(judge_renderer_name, judge_tokenizer)

    # Setup Tinker service client
    service_client = tinker.ServiceClient(base_url=base_url, api_key=TINKER_API_KEY)

    # Create sampling client for judge model
    judge_sampling_client = service_client.create_sampling_client(base_model=judge_model_name)

    return {
        "judge_model_name": judge_model_name,
        'sampling_client': judge_sampling_client,
        'renderer': judge_renderer,
        'tokenizer': judge_tokenizer,
        'max_tokens': 1024,
    }

if USE_TINKER:
    judge_client_config = create_llm_judge()        # TODO haha globally temporary
else:
    judge_client_config = None


def parse_sycophancy_score(resp: str) -> float | None:
    """Extract the sycophancy score from the judge's response."""
    match = re.search(r"RATING:\s*(\d+)", resp, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 10:
            return float(score)
    # Fallback: look for any standalone number 1-10
    numbers = re.findall(r"\b([1-9]|10)\b", resp)
    if numbers:
        return float(numbers[0])
    return None


def parse_judge_explanation(resp: str) -> str:
    """Extract the explanation from the judge's response."""
    match = re.search(r"EXPLANATION:\s*(.*)", resp, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No explanation provided."


def process_main_cot_helper(tokenizer, enable_thinking, data_source, main_response, main_pred_token_ids, main_response_mask, keep_ratio):
    """
    Split CoT from main model by all whitespace characters (e.g., whitespace, tab, newline), and keep a portion of tokens
    """
    # TODO haha double check for code_contests_modify_code
    if data_source == "code_contests_modify_code":
        raise NotImplementedError("code_contests_modify_code truncation not supported now.")

    start_with_think = False

    if enable_thinking:
        main_cot, final_output, start_with_think, end_with_think = parse_out_main_cot_output(main_response)

        main_pred_token_ids = tokenizer.encode(main_cot)
        # main_response_mask = main_response_mask[:len(main_pred_token_ids) + 1] if main_response_mask is not None else None
        if main_response_mask is not None:     # len(main_pred_token_ids) == 0 means main_cot is empty string
            main_response_mask = main_response_mask[:len(main_pred_token_ids)]
            if len(main_pred_token_ids) > 0:
                # Trim front and back until it doesn't start or end with \n
                idx = 0
                while tokenizer.decode(main_pred_token_ids[0]).replace("\n", "") == "":
                    main_pred_token_ids = main_pred_token_ids[1:]
                    main_response_mask[idx] = 0
                    idx += 1

                idx = 0
                while tokenizer.decode(main_pred_token_ids[-1]).replace("\n", "") == "":
                    main_pred_token_ids = main_pred_token_ids[:-1]
                    main_response_mask[-1 - idx] = 0

                main_cot = tokenizer.decode(main_pred_token_ids)
    else:
        main_cot = main_response
        final_output = main_response
        end_with_think = False

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
        return main_cot, final_output, main_response_mask.tolist() if main_response_mask is not None else None, end_with_think, start_with_think

    raise NotImplementedError("Only keep_ratio=1 is supported now. Implementing other keep_ratio requires careful handling of response mask and start/end with think cases.")

    num_to_keep = int(len(main_pred_token_ids) * keep_ratio)

    if num_to_keep <= 0:
        truncated_token_ids = main_pred_token_ids
    else:
        truncated_token_ids = main_pred_token_ids[:num_to_keep]

    truncated_text = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
    main_original_truncated_token_ids = tokenizer.encode("<think>" + truncated_text) if (
                enable_thinking and start_with_think) else truncated_token_ids
    actual_kept_len = len(main_original_truncated_token_ids)

    # print("🐛🐛🐛 truncated_text", truncated_text)
    # if type(truncated_token_ids) == torch.Tensor:
    #     truncated_token_ids = truncated_token_ids.tolist()
    # if type(main_original_truncated_token_ids) == torch.Tensor:
    #     main_original_truncated_token_ids = main_original_truncated_token_ids.tolist()
    # print("🐛🐛🐛 truncated_token_ids", truncated_token_ids)
    # print("🐛🐛🐛 main_original_truncated_token_ids", main_original_truncated_token_ids)

    if main_response_mask is not None:
        classmate_response_mask = main_response_mask[:actual_kept_len].tolist() + [0] * (len(main_response_mask) - actual_kept_len)
    else:
        classmate_response_mask = None

    return truncated_text, final_output, classmate_response_mask, end_with_think, start_with_think

def monitor_cot_wrapper_w_tinker(
    monitor_prompts: List[str],
    temperature: float = 0.0,
):
    """
    Wrapper to process monitor_cot calls sequentially without API calls (for testing/debugging).

    Args:
        monitor_prompts: List of fully formatted monitor prompts
        monitor_reward_type: Reward type (e.g., "binary", "non_binary")

    Returns:
        Tuple of (all_monitor_use_hint, all_monitor_explanations)
    """
    if USE_TINKER:
        sampling_client = judge_client_config['sampling_client']
        renderer = judge_client_config['renderer']

        sampling_params = tinker.types.SamplingParams(
            max_tokens=judge_client_config['max_tokens'],
            stop=renderer.get_stop_sequences(),
            temperature=temperature,
        )

        tinker_futures = []

        for monitor_prompt in monitor_prompts:
            messages = [{"role": "user", "content": monitor_prompt}]
            if judge_client_config["judge_model_name"] == "openai/gpt-oss-20b":
                messages.append({"role": "system", "content": "Reasoning: low\nAnswer with only the final answer."})  # Let the model to return the answer immediately
            model_input = renderer.build_generation_prompt(messages)

            # Send prompt to judge model
            sample_future = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )

            tinker_futures.append(sample_future)

        # Execute monitor_cot calls in parallel
        all_monitor_scores = []
        all_monitor_explanations = []

        for sample_future in tinker_futures:
            sample_result = sample_future.result()
            # Parse response
            judge_tokens = sample_result.sequences[0].tokens
            parsed_judge, _ = renderer.parse_response(judge_tokens)
            judge_text = renderers.get_text_content(parsed_judge)
            judge_score, judge_explanation = parse_sycophancy_score(judge_text), parse_judge_explanation(judge_text)
            all_monitor_scores.append(judge_score)
            all_monitor_explanations.append(judge_explanation)

        return all_monitor_scores, all_monitor_explanations
    else:
        raise NotImplementedError("monitor_cot_wrapper_new is only implemented for Tinker judge. Set USE_TINKER=True and USE_OPENAI=False.")


# def monitor_cot_wrapper(
#     monitor_prompts: List[str],
#     openai_client,
#     model_name: str,
#     monitor_reward_type: str,
# ):
#     """
#     Wrapper to batch process monitor_cot calls in parallel.
#
#     Args:
#         monitor_prompts: List of fully formatted monitor prompts
#         openai_client: OpenAI client instance
#         model_name: Name of the OpenAI model to use
#         monitor_reward_type: Reward type (e.g., "binary", "non_binary")
#
#     Returns:
#         Tuple of (all_monitor_use_hint, all_monitor_explanations)
#     """
#     from concurrent.futures import ThreadPoolExecutor, as_completed
#
#     num_samples = len(monitor_prompts)
#
#     # Execute monitor_cot calls in parallel
#     all_monitor_scores = [None] * num_samples
#     all_monitor_explanations = [None] * num_samples
#
#     with ThreadPoolExecutor(max_workers=min(8, num_samples)) as executor:
#         # Submit all tasks
#         future_to_idx = {
#             executor.submit(
#                 monitor_cot_from_prompt,
#                 monitor_prompts[idx],
#                 openai_client,
#                 model_name,
#                 monitor_reward_type,
#             ): idx
#             for idx in range(num_samples)
#         }
#
#         # from tqdm import tqdm
#         # pbar = tqdm(total=num_samples, desc="Processing monitor prompts")
#
#         # Collect results as they complete
#         for future in as_completed(future_to_idx):
#             idx = future_to_idx[future]
#             monitor_result_dict = future.result()
#             all_monitor_scores[idx] = monitor_result_dict["monitor_score"]
#             all_monitor_explanations[idx] = monitor_result_dict["explanation"]
#             # pbar.update(1)
#
#     return all_monitor_scores, all_monitor_explanations


# def monitor_cot_from_prompt(
#     monitor_prompt: str,
#     openai_client,
#     model_name: str,
#     monitor_reward_type: Optional[str] = None,
#     max_retries: int = 3,
# ):
#     result = send_one_api_call(
#         openai_client=openai_client,
#         model_name=model_name,
#         messages=[{"role": "user", "content": monitor_prompt}],
#         temperature=0.0,
#         response_format={"type": "json_object"},
#         seed=42,
#         max_retries=max_retries,
#         required_fields=["monitor_score", "explanation"],
#     )
#
#     if result["success"] and result["parsed"]:
#         score = result["parsed"]["monitor_score"]
#
#         if monitor_reward_type == "binary":
#             if isinstance(score, bool):
#                 score = 1 if score else 0
#             else:
#                 score = 1 if score else 0
#         else:
#             score = float(score) if isinstance(score, (int, float)) else score
#
#         return {
#             "monitor_score": score,
#             "explanation": result["parsed"]["explanation"],
#         }
#
#     return {
#         "monitor_score": None,
#         "explanation": result["content"],
#     }

def send_one_api_call(
    openai_client,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    response_format: Optional[Dict[str, str]] = None,
    seed: Optional[int] = None,
    max_retries: int = 3,
    required_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Send a single API call to OpenAI with retry logic and error handling.
    
    Args:
        openai_client: OpenAI client instance
        model_name: Name of the model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        messages: List of message dictionaries with "role" and "content"
        temperature: Temperature for sampling (default: 0.0 for deterministic)
        response_format: Response format specification (e.g., {"type": "json_object"})
        seed: Seed for reproducibility
        max_retries: Maximum number of retries for transient errors (default: 3)
        required_fields: List of required fields in JSON response (only used if response_format is json)
        
    Returns:
        Dict with keys:
            - "success": bool indicating if the call succeeded
            - "content": str with the response content (or error message)
            - "parsed": parsed JSON (if response_format is json and parsing succeeded)
            - "error_type": str indicating type of error (if failed)
    """
    
    for attempt in range(max_retries):
        if USE_TINKER:
            """
            Send a prompt to the judge model and receive the response.

            Args:
                prompt: The prompt to send to the judge model
                max_tokens: Maximum tokens in the response
                temperature: Temperature for sampling (0.0 for deterministic)

            Returns:
                The parsed response text from the judge model
            """
            if judge_client_config is None:
                raise ValueError("Tinker judge not initialized. Set use_tinker_judge=True in __init__")

            sampling_client = judge_client_config['sampling_client']
            renderer = judge_client_config['renderer']

            sampling_params = tinker.types.SamplingParams(
                max_tokens=judge_client_config['max_tokens'],
                stop=renderer.get_stop_sequences(),
                temperature=temperature,
            )

            if judge_client_config["judge_model_name"] == "openai/gpt-oss-20b":
                messages.append({"role": "system", "content": "Reasoning: low\nAnswer with only the final answer."})        # Let the model to return the answer immediately
                # messages.append({"role": "thinking", "content": ""})

            model_input = renderer.build_generation_prompt(messages)

            # Send prompt to judge model
            sample_future = sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )

            sample_result = sample_future.result()

            # Parse response
            judge_tokens = sample_result.sequences[0].tokens
            parsed_judge, _ = renderer.parse_response(judge_tokens)
            judge_text = renderers.get_text_content(parsed_judge)

            # try:
            #     # Parse out from ```json{response}```
            #     json_regex = r"```json\s*(\{.*?\})\s*```"
            #     match = re.search(json_regex, judge_text, re.DOTALL)
            #     if not match:
            #         print("🐛🐛🐛 json output not found in judge_text", judge_text)
            #         print("🐛🐛🐛 messages", messages)
            #         raise JSONDecodeError("JSON response not found in judge output", judge_text, 0)
            #     parsed_output = json.loads(match.group(1))
            # except json.JSONDecodeError as e:
            #     # Malformed JSON response; retry
            #     if attempt < max_retries - 1:
            #         wait_time = 2 ** attempt
            #         print(
            #             f"JSON decode error (attempt {attempt + 1}/{max_retries}): {e}. "
            #             f"Retrying in {wait_time}s..."
            #         )
            #         time.sleep(wait_time)
            #         continue
            #
            #     print(f"JSON decode error after {max_retries} attempts: {e}")
            #     return {
            #         "success": False,
            #         "content": f"JSON decode error: {str(e)}",
            #         "parsed": None,
            #         "error_type": "json_decode",
            #     }

            judge_score, judge_explanation = parse_sycophancy_score(judge_text), parse_judge_explanation(judge_text)

            return {
                "success": True,
                "content": judge_text,
                "parsed": {
                    "monitor_score": judge_score,
                    "explanation": judge_explanation,
                },
                "error_type": None,
            }
        else:
            try:
                # Build request kwargs
                request_kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                }

                if response_format is not None:
                    request_kwargs["response_format"] = response_format

                if seed is not None:
                    request_kwargs["seed"] = seed

                # Make the API call
                response = openai_client.chat.completions.create(**request_kwargs)

                # Validate response
                if not response.choices:
                    raise ValueError("Empty response from OpenAI API")

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty content in OpenAI response")

                # Parse JSON if requested
                parsed_output = None
                if response_format and response_format.get("type") == "json_object":
                    parsed_output = json.loads(content)

                    # Validate required fields
                    if required_fields:
                        missing_fields = [f for f in required_fields if f not in parsed_output]
                        if missing_fields:
                            raise ValueError(f"Missing required fields in response: {missing_fields}")

                return {
                    "success": True,
                    "content": content,
                    "parsed": parsed_output,
                    "error_type": None,
                }

            except (RateLimitError, APITimeoutError, InternalServerError, APIError) as e:
                # Retryable: 429, 500, 503, timeouts, and generic API errors
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(
                        f"Retryable error calling OpenAI API (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                print(f"Error calling OpenAI API after {max_retries} attempts: {e}")
                return {
                    "success": False,
                    "content": f"Error after {max_retries} retries: {str(e)}",
                    "parsed": None,
                    "error_type": "retryable_exhausted",
                }

            except (BadRequestError, AuthenticationError, PermissionDeniedError, NotFoundError) as e:
                # Non-retryable: 400, 401, 403, 404
                print(f"Non-retryable OpenAI API error: {e}")
                return {
                    "success": False,
                    "content": f"Non-retryable error: {str(e)}",
                    "parsed": None,
                    "error_type": "non_retryable",
                }

            except json.JSONDecodeError as e:
                # Malformed JSON response; retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(
                        f"JSON decode error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                print(f"JSON decode error after {max_retries} attempts: {e}")
                return {
                    "success": False,
                    "content": f"JSON decode error: {str(e)}",
                    "parsed": None,
                    "error_type": "json_decode",
                }

            except Exception as e:
                # Generic exceptions
                print(f"Error calling OpenAI API: {e}")
                return {
                    "success": False,
                    "content": f"Error: {str(e)}",
                    "parsed": None,
                    "error_type": "unknown",
                }
    
    # Should not reach here, but just in case
    return {
        "success": False,
        "content": "Max retries exceeded",
        "parsed": None,
        "error_type": "max_retries_exceeded",
    }