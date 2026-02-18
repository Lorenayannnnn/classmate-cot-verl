from concurrent.futures import Future
from typing import Dict, Any

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from keys import TINKER_API_KEY

def create_llm_judge(
    judge_model_name: str,
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
        'judge_model_name': judge_model_name,
        'sampling_client': judge_sampling_client,
        'renderer': judge_renderer,
        'tokenizer': judge_tokenizer,
        "max_tokens": 1024,
    }

def send_prompt_to_tinker(
    judge_client_config: Dict[str, Any],
    prompt: str,
    temperature: float = 0.0
) -> Future:
    """
    Send a prompt to the judge model and receive the response.

    Args:
        prompt: The prompt to send to the judge model
        max_tokens: Maximum tokens in the response
        temperature: Temperature for sampling (0.0 for deterministic)

    Returns:
        The parsed response text from the judge model
    """

    sampling_client = judge_client_config['sampling_client']
    renderer = judge_client_config['renderer']

    sampling_params = tinker.types.SamplingParams(
        max_tokens=judge_client_config["max_tokens"],
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
    )

    messages = [{"role": "user", "content": prompt}]

    # if judge_client_config["judge_model_name"] == "openai/gpt-oss-20b":
    #     messages.append({"role": "system",
    #                      "content": "Reasoning: low\nAnswer with only the final answer."})  # Let the model to return the answer immediately
        # messages.append({"role": "thinking", "content": ""})        # Let the model to return the answer immediately

    model_input = renderer.build_generation_prompt(messages)

    # Send prompt to judge model
    return sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    #
    # sample_result = sample_future.result()
    #
    # # Parse response
    # judge_tokens = sample_result.sequences[0].tokens
    # parsed_judge, _ = renderer.parse_response(judge_tokens)
    # judge_text = renderers.get_text_content(parsed_judge)
    #
    # return judge_text