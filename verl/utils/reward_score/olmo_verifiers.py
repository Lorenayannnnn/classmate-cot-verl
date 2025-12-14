import ast
import asyncio
import json
import logging
import os
import re
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
import requests
from litellm import acompletion

from verl.utils.math_utils import last_boxed_only_string, remove_boxed, normalize_final_answer, get_unnormalized_answer, \
    is_equiv, hendrycks_is_equiv
from verl.utils.reward_score.IFEvalG import instructions_registry
from verl.utils.reward_score.if_functions import IF_FUNCTIONS_MAP
from verl.utils.reward_score.olmo3_judge_utils import JUDGE_PROMPT_MAP, EXTRACTOR_MAP, PRICE_PER_TOKEN, build_messages

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    score: float
    cost: float = 0.0
    reasoning: str | None = None

@dataclass
class VerifierConfig:
    """For now this config exists to support LMJudgeVerifer, can be expanded to support other verifers"""

    @classmethod
    def from_args(cls, args) -> "VerifierConfig":
        """
        Create a VerifierConfig from an Args object by automatically matching field names.
        Only fields that exist in both Args and VerifierConfig will be passed through.
        """
        import dataclasses

        # Get all field names from VerifierConfig
        verifier_fields = {field.name for field in dataclasses.fields(cls)}

        # Get all attributes from args that match VerifierConfig field names
        matching_kwargs = {}
        for field_name in verifier_fields:
            if hasattr(args, field_name):
                matching_kwargs[field_name] = getattr(args, field_name)

        return cls(**matching_kwargs)

class VerifierFunction(ABC):
    """
    Base class for all verifier functions that evaluate model predictions against ground truth.

    Each verifier function takes a prediction and compares it to a ground truth label,s
    returning a VerificationResult with a score between 0.0 and 1.0.
    """

    def __init__(self, name: str, weight: float = 1.0, verifier_config: VerifierConfig | None = None) -> None:
        self.name = name
        self.weight = weight
        self.verifier_config = verifier_config

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return VerifierConfig

    @abstractmethod
    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
    ) -> VerificationResult:
        """
        Evaluate the given prediction against the ground truth (or constraint).

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query

        Returns:
            VerificationResult
        """

    async def async_call(
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
    ) -> VerificationResult:
        """
        Asynchronous version of __call__. By default, it runs the synchronous __call__ in a thread pool.
        Subclasses can override this method for truly asynchronous implementation.

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query.

        Returns:
            VerificationResult
        """
        # Run the synchronous __call__ in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.__call__(tokenized_prediction, prediction, label, query))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight})"

@dataclass
class CodeVerifierConfig(VerifierConfig):
    code_api_url: str
    code_max_execution_time: float
    code_pass_rate_reward_threshold: float
    code_apply_perf_penalty: bool

def process_code_output(model_output: str, extract_CoT=False) -> str:
    """Extract the last code block between ``` markers from the model output."""
    # Find content between ``` markers
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)

    if not matches:
        return model_output
    else:
        if extract_CoT:
            # Split the model output at the first ``` marker
            split_output = re.split(r"```", model_output, maxsplit=1)
            # Return the part before the first ``` marker, stripped of whitespace
            return split_output[0].strip()

        # Return the last match, stripped of whitespace
        return matches[-1].strip()

class CodeVerifier(VerifierFunction):
    """
    Verifier that executes Python code against test cases using an external API.

    The label should be a list of test cases or a JSON string representation of a list.
    The API URL should be provided during initialization.
    """

    # Class-level session cache to reuse connections
    _session_cache = weakref.WeakKeyDictionary()

    def __init__(self, verifier_config: CodeVerifierConfig) -> None:
        super().__init__("code", verifier_config=verifier_config, weight=1.0)
        self.pass_rate_reward_threshold = verifier_config.code_pass_rate_reward_threshold
        self.apply_perf_penalty = verifier_config.code_apply_perf_penalty

    def extract_python_code(self, model_output: str) -> str:
        """Extract the last code block between ``` markers from the model output."""
        return process_code_output(model_output)

    # Create a session pool for better performance
    _session_pool = None

    @classmethod
    def _get_session(cls):
        if cls._session_pool is None:
            cls._session_pool = requests.Session()
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=100,
                pool_maxsize=100,
                max_retries=requests.adapters.Retry(
                    total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
                ),
            )
            cls._session_pool.mount("http://", adapter)
            cls._session_pool.mount("https://", adapter)
        return cls._session_pool

    async def async_call(
        self, prediction: str, label: Any, query: str | None = None
    ) -> VerificationResult:
        """
        Asynchronously verify code execution against test cases.

        Args:
            prediction: The model output containing Python code
            label: List of test cases or JSON string representation of a list
            query: Unused original query

        Returns:
            VerificationResult with score as the pass rate of test cases
        """
        # Extract Python code from the model output
        python_code = self.extract_python_code(prediction)

        # Test data
        payload = {
            "program": python_code,
            "tests": label,
            "max_execution_time": self.verifier_config.code_max_execution_time,
        }

        try:
            # Use connection pooling session
            session = self._get_session()

            # Calculate timeout
            http_timeout = max(30, min(300, self.verifier_config.code_max_execution_time * 10))

            # Make request in thread pool to keep it async
            def make_request():
                response = session.post(
                    self.verifier_config.code_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=http_timeout,
                )
                response.raise_for_status()
                return response.json()

            result = await asyncio.to_thread(make_request)
            passes = result["results"]
            pass_rate = sum(passes) / len(passes) if passes else 0.0
            score = 0.0 if pass_rate < self.pass_rate_reward_threshold else pass_rate
            if self.apply_perf_penalty and score > 0.0:
                runtimes = result["runtimes"]
                # for each runtime, multiply the percent of the timeout that was used
                multipliers = [
                    (self.verifier_config.code_max_execution_time - runtime)
                    / self.verifier_config.code_max_execution_time
                    for runtime in runtimes
                ]
                penalized_passes = [passes[i] * multipliers[i] for i in range(len(passes))]
                score = sum(penalized_passes) / len(penalized_passes)
            return VerificationResult(score=score)
        except Exception as e:
            logger.warning(f"Error verifying code sample: {e}")
            return VerificationResult(score=0.0)

    def __call__(
        self, prediction: str, label: Any, query: str | None = None
    ) -> VerificationResult:
        """
        Synchronously verify code execution against test cases.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous __call__ method from within an async context. Use async_call instead."
                )
            else:
                return asyncio.run(self.async_call(prediction, label, query))
        except RuntimeError:
            return asyncio.run(self.async_call(prediction, label, query))

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return CodeVerifierConfig


# small helper to optionally remove thinking section + answer output.
# assumes a certain format, so might not always be useful.
# we don't always need this -- for example, math evaluations just extract a final
# number, so we don't need to remove the thinking section.
def remove_thinking_section(prediction: str) -> str:
    prediction = prediction.replace("<|assistant|>", "").strip()
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


@dataclass
class LMJudgeVerifierConfig(VerifierConfig):
    # judge args
    llm_judge_model: str
    llm_judge_max_tokens: int
    llm_judge_max_context_length: int
    llm_judge_temperature: float
    llm_judge_timeout: int
    seed: int = None


def extract_final_answer(prediction: str) -> str:
    """
    Extract the substring between <answer> and </answer>.
    If no match is found, extract the substring after </think>.
    If neither condition matches, clean the prediction by removing the <|assistant|> tag.
    If none of the above applies, return the original string.

    Args:
        prediction (str): The input string.

    Returns:
        str: The extracted substring or the cleaned/original string.
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    think_match = re.search(r"</think>(.*)", prediction, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    cleaned = re.sub(r"<\|assistant\|>", "", prediction)
    if cleaned != prediction:
        return cleaned.strip()

    return prediction

class LMJudgeVerifier(VerifierFunction):
    """
    Verifier that uses a language model's judgement to score a response.
    """

    # Use WeakKeyDictionary to automatically clean up clients when event loops are garbage collected
    _client_cache = weakref.WeakKeyDictionary()

    def __init__(self, judge_type: str, verifier_config: LMJudgeVerifierConfig) -> None:
        super().__init__(f"general-{judge_type}", verifier_config=verifier_config, weight=1.0)
        self.prompt_template = JUDGE_PROMPT_MAP[judge_type]
        self.extractor = EXTRACTOR_MAP[judge_type]
        os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"

    def parse_completion(self, completion):
        """
        Extract reasoning and score from an OpenAI API completion response.

        Args:
            completion: The OpenAI API completion response object

        Returns:
            tuple: (reasoning, score) extracted from the response
        """
        reasoning = ""
        score = 0.0

        if not completion:
            print("No completion received from the model.")
            return reasoning, score

        try:
            # remove anything between <think> and </think> including the tags using regex
            pattern = r"<think>\s*.*?\s*</think>\s*"
            content = re.sub(pattern, "", completion.choices[0].message.content, flags=re.DOTALL)
            content = content.replace("<answer>", "").replace("</answer>", "")
            reasoning, score = self.extractor(content)

        except Exception as e:
            print(f"Error processing model response: {str(e)}")
            if hasattr(completion, "choices") and completion.choices is not None and len(completion.choices) > 0:
                print(f"Response content: {getattr(completion.choices[0].message, 'content', 'No content available')}")

        return reasoning, score

    def get_cost(self, response, model: str):
        """
        Get the cost of the response.
        """
        model_name = model.split("/")[-1]  # for litellm, discard the namespace
        model_name = model_name.replace("-standard", "")  # azure OAI models have -standard in the name
        return (
            PRICE_PER_TOKEN.get(model_name, {}).get("input", 0) * response.usage.prompt_tokens
            + PRICE_PER_TOKEN.get(model_name, {}).get("output", 0) * response.usage.completion_tokens
        )

    async def async_call(
        self, prediction: str, label: str, query: str
    ) -> VerificationResult:
        """
        Asynchronous version of __call__ that properly handles the async OpenAI client.
        """
        # client = self._get_client()
        final_answer = extract_final_answer(prediction)
        prompt = self.prompt_template.format(input=query, output=final_answer, label=label)

        max_retries = 3  # for rate limits
        retry_delay = 1.0

        for attempt in range(max_retries):
            # judges the quality of a response
            try:
                messages = build_messages(prompt)

                # Faeze: check if the request would exceed context window
                # Import the context window checker
                try:
                    from verl.context_window_checker import (
                        check_context_window_limit,
                        truncate_messages_to_fit_context,
                    )

                    context_check_available = True
                except ImportError:
                    logger.warning("Context window checker not available. Proceeding without context checking.")
                    context_check_available = False

                # Check if the request would exceed context window
                if context_check_available and not check_context_window_limit(
                    messages=messages,
                    max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                    model_name=self.verifier_config.llm_judge_model,
                    max_context_length=self.verifier_config.llm_judge_max_context_length,  # Adjust based on your model
                    safety_margin=150,
                ):
                    # Try to truncate messages to fit
                    messages = truncate_messages_to_fit_context(
                        messages=messages,
                        max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                        model_name=self.verifier_config.llm_judge_model,
                        max_context_length=self.verifier_config.llm_judge_max_context_length,
                        safety_margin=200,
                    )

                    # Check again after truncation
                    if not check_context_window_limit(
                        messages=messages,
                        max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                        model_name=self.verifier_config.llm_judge_model,
                        max_context_length=self.verifier_config.llm_judge_max_context_length,
                        safety_margin=150,
                    ):
                        logger.error("Cannot fit request within context window even after truncation.")
                        return VerificationResult(score=0.0, cost=0.0, reasoning="Error: Context window exceeded")
                # end of Faeze's context window check
                response = await acompletion(
                    model=self.verifier_config.llm_judge_model,
                    messages=messages,
                    temperature=self.verifier_config.llm_judge_temperature,
                    max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                    seed=self.verifier_config.seed,
                    timeout=self.verifier_config.llm_judge_timeout,
                    drop_params=True,
                )
                # print("🐛🐛🐛 query:", query)
                # print("🐛🐛🐛 prediction:", prediction)
                # print("🐛🐛🐛 label:", label)
                # print("🐛🐛🐛 llm judge raw response:", response)
                reasoning, score = self.parse_completion(response)
                # print("🐛🐛🐛 llm judge parsed response:", response)
                cost = self.get_cost(response, self.verifier_config.llm_judge_model)
                # cost = response.usage.estimated_cost
                # print("🐛🐛🐛 llm judge estimated cost:", cost)
                # normalize score to be between 0 and 1
                return VerificationResult(score=score, cost=cost, reasoning=reasoning)

            except Exception as e:
                logger.warning(f"LLM judge attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"LLM judge failed after {max_retries} attempts. Returning default score of 0.0")
                    return VerificationResult(score=0.0, cost=0.0, reasoning=f"Error: {str(e)}")
                else:
                    await asyncio.sleep(retry_delay * (2**attempt))  # Exponential backoff
        return VerificationResult(score=0.0, cost=0.0, reasoning="Unknown error after all retries.")

    def __call__(self, prediction: str, label: str, query: str) -> VerificationResult:
        """
        Evaluates the prediction based on an LLM's judgement.

        Args:
            # tokenized_prediction (List[int]): Tokenized representation of the prediction (unused).
            prediction (str): The model output string that was judged.
            label (str): An optional reference for the judge. Can be a reference answer or a rubric.
        Returns:
            float: The calculated reward (parsed_rating)
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous __call__ method from within an async context. Use async_call instead."
                )
            else:
                return asyncio.run(self.async_call(prediction, label, query))
        except RuntimeError:
            return asyncio.run(self.async_call(prediction, label, query))

    @classmethod
    async def cleanup_all_clients(cls):
        """
        Manually close all cached clients. Call this before shutting down to avoid cleanup warnings.
        """
        clients_to_close = list(cls._client_cache.values())
        cls._client_cache.clear()

        for client in clients_to_close:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {e}")
                # Suppress the error to avoid breaking shutdown

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return LMJudgeVerifierConfig


def verify_math(prediction: str, label: str) -> VerificationResult:
    raw_answer = prediction
    all_answers = []

    # Attempt extraction from \boxed{}.
    boxed_answer = last_boxed_only_string(raw_answer)
    if boxed_answer is not None:
        try:
            boxed_answer = remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    if boxed_answer is not None:
        all_answers.append(boxed_answer)

    # Attempt extraction via Minerva format.
    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)

    # Attempt extraction from the last LaTeX-formatted answer.
    if not all_answers:
        dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
        if len(dollars) > 1:
            answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)

    # Fallback to the full output.
    if not all_answers:
        all_answers.append(normalize_final_answer(prediction))
        # also provide original string in case normalization fails
        all_answers.append(prediction)

    # Compare each candidate answer to the ground truth.
    for answer in all_answers:
        if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
            return VerificationResult(score=1.0)
    return VerificationResult(score=0.0)

def verify_ifeval(prediction: str, label: str | dict) -> VerificationResult:
    # Copied from IFEvalVerifierOld from open-instruct
    constraint = label
    answer = remove_thinking_section(prediction)
    if isinstance(constraint, str):
        # constraint = json.loads(constraint)
        constraint = eval(constraint)
    if "func_name" not in constraint:
        logger.warning("Constraint missing 'func_name': %s", constraint)
        return VerificationResult(score=0.0)
    func_name = constraint.pop("func_name")
    func = IF_FUNCTIONS_MAP[func_name]
    non_none_args = {k: v for k, v in constraint.items() if v is not None}
    if not constraint:
        return VerificationResult(score=float(func(answer)))
    return VerificationResult(score=float(func(answer, **non_none_args)))

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    code_api_url=None,
    # llm_judge_config_dict=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    method=None,
    return_dict=True,
    # code_verifier_src_to_verifier=None,
    **kwargs,
):
    start_time = time.time()
    if data_source == "gsm8k":
        from verl.utils.reward_score import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth, method=method)
        if return_dict:
            return {
                "score": res
            }
    elif data_source.lower() == "math":
        score = verify_math(solution_str, ground_truth).score
    elif data_source == "ifeval":
        score = verify_ifeval(solution_str, ground_truth).score
    elif "code" in data_source:
        # if data_source == "code_stdio":
        #     tmp_url = code_api_url + "/test_program_stdio"
        if data_source == "code":
            # tmp_url = code_api_url + "/test_program"
            ground_truth = eval(ground_truth)
        # if code_verifier_src_to_verifier is None:
        #     print("WARNING: creating new code verifier inside compute_score")
        #     code_verifier_src_to_verifier = {
        #         "code": CodeVerifier(
        #             verifier_config=CodeVerifierConfig(
        #                 code_api_url=code_api_url + "/test_program",
        #                 code_max_execution_time=1.0,
        #                 code_pass_rate_reward_threshold=0.99,
        #                 code_apply_perf_penalty=False,
        #             )
        #         ),
        #         "code_stdio": CodeVerifier(
        #             verifier_config=CodeVerifierConfig(
        #                 code_api_url=code_api_url + "/test_program_stdio",
        #                 code_max_execution_time=1.0,
        #                 code_pass_rate_reward_threshold=0.99,
        #                 code_apply_perf_penalty=False,
        #             )
        #         ),
        #     }
        # score = code_verifier_src_to_verifier[data_source](prediction=solution_str, label=ground_truth).score
    elif "general" in data_source:
        raise NotImplementedError("Not training on this subset for now due to long sampling time")
        # judge_type = data_source.split("general-")[-1]
        # # llm_judge_config_dict = {
        # #   "llm_judge_model": null,
        # #   "llm_judge_timeout": 600,
        # #   "llm_judge_max_tokens": 2048,
        # #   "llm_judge_max_context_length": 32768,
        # #   "llm_judge_temperature": 1.0,
        # #   "seed": 42,
        # # }
        # verifier = LMJudgeVerifier(
        #     judge_type=judge_type,
        #     verifier_config=LMJudgeVerifierConfig(**llm_judge_config_dict)
        # )
        # # prediction: str, label: str, query: str
        # result = verifier(prediction=solution_str, label=ground_truth, query=extra_info["raw_prompt"])
        # score = result.score
        # print(f"🐛🐛🐛 cost for {data_source} verifier: {result.cost:.6f} USD")
    else:
        raise NotImplementedError(f"unknown data source for olmo3 verifier: {data_source}")

    # if "code" in data_source:
    #     end_time = time.time()
        # print(f"🐛🐛🐛 {data_source} compute score took {end_time - start_time:.2f} seconds.")

    # print("🐛🐛🐛 prompt:", extra_info["raw_prompt"])
    # print("🐛🐛🐛 prediction:", solution_str)
    # print("🐛🐛🐛 label:", ground_truth)
    if return_dict:
        return {"score": score}
    else:
        return score