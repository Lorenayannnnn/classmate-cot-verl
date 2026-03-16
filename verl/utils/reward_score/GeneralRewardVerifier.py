from typing import Any, Dict

from verl.utils.reward_score.BaseVerifier import BaseVerifier


class GeneralRewardVerifier(BaseVerifier):
    """
    Dummy verifier for the 'general_reward' data source.

    For general_reward, scoring is handled externally by the reward model
    (e.g. hf_scoring backend) or per-behavior judge pipelines in _validate.
    This verifier exists only so that get_verifier("general_reward") does not
    raise, satisfying reward-manager code paths that instantiate a verifier
    before dispatching to the backend.

    """

    def __init__(self, **kwargs):
        super().__init__(reward_type="non_binary")

    # ── Methods used by generative judge backends (tinker / vllm_generative) ──

    def format_llm_judge_prompt(self, user_prompt: str, continuation: str, **kwargs):
        raise NotImplementedError("No prompt for classifier reward model")
        """Format a generic judge prompt for general_reward items."""
        # if continuation is None:
        #     return None
        # return (
        #     f"Instruction:\n{user_prompt}\n\n"
        #     f"Response:\n{continuation}\n\n"
        #     "Rate the quality of the response on a scale from 1 to 10.\n"
        #     "RATING: <number>"
        # )

    def parse_llm_judge_output(self, judge_result: str, **kwargs):
        if judge_result is None:
            return self.invalid_score, "empty judge output"
        raise NotImplementedError("No parsing needed for classifier reward model; should be handled by backend")
        # """Parse a numeric RATING from a generative judge response."""
        # import re
        #
        # if judge_result is None:
        #     return self.invalid_score, "empty judge output"
        #
        # match = re.search(r"RATING:\s*(\d+(?:\.\d+)?)", judge_result, re.IGNORECASE)
        # if match:
        #     score = float(match.group(1))
        #     return score, judge_result
        #
        # numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", judge_result)
        # if numbers:
        #     return float(numbers[0]), judge_result
        #
        # return self.invalid_score, judge_result

    # ── Methods that must NOT be called for general_reward ──

    def create_monitor_message(self, doc: Dict[str, Any]):
        raise NotImplementedError(
            "create_monitor_message should not be called for general_reward. "
            "general_reward uses per-behavior monitor pipelines in _validate."
        )

    def parse_monitor_output(self, monitor_response, **kwargs):
        raise NotImplementedError(
            "parse_monitor_output should not be called for general_reward. "
            "general_reward uses per-behavior monitor pipelines in _validate."
        )

    def compute_score(self, continuation, gt, **kwargs):
        raise NotImplementedError(
            "compute_score should not be called for general_reward. "
            "general_reward scores are provided by the external reward model backend."
        )

    def process_doc(self, doc):
        """
        Used only in my_eval
        """
        out_doc = {
            "question": doc["prompt"][0]["content"],
            "query": doc["prompt"][0]["content"],
            "gt": None,
            "raw_question_for_monitor": doc["reward_model"]["raw_question_for_monitor"]
        }
        return out_doc
