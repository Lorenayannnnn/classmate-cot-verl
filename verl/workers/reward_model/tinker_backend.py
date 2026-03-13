
from concurrent.futures import Future

from verl.workers.reward_model.judge_backend import LLMJudgeBackend
from verl.utils.reward_score.monitor import send_prompt_to_tinker
from verl.utils.reward_score.BaseVerifier import BaseVerifier


class TinkerJudgeBackend(LLMJudgeBackend):
    """LLM judge backend using the Tinker API.

    Requests are submitted concurrently (one future per non-None prompt) and
    gathered after all have been submitted, preserving the original two-loop
    concurrency pattern while keeping the reward manager free of Tinker-specific
    logic.

    Text is extracted from the Tinker response object inside run_batch, so
    callers receive plain strings and can call
    ``verifier.parse_llm_judge_output(text, None)``.
    """

    def __init__(self, judge_client_config: dict):
        """
        Args:
            judge_client_config: Dict returned by create_llm_judge(), containing
                "sampling_client", "renderer", "tokenizer", "max_tokens", etc.
        """
        self._config = judge_client_config

    def prepare_input(self, verifier, question: str, output: str) -> str:
        return verifier.format_llm_judge_prompt(question, output)

    def run_batch(self, prompts: list[str | None]) -> list[str | None]:
        # Submit all non-None prompts concurrently (non-blocking).
        futures: list[Future | None] = []
        for p in prompts:
            if p is None:
                futures.append(None)
            else:
                futures.append(send_prompt_to_tinker(self._config, p))

        # Gather results and extract plain text.
        results: list[str | None] = []
        for f in futures:
            if f is None:
                results.append(None)
            else:
                raw = f.result()
                text = BaseVerifier._extract_response_text(raw, self._config)
                results.append(text)
        return results
