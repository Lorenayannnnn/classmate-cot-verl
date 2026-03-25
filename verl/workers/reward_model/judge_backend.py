from abc import ABC, abstractmethod
from typing import Any


class BaseBackend(ABC):
    """Abstract base class for LLM inference backends (judge and monitor).

    A single backend instance can serve both LLM-judge and monitor roles —
    the distinction is which verifier method is called to prepare prompts and
    parse results.

    All backends accept a list of inputs prepared by ``prepare_input`` (for
    judge) or pre-formatted prompt strings (for monitor) and return a list of
    raw outputs in the same order.  None inputs pass through as None outputs.

    Output types per backend
    ------------------------
    TinkerJudgeBackend         : str | None
        Plain text extracted from the Tinker response object.
    VLLMGenerativeJudgeBackend : str | None
        Generated text from the model.
    HFScoringJudgeBackend      : dict | None
        {"score": float, "judge_explanation": "hf_scoring"} — score is
        ready to use, no further parsing required.  Only suitable for judge
        use; monitor always uses a generative backend.

    LLM-judge usage in reward managers
    ------------------------------------
    inp = backend.prepare_input(verifier, question, output)  # or None
    result = backend.run_batch([inp, ...])[i]
    # generative: score, explanation = verifier.parse_llm_judge_output(result)
    # scoring:    isinstance(result, dict) → use directly.

    Monitor usage in ray trainers
    ------------------------------
    prompt = verifier.create_monitor_message(monitor_doc)  # or None
    raw_out = backend.run_batch([prompt, ...])[i]
    score, explanation = verifier.parse_monitor_output(raw_out)
    """

    @abstractmethod
    def prepare_input(self, verifier, question: str, output: str) -> Any:
        """Prepare a single judge input from a raw question and model output.

        Each backend decides the format:
        - Generative backends return the formatted LLM judge prompt string.
        - Scoring backends return a (question, output) tuple that run_batch
          formats as a chat conversation internally.

        Args:
            verifier: BaseVerifier instance for the current data source.
            question: The raw user-facing question.
            output: The model's response to score.

        Returns:
            Backend-specific input object.
        """
        ...

    @abstractmethod
    def run_batch(
        self,
        inputs: list[Any | None],
        icl_pairs: list[tuple[str, str]] | None = None,
    ) -> list[Any]:
        """Run inference on a batch of inputs prepared by ``prepare_input``.

        Args:
            inputs:    List of backend-specific inputs.  None entries are skipped
                       and produce None at the corresponding output position.
            icl_pairs: Optional list of (user_msg, assistant_msg) few-shot
                       demonstration pairs to prepend to every prompt.
                       - Generative backends (vLLM, OpenAI) format them as
                         alternating user/assistant messages so the chat template
                         handles them correctly.
                       - Tinker backend prepends them as flat text separated by
                         ``\\n\\n---\\n\\n``.

        Returns:
            List of raw outputs with the same length as ``inputs``.
        """
        ...

    def close(self):
        """Optional cleanup (free GPU memory, close connections, etc.)."""
        pass
