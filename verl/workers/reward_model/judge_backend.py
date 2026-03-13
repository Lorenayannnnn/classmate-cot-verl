from abc import ABC, abstractmethod
from typing import Any


class LLMJudgeBackend(ABC):
    """Abstract base class for LLM judge inference backends.

    All backends accept a list of inputs prepared by ``prepare_input`` and
    return a list of raw outputs in the same order.  None inputs pass through
    as None outputs (skipped during inference).

    Output types per backend
    ------------------------
    TinkerJudgeBackend         : str | None
        Plain text extracted from the Tinker response object.
    VLLMGenerativeJudgeBackend : str | None
        Generated text from the judge model.
    HFScoringJudgeBackend      : dict | None
        {"score": float, "judge_explanation": "hf_scoring"} — score is
        ready to use, no further parsing required.

    Usage in reward managers
    ------------------------
    # Build the input (backend decides the format):
    inp = backend.prepare_input(verifier, question, output)  # or None

    # Run inference:
    result = backend.run_batch([inp, ...])[i]

    For generative backends (Tinker + vLLM generative):
        score, explanation = verifier.parse_llm_judge_output(result)

    For scoring backends:
        # isinstance(result, dict) and "score" in result → use directly.
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
    def run_batch(self, inputs: list[Any | None]) -> list[Any]:
        """Run inference on a batch of inputs prepared by ``prepare_input``.

        Args:
            inputs: List of backend-specific inputs.  None entries are skipped
                    and produce None at the corresponding output position.

        Returns:
            List of raw outputs with the same length as ``inputs``.
        """
        ...

    def close(self):
        """Optional cleanup (free GPU memory, close connections, etc.)."""
        pass
