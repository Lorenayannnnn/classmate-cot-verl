import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from verl.workers.reward_model.judge_backend import BaseBackend


class HFScoringJudgeBackend(BaseBackend):
    """LLM judge backend using a HuggingFace reward (scoring) model directly.

    Unlike generative backends, this backend does not call
    ``verifier.format_llm_judge_prompt``; instead ``prepare_input`` returns a
    raw ``(question, output)`` tuple, and ``run_batch`` formats it as a
    two-turn conversation with ``apply_chat_template`` before scoring.

    Runs ``model(**tokenized).logits[0][0].item()`` per the reference scoring
    pattern. All valid inputs are batched into a single forward pass.
    Uses all available GPUs via ``torch.nn.DataParallel``.

    Returns list of {"score": float, "judge_explanation": "hf_scoring"} dicts,
    which the reward manager uses directly without calling parse_llm_judge_output.
    """

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model_name_or_path: HuggingFace model id or local path.
            torch_dtype: Model weight dtype.
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        )
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        self._model = model
        self._model.eval()
        self._model.to("cpu")

    def prepare_input(self, verifier, question: str, output: str) -> tuple[str, str]:
        """Return (question, output) directly; formatting is done in run_batch."""
        return (question, output)

    def run_batch(self, inputs: list[tuple[str, str] | None]) -> list[dict | None]:
        valid = [(i, item) for i, item in enumerate(inputs) if item is not None]
        if not valid:
            return [None] * len(inputs)

        self._model.to(self._device)
        try:
            formatted_texts = []
            for _, (question, output) in valid:
                conv = [{"role": "user", "content": question},
                        {"role": "assistant", "content": output}]
                formatted = self._tokenizer.apply_chat_template(conv, tokenize=False)
                if (self._tokenizer.bos_token is not None
                        and formatted.startswith(self._tokenizer.bos_token)):
                    formatted = formatted[len(self._tokenizer.bos_token):]
                formatted_texts.append(formatted)

            tokenized = self._tokenizer(
                formatted_texts, return_tensors="pt", padding=True, truncation=False,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**tokenized).logits  # (batch, num_labels)

            results: list[dict | None] = [None] * len(inputs)
            for (i, _), score in zip(valid, logits[:, 0].tolist()):
                results[i] = {"score": score, "judge_explanation": "hf_scoring"}
            return results
        finally:
            self._model.to("cpu")
            torch.cuda.empty_cache()

    def close(self):
        del self._model
        del self._tokenizer
