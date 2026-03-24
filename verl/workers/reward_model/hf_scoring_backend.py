"""Direct HuggingFace sequence-classification backend for standalone inference.

Unlike ScoringJudgeWorkerBackend (which requires a Ray worker group), this
backend loads the reward model directly and is suitable for use outside of
training (e.g. inference_main_model.py).
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from verl.workers.reward_model.judge_backend import BaseBackend


class HFScoringJudgeBackend(BaseBackend):
    """Runs AutoModelForSequenceClassification directly on the current device.
    Functionally equivalent to scoring_judge_worker_backend.py but without self.worker_group just for Inference

    Returns dicts of {"score": float, "extracted_solution": str,
    "judge_explanation": "hf_scoring"}, matching the format expected by
    InferenceExpRunner's general_reward scoring path.
    """

    def __init__(self, model_name_or_path: str, dtype: torch.dtype = torch.bfloat16):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self._model.eval()
        self._model.to(self._device)
        print(f"Initialized HFScoringJudgeBackend for {model_name_or_path} on {self._device}")

    def prepare_input(self, verifier, question: str, output: str):
        return (question, output)

    def run_batch(self, inputs: list) -> list:
        valid_indices = [i for i, x in enumerate(inputs) if x is not None]
        results = [None] * len(inputs)

        if not valid_indices:
            return results

        formatted_texts = []
        for i in valid_indices:
            question, output = inputs[i]
            conv = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": output if output is not None else ""},
            ]
            text = self._tokenizer.apply_chat_template(conv, tokenize=False)
            if self._tokenizer.bos_token and text.startswith(self._tokenizer.bos_token):
                text = text[len(self._tokenizer.bos_token):]
            formatted_texts.append(text)

        tokenized = self._tokenizer(
            formatted_texts, return_tensors="pt", padding=True, truncation=False,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**tokenized).logits  # (batch, 1)

        for list_idx, batch_idx in enumerate(valid_indices):
            score = float(logits[list_idx, 0].item())
            results[batch_idx] = {
                "score": score,
                "extracted_solution": inputs[batch_idx][1],
                "judge_explanation": "hf_scoring",
            }

        return results

    def close(self):
        del self._model
        del self._tokenizer
        torch.cuda.empty_cache()